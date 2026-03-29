"""
PayFlow — GPU Priority Queue & Concurrency Manager
====================================================
Replaces the binary exclusive-mode VRAM model with cooperative coexistence.

The LLM (Qwen 3.5 9B) stays permanently resident in VRAM.  GNN and
XGBoost workloads share the ~1.1 GB headroom under priority-based
arbitration with dynamic KV-cache scaling and automatic CPU fallback.

Priority order:  LLM (always)  >  GNN (semaphore)  >  ML Training (semaphore)

VRAM pressure levels (with hysteresis):
    NORMAL   — < 6 500 MB  -> full context (16 384 tokens)
    HIGH     — 6 500–7 800 MB -> reduced context (8 192 tokens)
    CRITICAL — ≥ 7 800 MB  -> minimal context (4 096) + GNN -> CPU
"""

from __future__ import annotations

import asyncio
import enum
import logging
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import AsyncGenerator, Callable, Optional

from config.settings import GPUConcurrencyConfig, GPU_CONCURRENCY_CFG

logger = logging.getLogger(__name__)


# ── Enums ─────────────────────────────────────────────────────────────────────

class GPUPriority(enum.IntEnum):
    """Lower value = higher priority."""
    LLM = 0
    GNN = 1
    ML_TRAINING = 2


class VRAMPressureLevel(enum.Enum):
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


# ── Metrics ───────────────────────────────────────────────────────────────────

@dataclass
class GPUConcurrencyMetrics:
    """Mutable counters for observability / dashboard integration."""
    llm_acquisitions: int = 0
    gnn_gpu_acquisitions: int = 0
    gnn_cpu_fallbacks: int = 0
    ml_acquisitions: int = 0
    kv_cache_reductions: int = 0
    kv_cache_restorations: int = 0
    current_pressure: str = "normal"
    current_num_ctx: int = 16384


# ── GPU Priority Queue ───────────────────────────────────────────────────────

class GPUPriorityQueue:
    """
    Async-safe GPU arbitrator.

    Thread-safe pressure updates via ``update_pressure()`` (called from
    HardwareProfiler's thread pool through ``run_coroutine_threadsafe``).
    All acquire methods are async context managers for clean RAII semantics.
    """

    _instance: Optional["GPUPriorityQueue"] = None

    def __init__(self, config: GPUConcurrencyConfig | None = None) -> None:
        self._config = config or GPU_CONCURRENCY_CFG
        self._pressure = VRAMPressureLevel.NORMAL
        self._current_num_ctx: int = self._config.num_ctx_full
        self._gnn_semaphore = asyncio.Semaphore(self._config.max_concurrent_gnn)
        self._ml_semaphore = asyncio.Semaphore(self._config.max_concurrent_ml)
        self._state_lock = asyncio.Lock()
        self._ctx_change_callback: Optional[Callable[[int], None]] = None
        self._metrics = GPUConcurrencyMetrics(
            current_num_ctx=self._config.num_ctx_full,
        )
        self._last_pressure_change = time.monotonic()

    # ── Singleton ─────────────────────────────────────────────────────────

    @classmethod
    def get(cls, config: GPUConcurrencyConfig | None = None) -> "GPUPriorityQueue":
        if cls._instance is None:
            cls._instance = cls(config)
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Reset singleton — used by tests."""
        cls._instance = None

    # ── Pressure management ───────────────────────────────────────────────

    async def update_pressure(self, vram_used_mb: float) -> VRAMPressureLevel:
        """
        Update VRAM pressure level based on current usage.

        Uses hysteresis: going *up* uses high/critical thresholds,
        coming *down* requires dropping below ``vram_normal_threshold_mb``.
        """
        async with self._state_lock:
            old = self._pressure
            cfg = self._config

            if vram_used_mb >= cfg.vram_critical_threshold_mb:
                new = VRAMPressureLevel.CRITICAL
            elif vram_used_mb >= cfg.vram_high_threshold_mb:
                new = VRAMPressureLevel.HIGH
            elif vram_used_mb <= cfg.vram_normal_threshold_mb:
                new = VRAMPressureLevel.NORMAL
            else:
                # Between normal and high thresholds: maintain current state
                new = old

            if new != old:
                self._pressure = new
                self._last_pressure_change = time.monotonic()
                logger.info(
                    "VRAM pressure: %s -> %s  (%.0f MB used)",
                    old.value, new.value, vram_used_mb,
                )
                self._update_num_ctx(new)
                self._metrics.current_pressure = new.value

            return self._pressure

    def _update_num_ctx(self, pressure: VRAMPressureLevel) -> None:
        """Adjust dynamic context window based on pressure level."""
        cfg = self._config
        if pressure == VRAMPressureLevel.NORMAL:
            target = cfg.num_ctx_full
        elif pressure == VRAMPressureLevel.HIGH:
            target = cfg.num_ctx_medium
        else:
            target = cfg.num_ctx_minimal

        if target != self._current_num_ctx:
            old_ctx = self._current_num_ctx
            self._current_num_ctx = target
            self._metrics.current_num_ctx = target

            if target < old_ctx:
                self._metrics.kv_cache_reductions += 1
                logger.info("KV cache reduced: num_ctx %d -> %d", old_ctx, target)
            else:
                self._metrics.kv_cache_restorations += 1
                logger.info("KV cache restored: num_ctx %d -> %d", old_ctx, target)

            if self._ctx_change_callback:
                try:
                    self._ctx_change_callback(target)
                except Exception:
                    logger.warning("ctx_change_callback failed", exc_info=True)

    # ── Acquire context managers ──────────────────────────────────────────

    @asynccontextmanager
    async def acquire_llm(self) -> AsyncGenerator[int, None]:
        """
        Acquire GPU for LLM inference.  Always grants immediately.
        Yields the current dynamic ``num_ctx`` value.
        """
        self._metrics.llm_acquisitions += 1
        yield self._current_num_ctx

    @asynccontextmanager
    async def acquire_gnn(self) -> AsyncGenerator[bool, None]:
        """
        Acquire GPU for GNN inference.  Semaphore-gated.
        Yields ``use_gpu`` — ``False`` at CRITICAL pressure (caller should
        use CPU fallback).
        """
        try:
            await asyncio.wait_for(
                self._gnn_semaphore.acquire(),
                timeout=self._config.gpu_acquire_timeout_sec,
            )
        except asyncio.TimeoutError:
            logger.warning("GNN GPU acquire timed out — falling back to CPU")
            self._metrics.gnn_cpu_fallbacks += 1
            yield False
            return

        try:
            if self._pressure == VRAMPressureLevel.CRITICAL:
                self._metrics.gnn_cpu_fallbacks += 1
                logger.info("VRAM CRITICAL — GNN offloaded to CPU")
                yield False
            else:
                self._metrics.gnn_gpu_acquisitions += 1
                yield True
        finally:
            self._gnn_semaphore.release()

    @asynccontextmanager
    async def acquire_ml(self) -> AsyncGenerator[None, None]:
        """
        Acquire GPU for ML training (XGBoost).  Semaphore-gated, lowest priority.
        """
        try:
            await asyncio.wait_for(
                self._ml_semaphore.acquire(),
                timeout=self._config.gpu_acquire_timeout_sec,
            )
        except asyncio.TimeoutError:
            logger.warning("ML GPU acquire timed out")
            raise

        try:
            self._metrics.ml_acquisitions += 1
            yield
        finally:
            self._ml_semaphore.release()

    # ── Configuration ─────────────────────────────────────────────────────

    def set_ctx_change_callback(self, callback: Callable[[int], None]) -> None:
        """Register a callback fired when ``num_ctx`` changes."""
        self._ctx_change_callback = callback

    # ── Observability ─────────────────────────────────────────────────────

    @property
    def pressure(self) -> VRAMPressureLevel:
        return self._pressure

    @property
    def current_num_ctx(self) -> int:
        return self._current_num_ctx

    def snapshot(self) -> dict:
        """Return current state for dashboard / monitoring."""
        return {
            "pressure": self._pressure.value,
            "num_ctx": self._current_num_ctx,
            "llm_acquisitions": self._metrics.llm_acquisitions,
            "gnn_gpu_acquisitions": self._metrics.gnn_gpu_acquisitions,
            "gnn_cpu_fallbacks": self._metrics.gnn_cpu_fallbacks,
            "ml_acquisitions": self._metrics.ml_acquisitions,
            "kv_cache_reductions": self._metrics.kv_cache_reductions,
            "kv_cache_restorations": self._metrics.kv_cache_restorations,
        }

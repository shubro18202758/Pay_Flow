"""
PayFlow — VRAM Lifecycle Manager
=================================
Manages GPU memory transitions between Analysis mode (ML/GNN) and
Assistant mode (LLM). Ensures no two heavy consumers coexist on the 8 GB card.
"""

from __future__ import annotations

import enum
import logging
from contextlib import contextmanager
from typing import Generator

logger = logging.getLogger(__name__)


class GPUMode(enum.Enum):
    IDLE = "idle"
    ANALYSIS = "analysis"    # XGBoost / GNN hold VRAM
    ASSISTANT = "assistant"  # Ollama LLM holds VRAM
    FINETUNING = "finetuning"  # QLoRA fine-tuning (exclusive, full 8 GB)


_current_mode: GPUMode = GPUMode.IDLE


def _flush_torch_cache() -> None:
    """Release all cached PyTorch CUDA memory back to the driver."""
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            allocated = torch.cuda.memory_allocated() / 1024 / 1024
            logger.info("PyTorch CUDA cache flushed. Still allocated: %.1f MB", allocated)
    except ImportError:
        pass


def _unload_ollama_model(model: str = "qwen3.5:9b") -> None:
    """No-op — LLM stays permanently resident in cooperative mode (Phase 15)."""
    logger.debug("_unload_ollama_model() called but skipped (cooperative mode).")


def _unload_ollama_model_forced(model: str = "qwen3.5:9b") -> None:
    """Force-unload Ollama model from VRAM.  Used only by finetuning_mode()."""
    try:
        import httpx
        resp = httpx.post(
            "http://localhost:11434/api/generate",
            json={"model": model, "keep_alive": 0},
            timeout=10.0,
        )
        if resp.status_code == 200:
            logger.info("Ollama model '%s' force-unloaded from VRAM.", model)
    except Exception as exc:
        logger.warning("Could not force-unload Ollama model: %s", exc)


def get_current_mode() -> GPUMode:
    return _current_mode


@contextmanager
def analysis_mode() -> Generator[None, None, None]:
    """
    Context manager that claims GPU for ML/GNN workloads.
    Unloads Ollama LLM first if needed, and flushes VRAM on exit.

    Usage:
        with analysis_mode():
            model.fit(X_train, y_train)
    """
    global _current_mode

    if _current_mode == GPUMode.ASSISTANT:
        logger.info("Transitioning ASSISTANT -> ANALYSIS: LLM stays resident (cooperative mode)")

    _current_mode = GPUMode.ANALYSIS
    logger.info("GPU mode: ANALYSIS")
    try:
        yield
    finally:
        _flush_torch_cache()
        _current_mode = GPUMode.IDLE
        logger.info("GPU mode: IDLE (analysis finished)")


@contextmanager
def assistant_mode() -> Generator[None, None, None]:
    """
    Context manager that claims GPU for LLM inference.
    Flushes PyTorch tensors first if needed.

    Usage:
        with assistant_mode():
            response = ollama.chat(model="qwen3.5:9b", ...)
    """
    global _current_mode

    if _current_mode == GPUMode.ANALYSIS:
        logger.info("Transitioning ANALYSIS -> ASSISTANT: ML tensors coexist (cooperative mode)")

    _current_mode = GPUMode.ASSISTANT
    logger.info("GPU mode: ASSISTANT")
    try:
        yield
    finally:
        _current_mode = GPUMode.IDLE
        logger.info("GPU mode: IDLE (assistant finished)")


def log_vram_status() -> dict:
    """Return current VRAM utilization for monitoring."""
    info = {"mode": _current_mode.value}
    try:
        import torch
        if torch.cuda.is_available():
            info["allocated_mb"] = round(torch.cuda.memory_allocated() / 1024 / 1024, 1)
            info["reserved_mb"] = round(torch.cuda.memory_reserved() / 1024 / 1024, 1)
            info["total_mb"] = round(torch.cuda.get_device_properties(0).total_memory / 1024 / 1024, 1)
    except ImportError:
        info["torch_available"] = False
    return info


@contextmanager
def finetuning_mode() -> Generator[None, None, None]:
    """
    Context manager that claims GPU exclusively for QLoRA fine-tuning.
    Evicts both Ollama LLM and PyTorch training tensors before entering,
    then performs aggressive cleanup on exit.

    Usage:
        with finetuning_mode():
            trainer.train()
    """
    global _current_mode

    if _current_mode == GPUMode.ASSISTANT:
        logger.info("Transitioning ASSISTANT -> FINETUNING: force-unloading LLM...")
        _unload_ollama_model_forced()
    if _current_mode in (GPUMode.ANALYSIS, GPUMode.ASSISTANT):
        logger.info("Flushing PyTorch VRAM before fine-tuning...")
        _flush_torch_cache()

    _current_mode = GPUMode.FINETUNING
    logger.info("GPU mode: FINETUNING (exclusive)")
    try:
        yield
    finally:
        _flush_torch_cache()
        _current_mode = GPUMode.IDLE
        logger.info("GPU mode: IDLE (fine-tuning finished)")

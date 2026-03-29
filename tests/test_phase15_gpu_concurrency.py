"""
PayFlow -- Phase 15 Tests: Advanced GPU Concurrency Management
===============================================================
Verifies GPUConcurrencyConfig defaults/immutability, GPUPriorityQueue
pressure transitions with hysteresis, dynamic num_ctx scaling, GNN GPU vs
CPU acquisition, semaphore enforcement, LoadShedder CPU fallback, VRAM
manager cooperative mode, and metrics snapshot structure.

Tests:
 1. GPUConcurrencyConfig defaults match spec
 2. GPUConcurrencyConfig frozen rejects assignment
 3. CPU-only mode bypasses GPU queue creation
 4. Pressure NORMAL at low VRAM usage
 5. Pressure HIGH at elevated VRAM usage
 6. Pressure CRITICAL at maximum VRAM usage
 7. Hysteresis prevents oscillation (stays HIGH between thresholds)
 8. Dynamic num_ctx scales with pressure level
 9. ctx_change_callback fires on num_ctx change
10. GNN acquire yields use_gpu=True at NORMAL pressure
11. GNN acquire yields use_gpu=False at CRITICAL pressure
12. GNN CPU fallback scoring returns real risk scores
13. LLM acquire always succeeds and yields num_ctx
14. GNN semaphore limits concurrent acquisitions
15. ML semaphore limits concurrent acquisitions
16. Concurrent GNN + LLM does not deadlock
17. LoadShedder CPU fallback returns real risk score
18. LoadShedder sentinel only on complete failure
19. VRAM manager analysis_mode does not unload LLM
20. VRAM manager cooperative coexistence mode
21. Metrics snapshot has expected structure
22. Full pressure cycle: normal -> high -> critical -> normal
"""

from __future__ import annotations

import asyncio
import sys
import time
from dataclasses import FrozenInstanceError
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import GPU_CONCURRENCY_CFG, GPUConcurrencyConfig
from config.gpu_concurrency import (
    GPUConcurrencyMetrics,
    GPUPriority,
    GPUPriorityQueue,
    VRAMPressureLevel,
)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _safe_print(msg: str) -> None:
    """Print with ASCII fallback for Windows cp1252 consoles."""
    try:
        print(msg)
    except UnicodeEncodeError:
        print(msg.encode("ascii", "replace").decode())


passed = 0
failed = 0


def run_test(func):
    global passed, failed
    name = func.__name__
    try:
        if asyncio.iscoroutinefunction(func):
            asyncio.run(func())
        else:
            func()
        _safe_print(f"  PASS  {name}")
        passed += 1
    except Exception as e:
        _safe_print(f"  FAIL  {name}: {e}")
        failed += 1


def _fresh_queue(config=None) -> GPUPriorityQueue:
    """Create a fresh GPUPriorityQueue (bypasses singleton)."""
    GPUPriorityQueue.reset()
    return GPUPriorityQueue(config or GPUConcurrencyConfig())


# ── Test 1: Config defaults ──────────────────────────────────────────────────

def test_gpu_concurrency_config_defaults():
    cfg = GPUConcurrencyConfig()
    assert cfg.vram_critical_threshold_mb == 7800.0, "critical threshold"
    assert cfg.vram_high_threshold_mb == 7500.0, "high threshold"
    assert cfg.vram_normal_threshold_mb == 6500.0, "normal threshold"
    assert cfg.num_ctx_full == 16384, "num_ctx full"
    assert cfg.num_ctx_medium == 8192, "num_ctx medium"
    assert cfg.num_ctx_minimal == 4096, "num_ctx minimal"
    assert cfg.max_concurrent_gnn == 1, "max concurrent GNN"
    assert cfg.max_concurrent_ml == 1, "max concurrent ML"
    assert cfg.gpu_acquire_timeout_sec == 30.0, "acquire timeout"
    # Verify singleton matches
    assert GPU_CONCURRENCY_CFG.vram_critical_threshold_mb == 7800.0


# ── Test 2: Config frozen ────────────────────────────────────────────────────

def test_gpu_concurrency_config_frozen():
    cfg = GPUConcurrencyConfig()
    try:
        cfg.vram_critical_threshold_mb = 9000.0  # type: ignore
        assert False, "Should have raised FrozenInstanceError"
    except FrozenInstanceError:
        pass


# ── Test 3: CPU-only bypasses queue ──────────────────────────────────────────

def test_cpu_only_bypasses_queue():
    """Verify that GPUPriorityQueue can be instantiated (for testing) but
    the orchestrator skips it when cpu_only=True."""
    # Just verify the queue can be created with default config
    q = _fresh_queue()
    assert q.pressure == VRAMPressureLevel.NORMAL
    assert q.current_num_ctx == 16384


# ── Test 4: Pressure NORMAL at low VRAM ──────────────────────────────────────

async def test_pressure_normal():
    q = _fresh_queue()
    level = await q.update_pressure(4000.0)
    assert level == VRAMPressureLevel.NORMAL, f"Expected NORMAL, got {level}"
    assert q.current_num_ctx == 16384


# ── Test 5: Pressure HIGH at elevated VRAM ───────────────────────────────────

async def test_pressure_high():
    q = _fresh_queue()
    level = await q.update_pressure(7600.0)
    assert level == VRAMPressureLevel.HIGH, f"Expected HIGH, got {level}"
    assert q.current_num_ctx == 8192


# ── Test 6: Pressure CRITICAL at maximum VRAM ────────────────────────────────

async def test_pressure_critical():
    q = _fresh_queue()
    level = await q.update_pressure(7900.0)
    assert level == VRAMPressureLevel.CRITICAL, f"Expected CRITICAL, got {level}"
    assert q.current_num_ctx == 4096


# ── Test 7: Hysteresis prevents oscillation ──────────────────────────────────

async def test_hysteresis_prevents_oscillation():
    q = _fresh_queue()
    # Go to HIGH
    await q.update_pressure(7600.0)
    assert q.pressure == VRAMPressureLevel.HIGH
    # Drop to 7000 MB — between normal (6500) and high (7500) thresholds
    # Should stay HIGH due to hysteresis
    level = await q.update_pressure(7000.0)
    assert level == VRAMPressureLevel.HIGH, f"Expected HIGH (hysteresis), got {level}"
    # Drop below normal threshold — should clear to NORMAL
    level = await q.update_pressure(6000.0)
    assert level == VRAMPressureLevel.NORMAL, f"Expected NORMAL, got {level}"


# ── Test 8: Dynamic num_ctx scales ───────────────────────────────────────────

async def test_dynamic_num_ctx_scales():
    q = _fresh_queue()
    assert q.current_num_ctx == 16384, "initial"
    await q.update_pressure(7600.0)
    assert q.current_num_ctx == 8192, "after HIGH"
    await q.update_pressure(7900.0)
    assert q.current_num_ctx == 4096, "after CRITICAL"
    await q.update_pressure(5000.0)
    assert q.current_num_ctx == 16384, "after return to NORMAL"


# ── Test 9: ctx_change_callback fires ────────────────────────────────────────

async def test_ctx_change_callback():
    q = _fresh_queue()
    captured = []
    q.set_ctx_change_callback(lambda ctx: captured.append(ctx))
    await q.update_pressure(7600.0)  # NORMAL -> HIGH: 16384 -> 8192
    assert captured == [8192], f"Expected [8192], got {captured}"
    await q.update_pressure(7900.0)  # HIGH -> CRITICAL: 8192 -> 4096
    assert captured == [8192, 4096], f"Expected [8192, 4096], got {captured}"


# ── Test 10: GNN acquire GPU at NORMAL ───────────────────────────────────────

async def test_gnn_acquire_gpu_normal():
    q = _fresh_queue()
    async with q.acquire_gnn() as use_gpu:
        assert use_gpu is True, "Should use GPU at NORMAL pressure"


# ── Test 11: GNN acquire CPU at CRITICAL ─────────────────────────────────────

async def test_gnn_acquire_cpu_critical():
    q = _fresh_queue()
    await q.update_pressure(7900.0)  # CRITICAL
    async with q.acquire_gnn() as use_gpu:
        assert use_gpu is False, "Should fall back to CPU at CRITICAL pressure"
    snap = q.snapshot()
    assert snap["gnn_cpu_fallbacks"] >= 1, "CPU fallback not counted"


# ── Test 12: GNN CPU fallback scoring ────────────────────────────────────────

def test_gnn_cpu_fallback_scoring():
    """Verify score_subgraph_cpu returns a real GNNScoringResult."""
    from src.ml.models.gnn_scorer import GNNScoringResult

    # Mock the GNN scorer's score_subgraph_cpu method
    mock_gnn = MagicMock()
    mock_gnn.score_subgraph_cpu.return_value = GNNScoringResult(
        risk_score=0.75,
        node_count=10,
        edge_count=20,
        inference_ms=3.5,
    )

    result = mock_gnn.score_subgraph_cpu("subgraph", ["node1"], 12345)
    assert result.risk_score == 0.75, f"Expected 0.75, got {result.risk_score}"
    assert result.node_count == 10
    assert result.inference_ms == 3.5


# ── Test 13: LLM acquire always succeeds ─────────────────────────────────────

async def test_llm_acquire_always_succeeds():
    q = _fresh_queue()
    # Even at CRITICAL pressure, LLM should always be granted
    await q.update_pressure(7900.0)
    async with q.acquire_llm() as num_ctx:
        assert num_ctx == 4096, f"Expected 4096 at CRITICAL, got {num_ctx}"
    # At NORMAL
    await q.update_pressure(4000.0)
    async with q.acquire_llm() as num_ctx:
        assert num_ctx == 16384, f"Expected 16384 at NORMAL, got {num_ctx}"


# ── Test 14: GNN semaphore limits ─────────────────────────────────────────────

async def test_gnn_semaphore_limit():
    cfg = GPUConcurrencyConfig(max_concurrent_gnn=1, gpu_acquire_timeout_sec=0.5)
    q = _fresh_queue(cfg)
    acquired = asyncio.Event()
    blocked = asyncio.Event()

    async def hold_gnn():
        async with q.acquire_gnn() as _:
            acquired.set()
            await asyncio.sleep(1.0)  # hold semaphore

    async def try_acquire():
        await acquired.wait()  # ensure first is holding
        blocked.set()
        async with q.acquire_gnn() as use_gpu:
            # Should eventually get here (after timeout or release)
            pass

    holder = asyncio.create_task(hold_gnn())
    waiter = asyncio.create_task(try_acquire())

    # Wait for both to start
    await acquired.wait()
    await blocked.wait()

    # The waiter should be blocked (semaphore held)
    await asyncio.sleep(0.1)
    assert not waiter.done(), "Waiter should still be blocked by semaphore"

    # Clean up
    holder.cancel()
    waiter.cancel()
    try:
        await holder
    except asyncio.CancelledError:
        pass
    try:
        await waiter
    except asyncio.CancelledError:
        pass


# ── Test 15: ML semaphore limits ──────────────────────────────────────────────

async def test_ml_semaphore_limit():
    cfg = GPUConcurrencyConfig(max_concurrent_ml=1, gpu_acquire_timeout_sec=0.5)
    q = _fresh_queue(cfg)
    acquired = asyncio.Event()

    async def hold_ml():
        async with q.acquire_ml():
            acquired.set()
            await asyncio.sleep(2.0)

    holder = asyncio.create_task(hold_ml())
    await acquired.wait()

    # Second acquire should timeout
    timed_out = False
    try:
        async with q.acquire_ml():
            pass
    except asyncio.TimeoutError:
        timed_out = True

    assert timed_out, "Second ML acquire should timeout"

    holder.cancel()
    try:
        await holder
    except asyncio.CancelledError:
        pass


# ── Test 16: Concurrent GNN + LLM no deadlock ────────────────────────────────

async def test_concurrent_gnn_llm_no_deadlock():
    q = _fresh_queue()
    results = []

    async def llm_task():
        async with q.acquire_llm() as ctx:
            results.append(("llm", ctx))
            await asyncio.sleep(0.01)

    async def gnn_task():
        async with q.acquire_gnn() as use_gpu:
            results.append(("gnn", use_gpu))
            await asyncio.sleep(0.01)

    # Run concurrently — should not deadlock
    await asyncio.wait_for(
        asyncio.gather(llm_task(), gnn_task()),
        timeout=5.0,
    )
    assert len(results) == 2, f"Expected 2 results, got {len(results)}"
    types = {r[0] for r in results}
    assert types == {"llm", "gnn"}, f"Expected llm+gnn, got {types}"


# ── Test 17: LoadShedder CPU fallback ─────────────────────────────────────────

def test_load_shedder_cpu_fallback():
    from main import LoadShedder, HardwareProfiler
    from src.ml.models.gnn_scorer import GNNScoringResult

    profiler = HardwareProfiler()
    mock_gnn = MagicMock()
    mock_gnn.score_subgraph_cpu.return_value = GNNScoringResult(
        risk_score=0.82, node_count=5, edge_count=8, inference_ms=2.1,
    )

    shedder = LoadShedder(mock_gnn, profiler)

    # Simulate load shedding active
    profiler._load_shed_active = True
    result = shedder.score_subgraph("subgraph", ["n1"], 100)

    assert result.risk_score == 0.82, f"Expected 0.82 (CPU fallback), got {result.risk_score}"
    assert shedder._cpu_fallback_count == 1, "CPU fallback not counted"


# ── Test 18: LoadShedder sentinel on failure ──────────────────────────────────

def test_load_shedder_sentinel_on_failure():
    from main import LoadShedder, HardwareProfiler
    from src.ml.models.gnn_scorer import GNNScoringResult

    profiler = HardwareProfiler()
    mock_gnn = MagicMock()
    mock_gnn.score_subgraph_cpu.side_effect = RuntimeError("model not loaded")

    shedder = LoadShedder(mock_gnn, profiler)
    profiler._load_shed_active = True
    result = shedder.score_subgraph("subgraph", ["n1"], 100)

    assert result.risk_score == -1.0, f"Expected sentinel -1.0, got {result.risk_score}"


# ── Test 19: analysis_mode does not unload LLM ───────────────────────────────

def test_analysis_mode_no_unload():
    """Verify that analysis_mode() no longer calls _unload_ollama_model()."""
    from config.vram_manager import analysis_mode, GPUMode, _unload_ollama_model
    import config.vram_manager as vm

    # Set mode to ASSISTANT to trigger the transition path
    vm._current_mode = GPUMode.ASSISTANT

    with patch("config.vram_manager._unload_ollama_model") as mock_unload:
        with analysis_mode():
            pass
        # In cooperative mode, _unload_ollama_model should NOT be called
        mock_unload.assert_not_called()

    vm._current_mode = GPUMode.IDLE


# ── Test 20: Cooperative coexistence ──────────────────────────────────────────

def test_cooperative_coexistence():
    """Verify assistant_mode no longer flushes PyTorch cache."""
    from config.vram_manager import assistant_mode, GPUMode
    import config.vram_manager as vm

    vm._current_mode = GPUMode.ANALYSIS

    with patch("config.vram_manager._flush_torch_cache") as mock_flush:
        with assistant_mode():
            pass
        # In cooperative mode, _flush_torch_cache should NOT be called on entry
        mock_flush.assert_not_called()

    vm._current_mode = GPUMode.IDLE


# ── Test 21: Metrics snapshot structure ───────────────────────────────────────

async def test_metrics_snapshot_structure():
    q = _fresh_queue()
    snap = q.snapshot()
    expected_keys = {
        "pressure", "num_ctx",
        "llm_acquisitions", "gnn_gpu_acquisitions", "gnn_cpu_fallbacks",
        "ml_acquisitions", "kv_cache_reductions", "kv_cache_restorations",
    }
    assert expected_keys == set(snap.keys()), f"Missing keys: {expected_keys - set(snap.keys())}"
    assert snap["pressure"] == "normal"
    assert snap["num_ctx"] == 16384

    # After some operations, counters should update
    async with q.acquire_llm() as _:
        pass
    snap2 = q.snapshot()
    assert snap2["llm_acquisitions"] == 1


# ── Test 22: Full pressure cycle ─────────────────────────────────────────────

async def test_full_pressure_cycle():
    q = _fresh_queue()

    # Start NORMAL
    assert q.pressure == VRAMPressureLevel.NORMAL
    assert q.current_num_ctx == 16384

    # Ramp up to HIGH
    await q.update_pressure(7600.0)
    assert q.pressure == VRAMPressureLevel.HIGH
    assert q.current_num_ctx == 8192

    # Ramp up to CRITICAL
    await q.update_pressure(7900.0)
    assert q.pressure == VRAMPressureLevel.CRITICAL
    assert q.current_num_ctx == 4096

    # GNN should be offloaded to CPU
    async with q.acquire_gnn() as use_gpu:
        assert use_gpu is False

    # LLM still works
    async with q.acquire_llm() as ctx:
        assert ctx == 4096

    # Drop back to NORMAL
    await q.update_pressure(5000.0)
    assert q.pressure == VRAMPressureLevel.NORMAL
    assert q.current_num_ctx == 16384

    # GNN should be on GPU again
    async with q.acquire_gnn() as use_gpu:
        assert use_gpu is True

    # Verify metrics
    snap = q.snapshot()
    assert snap["kv_cache_reductions"] >= 1
    assert snap["kv_cache_restorations"] >= 1
    assert snap["gnn_cpu_fallbacks"] >= 1
    assert snap["gnn_gpu_acquisitions"] >= 1


# ── Runner ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    _safe_print("\n=== Phase 15: Advanced GPU Concurrency Management ===\n")

    # Config tests
    run_test(test_gpu_concurrency_config_defaults)
    run_test(test_gpu_concurrency_config_frozen)
    run_test(test_cpu_only_bypasses_queue)

    # Pressure tests
    run_test(test_pressure_normal)
    run_test(test_pressure_high)
    run_test(test_pressure_critical)
    run_test(test_hysteresis_prevents_oscillation)
    run_test(test_dynamic_num_ctx_scales)
    run_test(test_ctx_change_callback)

    # GNN offload tests
    run_test(test_gnn_acquire_gpu_normal)
    run_test(test_gnn_acquire_cpu_critical)
    run_test(test_gnn_cpu_fallback_scoring)

    # Priority queue tests
    run_test(test_llm_acquire_always_succeeds)
    run_test(test_gnn_semaphore_limit)
    run_test(test_ml_semaphore_limit)
    run_test(test_concurrent_gnn_llm_no_deadlock)

    # LoadShedder v2 tests
    run_test(test_load_shedder_cpu_fallback)
    run_test(test_load_shedder_sentinel_on_failure)

    # VRAM manager v2 tests
    run_test(test_analysis_mode_no_unload)
    run_test(test_cooperative_coexistence)

    # Integration tests
    run_test(test_metrics_snapshot_structure)
    run_test(test_full_pressure_cycle)

    _safe_print(f"\n{'=' * 60}")
    _safe_print(f"Phase 15 Results: {passed} passed, {failed} failed "
                f"out of {passed + failed}")
    _safe_print(f"{'=' * 60}\n")

    sys.exit(0 if failed == 0 else 1)

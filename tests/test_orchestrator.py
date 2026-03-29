"""
PayFlow — Tests for Unified Execution Engine (main.py)
=======================================================
Covers: HardwareProfiler, LoadShedder, PayFlowOrchestrator, CLI parsing.

All heavy module dependencies are mocked so these tests run instantly
without GPU, Ollama, or trained models.

Run:
    PYTHONIOENCODING=utf-8 PYTHONPATH=. python tests/test_orchestrator.py
"""

from __future__ import annotations

import asyncio
import os
import sys
import time
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock
from typing import NamedTuple

# Force CPU mode before anything else
os.environ["PAYFLOW_CPU_ONLY"] = "1"

# ── Project root on path ─────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ── Import units under test ──────────────────────────────────────────────────
from main import (
    HardwareSnapshot,
    HardwareProfiler,
    LoadShedder,
    OrchestratorMetrics,
    PayFlowOrchestrator,
    _build_parser,
)

# ── Helpers ──────────────────────────────────────────────────────────────────

passed = 0
failed = 0


def _safe_print(msg: str) -> None:
    try:
        print(msg)
    except UnicodeEncodeError:
        print(msg.encode("ascii", errors="replace").decode())


def run_test(func):
    """Run a sync or async test, update counters."""
    global passed, failed
    name = func.__name__
    try:
        if asyncio.iscoroutinefunction(func):
            asyncio.run(func())
        else:
            func()
        _safe_print(f"  PASS  {name}")
        passed += 1
    except Exception as exc:
        _safe_print(f"  FAIL  {name}: {exc}")
        failed += 1


# ═══════════════════════════════════════════════════════════════════════════════
# 1. HardwareSnapshot
# ═══════════════════════════════════════════════════════════════════════════════

def test_hardware_snapshot_defaults():
    """HardwareSnapshot initialises with sensible zero-state defaults."""
    snap = HardwareSnapshot(timestamp=time.time())
    assert snap.gpu_vram_used_mb == 0.0
    assert snap.gpu_vram_total_mb == 0.0
    assert snap.gpu_vram_free_mb == 0.0
    assert snap.gpu_utilization_pct == -1
    assert snap.gpu_temperature_c == -1
    assert snap.cpu_utilization_pct == 0.0
    assert snap.llm_tokens_generated == 0
    assert snap.llm_tps == 0.0
    assert snap.load_shed_active is False


# ═══════════════════════════════════════════════════════════════════════════════
# 2. HardwareProfiler — unit tests (no polling)
# ═══════════════════════════════════════════════════════════════════════════════

def test_profiler_initial_state():
    """Fresh profiler has shedding off and zero tokens."""
    p = HardwareProfiler(vram_shed_threshold_mb=7500, vram_resume_threshold_mb=6500)
    assert p.load_shed_active is False
    snap = p.snapshot()
    assert snap["llm_tokens_total"] == 0
    assert snap["load_shed_active"] is False


def test_profiler_record_tokens():
    """record_tokens() accumulates correctly."""
    p = HardwareProfiler()
    p.record_tokens(100)
    p.record_tokens(50)
    snap = p.snapshot()
    assert snap["llm_tokens_total"] == 150


def test_profiler_shed_trigger():
    """_evaluate_load_shedding triggers when VRAM exceeds threshold."""
    p = HardwareProfiler(vram_shed_threshold_mb=7500, vram_resume_threshold_mb=6500)

    on_shed = asyncio.Event()
    on_resume = asyncio.Event()
    p.set_shed_events(on_shed, on_resume)

    # Below threshold — no shed
    snap_ok = HardwareSnapshot(
        timestamp=time.time(),
        gpu_vram_used_mb=6000.0,
        gpu_vram_total_mb=8192.0,
    )
    p._evaluate_load_shedding(snap_ok)
    assert p.load_shed_active is False
    assert not on_shed.is_set()

    # Exceed threshold — triggers shed
    snap_high = HardwareSnapshot(
        timestamp=time.time(),
        gpu_vram_used_mb=7600.0,
        gpu_vram_total_mb=8192.0,
    )
    p._evaluate_load_shedding(snap_high)
    assert p.load_shed_active is True
    assert on_shed.is_set()
    assert not on_resume.is_set()


def test_profiler_shed_hysteresis():
    """VRAM must drop below resume threshold (not just shed threshold) to clear."""
    p = HardwareProfiler(vram_shed_threshold_mb=7500, vram_resume_threshold_mb=6500)

    on_shed = asyncio.Event()
    on_resume = asyncio.Event()
    p.set_shed_events(on_shed, on_resume)

    # Trigger shed
    p._evaluate_load_shedding(HardwareSnapshot(
        timestamp=time.time(), gpu_vram_used_mb=7600, gpu_vram_total_mb=8192,
    ))
    assert p.load_shed_active is True

    # Drop to 7000 (below shed, but ABOVE resume) — still shedding
    p._evaluate_load_shedding(HardwareSnapshot(
        timestamp=time.time(), gpu_vram_used_mb=7000, gpu_vram_total_mb=8192,
    ))
    assert p.load_shed_active is True, "Should still be shedding (7000 > 6500 resume)"

    # Drop to 6400 (below resume) — clears
    p._evaluate_load_shedding(HardwareSnapshot(
        timestamp=time.time(), gpu_vram_used_mb=6400, gpu_vram_total_mb=8192,
    ))
    assert p.load_shed_active is False
    assert on_resume.is_set()
    assert not on_shed.is_set()


def test_profiler_no_shed_without_gpu():
    """No load-shedding when no GPU data (total=0)."""
    p = HardwareProfiler(vram_shed_threshold_mb=7500, vram_resume_threshold_mb=6500)
    snap = HardwareSnapshot(timestamp=time.time(), gpu_vram_used_mb=0, gpu_vram_total_mb=0)
    p._evaluate_load_shedding(snap)
    assert p.load_shed_active is False


async def test_profiler_start_stop():
    """Profiler starts and stops cleanly."""
    p = HardwareProfiler(poll_interval_sec=0.05)
    await p.start()
    assert p._running is True
    await asyncio.sleep(0.15)  # Let 2-3 polls run
    await p.stop()
    assert p._running is False


def test_profiler_snapshot_format():
    """snapshot() returns correctly-keyed dict."""
    p = HardwareProfiler()
    snap = p.snapshot()
    expected_keys = {
        "gpu_vram_used_mb", "gpu_vram_total_mb", "gpu_vram_free_mb",
        "gpu_utilization_pct", "gpu_temperature_c", "cpu_utilization_pct",
        "llm_tps", "llm_tokens_total", "load_shed_active",
    }
    assert set(snap.keys()) == expected_keys


# ═══════════════════════════════════════════════════════════════════════════════
# 3. LoadShedder
# ═══════════════════════════════════════════════════════════════════════════════

def test_load_shedder_passthrough():
    """When profiler is not shedding, calls go through to the real GNN."""
    mock_gnn = MagicMock()
    mock_profiler = MagicMock(spec=HardwareProfiler)
    mock_profiler.load_shed_active = False

    # Fake GNNScoringResult
    class FakeResult(NamedTuple):
        risk_score: float
        node_count: int
        edge_count: int
        inference_ms: float

    mock_gnn.score_subgraph.return_value = FakeResult(0.75, 10, 20, 5.2)

    shedder = LoadShedder(mock_gnn, mock_profiler)
    result = shedder.score_subgraph("subgraph", ["n1"], 1000)

    assert result.risk_score == 0.75
    mock_gnn.score_subgraph.assert_called_once_with("subgraph", ["n1"], 1000)
    assert shedder.snapshot()["pass_count"] == 1
    assert shedder.snapshot()["shed_count"] == 0


def test_load_shedder_sentinel():
    """When profiler is shedding, returns sentinel GNNScoringResult(-1.0)."""
    mock_gnn = MagicMock()
    mock_profiler = MagicMock(spec=HardwareProfiler)
    mock_profiler.load_shed_active = True

    shedder = LoadShedder(mock_gnn, mock_profiler)
    result = shedder.score_subgraph("sg", ["n1"], 500)

    assert result.risk_score == -1.0
    assert result.node_count == 0
    assert result.inference_ms == 0.0
    # GNN should NOT be called when shedding
    mock_gnn.score_subgraph.assert_not_called()
    assert shedder.snapshot()["shed_count"] == 1
    assert shedder.snapshot()["currently_shedding"] is True


def test_load_shedder_attribute_forwarding():
    """LoadShedder forwards unknown attributes to underlying GNN."""
    mock_gnn = MagicMock()
    mock_gnn.device = "cuda:0"
    mock_gnn.model_name = "gnn_v2"
    mock_profiler = MagicMock(spec=HardwareProfiler)
    mock_profiler.load_shed_active = False

    shedder = LoadShedder(mock_gnn, mock_profiler)
    assert shedder.device == "cuda:0"
    assert shedder.model_name == "gnn_v2"


# ═══════════════════════════════════════════════════════════════════════════════
# 4. OrchestratorMetrics
# ═══════════════════════════════════════════════════════════════════════════════

def test_metrics_elapsed_and_throughput():
    """elapsed_sec and events_per_sec compute correctly."""
    m = OrchestratorMetrics()
    m.pipeline_start_time = 1000.0
    m.pipeline_end_time = 1010.0
    m.events_ingested = 5000

    assert m.elapsed_sec == 10.0
    assert m.events_per_sec == 500.0


def test_metrics_zero_elapsed():
    """No division-by-zero when pipeline hasn't started."""
    m = OrchestratorMetrics()
    assert m.elapsed_sec == 0.0
    assert m.events_per_sec == 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# 5. PayFlowOrchestrator — mocked integration
# ═══════════════════════════════════════════════════════════════════════════════

def _make_mock_ledger():
    """Create a convincing AuditLedger mock."""
    ledger = AsyncMock()
    ledger.open = AsyncMock()
    ledger.close = AsyncMock()
    ledger.anchor_system_event = AsyncMock()
    ledger.anchor_alert = AsyncMock()
    ledger.anchor_model_event = AsyncMock()
    ledger.ingest = AsyncMock()
    ledger.head_index = 0

    # verify_chain returns a valid result
    verification = MagicMock()
    verification.valid = True
    verification.blocks_checked = 10
    ledger.verify_chain = AsyncMock(return_value=verification)

    return ledger


def _make_mock_breaker():
    """Create a CircuitBreaker mock."""
    breaker = MagicMock()
    breaker.on_alert = AsyncMock()
    breaker.cleanup_expired = AsyncMock(return_value=0)
    breaker.get_frozen_nodes.return_value = []
    breaker.snapshot.return_value = {"frozen_count": 0}
    breaker._cfg = MagicMock()
    breaker._cfg.consensus_threshold = 0.7
    return breaker


def _make_mock_gnn():
    """Create a GNNScorer mock."""
    gnn = MagicMock()
    gnn.device = "cpu"
    from src.ml.models.gnn_scorer import GNNScoringResult
    gnn.score_subgraph.return_value = GNNScoringResult(
        risk_score=0.5, node_count=5, edge_count=8, inference_ms=2.0,
    )
    return gnn


def _make_mock_engine():
    """Create a FeatureEngine mock."""
    import numpy as np
    engine = MagicMock()
    engine.ingest = AsyncMock()
    engine.metrics = MagicMock()
    engine.metrics.batches_processed = 5

    # get_training_data() returns a mock with features/labels
    train_data = MagicMock()
    train_data.features = np.random.rand(200, 30).astype(np.float32)
    train_data.labels = np.random.randint(0, 2, size=200).astype(np.float32)
    train_data.txn_ids = [f"txn_{i}" for i in range(200)]
    train_data.sender_ids = [f"sender_{i % 20}" for i in range(200)]
    train_data.receiver_ids = [f"receiver_{i % 20}" for i in range(200)]
    train_data.timestamps = np.arange(200, dtype=np.int64)
    engine.get_training_data.return_value = train_data

    return engine


def _make_mock_classifier():
    """Create a FraudClassifier mock."""
    import numpy as np
    clf = MagicMock()
    clf.device = "cpu"
    clf.is_fitted = True

    # train() result
    train_result = MagicMock()
    train_result.best_aucpr = 0.92
    train_result.best_iteration = 100
    train_result.n_train = 160
    train_result.n_eval = 40
    train_result.device_used = "cpu"
    clf.train.return_value = train_result

    # validate() result
    val_result = MagicMock()
    val_result.aucpr = 0.91
    val_result.auc_roc = 0.95
    val_result.f1 = 0.88
    val_result.precision = 0.90
    val_result.recall = 0.86
    clf.validate.return_value = val_result

    # predict() result
    pred_result = MagicMock()
    pred_result.risk_scores = np.random.rand(200).astype(np.float32)
    pred_result.predicted_labels = (pred_result.risk_scores > 0.5).astype(np.float32)
    pred_result.inference_ms = 12.5
    clf.predict.return_value = pred_result

    return clf


def _make_mock_threshold():
    """Create a DynamicThreshold mock."""
    threshold = MagicMock()
    threshold.current_threshold = 0.5
    threshold.snapshot.return_value = {"threshold": 0.5}

    # evaluate_batch returns ThresholdResult-like objects
    def _eval_batch(scores):
        results = []
        for i, s in enumerate(scores):
            r = MagicMock()
            if s > 0.7:
                r.tier = "HIGH"
            elif s > 0.4:
                r.tier = "MEDIUM"
            else:
                r.tier = "LOW"
            r.score = float(s)
            r.txn_index = i
            results.append(r)
        return results

    threshold.evaluate_batch.side_effect = _eval_batch
    return threshold


def _make_mock_router():
    """Create an AlertRouter mock."""
    router = MagicMock()
    router.register_graph_consumer = MagicMock()
    router.register_ledger_consumer = MagicMock()
    router.register_circuit_breaker_consumer = MagicMock()
    router.register_agent_consumer = MagicMock()
    router.route = AsyncMock()
    router.shutdown = AsyncMock()
    router.metrics = MagicMock()
    return router


def _make_mock_pipeline():
    """Create an IngestionPipeline mock."""
    pipeline = MagicMock()
    pipeline.add_consumer = MagicMock()
    pipeline.ingest = AsyncMock()
    pipeline.start = AsyncMock()
    pipeline.stop = AsyncMock()
    pipeline.__aenter__ = AsyncMock(return_value=pipeline)
    pipeline.__aexit__ = AsyncMock(return_value=False)
    pipeline.metrics = MagicMock()
    pipeline.metrics.events_ingested = 100
    return pipeline


async def test_orchestrator_initialize_wires_consumers():
    """initialize() wires the correct consumers to router and pipeline."""
    orch = PayFlowOrchestrator(
        num_events=10, num_accounts=5, skip_llm=True, cpu_only=True,
        profiler_interval=60.0,  # don't actually poll
    )

    mock_ledger = _make_mock_ledger()
    mock_breaker = _make_mock_breaker()
    mock_gnn = _make_mock_gnn()
    mock_graph = MagicMock()
    mock_graph.investigate = AsyncMock()
    mock_graph.ingest = AsyncMock()
    mock_graph.metrics = MagicMock(nodes_added=0, transactions_added=0)
    mock_graph.snapshot.return_value = {}
    mock_engine = _make_mock_engine()
    mock_classifier = _make_mock_classifier()
    mock_threshold = _make_mock_threshold()
    mock_router = _make_mock_router()
    mock_pipeline = _make_mock_pipeline()

    mock_world = MagicMock()
    mock_world.accounts = [MagicMock()] * 5
    mock_world.dormant_accounts = []

    # Patch at source modules (inline imports target original locations)
    with patch("src.blockchain.ledger.AuditLedger", return_value=mock_ledger), \
         patch("src.blockchain.circuit_breaker.CircuitBreaker", return_value=mock_breaker), \
         patch("src.ml.models.gnn_scorer.GNNScorer", return_value=mock_gnn), \
         patch("src.graph.builder.TransactionGraph", return_value=mock_graph), \
         patch("src.ml.feature_engine.FeatureEngine", return_value=mock_engine), \
         patch("src.ml.models.xgboost_classifier.FraudClassifier", return_value=mock_classifier), \
         patch("src.ml.models.threshold.DynamicThreshold", return_value=mock_threshold), \
         patch("src.ml.models.alert_router.AlertRouter", return_value=mock_router), \
         patch("src.ingestion.stream_processor.IngestionPipeline", return_value=mock_pipeline), \
         patch("src.ingestion.generators.synthetic_transactions.build_world", return_value=mock_world):

        await orch.initialize()

        # Router should have 3 consumers (no agent since skip_llm)
        mock_router.register_graph_consumer.assert_called_once()
        mock_router.register_ledger_consumer.assert_called_once()
        mock_router.register_circuit_breaker_consumer.assert_called_once()
        mock_router.register_agent_consumer.assert_not_called()

        # Pipeline should have 3 consumers
        assert mock_pipeline.add_consumer.call_count == 3

        # Ledger should be opened
        mock_ledger.open.assert_awaited_once()

        await orch.profiler.stop()


async def test_orchestrator_shutdown_verifies_chain():
    """shutdown() verifies ledger chain integrity and closes ledger."""
    orch = PayFlowOrchestrator(skip_llm=True, cpu_only=True)
    mock_ledger = _make_mock_ledger()
    mock_router = _make_mock_router()

    orch._ledger = mock_ledger
    orch._router = mock_router
    orch._llm = None

    await orch.shutdown()

    mock_ledger.anchor_system_event.assert_awaited()
    mock_ledger.verify_chain.assert_awaited_once()
    mock_ledger.close.assert_awaited_once()
    mock_router.shutdown.assert_awaited_once()


async def test_orchestrator_context_manager():
    """async with PayFlowOrchestrator runs initialize + shutdown."""
    orch = PayFlowOrchestrator(
        num_events=5, num_accounts=3, skip_llm=True, cpu_only=True,
        profiler_interval=60.0,
    )

    mock_ledger = _make_mock_ledger()
    mock_breaker = _make_mock_breaker()
    mock_gnn = _make_mock_gnn()
    mock_graph = MagicMock()
    mock_graph.investigate = AsyncMock()
    mock_graph.ingest = AsyncMock()
    mock_graph.metrics = MagicMock(nodes_added=0, transactions_added=0)
    mock_graph.snapshot.return_value = {}
    mock_engine = _make_mock_engine()
    mock_classifier = _make_mock_classifier()
    mock_threshold = _make_mock_threshold()
    mock_router = _make_mock_router()
    mock_pipeline = _make_mock_pipeline()

    mock_world = MagicMock()
    mock_world.accounts = [MagicMock()] * 3
    mock_world.dormant_accounts = []

    with patch("src.blockchain.ledger.AuditLedger", return_value=mock_ledger), \
         patch("src.blockchain.circuit_breaker.CircuitBreaker", return_value=mock_breaker), \
         patch("src.ml.models.gnn_scorer.GNNScorer", return_value=mock_gnn), \
         patch("src.graph.builder.TransactionGraph", return_value=mock_graph), \
         patch("src.ml.feature_engine.FeatureEngine", return_value=mock_engine), \
         patch("src.ml.models.xgboost_classifier.FraudClassifier", return_value=mock_classifier), \
         patch("src.ml.models.threshold.DynamicThreshold", return_value=mock_threshold), \
         patch("src.ml.models.alert_router.AlertRouter", return_value=mock_router), \
         patch("src.ingestion.stream_processor.IngestionPipeline", return_value=mock_pipeline), \
         patch("src.ingestion.generators.synthetic_transactions.build_world", return_value=mock_world):

        async with orch:
            # Inside context — modules should be initialised
            assert orch._ledger is not None

    # After exit — ledger should be closed
    mock_ledger.close.assert_awaited_once()


async def test_orchestrator_full_snapshot():
    """full_snapshot() returns correctly structured diagnostics dict."""
    orch = PayFlowOrchestrator(skip_llm=True, cpu_only=True)
    orch._gnn_proxy = MagicMock()
    orch._gnn_proxy.snapshot.return_value = {"shed_count": 0, "pass_count": 10}
    orch._graph = MagicMock()
    orch._graph.snapshot.return_value = {"nodes": 50, "edges": 120}
    orch._breaker = MagicMock()
    orch._breaker.snapshot.return_value = {"frozen_count": 2}
    orch._threshold = MagicMock()
    orch._threshold.snapshot.return_value = {"threshold": 0.5}
    orch._agent = None  # skip_llm

    snap = orch.full_snapshot()

    assert "orchestrator" in snap
    assert "hardware" in snap
    assert "load_shedder" in snap
    assert "graph" in snap
    assert "circuit_breaker" in snap
    assert "threshold" in snap
    assert "agent" not in snap  # no agent when skip_llm


# ═══════════════════════════════════════════════════════════════════════════════
# 6. CLI Argument Parsing
# ═══════════════════════════════════════════════════════════════════════════════

def test_cli_defaults():
    """Parser defaults match expected values."""
    parser = _build_parser()
    args = parser.parse_args([])
    assert args.events == 10_000
    assert args.accounts == 2000
    assert args.fraud_ratio == 0.05
    assert args.batch_size == 512
    assert args.cpu_only is False
    assert args.skip_llm is False
    assert args.vram_shed_mb == 7500.0
    assert args.vram_resume_mb == 6500.0


def test_cli_custom_values():
    """Parser correctly reads custom arguments."""
    parser = _build_parser()
    args = parser.parse_args([
        "--events", "500",
        "--accounts", "50",
        "--fraud-ratio", "0.10",
        "--batch-size", "64",
        "--cpu-only",
        "--skip-llm",
        "--vram-shed-mb", "8000",
        "--vram-resume-mb", "7000",
    ])
    assert args.events == 500
    assert args.accounts == 50
    assert args.fraud_ratio == 0.10
    assert args.batch_size == 64
    assert args.cpu_only is True
    assert args.skip_llm is True
    assert args.vram_shed_mb == 8000.0
    assert args.vram_resume_mb == 7000.0


# ═══════════════════════════════════════════════════════════════════════════════
# 7. Phase execution — mocked
# ═══════════════════════════════════════════════════════════════════════════════

async def test_phase_ingest():
    """_phase_ingest streams events through the pipeline."""
    orch = PayFlowOrchestrator(num_events=20, num_accounts=5, skip_llm=True, cpu_only=True)

    mock_pipeline = _make_mock_pipeline()
    mock_engine = _make_mock_engine()
    mock_graph = MagicMock()
    mock_graph.metrics = MagicMock(nodes_added=15, transactions_added=25)
    mock_ledger = _make_mock_ledger()

    orch._pipeline = mock_pipeline
    orch._engine = mock_engine
    orch._graph = mock_graph
    orch._ledger = mock_ledger

    # Mock generate_event_stream
    fake_events = [MagicMock() for _ in range(20)]
    with patch("src.ingestion.generators.synthetic_transactions.generate_event_stream",
               return_value=fake_events):
        mock_world = MagicMock()
        orch._world = mock_world
        await orch._phase_ingest()

    # Pipeline ingest should be called once per event
    assert mock_pipeline.ingest.await_count == 20
    assert orch.metrics.events_ingested == 100  # from pipeline.metrics mock


async def test_phase_train():
    """_phase_train trains XGBoost and anchors to ledger."""
    orch = PayFlowOrchestrator(skip_llm=True, cpu_only=True)

    mock_engine = _make_mock_engine()
    mock_classifier = _make_mock_classifier()
    mock_ledger = _make_mock_ledger()

    orch._engine = mock_engine
    orch._classifier = mock_classifier
    orch._ledger = mock_ledger

    with patch("config.vram_manager.analysis_mode") as mock_ctx:
        # Make analysis_mode() a real context manager
        from contextlib import contextmanager
        @contextmanager
        def _fake_analysis():
            yield
        mock_ctx.side_effect = _fake_analysis

        await orch._phase_train()

    mock_classifier.train.assert_called_once()
    mock_classifier.validate.assert_called_once()
    mock_ledger.anchor_model_event.assert_awaited_once()


async def test_phase_inference():
    """_phase_inference runs prediction, thresholding, and routing."""
    import numpy as np
    from src.ml.feature_engine import FEATURE_COLUMNS

    orch = PayFlowOrchestrator(skip_llm=True, cpu_only=True)

    mock_engine = _make_mock_engine()
    mock_classifier = _make_mock_classifier()
    mock_threshold = _make_mock_threshold()
    mock_router = _make_mock_router()
    mock_breaker = _make_mock_breaker()
    mock_ledger = _make_mock_ledger()

    orch._engine = mock_engine
    orch._classifier = mock_classifier
    orch._threshold = mock_threshold
    orch._router = mock_router
    orch._breaker = mock_breaker
    orch._ledger = mock_ledger

    # Patch AlertRouter.build_payloads at class level (static method)
    fake_payloads = []
    for i in range(5):
        p = MagicMock()
        p.tier = "HIGH" if i < 2 else "MEDIUM"
        fake_payloads.append(p)

    with patch("config.vram_manager.analysis_mode") as mock_ctx, \
         patch("src.ml.models.alert_router.AlertRouter") as mock_router_cls:

        from contextlib import contextmanager
        @contextmanager
        def _fake_analysis():
            yield
        mock_ctx.side_effect = _fake_analysis
        mock_router_cls.build_payloads.return_value = fake_payloads

        await orch._phase_inference()

    assert orch.metrics.ml_inferences == 200  # 200 features in mock engine
    assert orch.metrics.alerts_routed == 5
    assert mock_router.route.await_count == 5


# ═══════════════════════════════════════════════════════════════════════════════
# 8. Load-shedding integration
# ═══════════════════════════════════════════════════════════════════════════════

def test_load_shedder_transition_counts():
    """LoadShedder tracks shed/pass counts correctly across transitions."""
    from src.ml.models.gnn_scorer import GNNScoringResult

    mock_gnn = MagicMock()
    mock_gnn.score_subgraph.return_value = GNNScoringResult(0.5, 5, 8, 2.0)

    profiler = HardwareProfiler(vram_shed_threshold_mb=7500, vram_resume_threshold_mb=6500)
    shedder = LoadShedder(mock_gnn, profiler)

    # Normal scoring (2 calls)
    shedder.score_subgraph("sg", ["n1"], 100)
    shedder.score_subgraph("sg", ["n2"], 200)
    assert shedder.snapshot()["pass_count"] == 2
    assert shedder.snapshot()["shed_count"] == 0

    # Trigger shedding
    profiler._evaluate_load_shedding(HardwareSnapshot(
        timestamp=time.time(), gpu_vram_used_mb=7800, gpu_vram_total_mb=8192,
    ))

    # Shed scoring (3 calls)
    for i in range(3):
        result = shedder.score_subgraph("sg", [f"n{i}"], 300)
        assert result.risk_score == -1.0

    assert shedder.snapshot()["pass_count"] == 2
    assert shedder.snapshot()["shed_count"] == 3

    # Resume
    profiler._evaluate_load_shedding(HardwareSnapshot(
        timestamp=time.time(), gpu_vram_used_mb=6000, gpu_vram_total_mb=8192,
    ))

    # Normal again
    shedder.score_subgraph("sg", ["n10"], 400)
    assert shedder.snapshot()["pass_count"] == 3
    assert shedder.snapshot()["shed_count"] == 3


# ═══════════════════════════════════════════════════════════════════════════════
# Runner
# ═══════════════════════════════════════════════════════════════════════════════

ALL_TESTS = [
    # 1. HardwareSnapshot
    test_hardware_snapshot_defaults,
    # 2. HardwareProfiler
    test_profiler_initial_state,
    test_profiler_record_tokens,
    test_profiler_shed_trigger,
    test_profiler_shed_hysteresis,
    test_profiler_no_shed_without_gpu,
    test_profiler_start_stop,
    test_profiler_snapshot_format,
    # 3. LoadShedder
    test_load_shedder_passthrough,
    test_load_shedder_sentinel,
    test_load_shedder_attribute_forwarding,
    # 4. OrchestratorMetrics
    test_metrics_elapsed_and_throughput,
    test_metrics_zero_elapsed,
    # 5. PayFlowOrchestrator
    test_orchestrator_initialize_wires_consumers,
    test_orchestrator_shutdown_verifies_chain,
    test_orchestrator_context_manager,
    test_orchestrator_full_snapshot,
    # 6. CLI
    test_cli_defaults,
    test_cli_custom_values,
    # 7. Phases
    test_phase_ingest,
    test_phase_train,
    test_phase_inference,
    # 8. Load-shedding integration
    test_load_shedder_transition_counts,
]


def main_runner():
    _safe_print(f"\n{'=' * 60}")
    _safe_print("  PayFlow Orchestrator Tests (main.py)")
    _safe_print(f"{'=' * 60}\n")

    for test_fn in ALL_TESTS:
        run_test(test_fn)

    _safe_print(f"\n{'=' * 60}")
    _safe_print(f"  Results: {passed} passed, {failed} failed, {passed + failed} total")
    _safe_print(f"{'=' * 60}\n")

    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main_runner()

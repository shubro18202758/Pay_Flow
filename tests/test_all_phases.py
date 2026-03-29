"""
PayFlow — Comprehensive Phase 1–7 Functional Verification
============================================================
Exercises every module across all 7 implementation phases:
  Phase 1: Config, VRAM Budget, VRAM Manager
  Phase 2: Schemas, Validators, Generators, Stream Processor
  Phase 3: Velocity, Behavioral, Text Anomaly, Feature Engine
  Phase 4: XGBoost Classifier, Dynamic Threshold, Alert Router
  Phase 5: Graph Builder, Mule Detector, Cycle Detector
  Phase 6: GNN Scorer (model build + nx→PyG conversion), LLM module imports
  Phase 7: Audit Ledger (crypto, storage, ledger, verification)

Tests are self-contained: no Ollama, no GPU, no external services required.
"""

from __future__ import annotations

import asyncio
import math
import os
import sys
import tempfile
import time

# Ensure CPU-only mode for reproducible testing
os.environ["PAYFLOW_CPU_ONLY"] = "1"

PASSED = 0
FAILED = 0
ERRORS: list[str] = []


def _safe_print(text: str) -> None:
    """Print with fallback for Windows cp1252 console."""
    try:
        print(text)
    except UnicodeEncodeError:
        print(text.encode("ascii", errors="replace").decode("ascii"))


def test(name: str):
    """Decorator that catches exceptions and tallies pass/fail."""
    def decorator(fn):
        def wrapper():
            global PASSED, FAILED
            try:
                fn()
                PASSED += 1
                _safe_print(f"  [PASS] {name}")
            except Exception as exc:
                FAILED += 1
                ERRORS.append(f"{name}: {exc}")
                _safe_print(f"  [FAIL] {name} -- {exc}")
        return wrapper
    return decorator


def async_test(name: str):
    """Decorator for async test functions."""
    def decorator(fn):
        def wrapper():
            global PASSED, FAILED
            try:
                asyncio.run(fn())
                PASSED += 1
                _safe_print(f"  [PASS] {name}")
            except Exception as exc:
                FAILED += 1
                ERRORS.append(f"{name}: {exc}")
                _safe_print(f"  [FAIL] {name} -- {exc}")
        return wrapper
    return decorator


# ════════════════════════════════════════════════════════════════════════════
# PHASE 1 — Config & VRAM Management
# ════════════════════════════════════════════════════════════════════════════

@test("Phase 1: Settings singletons and paths")
def test_p1_settings():
    from config.settings import (
        ARTIFACTS_DIR, DATA_DIR, FRAUD_THRESHOLDS, GNN_CFG,
        GPU_VRAM_TOTAL_MB, LEDGER_CFG, LOG_DIR, OLLAMA_CFG,
        PROJECT_ROOT, VRAM, XGBOOST_CFG,
    )
    assert PROJECT_ROOT.exists(), "PROJECT_ROOT does not exist"
    assert GPU_VRAM_TOTAL_MB == 8192
    assert VRAM.total_mb == 8192
    assert XGBOOST_CFG.eval_metric == "aucpr"
    assert GNN_CFG.hidden_channels == 128
    assert GNN_CFG.num_layers == 3
    assert OLLAMA_CFG.temperature == 0.3
    assert FRAUD_THRESHOLDS.ctr_threshold_inr == 10_00_000
    assert FRAUD_THRESHOLDS.max_cycle_length == 10
    assert LEDGER_CFG.checkpoint_interval == 100


@test("Phase 1: VRAMBudget validation passes")
def test_p1_vram_budget():
    from config.settings import VRAMBudget
    budget = VRAMBudget()
    budget.validate()  # should not raise
    assert budget.analysis_reserve_mb <= budget.total_mb
    assert budget.assistant_reserve_mb <= budget.total_mb


@test("Phase 1: VRAM Manager mode transitions")
def test_p1_vram_manager():
    from config.vram_manager import (
        GPUMode, analysis_mode, assistant_mode,
        get_current_mode, log_vram_status,
    )
    assert get_current_mode() == GPUMode.IDLE

    with analysis_mode():
        assert get_current_mode() == GPUMode.ANALYSIS
    assert get_current_mode() == GPUMode.IDLE

    with assistant_mode():
        assert get_current_mode() == GPUMode.ASSISTANT
    assert get_current_mode() == GPUMode.IDLE

    status = log_vram_status()
    assert "mode" in status


@test("Phase 1: Config __init__ re-exports")
def test_p1_config_init():
    from config import (
        ARTIFACTS_DIR, DATA_DIR, FRAUD_THRESHOLDS, GNN_CFG,
        LEDGER_CFG, LOG_DIR, OLLAMA_CFG, PROJECT_ROOT,
        XGBOOST_CFG, LedgerConfig,
        analysis_mode, assistant_mode, log_vram_status,
    )
    assert PROJECT_ROOT is not None
    assert callable(analysis_mode)


# ════════════════════════════════════════════════════════════════════════════
# PHASE 2 — Data Ingestion Pipeline
# ════════════════════════════════════════════════════════════════════════════

@test("Phase 2: Schema instantiation + msgpack round-trip")
def test_p2_schemas():
    from src.ingestion.schemas import (
        AccountType, AuthAction, AuthEvent, Channel,
        EventBatch, FraudPattern, InterbankMessage, Transaction,
        encode_transaction, decode_transaction,
        encode_batch_json, decode_batch_json,
    )
    from src.ingestion.validators import (
        compute_transaction_checksum,
        compute_interbank_checksum,
        compute_auth_checksum,
    )

    # Transaction round-trip
    crc = compute_transaction_checksum("TXN-abc123def456", 1700000000, "00011234567890", "00029876543210", 500000, 4)
    txn = Transaction(
        txn_id="TXN-abc123def456", timestamp=1700000000,
        sender_id="00011234567890", receiver_id="00029876543210",
        amount_paisa=500000, channel=Channel.UPI,
        sender_branch="0001", receiver_branch="0002",
        sender_geo_lat=12.97, sender_geo_lon=77.59,
        receiver_geo_lat=13.08, receiver_geo_lon=80.27,
        device_fingerprint="a1b2c3d4e5f6g7h8",
        sender_account_type=AccountType.SAVINGS,
        receiver_account_type=AccountType.CURRENT,
        checksum=crc,
    )
    raw = encode_transaction(txn)
    assert isinstance(raw, bytes) and len(raw) > 0
    decoded = decode_transaction(raw)
    assert decoded.txn_id == txn.txn_id
    assert decoded.amount_paisa == 500000
    assert decoded.checksum == crc

    # EventBatch JSON round-trip
    batch = EventBatch(
        transactions=[txn], interbank_messages=[], auth_events=[],
        batch_id=1, batch_timestamp=1700000000, event_count=1,
    )
    json_bytes = encode_batch_json(batch)
    decoded_batch = decode_batch_json(json_bytes)
    assert decoded_batch.batch_id == 1
    assert len(decoded_batch.transactions) == 1


@test("Phase 2: Validators — valid + invalid events")
def test_p2_validators():
    from src.ingestion.schemas import AccountType, Channel, FraudPattern, Transaction
    from src.ingestion.validators import (
        ValidationResult, compute_transaction_checksum, validate_transaction,
    )

    # Valid transaction
    crc = compute_transaction_checksum("TXN-valid0000001", 1700000000, "00011234567890", "00029876543210", 100000, 4)
    txn = Transaction(
        txn_id="TXN-valid0000001", timestamp=1700000000,
        sender_id="00011234567890", receiver_id="00029876543210",
        amount_paisa=100000, channel=Channel.UPI,
        sender_branch="0001", receiver_branch="0002",
        sender_geo_lat=12.97, sender_geo_lon=77.59,
        receiver_geo_lat=13.08, receiver_geo_lon=80.27,
        device_fingerprint="a1b2c3d4e5f67890",
        sender_account_type=AccountType.SAVINGS,
        receiver_account_type=AccountType.CURRENT,
        checksum=crc,
    )
    result = validate_transaction(txn)
    assert result.valid, f"Valid txn rejected: {result.errors}"

    # Invalid: wrong checksum
    bad_txn = Transaction(
        txn_id="TXN-bad000000001", timestamp=1700000000,
        sender_id="00011234567890", receiver_id="00029876543210",
        amount_paisa=100000, channel=Channel.UPI,
        sender_branch="0001", receiver_branch="0002",
        sender_geo_lat=12.97, sender_geo_lon=77.59,
        receiver_geo_lat=13.08, receiver_geo_lon=80.27,
        device_fingerprint="a1b2c3d4e5f67890",
        sender_account_type=AccountType.SAVINGS,
        receiver_account_type=AccountType.CURRENT,
        checksum=12345,  # wrong checksum
    )
    result2 = validate_transaction(bad_txn)
    assert not result2.valid, "Bad checksum should be rejected"
    assert any("checksum" in e for e in result2.errors)


@test("Phase 2: Synthetic generator — world + event stream")
def test_p2_generators():
    from src.ingestion.generators.synthetic_transactions import (
        build_world, generate_event_stream,
        generate_layering_chain, generate_round_trip,
        generate_structuring_burst, generate_dormant_activation,
        generate_profile_mismatch,
    )
    from src.ingestion.schemas import FraudPattern, Transaction

    world = build_world(num_accounts=200, dormant_ratio=0.1)
    assert len(world.accounts) == 200
    assert len(world.dormant_accounts) >= 10

    # Each fraud pattern
    ts = int(time.time())
    layering = generate_layering_chain(world, ts, chain_length=4)
    assert len(layering) == 4  # chain_length=4 → 5 accounts sampled → 4 hops
    assert all(t.fraud_label == FraudPattern.LAYERING for t in layering)

    round_trip = generate_round_trip(world, ts, cycle_size=4)
    assert len(round_trip) == 4
    assert all(t.fraud_label == FraudPattern.ROUND_TRIPPING for t in round_trip)

    structuring = generate_structuring_burst(world, ts, num_transactions=5)
    assert len(structuring) == 5
    assert all(t.fraud_label == FraudPattern.STRUCTURING for t in structuring)

    dormant = generate_dormant_activation(world, ts)
    assert len(dormant) > 0
    assert all(t.fraud_label == FraudPattern.DORMANT_ACTIVATION for t in dormant)

    mismatch = generate_profile_mismatch(world, ts)
    assert len(mismatch) > 0
    assert all(t.fraud_label == FraudPattern.PROFILE_MISMATCH for t in mismatch)

    # Mixed stream
    events = list(generate_event_stream(world, num_events=500, fraud_ratio=0.10))
    assert len(events) > 400  # may exceed due to fraud chains


@async_test("Phase 2: Ingestion pipeline — ingest + validate + batch + consumer")
async def test_p2_pipeline():
    from src.ingestion.generators.synthetic_transactions import build_world, generate_event_stream
    from src.ingestion.stream_processor import IngestionPipeline

    world = build_world(num_accounts=100)
    events = list(generate_event_stream(world, num_events=200, fraud_ratio=0.05))

    received_batches = []

    async def mock_consumer(batch):
        received_batches.append(batch)

    pipeline = IngestionPipeline(batch_size=64, batch_timeout_sec=0.5, queue_capacity=500)
    pipeline.add_consumer(mock_consumer)

    async with pipeline:
        for event in events:
            await pipeline.ingest(event)

    assert pipeline.metrics.events_ingested == len(events)
    assert pipeline.metrics.events_validated + pipeline.metrics.events_rejected == pipeline.metrics.events_ingested
    assert pipeline.metrics.events_validated > 0
    assert len(received_batches) > 0
    total_dispatched = sum(b.event_count for b in received_batches)
    assert total_dispatched == pipeline.metrics.events_dispatched


# ════════════════════════════════════════════════════════════════════════════
# PHASE 3 — Feature Engineering
# ════════════════════════════════════════════════════════════════════════════

@test("Phase 3: VelocityTracker — record + extract features")
def test_p3_velocity():
    from src.ml.velocity import VelocityTracker, VelocityFeatures

    tracker = VelocityTracker()
    ts = 1700000000

    # Record 10 transactions from the same sender
    for i in range(10):
        features = tracker.record_and_extract(
            sender_id="ACC001", receiver_id=f"RECV{i:03d}",
            amount_paisa=100000 * (i + 1), timestamp=ts + i * 300,
        )

    assert isinstance(features, VelocityFeatures)
    assert features.txn_count_1h == 10  # all within 1 hour
    assert features.txn_count_7d == 10
    assert features.unique_receivers_1h == 10
    assert features.amount_sum_1h > 0
    assert tracker.account_count() == 1
    assert tracker.total_records() == 10


@test("Phase 3: BehavioralAnalyzer — profiles and deviation features")
def test_p3_behavioral():
    from src.ml.behavioral import BehavioralAnalyzer, BehavioralFeatures

    analyzer = BehavioralAnalyzer()
    ts = 1700000000

    # Build baseline with 10 transactions
    for i in range(10):
        features = analyzer.analyze_transaction(
            sender_id="ACC001", amount_paisa=50000,
            timestamp=ts + i * 3600,
            sender_lat=12.97, sender_lon=77.59,
            device_fingerprint="device_a",
        )

    assert isinstance(features, BehavioralFeatures)
    assert len(features) == 10

    # Now an anomalous transaction: different location, huge amount, new device
    anomaly = analyzer.analyze_transaction(
        sender_id="ACC001", amount_paisa=99_000_000,
        timestamp=ts + 20 * 3600,
        sender_lat=28.61, sender_lon=77.21,  # Delhi instead of Bangalore
        device_fingerprint="device_b",
    )
    assert anomaly.device_change_flag == 1
    assert anomaly.geo_distance_km > 1000  # Bangalore → Delhi
    assert anomaly.session_anomaly > 0  # composite score should be elevated


@test("Phase 3: TextAnomalyAnalyzer — homoglyphs, entropy, patterns")
def test_p3_text_anomaly():
    from src.ml.text_anomaly import TextAnomalyAnalyzer, TextAnomalyFeatures

    analyzer = TextAnomalyAnalyzer()

    # Normal name
    normal = analyzer.analyze("Rahul Sharma")
    assert isinstance(normal, TextAnomalyFeatures)
    assert normal.homoglyph_count == 0
    assert normal.suspicious_token_flag == 0

    # Suspicious text with phishing tokens
    phish = analyzer.analyze("URGENT verify your OTP now")
    assert phish.suspicious_token_flag == 1
    assert phish.text_anomaly_score > normal.text_anomaly_score

    # Homoglyph attack (Cyrillic а instead of Latin a)
    homoglyph = analyzer.analyze("R\u0430hul Sh\u0430rm\u0430")
    assert homoglyph.homoglyph_count >= 3

    # Empty string
    empty = analyzer.analyze("")
    assert empty.text_anomaly_score == 0.0


@test("Phase 3: FeatureEngine — full 30-dim extraction from EventBatch")
def test_p3_feature_engine():
    from src.ml.feature_engine import FeatureEngine, TOTAL_FEATURE_DIM, FEATURE_COLUMNS, ExtractionResult
    from src.ingestion.generators.synthetic_transactions import build_world, generate_event_stream
    from src.ingestion.schemas import EventBatch, Transaction, InterbankMessage, AuthEvent

    engine = FeatureEngine()
    world = build_world(num_accounts=100)
    events = list(generate_event_stream(world, num_events=300, fraud_ratio=0.10))

    # Separate by type
    txns = [e for e in events if isinstance(e, Transaction)]
    msgs = [e for e in events if isinstance(e, InterbankMessage)]
    auths = [e for e in events if isinstance(e, AuthEvent)]

    batch = EventBatch(
        transactions=txns[:50], interbank_messages=msgs[:10], auth_events=auths[:10],
        batch_id=1, batch_timestamp=int(time.time()), event_count=50 + 10 + 10,
    )

    result = engine.extract_batch(batch)
    assert isinstance(result, ExtractionResult)
    assert result.features.shape == (50, TOTAL_FEATURE_DIM)
    assert result.features.dtype.name == "float32"
    assert len(result.txn_ids) == 50
    assert len(result.sender_ids) == 50
    assert len(result.receiver_ids) == 50
    assert TOTAL_FEATURE_DIM == 30
    assert len(FEATURE_COLUMNS) == 30


# ════════════════════════════════════════════════════════════════════════════
# PHASE 4 — ML Models (XGBoost, Threshold, Alert Router)
# ════════════════════════════════════════════════════════════════════════════

@test("Phase 4: DynamicThreshold — EMA warmup + tier classification")
def test_p4_threshold():
    from src.ml.models.threshold import DynamicThreshold, ThresholdConfig, RiskTier

    config = ThresholdConfig(warmup_samples=20, floor=0.5, ceiling=0.95)
    threshold = DynamicThreshold(config)

    assert not threshold.is_warmed_up
    assert threshold.current_threshold == config.initial_threshold

    # Feed warmup scores (mostly low)
    import numpy as np
    rng = np.random.RandomState(42)
    scores = rng.uniform(0.0, 0.3, size=50).tolist()

    results = threshold.evaluate_batch(np.array(scores))
    assert len(results) == 50
    assert threshold.is_warmed_up
    assert threshold.samples_seen == 50

    # Very high score should trigger HIGH tier
    high_result = threshold.evaluate(0.99)
    assert high_result.tier == RiskTier.HIGH
    assert high_result.exceeds_threshold

    # Very low score should be LOW tier
    low_result = threshold.evaluate(0.01)
    assert low_result.tier == RiskTier.LOW
    assert not low_result.exceeds_threshold

    snap = threshold.snapshot()
    assert "threshold" in snap and "ema_mean" in snap


@test("Phase 4: AlertRouter — build payloads + routing logic")
def test_p4_alert_router():
    from src.ml.models.alert_router import AlertRouter, AlertPayload
    from src.ml.models.threshold import RiskTier, ThresholdResult
    import numpy as np

    # Build payloads from threshold results
    threshold_results = [
        ThresholdResult(0.95, RiskTier.HIGH, 0.85, True),
        ThresholdResult(0.70, RiskTier.MEDIUM, 0.85, False),
        ThresholdResult(0.20, RiskTier.LOW, 0.85, False),
    ]
    features = np.random.rand(3, 30).astype(np.float32)
    names = [f"f{i}" for i in range(30)]

    payloads = AlertRouter.build_payloads(
        threshold_results=threshold_results,
        txn_ids=["TXN-001", "TXN-002", "TXN-003"],
        sender_ids=["S1", "S2", "S3"],
        receiver_ids=["R1", "R2", "R3"],
        timestamps=np.array([100, 200, 300]),
        features=features,
        feature_names=names,
    )

    # LOW tier should be filtered out
    assert len(payloads) == 2
    assert payloads[0].tier == RiskTier.HIGH
    assert payloads[1].tier == RiskTier.MEDIUM

    # to_dict serializable
    d = payloads[0].to_dict()
    assert "txn_id" in d and "top_features" in d


@async_test("Phase 4: AlertRouter — async dispatch to mock consumers")
async def test_p4_router_dispatch():
    from src.ml.models.alert_router import AlertRouter, AlertPayload
    from src.ml.models.threshold import RiskTier
    import numpy as np

    router = AlertRouter()
    graph_calls = []
    llm_calls = []

    async def mock_graph(payload):
        graph_calls.append(payload.txn_id)

    async def mock_llm(payload):
        llm_calls.append(payload.txn_id)

    router.register_graph_consumer(mock_graph)
    router.register_llm_consumer(mock_llm)

    # HIGH alert → both consumers
    high_payload = AlertPayload(
        txn_id="TXN-H1", sender_id="S1", receiver_id="R1",
        timestamp=100, risk_score=0.95, tier=RiskTier.HIGH,
        threshold_at_eval=0.85, features=np.zeros(30, dtype=np.float32),
        feature_names=[f"f{i}" for i in range(30)],
    )
    await router.route(high_payload)

    # MEDIUM alert → graph only
    med_payload = AlertPayload(
        txn_id="TXN-M1", sender_id="S2", receiver_id="R2",
        timestamp=200, risk_score=0.70, tier=RiskTier.MEDIUM,
        threshold_at_eval=0.85, features=np.zeros(30, dtype=np.float32),
        feature_names=[f"f{i}" for i in range(30)],
    )
    await router.route(med_payload)

    assert "TXN-H1" in graph_calls and "TXN-H1" in llm_calls
    assert "TXN-M1" in graph_calls and "TXN-M1" not in llm_calls
    assert router.metrics.high_alerts == 1
    assert router.metrics.medium_alerts == 1


@test("Phase 4: FraudClassifier — instantiation (CPU mode)")
def test_p4_classifier_init():
    from src.ml.models.xgboost_classifier import FraudClassifier, PredictionResult, ValidationMetrics
    clf = FraudClassifier()
    assert clf.device == "cpu"  # PAYFLOW_CPU_ONLY is set
    assert not clf.is_fitted


# ════════════════════════════════════════════════════════════════════════════
# PHASE 5 — Graph Analytics
# ════════════════════════════════════════════════════════════════════════════

@test("Phase 5: MuleDetector — star pattern detection")
def test_p5_mule_detector():
    import networkx as nx
    from src.graph.algorithms import MuleDetector, MuleNetwork

    g = nx.MultiDiGraph()
    ts = 1700000000

    # Build star: 6 senders → 1 mule
    mule = "MULE_001"
    for i in range(6):
        sender = f"SENDER_{i:03d}"
        g.add_node(sender, first_seen=ts, last_seen=ts, txn_count=1)
        g.add_edge(sender, mule, key=f"TXN-{i}",
                   timestamp=ts + i * 60, amount_paisa=500000, channel=4, fraud_label=0)
    g.add_node(mule, first_seen=ts, last_seen=ts + 300, txn_count=6)

    detector = MuleDetector(min_distinct_senders=5, window_seconds=3600)
    findings = detector.detect(g, current_time=ts + 600)

    assert len(findings) >= 1
    assert findings[0].mule_node == mule
    assert findings[0].distinct_senders >= 6


@test("Phase 5: CycleDetector — round-tripping detection")
def test_p5_cycle_detector():
    import networkx as nx
    from src.graph.algorithms import CycleDetector, TransactionCycle

    g = nx.MultiDiGraph()
    ts = 1700000000
    nodes = ["A", "B", "C", "D"]

    # Build cycle: A→B→C→D→A
    for i in range(4):
        u, v = nodes[i], nodes[(i + 1) % 4]
        g.add_node(u, first_seen=ts, last_seen=ts, txn_count=1)
        g.add_edge(u, v, key=f"TXN-CYCLE-{i}",
                   timestamp=ts + i * 120, amount_paisa=1000000, channel=6, fraud_label=0)

    detector = CycleDetector(max_cycle_length=10, window_seconds=3600)
    cycles = detector.detect(g, current_time=ts + 600)

    assert len(cycles) >= 1
    assert cycles[0].cycle_length == 4


@async_test("Phase 5: TransactionGraph — ingest + investigation")
async def test_p5_graph_builder():
    from src.graph.builder import TransactionGraph, GraphConfig, InvestigationResult
    from src.ingestion.generators.synthetic_transactions import build_world, generate_event_stream
    from src.ingestion.schemas import EventBatch, Transaction
    from src.ml.models.alert_router import AlertPayload
    from src.ml.models.threshold import RiskTier
    import numpy as np

    world = build_world(num_accounts=200)
    events = list(generate_event_stream(world, num_events=500, fraud_ratio=0.10))
    txns = [e for e in events if isinstance(e, Transaction)]

    graph = TransactionGraph(config=GraphConfig(scan_interval_batches=9999))

    # Ingest as batches
    batch_size = 100
    for i in range(0, len(txns), batch_size):
        chunk = txns[i:i + batch_size]
        batch = EventBatch(
            transactions=chunk, interbank_messages=[], auth_events=[],
            batch_id=i // batch_size + 1,
            batch_timestamp=int(time.time()),
            event_count=len(chunk),
        )
        await graph.ingest(batch)

    assert graph.metrics.transactions_added == len(txns)
    assert graph.graph.number_of_nodes() > 0
    assert graph.graph.number_of_edges() > 0

    # Investigation
    sample_sender = txns[0].sender_id
    sample_receiver = txns[0].receiver_id
    payload = AlertPayload(
        txn_id=txns[0].txn_id, sender_id=sample_sender, receiver_id=sample_receiver,
        timestamp=txns[0].timestamp, risk_score=0.90, tier=RiskTier.HIGH,
        threshold_at_eval=0.85, features=np.zeros(30, dtype=np.float32),
        feature_names=[f"f{i}" for i in range(30)],
    )
    await graph.investigate(payload)
    assert graph.metrics.investigations_completed == 1


# ════════════════════════════════════════════════════════════════════════════
# PHASE 6 — GNN + LLM Modules
# ════════════════════════════════════════════════════════════════════════════

@test("Phase 6: GNN nx→PyG conversion + model build")
def test_p6_gnn():
    import networkx as nx
    from src.ml.models.gnn_scorer import nx_to_pyg_data, GNNScorer, _build_fraud_gat, NODE_FEATURE_DIM, EDGE_FEATURE_DIM
    from config.settings import GNN_CFG

    # Build a small test subgraph
    g = nx.MultiDiGraph()
    ts = 1700000000
    for i in range(5):
        g.add_node(f"N{i}", first_seen=ts - 3600, last_seen=ts, txn_count=i + 1)
    for i in range(4):
        g.add_edge(f"N{i}", f"N{i+1}", key=f"E{i}",
                   timestamp=ts - i * 60, amount_paisa=100000 * (i + 1),
                   channel=4, fraud_label=0)
    g.add_edge("N4", "N0", key="E4", timestamp=ts - 10,
               amount_paisa=500000, channel=6, fraud_label=1)

    data = nx_to_pyg_data(g, ["N0", "N4"], ts)
    assert data.x.shape[0] == 5
    assert data.x.shape[1] == NODE_FEATURE_DIM  # 7
    assert data.edge_index.shape[0] == 2
    assert data.edge_attr.shape[1] == EDGE_FEATURE_DIM  # 3
    assert float(data.y) == 1.0  # has fraud edge

    # Build model and check parameter count
    model = _build_fraud_gat(GNN_CFG)
    param_count = sum(p.numel() for p in model.parameters())
    assert param_count > 200_000  # ~278K expected


@test("Phase 6: LLM module imports and class structure")
def test_p6_llm_imports():
    from src.llm.orchestrator import PayFlowLLM, LLMResponse, VRAMInsufficientError
    from src.llm.health_check import check_vram_for_llm

    # Health check returns a result object (GPU may not be present; that's fine)
    result = check_vram_for_llm()
    assert hasattr(result, "passed")
    assert hasattr(result, "message")


# ════════════════════════════════════════════════════════════════════════════
# PHASE 7 — Audit Ledger (subset — full suite in test_blockchain.py)
# ════════════════════════════════════════════════════════════════════════════

@async_test("Phase 7: Ledger — genesis + anchor + verify chain")
async def test_p7_ledger():
    from src.blockchain.ledger import AuditLedger
    from src.blockchain.models import EventType, VerificationResult
    from config.settings import LedgerConfig

    with tempfile.TemporaryDirectory() as tmpdir:
        from pathlib import Path
        cfg = LedgerConfig(
            db_path=Path(tmpdir) / "test_ledger.db",
            key_dir=Path(tmpdir) / "keys",
            checkpoint_interval=5,
            enable_signing=True,
        )

        async with AuditLedger(config=cfg) as ledger:
            stats = await ledger.get_stats()
            assert stats.total_blocks >= 1  # genesis

            # Anchor 5 events
            for i in range(5):
                await ledger.anchor(EventType.ALERT, {"test": i})

            # Verify chain
            result = await ledger.verify_chain()
            assert result.valid, f"Chain verification failed: {result.error_message}"
            assert result.blocks_checked >= 6  # genesis + 5


# ════════════════════════════════════════════════════════════════════════════
# CROSS-PHASE INTEGRATION
# ════════════════════════════════════════════════════════════════════════════

@async_test("Integration: Pipeline → Features → Threshold → Router → Graph")
async def test_integration_pipeline_to_graph():
    """End-to-end integration test simulating the core data flow."""
    from src.ingestion.generators.synthetic_transactions import build_world, generate_event_stream
    from src.ingestion.stream_processor import IngestionPipeline
    from src.ingestion.schemas import EventBatch, Transaction
    from src.ml.feature_engine import FeatureEngine
    from src.ml.models.threshold import DynamicThreshold, ThresholdConfig
    from src.ml.models.alert_router import AlertRouter
    from src.graph.builder import TransactionGraph, GraphConfig
    import numpy as np

    # Initialize all components
    world = build_world(num_accounts=200)
    events = list(generate_event_stream(world, num_events=500, fraud_ratio=0.10))

    engine = FeatureEngine()
    threshold = DynamicThreshold(ThresholdConfig(warmup_samples=10))
    router = AlertRouter()
    graph = TransactionGraph(config=GraphConfig(scan_interval_batches=9999))

    graph_investigations = []

    async def mock_graph_investigate(payload):
        graph_investigations.append(payload.txn_id)

    router.register_graph_consumer(mock_graph_investigate)

    pipeline = IngestionPipeline(batch_size=100, batch_timeout_sec=0.3)
    pipeline.add_consumer(engine.ingest)
    pipeline.add_consumer(graph.ingest)

    async with pipeline:
        for event in events:
            await pipeline.ingest(event)

    # Features gathered
    train_data = engine.get_training_data()
    assert train_data is not None
    assert train_data.features.shape[1] == 30

    # Run threshold evaluation on extracted features
    if train_data.features.shape[0] > 0:
        # Simulate risk scores (since we skip XGBoost training in this test)
        fake_scores = np.random.rand(train_data.features.shape[0])
        results = threshold.evaluate_batch(fake_scores)
        from src.ml.feature_engine import FEATURE_COLUMNS
        payloads = AlertRouter.build_payloads(
            threshold_results=results,
            txn_ids=train_data.txn_ids,
            sender_ids=train_data.sender_ids,
            receiver_ids=train_data.receiver_ids,
            timestamps=train_data.timestamps,
            features=train_data.features,
            feature_names=FEATURE_COLUMNS,
        )

        # Route alerts
        for p in payloads[:5]:  # route first 5 to avoid slowness
            await router.route(p)

    # Verify integration worked
    assert engine.metrics.batches_processed > 0
    assert graph.metrics.transactions_added > 0
    assert pipeline.metrics.events_ingested == len(events)
    summary = engine.summary()
    assert summary["total_rows"] > 0


# ════════════════════════════════════════════════════════════════════════════
# MAIN RUNNER
# ════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("PayFlow — Comprehensive Phase 1-7 Verification")
    print("=" * 60)

    # Phase 1
    print("\n--- Phase 1: Config & VRAM Management ---")
    test_p1_settings()
    test_p1_vram_budget()
    test_p1_vram_manager()
    test_p1_config_init()

    # Phase 2
    print("\n--- Phase 2: Data Ingestion Pipeline ---")
    test_p2_schemas()
    test_p2_validators()
    test_p2_generators()
    test_p2_pipeline()

    # Phase 3
    print("\n--- Phase 3: Feature Engineering ---")
    test_p3_velocity()
    test_p3_behavioral()
    test_p3_text_anomaly()
    test_p3_feature_engine()

    # Phase 4
    print("\n--- Phase 4: ML Models ---")
    test_p4_threshold()
    test_p4_alert_router()
    test_p4_router_dispatch()
    test_p4_classifier_init()

    # Phase 5
    print("\n--- Phase 5: Graph Analytics ---")
    test_p5_mule_detector()
    test_p5_cycle_detector()
    test_p5_graph_builder()

    # Phase 6
    print("\n--- Phase 6: GNN + LLM Modules ---")
    test_p6_gnn()
    test_p6_llm_imports()

    # Phase 7
    print("\n--- Phase 7: Audit Ledger ---")
    test_p7_ledger()

    # Cross-phase integration
    print("\n--- Cross-Phase Integration ---")
    test_integration_pipeline_to_graph()

    # Summary
    print("\n" + "=" * 60)
    total = PASSED + FAILED
    print(f"Results: {PASSED} passed, {FAILED} failed out of {total}")
    if ERRORS:
        print("\nFailures:")
        for err in ERRORS:
            print(f"  ✗ {err}")
    else:
        print("All tests passed!")
    print("=" * 60)

    return 0 if FAILED == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

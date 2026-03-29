"""
PayFlow — Feature Extraction Engine (Orchestrator)
=====================================================
Central feature engineering module that consumes EventBatch objects from the
ingestion pipeline and produces GPU-ready feature tensors for downstream
XGBoost / GNN models.

Architecture:
    EventBatch ──→ FeatureEngine.ingest(batch) ──→ FeatureStore
                       │
                       ├─ VelocityTracker  (11 features)
                       ├─ BehavioralAnalyzer (10 features)
                       └─ TextAnomalyAnalyzer (9 features)
                                                    │
                                                    ▼
                                        numpy float32 array (30 cols)
                                              ↓
                                        PyArrow RecordBatch
                                        (zero-copy → Polars / XGBoost)

Feature Vector Layout (30 columns):
  [0:11]  Velocity features
  [11:21] Behavioral features
  [21:30] Text anomaly features

Memory Budget:
  - VelocityTracker   : ~400 MB @ 100K accounts (7-day window)
  - BehavioralAnalyzer: ~40 MB @ 100K accounts
  - TextAnomalyAnalyzer: ~1 MB (stateless except length stats)
  - Feature arrays     : ~120 bytes/row × batch_size
  Total: ~450 MB CPU RAM — fits within analysis mode VRAM budget (GPU
  arrays are created on-demand during XGBoost DMatrix construction).

Pipeline Integration:
  The FeatureEngine exposes an async `ingest(batch)` coroutine that
  conforms to the Consumer protocol in stream_processor.py:
      Consumer = Callable[[EventBatch], Coroutine[Any, Any, None]]
  Register it via: pipeline.add_consumer(feature_engine.ingest)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import NamedTuple

import numpy as np

from src.ingestion.schemas import (
    AuthEvent,
    EventBatch,
    FraudPattern,
    InterbankMessage,
    Transaction,
)
import networkx as nx

from src.ml.behavioral import BehavioralAnalyzer, BehavioralFeatures
from src.ml.text_anomaly import TextAnomalyAnalyzer, TextAnomalyFeatures
from src.ml.velocity import VelocityFeatures, VelocityTracker

logger = logging.getLogger(__name__)


# ── Feature Dimensions ──────────────────────────────────────────────────────

VELOCITY_DIM = 11
BEHAVIORAL_DIM = 10
TEXT_ANOMALY_DIM = 9
EXTENDED_DIM = 6   # build-prompt required extra features (f05, f19, f20, f31, f39, f40)
NETWORK_DIM = 6
BASE_FEATURE_DIM = VELOCITY_DIM + BEHAVIORAL_DIM + TEXT_ANOMALY_DIM + EXTENDED_DIM  # 36
TOTAL_FEATURE_DIM = BASE_FEATURE_DIM  # default; extended to 42 when graph is attached


# ── Column Names (for Polars/Arrow schema) ──────────────────────────────────

FEATURE_COLUMNS: list[str] = [
    # Velocity (11)
    "vel_txn_count_1h", "vel_txn_count_6h", "vel_txn_count_24h", "vel_txn_count_7d",
    "vel_unique_recv_1h", "vel_unique_recv_24h",
    "vel_amount_sum_1h", "vel_amount_sum_24h",
    "vel_avg_gap_sec", "vel_min_gap_sec", "vel_zscore",
    # Behavioral (10)
    "beh_hour_deviation", "beh_is_off_hours", "beh_geo_distance_km", "beh_geo_deviation",
    "beh_amount_zscore", "beh_amount_percentile", "beh_login_fail_rate_7d",
    "beh_unique_ips_24h", "beh_device_change", "beh_session_anomaly",
    # Text anomaly (9)
    "txt_homoglyph_count", "txt_mixed_script", "txt_digit_ratio", "txt_special_char_ratio",
    "txt_entropy", "txt_min_edit_dist", "txt_suspicious_token", "txt_name_length_z",
    "txt_anomaly_score",
    # Extended build-prompt features (6)
    "ext_amount_round_number", "ext_cfr_match", "ext_prior_fraud_reports",
    "ext_is_cross_border", "ext_pass_through_detected", "ext_dormant_activation",
]

NETWORK_COLUMNS: list[str] = [
    "net_in_degree", "net_out_degree", "net_degree_centrality",
    "net_betweenness", "net_pagerank", "net_community_id",
]

assert len(FEATURE_COLUMNS) == BASE_FEATURE_DIM
assert len(NETWORK_COLUMNS) == NETWORK_DIM


# ── Extraction Result ───────────────────────────────────────────────────────

class ExtractionResult(NamedTuple):
    """Result of processing a single EventBatch."""
    features: np.ndarray          # shape (N, 30), dtype float32
    labels: np.ndarray            # shape (N,), dtype int8 (FraudPattern enum)
    txn_ids: list[str]            # transaction IDs for traceability
    timestamps: np.ndarray        # shape (N,), dtype int64
    sender_ids: list[str]         # for graph edge construction
    receiver_ids: list[str]       # for graph edge construction


# ── Engine Metrics ──────────────────────────────────────────────────────────

@dataclass
class FeatureEngineMetrics:
    """Performance counters for the feature extraction pipeline."""
    transactions_processed: int = 0
    auth_events_processed: int = 0
    interbank_processed: int = 0
    batches_processed: int = 0
    total_extraction_ms: float = 0.0
    _start_time: float = field(default_factory=time.monotonic)

    @property
    def uptime_sec(self) -> float:
        return time.monotonic() - self._start_time

    @property
    def avg_extraction_ms(self) -> float:
        if self.batches_processed == 0:
            return 0.0
        return self.total_extraction_ms / self.batches_processed

    def snapshot(self) -> dict:
        return {
            "txns": self.transactions_processed,
            "auths": self.auth_events_processed,
            "interbank": self.interbank_processed,
            "batches": self.batches_processed,
            "avg_extract_ms": round(self.avg_extraction_ms, 2),
            "uptime_sec": round(self.uptime_sec, 1),
        }


# ── Feature Engine ──────────────────────────────────────────────────────────

class FeatureEngine:
    """
    Central feature extraction orchestrator.

    Consumes EventBatch from the ingestion pipeline, runs all three
    sub-engines (velocity, behavioral, text anomaly), and produces
    GPU-ready numpy arrays.

    Usage:
        engine = FeatureEngine()
        pipeline.add_consumer(engine.ingest)  # wire to pipeline

        # Or standalone:
        result = engine.extract_batch(batch)
        xgb_dmatrix = xgb.DMatrix(result.features, label=result.labels)
    """

    def __init__(self) -> None:
        self.velocity = VelocityTracker()
        self.behavioral = BehavioralAnalyzer()
        self.text_anomaly = TextAnomalyAnalyzer()
        self.metrics = FeatureEngineMetrics()

        # Accumulated results for bulk model training
        self._feature_store: list[ExtractionResult] = []

        # Per-txn feature cache for tool lookups (txn_id -> (features, names))
        self._feature_cache: dict[str, tuple[np.ndarray, list[str]]] = {}

        # Optional graph reference for network-derived features
        self._graph: nx.MultiDiGraph | None = None
        self._community_map: dict[str, int] = {}
        self._pagerank_cache: dict[str, float] = {}
        self._pagerank_stale: bool = True
        self._betweenness_cache: dict[str, float] = {}
        self._betweenness_stale: bool = True

        # Extended feature support
        self._cfr_registry = None  # CentralFraudRegistry, set via attach_cfr()
        self._last_activity: dict[str, int] = {}  # account_id -> last txn timestamp

    def attach_graph(
        self,
        graph: nx.MultiDiGraph,
        community_map: dict[str, int] | None = None,
    ) -> None:
        """Attach a transaction graph for network feature extraction."""
        self._graph = graph
        if community_map is not None:
            self._community_map = community_map
        self._pagerank_stale = True
        self._betweenness_stale = True

    def update_community_map(self, community_map: dict[str, int]) -> None:
        self._community_map = community_map

    def attach_cfr(self, registry) -> None:
        """Attach Central Fraud Registry for CFR match features."""
        self._cfr_registry = registry

    # ── Pipeline Consumer Interface ─────────────────────────────────────

    async def ingest(self, batch: EventBatch) -> None:
        """
        Async consumer callable — conforms to Consumer protocol.
        Register via: pipeline.add_consumer(engine.ingest)
        """
        result = self.extract_batch(batch)
        if result.features.shape[0] > 0:
            self._feature_store.append(result)

    # ── Core Extraction ─────────────────────────────────────────────────

    def extract_batch(self, batch: EventBatch) -> ExtractionResult:
        """
        Process a complete EventBatch and return vectorized features.

        Steps:
          1. Process auth events (enrich behavioral profiles, no features)
          2. Process interbank messages (velocity + behavioral features)
          3. Process transactions (all three feature engines)
          4. Stack into contiguous numpy float32 array
        """
        t0 = time.perf_counter()

        # Phase 1: Auth events enrich behavioral profiles
        for auth in batch.auth_events:
            self._process_auth(auth)

        # Phase 2: Interbank messages update velocity state
        for msg in batch.interbank_messages:
            self._process_interbank(msg)

        # Phase 3: Transactions → full feature vectors
        n_txn = len(batch.transactions)
        if n_txn == 0:
            self.metrics.batches_processed += 1
            elapsed_ms = (time.perf_counter() - t0) * 1000
            self.metrics.total_extraction_ms += elapsed_ms
            return ExtractionResult(
                features=np.empty((0, self.active_feature_dim), dtype=np.float32),
                labels=np.empty(0, dtype=np.int8),
                txn_ids=[],
                timestamps=np.empty(0, dtype=np.int64),
                sender_ids=[],
                receiver_ids=[],
            )

        # Pre-allocate arrays
        dim = self.active_feature_dim
        features = np.empty((n_txn, dim), dtype=np.float32)
        labels = np.empty(n_txn, dtype=np.int8)
        timestamps = np.empty(n_txn, dtype=np.int64)
        txn_ids: list[str] = []
        sender_ids: list[str] = []
        receiver_ids: list[str] = []

        for i, txn in enumerate(batch.transactions):
            row = self._extract_transaction_features(txn)
            features[i] = row
            labels[i] = int(txn.fraud_label)
            timestamps[i] = txn.timestamp
            txn_ids.append(txn.txn_id)
            sender_ids.append(txn.sender_id)
            receiver_ids.append(txn.receiver_id)
            # Cache per-txn features for LLM tool lookups
            self._feature_cache[txn.txn_id] = (row.copy(), self.active_feature_columns)

        self.metrics.transactions_processed += n_txn
        self.metrics.batches_processed += 1
        elapsed_ms = (time.perf_counter() - t0) * 1000
        self.metrics.total_extraction_ms += elapsed_ms

        if self.metrics.batches_processed % 50 == 0:
            logger.info("FeatureEngine metrics: %s", self.metrics.snapshot())

        return ExtractionResult(
            features=features,
            labels=labels,
            txn_ids=txn_ids,
            timestamps=timestamps,
            sender_ids=sender_ids,
            receiver_ids=receiver_ids,
        )

    def _extract_transaction_features(self, txn: Transaction) -> np.ndarray:
        """Extract all 36 base features for a single transaction."""
        # Velocity (11 features)
        vel: VelocityFeatures = self.velocity.record_and_extract(
            sender_id=txn.sender_id,
            receiver_id=txn.receiver_id,
            amount_paisa=txn.amount_paisa,
            timestamp=txn.timestamp,
        )

        # Behavioral (10 features)
        beh: BehavioralFeatures = self.behavioral.analyze_transaction(
            sender_id=txn.sender_id,
            amount_paisa=txn.amount_paisa,
            timestamp=txn.timestamp,
            sender_lat=txn.sender_geo_lat,
            sender_lon=txn.sender_geo_lon,
            device_fingerprint=txn.device_fingerprint,
        )

        # Text anomaly on receiver_id (in production, this would be
        # the beneficiary name from MT103 field :59 or NEFT remittance).
        # For the prototype, we analyze the receiver's account ID and
        # branch code as proxy fields — the pipeline will be extended
        # when real SFMS message parsing is added.
        text_field = txn.receiver_id  # proxy for beneficiary name
        txt: TextAnomalyFeatures = self.text_anomaly.analyze(text_field)

        # Extended build-prompt features (6)
        ext = self._compute_extended_features(txn)

        # Stack into flat float32 row (36 base elements)
        base = list(vel) + list(beh) + list(txt) + ext

        # Append network features if graph is attached
        if self._graph is not None:
            net = self._compute_network_features(txn.sender_id)
            base.extend(net)

        row = np.array(base, dtype=np.float32)
        return row

    def _compute_extended_features(self, txn: Transaction) -> list[float]:
        """
        Compute 6 build-prompt extended features:
          0: amount_round_number — is amount divisible by ₹1,000?
          1: cfr_match — Central Fraud Registry match (0/1)
          2: prior_fraud_reports — count of prior fraud reports
          3: is_cross_border — cross-border transaction flag
          4: pass_through_detected — rapid pass-through pattern (0/1)
          5: dormant_activation — dormant account reactivation (0/1)
        """
        # f05: Round number (divisible by ₹1,000 = 100,000 paisa)
        amount_round = 1.0 if txn.amount_paisa > 0 and txn.amount_paisa % 100_000 == 0 else 0.0

        # f19: CFR match — check if sender is in fraud registry
        cfr_match = 0.0
        if self._cfr_registry is not None:
            try:
                entry = self._cfr_registry.query(txn.sender_id)
                if entry is not None:
                    cfr_match = 1.0
            except Exception:
                pass

        # f20: Prior fraud reports count
        prior_fraud = 0.0
        if self._cfr_registry is not None:
            try:
                entry = self._cfr_registry.query(txn.sender_id)
                if entry is not None:
                    prior_fraud = float(getattr(entry, "fraud_count", 1))
            except Exception:
                pass

        # f31: Cross-border (channel is SWIFT or receiver has international pattern)
        is_cross_border = 1.0 if txn.channel in (7, 8) else 0.0  # SWIFT/INTERNATIONAL

        # f39: Pass-through (checks if account received + forwarded >80% within 30min)
        pass_through = self._detect_pass_through(txn.sender_id, txn.timestamp)

        # f40: Dormant activation (inactive >180 days, now high-value)
        dormant = self._detect_dormant_activation(txn.sender_id, txn.amount_paisa, txn.timestamp)

        return [amount_round, cfr_match, prior_fraud, is_cross_border, pass_through, dormant]

    def _detect_pass_through(self, account_id: str, current_ts: int) -> float:
        """Check if an account received and forwarded >80% of funds within 30min."""
        if self._graph is None:
            return 0.0
        g = self._graph
        if not g.has_node(account_id):
            return 0.0

        window = 30 * 60  # 30 minutes in seconds

        # Check incoming edges in window
        in_total = 0.0
        for u, _, data in g.in_edges(account_id, data=True):
            ts = data.get("timestamp", 0)
            if current_ts - ts < window:
                in_total += data.get("amount_paisa", 0)

        if in_total == 0:
            return 0.0

        # Check outgoing edges after last incoming
        out_total = 0.0
        for _, v, data in g.out_edges(account_id, data=True):
            ts = data.get("timestamp", 0)
            if current_ts - ts < window:
                out_total += data.get("amount_paisa", 0)

        if out_total / in_total > 0.8:
            return 1.0
        return 0.0

    def _detect_dormant_activation(self, account_id: str, amount_paisa: int, current_ts: int) -> float:
        """Detect dormant account reactivation (>180 days inactive, high-value txn)."""
        last_ts = self._last_activity.get(account_id)
        self._last_activity[account_id] = current_ts

        if last_ts is None:
            return 0.0

        idle_days = (current_ts - last_ts) / 86400.0
        if idle_days > 180 and amount_paisa > 2_000_000_00:  # >₹20L after 180 days
            return 1.0
        return 0.0

    def _process_auth(self, auth: AuthEvent) -> None:
        """Feed auth event into behavioral profiler."""
        self.behavioral.record_auth_event(
            account_id=auth.account_id,
            timestamp=auth.timestamp,
            success=auth.success,
            ip_address=auth.ip_address,
            device_fingerprint=auth.device_fingerprint,
        )
        self.metrics.auth_events_processed += 1

    def _process_interbank(self, msg: InterbankMessage) -> None:
        """Feed interbank message into velocity tracker."""
        self.velocity.record_and_extract(
            sender_id=msg.sender_account,
            receiver_id=msg.receiver_account,
            amount_paisa=msg.amount_paisa,
            timestamp=msg.timestamp,
        )
        self.metrics.interbank_processed += 1

    # ── Network Features ────────────────────────────────────────────────

    def _compute_network_features(self, node_id: str) -> list[float]:
        """
        Compute 6 network-derived features for a node:
          0: in-degree   (normalized)
          1: out-degree   (normalized)
          2: degree centrality
          3: betweenness centrality
          4: PageRank score
          5: community_id (as float)

        Betweenness centrality measures how often a node appears on shortest
        paths between other nodes — high values indicate transaction
        intermediaries (potential mule accounts).  Uses approximate
        betweenness (k=min(200, n)) for graphs > 500 nodes to keep
        computation tractable.
        """
        g = self._graph
        if g is None or not g.has_node(node_id):
            return [0.0] * NETWORK_DIM

        n_nodes = max(g.number_of_nodes(), 1)

        in_deg = g.in_degree(node_id) / max(n_nodes - 1, 1)
        out_deg = g.out_degree(node_id) / max(n_nodes - 1, 1)
        deg_centrality = (g.in_degree(node_id) + g.out_degree(node_id)) / max(
            2 * (n_nodes - 1), 1
        )

        # Betweenness centrality — CB(v) = Σ σ(s,t|v) / σ(s,t)
        # Recompute only when stale; use approximation for large graphs
        if self._betweenness_stale:
            try:
                if n_nodes > 500:
                    # Approximate: sample k pivot nodes for O(k(V+E))
                    k = min(200, n_nodes)
                    self._betweenness_cache = nx.betweenness_centrality(
                        g, k=k, normalized=True, weight=None,
                    )
                else:
                    self._betweenness_cache = nx.betweenness_centrality(
                        g, normalized=True, weight=None,
                    )
            except nx.NetworkXError:
                self._betweenness_cache = {}
            self._betweenness_stale = False
        bc = self._betweenness_cache.get(node_id, 0.0)

        # PageRank — recompute only when stale (expensive on large graphs)
        if self._pagerank_stale:
            try:
                self._pagerank_cache = nx.pagerank(
                    g, alpha=0.85, max_iter=50, tol=1e-4
                )
            except (nx.NetworkXError, nx.PowerIterationFailedConvergence):
                self._pagerank_cache = {}
            self._pagerank_stale = False
        pr = self._pagerank_cache.get(node_id, 0.0)

        community = float(self._community_map.get(node_id, -1))

        return [in_deg, out_deg, deg_centrality, bc, pr, community]

    @property
    def active_feature_dim(self) -> int:
        """Current feature dimension (36 base or 42 with graph attached)."""
        return BASE_FEATURE_DIM + (NETWORK_DIM if self._graph is not None else 0)

    @property
    def active_feature_columns(self) -> list[str]:
        """Current column names including network features if attached."""
        if self._graph is not None:
            return FEATURE_COLUMNS + NETWORK_COLUMNS
        return FEATURE_COLUMNS

    # ── Bulk Feature Access ─────────────────────────────────────────────

    def get_training_data(self) -> ExtractionResult | None:
        """
        Concatenate all accumulated feature batches into a single
        contiguous array suitable for XGBoost DMatrix construction.

        Returns None if no data has been accumulated.
        """
        if not self._feature_store:
            return None

        all_features = np.concatenate(
            [r.features for r in self._feature_store], axis=0
        )
        all_labels = np.concatenate(
            [r.labels for r in self._feature_store], axis=0
        )
        all_timestamps = np.concatenate(
            [r.timestamps for r in self._feature_store], axis=0
        )
        all_txn_ids = [tid for r in self._feature_store for tid in r.txn_ids]
        all_senders = [sid for r in self._feature_store for sid in r.sender_ids]
        all_receivers = [rid for r in self._feature_store for rid in r.receiver_ids]

        return ExtractionResult(
            features=all_features,
            labels=all_labels,
            txn_ids=all_txn_ids,
            timestamps=all_timestamps,
            sender_ids=all_senders,
            receiver_ids=all_receivers,
        )

    def to_arrow(self) -> "pa.RecordBatch":
        """
        Export accumulated features as a PyArrow RecordBatch.

        Zero-copy pathway to Polars DataFrame:
            df = pl.from_arrow(engine.to_arrow())

        And to XGBoost DMatrix:
            dm = xgb.DMatrix(engine.to_arrow())
        """
        import pyarrow as pa

        data = self.get_training_data()
        if data is None:
            return pa.record_batch(
                {col: pa.array([], type=pa.float32()) for col in FEATURE_COLUMNS},
            )

        arrays = {
            col: pa.array(data.features[:, i], type=pa.float32())
            for i, col in enumerate(FEATURE_COLUMNS)
        }
        arrays["fraud_label"] = pa.array(data.labels, type=pa.int8())
        arrays["txn_id"] = pa.array(data.txn_ids, type=pa.string())
        arrays["timestamp"] = pa.array(data.timestamps, type=pa.int64())
        arrays["sender_id"] = pa.array(data.sender_ids, type=pa.string())
        arrays["receiver_id"] = pa.array(data.receiver_ids, type=pa.string())

        return pa.record_batch(arrays)

    def clear_store(self) -> None:
        """Release accumulated feature data to free memory."""
        self._feature_store.clear()

    def summary(self) -> dict:
        """Engine state summary for diagnostics."""
        data = self.get_training_data()
        n_rows = data.features.shape[0] if data else 0
        fraud_counts = {}
        if data is not None:
            for label_val in range(6):  # FraudPattern has 6 values
                count = int((data.labels == label_val).sum())
                if count > 0:
                    fraud_counts[FraudPattern(label_val).name] = count

        return {
            "total_rows": n_rows,
            "feature_dim": self.active_feature_dim,
            "feature_columns": len(self.active_feature_columns),
            "graph_attached": self._graph is not None,
            "velocity_accounts": self.velocity.account_count(),
            "behavioral_accounts": self.behavioral.account_count(),
            "text_analyzed": self.text_anomaly.analyzed_count(),
            "fraud_distribution": fraud_counts,
            "metrics": self.metrics.snapshot(),
        }

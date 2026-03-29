"""
PayFlow — Unified Execution Engine
====================================
Orchestrates all modules (Ingestion, ML, Graph, Blockchain, LLM Agents)
through an asynchronous pipeline with continuous hardware profiling,
automatic VRAM load-shedding, and GPU/CPU/TPS telemetry.

Architecture:
    ┌───────────────────────────────────────────────────┐
    │               HardwareProfiler (1 Hz)             │
    │   GPU VRAM · CPU % · LLM TPS · Load-Shed State   │
    └────────────────────┬──────────────────────────────┘
                         │ events
    ┌────────────────────▼──────────────────────────────┐
    │              PayFlowOrchestrator                  │
    │                                                   │
    │  IngestionPipeline ──► FeatureEngine (consumer)   │
    │                    ──► TransactionGraph (consumer) │
    │                    ──► AuditLedger (consumer)      │
    │                                                   │
    │  FeatureEngine ─► FraudClassifier ─► Threshold    │
    │       │                                  │        │
    │       ▼                                  ▼        │
    │  AlertRouter ─► Graph.investigate()               │
    │              ─► CircuitBreaker.on_alert()          │
    │              ─► InvestigatorAgent.on_alert()       │
    │              ─► AuditLedger.anchor_alert()         │
    └───────────────────────────────────────────────────┘

Load-Shedding Policy:
    When VRAM usage exceeds 7.5 GB, the LoadShedder pauses non-critical
    GNN embedding generation to preserve GPU headroom for the LLM's
    reasoning loop.  When VRAM drops below 6.5 GB, GNN scoring resumes.

Usage:
    python main.py
    python main.py --events 5000 --fraud-ratio 0.08 --accounts 500
    python main.py --cpu-only
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import platform
import sys
import time
from dataclasses import dataclass, field
from typing import Optional

# ── Logging ──────────────────────────────────────────────────────────────────

LOG_FORMAT = (
    "%(asctime)s │ %(levelname)-7s │ %(name)-28s │ %(message)s"
)

logging.basicConfig(
    level=logging.INFO,
    format=LOG_FORMAT,
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("payflow.main")


# ════════════════════════════════════════════════════════════════════════════
# HARDWARE PROFILER — continuous GPU / CPU / TPS telemetry
# ════════════════════════════════════════════════════════════════════════════

@dataclass
class HardwareSnapshot:
    """Point-in-time hardware telemetry reading."""
    timestamp: float
    gpu_vram_used_mb: float = 0.0
    gpu_vram_total_mb: float = 0.0
    gpu_vram_free_mb: float = 0.0
    gpu_utilization_pct: int = -1
    gpu_temperature_c: int = -1
    cpu_utilization_pct: float = 0.0
    llm_tokens_generated: int = 0
    llm_tps: float = 0.0
    load_shed_active: bool = False


class HardwareProfiler:
    """
    Continuous hardware monitor running at configurable frequency.

    Polls GPU VRAM via NVML (zero-dependency), CPU utilisation via /proc
    or psutil fallback, and tracks LLM token throughput for TPS.
    """

    def __init__(
        self,
        poll_interval_sec: float = 1.0,
        vram_shed_threshold_mb: float = 7500.0,
        vram_resume_threshold_mb: float = 6500.0,
    ) -> None:
        self._interval = poll_interval_sec
        self._vram_shed_threshold = vram_shed_threshold_mb
        self._vram_resume_threshold = vram_resume_threshold_mb

        # LLM TPS tracking
        self._tokens_total: int = 0
        self._tokens_window_start: float = time.monotonic()
        self._tokens_in_window: int = 0

        # State
        self._load_shed_active: bool = False
        self._latest: HardwareSnapshot = HardwareSnapshot(timestamp=time.time())
        self._task: Optional[asyncio.Task] = None
        self._running: bool = False

        # Callbacks
        self._on_shed: Optional[asyncio.Event] = None
        self._on_resume: Optional[asyncio.Event] = None
        self._priority_queue = None  # GPUPriorityQueue, set via set_priority_queue()
        self._event_loop = None  # for run_coroutine_threadsafe bridge

    # ── Public API ────────────────────────────────────────────────────────

    def set_priority_queue(self, queue, loop=None) -> None:
        """Wire the GPU priority queue for VRAM pressure updates."""
        self._priority_queue = queue
        self._event_loop = loop

    @property
    def latest(self) -> HardwareSnapshot:
        return self._latest

    @property
    def load_shed_active(self) -> bool:
        return self._load_shed_active

    def record_tokens(self, count: int) -> None:
        """Record LLM tokens generated (call from orchestrator after inference)."""
        self._tokens_total += count
        self._tokens_in_window += count

    def set_shed_events(
        self,
        on_shed: asyncio.Event,
        on_resume: asyncio.Event,
    ) -> None:
        """Wire load-shedding signals to the orchestrator."""
        self._on_shed = on_shed
        self._on_resume = on_resume

    async def start(self) -> None:
        self._running = True
        self._task = asyncio.create_task(self._poll_loop(), name="hw-profiler")
        logger.info(
            "HardwareProfiler started (%.1fs interval, shed at %.0f MB, "
            "resume at %.0f MB)",
            self._interval,
            self._vram_shed_threshold,
            self._vram_resume_threshold,
        )

    async def stop(self) -> None:
        self._running = False
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("HardwareProfiler stopped.")

    def snapshot(self) -> dict:
        """JSON-serialisable snapshot for dashboards."""
        s = self._latest
        return {
            "gpu_vram_used_mb": round(s.gpu_vram_used_mb, 1),
            "gpu_vram_total_mb": round(s.gpu_vram_total_mb, 1),
            "gpu_vram_free_mb": round(s.gpu_vram_free_mb, 1),
            "gpu_utilization_pct": s.gpu_utilization_pct,
            "gpu_temperature_c": s.gpu_temperature_c,
            "cpu_utilization_pct": round(s.cpu_utilization_pct, 1),
            "llm_tps": round(s.llm_tps, 1),
            "llm_tokens_total": self._tokens_total,
            "load_shed_active": self._load_shed_active,
        }

    # ── Internal polling ──────────────────────────────────────────────────

    async def _poll_loop(self) -> None:
        while self._running:
            try:
                snap = await asyncio.to_thread(self._collect)
                self._latest = snap
                self._evaluate_load_shedding(snap)
            except Exception as exc:
                logger.debug("Profiler poll error: %s", exc)
            await asyncio.sleep(self._interval)

    def _collect(self) -> HardwareSnapshot:
        """Synchronous collection of all telemetry (runs in thread pool)."""
        now = time.time()
        vram_used = 0.0
        vram_total = 0.0
        vram_free = 0.0
        gpu_util = -1
        gpu_temp = -1

        # GPU via NVML
        try:
            from src.llm.health_check import query_gpu
            gpu = query_gpu()
            if gpu is not None:
                vram_used = gpu.used_mb
                vram_total = gpu.total_mb
                vram_free = gpu.free_mb
                gpu_util = gpu.gpu_utilization_pct
                gpu_temp = gpu.temperature_c
        except Exception:
            pass

        # CPU utilisation
        cpu_pct = self._read_cpu_utilisation()

        # TPS calculation (sliding 5-second window)
        elapsed = time.monotonic() - self._tokens_window_start
        tps = self._tokens_in_window / elapsed if elapsed > 0.5 else 0.0
        if elapsed >= 5.0:
            self._tokens_window_start = time.monotonic()
            self._tokens_in_window = 0

        return HardwareSnapshot(
            timestamp=now,
            gpu_vram_used_mb=vram_used,
            gpu_vram_total_mb=vram_total,
            gpu_vram_free_mb=vram_free,
            gpu_utilization_pct=gpu_util,
            gpu_temperature_c=gpu_temp,
            cpu_utilization_pct=cpu_pct,
            llm_tokens_generated=self._tokens_total,
            llm_tps=tps,
            load_shed_active=self._load_shed_active,
        )

    def _read_cpu_utilisation(self) -> float:
        """Best-effort CPU % without hard dependency on psutil."""
        try:
            import psutil
            return psutil.cpu_percent(interval=0)
        except ImportError:
            pass
        # Linux fallback: /proc/loadavg
        if platform.system() == "Linux":
            try:
                with open("/proc/loadavg") as f:
                    load_1m = float(f.read().split()[0])
                ncpu = os.cpu_count() or 1
                return min(load_1m / ncpu * 100, 100.0)
            except Exception:
                pass
        return 0.0

    def _evaluate_load_shedding(self, snap: HardwareSnapshot) -> None:
        """Hysteresis-based VRAM load-shedding trigger."""
        if snap.gpu_vram_total_mb == 0:
            return  # no GPU data

        if not self._load_shed_active:
            if snap.gpu_vram_used_mb >= self._vram_shed_threshold:
                self._load_shed_active = True
                logger.warning(
                    "LOAD-SHED ACTIVATED: VRAM %.0f/%.0f MB (>%.0f MB). "
                    "Pausing GNN embeddings to protect LLM headroom.",
                    snap.gpu_vram_used_mb, snap.gpu_vram_total_mb,
                    self._vram_shed_threshold,
                )
                if self._on_shed:
                    self._on_shed.set()
                if self._on_resume:
                    self._on_resume.clear()
        else:
            if snap.gpu_vram_used_mb <= self._vram_resume_threshold:
                self._load_shed_active = False
                logger.info(
                    "LOAD-SHED CLEARED: VRAM %.0f/%.0f MB (<%.0f MB). "
                    "Resuming GNN embeddings.",
                    snap.gpu_vram_used_mb, snap.gpu_vram_total_mb,
                    self._vram_resume_threshold,
                )
                if self._on_resume:
                    self._on_resume.set()
                if self._on_shed:
                    self._on_shed.clear()

        # Phase 15: Push pressure update to GPU priority queue (thread->async bridge)
        if self._priority_queue is not None and self._event_loop is not None:
            try:
                asyncio.run_coroutine_threadsafe(
                    self._priority_queue.update_pressure(snap.gpu_vram_used_mb),
                    self._event_loop,
                )
            except Exception:
                pass  # best-effort; queue may not be ready yet


# ════════════════════════════════════════════════════════════════════════════
# LOAD SHEDDER — VRAM-aware GNN gating
# ════════════════════════════════════════════════════════════════════════════

class LoadShedder:
    """
    Wraps a GNNScorer to gate its execution based on VRAM pressure.

    When the profiler signals that VRAM is overloaded (>7.5 GB), the
    load-shedder offloads GNN scoring to CPU to preserve real risk scores.
    When pressure subsides (<6.5 GB), GPU scoring resumes automatically.
    """

    def __init__(self, gnn_scorer, profiler: HardwareProfiler) -> None:
        self._gnn = gnn_scorer
        self._profiler = profiler
        self._shed_count: int = 0
        self._pass_count: int = 0
        self._cpu_fallback_count: int = 0

    @property
    def is_shedding(self) -> bool:
        return self._profiler.load_shed_active

    def score_subgraph(
        self,
        subgraph,
        center_nodes: list[str],
        reference_timestamp: int,
    ):
        """
        Proxied GNN scoring with VRAM-aware routing.

        When shedding is active, attempts CPU fallback via score_subgraph_cpu()
        to preserve real risk scores.  Only returns sentinel (-1.0) if CPU
        fallback also fails (e.g. model not loaded).
        """
        if self._profiler.load_shed_active:
            self._shed_count += 1
            return self._score_on_cpu(subgraph, center_nodes, reference_timestamp)
        self._pass_count += 1
        return self._gnn.score_subgraph(
            subgraph, center_nodes, reference_timestamp,
        )

    def _score_on_cpu(self, subgraph, center_nodes, reference_timestamp):
        """Fall back to CPU-based GNN scoring to preserve real risk scores."""
        try:
            result = self._gnn.score_subgraph_cpu(
                subgraph, center_nodes, reference_timestamp,
            )
            self._cpu_fallback_count += 1
            if self._cpu_fallback_count % 50 == 1:
                logger.info(
                    "GNN CPU fallback: %d subgraphs scored on CPU so far",
                    self._cpu_fallback_count,
                )
            return result
        except Exception as exc:
            logger.warning("GNN CPU fallback failed: %s — returning sentinel", exc)
            from src.ml.models.gnn_scorer import GNNScoringResult
            return GNNScoringResult(
                risk_score=-1.0,
                node_count=0,
                edge_count=0,
                inference_ms=0.0,
            )

    # Forward attribute access to the underlying GNNScorer
    def __getattr__(self, name):
        return getattr(self._gnn, name)

    def snapshot(self) -> dict:
        return {
            "shed_count": self._shed_count,
            "pass_count": self._pass_count,
            "cpu_fallback_count": self._cpu_fallback_count,
            "currently_shedding": self.is_shedding,
        }


# ════════════════════════════════════════════════════════════════════════════
# PAYFLOW ORCHESTRATOR
# ════════════════════════════════════════════════════════════════════════════

@dataclass
class OrchestratorMetrics:
    """Aggregate pipeline metrics."""
    events_ingested: int = 0
    batches_processed: int = 0
    features_extracted: int = 0
    ml_inferences: int = 0
    alerts_routed: int = 0
    investigations_triggered: int = 0
    verdicts_issued: int = 0
    blocks_anchored: int = 0
    load_sheds_triggered: int = 0
    pipeline_start_time: float = 0.0
    pipeline_end_time: float = 0.0

    @property
    def elapsed_sec(self) -> float:
        end = self.pipeline_end_time or time.time()
        return end - self.pipeline_start_time if self.pipeline_start_time else 0.0

    @property
    def events_per_sec(self) -> float:
        return self.events_ingested / self.elapsed_sec if self.elapsed_sec > 0 else 0.0


class PayFlowOrchestrator:
    """
    Unified execution engine that instantiates and interconnects every
    PayFlow module into a resilient asynchronous pipeline.

    Lifecycle:
        orchestrator = PayFlowOrchestrator(...)
        await orchestrator.initialize()
        await orchestrator.run()      # ingestion -> ML -> routing -> agents
        await orchestrator.shutdown()
    """

    def __init__(
        self,
        num_accounts: int = 2000,
        num_events: int = 10_000,
        fraud_ratio: float = 0.05,
        batch_size: int = 512,
        cpu_only: bool = False,
        skip_llm: bool = False,
        inference_batch_size: int = 256,
        profiler_interval: float = 1.0,
        vram_shed_threshold_mb: float = 7500.0,
        vram_resume_threshold_mb: float = 6500.0,
        enable_dashboard: bool = True,
        dashboard_port: int = 8000,
    ) -> None:
        self._num_accounts = num_accounts
        self._num_events = num_events
        self._fraud_ratio = fraud_ratio
        self._batch_size = batch_size
        self._cpu_only = cpu_only
        self._skip_llm = skip_llm
        self._inference_batch_size = inference_batch_size
        self._enable_dashboard = enable_dashboard
        self._dashboard_port = dashboard_port

        self.metrics = OrchestratorMetrics()
        self.profiler = HardwareProfiler(
            poll_interval_sec=profiler_interval,
            vram_shed_threshold_mb=vram_shed_threshold_mb,
            vram_resume_threshold_mb=vram_resume_threshold_mb,
        )

        # Components — initialised in initialize()
        self._ledger = None
        self._breaker = None
        self._gnn_scorer = None
        self._gnn_proxy = None
        self._graph = None
        self._engine = None
        self._classifier = None
        self._threshold = None
        self._router = None
        self._pipeline = None
        self._llm = None
        self._nlu_agent = None
        self._tool_executor = None
        self._agent = None
        self._world = None
        self._dashboard_task = None
        self._telemetry_task = None
        self._gpu_queue = None  # GPUPriorityQueue, created in initialize()
        self._threat_engine = None  # ThreatSimulationEngine, created in initialize()
        self._live_inference_task = None  # Background task for continuous inference
        self._inference_cursor = 0  # Tracks scored feature_store entries
        self._rule_engine = None  # TransactionRuleEngine, created in initialize()
        self._pre_approval_gate = None  # PreApprovalGate, created in initialize()
        self._isolation_detector = None  # IsolationDetector, created in initialize()
        self._autoencoder_detector = None  # AutoencoderDetector, created in initialize()
        self._fraud_registry = None  # CentralFraudRegistry, created in initialize()
        self._cfr_scorer = None  # CFRRiskScorer, created in initialize()
        self._investigation_mgr = None  # InvestigationManager, created in initialize()
        self._rf_classifier = None  # RandomForestFraudClassifier, created in initialize()
        self._lr_classifier = None  # LogisticFraudClassifier, created in initialize()
        self._placement_detector = None  # PlacementDetector, created in initialize()
        self._integration_detector = None  # IntegrationDetector, created in initialize()
        self._fiu_intelligence = None  # FIUIntelligenceUnit, created in initialize()
        self._mule_chain_detector = None  # MuleChainDetector, created in initialize()
        self._mule_scorer = None  # MuleAccountScorer, created in initialize()
        self._victim_tracer = None  # VictimFundTracer, created in initialize()
        self._fraud_ensemble = None  # FraudEnsemble, created in initialize()
        self._shap_explainer = None  # SHAPExplainer, created in initialize()
        self._drift_detector = None  # ModelDriftDetector, created in initialize()
        self._nl_query_engine = None  # NLQueryEngine, created in initialize()
        self._consortium_hub = None  # ConsortiumHub, created in initialize()

    # ── Initialization ────────────────────────────────────────────────────

    async def initialize(self) -> None:
        """
        Boot all modules in dependency order:
        1. Directories
        2. AuditLedger (async — needs DB open)
        3. CircuitBreaker
        4. GNNScorer + LoadShedder proxy
        5. TransactionGraph (with GNN proxy)
        6. FeatureEngine
        7. FraudClassifier (XGBoost)
        8. DynamicThreshold
        9. LLM stack (PayFlowLLM, NLU agent, ToolExecutor, InvestigatorAgent)
        10. AlertRouter (wire all consumers)
        11. IngestionPipeline (wire all batch consumers)
        12. HardwareProfiler
        """
        logger.info("=" * 60)
        logger.info("PayFlow Orchestrator — Initializing")
        logger.info("=" * 60)

        if self._cpu_only:
            os.environ["PAYFLOW_CPU_ONLY"] = "1"

        # 1. Ensure directories
        from config.settings import DATA_DIR, ARTIFACTS_DIR, LOG_DIR
        for d in (DATA_DIR, ARTIFACTS_DIR, LOG_DIR):
            d.mkdir(parents=True, exist_ok=True)
        logger.info("Directories verified: data/, artifacts/, logs/")

        # 2. AuditLedger
        from src.blockchain.ledger import AuditLedger
        from config.settings import LEDGER_CFG
        self._ledger = AuditLedger(config=LEDGER_CFG)
        await self._ledger.open()
        await self._ledger.anchor_system_event("pipeline_init", {
            "num_accounts": self._num_accounts,
            "num_events": self._num_events,
            "fraud_ratio": self._fraud_ratio,
            "cpu_only": self._cpu_only,
        })
        logger.info("AuditLedger opened (chain head: %d)", self._ledger.head_index)

        # 3. CircuitBreaker
        from src.blockchain.circuit_breaker import CircuitBreaker
        self._breaker = CircuitBreaker(audit_ledger=self._ledger)
        logger.info("CircuitBreaker initialized (consensus threshold: %.2f)",
                     self._breaker._cfg.consensus_threshold)

        # 3.5. Rule Engine + Pre-Approval Gate (RTPM monitoring pipeline)
        from src.ml.rule_engine import TransactionRuleEngine
        from src.ml.pre_approval_gate import PreApprovalGate
        from src.ml.velocity import VelocityTracker
        from src.ml.behavioral import BehavioralAnalyzer
        from src.ml.risk_scorer import TransactionRiskScorer
        from src.ml.device_verifier import DeviceVerifier

        self._rule_engine = TransactionRuleEngine()
        _velocity = VelocityTracker()
        _behavioral = BehavioralAnalyzer()
        _risk_scorer = TransactionRiskScorer()
        _device_verifier = DeviceVerifier()

        # 3.6. Central Fraud Registry (CFR) + CFR-aware scorer
        from src.cfr.registry import CentralFraudRegistry
        from src.cfr.scoring import CFRRiskScorer
        self._fraud_registry = CentralFraudRegistry()
        self._cfr_scorer = CFRRiskScorer(registry=self._fraud_registry)

        # 3.7. Investigation Manager (law enforcement + legal proceedings)
        from src.ml.investigation import InvestigationManager
        self._investigation_mgr = InvestigationManager()

        # 3.8. Random Forest ensemble member
        from src.ml.models.random_forest_classifier import RandomForestFraudClassifier
        self._rf_classifier = RandomForestFraudClassifier()

        # 3.9. Logistic Regression (interpretable baseline for AML audit)
        from src.ml.models.logistic_classifier import LogisticFraudClassifier
        self._lr_classifier = LogisticFraudClassifier()

        # 3.10. AML Stage Detectors (Placement + Integration)
        from src.ml.aml_stages import PlacementDetector, IntegrationDetector
        self._placement_detector = PlacementDetector()
        self._integration_detector = IntegrationDetector()

        # 3.11. FIU-IND Intelligence Unit
        from src.ml.fiu_intelligence import FIUIntelligenceUnit
        self._fiu_intelligence = FIUIntelligenceUnit()

        # 3.12. Mule Chain Detector + Mule Account Scorer (Carbanak)
        from src.graph.chain_detector import MuleChainDetector
        from src.ml.mule_scorer import MuleAccountScorer
        self._mule_chain_detector = MuleChainDetector()
        self._mule_scorer = MuleAccountScorer()

        # 3.13. Victim Fund Tracer (Digital Banking Fraud Flow)
        from src.ml.victim_tracer import VictimFundTracer
        self._victim_tracer = VictimFundTracer()

        # 3.14. Fraud Ensemble Combiner (0.5 XGB + 0.3 ISO + 0.2 AE)
        from src.ml.ensemble import FraudEnsemble
        self._fraud_ensemble = FraudEnsemble()
        logger.info("FraudEnsemble combiner ready (weights: %.1f/%.1f/%.1f)",
                     self._fraud_ensemble._weights.xgboost,
                     self._fraud_ensemble._weights.isolation_forest,
                     self._fraud_ensemble._weights.autoencoder)

        self._pre_approval_gate = PreApprovalGate(
            velocity_tracker=_velocity,
            behavioral_analyzer=_behavioral,
            risk_scorer=_risk_scorer,
            circuit_breaker=self._breaker,
            device_verifier=_device_verifier,
            fraud_registry=self._fraud_registry,
        )
        logger.info(
            "RuleEngine (%d rules) + PreApprovalGate + CFR wired into pipeline",
            len(self._rule_engine.list_rules()),
        )

        # 4. GNNScorer + LoadShedder
        from src.ml.models.gnn_scorer import GNNScorer
        self._gnn_scorer = GNNScorer()
        self._gnn_proxy = LoadShedder(self._gnn_scorer, self.profiler)
        logger.info("GNNScorer + LoadShedder proxy ready (device: %s)",
                     self._gnn_scorer.device)

        # 5. TransactionGraph
        from src.graph.builder import TransactionGraph
        self._graph = TransactionGraph(
            gnn_scorer=self._gnn_proxy,
            audit_ledger=self._ledger,
            circuit_breaker=self._breaker,
        )
        logger.info("TransactionGraph initialized")

        # 6. FeatureEngine
        from src.ml.feature_engine import FeatureEngine
        self._engine = FeatureEngine()
        # Attach CFR registry for extended features (ext_cfr_match, ext_prior_fraud_reports)
        if self._fraud_registry:
            self._engine.attach_cfr(self._fraud_registry)
        logger.info("FeatureEngine ready (%d-dim feature vector)",
                     self._engine.active_feature_dim)

        # 7. FraudClassifier
        from src.ml.models.xgboost_classifier import FraudClassifier
        self._classifier = FraudClassifier()
        logger.info("FraudClassifier ready (device: %s)", self._classifier.device)

        # 8. DynamicThreshold
        from src.ml.models.threshold import DynamicThreshold
        self._threshold = DynamicThreshold()
        logger.info("DynamicThreshold ready (initial: %.2f)",
                     self._threshold.current_threshold)

        # 8.5. Unsupervised Anomaly Detectors (Isolation Forest + Autoencoder)
        from src.ml.isolation_detector import IsolationDetector
        from src.ml.autoencoder_detector import AutoencoderDetector
        self._isolation_detector = IsolationDetector()
        self._autoencoder_detector = AutoencoderDetector()
        logger.info("Anomaly detectors initialized (IsolationForest + Autoencoder)")

        # 8.6. SHAP Explainability + Model Drift Detector
        from src.ml.explainability import SHAPExplainer
        from src.ml.drift_detector import ModelDriftDetector
        self._shap_explainer = SHAPExplainer()
        self._drift_detector = ModelDriftDetector()
        logger.info("SHAPExplainer + ModelDriftDetector ready")

        # 8.7. Cross-Bank Consortium (ZKP-based intelligence sharing)
        from src.blockchain.consortium import ConsortiumHub
        self._consortium_hub = ConsortiumHub()
        # Simulate peer bank intelligence
        peer_alerts = self._consortium_hub.simulate_peer_intelligence(n_alerts=8)
        logger.info("ConsortiumHub initialized (%d member banks, %d peer alerts)",
                     self._consortium_hub.snapshot()["member_banks"],
                     len(peer_alerts))

        # 9. LLM stack
        if not self._skip_llm:
            try:
                from src.llm.orchestrator import PayFlowLLM
                from src.llm.unstructured_agent import UnstructuredAnalysisAgent
                from src.llm.tools import ToolExecutor
                from src.llm.agent import InvestigatorAgent

                self._llm = PayFlowLLM(skip_health_check=self._cpu_only)
                self._nlu_agent = UnstructuredAnalysisAgent(llm_client=self._llm)
                self._tool_executor = ToolExecutor(
                    transaction_graph=self._graph,
                    feature_engine=self._engine,
                    audit_ledger=self._ledger,
                    circuit_breaker=self._breaker,
                    unstructured_agent=self._nlu_agent,
                )
                self._agent = InvestigatorAgent(
                    llm_client=self._llm,
                    tool_executor=self._tool_executor,
                    audit_ledger=self._ledger,
                    unstructured_agent=self._nlu_agent,
                )
                logger.info("LLM stack initialized (InvestigatorAgent + NLU sub-agent)")
            except Exception as exc:
                logger.warning("LLM stack unavailable, continuing without: %s", exc)
                self._skip_llm = True

        # 9.3. NL Query Engine (Qwen 3.5 powered)
        from src.llm.nl_query import NLQueryEngine
        self._nl_query_engine = NLQueryEngine(
            llm_client=self._llm,
            orchestrator=self,
        )
        logger.info("NLQueryEngine ready (LLM=%s)", self._llm is not None)

        # 9.5. Wire CircuitBreaker graph + LLM + reporter attachments
        if self._graph:
            self._breaker.attach_graph(self._graph)
        if self._agent:
            self._breaker.attach_llm_agent(self._agent)
        logger.info("CircuitBreaker wired (graph=%s, llm=%s)",
                     self._graph is not None, self._agent is not None)

        # 10. AlertRouter
        from src.ml.models.alert_router import AlertRouter
        self._router = AlertRouter()
        self._router.register_graph_consumer(self._graph.investigate)
        self._router.register_ledger_consumer(self._ledger.anchor_alert)
        self._router.register_circuit_breaker_consumer(self._breaker.on_alert)
        if self._agent:
            self._router.register_agent_consumer(self._agent.on_alert)
        logger.info("AlertRouter wired (%d consumer groups)",
                     3 + (1 if self._agent else 0))

        # 11. IngestionPipeline
        from src.ingestion.stream_processor import IngestionPipeline
        self._pipeline = IngestionPipeline(
            batch_size=self._batch_size,
            batch_timeout_sec=1.0,
        )
        self._pipeline.add_consumer(self._engine.ingest)
        self._pipeline.add_consumer(self._graph.ingest)
        self._pipeline.add_consumer(self._ledger.ingest)
        logger.info("IngestionPipeline ready (batch_size=%d, 3 consumers)",
                     self._batch_size)

        # 12. Synthetic world
        from src.ingestion.generators.synthetic_transactions import build_world
        self._world = build_world(num_accounts=self._num_accounts)
        logger.info("Synthetic world built (%d accounts, %d dormant)",
                     len(self._world.accounts), len(self._world.dormant_accounts))

        # 12.5. Threat Simulation Engine
        from src.simulation.threat_engine import ThreatSimulationEngine
        self._threat_engine = ThreatSimulationEngine(
            pipeline=self._pipeline,
            world=self._world,
        )
        logger.info("ThreatSimulationEngine ready (%d attack types)",
                     len(self._threat_engine.available_attacks()))

        # 13. HardwareProfiler
        shed_event = asyncio.Event()
        resume_event = asyncio.Event()
        self.profiler.set_shed_events(shed_event, resume_event)
        await self.profiler.start()

        # 14. GPU Priority Queue (Phase 15 — cooperative VRAM sharing)
        if not self._cpu_only:
            try:
                from config.gpu_concurrency import GPUPriorityQueue
                from config.settings import GPU_CONCURRENCY_CFG
                self._gpu_queue = GPUPriorityQueue(GPU_CONCURRENCY_CFG)
                # Wire profiler -> priority queue (thread->async bridge)
                loop = asyncio.get_running_loop()
                self.profiler.set_priority_queue(self._gpu_queue, loop)
                # Wire LLM -> dynamic num_ctx
                if self._llm is not None:
                    self._llm.set_priority_queue(self._gpu_queue)
                # Register ctx_change callback for dashboard events
                def _on_ctx_change(new_ctx: int) -> None:
                    try:
                        from src.api.events import EventBroadcaster
                        EventBroadcaster.get().publish_sync("system", {
                            "type": "gpu_pressure",
                            "num_ctx": new_ctx,
                            "pressure": self._gpu_queue.pressure.value,
                        })
                    except Exception:
                        pass
                self._gpu_queue.set_ctx_change_callback(_on_ctx_change)
                logger.info("GPUPriorityQueue initialized (cooperative VRAM sharing)")
            except Exception as exc:
                logger.warning("GPU priority queue unavailable: %s", exc)

        # 15. Dashboard (optional — FastAPI + SSE)
        if self._enable_dashboard:
            try:
                from src.api.app import create_app
                import uvicorn

                self._dashboard_app = create_app(orchestrator=self)
                config = uvicorn.Config(
                    self._dashboard_app,
                    host="127.0.0.1",
                    port=self._dashboard_port,
                    log_level="warning",
                )
                server = uvicorn.Server(config)
                self._dashboard_task = asyncio.create_task(
                    server.serve(), name="dashboard-server",
                )
                self._telemetry_task = asyncio.create_task(
                    self._broadcast_telemetry(), name="telemetry-broadcaster",
                )
                logger.info(
                    "Dashboard server started on http://127.0.0.1:%d",
                    self._dashboard_port,
                )
            except Exception as exc:
                logger.warning("Dashboard startup failed: %s", exc)

        logger.info("=" * 60)
        logger.info("Initialization complete — all modules online")
        logger.info("=" * 60)

    # ── Main Execution Loop ───────────────────────────────────────────────

    async def run(self) -> OrchestratorMetrics:
        """
        Execute the full pipeline:
        Phase A — Ingest synthetic event stream (concurrent feature + graph)
        Phase B — Train ML models on accumulated features
        Phase C — Run inference -> threshold -> alert routing -> investigation
        """
        self.metrics.pipeline_start_time = time.time()
        logger.info("Pipeline execution starting (%d events, %.0f%% fraud)",
                     self._num_events, self._fraud_ratio * 100)

        # ─── Phase A: Data Ingestion ──────────────────────────────────────
        await self._phase_ingest()

        # ─── Phase B: Model Training ─────────────────────────────────────
        await self._phase_train()

        # ─── Phase C: Inference + Routing ────────────────────────────────
        await self._phase_inference()

        # Set inference cursor to current position so live loop only
        # scores NEW events arriving from simulation/threat engine
        self._inference_cursor = len(self._engine._feature_store)

        # Start live inference loop (continuous scoring for simulation events)
        self._live_inference_task = asyncio.create_task(
            self._live_inference_loop(), name="live-inference",
        )

        self.metrics.pipeline_end_time = time.time()
        logger.info(
            "Pipeline complete: %d events in %.1fs (%.0f events/sec)",
            self.metrics.events_ingested,
            self.metrics.elapsed_sec,
            self.metrics.events_per_sec,
        )
        return self.metrics

    async def _phase_ingest(self) -> None:
        """Phase A: stream synthetic events through the ingestion pipeline."""
        logger.info("─── Phase A: Data Ingestion ───")

        from src.ingestion.generators.synthetic_transactions import (
            generate_event_stream,
        )

        events = list(generate_event_stream(
            self._world,
            num_events=self._num_events,
            fraud_ratio=self._fraud_ratio,
        ))

        await self._ledger.anchor_system_event("ingestion_start", {
            "total_events": len(events),
        })

        # Start pipeline (kept alive for simulation events in --serve mode)
        await self._pipeline.start()

        for event in events:
            await self._pipeline.ingest(event)

        # Save metrics before flush-restart cycle
        ingested_count = self._pipeline.metrics.events_ingested
        logger.info("Captured ingested_count = %d before pipeline restart", ingested_count)

        # Flush remaining events by sending shutdown and restarting
        await self._pipeline.stop()
        await self._pipeline.start()

        self.metrics.events_ingested = ingested_count
        self.metrics.batches_processed = self._engine.metrics.batches_processed
        self.metrics.features_extracted = (
            self._engine.get_training_data().features.shape[0]
            if self._engine.get_training_data() is not None else 0
        )

        logger.info(
            "Ingestion done: %d events -> %d batches -> %d feature rows, "
            "graph has %d nodes / %d edges",
            self.metrics.events_ingested,
            self.metrics.batches_processed,
            self.metrics.features_extracted,
            self._graph.metrics.nodes,
            self._graph.metrics.transactions_added,
        )

    async def _phase_train(self) -> None:
        """Phase B: train XGBoost and GNN on accumulated data."""
        logger.info("─── Phase B: Model Training ───")

        train_data = self._engine.get_training_data()
        if train_data is None or train_data.features.shape[0] == 0:
            logger.warning("No training data available — skipping training")
            return

        import numpy as np
        from config.vram_manager import analysis_mode
        from src.ml.feature_engine import FEATURE_COLUMNS

        features = train_data.features
        labels = train_data.labels

        # Split 80/20
        n = features.shape[0]
        split = int(n * 0.8)
        indices = np.random.default_rng(42).permutation(n)
        train_idx, eval_idx = indices[:split], indices[split:]

        X_train, y_train = features[train_idx], labels[train_idx]
        X_eval, y_eval = features[eval_idx], labels[eval_idx]

        # Train XGBoost under analysis VRAM mode
        with analysis_mode():
            logger.info("Training XGBoost on %d samples (%d eval)...",
                         len(train_idx), len(eval_idx))
            t0 = time.monotonic()
            train_metrics = self._classifier.train(
                X_train, y_train, X_eval, y_eval,
                feature_names=FEATURE_COLUMNS,
            )
            elapsed = time.monotonic() - t0
            logger.info(
                "XGBoost trained in %.1fs — AUCPR: %.4f, "
                "best_iter: %d, device: %s",
                elapsed,
                train_metrics.best_aucpr,
                train_metrics.best_iteration,
                train_metrics.device_used,
            )

            # Anchor model event
            await self._ledger.anchor_model_event("train", "xgboost", {
                "aucpr": train_metrics.best_aucpr,
                "n_train": train_metrics.n_train,
                "n_eval": train_metrics.n_eval,
                "device": train_metrics.device_used,
                "elapsed_ms": elapsed * 1000,
            })

            # Validate
            val_metrics = self._classifier.validate(X_eval, y_eval)
            logger.info(
                "Validation — AUCPR: %.4f, ROC-AUC: %.4f, F1: %.4f, "
                "Precision: %.4f, Recall: %.4f",
                val_metrics.aucpr, val_metrics.auc_roc, val_metrics.f1,
                val_metrics.precision, val_metrics.recall,
            )

        # Train unsupervised anomaly detectors
        if self._isolation_detector is not None:
            t0 = time.monotonic()
            self._isolation_detector.fit(features)
            elapsed = time.monotonic() - t0
            logger.info(
                "IsolationForest fitted on %d samples in %.1fs",
                features.shape[0], elapsed,
            )

        if self._autoencoder_detector is not None:
            t0 = time.monotonic()
            self._autoencoder_detector.fit(features, labels)
            elapsed = time.monotonic() - t0
            ae_snap = self._autoencoder_detector.snapshot()
            logger.info(
                "Autoencoder fitted on %d legitimate samples in %.1fs "
                "(threshold: %.6f)",
                ae_snap.get("fit_count", 0), elapsed,
                ae_snap.get("threshold", 0.0),
            )

        # Train Random Forest ensemble member
        if self._rf_classifier is not None:
            t0 = time.monotonic()
            rf_metrics = self._rf_classifier.train(
                X_train, y_train, X_eval, y_eval,
                feature_names=FEATURE_COLUMNS,
            )
            elapsed = time.monotonic() - t0
            logger.info(
                "RandomForest fitted on %d samples in %.1fs — AUCPR: %.4f, OOB: %.4f",
                rf_metrics.n_train, elapsed,
                rf_metrics.best_aucpr, rf_metrics.oob_score,
            )

        # Train Logistic Regression baseline
        if self._lr_classifier is not None:
            t0 = time.monotonic()
            lr_metrics = self._lr_classifier.train(
                X_train, y_train, X_eval, y_eval,
                feature_names=FEATURE_COLUMNS,
            )
            elapsed = time.monotonic() - t0
            logger.info(
                "LogisticRegression fitted on %d samples in %.1fs — AUCPR: %.4f, converged: %s",
                lr_metrics.n_train, elapsed,
                lr_metrics.best_aucpr, lr_metrics.converged,
            )

        # Attach SHAP explainer to trained XGBoost model
        if self._shap_explainer and self._classifier.is_fitted:
            self._shap_explainer.attach_model(self._classifier)
            logger.info("SHAPExplainer attached to trained XGBoost model")

        # Set drift detector reference distribution
        if self._drift_detector:
            import numpy as _np
            self._drift_detector.set_reference(
                pred_result.risk_scores
                if hasattr(self, '_last_train_scores') else
                _np.random.default_rng(42).uniform(0, 0.4, size=n).astype(_np.float32)
            )
            logger.info("DriftDetector reference distribution set (%d samples)", n)

    async def _phase_inference(self) -> None:
        """Phase C: run inference, threshold, and route alerts."""
        logger.info("─── Phase C: Inference + Alert Routing ───")

        train_data = self._engine.get_training_data()
        if train_data is None or not self._classifier.is_fitted:
            logger.warning(
                "Skipping inference: %s",
                "no data" if train_data is None else "model not fitted",
            )
            return

        import numpy as np
        from config.vram_manager import analysis_mode
        from src.ml.feature_engine import FEATURE_COLUMNS
        from src.ml.models.alert_router import AlertRouter

        features = train_data.features
        n = features.shape[0]

        # Run XGBoost inference under analysis VRAM mode
        with analysis_mode():
            logger.info("Running inference on %d transactions...", n)
            t0 = time.monotonic()
            pred_result = self._classifier.predict(features)
            elapsed = time.monotonic() - t0
            self.metrics.ml_inferences = n
            logger.info(
                "Inference complete in %.1fms — %d flagged (>0.5)",
                pred_result.inference_ms,
                int(pred_result.predicted_labels.sum()),
            )

            # Broadcast initial ML scoring stage to frontend
            try:
                from src.api.events import EventBroadcaster
                broadcaster = EventBroadcaster.get()
                await broadcaster.publish("pipeline", {
                    "type": "stage_complete",
                    "stage": "ml_scored",
                    "batch_size": n,
                    "duration_ms": round(elapsed * 1000, 1),
                    "flagged_count": int(pred_result.predicted_labels.sum()),
                    "txn_ids": train_data.txn_ids[:20],
                })
            except Exception:
                pass

        # Threshold evaluation
        threshold_results = self._threshold.evaluate_batch(pred_result.risk_scores)

        # ── Rule Engine batch evaluation (RTPM monitoring) ──
        # Run flagged transactions through the rule engine for auditable
        # rule-based detection alongside ML scoring.
        if self._rule_engine:
            flagged_indices = np.where(pred_result.risk_scores > 0.3)[0]
            rule_violations_total = 0
            for idx in flagged_indices[:500]:  # cap to avoid startup lag
                self._rule_engine.evaluate(
                    amount_paisa=int(features[idx][0] * 100) if features.shape[1] > 0 else 0,
                    sender_id=train_data.sender_ids[idx],
                    receiver_id=train_data.receiver_ids[idx],
                    timestamp=int(train_data.timestamps[idx]),
                    device_trusted=True,
                )
            logger.info(
                "RuleEngine: evaluated %d flagged transactions (%d rules × %d txns)",
                min(len(flagged_indices), 500),
                len(self._rule_engine.list_rules()),
                min(len(flagged_indices), 500),
            )

        # Build alert payloads (filters out LOW tier)
        payloads = AlertRouter.build_payloads(
            threshold_results=threshold_results,
            txn_ids=train_data.txn_ids,
            sender_ids=train_data.sender_ids,
            receiver_ids=train_data.receiver_ids,
            timestamps=train_data.timestamps,
            features=features,
            feature_names=FEATURE_COLUMNS,
        )

        high_count = sum(1 for p in payloads if p.tier == "HIGH")
        med_count = sum(1 for p in payloads if p.tier == "MEDIUM")
        logger.info(
            "%d alerts generated: %d HIGH, %d MEDIUM (from %d threshold results)",
            len(payloads), high_count, med_count, len(threshold_results),
        )

        # Route alerts (graph investigation, circuit breaker, agent — all async)
        for payload in payloads:
            await self._router.route(payload)
            self.metrics.alerts_routed += 1

        # Allow any pending async work to drain
        await asyncio.sleep(0.1)

        # Expire old circuit breaker entries
        expired = await self._breaker.cleanup_expired()
        if expired:
            logger.info("CircuitBreaker: %d expired freeze orders cleaned up", expired)

        logger.info(
            "Routing complete: %d alerts dispatched, %d nodes frozen",
            self.metrics.alerts_routed,
            len(self._breaker.get_frozen_nodes()),
        )

    # ── Live Inference Loop (continuous scoring for simulation events) ──

    async def _live_inference_loop(self) -> None:
        """
        Background task that continuously scores new features as they
        arrive from the IngestionPipeline (e.g. simulation events).

        Polls the FeatureEngine's _feature_store for new entries beyond
        the _inference_cursor and runs XGBoost → Threshold → AlertRouter
        on the delta.
        """
        import numpy as np
        from src.ml.feature_engine import FEATURE_COLUMNS
        from src.ml.models.alert_router import AlertRouter

        logger.info("Live inference loop started (polling every 2s)")

        while True:
            try:
                await asyncio.sleep(2.0)

                if not self._classifier or not self._classifier.is_fitted:
                    continue

                store = self._engine._feature_store
                current_len = len(store)

                if current_len <= self._inference_cursor:
                    continue  # No new data

                # Collect new ExtractionResults since last cursor
                new_results = store[self._inference_cursor:current_len]
                self._inference_cursor = current_len

                # Concatenate new features
                all_features = np.concatenate(
                    [r.features for r in new_results], axis=0
                )
                all_txn_ids = [tid for r in new_results for tid in r.txn_ids]
                all_senders = [sid for r in new_results for sid in r.sender_ids]
                all_receivers = [rid for r in new_results for rid in r.receiver_ids]
                all_timestamps = np.concatenate(
                    [r.timestamps for r in new_results], axis=0
                )

                n = all_features.shape[0]
                if n == 0:
                    continue

                logger.info(
                    "Live inference: scoring %d new transactions...", n,
                )

                # XGBoost inference (run in thread to avoid blocking event loop)
                t_ml = time.monotonic()
                pred_result = await asyncio.to_thread(
                    self._classifier.predict, all_features,
                )
                ml_elapsed_ms = (time.monotonic() - t_ml) * 1000
                self.metrics.ml_inferences += n

                # Broadcast per-batch ML scoring stage to frontend
                try:
                    from src.api.events import EventBroadcaster
                    broadcaster = EventBroadcaster.get()
                    await broadcaster.publish("pipeline", {
                        "type": "stage_complete",
                        "stage": "ml_scored",
                        "batch_size": n,
                        "duration_ms": round(ml_elapsed_ms, 1),
                        "flagged_count": int(pred_result.predicted_labels.sum()),
                        "txn_ids": all_txn_ids[:20],  # cap for SSE payload size
                    })
                except Exception:
                    pass

                # Threshold evaluation
                threshold_results = self._threshold.evaluate_batch(
                    pred_result.risk_scores,
                )

                # ── Rule Engine + Pre-Approval Gate (RTPM monitoring) ──
                # Run each transaction through the rule engine and gate,
                # broadcasting decisions on the transaction_decision channel.
                gate_decisions = {"APPROVE": 0, "HOLD": 0, "BLOCK": 0}
                if self._rule_engine and self._pre_approval_gate:
                    try:
                        from src.api.events import EventBroadcaster
                        _bc = EventBroadcaster.get()
                        for i in range(n):
                            # Rule engine evaluation
                            rule_result = self._rule_engine.evaluate(
                                amount_paisa=int(all_features[i][0] * 100) if all_features.shape[1] > 0 else 0,
                                sender_id=all_senders[i],
                                receiver_id=all_receivers[i],
                                timestamp=int(all_timestamps[i]),
                                device_trusted=True,
                            )

                            # Pre-approval gate evaluation
                            gate_result = self._pre_approval_gate.evaluate(
                                sender_id=all_senders[i],
                                receiver_id=all_receivers[i],
                                amount_paisa=int(all_features[i][0] * 100) if all_features.shape[1] > 0 else 0,
                                timestamp=int(all_timestamps[i]),
                                sender_lat=0.0,
                                sender_lon=0.0,
                                device_fingerprint="",
                            )

                            decision = gate_result.decision.value
                            gate_decisions[decision] = gate_decisions.get(decision, 0) + 1

                            # Broadcast HIGH-risk gate decisions (HOLD/BLOCK)
                            if decision != "APPROVE":
                                await _bc.publish("transaction_decision", {
                                    "type": "gate_decision",
                                    "txn_id": all_txn_ids[i] if i < len(all_txn_ids) else f"txn_{i}",
                                    "sender_id": all_senders[i],
                                    "receiver_id": all_receivers[i],
                                    "decision": decision,
                                    "risk_score": round(gate_result.risk_score, 4),
                                    "reason": gate_result.reason,
                                    "rule_violations": len(rule_result.violations),
                                    "rule_max_severity": rule_result.max_severity.value,
                                    "evaluation_ms": round(gate_result.evaluation_ms, 2),
                                })

                        # Broadcast batch summary
                        await _bc.publish("transaction_decision", {
                            "type": "batch_summary",
                            "batch_size": n,
                            "approved": gate_decisions["APPROVE"],
                            "held": gate_decisions["HOLD"],
                            "blocked": gate_decisions["BLOCK"],
                        })
                    except Exception as exc:
                        logger.debug("Gate evaluation error: %s", exc)

                # Build alert payloads (filters out LOW tier)
                payloads = AlertRouter.build_payloads(
                    threshold_results=threshold_results,
                    txn_ids=all_txn_ids,
                    sender_ids=all_senders,
                    receiver_ids=all_receivers,
                    timestamps=all_timestamps,
                    features=all_features,
                    feature_names=FEATURE_COLUMNS,
                )

                if payloads:
                    high_count = sum(1 for p in payloads if p.tier == "HIGH")
                    med_count = sum(1 for p in payloads if p.tier == "MEDIUM")
                    logger.info(
                        "Live inference: %d alerts (%d HIGH, %d MEDIUM) "
                        "from %d transactions",
                        len(payloads), high_count, med_count, n,
                    )

                    # Route alerts through the full pipeline
                    for payload in payloads:
                        await self._router.route(payload)
                        self.metrics.alerts_routed += 1

                # Update orchestrator metrics for dashboard (accumulate, don't overwrite)
                live_ingested = self._pipeline.metrics.events_ingested
                if live_ingested > 0:
                    self.metrics.events_ingested += live_ingested
                    # Reset pipeline counter to avoid double-counting
                    self._pipeline.metrics.events_ingested = 0

            except asyncio.CancelledError:
                logger.info("Live inference loop stopped")
                break
            except Exception as exc:
                logger.error("Live inference error: %s", exc, exc_info=True)
                await asyncio.sleep(5.0)  # Back off on error

    # ── Shutdown ──────────────────────────────────────────────────────────

    async def shutdown(self) -> None:
        """Graceful teardown in reverse dependency order."""
        logger.info("─── Shutdown ───")

        # Stop threat simulation engine
        if self._threat_engine:
            await self._threat_engine.stop_all()

        # Stop live inference loop
        if self._live_inference_task and not self._live_inference_task.done():
            self._live_inference_task.cancel()
            try:
                await self._live_inference_task
            except (asyncio.CancelledError, Exception):
                pass

        # Stop the ingestion pipeline (kept alive for simulation)
        if self._pipeline and self._pipeline._running:
            await self._pipeline.stop()

        # Cancel dashboard tasks
        for task in (self._telemetry_task, self._dashboard_task):
            if task is not None and not task.done():
                task.cancel()
                try:
                    await task
                except (asyncio.CancelledError, Exception):
                    pass

        if self._router:
            await self._router.shutdown()

        await self.profiler.stop()

        if self._llm:
            try:
                self._llm.unload()
            except Exception:
                pass

        # Anchor final system state
        if self._ledger:
            await self._ledger.anchor_system_event("pipeline_shutdown", {
                "events_ingested": self.metrics.events_ingested,
                "alerts_routed": self.metrics.alerts_routed,
                "elapsed_sec": round(self.metrics.elapsed_sec, 2),
            })

        # Verify chain integrity
        if self._ledger:
            verification = await self._ledger.verify_chain()
            if verification.valid:
                logger.info(
                    "Ledger integrity verified: %d blocks, chain valid",
                    verification.blocks_checked,
                )
            else:
                logger.error(
                    "LEDGER INTEGRITY FAILURE at block %d: %s",
                    verification.first_invalid_index,
                    verification.error_message,
                )
            await self._ledger.close()

        logger.info("Shutdown complete.")

    # ── Diagnostics ───────────────────────────────────────────────────────

    # ── Dashboard Telemetry ─────────────────────────────────────────────

    async def _broadcast_telemetry(self) -> None:
        """Publish system-wide telemetry snapshots to the dashboard at 1 Hz."""
        from src.api.events import EventBroadcaster
        broadcaster = EventBroadcaster.get()
        while True:
            try:
                snap = self.full_snapshot()
                await broadcaster.publish("system", {
                    "type": "telemetry", **snap,
                })
            except Exception:
                pass
            await asyncio.sleep(1.0)

    # ── Diagnostics ───────────────────────────────────────────────────────

    def full_snapshot(self) -> dict:
        """Complete system state for monitoring dashboards."""
        snap = {
            "orchestrator": {
                "events_ingested": self.metrics.events_ingested,
                "features_extracted": self.metrics.features_extracted,
                "ml_inferences": self.metrics.ml_inferences,
                "alerts_routed": self.metrics.alerts_routed,
                "elapsed_sec": round(self.metrics.elapsed_sec, 2),
                "events_per_sec": round(self.metrics.events_per_sec, 1),
            },
            "hardware": self.profiler.snapshot(),
        }
        if self._gnn_proxy:
            snap["load_shedder"] = self._gnn_proxy.snapshot()
        if self._graph:
            snap["graph"] = self._graph.snapshot()
        if self._breaker:
            snap["circuit_breaker"] = self._breaker.snapshot()
        if self._agent:
            snap["agent"] = self._agent.snapshot()
        if self._threshold:
            snap["threshold"] = self._threshold.snapshot()
        if self._gpu_queue:
            snap["gpu_concurrency"] = self._gpu_queue.snapshot()
        if self._threat_engine:
            snap["threat_simulation"] = self._threat_engine.snapshot()
        if self._rule_engine:
            snap["rule_engine"] = self._rule_engine.snapshot()
        if self._pre_approval_gate:
            snap["pre_approval_gate"] = self._pre_approval_gate.metrics.snapshot()
        if self._isolation_detector:
            snap["isolation_detector"] = self._isolation_detector.snapshot()
        if self._autoencoder_detector:
            snap["autoencoder_detector"] = self._autoencoder_detector.snapshot()
        if self._fraud_registry:
            snap["cfr_registry"] = self._fraud_registry.snapshot()
        if self._cfr_scorer:
            snap["cfr_scorer"] = self._cfr_scorer.snapshot()
        if self._investigation_mgr:
            snap["investigation"] = self._investigation_mgr.snapshot()
        if self._rf_classifier and self._rf_classifier.is_fitted:
            snap["random_forest"] = self._rf_classifier.training_info
        if self._lr_classifier and self._lr_classifier.is_fitted:
            snap["logistic_regression"] = self._lr_classifier.training_info
        if self._placement_detector:
            snap["aml_placement"] = self._placement_detector.snapshot()
        if self._integration_detector:
            snap["aml_integration"] = self._integration_detector.snapshot()
        if self._fiu_intelligence:
            snap["fiu_intelligence"] = self._fiu_intelligence.snapshot()
        if self._mule_chain_detector:
            snap["mule_chain_detector"] = self._mule_chain_detector.snapshot()
        if self._mule_scorer:
            snap["mule_scorer"] = self._mule_scorer.snapshot()
        if self._victim_tracer:
            snap["victim_tracer"] = self._victim_tracer.snapshot()
        if self._fraud_ensemble:
            snap["fraud_ensemble"] = self._fraud_ensemble.snapshot()
        if self._shap_explainer:
            snap["explainability"] = self._shap_explainer.snapshot()
        if self._drift_detector:
            snap["model_drift"] = self._drift_detector.snapshot()
        if self._nl_query_engine:
            snap["nl_query"] = self._nl_query_engine.snapshot()
        if self._consortium_hub:
            snap["consortium"] = self._consortium_hub.snapshot()
        return snap

    # ── Context Manager ───────────────────────────────────────────────────

    async def __aenter__(self):
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.shutdown()
        return False


# ════════════════════════════════════════════════════════════════════════════
# CLI ENTRY POINT
# ════════════════════════════════════════════════════════════════════════════

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="payflow",
        description="PayFlow Fraud Detection — Unified Execution Engine",
    )
    p.add_argument(
        "--events", type=int, default=10_000,
        help="Number of synthetic events to generate (default: 10000)",
    )
    p.add_argument(
        "--accounts", type=int, default=2000,
        help="Number of synthetic accounts (default: 2000)",
    )
    p.add_argument(
        "--fraud-ratio", type=float, default=0.05,
        help="Fraction of events that are fraudulent (default: 0.05)",
    )
    p.add_argument(
        "--batch-size", type=int, default=512,
        help="Ingestion pipeline micro-batch size (default: 512)",
    )
    p.add_argument(
        "--cpu-only", action="store_true",
        help="Force CPU-only execution (no GPU required)",
    )
    p.add_argument(
        "--skip-llm", action="store_true",
        help="Skip LLM agent initialization (for testing without Ollama)",
    )
    p.add_argument(
        "--vram-shed-mb", type=float, default=7500.0,
        help="VRAM threshold (MB) to trigger GNN load-shedding (default: 7500)",
    )
    p.add_argument(
        "--vram-resume-mb", type=float, default=6500.0,
        help="VRAM threshold (MB) to resume GNN scoring (default: 6500)",
    )
    p.add_argument(
        "--no-dashboard", action="store_true",
        help="Disable the real-time web dashboard",
    )
    p.add_argument(
        "--dashboard-port", type=int, default=8000,
        help="Port for the dashboard web server (default: 8000)",
    )
    p.add_argument(
        "--serve", action="store_true",
        help="Keep dashboard server alive after pipeline completes (demo mode)",
    )
    return p


def _print_banner() -> None:
    print("""
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║   ██████╗  █████╗ ██╗   ██╗███████╗██╗      ██████╗ ██╗    ║
║   ██╔══██╗██╔══██╗╚██╗ ██╔╝██╔════╝██║     ██╔═══██╗██║    ║
║   ██████╔╝███████║ ╚████╔╝ █████╗  ██║     ██║   ██║██║ █╗ ║
║   ██╔═══╝ ██╔══██║  ╚██╔╝  ██╔══╝  ██║     ██║   ██║██║███╗║
║   ██║     ██║  ██║   ██║   ██║     ███████╗╚██████╔╝╚███╔██║
║   ╚═╝     ╚═╝  ╚═╝   ╚═╝   ╚═╝     ╚══════╝ ╚═════╝  ╚══╝╚╝║
║                                                              ║
║   Real-Time Fraud Intelligence for Union Bank of India       ║
║   ML · GNN · Blockchain · LLM Agents · ZKP Privacy          ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
""")


def _print_summary(metrics: OrchestratorMetrics, hw: dict) -> None:
    print("\n" + "═" * 60)
    print("  EXECUTION SUMMARY")
    print("═" * 60)
    print(f"  Events ingested:    {metrics.events_ingested:>10,}")
    print(f"  Features extracted: {metrics.features_extracted:>10,}")
    print(f"  ML inferences:      {metrics.ml_inferences:>10,}")
    print(f"  Alerts routed:      {metrics.alerts_routed:>10,}")
    print(f"  Elapsed:            {metrics.elapsed_sec:>9.1f}s")
    print(f"  Throughput:         {metrics.events_per_sec:>9.0f} events/sec")
    print("─" * 60)
    print(f"  GPU VRAM Used:      {hw.get('gpu_vram_used_mb', 0):>8.0f} MB")
    print(f"  GPU VRAM Total:     {hw.get('gpu_vram_total_mb', 0):>8.0f} MB")
    print(f"  GPU Utilization:    {hw.get('gpu_utilization_pct', -1):>7d} %")
    print(f"  CPU Utilization:    {hw.get('cpu_utilization_pct', 0):>9.1f} %")
    print(f"  LLM TPS:            {hw.get('llm_tps', 0):>9.1f} tok/s")
    print(f"  LLM Tokens Total:   {hw.get('llm_tokens_total', 0):>10,}")
    print(f"  Load-Shed Active:   {'YES' if hw.get('load_shed_active') else 'NO':>10s}")
    print("═" * 60)


async def async_main(args: argparse.Namespace) -> None:
    orchestrator = PayFlowOrchestrator(
        num_accounts=args.accounts,
        num_events=args.events,
        fraud_ratio=args.fraud_ratio,
        batch_size=args.batch_size,
        cpu_only=args.cpu_only,
        skip_llm=args.skip_llm,
        vram_shed_threshold_mb=args.vram_shed_mb,
        vram_resume_threshold_mb=args.vram_resume_mb,
        enable_dashboard=not args.no_dashboard,
        dashboard_port=args.dashboard_port,
    )

    try:
        await orchestrator.initialize()
        try:
            metrics = await orchestrator.run()
            hw = orchestrator.profiler.snapshot()
            _print_summary(metrics, hw)
        except Exception as exc:
            logger.warning("Pipeline run failed: %s — dashboard still alive", exc)
        if args.serve:
            logger.info("Serve mode: dashboard staying alive at http://127.0.0.1:%d — Ctrl+C to stop", args.dashboard_port)
            await asyncio.sleep(float('inf'))
    finally:
        await orchestrator.shutdown()


def main() -> None:
    _print_banner()
    parser = _build_parser()
    args = parser.parse_args()

    logger.info("Configuration: accounts=%d, events=%d, fraud=%.0f%%, "
                "cpu_only=%s, skip_llm=%s",
                args.accounts, args.events, args.fraud_ratio * 100,
                args.cpu_only, args.skip_llm)

    asyncio.run(async_main(args))


if __name__ == "__main__":
    main()

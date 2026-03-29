"""
PayFlow — Async Alert Router
==============================
Routes high-risk transaction payloads to downstream investigation modules
when the dynamic threshold is breached.

Pipeline Position:
    FeatureEngine → FraudClassifier → DynamicThreshold → AlertRouter
                                                              │
                                                    ┌────────┴────────┐
                                                    ▼                 ▼
                                            Graph Analytics      LLM Orchestrator
                                            (GNN subgraph)     (forensic analysis)

Design:
    - Fully async (asyncio) — non-blocking dispatch to both downstream modules.
    - Fire-and-forget delivery with bounded retry (max 2 attempts).
    - Alert payloads carry full provenance: txn_id, features, risk_score,
      threshold at time of evaluation, and fraud tier.
    - In-memory asyncio.Queue for decoupling production rate from consumption
      rate; backpressure at 10K pending alerts triggers warning + drop oldest.

Integration Points:
    - Graph module: `src.graph` (future phase) — receives sender/receiver IDs
      for subgraph extraction and GNN scoring.
    - LLM module: `src.llm.orchestrator.PayFlowLLM` — receives transaction
      context for natural language forensic analysis.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Coroutine

import numpy as np

from src.ml.models.threshold import RiskTier, ThresholdResult

logger = logging.getLogger(__name__)


# ── Alert Payload ────────────────────────────────────────────────────────────

@dataclass
class AlertPayload:
    """
    Enriched alert package dispatched to downstream investigation modules.

    Contains everything needed for graph subgraph extraction and LLM analysis
    without requiring a round-trip to the feature store.
    """
    txn_id: str
    sender_id: str
    receiver_id: str
    timestamp: int
    risk_score: float
    tier: str                    # RiskTier value
    threshold_at_eval: float     # dynamic threshold when alert was raised
    features: np.ndarray         # (30,) float32 — full feature vector
    feature_names: list[str]     # column names for interpretability

    def to_dict(self) -> dict:
        """Serializable representation for LLM context injection."""
        top_features = {}
        if self.features is not None and self.feature_names:
            # Include top-5 features by absolute value for LLM context
            indices = np.argsort(np.abs(self.features))[::-1][:5]
            for idx in indices:
                top_features[self.feature_names[idx]] = round(float(self.features[idx]), 4)

        return {
            "txn_id": self.txn_id,
            "sender_id": self.sender_id,
            "receiver_id": self.receiver_id,
            "timestamp": self.timestamp,
            "risk_score": round(self.risk_score, 4),
            "tier": self.tier,
            "threshold": round(self.threshold_at_eval, 4),
            "top_features": top_features,
        }


# ── Router Metrics ───────────────────────────────────────────────────────────

@dataclass
class RouterMetrics:
    alerts_dispatched: int = 0
    graph_deliveries: int = 0
    llm_deliveries: int = 0
    graph_failures: int = 0
    llm_failures: int = 0
    high_alerts: int = 0
    medium_alerts: int = 0
    dropped_backpressure: int = 0
    _start: float = field(default_factory=time.monotonic)

    def snapshot(self) -> dict:
        return {
            "dispatched": self.alerts_dispatched,
            "graph_ok": self.graph_deliveries,
            "llm_ok": self.llm_deliveries,
            "graph_fail": self.graph_failures,
            "llm_fail": self.llm_failures,
            "high": self.high_alerts,
            "medium": self.medium_alerts,
            "dropped": self.dropped_backpressure,
            "uptime_sec": round(time.monotonic() - self._start, 1),
        }


# ── Type aliases for consumer callbacks ──────────────────────────────────────

GraphConsumer = Callable[[AlertPayload], Coroutine[Any, Any, None]]
LLMConsumer = Callable[[AlertPayload], Coroutine[Any, Any, None]]
LedgerConsumer = Callable[[AlertPayload], Coroutine[Any, Any, None]]
CircuitBreakerConsumer = Callable[[AlertPayload], Coroutine[Any, Any, None]]
AgentConsumer = Callable[[AlertPayload], Coroutine[Any, Any, None]]


# ── Alert Router ─────────────────────────────────────────────────────────────

class AlertRouter:
    """
    Async dispatcher for high-risk alerts to Graph Analytics and LLM modules.

    Both consumers are optional — the router gracefully skips unregistered
    modules (e.g., before the Graph module is implemented in a later phase).

    Usage:
        router = AlertRouter()

        # Register consumers (phase-dependent — LLM available now, Graph later)
        router.register_graph_consumer(graph_module.investigate)
        router.register_llm_consumer(llm_investigate)

        # Route alerts from threshold evaluation
        await router.route(alert_payload)

        # Or batch-route after inference
        await router.route_batch(payloads)
    """

    def __init__(self, max_queue: int = 10_000) -> None:
        self._graph_consumer: GraphConsumer | None = None
        self._llm_consumer: LLMConsumer | None = None
        self._ledger_consumer: LedgerConsumer | None = None
        self._circuit_breaker_consumer: CircuitBreakerConsumer | None = None
        self._agent_consumer: AgentConsumer | None = None
        self._queue: asyncio.Queue[AlertPayload] = asyncio.Queue(maxsize=max_queue)
        self.metrics = RouterMetrics()
        self._worker_task: asyncio.Task | None = None
        self._agent_tasks: set[asyncio.Task] = set()  # fire-and-forget agent investigations
        self._agent_semaphore = asyncio.Semaphore(2)  # limit concurrent LLM investigations

    # ── Consumer Registration ─────────────────────────────────────────────

    def register_graph_consumer(self, consumer: GraphConsumer) -> None:
        """Register the Graph Analytics investigation callback."""
        self._graph_consumer = consumer
        logger.info("Graph consumer registered.")

    def register_llm_consumer(self, consumer: LLMConsumer) -> None:
        """Register the LLM Orchestrator investigation callback."""
        self._llm_consumer = consumer
        logger.info("LLM consumer registered.")

    def register_ledger_consumer(self, consumer: LedgerConsumer) -> None:
        """Register the audit ledger anchoring callback."""
        self._ledger_consumer = consumer
        logger.info("Ledger consumer registered.")

    def register_circuit_breaker_consumer(self, consumer: CircuitBreakerConsumer) -> None:
        """Register the circuit breaker callback."""
        self._circuit_breaker_consumer = consumer
        logger.info("Circuit breaker consumer registered.")

    def register_agent_consumer(self, consumer: AgentConsumer) -> None:
        """Register the investigator agent callback."""
        self._agent_consumer = consumer
        logger.info("Agent consumer registered.")

    # ── Routing ───────────────────────────────────────────────────────────

    async def route(self, payload: AlertPayload) -> None:
        """
        Dispatch a single alert to registered consumers.

        HIGH-tier alerts go to both Graph and LLM.
        MEDIUM-tier alerts go to Graph only (LLM is expensive).
        LOW-tier alerts are logged but not routed.
        """
        if payload.tier == RiskTier.LOW:
            return

        if payload.tier == RiskTier.HIGH:
            self.metrics.high_alerts += 1
        else:
            self.metrics.medium_alerts += 1

        self.metrics.alerts_dispatched += 1

        # Broadcast risk score to frontend via SSE (best-effort)
        try:
            from src.api.events import EventBroadcaster
            top_features_list = []
            if payload.features is not None and payload.feature_names:
                indices = np.argsort(np.abs(payload.features))[::-1][:5]
                top_features_list = [payload.feature_names[idx] for idx in indices]
            await EventBroadcaster.get().publish("risk_scores", {
                "type": "alert_scored",
                "txn_id": payload.txn_id,
                "risk_score": round(payload.risk_score, 4),
                "tier": payload.tier,
                "top_features": top_features_list,
            })
        except Exception:
            pass

        # Dispatch to both modules concurrently
        tasks = []

        if self._graph_consumer is not None:
            tasks.append(self._deliver_graph(payload))

        if payload.tier == RiskTier.HIGH and self._llm_consumer is not None:
            tasks.append(self._deliver_llm(payload))

        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

        # Anchor to audit ledger (fire-and-forget, never blocks routing)
        if self._ledger_consumer is not None:
            try:
                await self._ledger_consumer(payload)
            except Exception as exc:
                logger.warning("Ledger anchoring failed for %s: %s", payload.txn_id, exc)

        # Feed circuit breaker (fire-and-forget, never blocks routing)
        if self._circuit_breaker_consumer is not None:
            try:
                await self._circuit_breaker_consumer(payload)
            except Exception as exc:
                logger.warning("Circuit breaker feed failed for %s: %s", payload.txn_id, exc)

        # Feed investigator agent for HIGH-tier alerts (true fire-and-forget)
        if payload.tier == RiskTier.HIGH and self._agent_consumer is not None:
            try:
                task = asyncio.create_task(
                    self._throttled_agent(payload),
                    name=f"agent-{payload.txn_id}",
                )
                self._agent_tasks.add(task)
                task.add_done_callback(self._agent_tasks.discard)
            except Exception as exc:
                logger.warning("Agent feed failed for %s: %s", payload.txn_id, exc)

    async def route_batch(self, payloads: list[AlertPayload]) -> None:
        """Route multiple alerts. Filters LOW-tier before dispatching."""
        for p in payloads:
            await self.route(p)

    # ── Delivery with Retry ───────────────────────────────────────────────

    async def _throttled_agent(self, payload: AlertPayload) -> None:
        """Run agent investigation with concurrency limit to avoid GPU overload."""
        async with self._agent_semaphore:
            try:
                await self._agent_consumer(payload)
            except Exception as exc:
                logger.warning("Agent investigation failed for %s: %s", payload.txn_id, exc)

    async def _deliver_graph(self, payload: AlertPayload, retries: int = 2) -> None:
        """Deliver alert to Graph Analytics with bounded retry."""
        for attempt in range(retries):
            try:
                await self._graph_consumer(payload)
                self.metrics.graph_deliveries += 1
                return
            except Exception as exc:
                logger.warning(
                    "Graph delivery attempt %d/%d failed for %s: %s",
                    attempt + 1, retries, payload.txn_id, exc,
                )
        self.metrics.graph_failures += 1
        logger.error("Graph delivery exhausted retries for %s", payload.txn_id)

    async def _deliver_llm(self, payload: AlertPayload, retries: int = 2) -> None:
        """Deliver alert to LLM Orchestrator with bounded retry."""
        for attempt in range(retries):
            try:
                await self._llm_consumer(payload)
                self.metrics.llm_deliveries += 1
                return
            except Exception as exc:
                logger.warning(
                    "LLM delivery attempt %d/%d failed for %s: %s",
                    attempt + 1, retries, payload.txn_id, exc,
                )
        self.metrics.llm_failures += 1
        logger.error("LLM delivery exhausted retries for %s", payload.txn_id)

    # ── Background Worker (Queue-based) ───────────────────────────────────

    def start_worker(self) -> None:
        """Start a background asyncio task that drains the alert queue."""
        if self._worker_task is not None:
            return
        self._worker_task = asyncio.ensure_future(self._worker_loop())
        logger.info("Alert router worker started.")

    async def enqueue(self, payload: AlertPayload) -> None:
        """
        Non-blocking enqueue. Drops oldest on backpressure.
        Use with start_worker() for decoupled production/consumption.
        """
        if self._queue.full():
            # Drop oldest to prevent unbounded growth
            try:
                self._queue.get_nowait()
                self.metrics.dropped_backpressure += 1
            except asyncio.QueueEmpty:
                pass
        await self._queue.put(payload)

    async def _worker_loop(self) -> None:
        """Drain queue and route alerts."""
        while True:
            try:
                payload = await self._queue.get()
                await self.route(payload)
                self._queue.task_done()
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.error("Worker loop error: %s", exc)

    async def shutdown(self) -> None:
        """Graceful shutdown: drain queue then cancel worker."""
        if self._worker_task is not None:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass
            self._worker_task = None
            logger.info("Alert router worker stopped.")

    # ── Factory: Build Payloads from Inference Results ────────────────────

    @staticmethod
    def build_payloads(
        threshold_results: list[ThresholdResult],
        txn_ids: list[str],
        sender_ids: list[str],
        receiver_ids: list[str],
        timestamps: np.ndarray,
        features: np.ndarray,
        feature_names: list[str],
    ) -> list[AlertPayload]:
        """
        Build AlertPayload objects from threshold evaluation results.

        Filters to only HIGH and MEDIUM tiers (LOW are ignored).
        """
        payloads = []
        for i, tr in enumerate(threshold_results):
            if tr.tier == RiskTier.LOW:
                continue
            payloads.append(AlertPayload(
                txn_id=txn_ids[i],
                sender_id=sender_ids[i],
                receiver_id=receiver_ids[i],
                timestamp=int(timestamps[i]),
                risk_score=tr.risk_score,
                tier=tr.tier,
                threshold_at_eval=tr.threshold,
                features=features[i],
                feature_names=feature_names,
            ))
        return payloads

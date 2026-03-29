"""
PayFlow -- Human-in-the-Loop (HITL) Escalation Framework
==========================================================
Implements confidence-threshold-based escalation of fraud investigations
to human analysts when the autonomous agent cannot reach sufficient
confidence on complex money laundering typologies.

Components:
    - ``ConfidenceEvaluator``:  Per-typology threshold checks
    - ``GraphContextPackager``: k-hop subgraph extraction & serialization
    - ``HITLEscalationPayload``: Immutable escalation record
    - ``HITLDispatcher``:       Async HTTP dispatch to analyst API
    - ``HITLMetrics``:          Escalation tracking counters

Integration::

    evaluator = ConfidenceEvaluator()
    if evaluator.should_escalate(confidence=0.65, typology="LAYERING"):
        ctx = packager.package(graph, node_id, k_hops=3)
        payload = HITLEscalationPayload(...)
        result = await dispatcher.dispatch(payload)

Dependencies:
    - httpx (async HTTP client, already in deps)
    - config.settings.HITLConfig
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


# ── Confidence Evaluator ─────────────────────────────────────────────────────

class ConfidenceEvaluator:
    """
    Determines whether an investigation should be escalated to a human
    analyst based on the agent's confidence and the detected fraud typology.

    Complex typologies (LAYERING, ROUND_TRIPPING, STRUCTURING) require
    higher confidence for autonomous action. If confidence falls below the
    applicable threshold, escalation is recommended.
    """

    @staticmethod
    def get_threshold(typology: str | None, config) -> float:
        """
        Return the confidence threshold for a given typology.

        Falls back to the default threshold if typology is None or unknown.
        """
        if typology is None:
            return config.default_confidence_threshold
        return config.typology_thresholds.get(
            typology.upper(), config.default_confidence_threshold,
        )

    @staticmethod
    def should_escalate(
        confidence: float,
        typology: str | None,
        config,
    ) -> bool:
        """
        Check whether the investigation should be escalated to HITL.

        Returns True if the agent's confidence is strictly below the
        threshold for the detected typology.
        """
        threshold = ConfidenceEvaluator.get_threshold(typology, config)
        return confidence < threshold


# ── Graph Context Packager ───────────────────────────────────────────────────

class GraphContextPackager:
    """
    Extracts and serializes k-hop subgraph context from the TransactionGraph
    for inclusion in HITL escalation payloads.

    Handles missing graphs and unknown nodes gracefully, returning empty
    context dictionaries rather than raising exceptions.
    """

    @staticmethod
    def package(
        transaction_graph,
        node_id: str,
        k_hops: int = 3,
    ) -> dict:
        """
        Package graph context around a node for human analyst review.

        Args:
            transaction_graph: TransactionGraph instance (or None).
            node_id: Account node to extract context for.
            k_hops: Neighborhood extraction radius.

        Returns:
            Dict with subgraph summary, node attributes, connections,
            and pattern findings. Empty dict if graph is unavailable.
        """
        if transaction_graph is None:
            return {
                "available": False,
                "reason": "TransactionGraph not available",
            }

        g = transaction_graph.graph
        if not g.has_node(node_id):
            return {
                "available": False,
                "reason": f"Node {node_id} not found in graph",
                "node_id": node_id,
            }

        try:
            subgraph = transaction_graph._extract_k_hop_subgraph(
                [node_id], min(k_hops, 5),
            )
            node_data = g.nodes.get(node_id, {})

            context: dict = {
                "available": True,
                "node_id": node_id,
                "node_attributes": {
                    "first_seen": node_data.get("first_seen", 0),
                    "last_seen": node_data.get("last_seen", 0),
                    "txn_count": node_data.get("txn_count", 0),
                },
                "subgraph": {
                    "nodes": subgraph.number_of_nodes(),
                    "edges": subgraph.number_of_edges(),
                    "k_hops": k_hops,
                },
                "connections": {
                    "in_degree": g.in_degree(node_id),
                    "out_degree": g.out_degree(node_id),
                    "distinct_senders": len(
                        set(u for u, _ in g.in_edges(node_id))
                    ),
                    "distinct_receivers": len(
                        set(v for _, v in g.out_edges(node_id))
                    ),
                },
                "patterns": {},
            }

            # Mule detection
            try:
                mule_finding = (
                    transaction_graph._mule_detector.detect_around_node(
                        subgraph, node_id,
                        transaction_graph._latest_timestamp,
                    )
                )
                context["patterns"]["mule_network_detected"] = (
                    mule_finding is not None
                )
                context["patterns"]["mule_distinct_senders"] = (
                    mule_finding.distinct_senders if mule_finding else 0
                )
            except Exception:
                context["patterns"]["mule_network_detected"] = False

            # Cycle detection
            try:
                cycles = (
                    transaction_graph._cycle_detector.detect_around_nodes(
                        subgraph, [node_id],
                        transaction_graph._latest_timestamp,
                        k_hops=k_hops,
                    )
                )
                context["patterns"]["cycles_found"] = len(cycles)
                context["patterns"]["cycle_lengths"] = [
                    c.length for c in cycles[:5]
                ]
            except Exception:
                context["patterns"]["cycles_found"] = 0

            return context

        except Exception as exc:
            logger.warning(
                "Graph context packaging failed for %s: %s", node_id, exc,
            )
            return {
                "available": False,
                "reason": f"Packaging error: {exc}",
                "node_id": node_id,
            }


# ── HITL Escalation Payload ──────────────────────────────────────────────────

@dataclass(frozen=True)
class HITLEscalationPayload:
    """
    Immutable escalation record sent to the human analyst API.

    Contains everything the analyst needs to make an informed decision:
    the agent's reasoning trace, all accumulated evidence, graph context,
    ML/GNN scores, and the confidence threshold that triggered escalation.
    """
    escalation_id: str
    txn_id: str
    node_id: str
    agent_confidence: float
    confidence_threshold: float
    detected_typology: str | None
    reasoning_trace: list[str]
    evidence_collected: dict
    graph_context: dict
    ml_score: float
    gnn_score: float
    nlu_findings: dict | None
    recommended_action: str  # always "ESCALATE_TO_HUMAN"
    escalated_at: float

    def to_dict(self) -> dict:
        return {
            "escalation_id": self.escalation_id,
            "txn_id": self.txn_id,
            "node_id": self.node_id,
            "agent_confidence": round(self.agent_confidence, 4),
            "confidence_threshold": round(self.confidence_threshold, 4),
            "detected_typology": self.detected_typology,
            "reasoning_trace": list(self.reasoning_trace),
            "evidence_collected": dict(self.evidence_collected),
            "graph_context": dict(self.graph_context),
            "ml_score": round(self.ml_score, 4),
            "gnn_score": round(self.gnn_score, 4),
            "nlu_findings": (
                dict(self.nlu_findings) if self.nlu_findings else None
            ),
            "recommended_action": self.recommended_action,
            "escalated_at": self.escalated_at,
        }


# ── HITL Dispatch Result ─────────────────────────────────────────────────────

@dataclass(frozen=True)
class HITLDispatchResult:
    """Result of dispatching an escalation to the human analyst API."""
    success: bool
    status_code: int
    analyst_ack_id: str | None = None
    error: str | None = None


# ── HITL Dispatcher ──────────────────────────────────────────────────────────

class HITLDispatcher:
    """
    Async HTTP dispatcher for HITL escalation payloads.

    Sends escalation records to the configured human analyst API endpoint
    with bounded retries and timeout handling.
    """

    def __init__(self, config=None) -> None:
        from config.settings import HITL_CFG
        self._cfg = config or HITL_CFG
        self._pending: dict[str, HITLEscalationPayload] = {}

    async def dispatch(
        self, payload: HITLEscalationPayload,
    ) -> HITLDispatchResult:
        """
        POST the escalation payload to the human analyst API.

        Retries on 5xx responses up to ``max_retries``. Returns a
        ``HITLDispatchResult`` with success/failure details.
        """
        import httpx

        # Track pending escalation
        if len(self._pending) >= self._cfg.max_pending_escalations:
            # Evict oldest
            oldest_key = next(iter(self._pending))
            del self._pending[oldest_key]
        self._pending[payload.escalation_id] = payload

        last_error: str | None = None
        last_status: int = 0

        for attempt in range(self._cfg.analyst_max_retries):
            try:
                async with httpx.AsyncClient(
                    timeout=self._cfg.analyst_timeout_seconds,
                ) as client:
                    response = await client.post(
                        self._cfg.analyst_endpoint_url,
                        json=payload.to_dict(),
                    )

                last_status = response.status_code

                if 200 <= response.status_code < 300:
                    body = response.json()
                    ack_id = body.get("ack_id")
                    # Remove from pending on success
                    self._pending.pop(payload.escalation_id, None)
                    logger.info(
                        "HITL escalation dispatched: %s → ack=%s",
                        payload.escalation_id, ack_id,
                    )
                    return HITLDispatchResult(
                        success=True,
                        status_code=response.status_code,
                        analyst_ack_id=ack_id,
                    )

                if response.status_code >= 500:
                    last_error = (
                        f"Server error {response.status_code}"
                    )
                    logger.warning(
                        "HITL dispatch attempt %d/%d failed: %s",
                        attempt + 1, self._cfg.analyst_max_retries,
                        last_error,
                    )
                    continue

                # 4xx — client error, don't retry
                last_error = (
                    f"Client error {response.status_code}: "
                    f"{response.text[:200]}"
                )
                break

            except httpx.TimeoutException:
                last_error = "Request timed out"
                logger.warning(
                    "HITL dispatch attempt %d/%d timed out",
                    attempt + 1, self._cfg.analyst_max_retries,
                )
            except Exception as exc:
                last_error = str(exc)
                logger.warning(
                    "HITL dispatch attempt %d/%d error: %s",
                    attempt + 1, self._cfg.analyst_max_retries, exc,
                )

        logger.error(
            "HITL escalation failed after %d attempts for %s: %s",
            self._cfg.analyst_max_retries,
            payload.escalation_id,
            last_error,
        )
        return HITLDispatchResult(
            success=False,
            status_code=last_status,
            error=last_error,
        )

    @property
    def pending_count(self) -> int:
        return len(self._pending)


# ── HITL Metrics ─────────────────────────────────────────────────────────────

@dataclass
class HITLMetrics:
    """Tracks HITL escalation statistics."""
    escalations_triggered: int = 0
    escalations_dispatched: int = 0
    escalations_failed: int = 0
    escalations_recovered: int = 0  # second-chance gate returned to think
    escalations_by_typology: dict = field(default_factory=dict)
    total_escalation_latency_ms: float = 0.0
    _start_time: float = field(default_factory=time.monotonic)

    def record_escalation(
        self,
        typology: str | None,
        dispatched: bool,
        latency_ms: float,
    ) -> None:
        """Record an escalation event."""
        self.escalations_triggered += 1
        if dispatched:
            self.escalations_dispatched += 1
        else:
            self.escalations_failed += 1
        self.total_escalation_latency_ms += latency_ms
        key = typology or "UNKNOWN"
        self.escalations_by_typology[key] = (
            self.escalations_by_typology.get(key, 0) + 1
        )

    def snapshot(self) -> dict:
        return {
            "triggered": self.escalations_triggered,
            "dispatched": self.escalations_dispatched,
            "failed": self.escalations_failed,
            "recovered": self.escalations_recovered,
            "by_typology": dict(self.escalations_by_typology),
            "avg_latency_ms": round(
                self.total_escalation_latency_ms
                / max(self.escalations_triggered, 1),
                2,
            ),
            "uptime_sec": round(
                time.monotonic() - self._start_time, 1,
            ),
        }


# ── Factory Helpers ──────────────────────────────────────────────────────────

def build_escalation_payload(
    state: dict,
    graph_context: dict,
    confidence: float,
    threshold: float,
    typology: str | None,
) -> HITLEscalationPayload:
    """
    Build an HITLEscalationPayload from the current agent state.

    Convenience factory used by the agent's escalate_hitl node.
    """
    alert = state.get("alert", {})
    return HITLEscalationPayload(
        escalation_id=str(uuid.uuid4()),
        txn_id=alert.get("txn_id", "unknown"),
        node_id=alert.get("sender_id", "unknown"),
        agent_confidence=confidence,
        confidence_threshold=threshold,
        detected_typology=typology,
        reasoning_trace=list(state.get("thinking_trace", [])),
        evidence_collected=dict(state.get("evidence_collected", {})),
        graph_context=graph_context,
        ml_score=state.get("ml_score", 0.0),
        gnn_score=state.get("gnn_score", -1.0),
        nlu_findings=None,
        recommended_action="ESCALATE_TO_HUMAN",
        escalated_at=time.time(),
    )

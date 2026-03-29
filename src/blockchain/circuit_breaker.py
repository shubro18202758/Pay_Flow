"""
PayFlow -- Smart Contract Circuit Breaker
===========================================
Multi-model consensus engine that subscribes to both ML (AlertRouter) and
GNN (TransactionGraph) event streams's.  When mathematical consensus of high
fraud probability is reached across models, the circuit breaker autonomously
freezes suspicious nodes and halts their transaction flow.

Integration::

    breaker = CircuitBreaker(audit_ledger=ledger)
    router.register_circuit_breaker_consumer(breaker.on_alert)
    # TransactionGraph calls breaker.on_investigation() after investigate()

Consensus Formula::

    graph_evidence = min(1.0, mule_count * 0.5 + cycle_count * 0.5)

    If GNN available (score >= 0):
        consensus = 0.35 * ml + 0.35 * gnn + 0.30 * graph_evidence

    If GNN unavailable (score == -1.0):
        consensus = 0.55 * ml + 0.45 * graph_evidence

Freeze Flow:
    consensus >= threshold
        -> check cooldown
        -> create FreezeOrder
        -> add to frozen set
        -> generate ZKP proof of risk compliance
        -> anchor freeze + ZKP to audit ledger
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


# ── Data Structures ──────────────────────────────────────────────────────────

@dataclass(frozen=True)
class FreezeOrder:
    """Immutable record of a node freeze action."""
    node_id: str
    freeze_timestamp: float
    trigger_txn_id: str
    ml_risk_score: float
    gnn_risk_score: float
    graph_evidence_score: float
    consensus_score: float
    reason: str
    ttl_seconds: int          # auto-unfreeze after TTL (0 = manual only)

    def to_dict(self) -> dict:
        """Serialise for ledger anchoring."""
        return {
            "node_id": self.node_id,
            "freeze_timestamp": self.freeze_timestamp,
            "trigger_txn_id": self.trigger_txn_id,
            "ml_risk_score": round(self.ml_risk_score, 4),
            "gnn_risk_score": round(self.gnn_risk_score, 4),
            "graph_evidence_score": round(self.graph_evidence_score, 4),
            "consensus_score": round(self.consensus_score, 4),
            "reason": self.reason,
            "ttl_seconds": self.ttl_seconds,
        }


# ── Metrics ──────────────────────────────────────────────────────────────────

@dataclass
class CircuitBreakerMetrics:
    """Runtime performance counters for the circuit breaker."""
    alerts_evaluated: int = 0
    investigations_evaluated: int = 0
    consensus_triggers: int = 0
    nodes_frozen: int = 0
    nodes_unfrozen: int = 0
    _start_time: float = field(default_factory=time.monotonic)

    @property
    def uptime_sec(self) -> float:
        return time.monotonic() - self._start_time

    def snapshot(self) -> dict:
        return {
            "alerts_evaluated": self.alerts_evaluated,
            "investigations_evaluated": self.investigations_evaluated,
            "consensus_triggers": self.consensus_triggers,
            "nodes_frozen": self.nodes_frozen,
            "nodes_unfrozen": self.nodes_unfrozen,
            "uptime_sec": round(self.uptime_sec, 1),
        }


# ── Circuit Breaker ──────────────────────────────────────────────────────────

class CircuitBreaker:
    """
    Smart contract circuit breaker — multi-model consensus freeze.

    Subscribes to AlertRouter (ML scores) and TransactionGraph (GNN + graph
    structural evidence).  When the weighted consensus score exceeds the
    configured threshold, freezes the suspicious node(s) and anchors the
    action to the audit ledger with a ZKP proof of compliance.
    """

    def __init__(self, config=None, audit_ledger=None) -> None:
        from config.settings import CIRCUIT_BREAKER_CFG
        self._cfg = config or CIRCUIT_BREAKER_CFG

        self._audit_ledger = audit_ledger
        self._frozen_nodes: dict[str, FreezeOrder] = {}
        self._pending_alerts: dict[str, object] = {}   # txn_id -> AlertPayload
        self._cooldowns: dict[str, float] = {}          # node_id -> last trigger ts
        self._agent_listener = None                     # AgentCircuitBreakerListener
        self._graph = None                              # TransactionGraph for 1-hop freeze
        self._llm_agent = None                          # InvestigatorAgent for auto-trigger
        self._regulatory_reporter = None                # RegulatoryReporter for STR queue
        self._lock = asyncio.Lock()
        self.metrics = CircuitBreakerMetrics()

    # ── Agent Listener Registration ────────────────────────────────────

    def register_agent_listener(self, listener) -> None:
        """
        Register an AgentCircuitBreakerListener for device ban and
        routing pause checks within ``check_transaction()``.
        """
        self._agent_listener = listener

    def attach_graph(self, graph) -> None:
        """Attach the TransactionGraph for 1-hop neighbor freeze."""
        self._graph = graph

    def attach_llm_agent(self, agent) -> None:
        """Attach the LLM InvestigatorAgent for auto-trigger on OPEN."""
        self._llm_agent = agent

    def attach_regulatory_reporter(self, reporter) -> None:
        """Attach the RegulatoryReporter for automatic STR queuing."""
        self._regulatory_reporter = reporter

    # ── Consumer Protocol ─────────────────────────────────────────────────

    async def on_alert(self, payload) -> None:
        """
        AlertRouter consumer — receives ML risk scores.

        If ``require_investigation`` is True, stashes the alert and waits
        for the matching InvestigationResult from the graph module before
        evaluating consensus.  Otherwise, evaluates immediately with
        gnn_risk_score=-1.0.
        """
        async with self._lock:
            self.metrics.alerts_evaluated += 1

            if self._cfg.require_investigation:
                # Stash and wait for matching investigation
                self._pending_alerts[payload.txn_id] = payload
                logger.debug(
                    "Circuit breaker: alert %s stashed, waiting for investigation",
                    payload.txn_id,
                )
            else:
                # Evaluate immediately without GNN
                await self._evaluate_and_freeze(
                    txn_id=payload.txn_id,
                    alert=payload,
                    ml_score=payload.risk_score,
                    gnn_score=-1.0,
                    mule_count=0,
                    cycle_count=0,
                )

    async def on_investigation(self, result) -> None:
        """
        TransactionGraph consumer — receives GNN + graph structural evidence.

        Matches with a pending alert by ``txn_id`` and runs the consensus
        evaluation.
        """
        async with self._lock:
            self.metrics.investigations_evaluated += 1

            alert = self._pending_alerts.pop(result.txn_id, None)
            if alert is None:
                logger.debug(
                    "Circuit breaker: investigation %s has no pending alert",
                    result.txn_id,
                )
                return

            await self._evaluate_and_freeze(
                txn_id=result.txn_id,
                alert=alert,
                ml_score=alert.risk_score,
                gnn_score=result.gnn_risk_score,
                mule_count=len(result.mule_findings),
                cycle_count=len(result.cycle_findings),
            )

    # ── Consensus Engine ──────────────────────────────────────────────────

    def _compute_consensus(
        self,
        ml_score: float,
        gnn_score: float,
        mule_count: int,
        cycle_count: int,
    ) -> float:
        """
        Weighted consensus score combining ML, GNN, and graph structural evidence.

        Returns a float in [0.0, 1.0].
        """
        graph_evidence = min(1.0, mule_count * 0.5 + cycle_count * 0.5)

        if gnn_score >= 0.0:
            # Full consensus with all three signals
            return (
                self._cfg.ml_weight * ml_score
                + self._cfg.gnn_weight * gnn_score
                + self._cfg.graph_evidence_weight * graph_evidence
            )
        else:
            # GNN unavailable — redistribute weight to ML + graph evidence
            total_non_gnn = self._cfg.ml_weight + self._cfg.graph_evidence_weight
            if total_non_gnn <= 0:
                return 0.0
            ml_adj = self._cfg.ml_weight / total_non_gnn
            ge_adj = self._cfg.graph_evidence_weight / total_non_gnn
            return ml_adj * ml_score + ge_adj * graph_evidence

    async def _evaluate_and_freeze(
        self,
        txn_id: str,
        alert,
        ml_score: float,
        gnn_score: float,
        mule_count: int,
        cycle_count: int,
    ) -> None:
        """Check consensus, apply freeze if threshold breached."""
        consensus = self._compute_consensus(ml_score, gnn_score, mule_count, cycle_count)
        graph_evidence = min(1.0, mule_count * 0.5 + cycle_count * 0.5)

        # Broadcast evaluation result to frontend (per-txn stage regardless of freeze)
        try:
            from src.api.events import EventBroadcaster
            EventBroadcaster.get().publish_sync("pipeline", {
                "type": "stage_complete",
                "stage": "cb_evaluated",
                "txn_id": txn_id,
                "consensus_score": round(consensus, 4),
                "threshold": self._cfg.consensus_threshold,
                "ml_score": round(ml_score, 4),
                "gnn_score": round(gnn_score, 4),
                "graph_evidence": round(graph_evidence, 4),
                "freeze_triggered": consensus >= self._cfg.consensus_threshold,
            })
        except Exception:
            pass

        if consensus < self._cfg.consensus_threshold:
            return

        self.metrics.consensus_triggers += 1
        now = time.time()

        # Freeze both sender and receiver
        for node_id in [alert.sender_id, alert.receiver_id]:
            # Check cooldown
            last_trigger = self._cooldowns.get(node_id, 0.0)
            if now - last_trigger < self._cfg.cooldown_seconds:
                logger.debug(
                    "Circuit breaker: node %s in cooldown (%.0f sec remaining)",
                    node_id, self._cfg.cooldown_seconds - (now - last_trigger),
                )
                continue

            # Check capacity
            if len(self._frozen_nodes) >= self._cfg.max_frozen_nodes:
                logger.warning(
                    "Circuit breaker: max frozen nodes (%d) reached, skipping %s",
                    self._cfg.max_frozen_nodes, node_id,
                )
                continue

            # Skip if already frozen
            if node_id in self._frozen_nodes:
                continue

            order = FreezeOrder(
                node_id=node_id,
                freeze_timestamp=now,
                trigger_txn_id=txn_id,
                ml_risk_score=ml_score,
                gnn_risk_score=gnn_score,
                graph_evidence_score=graph_evidence,
                consensus_score=consensus,
                reason=(
                    f"Multi-model consensus {consensus:.4f} >= "
                    f"{self._cfg.consensus_threshold} "
                    f"(ML={ml_score:.4f}, GNN={gnn_score:.4f}, "
                    f"GE={graph_evidence:.4f})"
                ),
                ttl_seconds=self._cfg.freeze_ttl_seconds,
            )

            await self.freeze_node(order)

        # ── Action 3: 1-hop neighbor freeze ────────────────────────────────
        frozen_primary = [
            nid for nid in [alert.sender_id, alert.receiver_id]
            if nid in self._frozen_nodes
        ]
        if self._graph is not None and frozen_primary:
            for primary in frozen_primary:
                await self._freeze_1hop_neighbors(primary, txn_id, ml_score, gnn_score, graph_evidence, consensus)

        # ── Action 4: Team notification broadcast ─────────────────────────
        try:
            from src.api.events import EventBroadcaster
            EventBroadcaster.get().publish_sync("circuit_breaker", {
                "type": "team_alert",
                "txn_id": txn_id,
                "consensus_score": round(consensus, 4),
                "frozen_nodes": frozen_primary,
                "reason": f"Consensus {consensus:.4f} exceeded threshold",
            })
        except Exception:
            pass

        # ── Action 5: LLM agent auto-investigation ───────────────────────
        if self._llm_agent is not None and frozen_primary:
            try:
                asyncio.create_task(self._llm_agent.investigate(
                    case_id=txn_id,
                    account_ids=frozen_primary,
                    trigger="circuit_breaker_open",
                ))
            except Exception as exc:
                logger.warning("LLM agent auto-trigger failed: %s", exc)

        # ── Action 6: STR auto-queue for FIU-IND ─────────────────────────
        if self._regulatory_reporter is not None and frozen_primary:
            try:
                for nid in frozen_primary:
                    await self._regulatory_reporter.queue_str(
                        account_id=nid,
                        txn_id=txn_id,
                        reason=f"Circuit breaker consensus {consensus:.4f}",
                    )
            except Exception as exc:
                logger.warning("STR auto-queue failed: %s", exc)

    async def _freeze_1hop_neighbors(
        self,
        primary_node: str,
        txn_id: str,
        ml_score: float,
        gnn_score: float,
        graph_evidence: float,
        consensus: float,
    ) -> None:
        """Freeze all 1-hop neighbors of a frozen primary node."""
        try:
            neighbors = list(self._graph.graph.neighbors(primary_node))
        except Exception:
            neighbors = []

        now = time.time()
        for neighbor_id in neighbors:
            if neighbor_id in self._frozen_nodes:
                continue
            if len(self._frozen_nodes) >= self._cfg.max_frozen_nodes:
                break
            order = FreezeOrder(
                node_id=neighbor_id,
                freeze_timestamp=now,
                trigger_txn_id=txn_id,
                ml_risk_score=ml_score,
                gnn_risk_score=gnn_score,
                graph_evidence_score=graph_evidence,
                consensus_score=consensus,
                reason=f"1-hop neighbor of frozen node {primary_node}",
                ttl_seconds=self._cfg.freeze_ttl_seconds,
            )
            await self.freeze_node(order)
            logger.info(
                "Circuit breaker: 1-hop freeze %s (neighbor of %s)",
                neighbor_id, primary_node,
            )

    # ── Freeze Management ─────────────────────────────────────────────────

    async def freeze_node(self, order: FreezeOrder) -> None:
        """
        Add a node to the frozen set and anchor the action to the audit ledger.

        Also generates a ZKP proof attesting that the node's risk exceeds
        the consensus threshold, without revealing the exact scores to the
        ledger.
        """
        self._frozen_nodes[order.node_id] = order
        self._cooldowns[order.node_id] = order.freeze_timestamp
        self.metrics.nodes_frozen += 1

        logger.warning(
            "CIRCUIT BREAKER TRIGGERED: node %s frozen | txn=%s | consensus=%.4f | %s",
            order.node_id, order.trigger_txn_id, order.consensus_score, order.reason,
        )

        # Anchor freeze event to audit ledger
        if self._audit_ledger is not None:
            try:
                from src.blockchain.models import EventType

                # Anchor the freeze action
                await self._audit_ledger.anchor_circuit_breaker(
                    action="freeze_node",
                    details=order.to_dict(),
                )

                # Generate and anchor ZKP proof
                from src.blockchain.zkp import ZKPProver

                commitment = ZKPProver.commit(order.consensus_score, "consensus_score")
                proof = ZKPProver.prove_threshold(
                    commitment=commitment,
                    actual_value=order.consensus_score,
                    threshold=self._cfg.consensus_threshold,
                    predicate=f"consensus_score >= {self._cfg.consensus_threshold}",
                    node_id=order.node_id,
                )
                await self._audit_ledger.anchor_zkp_proof(proof)

            except Exception as exc:
                logger.warning(
                    "Ledger anchoring failed for circuit breaker freeze %s: %s",
                    order.node_id, exc,
                )

        # Broadcast freeze event to dashboard (best-effort)
        try:
            from src.api.events import EventBroadcaster
            EventBroadcaster.get().publish_sync("circuit_breaker", {
                "type": "node_frozen",
                "node_id": order.node_id,
                "order": order.to_dict(),
            })
        except Exception:
            pass

    async def unfreeze_node(self, node_id: str, reason: str = "manual") -> None:
        """Remove a node from the frozen set and anchor the action."""
        order = self._frozen_nodes.pop(node_id, None)
        if order is None:
            return

        self.metrics.nodes_unfrozen += 1
        logger.info(
            "Circuit breaker: node %s unfrozen (reason: %s)", node_id, reason,
        )

        if self._audit_ledger is not None:
            try:
                await self._audit_ledger.anchor_circuit_breaker(
                    action="unfreeze_node",
                    details={
                        "node_id": node_id,
                        "reason": reason,
                        "original_freeze_txn": order.trigger_txn_id,
                        "frozen_duration_sec": round(time.time() - order.freeze_timestamp, 1),
                    },
                )
            except Exception as exc:
                logger.warning(
                    "Ledger anchoring failed for unfreeze %s: %s", node_id, exc,
                )

        # Broadcast unfreeze event to dashboard (best-effort)
        try:
            from src.api.events import EventBroadcaster
            EventBroadcaster.get().publish_sync("circuit_breaker", {
                "type": "node_unfrozen",
                "node_id": node_id,
                "reason": reason,
            })
        except Exception:
            pass

    async def cleanup_expired(self) -> int:
        """
        Remove nodes whose freeze TTL has expired.

        Returns:
            Number of nodes unfrozen.
        """
        now = time.time()
        expired: list[str] = []
        for node_id, order in self._frozen_nodes.items():
            if order.ttl_seconds > 0:
                if now - order.freeze_timestamp >= order.ttl_seconds:
                    expired.append(node_id)

        for node_id in expired:
            await self.unfreeze_node(node_id, reason="ttl_expired")

        return len(expired)

    def is_frozen(self, node_id: str) -> bool:
        """Check if a node is currently frozen."""
        return node_id in self._frozen_nodes

    def get_frozen_nodes(self) -> list[FreezeOrder]:
        """Return all active freeze orders."""
        return list(self._frozen_nodes.values())

    # ── Pipeline Integration ──────────────────────────────────────────────

    def check_transaction(
        self,
        sender_id: str,
        receiver_id: str,
        device_fingerprint: str | None = None,
    ) -> bool:
        """
        Check if a transaction should be BLOCKED because the sender or
        receiver is frozen, the device fingerprint is banned, or routing
        is paused for either party.

        Called by the ingestion pipeline / validator before admitting
        a transaction into the system.

        Returns:
            True if the transaction should be BLOCKED.
        """
        # Existing freeze check
        if sender_id in self._frozen_nodes or receiver_id in self._frozen_nodes:
            return True

        # Agent listener checks (device ban + routing pause)
        if self._agent_listener is not None:
            if (
                device_fingerprint is not None
                and self._agent_listener.is_device_banned(device_fingerprint)
            ):
                return True
            if (
                self._agent_listener.is_routing_paused(sender_id)
                or self._agent_listener.is_routing_paused(receiver_id)
            ):
                return True

        return False

    # ── Diagnostics ───────────────────────────────────────────────────────

    def snapshot(self) -> dict:
        """Full circuit breaker state for monitoring dashboards."""
        return {
            "frozen_count": len(self._frozen_nodes),
            "pending_alerts": len(self._pending_alerts),
            "cooldowns_active": len(self._cooldowns),
            "metrics": self.metrics.snapshot(),
            "config": {
                "consensus_threshold": self._cfg.consensus_threshold,
                "ml_weight": self._cfg.ml_weight,
                "gnn_weight": self._cfg.gnn_weight,
                "graph_evidence_weight": self._cfg.graph_evidence_weight,
                "cooldown_seconds": self._cfg.cooldown_seconds,
                "freeze_ttl_seconds": self._cfg.freeze_ttl_seconds,
            },
        }

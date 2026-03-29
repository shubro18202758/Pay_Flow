"""
PayFlow -- Agent-Triggered Blockchain Circuit Breaker Execution Listeners
==========================================================================
Binds the InvestigatorAgent's verdict outputs to the local blockchain
circuit breaker.  When the agent issues a CRITICAL_FRAUD_DETECTED state
(FRAUDULENT + confidence >= threshold + FREEZE action), the listener
deterministically:

1. Freezes compromised nodes (sender + receiver + mule cascade)
2. Pauses transaction routing for affected nodes
3. Bans the associated device fingerprint
4. Hashes the AI's full reasoning chain (SHA-256 cryptographic proof)
5. Generates a ZKP proof of confidence >= critical threshold
6. Anchors all actions to the immutable audit ledger

Two-tier response::

    CRITICAL  (confidence >= 0.95 + FRAUDULENT + FREEZE)
        -> full freeze + device ban + routing pause + ledger anchor + ZKP

    HIGH_SUSPICION  (confidence >= 0.80 + FRAUDULENT/SUSPICIOUS)
        -> routing pause only (no formal freeze, no ledger anchor)

Integration::

    listener = AgentCircuitBreakerListener(breaker, ledger, graph)
    breaker.register_agent_listener(listener)
    agent = InvestigatorAgent(..., agent_breaker_listener=listener)
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
import uuid
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


# ── Data Structures ──────────────────────────────────────────────────────────

@dataclass(frozen=True)
class DeviceBanEntry:
    """Immutable record of a device fingerprint ban."""
    device_fingerprint: str
    banned_at: float
    trigger_txn_id: str
    ttl_seconds: int
    reason: str

    @property
    def is_expired(self) -> bool:
        """Check if this ban has exceeded its TTL."""
        return time.time() - self.banned_at >= self.ttl_seconds

    def to_dict(self) -> dict:
        return {
            "device_fingerprint": self.device_fingerprint,
            "banned_at": self.banned_at,
            "trigger_txn_id": self.trigger_txn_id,
            "ttl_seconds": self.ttl_seconds,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class AgentBreakerEvent:
    """Immutable record of a circuit breaker execution triggered by the agent."""
    event_id: str
    trigger_verdict: dict
    nodes_frozen: list[str]
    devices_banned: list[str]
    routing_paused_nodes: list[str]
    reasoning_hash: str
    zkp_proof_id: str | None
    latency_ms: float
    timestamp: float

    def to_dict(self) -> dict:
        return {
            "event_id": self.event_id,
            "trigger_verdict": self.trigger_verdict,
            "nodes_frozen": self.nodes_frozen,
            "devices_banned": self.devices_banned,
            "routing_paused_nodes": self.routing_paused_nodes,
            "reasoning_hash": self.reasoning_hash,
            "zkp_proof_id": self.zkp_proof_id,
            "latency_ms": round(self.latency_ms, 3),
            "timestamp": self.timestamp,
        }


# ── Metrics ──────────────────────────────────────────────────────────────────

@dataclass
class AgentBreakerMetrics:
    """Runtime performance counters for agent-triggered circuit breaking."""
    critical_events: int = 0
    high_suspicion_events: int = 0
    nodes_frozen_by_agent: int = 0
    devices_banned: int = 0
    routing_pauses: int = 0
    total_latency_ms: float = 0.0
    _start_time: float = field(default_factory=time.monotonic)

    @property
    def avg_latency_ms(self) -> float:
        total = self.critical_events + self.high_suspicion_events
        if total == 0:
            return 0.0
        return self.total_latency_ms / total

    @property
    def uptime_sec(self) -> float:
        return time.monotonic() - self._start_time

    def snapshot(self) -> dict:
        return {
            "critical_events": self.critical_events,
            "high_suspicion_events": self.high_suspicion_events,
            "nodes_frozen_by_agent": self.nodes_frozen_by_agent,
            "devices_banned": self.devices_banned,
            "routing_pauses": self.routing_pauses,
            "avg_latency_ms": round(self.avg_latency_ms, 3),
            "uptime_sec": round(self.uptime_sec, 1),
        }


# ── Reasoning Chain Hashing ──────────────────────────────────────────────────

def hash_reasoning_chain(verdict) -> str:
    """
    Compute a SHA-256 cryptographic proof of the AI agent's reasoning.

    Builds a canonical JSON representation of the verdict's key fields and
    returns a deterministic 64-character hex digest.  This hash is anchored
    to the immutable audit ledger so that regulators can verify the AI's
    decision-making process has not been tampered with post-hoc.

    Args:
        verdict: VerdictPayload instance (or any object with the required fields).

    Returns:
        64-character lowercase hex SHA-256 digest.
    """
    canonical = {
        "verdict": getattr(verdict, "verdict", ""),
        "confidence": round(getattr(verdict, "confidence", 0.0), 6),
        "fraud_typology": getattr(verdict, "fraud_typology", None),
        "reasoning_summary": getattr(verdict, "reasoning_summary", ""),
        "evidence_cited": list(getattr(verdict, "evidence_cited", [])),
        "tools_used": sorted(getattr(verdict, "tools_used", [])),
    }
    payload_bytes = json.dumps(
        canonical,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(payload_bytes).hexdigest()


# ── Execution Listener ───────────────────────────────────────────────────────

class AgentCircuitBreakerListener:
    """
    Execution listener binding the InvestigatorAgent's outputs to the
    blockchain circuit breaker.

    Subscribes to agent verdicts and, when critical fraud is detected,
    executes deterministic defensive actions in milliseconds:
    node freeze, device ban, routing pause, reasoning proof, ZKP,
    and immutable ledger anchoring.
    """

    def __init__(
        self,
        circuit_breaker,
        audit_ledger=None,
        transaction_graph=None,
        config=None,
    ) -> None:
        from config.settings import AGENT_BREAKER_CFG
        self._cfg = config or AGENT_BREAKER_CFG

        self._circuit_breaker = circuit_breaker
        self._audit_ledger = audit_ledger
        self._transaction_graph = transaction_graph

        # Internal state
        self._banned_devices: dict[str, DeviceBanEntry] = {}
        self._routing_paused_nodes: dict[str, float] = {}  # node_id -> expiry ts
        self._lock = asyncio.Lock()
        self.metrics = AgentBreakerMetrics()

    # ── Public API ────────────────────────────────────────────────────────

    async def on_verdict(
        self,
        verdict,
        alert_context: dict,
    ) -> AgentBreakerEvent | None:
        """
        Core execution listener — called after every agent investigation.

        Evaluates the verdict against two tiers:

        - **CRITICAL**: FRAUDULENT + confidence >= critical_threshold + FREEZE
          → full freeze + device ban + routing pause + ZKP + ledger anchor.
        - **HIGH_SUSPICION**: FRAUDULENT/SUSPICIOUS + confidence >= high_threshold
          → routing pause only.
        - Below both thresholds → no action (returns None).

        Args:
            verdict: VerdictPayload from InvestigatorAgent.investigate().
            alert_context: Original alert dict (must contain sender_id,
                           receiver_id; optionally device_fingerprint).

        Returns:
            AgentBreakerEvent if action was taken, None otherwise.
        """
        v_type = getattr(verdict, "verdict", "")
        confidence = getattr(verdict, "confidence", 0.0)
        action = getattr(verdict, "recommended_action", "")
        typology = getattr(verdict, "fraud_typology", None)

        is_critical = (
            v_type == "FRAUDULENT"
            and confidence >= self._cfg.critical_confidence_threshold
            and action == "FREEZE"
        )
        is_high_suspicion = (
            not is_critical
            and v_type in ("FRAUDULENT", "SUSPICIOUS")
            and confidence >= self._cfg.high_suspicion_threshold
        )

        if not is_critical and not is_high_suspicion:
            return None

        t0 = time.perf_counter()

        sender_id = alert_context.get("sender_id", "unknown")
        receiver_id = alert_context.get("receiver_id", "unknown")
        txn_id = alert_context.get("txn_id", "unknown")
        device_fp = alert_context.get("device_fingerprint")

        nodes_frozen: list[str] = []
        devices_banned: list[str] = []
        routing_paused: list[str] = []
        zkp_proof_id: str | None = None

        async with self._lock:
            if is_critical:
                # ── Step 1: Freeze compromised nodes ──────────────────
                nodes_frozen = await self._freeze_compromised_nodes(
                    sender_id=sender_id,
                    receiver_id=receiver_id,
                    txn_id=txn_id,
                    verdict=verdict,
                    alert_context=alert_context,
                )

                # ── Step 2: Pause routing ─────────────────────────────
                routing_paused = self._pause_routing(nodes_frozen)

                # ── Step 3: Ban device fingerprint ────────────────────
                if device_fp:
                    self._ban_device(device_fp, txn_id, typology)
                    devices_banned = [device_fp]

                self.metrics.critical_events += 1
            else:
                # HIGH_SUSPICION: routing pause only
                routing_paused = self._pause_routing([sender_id, receiver_id])
                self.metrics.high_suspicion_events += 1

        # ── Step 4: Hash reasoning chain (no lock needed — pure function)
        reasoning_hash = hash_reasoning_chain(verdict)

        # ── Step 5: Generate ZKP proof ────────────────────────────────
        if is_critical:
            zkp_proof_id = await self._generate_and_anchor_zkp(
                confidence=confidence,
                node_id=sender_id,
            )

        # Build event record
        elapsed = (time.perf_counter() - t0) * 1000
        event = AgentBreakerEvent(
            event_id=str(uuid.uuid4()),
            trigger_verdict=self._serialize_verdict(verdict),
            nodes_frozen=nodes_frozen,
            devices_banned=devices_banned,
            routing_paused_nodes=routing_paused,
            reasoning_hash=reasoning_hash,
            zkp_proof_id=zkp_proof_id,
            latency_ms=elapsed,
            timestamp=time.time(),
        )

        # ── Step 6: Anchor to immutable ledger ────────────────────────
        if is_critical and self._audit_ledger is not None:
            try:
                await self._audit_ledger.anchor_circuit_breaker(
                    action="agent_critical_freeze",
                    details=event.to_dict(),
                )
            except Exception as exc:
                logger.warning(
                    "Ledger anchoring failed for agent breaker event %s: %s",
                    event.event_id, exc,
                )

        # Update latency metrics
        self.metrics.total_latency_ms += elapsed

        logger.warning(
            "AGENT CIRCUIT BREAKER %s: txn=%s confidence=%.4f "
            "frozen=%d banned=%d paused=%d (%.2f ms)",
            "CRITICAL" if is_critical else "HIGH_SUSPICION",
            txn_id, confidence,
            len(nodes_frozen), len(devices_banned), len(routing_paused),
            elapsed,
        )

        return event

    def is_device_banned(self, fingerprint: str) -> bool:
        """Check if a device fingerprint is currently banned (respects TTL)."""
        entry = self._banned_devices.get(fingerprint)
        if entry is None:
            return False
        if entry.is_expired:
            del self._banned_devices[fingerprint]
            return False
        return True

    def is_routing_paused(self, node_id: str) -> bool:
        """Check if a node's routing is currently paused (respects TTL)."""
        expiry = self._routing_paused_nodes.get(node_id)
        if expiry is None:
            return False
        if time.time() >= expiry:
            del self._routing_paused_nodes[node_id]
            return False
        return True

    async def cleanup_expired(self) -> int:
        """Remove expired device bans and routing pauses. Returns count removed."""
        removed = 0
        now = time.time()

        expired_devices = [
            fp for fp, entry in self._banned_devices.items()
            if entry.is_expired
        ]
        for fp in expired_devices:
            del self._banned_devices[fp]
            removed += 1

        expired_pauses = [
            nid for nid, expiry in self._routing_paused_nodes.items()
            if now >= expiry
        ]
        for nid in expired_pauses:
            del self._routing_paused_nodes[nid]
            removed += 1

        return removed

    def snapshot(self) -> dict:
        """Full listener state for monitoring dashboards."""
        return {
            "banned_devices": len(self._banned_devices),
            "routing_paused_nodes": len(self._routing_paused_nodes),
            "metrics": self.metrics.snapshot(),
            "config": {
                "critical_confidence_threshold": self._cfg.critical_confidence_threshold,
                "high_suspicion_threshold": self._cfg.high_suspicion_threshold,
                "device_ban_ttl_seconds": self._cfg.device_ban_ttl_seconds,
                "routing_pause_ttl_seconds": self._cfg.routing_pause_ttl_seconds,
                "enable_mule_cascade_freeze": self._cfg.enable_mule_cascade_freeze,
                "max_cascade_nodes": self._cfg.max_cascade_nodes,
            },
        }

    # ── Internal: Freeze Logic ────────────────────────────────────────────

    async def _freeze_compromised_nodes(
        self,
        sender_id: str,
        receiver_id: str,
        txn_id: str,
        verdict,
        alert_context: dict,
    ) -> list[str]:
        """
        Freeze sender + receiver + mule cascade nodes via the CircuitBreaker.

        Constructs a FreezeOrder for each node and delegates to the
        CircuitBreaker.freeze_node() method, which handles ZKP generation
        and ledger anchoring per-freeze.

        Returns:
            List of node IDs that were successfully frozen.
        """
        from src.blockchain.circuit_breaker import FreezeOrder

        confidence = getattr(verdict, "confidence", 0.0)
        typology = getattr(verdict, "fraud_typology", None)
        ml_score = alert_context.get("risk_score", 0.0)
        now = time.time()

        # Collect target nodes: sender + receiver
        target_nodes: list[str] = [sender_id, receiver_id]

        # Mule cascade: detect mule nodes around sender and receiver
        if (
            self._cfg.enable_mule_cascade_freeze
            and self._transaction_graph is not None
        ):
            cascade_count = 0
            for center in [sender_id, receiver_id]:
                if cascade_count >= self._cfg.max_cascade_nodes:
                    break
                try:
                    mule_ids = self._transaction_graph.detect_mule_around(center)
                    for mid in mule_ids:
                        if mid not in target_nodes and cascade_count < self._cfg.max_cascade_nodes:
                            target_nodes.append(mid)
                            cascade_count += 1
                except Exception as exc:
                    logger.warning(
                        "Mule cascade detection failed for %s: %s",
                        center, exc,
                    )

        # Freeze each target
        frozen: list[str] = []
        for node_id in target_nodes:
            if self._circuit_breaker.is_frozen(node_id):
                frozen.append(node_id)  # already frozen — still count
                continue

            order = FreezeOrder(
                node_id=node_id,
                freeze_timestamp=now,
                trigger_txn_id=txn_id,
                ml_risk_score=ml_score,
                gnn_risk_score=-1.0,
                graph_evidence_score=0.0,
                consensus_score=confidence,
                reason=(
                    f"Agent CRITICAL_FRAUD_DETECTED: "
                    f"confidence={confidence:.4f}, typology={typology}"
                ),
                ttl_seconds=self._circuit_breaker._cfg.freeze_ttl_seconds,
            )
            try:
                await self._circuit_breaker.freeze_node(order)
                frozen.append(node_id)
                self.metrics.nodes_frozen_by_agent += 1
            except Exception as exc:
                logger.warning(
                    "Failed to freeze node %s: %s", node_id, exc,
                )

        return frozen

    def _pause_routing(self, node_ids: list[str]) -> list[str]:
        """Add nodes to the routing pause set with TTL-based expiry."""
        expiry = time.time() + self._cfg.routing_pause_ttl_seconds
        paused: list[str] = []
        for nid in node_ids:
            if nid not in self._routing_paused_nodes:
                self._routing_paused_nodes[nid] = expiry
                self.metrics.routing_pauses += 1
                paused.append(nid)
            else:
                paused.append(nid)  # already paused
        return paused

    def _ban_device(
        self,
        fingerprint: str,
        txn_id: str,
        typology: str | None,
    ) -> None:
        """Add a device fingerprint to the ban list."""
        if len(self._banned_devices) >= self._cfg.max_banned_devices:
            logger.warning(
                "Device ban list full (%d), skipping ban for %s",
                self._cfg.max_banned_devices, fingerprint[:8],
            )
            return

        entry = DeviceBanEntry(
            device_fingerprint=fingerprint,
            banned_at=time.time(),
            trigger_txn_id=txn_id,
            ttl_seconds=self._cfg.device_ban_ttl_seconds,
            reason=f"Agent critical fraud: typology={typology}",
        )
        self._banned_devices[fingerprint] = entry
        self.metrics.devices_banned += 1

    # ── Internal: ZKP & Proof ─────────────────────────────────────────────

    async def _generate_and_anchor_zkp(
        self,
        confidence: float,
        node_id: str,
    ) -> str | None:
        """
        Generate a ZKP proof that agent confidence >= critical threshold
        and anchor it to the audit ledger.

        Returns:
            The proof's predicate string as an identifier, or None on failure.
        """
        try:
            from src.blockchain.zkp import ZKPProver

            commitment = ZKPProver.commit(confidence, "agent_confidence")
            proof = ZKPProver.prove_threshold(
                commitment=commitment,
                actual_value=confidence,
                threshold=self._cfg.critical_confidence_threshold,
                predicate=(
                    f"agent_confidence >= "
                    f"{self._cfg.critical_confidence_threshold}"
                ),
                node_id=node_id,
            )

            if self._audit_ledger is not None:
                await self._audit_ledger.anchor_zkp_proof(proof)

            return proof.predicate
        except Exception as exc:
            logger.warning("ZKP proof generation failed: %s", exc)
            return None

    # ── Internal: Serialization ───────────────────────────────────────────

    @staticmethod
    def _serialize_verdict(verdict) -> dict:
        """Convert VerdictPayload to a JSON-safe dict."""
        if hasattr(verdict, "to_dict"):
            return verdict.to_dict()
        return {
            "txn_id": getattr(verdict, "txn_id", ""),
            "node_id": getattr(verdict, "node_id", ""),
            "verdict": getattr(verdict, "verdict", ""),
            "confidence": round(getattr(verdict, "confidence", 0.0), 4),
            "fraud_typology": getattr(verdict, "fraud_typology", None),
            "reasoning_summary": getattr(verdict, "reasoning_summary", ""),
            "evidence_cited": list(getattr(verdict, "evidence_cited", [])),
            "recommended_action": getattr(verdict, "recommended_action", ""),
            "thinking_steps": getattr(verdict, "thinking_steps", 0),
            "tools_used": list(getattr(verdict, "tools_used", [])),
            "total_duration_ms": round(
                getattr(verdict, "total_duration_ms", 0.0), 2,
            ),
        }

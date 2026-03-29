"""
PayFlow — Real-Time Pre-Approval Transaction Gate
====================================================
Implements a synchronous fraud-check gate that evaluates transactions
BEFORE they are settled/completed.  This addresses the critical gap
in banking fraud detection: catching fraud at the point of authorization
rather than post-hoc.

Architecture::

    Customer ──→ Payment Processing ──→ PreApprovalGate.evaluate()
                                            │
                                    ┌───────┴───────┐
                                    │               │
                              APPROVE (fast)    HOLD / BLOCK
                              < 50ms target     + reason + risk

The gate runs a lightweight subset of the full ML pipeline:
  1. Velocity check (cached sliding windows — O(1))
  2. Behavioral deviation (Welford stats — O(1))
  3. Risk score composite (no GNN — too slow for pre-auth)
  4. Device fingerprint consistency
  5. Known frozen account check

If risk exceeds threshold → HOLD (transaction queued for deep review)
If risk exceeds critical  → BLOCK (transaction rejected immediately)
Otherwise                 → APPROVE
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import NamedTuple

logger = logging.getLogger(__name__)


class GateDecision(str, Enum):
    """Pre-approval gate decision."""
    APPROVE = "APPROVE"
    HOLD = "HOLD"           # queued for deeper review, not yet settled
    BLOCK = "BLOCK"         # rejected outright


class GateResult(NamedTuple):
    """Result of a pre-approval evaluation."""
    decision: GateDecision
    risk_score: float               # composite ∈ [0.0, 1.0]
    reason: str                     # human-readable explanation
    evaluation_ms: float            # latency of the gate check
    components: dict                # per-factor breakdown

    def to_dict(self) -> dict:
        return {
            "decision": self.decision.value,
            "risk_score": round(self.risk_score, 4),
            "reason": self.reason,
            "evaluation_ms": round(self.evaluation_ms, 2),
            "components": self.components,
        }


@dataclass(frozen=True)
class GateConfig:
    """Tunable thresholds for the pre-approval gate."""
    hold_threshold: float = 0.55        # ≥ this → HOLD
    block_threshold: float = 0.85       # ≥ this → BLOCK
    max_evaluation_ms: float = 100.0    # latency target


@dataclass
class GateMetrics:
    """Runtime counters for the pre-approval gate."""
    evaluations: int = 0
    approved: int = 0
    held: int = 0
    blocked: int = 0
    avg_latency_ms: float = 0.0
    _total_latency_ms: float = 0.0

    def record(self, decision: GateDecision, latency_ms: float) -> None:
        self.evaluations += 1
        self._total_latency_ms += latency_ms
        self.avg_latency_ms = self._total_latency_ms / self.evaluations
        if decision == GateDecision.APPROVE:
            self.approved += 1
        elif decision == GateDecision.HOLD:
            self.held += 1
        else:
            self.blocked += 1

    def snapshot(self) -> dict:
        return {
            "evaluations": self.evaluations,
            "approved": self.approved,
            "held": self.held,
            "blocked": self.blocked,
            "avg_latency_ms": round(self.avg_latency_ms, 2),
            "approval_rate": round(
                self.approved / max(self.evaluations, 1), 4,
            ),
        }


class PreApprovalGate:
    """
    Real-time transaction authorization gate.

    Integrates with:
      - VelocityTracker (fast sliding-window counters)
      - BehavioralAnalyzer (Welford incremental stats)
      - TransactionRiskScorer (weighted composite)
      - CircuitBreaker (frozen account lookups)
      - DeviceVerifier (fingerprint consistency)
    """

    def __init__(
        self,
        velocity_tracker=None,
        behavioral_analyzer=None,
        risk_scorer=None,
        circuit_breaker=None,
        device_verifier=None,
        fraud_registry=None,
        config: GateConfig | None = None,
    ) -> None:
        self._velocity = velocity_tracker
        self._behavioral = behavioral_analyzer
        self._risk_scorer = risk_scorer
        self._circuit_breaker = circuit_breaker
        self._device_verifier = device_verifier
        self._fraud_registry = fraud_registry
        self._cfg = config or GateConfig()
        self.metrics = GateMetrics()

    def evaluate(
        self,
        sender_id: str,
        receiver_id: str,
        amount_paisa: int,
        timestamp: int,
        sender_lat: float,
        sender_lon: float,
        device_fingerprint: str,
        channel: int = 4,  # default UPI
    ) -> GateResult:
        """
        Synchronous pre-approval check.  Target: < 50ms.

        Returns a GateResult with APPROVE / HOLD / BLOCK decision
        and the risk breakdown.
        """
        t0 = time.perf_counter()
        reasons: list[str] = []
        components: dict = {}
        risk = 0.0

        # Check 1: Frozen account (instant lookup)
        if self._circuit_breaker:
            if self._circuit_breaker.is_frozen(sender_id):
                elapsed = (time.perf_counter() - t0) * 1000
                result = GateResult(
                    decision=GateDecision.BLOCK,
                    risk_score=1.0,
                    reason="Sender account is frozen by circuit breaker",
                    evaluation_ms=elapsed,
                    components={"frozen_account": True},
                )
                self.metrics.record(GateDecision.BLOCK, elapsed)
                return result
            if self._circuit_breaker.is_frozen(receiver_id):
                elapsed = (time.perf_counter() - t0) * 1000
                result = GateResult(
                    decision=GateDecision.BLOCK,
                    risk_score=1.0,
                    reason="Receiver account is frozen by circuit breaker",
                    evaluation_ms=elapsed,
                    components={"frozen_account": True},
                )
                self.metrics.record(GateDecision.BLOCK, elapsed)
                return result

        # Check 2: Velocity features
        vel_score = 0.0
        if self._velocity:
            vel = self._velocity.record_and_extract(
                sender_id=sender_id,
                receiver_id=receiver_id,
                amount_paisa=amount_paisa,
                timestamp=timestamp,
            )
            txn_count_1h = vel.txn_count_1h
            txn_count_24h = vel.txn_count_24h
            velocity_zscore = vel.velocity_zscore
            components["txn_count_1h"] = txn_count_1h
            components["txn_count_24h"] = txn_count_24h
            components["velocity_zscore"] = round(velocity_zscore, 3)
        else:
            txn_count_1h = 0
            txn_count_24h = 0
            velocity_zscore = 0.0

        # Check 3: Behavioral deviation
        geo_distance = 0.0
        geo_deviation = 0.0
        if self._behavioral:
            beh = self._behavioral.analyze_transaction(
                sender_id=sender_id,
                amount_paisa=amount_paisa,
                timestamp=timestamp,
                sender_lat=sender_lat,
                sender_lon=sender_lon,
                device_fingerprint=device_fingerprint,
            )
            geo_distance = beh.geo_distance_km
            geo_deviation = beh.geo_deviation
            components["geo_distance_km"] = round(geo_distance, 1)
            components["amount_zscore"] = round(beh.amount_zscore, 3)
            if beh.device_change_flag:
                reasons.append("Device fingerprint changed")
                components["device_changed"] = True

        # Check 4: Device verification
        if self._device_verifier:
            dev_result = self._device_verifier.verify(
                sender_id, device_fingerprint,
            )
            components["device_trusted"] = dev_result.trusted
            if not dev_result.trusted:
                reasons.append(f"Untrusted device: {dev_result.reason}")

        # Check 5: Composite risk score
        if self._risk_scorer:
            risk_result = self._risk_scorer.score(
                amount_paisa=amount_paisa,
                txn_count_1h=txn_count_1h,
                txn_count_24h=txn_count_24h,
                velocity_zscore=velocity_zscore,
                sender_id=sender_id,
                receiver_id=receiver_id,
                geo_distance_km=geo_distance,
                geo_deviation=geo_deviation,
            )
            risk = risk_result.composite_score
            components["risk_components"] = risk_result.to_dict()["components"]
        else:
            # Fallback: simple heuristic
            risk = min(1.0, (
                0.3 * min(1.0, amount_paisa / 200_000_00)
                + 0.3 * min(1.0, txn_count_1h / 10)
                + 0.2 * min(1.0, geo_distance / 500.0)
                + 0.2 * (1.0 if velocity_zscore > 3.0 else 0.0)
            ))

        # Check 6: CFR counterparty lookup
        if self._fraud_registry:
            cfr_match = self._fraud_registry.check_account(receiver_id)
            components["cfr_match"] = cfr_match.is_match
            if cfr_match.is_match:
                components["cfr_severity"] = cfr_match.highest_severity
                components["cfr_score"] = round(cfr_match.match_score, 3)
                # CFR match boosts risk significantly (40% weight per RBI spec)
                cfr_boost = 0.40 * cfr_match.match_score
                risk = min(1.0, risk + cfr_boost)
                reasons.append(
                    f"Receiver {receiver_id} found in CFR "
                    f"(severity={cfr_match.highest_severity})"
                )
            sender_cfr = self._fraud_registry.check_account(sender_id)
            if sender_cfr.is_match:
                components["cfr_sender_match"] = True
                components["cfr_sender_severity"] = sender_cfr.highest_severity
                risk = min(1.0, risk + 0.40 * sender_cfr.match_score)
                reasons.append(
                    f"Sender {sender_id} found in CFR "
                    f"(severity={sender_cfr.highest_severity})"
                )

        # Decision
        if risk >= self._cfg.block_threshold:
            decision = GateDecision.BLOCK
            reasons.append(f"Risk {risk:.2f} exceeds block threshold")
        elif risk >= self._cfg.hold_threshold:
            decision = GateDecision.HOLD
            reasons.append(f"Risk {risk:.2f} exceeds hold threshold")
        else:
            decision = GateDecision.APPROVE

        elapsed = (time.perf_counter() - t0) * 1000
        if elapsed > self._cfg.max_evaluation_ms:
            logger.warning(
                "Pre-approval gate latency %.1f ms exceeded target %.1f ms",
                elapsed, self._cfg.max_evaluation_ms,
            )

        reason_str = "; ".join(reasons) if reasons else "Transaction approved"

        result = GateResult(
            decision=decision,
            risk_score=risk,
            reason=reason_str,
            evaluation_ms=elapsed,
            components=components,
        )
        self.metrics.record(decision, elapsed)
        return result

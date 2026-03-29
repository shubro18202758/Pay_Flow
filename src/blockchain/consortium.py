"""
PayFlow — Cross-Bank Consortium for Privacy-Preserving Fraud Intelligence
============================================================================
Enables multi-bank fraud intelligence sharing using Zero-Knowledge Proofs.
Banks can share fraud signals (high-risk accounts, mule network patterns,
suspicious transaction typologies) WITHOUT revealing customer PII or
proprietary transaction data.

Architecture:
    ┌──────────┐   ZKP Alerts    ┌──────────────────┐
    │  Bank A  │ ───────────────►│                  │
    ├──────────┤                 │  Consortium Hub  │
    │  Bank B  │ ───────────────►│  (Aggregator)    │
    ├──────────┤                 │                  │
    │  Bank C  │ ◄───────────────│  Broadcasts ZKP  │
    └──────────┘   Verified      │  verified alerts │
                   Intelligence  └──────────────────┘

Each alert contains:
  - ZKP commitment proving risk_score >= threshold (no raw score)
  - Fraud typology label (MULE, LAUNDERING, etc.)
  - Hashed account identifier (SHA-256, not reversible)
  - Timestamp and originating bank ID
  - Verification proof that the alert is authentic
"""

from __future__ import annotations

import hashlib
import logging
import secrets
import time
from dataclasses import dataclass, field
from enum import IntEnum

from src.blockchain.zkp import ZKPCommitment, ZKPProver, ZKPVerifier, ComplianceProof

logger = logging.getLogger(__name__)


# ── Enums ────────────────────────────────────────────────────────────────────

class AlertSeverity(IntEnum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


class ConsortiumFraudType(IntEnum):
    MULE_NETWORK = 1
    CIRCULAR_LAUNDERING = 2
    VELOCITY_ABUSE = 3
    CROSS_BORDER_HEIST = 4
    ACCOUNT_TAKEOVER = 5
    IDENTITY_FRAUD = 6
    INSIDER_THREAT = 7
    SYNTHETIC_IDENTITY = 8


# ── Data Structures ──────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ConsortiumAlert:
    """Privacy-preserving fraud alert shared between consortium members."""
    alert_id: str
    originating_bank: str
    account_hash: str          # SHA-256 of account_id — not reversible
    fraud_type: ConsortiumFraudType
    severity: AlertSeverity
    zkp_proof: ComplianceProof  # proves risk >= threshold
    timestamp: float
    ttl_hours: int = 72        # alert validity period

    @property
    def is_expired(self) -> bool:
        return time.time() > self.timestamp + (self.ttl_hours * 3600)

    def to_dict(self) -> dict:
        return {
            "alert_id": self.alert_id,
            "originating_bank": self.originating_bank,
            "account_hash": self.account_hash,
            "fraud_type": self.fraud_type.name,
            "severity": self.severity.name,
            "zkp_proof": self.zkp_proof.to_dict(),
            "timestamp": self.timestamp,
            "ttl_hours": self.ttl_hours,
            "expired": self.is_expired,
        }


@dataclass
class BankMember:
    """Represents a consortium member bank."""
    bank_id: str
    bank_name: str
    joined_at: float = field(default_factory=time.time)
    alerts_shared: int = 0
    alerts_received: int = 0
    trust_score: float = 1.0   # decreases on invalid proofs


# ── Consortium Hub ───────────────────────────────────────────────────────────

class ConsortiumHub:
    """
    Central aggregator for privacy-preserving cross-bank fraud intelligence.

    In production this would be a distributed ledger or MPC-based system.
    This implementation simulates the protocol for demonstration.
    """

    RISK_THRESHOLD = 0.70   # minimum risk to share via consortium
    TRUST_PENALTY = 0.05    # penalty for invalid proofs

    def __init__(self) -> None:
        self._members: dict[str, BankMember] = {}
        self._alerts: list[ConsortiumAlert] = []
        self._verified_alerts: list[ConsortiumAlert] = []
        self._rejected_count: int = 0

        # Register UBI as default member
        self.register_bank("UBI", "Union Bank of India")

    # ── Member Management ────────────────────────────────────────────────

    def register_bank(self, bank_id: str, bank_name: str) -> BankMember:
        """Register a new bank in the consortium."""
        if bank_id in self._members:
            return self._members[bank_id]
        member = BankMember(bank_id=bank_id, bank_name=bank_name)
        self._members[bank_id] = member
        logger.info("Consortium: Bank %s (%s) registered", bank_id, bank_name)
        return member

    def get_member(self, bank_id: str) -> BankMember | None:
        return self._members.get(bank_id)

    # ── Alert Publishing ─────────────────────────────────────────────────

    @staticmethod
    def _hash_account(account_id: str) -> str:
        """One-way hash of account ID for privacy."""
        return hashlib.sha256(account_id.encode("utf-8")).hexdigest()

    def publish_alert(
        self,
        bank_id: str,
        account_id: str,
        risk_score: float,
        fraud_type: ConsortiumFraudType | int,
        severity: AlertSeverity | int = AlertSeverity.HIGH,
    ) -> ConsortiumAlert | None:
        """
        Publish a privacy-preserving fraud alert to the consortium.

        Creates a ZKP commitment proving risk_score >= RISK_THRESHOLD
        without revealing the actual score.

        Returns:
            ConsortiumAlert if successful, None if risk too low or bank unknown.
        """
        member = self._members.get(bank_id)
        if not member:
            logger.warning("Unknown bank %s tried to publish", bank_id)
            return None

        if risk_score < self.RISK_THRESHOLD:
            return None  # below sharing threshold

        # Create ZKP commitment and proof
        commitment = ZKPProver.commit(risk_score, "risk_score")
        proof = ZKPProver.prove_threshold(
            commitment=commitment,
            actual_value=risk_score,
            threshold=self.RISK_THRESHOLD,
            predicate=f"risk_score >= {self.RISK_THRESHOLD}",
            node_id=self._hash_account(account_id),
        )

        if isinstance(fraud_type, int):
            fraud_type = ConsortiumFraudType(fraud_type)
        if isinstance(severity, int):
            severity = AlertSeverity(severity)

        alert = ConsortiumAlert(
            alert_id=secrets.token_hex(16),
            originating_bank=bank_id,
            account_hash=self._hash_account(account_id),
            fraud_type=fraud_type,
            severity=severity,
            zkp_proof=proof,
            timestamp=time.time(),
        )

        # Verify proof before accepting
        if not ZKPVerifier.verify(proof):
            self._rejected_count += 1
            member.trust_score = max(0.0, member.trust_score - self.TRUST_PENALTY)
            logger.warning("Consortium: Invalid proof from %s", bank_id)
            return None

        self._alerts.append(alert)
        self._verified_alerts.append(alert)
        member.alerts_shared += 1
        logger.info(
            "Consortium alert published: %s from %s [%s]",
            alert.alert_id[:12], bank_id, fraud_type.name,
        )
        return alert

    # ── Alert Querying ───────────────────────────────────────────────────

    def query_alerts(
        self,
        bank_id: str,
        fraud_type: ConsortiumFraudType | None = None,
        severity_min: AlertSeverity = AlertSeverity.LOW,
        limit: int = 50,
    ) -> list[ConsortiumAlert]:
        """
        Query consortium alerts visible to a member bank.

        Banks see alerts from OTHER banks only (not their own).
        Expired alerts are excluded.
        """
        member = self._members.get(bank_id)
        if not member:
            return []

        results = []
        for alert in reversed(self._verified_alerts):
            if alert.is_expired:
                continue
            if alert.originating_bank == bank_id:
                continue  # don't see own alerts
            if fraud_type and alert.fraud_type != fraud_type:
                continue
            if alert.severity < severity_min:
                continue
            results.append(alert)
            member.alerts_received += 1
            if len(results) >= limit:
                break
        return results

    def check_account(self, account_id: str) -> list[ConsortiumAlert]:
        """Check if an account hash appears in consortium alerts."""
        account_hash = self._hash_account(account_id)
        return [
            a for a in self._verified_alerts
            if a.account_hash == account_hash and not a.is_expired
        ]

    def get_active_alerts_count(self) -> dict[str, int]:
        """Count active alerts by fraud type."""
        counts: dict[str, int] = {}
        for alert in self._verified_alerts:
            if not alert.is_expired:
                key = alert.fraud_type.name
                counts[key] = counts.get(key, 0) + 1
        return counts

    # ── Simulated Peer Banks ─────────────────────────────────────────────

    def simulate_peer_intelligence(self, n_alerts: int = 5) -> list[ConsortiumAlert]:
        """
        Simulate incoming fraud intelligence from peer banks.

        Used for demonstration — generates realistic ZKP-backed alerts
        from simulated partner banks.
        """
        import random

        PEER_BANKS = [
            ("SBI", "State Bank of India"),
            ("BOB", "Bank of Baroda"),
            ("PNB", "Punjab National Bank"),
            ("CANARA", "Canara Bank"),
            ("IOB", "Indian Overseas Bank"),
        ]

        for bank_id, bank_name in PEER_BANKS:
            self.register_bank(bank_id, bank_name)

        published = []
        for _ in range(n_alerts):
            bank_id, _ = random.choice(PEER_BANKS)
            risk = round(random.uniform(0.72, 0.99), 4)
            fraud_type = random.choice(list(ConsortiumFraudType))
            severity = (
                AlertSeverity.CRITICAL if risk > 0.90
                else AlertSeverity.HIGH if risk > 0.80
                else AlertSeverity.MEDIUM
            )
            account_id = f"PEER_{bank_id}_{secrets.token_hex(4).upper()}"
            alert = self.publish_alert(bank_id, account_id, risk, fraud_type, severity)
            if alert:
                published.append(alert)

        return published

    # ── Snapshot ──────────────────────────────────────────────────────────

    def snapshot(self) -> dict:
        active_count = sum(1 for a in self._verified_alerts if not a.is_expired)
        return {
            "member_banks": len(self._members),
            "members": {
                bid: {
                    "name": m.bank_name,
                    "alerts_shared": m.alerts_shared,
                    "alerts_received": m.alerts_received,
                    "trust_score": round(m.trust_score, 2),
                }
                for bid, m in self._members.items()
            },
            "total_alerts": len(self._alerts),
            "active_alerts": active_count,
            "verified_alerts": len(self._verified_alerts),
            "rejected_proofs": self._rejected_count,
            "alerts_by_type": self.get_active_alerts_count(),
        }

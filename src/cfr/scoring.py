"""
PayFlow — CFR-Aware Risk Scorer
================================
Implements the RBI Central Fraud Registry risk scoring formula:

    RiskScore = 40 × (CFR Match)
              + 30 × (Large Transaction)
              + 20 × (New Account)
              + 10 × (Unusual Location)

This is a *separate* scorer from TransactionRiskScorer (which uses
30 / 30 / 20 / 20 weights without a CFR factor).  Both may be used
in parallel — TransactionRiskScorer for the ML pipeline and
CFRRiskScorer for the pre-approval gate.
"""

from __future__ import annotations

import logging
from typing import NamedTuple

from src.cfr.registry import CentralFraudRegistry

logger = logging.getLogger(__name__)


# ── Weights (per RBI CFR document) ──────────────────────────────────────────

class CFRWeights(NamedTuple):
    """
    Risk scoring weights defined by the CFR framework.

    Sum must equal 1.0.
    """
    cfr_match: float = 0.40
    large_transaction: float = 0.30
    new_account: float = 0.20
    unusual_location: float = 0.10


DEFAULT_WEIGHTS = CFRWeights()

# ── Thresholds ──────────────────────────────────────────────────────────────

LARGE_TXN_THRESHOLD_PAISA = 200_000_00   # ₹2,00,000
NEW_ACCOUNT_DAYS = 90                      # account < 90 days is "new"
LOCATION_DEVIATION_KM = 100.0             # deviation > 100 km is "unusual"


# ── Score result ────────────────────────────────────────────────────────────

class CFRScoreResult(NamedTuple):
    """Output of CFRRiskScorer.score()."""
    composite_score: float          # ∈ [0.0, 1.0]
    cfr_match_score: float          # raw CFR match factor
    large_txn_score: float          # raw large-txn factor
    new_account_score: float        # raw new-account factor
    location_score: float           # raw location factor
    risk_level: str                 # "LOW" | "MEDIUM" | "HIGH" | "CRITICAL"
    flags: list[str]                # human-readable risk flags


# ── CFR Risk Scorer ─────────────────────────────────────────────────────────

class CFRRiskScorer:
    """
    Compute a CFR-weighted risk score for a transaction.

    The scorer queries the Central Fraud Registry for counterparty
    matches and combines the result with transaction-level signals
    (amount, account age, location) using the 40/30/20/10 weights
    prescribed by the CFR framework.
    """

    def __init__(
        self,
        registry: CentralFraudRegistry,
        weights: CFRWeights | None = None,
    ) -> None:
        self._registry = registry
        self._w = weights or DEFAULT_WEIGHTS
        self._score_count = 0
        self._high_risk_count = 0

    def score(
        self,
        sender_id: str,
        receiver_id: str,
        amount_paisa: int,
        account_age_days: int = 365,
        geo_distance_km: float = 0.0,
        geo_deviation: float = 0.0,
    ) -> CFRScoreResult:
        """
        Score a transaction using CFR-aware weights.

        Parameters
        ----------
        sender_id : str
            Sender account ID.
        receiver_id : str
            Receiver / counterparty account ID.
        amount_paisa : int
            Transaction amount in paisa.
        account_age_days : int
            Age of the sender's account in days.
        geo_distance_km : float
            Distance from the sender's usual location.
        geo_deviation : float
            Normalised deviation from baseline (0.0 — 1.0).
        """
        self._score_count += 1
        flags: list[str] = []

        # ── Factor 1: CFR Match (weight = 40%) ─────────────────────────
        # Check BOTH sender and receiver against CFR
        sender_match = self._registry.check_account(sender_id)
        receiver_match = self._registry.check_account(receiver_id)
        cfr_score = max(sender_match.match_score, receiver_match.match_score)

        if sender_match.is_match:
            flags.append(f"CFR_SENDER_MATCH(severity={sender_match.highest_severity})")
        if receiver_match.is_match:
            flags.append(f"CFR_RECEIVER_MATCH(severity={receiver_match.highest_severity})")

        # ── Factor 2: Large Transaction (weight = 30%) ──────────────────
        if amount_paisa >= LARGE_TXN_THRESHOLD_PAISA:
            ratio = min(1.0, amount_paisa / (LARGE_TXN_THRESHOLD_PAISA * 5))
            large_txn_score = 0.5 + 0.5 * ratio
            flags.append(f"LARGE_TXN(₹{amount_paisa / 100:,.0f})")
        else:
            large_txn_score = amount_paisa / LARGE_TXN_THRESHOLD_PAISA * 0.5

        # ── Factor 3: New Account (weight = 20%) ────────────────────────
        if account_age_days < NEW_ACCOUNT_DAYS:
            new_account_score = 1.0 - (account_age_days / NEW_ACCOUNT_DAYS)
            flags.append(f"NEW_ACCOUNT({account_age_days}d)")
        else:
            new_account_score = 0.0

        # ── Factor 4: Unusual Location (weight = 10%) ──────────────────
        if geo_distance_km > LOCATION_DEVIATION_KM or geo_deviation > 0.6:
            location_score = min(1.0, max(
                geo_distance_km / (LOCATION_DEVIATION_KM * 5),
                geo_deviation,
            ))
            flags.append(f"UNUSUAL_LOCATION({geo_distance_km:.0f}km)")
        else:
            location_score = max(
                geo_distance_km / (LOCATION_DEVIATION_KM * 5),
                geo_deviation * 0.5,
            )

        # ── Composite Score ─────────────────────────────────────────────
        composite = (
            self._w.cfr_match * cfr_score
            + self._w.large_transaction * large_txn_score
            + self._w.new_account * new_account_score
            + self._w.unusual_location * location_score
        )
        composite = min(1.0, max(0.0, composite))

        # ── Risk Level ──────────────────────────────────────────────────
        if composite >= 0.80:
            risk_level = "CRITICAL"
        elif composite >= 0.55:
            risk_level = "HIGH"
        elif composite >= 0.30:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"

        if risk_level in ("HIGH", "CRITICAL"):
            self._high_risk_count += 1

        return CFRScoreResult(
            composite_score=composite,
            cfr_match_score=cfr_score,
            large_txn_score=large_txn_score,
            new_account_score=new_account_score,
            location_score=location_score,
            risk_level=risk_level,
            flags=flags,
        )

    def snapshot(self) -> dict:
        return {
            "scores_computed": self._score_count,
            "high_risk_flagged": self._high_risk_count,
            "weights": self._w._asdict(),
        }

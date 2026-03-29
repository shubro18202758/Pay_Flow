"""
PayFlow — Composite Weighted Risk Scoring Engine
===================================================
Implements the multi-factor risk scoring model described in banking
fraud detection protocols (per FIU-IND specifications):

    Risk Score = 30(Large Transaction) + 30(High Velocity)
              + 20(New Account) + 20(High Risk Country)

Each component yields a sub-score in [0.0, 1.0].  The weighted sum
produces the final composite risk score ∈ [0.0, 1.0].

The "New Account" factor combines two signals:
    1. Account dormancy — accounts reactivated after prolonged inactivity
       or recently opened accounts (< 30 days old).
    2. First-time beneficiary — sender has never transacted with this
       receiver before.

The "High Risk Country" factor combines:
    1. FATF Blacklist / Greylist / Sanctioned jurisdiction checks.
    2. Geo-distance from the sender's usual location.

Default Weights (per banking protocol):
    Large Transaction  : 0.30
    High Velocity      : 0.30
    New Account        : 0.20
    High Risk Country  : 0.20

Integration:
    scorer = TransactionRiskScorer()
    result = scorer.score(txn, velocity_features, behavioral_features)
    # result.composite_score ∈ [0.0, 1.0]
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import NamedTuple


# ── Configuration ───────────────────────────────────────────────────────────

@dataclass(frozen=True)
class RiskWeights:
    """Configurable risk component weights (must sum to 1.0)."""
    large_transaction: float = 0.30
    high_velocity: float = 0.30
    new_account: float = 0.20
    high_risk_country: float = 0.20

    def __post_init__(self) -> None:
        total = (self.large_transaction + self.high_velocity
                 + self.new_account + self.high_risk_country)
        if abs(total - 1.0) > 0.001:
            raise ValueError(f"Weights must sum to 1.0, got {total:.4f}")


@dataclass(frozen=True)
class RiskThresholds:
    """Thresholds for component sub-score computation."""
    large_txn_paisa: int = 50_000_00          # ₹50,000 (in paisa)
    very_large_txn_paisa: int = 200_000_00    # ₹2,00,000 (in paisa)
    high_velocity_count_1h: int = 10          # >10 txns/hour is suspicious
    high_velocity_count_24h: int = 50         # >50 txns/day is suspicious
    geo_distance_km_unusual: float = 100.0    # >100 km from usual location
    geo_distance_km_extreme: float = 500.0    # >500 km = very unusual
    new_account_days: int = 30                # accounts < 30 days old
    dormant_account_days: int = 90            # reactivated after 90+ days idle


# ── Risk Score Result ───────────────────────────────────────────────────────

class RiskScoreResult(NamedTuple):
    """Decomposed risk score with per-component breakdown."""
    composite_score: float          # weighted sum ∈ [0.0, 1.0]
    large_txn_score: float          # ∈ [0.0, 1.0]
    velocity_score: float           # ∈ [0.0, 1.0]
    new_account_score: float        # ∈ [0.0, 1.0] (beneficiary + dormancy)
    location_score: float           # ∈ [0.0, 1.0]
    risk_level: str                 # "LOW" | "MEDIUM" | "HIGH" | "CRITICAL"
    flags: list[str] = []           # human-readable risk flags

    def to_dict(self) -> dict:
        return {
            "composite_score": round(self.composite_score, 4),
            "components": {
                "large_transaction": round(self.large_txn_score, 4),
                "high_velocity": round(self.velocity_score, 4),
                "new_account": round(self.new_account_score, 4),
                "high_risk_country": round(self.location_score, 4),
            },
            "risk_level": self.risk_level,
            "flags": self.flags,
        }


# ── Risk Scorer ─────────────────────────────────────────────────────────────

class TransactionRiskScorer:
    """
    Multi-factor weighted risk scoring engine.

    Tracks known beneficiaries per sender and account activity history
    to detect new/first-time relationships and dormant account
    reactivation.  Integrates velocity and behavioral feature outputs
    from existing PayFlow engines.
    """

    def __init__(
        self,
        weights: RiskWeights | None = None,
        thresholds: RiskThresholds | None = None,
    ) -> None:
        self._weights = weights or RiskWeights()
        self._thresholds = thresholds or RiskThresholds()
        self._known_beneficiaries: dict[str, set[str]] = {}
        # Account lifecycle tracking (account_id -> age in days, last_active_day_offset)
        self._account_age_days: dict[str, int] = {}
        self._account_last_active: dict[str, float] = {}  # epoch seconds
        self._scored_count: int = 0

    def register_account(self, account_id: str, age_days: int) -> None:
        """Register an account's age for dormancy detection."""
        self._account_age_days[account_id] = age_days

    def score(
        self,
        amount_paisa: int,
        txn_count_1h: int,
        txn_count_24h: int,
        velocity_zscore: float,
        sender_id: str,
        receiver_id: str,
        geo_distance_km: float,
        geo_deviation: float,
        account_age_days: int | None = None,
        country_code: str = "",
    ) -> RiskScoreResult:
        """
        Compute composite risk score for a single transaction.

        Parameters
        ----------
        amount_paisa : int
            Transaction amount in paisa.
        txn_count_1h : int
            Number of transactions by sender in last hour.
        txn_count_24h : int
            Number of transactions by sender in last 24h.
        velocity_zscore : float
            Z-score of current hourly rate vs 7-day baseline.
        sender_id : str
            Sender account ID.
        receiver_id : str
            Receiver account ID.
        geo_distance_km : float
            Haversine distance from sender's usual location.
        geo_deviation : float
            Normalised geo deviation from historical baseline.
        account_age_days : int, optional
            Account age in days.  If None, falls back to registered age
            or assumes established account.
        country_code : str, optional
            ISO 3166-1 alpha-2 country code of the counterparty
            or transaction origin.  Used for FATF risk assessment.
        """
        # Component 1: Large Transaction Score
        large_txn = self._score_large_txn(amount_paisa)

        # Component 2: High Velocity Score
        velocity = self._score_velocity(txn_count_1h, txn_count_24h, velocity_zscore)

        # Component 3: New Account Score (combines beneficiary + dormancy)
        new_account, flags = self._score_new_account(
            sender_id, receiver_id, account_age_days,
        )

        # Component 4: High Risk Country Score
        location = self._score_high_risk_country(
            geo_distance_km, geo_deviation, country_code,
        )

        # Weighted composite
        composite = (
            self._weights.large_transaction * large_txn
            + self._weights.high_velocity * velocity
            + self._weights.new_account * new_account
            + self._weights.high_risk_country * location
        )
        composite = min(1.0, max(0.0, composite))

        # Risk level classification
        if composite >= 0.80:
            level = "CRITICAL"
        elif composite >= 0.60:
            level = "HIGH"
        elif composite >= 0.35:
            level = "MEDIUM"
        else:
            level = "LOW"

        self._scored_count += 1

        return RiskScoreResult(
            composite_score=composite,
            large_txn_score=large_txn,
            velocity_score=velocity,
            new_account_score=new_account,
            location_score=location,
            risk_level=level,
            flags=flags,
        )

    def score_from_features(
        self,
        amount_paisa: int,
        sender_id: str,
        receiver_id: str,
        velocity_features: tuple,
        behavioral_features: tuple,
        country_code: str = "",
    ) -> RiskScoreResult:
        """
        Score using raw VelocityFeatures and BehavioralFeatures tuples
        from the FeatureEngine.  Convenient for pipeline integration.
        """
        return self.score(
            amount_paisa=amount_paisa,
            txn_count_1h=velocity_features[0],       # txn_count_1h
            txn_count_24h=velocity_features[2],      # txn_count_24h
            velocity_zscore=velocity_features[10],   # velocity_zscore
            sender_id=sender_id,
            receiver_id=receiver_id,
            geo_distance_km=behavioral_features[2],  # geo_distance_km
            geo_deviation=behavioral_features[3],    # geo_deviation
            country_code=country_code,
        )

    @property
    def scored_count(self) -> int:
        return self._scored_count

    @property
    def tracked_senders(self) -> int:
        return len(self._known_beneficiaries)

    def snapshot(self) -> dict:
        return {
            "scored_count": self._scored_count,
            "tracked_senders": len(self._known_beneficiaries),
            "tracked_accounts": len(self._account_age_days),
            "weights": {
                "large_transaction": self._weights.large_transaction,
                "high_velocity": self._weights.high_velocity,
                "new_account": self._weights.new_account,
                "high_risk_country": self._weights.high_risk_country,
            },
        }

    # ── Component Scorers ───────────────────────────────────────────────

    def _score_large_txn(self, amount_paisa: int) -> float:
        """Sigmoid-like score: ramps between threshold and very_large."""
        t = self._thresholds
        if amount_paisa <= t.large_txn_paisa:
            return 0.0
        if amount_paisa >= t.very_large_txn_paisa:
            return 1.0
        # Linear ramp between thresholds
        return (amount_paisa - t.large_txn_paisa) / (
            t.very_large_txn_paisa - t.large_txn_paisa
        )

    def _score_velocity(
        self, count_1h: int, count_24h: int, zscore: float,
    ) -> float:
        """Combined velocity anomaly from counts and z-score."""
        t = self._thresholds
        # Hourly burst
        hourly = min(1.0, max(0.0, count_1h / t.high_velocity_count_1h))
        # Daily volume
        daily = min(1.0, max(0.0, count_24h / t.high_velocity_count_24h))
        # Z-score component (clamp to [0, 1])
        z_component = min(1.0, max(0.0, zscore / 3.0))
        # Max of all three signals
        return max(hourly, daily, z_component)

    def _score_new_account(
        self, sender_id: str, receiver_id: str,
        account_age_days: int | None,
    ) -> tuple[float, list[str]]:
        """
        Combined "New Account" factor:
          - First-time beneficiary relationship
          - Account dormancy / recently opened

        Returns (score, flags) where score ∈ [0.0, 1.0].
        """
        flags: list[str] = []
        t = self._thresholds

        # Sub-signal 1: first-time beneficiary
        known = self._known_beneficiaries.setdefault(sender_id, set())
        if receiver_id in known:
            beneficiary_score = 0.0
        else:
            known.add(receiver_id)
            beneficiary_score = 1.0
            flags.append("first_time_beneficiary")

        # Sub-signal 2: account age / dormancy
        age = account_age_days
        if age is None:
            age = self._account_age_days.get(sender_id)

        dormancy_score = 0.0
        if age is not None:
            if age <= t.new_account_days:
                # Very new account — linear ramp (0 days → 1.0, 30 days → 0.0)
                dormancy_score = max(0.0, 1.0 - age / t.new_account_days)
                flags.append("new_account")
            # Track activity gaps for dormancy detection
            import time as _time
            now = _time.time()
            last_active = self._account_last_active.get(sender_id)
            if last_active is not None:
                idle_days = (now - last_active) / 86400
                if idle_days >= t.dormant_account_days:
                    dormancy_score = max(
                        dormancy_score,
                        min(1.0, idle_days / (2 * t.dormant_account_days)),
                    )
                    flags.append("dormant_reactivation")
            self._account_last_active[sender_id] = now

        # Combined: max of both signals
        return max(beneficiary_score, dormancy_score), flags

    def _score_high_risk_country(
        self,
        geo_distance_km: float,
        geo_deviation: float,
        country_code: str = "",
    ) -> float:
        """High Risk Country factor: FATF jurisdiction risk + geo distance."""
        from src.ml.rule_engine import (
            FATF_BLACKLIST,
            FATF_GREYLIST,
            SANCTIONED_JURISDICTIONS,
        )

        # Sub-signal 1: Country jurisdiction risk
        country_score = 0.0
        cc = country_code.upper().strip()
        if cc:
            if cc in FATF_BLACKLIST or cc in SANCTIONED_JURISDICTIONS:
                country_score = 1.0
            elif cc in FATF_GREYLIST:
                country_score = 0.75

        # Sub-signal 2: Geo distance / deviation
        t = self._thresholds
        dist_score = 0.0
        if geo_distance_km > 0 or geo_deviation > 0:
            if geo_distance_km >= t.geo_distance_km_extreme:
                dist_score = 1.0
            elif geo_distance_km > t.geo_distance_km_unusual:
                dist_score = (geo_distance_km - t.geo_distance_km_unusual) / (
                    t.geo_distance_km_extreme - t.geo_distance_km_unusual
                )
            dev_score = min(1.0, max(0.0, geo_deviation / 3.0))
            dist_score = max(dist_score, dev_score)

        # Combined: country risk dominates when present
        return max(country_score, dist_score)


# ── Build-Prompt Point-Based Risk Score ─────────────────────────────────────

class PointRiskResult(NamedTuple):
    """Point-based risk score result (max 100)."""
    risk_score: int              # 0–100
    breakdown: dict              # category -> points awarded
    verdict: str                 # LEGITIMATE / MONITOR / INVESTIGATE / BLOCK
    fraud_patterns: list[str]    # detected pattern names

    def to_dict(self) -> dict:
        return {
            "risk_score": self.risk_score,
            "breakdown": dict(self.breakdown),
            "verdict": self.verdict,
            "fraud_patterns": list(self.fraud_patterns),
        }


def compute_point_risk_score(features: dict) -> PointRiskResult:
    """
    Build-prompt prescribed point-based risk scoring.

    Max score: 100. Block threshold: 70.

    Categories:
      - Transaction signals: max 30
      - Velocity signals: max 30
      - Account signals: max 40
      - Behavioral signals: max 30
      - Graph signals: max 50
      - Geographic risk: max 20
    """
    score = 0
    breakdown: dict[str, int] = {}
    patterns: list[str] = []

    amount = features.get("amount", 0)
    amount_paisa = features.get("amount_paisa", amount * 100 if amount else 0)
    # Normalize to INR if in paisa
    if amount == 0 and amount_paisa > 0:
        amount = amount_paisa / 100

    # ── TRANSACTION SIGNALS (max 30) ──
    if amount > 10_00_000:        # >₹10 Lakhs (CTR trigger)
        score += 30
        breakdown["large_transaction"] = 30
        patterns.append("CTR_THRESHOLD_BREACH")
    elif amount > 5_00_000:       # >₹5 Lakhs
        score += 20
        breakdown["large_transaction"] = 20
    elif amount > 1_00_000:       # >₹1 Lakh
        score += 10
        breakdown["large_transaction"] = 10

    # ── VELOCITY SIGNALS (max 30) ──
    txn_1h = features.get("txn_count_last_1hr", features.get("vel_txn_count_1h", 0))
    if txn_1h > 20:
        score += 30
        breakdown["high_velocity"] = 30
        patterns.append("EXTREME_VELOCITY")
    elif txn_1h > 10:
        score += 20
        breakdown["high_velocity"] = 20
        patterns.append("HIGH_VELOCITY")
    elif txn_1h > 5:
        score += 10
        breakdown["high_velocity"] = 10

    # ── ACCOUNT SIGNALS (max 40) ──
    if features.get("cfr_match", False):
        score += 40
        breakdown["cfr_match"] = 40
        patterns.append("CFR_REGISTRY_MATCH")
    else:
        account_age = features.get("account_age_days", features.get("sender_account_age_days", 999))
        if account_age < 30:
            score += 20
            breakdown["new_account"] = 20
            patterns.append("NEW_ACCOUNT")
        elif account_age < 90:
            score += 10
            breakdown["new_account"] = 10

    # ── BEHAVIORAL SIGNALS (max 30) ──
    is_night = features.get("is_night_transaction", features.get("beh_is_off_hours", False))
    if is_night:
        score += 10
        breakdown["night_transaction"] = 10
        patterns.append("NIGHT_TRANSACTION")

    location_dev = features.get("location_deviation_km", features.get("beh_geo_distance_km", 0))
    if location_dev > 500:
        score += 10
        breakdown["unusual_location"] = 10
        patterns.append("LOCATION_ANOMALY")

    device_known = features.get("device_known", True)
    if not device_known:
        score += 10
        breakdown["unknown_device"] = 10
        patterns.append("UNKNOWN_DEVICE")

    # ── GRAPH SIGNALS (max 50) ──
    if features.get("circular_flow_detected", False):
        score += 20
        breakdown["circular_flow"] = 20
        patterns.append("CIRCULAR_FLOW")

    betweenness = features.get("sender_betweenness", features.get("net_betweenness", 0))
    if betweenness > 0.1:
        score += 15
        breakdown["high_centrality"] = 15
        patterns.append("HIGH_BETWEENNESS_CENTRALITY")

    if features.get("pass_through_detected", False):
        score += 15
        breakdown["pass_through"] = 15
        patterns.append("RAPID_PASS_THROUGH")

    # ── GEOGRAPHIC RISK (max 20) ──
    if features.get("is_high_risk_country", features.get("ext_is_cross_border", False)):
        score += 20
        breakdown["high_risk_country"] = 20
        patterns.append("HIGH_RISK_JURISDICTION")

    final_score = min(score, 100)

    # Route verdict
    if final_score < 30:
        verdict = "LEGITIMATE"
    elif final_score < 50:
        verdict = "MONITOR"
    elif final_score < 70:
        verdict = "INVESTIGATE"
    else:
        verdict = "BLOCK"

    return PointRiskResult(
        risk_score=final_score,
        breakdown=breakdown,
        verdict=verdict,
        fraud_patterns=patterns,
    )

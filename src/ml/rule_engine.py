"""
PayFlow — Configurable Rule-Based Detection Engine
=====================================================
Implements explicit IF-THEN fraud detection rules that evaluate
transactions independently from the ML pipeline.  This provides
an interpretable, auditable first line of defence aligned with
RBI/FIU-IND regulatory expectations.

Rule Categories (per RTPM specification):
  1. Amount threshold rules  — large/suspicious amounts
  2. Velocity rules          — rapid-fire transactions
  3. Beneficiary rules       — transfers to new/untrusted recipients
  4. Location rules          — geo-anomalous transactions
  5. Temporal rules          — off-hours / holiday transactions
  6. Device rules            — unknown or shared devices

Each rule produces a RuleViolation with severity (LOW/MEDIUM/HIGH/CRITICAL)
and a human-readable explanation.  The engine returns ALL matched rules
so downstream consumers (PreApprovalGate, AlertRouter) can aggregate.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import NamedTuple

logger = logging.getLogger(__name__)


# ── Data Structures ───────────────────────────────────────────────────────────

class RuleSeverity(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class RuleViolation(NamedTuple):
    """Single rule match result."""
    rule_id: str
    rule_name: str
    severity: RuleSeverity
    score: float           # contribution ∈ [0.0, 1.0]
    reason: str            # human-readable explanation
    category: str          # amount | velocity | beneficiary | location | temporal | device


@dataclass(frozen=True)
class RuleEngineConfig:
    """Tunable thresholds for all built-in rules."""
    # Amount thresholds (in paisa — 100 paisa = ₹1)
    amount_suspicious: int = 50_000_00       # ₹50,000
    amount_very_large: int = 2_00_000_00     # ₹2,00,000
    amount_critical: int = 10_00_000_00      # ₹10,00,000 (CTR threshold)

    # Velocity thresholds
    velocity_txn_per_hour_warn: int = 5
    velocity_txn_per_hour_high: int = 10
    velocity_txn_per_day_warn: int = 20
    velocity_txn_per_day_high: int = 50
    velocity_zscore_threshold: float = 3.0

    # Location thresholds (km)
    geo_unusual_km: float = 100.0
    geo_extreme_km: float = 500.0

    # Temporal thresholds (24-hour format)
    off_hours_start: int = 23    # 11 PM
    off_hours_end: int = 5       # 5 AM

    # New beneficiary lookback (seconds)
    new_beneficiary_window_sec: int = 86400  # 24 hours

    # AML — high-risk country / sanctions
    enable_geo_risk: bool = True


@dataclass
class RuleEngineMetrics:
    """Runtime counters."""
    evaluations: int = 0
    total_violations: int = 0
    violations_by_severity: dict = field(default_factory=lambda: {
        "LOW": 0, "MEDIUM": 0, "HIGH": 0, "CRITICAL": 0,
    })
    violations_by_category: dict = field(default_factory=lambda: {
        "amount": 0, "velocity": 0, "beneficiary": 0,
        "location": 0, "temporal": 0, "device": 0,
        "geo_risk": 0,
    })
    avg_evaluation_us: float = 0.0
    _total_us: float = 0.0

    def record(self, violations: list[RuleViolation], elapsed_us: float) -> None:
        self.evaluations += 1
        self._total_us += elapsed_us
        self.avg_evaluation_us = self._total_us / self.evaluations
        for v in violations:
            self.total_violations += 1
            self.violations_by_severity[v.severity.value] = (
                self.violations_by_severity.get(v.severity.value, 0) + 1
            )
            self.violations_by_category[v.category] = (
                self.violations_by_category.get(v.category, 0) + 1
            )

    def snapshot(self) -> dict:
        return {
            "evaluations": self.evaluations,
            "total_violations": self.total_violations,
            "by_severity": dict(self.violations_by_severity),
            "by_category": dict(self.violations_by_category),
            "avg_evaluation_us": round(self.avg_evaluation_us, 1),
        }


class RuleEvaluationResult(NamedTuple):
    """Aggregate result of running all rules on a transaction."""
    violations: list[RuleViolation]
    max_severity: RuleSeverity
    composite_score: float     # max individual rule score
    evaluation_us: float       # microseconds

    def to_dict(self) -> dict:
        return {
            "violations": [
                {
                    "rule_id": v.rule_id,
                    "rule_name": v.rule_name,
                    "severity": v.severity.value,
                    "score": round(v.score, 4),
                    "reason": v.reason,
                    "category": v.category,
                }
                for v in self.violations
            ],
            "max_severity": self.max_severity.value,
            "composite_score": round(self.composite_score, 4),
            "evaluation_us": round(self.evaluation_us, 1),
            "triggered_count": len(self.violations),
        }


# ── Rule Engine ───────────────────────────────────────────────────────────────

_SEVERITY_ORDER = {
    RuleSeverity.LOW: 0,
    RuleSeverity.MEDIUM: 1,
    RuleSeverity.HIGH: 2,
    RuleSeverity.CRITICAL: 3,
}


# ── FATF AML Country Lists (2024) ─────────────────────────────────────────────
# Ref: Financial Action Task Force — High-Risk Jurisdictions

FATF_BLACKLIST: frozenset[str] = frozenset({
    "KP",  # North Korea
    "IR",  # Iran
    "MM",  # Myanmar
})

FATF_GREYLIST: frozenset[str] = frozenset({
    "BF",  # Burkina Faso
    "CM",  # Cameroon
    "HR",  # Croatia
    "CD",  # Congo (DRC)
    "HT",  # Haiti
    "KE",  # Kenya
    "ML",  # Mali
    "MZ",  # Mozambique
    "NG",  # Nigeria
    "PH",  # Philippines
    "SN",  # Senegal
    "SS",  # South Sudan
    "SY",  # Syria
    "TZ",  # Tanzania
    "VN",  # Vietnam
    "YE",  # Yemen
})

# Sanctions: OFAC SDN / UN Security Council consolidated list (ISO-3166 alpha-2)
SANCTIONED_JURISDICTIONS: frozenset[str] = frozenset({
    "KP", "IR", "SY", "CU", "VE", "RU", "BY",
})


class TransactionRuleEngine:
    """
    Configurable rule-based fraud detection engine.

    Evaluates a transaction against all enabled rules and returns
    every violation found.  Rules are pure functions with no side
    effects — they only read the provided transaction context.

    Usage::

        engine = TransactionRuleEngine()
        result = engine.evaluate(
            amount_paisa=500_000_00,
            txn_count_1h=12,
            ...
        )
        for v in result.violations:
            print(v.rule_name, v.severity, v.reason)
    """

    def __init__(self, config: RuleEngineConfig | None = None) -> None:
        self._cfg = config or RuleEngineConfig()
        self.metrics = RuleEngineMetrics()

        # Beneficiary tracking: sender_id → set of known receiver_ids
        self._known_beneficiaries: dict[str, set[str]] = {}

        # Rule registry: list of (rule_id, rule_fn) — all enabled by default
        self._disabled_rules: set[str] = set()

    # ── Public API ────────────────────────────────────────────────────────

    def evaluate(
        self,
        amount_paisa: int,
        sender_id: str,
        receiver_id: str,
        timestamp: int,
        txn_count_1h: int = 0,
        txn_count_24h: int = 0,
        velocity_zscore: float = 0.0,
        geo_distance_km: float = 0.0,
        device_trusted: bool = True,
        device_fingerprint: str = "",
        hour_of_day: int = -1,
        country_code: str = "",
    ) -> RuleEvaluationResult:
        """
        Evaluate a transaction against all enabled rules.

        Returns a RuleEvaluationResult containing every triggered rule,
        the maximum severity encountered, and a composite score.
        """
        t0 = time.perf_counter()
        violations: list[RuleViolation] = []

        # Run all rule categories
        violations.extend(self._check_amount_rules(amount_paisa))
        violations.extend(self._check_velocity_rules(
            txn_count_1h, txn_count_24h, velocity_zscore,
        ))
        violations.extend(self._check_beneficiary_rules(
            sender_id, receiver_id, timestamp,
        ))
        violations.extend(self._check_location_rules(geo_distance_km))
        violations.extend(self._check_temporal_rules(hour_of_day, timestamp))
        violations.extend(self._check_device_rules(device_trusted))
        violations.extend(self._check_geo_risk_rules(country_code))

        # Filter out disabled rules
        if self._disabled_rules:
            violations = [v for v in violations if v.rule_id not in self._disabled_rules]

        # Compute aggregate severity
        if violations:
            max_sev = max(violations, key=lambda v: _SEVERITY_ORDER[v.severity]).severity
            composite = max(v.score for v in violations)
        else:
            max_sev = RuleSeverity.LOW
            composite = 0.0

        elapsed_us = (time.perf_counter() - t0) * 1_000_000
        self.metrics.record(violations, elapsed_us)

        return RuleEvaluationResult(
            violations=violations,
            max_severity=max_sev,
            composite_score=composite,
            evaluation_us=elapsed_us,
        )

    def register_beneficiary(self, sender_id: str, receiver_id: str) -> None:
        """Record a known sender→receiver relationship."""
        self._known_beneficiaries.setdefault(sender_id, set()).add(receiver_id)

    def disable_rule(self, rule_id: str) -> None:
        """Disable a rule by ID."""
        self._disabled_rules.add(rule_id)

    def enable_rule(self, rule_id: str) -> None:
        """Re-enable a previously disabled rule."""
        self._disabled_rules.discard(rule_id)

    def list_rules(self) -> list[dict]:
        """Return metadata for all available rules."""
        rules = [
            {"id": "AMT_SUSPICIOUS", "name": "Suspicious Amount",
             "category": "amount", "threshold": f"≥ ₹{self._cfg.amount_suspicious // 100:,}"},
            {"id": "AMT_VERY_LARGE", "name": "Very Large Amount",
             "category": "amount", "threshold": f"≥ ₹{self._cfg.amount_very_large // 100:,}"},
            {"id": "AMT_CRITICAL", "name": "Critical Amount (CTR)",
             "category": "amount", "threshold": f"≥ ₹{self._cfg.amount_critical // 100:,}"},
            {"id": "VEL_HOURLY_WARN", "name": "High Hourly Velocity",
             "category": "velocity", "threshold": f"≥ {self._cfg.velocity_txn_per_hour_warn} txn/hr"},
            {"id": "VEL_HOURLY_HIGH", "name": "Extreme Hourly Velocity",
             "category": "velocity", "threshold": f"≥ {self._cfg.velocity_txn_per_hour_high} txn/hr"},
            {"id": "VEL_DAILY_WARN", "name": "High Daily Velocity",
             "category": "velocity", "threshold": f"≥ {self._cfg.velocity_txn_per_day_warn} txn/day"},
            {"id": "VEL_DAILY_HIGH", "name": "Extreme Daily Velocity",
             "category": "velocity", "threshold": f"≥ {self._cfg.velocity_txn_per_day_high} txn/day"},
            {"id": "VEL_ZSCORE", "name": "Velocity Z-Score Spike",
             "category": "velocity", "threshold": f"z > {self._cfg.velocity_zscore_threshold}"},
            {"id": "BEN_NEW", "name": "New Beneficiary",
             "category": "beneficiary", "threshold": "First transfer to recipient"},
            {"id": "LOC_UNUSUAL", "name": "Unusual Location",
             "category": "location", "threshold": f"≥ {self._cfg.geo_unusual_km} km"},
            {"id": "LOC_EXTREME", "name": "Extreme Location Jump",
             "category": "location", "threshold": f"≥ {self._cfg.geo_extreme_km} km"},
            {"id": "TMP_OFF_HOURS", "name": "Off-Hours Transaction",
             "category": "temporal", "threshold": f"{self._cfg.off_hours_start}:00–{self._cfg.off_hours_end}:00"},
            {"id": "DEV_UNTRUSTED", "name": "Untrusted Device",
             "category": "device", "threshold": "Device not in trusted list"},
            {"id": "GEO_FATF_BLACKLIST", "name": "FATF Blacklisted Country",
             "category": "geo_risk", "threshold": "Country on FATF blacklist"},
            {"id": "GEO_FATF_GREYLIST", "name": "FATF Greylisted Country",
             "category": "geo_risk", "threshold": "Country on FATF greylist"},
            {"id": "GEO_SANCTIONED", "name": "Sanctioned Jurisdiction",
             "category": "geo_risk", "threshold": "OFAC/UN sanctioned country"},
        ]
        for r in rules:
            r["enabled"] = r["id"] not in self._disabled_rules
        return rules

    @property
    def config(self) -> RuleEngineConfig:
        return self._cfg

    def snapshot(self) -> dict:
        return {
            **self.metrics.snapshot(),
            "disabled_rules": list(self._disabled_rules),
            "known_beneficiary_pairs": sum(
                len(v) for v in self._known_beneficiaries.values()
            ),
        }

    # ── Rule Implementations ──────────────────────────────────────────────

    def _check_amount_rules(self, amount_paisa: int) -> list[RuleViolation]:
        violations = []

        if amount_paisa >= self._cfg.amount_critical:
            violations.append(RuleViolation(
                rule_id="AMT_CRITICAL",
                rule_name="Critical Amount (CTR Threshold)",
                severity=RuleSeverity.CRITICAL,
                score=1.0,
                reason=f"Amount ₹{amount_paisa // 100:,} exceeds CTR threshold ₹{self._cfg.amount_critical // 100:,}",
                category="amount",
            ))
        elif amount_paisa >= self._cfg.amount_very_large:
            violations.append(RuleViolation(
                rule_id="AMT_VERY_LARGE",
                rule_name="Very Large Transaction",
                severity=RuleSeverity.HIGH,
                score=0.8,
                reason=f"Amount ₹{amount_paisa // 100:,} is very large (≥ ₹{self._cfg.amount_very_large // 100:,})",
                category="amount",
            ))
        elif amount_paisa >= self._cfg.amount_suspicious:
            violations.append(RuleViolation(
                rule_id="AMT_SUSPICIOUS",
                rule_name="Suspicious Amount",
                severity=RuleSeverity.MEDIUM,
                score=0.5,
                reason=f"Amount ₹{amount_paisa // 100:,} exceeds suspicious threshold ₹{self._cfg.amount_suspicious // 100:,}",
                category="amount",
            ))

        return violations

    def _check_velocity_rules(
        self,
        txn_count_1h: int,
        txn_count_24h: int,
        velocity_zscore: float,
    ) -> list[RuleViolation]:
        violations = []

        if txn_count_1h >= self._cfg.velocity_txn_per_hour_high:
            violations.append(RuleViolation(
                rule_id="VEL_HOURLY_HIGH",
                rule_name="Extreme Hourly Velocity",
                severity=RuleSeverity.HIGH,
                score=0.9,
                reason=f"{txn_count_1h} transactions in 1 hour (≥ {self._cfg.velocity_txn_per_hour_high})",
                category="velocity",
            ))
        elif txn_count_1h >= self._cfg.velocity_txn_per_hour_warn:
            violations.append(RuleViolation(
                rule_id="VEL_HOURLY_WARN",
                rule_name="High Hourly Velocity",
                severity=RuleSeverity.MEDIUM,
                score=0.5,
                reason=f"{txn_count_1h} transactions in 1 hour (≥ {self._cfg.velocity_txn_per_hour_warn})",
                category="velocity",
            ))

        if txn_count_24h >= self._cfg.velocity_txn_per_day_high:
            violations.append(RuleViolation(
                rule_id="VEL_DAILY_HIGH",
                rule_name="Extreme Daily Velocity",
                severity=RuleSeverity.HIGH,
                score=0.85,
                reason=f"{txn_count_24h} transactions in 24 hours (≥ {self._cfg.velocity_txn_per_day_high})",
                category="velocity",
            ))
        elif txn_count_24h >= self._cfg.velocity_txn_per_day_warn:
            violations.append(RuleViolation(
                rule_id="VEL_DAILY_WARN",
                rule_name="High Daily Velocity",
                severity=RuleSeverity.MEDIUM,
                score=0.45,
                reason=f"{txn_count_24h} transactions in 24 hours (≥ {self._cfg.velocity_txn_per_day_warn})",
                category="velocity",
            ))

        if velocity_zscore >= self._cfg.velocity_zscore_threshold:
            violations.append(RuleViolation(
                rule_id="VEL_ZSCORE",
                rule_name="Velocity Z-Score Spike",
                severity=RuleSeverity.HIGH,
                score=min(1.0, velocity_zscore / 5.0),
                reason=f"Velocity z-score {velocity_zscore:.2f} exceeds {self._cfg.velocity_zscore_threshold}",
                category="velocity",
            ))

        return violations

    def _check_beneficiary_rules(
        self,
        sender_id: str,
        receiver_id: str,
        timestamp: int,
    ) -> list[RuleViolation]:
        violations = []
        known = self._known_beneficiaries.get(sender_id, set())

        if receiver_id not in known:
            violations.append(RuleViolation(
                rule_id="BEN_NEW",
                rule_name="New Beneficiary Transfer",
                severity=RuleSeverity.MEDIUM,
                score=0.4,
                reason=f"First transfer from {sender_id[:8]}… to {receiver_id[:8]}…",
                category="beneficiary",
            ))
            # Auto-register for future lookups
            self._known_beneficiaries.setdefault(sender_id, set()).add(receiver_id)

        return violations

    def _check_location_rules(self, geo_distance_km: float) -> list[RuleViolation]:
        violations = []

        if geo_distance_km >= self._cfg.geo_extreme_km:
            violations.append(RuleViolation(
                rule_id="LOC_EXTREME",
                rule_name="Extreme Location Jump",
                severity=RuleSeverity.HIGH,
                score=0.85,
                reason=f"Transaction {geo_distance_km:.0f} km from usual location (≥ {self._cfg.geo_extreme_km} km)",
                category="location",
            ))
        elif geo_distance_km >= self._cfg.geo_unusual_km:
            violations.append(RuleViolation(
                rule_id="LOC_UNUSUAL",
                rule_name="Unusual Location",
                severity=RuleSeverity.MEDIUM,
                score=0.5,
                reason=f"Transaction {geo_distance_km:.0f} km from usual location (≥ {self._cfg.geo_unusual_km} km)",
                category="location",
            ))

        return violations

    def _check_temporal_rules(
        self, hour_of_day: int, timestamp: int,
    ) -> list[RuleViolation]:
        violations = []

        # Derive hour from timestamp if not provided
        if hour_of_day < 0:
            import datetime
            hour_of_day = datetime.datetime.fromtimestamp(
                timestamp, tz=datetime.timezone.utc,
            ).hour

        is_off_hours = (
            hour_of_day >= self._cfg.off_hours_start
            or hour_of_day < self._cfg.off_hours_end
        )

        if is_off_hours:
            violations.append(RuleViolation(
                rule_id="TMP_OFF_HOURS",
                rule_name="Off-Hours Transaction",
                severity=RuleSeverity.LOW,
                score=0.25,
                reason=f"Transaction at {hour_of_day:02d}:00 UTC (off-hours: {self._cfg.off_hours_start}:00–{self._cfg.off_hours_end}:00)",
                category="temporal",
            ))

        return violations

    def _check_device_rules(self, device_trusted: bool) -> list[RuleViolation]:
        violations = []

        if not device_trusted:
            violations.append(RuleViolation(
                rule_id="DEV_UNTRUSTED",
                rule_name="Untrusted Device",
                severity=RuleSeverity.MEDIUM,
                score=0.55,
                reason="Transaction originated from an untrusted or unknown device",
                category="device",
            ))

        return violations

    def _check_geo_risk_rules(self, country_code: str) -> list[RuleViolation]:
        """AML geo-risk: FATF blacklist/greylist + sanctions screening."""
        if not country_code or not self._cfg.enable_geo_risk:
            return []

        violations: list[RuleViolation] = []
        cc = country_code.upper()

        if cc in FATF_BLACKLIST:
            violations.append(RuleViolation(
                rule_id="GEO_FATF_BLACKLIST",
                rule_name="FATF Blacklisted Country",
                severity=RuleSeverity.CRITICAL,
                score=1.0,
                reason=f"Transaction involves FATF blacklisted jurisdiction: {cc}",
                category="geo_risk",
            ))
        elif cc in FATF_GREYLIST:
            violations.append(RuleViolation(
                rule_id="GEO_FATF_GREYLIST",
                rule_name="FATF Greylisted Country",
                severity=RuleSeverity.HIGH,
                score=0.75,
                reason=f"Transaction involves FATF greylisted jurisdiction: {cc}",
                category="geo_risk",
            ))

        if cc in SANCTIONED_JURISDICTIONS:
            violations.append(RuleViolation(
                rule_id="GEO_SANCTIONED",
                rule_name="Sanctioned Jurisdiction",
                severity=RuleSeverity.CRITICAL,
                score=1.0,
                reason=f"Transaction involves OFAC/UN sanctioned jurisdiction: {cc}",
                category="geo_risk",
            ))

        return violations

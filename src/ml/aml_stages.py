"""
PayFlow — AML Money Laundering Stage Detectors
=================================================
Implements detection logic for the three classic stages of money
laundering as defined by FATF/RBI AML guidelines:

  1. **Placement** — Illicit cash enters the financial system via
     deposits, currency exchanges, or structured sub-threshold
     transactions (smurfing).

  2. **Layering** — Already covered by CycleDetector and MuleDetector
     in the graph module (rapid multi-hop routing to obscure origin).

  3. **Integration** — Laundered funds re-enter the legitimate economy
     through asset purchases, investments, or business commingling.

This module adds Placement and Integration detectors; Layering is
handled by ``src/graph/algorithms.py``.

References:
    • FATF Recommendations (2012, updated 2023) — Recommendation 20
    • RBI Master Direction on KYC (2016, updated 2024) — Chapter V
    • Prevention of Money Laundering Act (PMLA) 2002 — Schedule Part A
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import NamedTuple

logger = logging.getLogger(__name__)


# ── Common Enums ──────────────────────────────────────────────────────────────

class AMLStage(str, Enum):
    PLACEMENT = "PLACEMENT"
    LAYERING = "LAYERING"
    INTEGRATION = "INTEGRATION"


class AMLRisk(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class AMLAlert(NamedTuple):
    """Single AML stage detection alert."""
    stage: AMLStage
    risk: AMLRisk
    score: float              # ∈ [0.0, 1.0]
    indicator: str            # machine-readable indicator code
    reason: str               # human-readable explanation
    account_id: str
    metadata: dict = {}


# ── Placement Detector ────────────────────────────────────────────────────────

_CTR_THRESHOLD_PAISA = 10_00_000_00  # ₹10,00,000 (CTR reporting threshold)
_STRUCTURING_THRESHOLD_PAISA = 9_50_000_00  # ₹9,50,000 (just-below-CTR)


@dataclass
class PlacementMetrics:
    """Runtime counters for PlacementDetector."""
    evaluations: int = 0
    cash_alerts: int = 0
    structuring_alerts: int = 0
    multi_channel_alerts: int = 0
    round_amount_alerts: int = 0

    def snapshot(self) -> dict:
        return {
            "evaluations": self.evaluations,
            "cash_alerts": self.cash_alerts,
            "structuring_alerts": self.structuring_alerts,
            "multi_channel_alerts": self.multi_channel_alerts,
            "round_amount_alerts": self.round_amount_alerts,
        }


class PlacementDetector:
    """
    Detects placement-stage money laundering patterns:

    1. **Large cash deposits** — Single cash deposit ≥ CTR threshold
    2. **Structuring / Smurfing** — Multiple deposits just below ₹10L
       within a window to evade CTR reporting
    3. **Multi-channel aggregation** — Same account receives funds via
       RTGS + NEFT + cash deposit within 24 hours
    4. **Round-amount deposits** — Frequent exact round-number deposits
       (e.g., ₹1,00,000, ₹5,00,000) indicating cash placement
    """

    def __init__(
        self,
        structuring_window_sec: int = 86400,
        structuring_min_count: int = 3,
        multi_channel_min: int = 3,
        round_amount_min_count: int = 3,
    ) -> None:
        self._structuring_window = structuring_window_sec
        self._structuring_min = structuring_min_count
        self._multi_channel_min = multi_channel_min
        self._round_amount_min = round_amount_min_count
        self.metrics = PlacementMetrics()

        # Rolling state per account
        self._deposit_history: dict[str, list[dict]] = {}  # account → [{amount, ts, channel}]

    # ── Public API ────────────────────────────────────────────────────────

    def evaluate(
        self,
        account_id: str,
        amount_paisa: int,
        timestamp: int,
        channel: str = "CASH",
        is_deposit: bool = True,
    ) -> list[AMLAlert]:
        """Evaluate a single deposit transaction for placement indicators."""
        self.metrics.evaluations += 1

        if not is_deposit:
            return []

        alerts: list[AMLAlert] = []

        # Record this deposit
        self._deposit_history.setdefault(account_id, []).append({
            "amount": amount_paisa,
            "ts": timestamp,
            "channel": channel.upper(),
        })

        # 1. Large cash deposit
        if channel.upper() == "CASH" and amount_paisa >= _CTR_THRESHOLD_PAISA:
            self.metrics.cash_alerts += 1
            alerts.append(AMLAlert(
                stage=AMLStage.PLACEMENT,
                risk=AMLRisk.HIGH,
                score=0.85,
                indicator="PLACEMENT_LARGE_CASH",
                reason=(
                    f"Large cash deposit ₹{amount_paisa // 100:,} exceeds "
                    f"CTR threshold ₹{_CTR_THRESHOLD_PAISA // 100:,}"
                ),
                account_id=account_id,
                metadata={"amount_paisa": amount_paisa, "channel": channel},
            ))

        # 2. Structuring detection (multiple just-below-threshold deposits)
        history = self._deposit_history.get(account_id, [])
        window_start = timestamp - self._structuring_window
        recent = [d for d in history if d["ts"] >= window_start]

        just_below = [
            d for d in recent
            if _STRUCTURING_THRESHOLD_PAISA <= d["amount"] < _CTR_THRESHOLD_PAISA
        ]
        if len(just_below) >= self._structuring_min:
            total = sum(d["amount"] for d in just_below)
            self.metrics.structuring_alerts += 1
            alerts.append(AMLAlert(
                stage=AMLStage.PLACEMENT,
                risk=AMLRisk.CRITICAL,
                score=0.95,
                indicator="PLACEMENT_STRUCTURING",
                reason=(
                    f"{len(just_below)} deposits just below CTR threshold "
                    f"within {self._structuring_window // 3600}h totalling "
                    f"₹{total // 100:,} — possible smurfing/structuring"
                ),
                account_id=account_id,
                metadata={
                    "count": len(just_below),
                    "total_paisa": total,
                    "window_sec": self._structuring_window,
                },
            ))

        # 3. Multi-channel aggregation
        channels = {d["channel"] for d in recent}
        if len(channels) >= self._multi_channel_min:
            self.metrics.multi_channel_alerts += 1
            alerts.append(AMLAlert(
                stage=AMLStage.PLACEMENT,
                risk=AMLRisk.HIGH,
                score=0.80,
                indicator="PLACEMENT_MULTI_CHANNEL",
                reason=(
                    f"Deposits via {len(channels)} channels ({', '.join(sorted(channels))}) "
                    f"within {self._structuring_window // 3600}h — multi-channel aggregation"
                ),
                account_id=account_id,
                metadata={"channels": sorted(channels)},
            ))

        # 4. Round-amount deposits
        round_deps = [
            d for d in recent
            if d["amount"] >= 1_00_000_00 and d["amount"] % 1_00_000_00 == 0
        ]
        if len(round_deps) >= self._round_amount_min:
            self.metrics.round_amount_alerts += 1
            alerts.append(AMLAlert(
                stage=AMLStage.PLACEMENT,
                risk=AMLRisk.MEDIUM,
                score=0.60,
                indicator="PLACEMENT_ROUND_AMOUNTS",
                reason=(
                    f"{len(round_deps)} round-number deposits (multiples of ₹1,00,000) "
                    f"within {self._structuring_window // 3600}h"
                ),
                account_id=account_id,
                metadata={"count": len(round_deps)},
            ))

        # Prune old history (keep last 72h to limit memory)
        cutoff = timestamp - 259200
        self._deposit_history[account_id] = [
            d for d in self._deposit_history[account_id] if d["ts"] >= cutoff
        ]

        return alerts

    def snapshot(self) -> dict:
        return {
            "stage": "PLACEMENT",
            "tracked_accounts": len(self._deposit_history),
            **self.metrics.snapshot(),
        }


# ── Integration Detector ──────────────────────────────────────────────────────

@dataclass
class IntegrationMetrics:
    """Runtime counters for IntegrationDetector."""
    evaluations: int = 0
    asset_purchase_alerts: int = 0
    investment_alerts: int = 0
    rapid_withdrawal_alerts: int = 0
    round_trip_alerts: int = 0

    def snapshot(self) -> dict:
        return {
            "evaluations": self.evaluations,
            "asset_purchase_alerts": self.asset_purchase_alerts,
            "investment_alerts": self.investment_alerts,
            "rapid_withdrawal_alerts": self.rapid_withdrawal_alerts,
            "round_trip_alerts": self.round_trip_alerts,
        }


# Transaction purpose codes that indicate asset purchases / investments
_ASSET_PURPOSE_CODES: frozenset[str] = frozenset({
    "PROPERTY", "REAL_ESTATE", "VEHICLE", "JEWELRY",
    "PRECIOUS_METALS", "ART", "LUXURY_GOODS",
})

_INVESTMENT_PURPOSE_CODES: frozenset[str] = frozenset({
    "MUTUAL_FUND", "STOCK", "BOND", "DEBENTURE", "FD",
    "INSURANCE_PREMIUM", "CRYPTO", "FOREX", "COMMODITY",
})


class IntegrationDetector:
    """
    Detects integration-stage money laundering patterns:

    1. **Large asset purchases** — High-value payments to real estate,
       vehicles, jewelry, or luxury goods shortly after large inflows.
    2. **Investment patterns** — Sudden investment activity from accounts
       with no prior investment history.
    3. **Rapid withdrawal** — Large withdrawals shortly after deposits
       (deposit-withdraw churn indicating round-tripping).
    4. **Round-trip settlement** — Funds flow out and return to the same
       account within a short window (self-transfers via intermediaries).
    """

    def __init__(
        self,
        lookback_sec: int = 604800,     # 7 days
        asset_threshold_paisa: int = 5_00_000_00,  # ₹5L
        withdrawal_ratio: float = 0.8,  # withdraw ≥ 80% of recent deposits
        round_trip_window_sec: int = 172800,  # 48h
    ) -> None:
        self._lookback = lookback_sec
        self._asset_threshold = asset_threshold_paisa
        self._withdrawal_ratio = withdrawal_ratio
        self._round_trip_window = round_trip_window_sec
        self.metrics = IntegrationMetrics()

        # Account flow state
        self._inflows: dict[str, list[dict]] = {}   # account → [{amount, ts}]
        self._outflows: dict[str, list[dict]] = {}   # account → [{amount, ts, purpose}]

    # ── Public API ────────────────────────────────────────────────────────

    def record_inflow(self, account_id: str, amount_paisa: int, timestamp: int) -> None:
        """Record an incoming credit to an account."""
        self._inflows.setdefault(account_id, []).append({
            "amount": amount_paisa, "ts": timestamp,
        })

    def evaluate(
        self,
        account_id: str,
        amount_paisa: int,
        timestamp: int,
        purpose_code: str = "",
        is_outflow: bool = True,
    ) -> list[AMLAlert]:
        """Evaluate an outgoing debit for integration-stage indicators."""
        self.metrics.evaluations += 1

        if not is_outflow:
            return []

        # Record outflow
        self._outflows.setdefault(account_id, []).append({
            "amount": amount_paisa, "ts": timestamp,
            "purpose": purpose_code.upper(),
        })

        alerts: list[AMLAlert] = []
        purpose = purpose_code.upper()

        # 1. Asset purchase detection
        if purpose in _ASSET_PURPOSE_CODES and amount_paisa >= self._asset_threshold:
            # Check if preceded by large inflows
            recent_in = self._recent_inflows(account_id, timestamp)
            total_in = sum(d["amount"] for d in recent_in)
            if total_in >= amount_paisa:
                self.metrics.asset_purchase_alerts += 1
                alerts.append(AMLAlert(
                    stage=AMLStage.INTEGRATION,
                    risk=AMLRisk.HIGH,
                    score=0.85,
                    indicator="INTEGRATION_ASSET_PURCHASE",
                    reason=(
                        f"₹{amount_paisa // 100:,} {purpose} purchase after "
                        f"₹{total_in // 100:,} inflows in {self._lookback // 86400}d"
                    ),
                    account_id=account_id,
                    metadata={
                        "purpose": purpose,
                        "amount_paisa": amount_paisa,
                        "recent_inflows_paisa": total_in,
                    },
                ))

        # 2. Investment pattern detection
        if purpose in _INVESTMENT_PURPOSE_CODES and amount_paisa >= self._asset_threshold:
            recent_in = self._recent_inflows(account_id, timestamp)
            total_in = sum(d["amount"] for d in recent_in)
            if total_in >= amount_paisa:
                self.metrics.investment_alerts += 1
                alerts.append(AMLAlert(
                    stage=AMLStage.INTEGRATION,
                    risk=AMLRisk.HIGH,
                    score=0.80,
                    indicator="INTEGRATION_INVESTMENT",
                    reason=(
                        f"₹{amount_paisa // 100:,} {purpose} investment after "
                        f"₹{total_in // 100:,} inflows in {self._lookback // 86400}d"
                    ),
                    account_id=account_id,
                    metadata={
                        "purpose": purpose,
                        "amount_paisa": amount_paisa,
                        "recent_inflows_paisa": total_in,
                    },
                ))

        # 3. Rapid withdrawal (deposit–withdraw churn)
        recent_in = self._recent_inflows(account_id, timestamp)
        total_in = sum(d["amount"] for d in recent_in)
        if total_in > 0 and amount_paisa >= total_in * self._withdrawal_ratio:
            self.metrics.rapid_withdrawal_alerts += 1
            alerts.append(AMLAlert(
                stage=AMLStage.INTEGRATION,
                risk=AMLRisk.MEDIUM,
                score=0.65,
                indicator="INTEGRATION_RAPID_WITHDRAWAL",
                reason=(
                    f"Withdrawal ₹{amount_paisa // 100:,} is ≥ {self._withdrawal_ratio * 100:.0f}% "
                    f"of recent deposits (₹{total_in // 100:,}) — deposit-withdraw churn"
                ),
                account_id=account_id,
                metadata={
                    "withdrawal_paisa": amount_paisa,
                    "deposit_total_paisa": total_in,
                    "ratio": round(amount_paisa / total_in, 2) if total_in else 0,
                },
            ))

        # 4. Round-trip detection (funds returned to same account)
        recent_out = [
            d for d in self._outflows.get(account_id, [])
            if d["ts"] >= timestamp - self._round_trip_window and d["ts"] < timestamp
        ]
        for prior in recent_out:
            # Look for matching inflows that could be the return leg
            matching_returns = [
                d for d in self._inflows.get(account_id, [])
                if d["ts"] > prior["ts"]
                and abs(d["amount"] - prior["amount"]) < prior["amount"] * 0.1
            ]
            if matching_returns:
                self.metrics.round_trip_alerts += 1
                alerts.append(AMLAlert(
                    stage=AMLStage.INTEGRATION,
                    risk=AMLRisk.CRITICAL,
                    score=0.90,
                    indicator="INTEGRATION_ROUND_TRIP",
                    reason=(
                        f"Round-trip detected: ₹{prior['amount'] // 100:,} outflow returned "
                        f"within {self._round_trip_window // 3600}h — possible self-laundering"
                    ),
                    account_id=account_id,
                    metadata={
                        "outflow_paisa": prior["amount"],
                        "return_paisa": matching_returns[0]["amount"],
                    },
                ))
                break  # one round-trip alert per evaluation

        # Prune history older than 2× lookback
        cutoff = timestamp - self._lookback * 2
        self._inflows[account_id] = [
            d for d in self._inflows.get(account_id, []) if d["ts"] >= cutoff
        ]
        self._outflows[account_id] = [
            d for d in self._outflows.get(account_id, []) if d["ts"] >= cutoff
        ]

        return alerts

    # ── Helpers ───────────────────────────────────────────────────────────

    def _recent_inflows(self, account_id: str, timestamp: int) -> list[dict]:
        """Return inflows within the lookback window."""
        cutoff = timestamp - self._lookback
        return [
            d for d in self._inflows.get(account_id, [])
            if d["ts"] >= cutoff
        ]

    def snapshot(self) -> dict:
        return {
            "stage": "INTEGRATION",
            "tracked_accounts": len(self._inflows),
            **self.metrics.snapshot(),
        }

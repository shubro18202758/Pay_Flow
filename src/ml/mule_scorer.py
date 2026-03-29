"""
PayFlow — Mule Account Behavioral Scoring (Carbanak Indicators)
=================================================================
Scores individual bank accounts on four behavioural indicators
characteristic of money mule recruitment, derived from the Carbanak
banking fraud case study (2013–2018):

    MuleScore = 25 · NewlyOpened
              + 25 · HighFrequency
              + 25 · RapidForward
              + 25 · LargeCashOut

Each sub-score is normalised to [0, 1].  Accounts exceeding the
configurable ``mule_threshold`` (default 0.65) are flagged as
suspected mules.

Indicators:
    1. **Newly Opened** — account age < 90 days (score 1.0 if < 30 days,
       linear decay 30–90 days, 0.0 beyond).
    2. **High Frequency** — transactions-per-day exceeds typical retail
       banking norms (>10 txn/day → 1.0, sigmoid ramp 3–10).
    3. **Rapid Forward** — median delay between inbound receipt and
       outbound forwarding is very low (< 5 min → 1.0, linear
       decay 5–60 min, 0.0 beyond).
    4. **Large Cash-Out** — ratio of cash withdrawals to total inbound
       value exceeds 0.80 (full score), linear ramp 0.40–0.80.

Pipeline Position:
    VelocityTracker features  ──┐
    Account metadata           ──┼──▶ MuleAccountScorer.score_account()
    Graph inbound/outbound     ──┘         │
                                      MuleAccountScore
"""

from __future__ import annotations

import logging
import math
from typing import NamedTuple

logger = logging.getLogger(__name__)


# ── Result Types ───────────────────────────────────────────────────────────────


class MuleIndicators(NamedTuple):
    """Individual sub-scores for each Carbanak mule indicator."""

    newly_opened: float     # [0, 1]
    high_frequency: float   # [0, 1]
    rapid_forward: float    # [0, 1]
    large_cashout: float    # [0, 1]


class MuleAccountScore(NamedTuple):
    """Composite mule risk assessment for a single account."""

    account_id: str
    mule_score: float             # [0, 1] weighted composite
    indicators: MuleIndicators
    risk_level: str               # LOW / MEDIUM / HIGH / CRITICAL
    is_suspected_mule: bool


# ── Scorer ─────────────────────────────────────────────────────────────────────


class MuleAccountScorer:
    """
    Score bank accounts on Carbanak-derived mule behavioural indicators.

    Usage::

        scorer = MuleAccountScorer()
        result = scorer.score_account(
            account_id="ACC-12345",
            account_age_days=15,
            txn_count_24h=22,
            median_forward_delay_sec=120,
            total_inbound_paisa=500_000_00,
            total_cashout_paisa=420_000_00,
        )
        if result.is_suspected_mule:
            ...
    """

    # Default weights — equal (25 each)
    _W_NEW = 0.25
    _W_FREQ = 0.25
    _W_FWD = 0.25
    _W_CASH = 0.25

    def __init__(
        self,
        mule_threshold: float = 0.65,
        new_account_high_days: int = 30,
        new_account_low_days: int = 90,
        high_freq_ceiling: int = 10,
        high_freq_floor: int = 3,
        rapid_fwd_high_sec: int = 300,    # 5 min
        rapid_fwd_low_sec: int = 3600,    # 60 min
        cashout_high_ratio: float = 0.80,
        cashout_low_ratio: float = 0.40,
    ) -> None:
        self._threshold = mule_threshold
        self._new_hi = new_account_high_days
        self._new_lo = new_account_low_days
        self._freq_ceil = high_freq_ceiling
        self._freq_floor = high_freq_floor
        self._fwd_hi = rapid_fwd_high_sec
        self._fwd_lo = rapid_fwd_low_sec
        self._cash_hi = cashout_high_ratio
        self._cash_lo = cashout_low_ratio
        self._scored: dict[str, MuleAccountScore] = {}

    # ── Public API ─────────────────────────────────────────────────────────

    def score_account(
        self,
        account_id: str,
        account_age_days: int,
        txn_count_24h: int,
        median_forward_delay_sec: float,
        total_inbound_paisa: int,
        total_cashout_paisa: int,
    ) -> MuleAccountScore:
        """
        Compute composite mule score for a single account.

        Args:
            account_id:               Unique account identifier.
            account_age_days:         Days since account opening.
            txn_count_24h:            Number of transactions in last 24 h.
            median_forward_delay_sec: Median seconds between inbound receipt
                                      and outbound forwarding (0 if no forwarding).
            total_inbound_paisa:      Total inbound value in the window (paisa).
            total_cashout_paisa:      Total cash withdrawals in the window (paisa).

        Returns:
            MuleAccountScore with composite score and breakdown.
        """
        s_new = self._score_newly_opened(account_age_days)
        s_freq = self._score_high_frequency(txn_count_24h)
        s_fwd = self._score_rapid_forward(median_forward_delay_sec)
        s_cash = self._score_large_cashout(total_inbound_paisa, total_cashout_paisa)

        indicators = MuleIndicators(
            newly_opened=round(s_new, 4),
            high_frequency=round(s_freq, 4),
            rapid_forward=round(s_fwd, 4),
            large_cashout=round(s_cash, 4),
        )

        composite = (
            self._W_NEW * s_new
            + self._W_FREQ * s_freq
            + self._W_FWD * s_fwd
            + self._W_CASH * s_cash
        )
        composite = round(composite, 4)

        if composite >= 0.85:
            risk = "CRITICAL"
        elif composite >= 0.65:
            risk = "HIGH"
        elif composite >= 0.40:
            risk = "MEDIUM"
        else:
            risk = "LOW"

        result = MuleAccountScore(
            account_id=account_id,
            mule_score=composite,
            indicators=indicators,
            risk_level=risk,
            is_suspected_mule=composite >= self._threshold,
        )
        self._scored[account_id] = result
        return result

    def get_suspected_mules(self) -> list[MuleAccountScore]:
        """Return all accounts previously scored as suspected mules."""
        return [s for s in self._scored.values() if s.is_suspected_mule]

    def get_score(self, account_id: str) -> MuleAccountScore | None:
        """Retrieve a previously computed score."""
        return self._scored.get(account_id)

    def snapshot(self) -> dict:
        return {
            "threshold": self._threshold,
            "total_scored": len(self._scored),
            "suspected_mules": sum(
                1 for s in self._scored.values() if s.is_suspected_mule
            ),
        }

    # ── Sub-scorers ────────────────────────────────────────────────────────

    def _score_newly_opened(self, age_days: int) -> float:
        """Newly opened accounts are high risk for mule recruitment."""
        if age_days <= self._new_hi:
            return 1.0
        if age_days >= self._new_lo:
            return 0.0
        # Linear decay between thresholds
        return (self._new_lo - age_days) / (self._new_lo - self._new_hi)

    def _score_high_frequency(self, txn_count_24h: int) -> float:
        """High transaction frequency indicates automated or mule behaviour."""
        if txn_count_24h >= self._freq_ceil:
            return 1.0
        if txn_count_24h <= self._freq_floor:
            return 0.0
        # Sigmoid-like smooth ramp using tanh
        mid = (self._freq_ceil + self._freq_floor) / 2.0
        scale = 2.0 / (self._freq_ceil - self._freq_floor)
        return 0.5 + 0.5 * math.tanh(scale * (txn_count_24h - mid))

    def _score_rapid_forward(self, delay_sec: float) -> float:
        """
        Rapid forwarding of received funds — short delay between
        inbound receipt and outbound transfer.
        """
        if delay_sec <= 0:
            return 0.0  # No forwarding activity
        if delay_sec <= self._fwd_hi:
            return 1.0
        if delay_sec >= self._fwd_lo:
            return 0.0
        return (self._fwd_lo - delay_sec) / (self._fwd_lo - self._fwd_hi)

    def _score_large_cashout(
        self, inbound_paisa: int, cashout_paisa: int,
    ) -> float:
        """Large cash withdrawals relative to inbound deposits."""
        if inbound_paisa <= 0 or cashout_paisa <= 0:
            return 0.0
        ratio = cashout_paisa / inbound_paisa
        if ratio >= self._cash_hi:
            return 1.0
        if ratio <= self._cash_lo:
            return 0.0
        return (ratio - self._cash_lo) / (self._cash_hi - self._cash_lo)

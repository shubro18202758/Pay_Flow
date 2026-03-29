"""
PayFlow — Velocity Check Engine
=================================
Detects abnormally high transaction frequencies within tight time windows.

AML Context:
  Velocity checks are the first line of defense in real-time fraud monitoring.
  FATF Recommendation 20 (Suspicious Transaction Reporting) specifically calls
  out "unusual frequency" as a red-flag indicator. Indian FIU-IND guidelines
  mandate banks to flag accounts exceeding 50 transactions/day via digital
  channels or >10 cash deposits/day.

Algorithm:
  Per-account sliding window counters maintained in memory using sorted
  timestamp deques. O(1) amortized insert, O(k) window cleanup where k
  is the number of expired entries. At 100K accounts × 30-day window ×
  avg 50 txns/account = ~5M entries → ~200 MB RAM (timestamps only).

Output Features (per transaction):
  - txn_count_1h    : transactions by sender in last 1 hour
  - txn_count_6h    : transactions by sender in last 6 hours
  - txn_count_24h   : transactions by sender in last 24 hours
  - txn_count_7d    : transactions by sender in last 7 days
  - unique_receivers_1h  : distinct receivers in last 1 hour
  - unique_receivers_24h : distinct receivers in last 24 hours
  - amount_sum_1h   : total paisa sent in last 1 hour
  - amount_sum_24h  : total paisa sent in last 24 hours
  - avg_gap_sec     : mean seconds between consecutive txns (last 24h)
  - min_gap_sec     : minimum seconds between consecutive txns (last 24h)
  - velocity_zscore : z-score of current hourly rate vs 7-day baseline
"""

from __future__ import annotations

import math
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import NamedTuple


# ── Time Windows (seconds) ──────────────────────────────────────────────────

_1H = 3600
_6H = 21600
_24H = 86400
_7D = 604800


# ── Per-Event Velocity Features ─────────────────────────────────────────────

class VelocityFeatures(NamedTuple):
    """11 velocity features per transaction, packed for numpy vectorization."""
    txn_count_1h: int
    txn_count_6h: int
    txn_count_24h: int
    txn_count_7d: int
    unique_receivers_1h: int
    unique_receivers_24h: int
    amount_sum_1h: int       # paisa
    amount_sum_24h: int      # paisa
    avg_gap_sec: float
    min_gap_sec: float
    velocity_zscore: float


# ── Transaction Record (minimal footprint for window storage) ───────────────

@dataclass(slots=True)
class _TxnRecord:
    timestamp: int
    receiver_id: str
    amount_paisa: int


# ── Velocity Tracker ────────────────────────────────────────────────────────

class VelocityTracker:
    """
    Per-account sliding window velocity tracker.

    Maintains sorted deques of recent transactions per sender account.
    All windows are lazily pruned on access — no background timers needed.

    Memory: ~80 bytes per _TxnRecord × avg 50 records/account × 100K accounts
            ≈ 400 MB worst case (single 30-day retention window).
    """

    def __init__(self, retention_sec: int = _7D):
        self._retention = retention_sec
        # account_id → deque of _TxnRecord, sorted by timestamp (append-order)
        self._windows: dict[str, deque[_TxnRecord]] = defaultdict(deque)
        # For z-score: rolling hourly counts over 7 days
        # account_id → deque of (hour_bucket, count)
        self._hourly_counts: dict[str, dict[int, int]] = defaultdict(lambda: defaultdict(int))

    def record_and_extract(
        self,
        sender_id: str,
        receiver_id: str,
        amount_paisa: int,
        timestamp: int,
    ) -> VelocityFeatures:
        """
        Record a transaction and compute velocity features atomically.

        Must be called in timestamp order per sender for correct gap
        calculations. Out-of-order events produce approximate results
        (acceptable for streaming — exact replay uses sorted batches).
        """
        record = _TxnRecord(timestamp, receiver_id, amount_paisa)
        window = self._windows[sender_id]

        # Prune expired entries
        cutoff = timestamp - self._retention
        while window and window[0].timestamp < cutoff:
            window.popleft()

        # Append new record
        window.append(record)

        # Update hourly bucket
        hour_bucket = timestamp // _1H
        self._hourly_counts[sender_id][hour_bucket] += 1
        self._prune_hourly(sender_id, hour_bucket)

        # ── Compute features over the window ────────────────────────────
        t_1h = timestamp - _1H
        t_6h = timestamp - _6H
        t_24h = timestamp - _24H

        count_1h = count_6h = count_24h = count_7d = 0
        receivers_1h: set[str] = set()
        receivers_24h: set[str] = set()
        sum_1h = sum_24h = 0
        timestamps_24h: list[int] = []

        for rec in window:
            count_7d += 1
            if rec.timestamp >= t_24h:
                count_24h += 1
                receivers_24h.add(rec.receiver_id)
                sum_24h += rec.amount_paisa
                timestamps_24h.append(rec.timestamp)
                if rec.timestamp >= t_6h:
                    count_6h += 1
                    if rec.timestamp >= t_1h:
                        count_1h += 1
                        receivers_1h.add(rec.receiver_id)
                        sum_1h += rec.amount_paisa

        # Inter-transaction gaps (last 24h)
        avg_gap = 0.0
        min_gap = float("inf")
        if len(timestamps_24h) >= 2:
            gaps = [
                timestamps_24h[i + 1] - timestamps_24h[i]
                for i in range(len(timestamps_24h) - 1)
            ]
            avg_gap = sum(gaps) / len(gaps)
            min_gap = min(gaps)
        else:
            min_gap = 0.0

        # Z-score: current hourly rate vs 7-day baseline
        zscore = self._compute_velocity_zscore(sender_id, hour_bucket)

        return VelocityFeatures(
            txn_count_1h=count_1h,
            txn_count_6h=count_6h,
            txn_count_24h=count_24h,
            txn_count_7d=count_7d,
            unique_receivers_1h=len(receivers_1h),
            unique_receivers_24h=len(receivers_24h),
            amount_sum_1h=sum_1h,
            amount_sum_24h=sum_24h,
            avg_gap_sec=avg_gap,
            min_gap_sec=min_gap,
            velocity_zscore=zscore,
        )

    def _compute_velocity_zscore(self, sender_id: str, current_bucket: int) -> float:
        """
        Z-score of the current hour's transaction count against the sender's
        historical hourly distribution (last 7 days = 168 hourly buckets).

        Z = (x - μ) / σ  where x = current hour's count

        High z-scores (>3.0) indicate a velocity spike — strong fraud signal.
        """
        hourly = self._hourly_counts[sender_id]
        current_count = hourly.get(current_bucket, 0)

        # Collect all hourly counts including zero-count hours
        min_bucket = current_bucket - (7 * 24)  # 168 hours back
        counts = [
            hourly.get(b, 0)
            for b in range(min_bucket, current_bucket + 1)
        ]

        n = len(counts)
        if n < 2:
            return 0.0

        mean = sum(counts) / n
        variance = sum((c - mean) ** 2 for c in counts) / n
        std = math.sqrt(variance) if variance > 0 else 0.0

        if std < 1e-9:
            return 0.0 if current_count <= mean else 5.0  # cap at 5 if zero-variance

        return (current_count - mean) / std

    def _prune_hourly(self, sender_id: str, current_bucket: int) -> None:
        """Remove hourly buckets older than 7 days."""
        cutoff_bucket = current_bucket - (7 * 24)
        hourly = self._hourly_counts[sender_id]
        stale = [b for b in hourly if b < cutoff_bucket]
        for b in stale:
            del hourly[b]

    def account_count(self) -> int:
        """Number of tracked accounts."""
        return len(self._windows)

    def total_records(self) -> int:
        """Total transaction records across all windows."""
        return sum(len(w) for w in self._windows.values())

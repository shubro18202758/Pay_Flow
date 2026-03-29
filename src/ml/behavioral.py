"""
PayFlow — Behavioral Analytics Engine
========================================
Detects statistical deviations from a user's established behavioral baseline.

AML Context:
  FATF Recommendation 10 (Customer Due Diligence) mandates ongoing monitoring
  of the business relationship, including scrutiny of transactions to ensure
  they are consistent with the institution's knowledge of the customer.
  Behavioral analytics operationalizes this by building per-account statistical
  profiles and flagging deviations.

Three behavioral dimensions tracked:
  1. Temporal Patterns  — login/transaction hour-of-day distribution
  2. Geographic Patterns — usual transaction origination locations
  3. Amount Patterns     — typical transaction value distribution

Algorithm:
  Welford's online algorithm for incremental mean/variance computation.
  O(1) per update, numerically stable for large sequences. For geographic
  deviation we maintain a centroid + max-radius model and compute Haversine
  distance from the centroid for each new event.

Output Features (per transaction):
  - hour_deviation     : |current_hour - mean_hour| / std_hour
  - is_off_hours       : 1 if between 23:00–05:00 local, else 0
  - geo_distance_km    : Haversine km from user's centroid location
  - geo_deviation      : geo_distance_km / historical_max_radius_km
  - amount_zscore      : (amount - mean) / std  of sender's history
  - amount_percentile  : approximate CDF rank (0.0–1.0) in sender history
  - login_fail_rate_7d : failed_logins / total_logins in last 7 days
  - unique_ips_24h     : distinct IP addresses in last 24 hours
  - device_change_flag : 1 if device_fingerprint differs from last known
  - session_anomaly    : combined score of temporal + geo + amount deviation
"""

from __future__ import annotations

import math
from collections import defaultdict, deque
from typing import NamedTuple


_24H = 86400
_7D = 604800


# ── Output Feature Vector ──────────────────────────────────────────────────

class BehavioralFeatures(NamedTuple):
    """10 behavioral features per event, packed for numpy vectorization."""
    hour_deviation: float
    is_off_hours: int
    geo_distance_km: float
    geo_deviation: float
    amount_zscore: float
    amount_percentile: float
    login_fail_rate_7d: float
    unique_ips_24h: int
    device_change_flag: int
    session_anomaly: float


# ── Welford Online Statistics ───────────────────────────────────────────────

class _WelfordStats:
    """
    Welford's online algorithm for incremental mean and variance.
    Numerically stable, O(1) per update, zero memory growth.
    """
    __slots__ = ("n", "mean", "m2")

    def __init__(self) -> None:
        self.n: int = 0
        self.mean: float = 0.0
        self.m2: float = 0.0

    def update(self, x: float) -> None:
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        delta2 = x - self.mean
        self.m2 += delta * delta2

    @property
    def variance(self) -> float:
        return self.m2 / self.n if self.n > 1 else 0.0

    @property
    def std(self) -> float:
        return math.sqrt(self.variance)

    def zscore(self, x: float) -> float:
        s = self.std
        if s < 1e-9:
            return 0.0 if abs(x - self.mean) < 1e-9 else 5.0
        return (x - self.mean) / s


# ── Haversine Distance ──────────────────────────────────────────────────────

def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Great-circle distance between two points on Earth (km).
    Optimized: avoids trig calls when points are identical.
    """
    if lat1 == lat2 and lon1 == lon2:
        return 0.0

    r = 6371.0  # Earth radius km
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(math.radians(lat1))
        * math.cos(math.radians(lat2))
        * math.sin(dlon / 2) ** 2
    )
    return r * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


# ── Per-Account Behavioral Profile ─────────────────────────────────────────

class _AccountProfile:
    """
    Running behavioral statistics for a single account.

    Memory: ~400 bytes base + O(k) for IP/device deques where k ≤ 168
    (hourly buckets in 7 days). At 100K accounts → ~40 MB.
    """
    __slots__ = (
        "hour_stats", "amount_stats",
        "geo_centroid_lat", "geo_centroid_lon", "geo_max_radius_km", "geo_n",
        "last_device_fingerprint", "last_timestamp",
        "login_attempts_7d", "login_failures_7d",
        "recent_ips",
        "amount_sorted_sample",
    )

    def __init__(self) -> None:
        self.hour_stats = _WelfordStats()
        self.amount_stats = _WelfordStats()

        # Geographic centroid (incremental mean)
        self.geo_centroid_lat: float = 0.0
        self.geo_centroid_lon: float = 0.0
        self.geo_max_radius_km: float = 0.0
        self.geo_n: int = 0

        # Device tracking
        self.last_device_fingerprint: str = ""
        self.last_timestamp: int = 0

        # Auth tracking (deque of (timestamp, is_failure))
        self.login_attempts_7d: deque[tuple[int, bool]] = deque()

        # IP tracking (deque of (timestamp, ip))
        self.recent_ips: deque[tuple[int, str]] = deque()

        # Approximate CDF via reservoir sample (capped at 500 for memory)
        self.amount_sorted_sample: list[int] = []

    def update_geo(self, lat: float, lon: float) -> tuple[float, float]:
        """
        Update geographic centroid and return (distance_km, deviation).
        Centroid uses incremental mean. Radius expands monotonically.
        """
        if lat == 0.0 and lon == 0.0:
            # Sentinel for "location unavailable" — skip geo update
            return 0.0, 0.0

        if self.geo_n == 0:
            self.geo_centroid_lat = lat
            self.geo_centroid_lon = lon
            self.geo_n = 1
            return 0.0, 0.0

        dist = _haversine_km(self.geo_centroid_lat, self.geo_centroid_lon, lat, lon)

        # Update centroid (incremental mean)
        self.geo_n += 1
        self.geo_centroid_lat += (lat - self.geo_centroid_lat) / self.geo_n
        self.geo_centroid_lon += (lon - self.geo_centroid_lon) / self.geo_n

        # Expand max radius
        if dist > self.geo_max_radius_km:
            self.geo_max_radius_km = dist

        # Deviation: ratio of this distance to historical max
        deviation = dist / self.geo_max_radius_km if self.geo_max_radius_km > 0.1 else 0.0
        return dist, deviation

    def update_auth(self, timestamp: int, is_failure: bool, ip: str) -> None:
        """Record a login attempt for fail-rate and IP tracking."""
        self.login_attempts_7d.append((timestamp, is_failure))
        self.recent_ips.append((timestamp, ip))

        # Prune entries older than 7 days
        cutoff_7d = timestamp - _7D
        while self.login_attempts_7d and self.login_attempts_7d[0][0] < cutoff_7d:
            self.login_attempts_7d.popleft()

        cutoff_24h = timestamp - _24H
        while self.recent_ips and self.recent_ips[0][0] < cutoff_24h:
            self.recent_ips.popleft()

    def get_login_fail_rate(self) -> float:
        """Failed login ratio over last 7 days."""
        if not self.login_attempts_7d:
            return 0.0
        failures = sum(1 for _, f in self.login_attempts_7d if f)
        return failures / len(self.login_attempts_7d)

    def get_unique_ips_24h(self) -> int:
        """Distinct IPs seen in last 24 hours."""
        return len({ip for _, ip in self.recent_ips})

    def update_amount_sample(self, amount_paisa: int) -> float:
        """
        Update the sorted sample and return approximate percentile.
        Uses bisect for O(log n) insertion, capped at 500 entries.
        """
        import bisect
        bisect.insort(self.amount_sorted_sample, amount_paisa)

        # Cap reservoir at 500 (downsample by removing middle entries)
        if len(self.amount_sorted_sample) > 500:
            self.amount_sorted_sample.pop(len(self.amount_sorted_sample) // 2)

        # Approximate CDF rank
        idx = bisect.bisect_left(self.amount_sorted_sample, amount_paisa)
        return idx / len(self.amount_sorted_sample)


# ── Behavioral Analyzer ────────────────────────────────────────────────────

class BehavioralAnalyzer:
    """
    Stateful per-account behavioral profiler.

    Maintains incremental statistics for each account and computes
    deviation features relative to the account's own baseline.

    Thread-safety: single-writer (pipeline validator runs in one task).
    """

    def __init__(self) -> None:
        self._profiles: dict[str, _AccountProfile] = defaultdict(_AccountProfile)

    def analyze_transaction(
        self,
        sender_id: str,
        amount_paisa: int,
        timestamp: int,
        sender_lat: float,
        sender_lon: float,
        device_fingerprint: str,
    ) -> BehavioralFeatures:
        """Extract behavioral deviation features for a transaction event."""
        profile = self._profiles[sender_id]

        # ── Temporal deviation ──────────────────────────────────────────
        hour = (timestamp % _24H) / 3600  # fractional hour of day (0.0 – 23.99)
        profile.hour_stats.update(hour)
        hour_dev = profile.hour_stats.zscore(hour) if profile.hour_stats.n > 5 else 0.0
        is_off_hours = 1 if (hour >= 23.0 or hour < 5.0) else 0

        # ── Geographic deviation ────────────────────────────────────────
        geo_dist, geo_dev = profile.update_geo(sender_lat, sender_lon)

        # ── Amount deviation ────────────────────────────────────────────
        profile.amount_stats.update(float(amount_paisa))
        amt_z = profile.amount_stats.zscore(float(amount_paisa)) if profile.amount_stats.n > 5 else 0.0
        amt_pct = profile.update_amount_sample(amount_paisa)

        # ── Device change ───────────────────────────────────────────────
        device_changed = 0
        if profile.last_device_fingerprint and profile.last_device_fingerprint != device_fingerprint:
            device_changed = 1
        profile.last_device_fingerprint = device_fingerprint
        profile.last_timestamp = timestamp

        # ── Auth-derived features (use stored state) ────────────────────
        fail_rate = profile.get_login_fail_rate()
        unique_ips = profile.get_unique_ips_24h()

        # ── Composite anomaly score ─────────────────────────────────────
        # Weighted combination: temporal (0.2) + geo (0.3) + amount (0.35) + device (0.15)
        session_anomaly = (
            0.20 * min(abs(hour_dev) / 5.0, 1.0)
            + 0.30 * min(geo_dev, 1.0)
            + 0.35 * min(abs(amt_z) / 5.0, 1.0)
            + 0.15 * device_changed
        )

        return BehavioralFeatures(
            hour_deviation=round(hour_dev, 4),
            is_off_hours=is_off_hours,
            geo_distance_km=round(geo_dist, 2),
            geo_deviation=round(geo_dev, 4),
            amount_zscore=round(amt_z, 4),
            amount_percentile=round(amt_pct, 4),
            login_fail_rate_7d=round(fail_rate, 4),
            unique_ips_24h=unique_ips,
            device_change_flag=device_changed,
            session_anomaly=round(session_anomaly, 4),
        )

    def record_auth_event(
        self,
        account_id: str,
        timestamp: int,
        success: bool,
        ip_address: str,
        device_fingerprint: str,
    ) -> None:
        """
        Update the behavioral profile with an auth event.
        Auth events don't produce features directly — they enrich the profile
        so that the next transaction from this account reflects login patterns.
        """
        profile = self._profiles[account_id]
        profile.update_auth(timestamp, not success, ip_address)

        if profile.last_device_fingerprint and profile.last_device_fingerprint != device_fingerprint:
            profile.last_device_fingerprint = device_fingerprint

    def account_count(self) -> int:
        return len(self._profiles)

"""
PayFlow — Device Fingerprint Verification
============================================
Tracks known device fingerprints per account and flags transactions
originating from unrecognised or suspicious devices.

Device risk signals:
  1. Unknown device → first time this fingerprint is used with this account.
  2. Device velocity → many accounts using the same device fingerprint
     (shared device / emulator farm).
  3. Recent device change → new fingerprint appeared after a long dormant
     period on the account.
  4. Device hopping → multiple different fingerprints within a short window
     for the same account.

Integration:
  - Called by PreApprovalGate during pre-authorisation checks.
  - Ingests Transaction.device_fingerprint (16-char truncated SHA-256).
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from typing import NamedTuple

logger = logging.getLogger(__name__)


class DeviceRiskResult(NamedTuple):
    """Result of device fingerprint verification."""
    account_id: str
    device_fingerprint: str
    is_known_device: bool
    device_risk_score: float       # ∈ [0.0, 1.0]
    risk_factors: list[str]        # human-readable reasons
    accounts_on_device: int        # how many accounts share this fingerprint
    devices_on_account: int        # how many fingerprints this account has used
    evaluation_time_ms: float


class DeviceVerifier:
    """
    Maintains per-account device fingerprint history and scores
    device-based risk.

    Parameters
    ----------
    max_known_devices : int
        Maximum stored fingerprints per account before oldest are evicted.
    shared_device_threshold : int
        Number of distinct accounts using the same device fingerprint
        before it is flagged as suspicious (emulator / shared device).
    hop_window_seconds : int
        Time window within which multiple different devices for the same
        account triggers a "device hopping" alert.
    hop_count_threshold : int
        Minimum distinct devices within ``hop_window_seconds`` to flag
        device hopping.
    """

    def __init__(
        self,
        max_known_devices: int = 10,
        shared_device_threshold: int = 5,
        hop_window_seconds: int = 3600,
        hop_count_threshold: int = 3,
    ) -> None:
        self._max_known = max_known_devices
        self._shared_threshold = shared_device_threshold
        self._hop_window = hop_window_seconds
        self._hop_count = hop_count_threshold

        # account_id → list[(fingerprint, timestamp)]  (ordered by time)
        self._account_devices: dict[str, list[tuple[str, int]]] = defaultdict(list)
        # device_fingerprint → set of account_ids
        self._device_accounts: dict[str, set[str]] = defaultdict(set)

    # ── Public API ────────────────────────────────────────────────────────────

    def verify(
        self,
        account_id: str,
        device_fingerprint: str,
        timestamp: int,
    ) -> DeviceRiskResult:
        """
        Verify a device fingerprint against the account's known devices
        and return a risk assessment.
        """
        t0 = time.perf_counter()
        risk_factors: list[str] = []

        history = self._account_devices[account_id]
        known_fps = {fp for fp, _ in history}
        is_known = device_fingerprint in known_fps

        # 1. Unknown device
        if not is_known:
            if history:
                risk_factors.append("new_device_for_account")
            else:
                # First ever transaction — lower risk
                risk_factors.append("first_device_for_account")

        # 2. Shared device (many accounts on same fingerprint)
        self._device_accounts[device_fingerprint].add(account_id)
        accounts_on_device = len(self._device_accounts[device_fingerprint])
        if accounts_on_device >= self._shared_threshold:
            risk_factors.append(
                f"shared_device_{accounts_on_device}_accounts"
            )

        # 3. Device hopping (many devices in short window)
        cutoff = timestamp - self._hop_window
        recent_fps = {fp for fp, ts in history if ts >= cutoff}
        recent_fps.add(device_fingerprint)
        devices_in_window = len(recent_fps)
        if devices_in_window >= self._hop_count:
            risk_factors.append(
                f"device_hopping_{devices_in_window}_devices_in_window"
            )

        # 4. Dormant account reactivation with new device
        if history and not is_known:
            last_ts = history[-1][1]
            dormancy_days = (timestamp - last_ts) / 86400
            if dormancy_days > 90:
                risk_factors.append(
                    f"dormant_reactivation_{int(dormancy_days)}d_new_device"
                )

        # Record the device usage
        self._record(account_id, device_fingerprint, timestamp)

        # Compute composite risk score
        score = self._compute_score(
            is_known=is_known,
            accounts_on_device=accounts_on_device,
            devices_in_window=devices_in_window,
            risk_factors=risk_factors,
        )

        total_devices = len({fp for fp, _ in self._account_devices[account_id]})

        return DeviceRiskResult(
            account_id=account_id,
            device_fingerprint=device_fingerprint,
            is_known_device=is_known,
            device_risk_score=round(score, 4),
            risk_factors=risk_factors,
            accounts_on_device=accounts_on_device,
            devices_on_account=total_devices,
            evaluation_time_ms=(time.perf_counter() - t0) * 1000,
        )

    def is_known_device(self, account_id: str, device_fingerprint: str) -> bool:
        """Quick check: has this account ever used this fingerprint?"""
        return any(
            fp == device_fingerprint
            for fp, _ in self._account_devices.get(account_id, [])
        )

    def known_devices(self, account_id: str) -> list[str]:
        """Return list of known fingerprints for an account."""
        return list({fp for fp, _ in self._account_devices.get(account_id, [])})

    @property
    def total_accounts_tracked(self) -> int:
        return len(self._account_devices)

    @property
    def total_devices_tracked(self) -> int:
        return len(self._device_accounts)

    # ── Internal ──────────────────────────────────────────────────────────────

    def _record(
        self,
        account_id: str,
        device_fingerprint: str,
        timestamp: int,
    ) -> None:
        """Record a device usage event for the account."""
        history = self._account_devices[account_id]
        history.append((device_fingerprint, timestamp))

        # Evict oldest if over limit
        if len(history) > self._max_known * 3:
            # Keep last max_known unique fingerprints' events
            seen: dict[str, list[tuple[str, int]]] = {}
            for entry in reversed(history):
                fp = entry[0]
                if fp not in seen:
                    seen[fp] = []
                if len(seen) <= self._max_known:
                    seen[fp].append(entry)
            kept: list[tuple[str, int]] = []
            for entries in seen.values():
                kept.extend(entries)
            kept.sort(key=lambda e: e[1])
            self._account_devices[account_id] = kept

    def _compute_score(
        self,
        is_known: bool,
        accounts_on_device: int,
        devices_in_window: int,
        risk_factors: list[str],
    ) -> float:
        """
        Composite device risk score ∈ [0.0, 1.0].

        Heuristic weighting:
          - Unknown device:  +0.30
          - Shared device:   +0.25 (scaled by how far over threshold)
          - Device hopping:  +0.25
          - Dormant reactivation: +0.20
          - First device:    only +0.05 (low baseline risk)
        """
        score = 0.0

        if "first_device_for_account" in risk_factors:
            score += 0.05
        elif not is_known:
            score += 0.30

        if accounts_on_device >= self._shared_threshold:
            excess = min(accounts_on_device / (self._shared_threshold * 2), 1.0)
            score += 0.25 * excess

        if devices_in_window >= self._hop_count:
            excess = min(devices_in_window / (self._hop_count * 2), 1.0)
            score += 0.25 * excess

        if any("dormant_reactivation" in f for f in risk_factors):
            score += 0.20

        return min(1.0, score)

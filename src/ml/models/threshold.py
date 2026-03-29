"""
PayFlow — Dynamic Anomaly Scoring Threshold
=============================================
Adaptive threshold mechanism that adjusts the fraud/legitimate decision
boundary based on the live score distribution.

Why Dynamic?
  Static thresholds (e.g., score > 0.5 = fraud) fail in production because:
  1. Model calibration drifts as transaction patterns change seasonally
  2. Fraud prevalence shifts over time (new modus operandi emerge)
  3. A fixed cutoff cannot balance precision/recall across regimes

Algorithm — Exponential Moving Average (EMA) Threshold:
  The threshold tracks the running mean and standard deviation of risk scores
  using an EMA with configurable half-life. "High risk" is defined as:

      threshold = μ_ema + k × σ_ema

  where k is a sensitivity multiplier (default 2.0 for ~95th percentile under
  Gaussian assumptions). The threshold is clamped to [floor, ceiling] to
  prevent pathological collapse.

  This automatically adapts to score distribution changes without retraining
  the underlying XGBoost model.

Tier System:
  Scores are classified into three risk tiers:
    - HIGH   : score ≥ threshold              → route to Graph + LLM
    - MEDIUM : score ≥ medium_factor × threshold → flag for review
    - LOW    : score < medium_factor × threshold → pass through
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import NamedTuple

import numpy as np

logger = logging.getLogger(__name__)


# ── Risk Tier ────────────────────────────────────────────────────────────────

class RiskTier:
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


class ThresholdResult(NamedTuple):
    """Threshold evaluation result for a single transaction."""
    risk_score: float
    tier: str           # RiskTier value
    threshold: float    # current dynamic threshold at evaluation time
    exceeds_threshold: bool


# ── Threshold Configuration ──────────────────────────────────────────────────

@dataclass(frozen=True)
class ThresholdConfig:
    """Tunable parameters for the dynamic threshold engine."""
    sensitivity_k: float = 2.0       # σ multiplier (2.0 ≈ 95th percentile)
    ema_alpha: float = 0.01          # EMA smoothing factor (lower = slower adapt)
    floor: float = 0.50              # minimum threshold (never below 50%)
    ceiling: float = 0.95            # maximum threshold (never above 95%)
    medium_factor: float = 0.70      # medium tier = 70% of high threshold
    warmup_samples: int = 100        # use static floor until this many scores seen
    initial_threshold: float = 0.85  # default until warmup completes


# ── Dynamic Threshold Engine ─────────────────────────────────────────────────

class DynamicThreshold:
    """
    Adaptive fraud threshold using exponential moving averages of the
    risk score distribution.

    Thread-safety: single-writer (pipeline inference runs in one task).

    Usage:
        threshold = DynamicThreshold()

        for score in risk_scores:
            result = threshold.evaluate(score)
            if result.exceeds_threshold:
                route_to_deep_investigation(...)
    """

    def __init__(self, config: ThresholdConfig | None = None) -> None:
        self._cfg = config or ThresholdConfig()
        self._n: int = 0
        self._ema_mean: float = 0.0
        self._ema_var: float = 0.0
        self._current_threshold: float = self._cfg.initial_threshold

    # ── Core API ──────────────────────────────────────────────────────────

    def evaluate(self, risk_score: float) -> ThresholdResult:
        """
        Evaluate a single risk score against the adaptive threshold.

        Updates the EMA statistics and returns the tier classification.
        """
        self._update_ema(risk_score)
        threshold = self._current_threshold
        medium_thresh = threshold * self._cfg.medium_factor

        if risk_score >= threshold:
            tier = RiskTier.HIGH
        elif risk_score >= medium_thresh:
            tier = RiskTier.MEDIUM
        else:
            tier = RiskTier.LOW

        return ThresholdResult(
            risk_score=risk_score,
            tier=tier,
            threshold=threshold,
            exceeds_threshold=risk_score >= threshold,
        )

    def evaluate_batch(self, risk_scores: np.ndarray) -> list[ThresholdResult]:
        """Evaluate an array of risk scores. Updates EMA per score."""
        return [self.evaluate(float(s)) for s in risk_scores]

    # ── EMA Update ────────────────────────────────────────────────────────

    def _update_ema(self, score: float) -> None:
        """Update exponential moving average of mean and variance."""
        self._n += 1
        alpha = self._cfg.ema_alpha

        if self._n == 1:
            self._ema_mean = score
            self._ema_var = 0.0
            return

        # EMA mean
        delta = score - self._ema_mean
        self._ema_mean += alpha * delta

        # EMA variance (Welford-like incremental)
        self._ema_var = (1 - alpha) * (self._ema_var + alpha * delta * delta)

        # Recompute threshold after warmup
        if self._n >= self._cfg.warmup_samples:
            std = math.sqrt(self._ema_var) if self._ema_var > 0 else 0.0
            raw = self._ema_mean + self._cfg.sensitivity_k * std
            self._current_threshold = max(
                self._cfg.floor,
                min(raw, self._cfg.ceiling),
            )

    # ── Diagnostics ───────────────────────────────────────────────────────

    @property
    def current_threshold(self) -> float:
        return self._current_threshold

    @property
    def samples_seen(self) -> int:
        return self._n

    @property
    def is_warmed_up(self) -> bool:
        return self._n >= self._cfg.warmup_samples

    def snapshot(self) -> dict:
        """Current state for monitoring dashboards."""
        return {
            "threshold": round(self._current_threshold, 4),
            "ema_mean": round(self._ema_mean, 4),
            "ema_std": round(math.sqrt(self._ema_var) if self._ema_var > 0 else 0.0, 4),
            "samples_seen": self._n,
            "warmed_up": self.is_warmed_up,
            "medium_threshold": round(
                self._current_threshold * self._cfg.medium_factor, 4
            ),
        }

    def reset(self) -> None:
        """Reset all state (e.g., after model retraining)."""
        self._n = 0
        self._ema_mean = 0.0
        self._ema_var = 0.0
        self._current_threshold = self._cfg.initial_threshold
        logger.info("DynamicThreshold reset to initial state.")

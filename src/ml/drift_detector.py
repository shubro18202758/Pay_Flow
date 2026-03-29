"""
PayFlow — Model Drift Detection Engine
========================================
Monitors ML model prediction distributions over time to detect
concept drift, data drift, and model staleness. Uses statistical
tests (PSI, KS-test, JS-divergence) to signal when model
performance may be degrading.

Implements Population Stability Index (PSI) for production
monitoring and automatic retraining recommendations.
"""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class DriftSeverity(str, Enum):
    NONE = "none"
    LOW = "low"          # PSI 0.1-0.2: minor shift
    MODERATE = "moderate"  # PSI 0.2-0.25: noticeable drift
    HIGH = "high"        # PSI > 0.25: significant drift, retrain recommended
    CRITICAL = "critical"  # PSI > 0.5: model is stale, urgent retrain


@dataclass
class DriftReport:
    """Single drift assessment report."""
    timestamp: float
    psi_score: float
    ks_statistic: float
    ks_p_value: float
    js_divergence: float
    severity: DriftSeverity
    window_size: int
    reference_mean: float
    current_mean: float
    reference_std: float
    current_std: float
    recommendation: str


@dataclass
class FeatureDriftReport:
    """Per-feature drift assessment."""
    feature_name: str
    psi_score: float
    severity: DriftSeverity
    reference_mean: float
    current_mean: float
    shift_direction: str  # "higher" | "lower" | "stable"


class ModelDriftDetector:
    """
    Continuous model drift monitor using sliding windows and
    statistical divergence tests.

    Maintains a reference distribution (from training) and compares
    against a rolling window of recent predictions. Triggers alerts
    when statistical tests exceed configurable thresholds.
    """

    PSI_THRESHOLDS = {
        0.10: DriftSeverity.NONE,
        0.20: DriftSeverity.LOW,
        0.25: DriftSeverity.MODERATE,
        0.50: DriftSeverity.HIGH,
    }  # >= 0.50 -> CRITICAL

    def __init__(
        self,
        reference_window: int = 2000,
        detection_window: int = 500,
        num_bins: int = 10,
        check_interval_sec: float = 60.0,
    ):
        self._reference_window = reference_window
        self._detection_window = detection_window
        self._num_bins = num_bins
        self._check_interval = check_interval_sec

        # Score buffers
        self._reference_scores: Optional[np.ndarray] = None
        self._current_scores: deque[float] = deque(maxlen=detection_window)
        self._all_scores: deque[float] = deque(maxlen=reference_window * 2)

        # Feature buffers for feature-level drift
        self._reference_features: Optional[np.ndarray] = None
        self._current_features: deque[np.ndarray] = deque(maxlen=detection_window)

        # Reports
        self._reports: deque[DriftReport] = deque(maxlen=100)
        self._feature_reports: list[FeatureDriftReport] = []
        self._last_check: float = 0.0

        # Metrics
        self._checks_performed: int = 0
        self._alerts_raised: int = 0
        self._current_severity: DriftSeverity = DriftSeverity.NONE

    def set_reference(
        self,
        scores: np.ndarray,
        features: Optional[np.ndarray] = None,
    ) -> None:
        """Set the reference distribution from training/validation data."""
        self._reference_scores = scores.copy()
        if features is not None:
            self._reference_features = features.copy()
        logger.info(
            "Drift detector: reference set (%d scores, mean=%.4f, std=%.4f)",
            len(scores), scores.mean(), scores.std(),
        )

    def record_prediction(
        self,
        score: float,
        features: Optional[np.ndarray] = None,
    ) -> None:
        """Record a single prediction score for drift monitoring."""
        self._current_scores.append(score)
        self._all_scores.append(score)
        if features is not None:
            self._current_features.append(features)

    def record_batch(
        self,
        scores: np.ndarray,
        features: Optional[np.ndarray] = None,
    ) -> None:
        """Record a batch of prediction scores."""
        for s in scores:
            self._current_scores.append(float(s))
            self._all_scores.append(float(s))
        if features is not None:
            for i in range(features.shape[0]):
                self._current_features.append(features[i])

    def check_drift(self, force: bool = False) -> Optional[DriftReport]:
        """
        Run drift detection if enough time has elapsed.

        Returns a DriftReport if drift is detected, None otherwise.
        """
        now = time.time()
        if not force and (now - self._last_check) < self._check_interval:
            return None

        if self._reference_scores is None:
            return None

        if len(self._current_scores) < self._detection_window // 2:
            return None

        self._last_check = now
        self._checks_performed += 1

        current = np.array(list(self._current_scores))
        reference = self._reference_scores

        # PSI
        psi = self._compute_psi(reference, current)

        # KS test
        ks_stat, ks_p = self._compute_ks(reference, current)

        # JS divergence
        js_div = self._compute_js_divergence(reference, current)

        # Determine severity
        severity = self._classify_severity(psi)

        # Build recommendation
        recommendation = self._build_recommendation(severity, psi, ks_p)

        report = DriftReport(
            timestamp=now,
            psi_score=round(psi, 6),
            ks_statistic=round(ks_stat, 6),
            ks_p_value=round(ks_p, 6),
            js_divergence=round(js_div, 6),
            severity=severity,
            window_size=len(current),
            reference_mean=round(float(reference.mean()), 6),
            current_mean=round(float(current.mean()), 6),
            reference_std=round(float(reference.std()), 6),
            current_std=round(float(current.std()), 6),
            recommendation=recommendation,
        )

        self._reports.append(report)
        self._current_severity = severity

        if severity in (DriftSeverity.HIGH, DriftSeverity.CRITICAL):
            self._alerts_raised += 1
            logger.warning(
                "MODEL DRIFT DETECTED: PSI=%.4f, severity=%s — %s",
                psi, severity.value, recommendation,
            )

        # Feature-level drift
        if self._reference_features is not None and len(self._current_features) > 50:
            self._check_feature_drift()

        return report

    def _compute_psi(self, reference: np.ndarray, current: np.ndarray) -> float:
        """Compute Population Stability Index."""
        eps = 1e-6
        min_val = min(reference.min(), current.min())
        max_val = max(reference.max(), current.max())

        bins = np.linspace(min_val - eps, max_val + eps, self._num_bins + 1)
        ref_hist, _ = np.histogram(reference, bins=bins)
        cur_hist, _ = np.histogram(current, bins=bins)

        ref_pct = (ref_hist + eps) / (ref_hist.sum() + eps * len(ref_hist))
        cur_pct = (cur_hist + eps) / (cur_hist.sum() + eps * len(cur_hist))

        psi = float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))
        return max(psi, 0.0)

    def _compute_ks(self, reference: np.ndarray, current: np.ndarray) -> tuple[float, float]:
        """Compute Kolmogorov-Smirnov test statistic."""
        try:
            from scipy import stats
            result = stats.ks_2samp(reference, current)
            return float(result.statistic), float(result.pvalue)
        except ImportError:
            # Manual KS computation
            all_data = np.concatenate([reference, current])
            all_data.sort()
            n1, n2 = len(reference), len(current)
            cdf1 = np.searchsorted(np.sort(reference), all_data, side='right') / n1
            cdf2 = np.searchsorted(np.sort(current), all_data, side='right') / n2
            ks_stat = float(np.max(np.abs(cdf1 - cdf2)))
            # Approximate p-value
            n_eff = (n1 * n2) / (n1 + n2)
            p_val = np.exp(-2.0 * n_eff * ks_stat ** 2)
            return ks_stat, float(min(p_val, 1.0))

    def _compute_js_divergence(self, reference: np.ndarray, current: np.ndarray) -> float:
        """Compute Jensen-Shannon divergence."""
        eps = 1e-6
        min_val = min(reference.min(), current.min())
        max_val = max(reference.max(), current.max())

        bins = np.linspace(min_val - eps, max_val + eps, self._num_bins + 1)
        ref_hist, _ = np.histogram(reference, bins=bins, density=True)
        cur_hist, _ = np.histogram(current, bins=bins, density=True)

        ref_p = (ref_hist + eps) / (ref_hist.sum() + eps * len(ref_hist))
        cur_p = (cur_hist + eps) / (cur_hist.sum() + eps * len(cur_hist))
        m = 0.5 * (ref_p + cur_p)

        kl_ref = float(np.sum(ref_p * np.log(ref_p / m)))
        kl_cur = float(np.sum(cur_p * np.log(cur_p / m)))

        return 0.5 * (kl_ref + kl_cur)

    def _classify_severity(self, psi: float) -> DriftSeverity:
        if psi >= 0.50:
            return DriftSeverity.CRITICAL
        if psi >= 0.25:
            return DriftSeverity.HIGH
        if psi >= 0.20:
            return DriftSeverity.MODERATE
        if psi >= 0.10:
            return DriftSeverity.LOW
        return DriftSeverity.NONE

    def _build_recommendation(self, severity: DriftSeverity, psi: float, ks_p: float) -> str:
        if severity == DriftSeverity.CRITICAL:
            return "URGENT: Model is stale. Immediate retraining required with recent data."
        if severity == DriftSeverity.HIGH:
            return "Model drift detected. Schedule retraining within 24 hours."
        if severity == DriftSeverity.MODERATE:
            return "Minor drift observed. Monitor closely; consider retraining."
        if severity == DriftSeverity.LOW:
            return "Slight distribution shift. Continue monitoring."
        return "No significant drift detected. Model is performing within expected bounds."

    def _check_feature_drift(self) -> None:
        """Check drift at the feature level."""
        if self._reference_features is None:
            return

        current_features = np.array(list(self._current_features))
        if current_features.ndim != 2:
            return

        from src.ml.feature_engine import FEATURE_COLUMNS
        names = FEATURE_COLUMNS

        self._feature_reports = []
        for i in range(min(current_features.shape[1], len(names))):
            ref_col = self._reference_features[:, i]
            cur_col = current_features[:, i]
            psi = self._compute_psi(ref_col, cur_col)
            sev = self._classify_severity(psi)

            ref_mean = float(ref_col.mean())
            cur_mean = float(cur_col.mean())
            if cur_mean > ref_mean * 1.1:
                shift = "higher"
            elif cur_mean < ref_mean * 0.9:
                shift = "lower"
            else:
                shift = "stable"

            self._feature_reports.append(FeatureDriftReport(
                feature_name=names[i] if i < len(names) else f"f{i}",
                psi_score=round(psi, 6),
                severity=sev,
                reference_mean=round(ref_mean, 6),
                current_mean=round(cur_mean, 6),
                shift_direction=shift,
            ))

    def get_latest_report(self) -> Optional[DriftReport]:
        return self._reports[-1] if self._reports else None

    def get_drifting_features(self) -> list[FeatureDriftReport]:
        """Return features with detected drift (severity > NONE)."""
        return [f for f in self._feature_reports if f.severity != DriftSeverity.NONE]

    def snapshot(self) -> dict:
        latest = self.get_latest_report()
        drifting = self.get_drifting_features()
        return {
            "checks_performed": self._checks_performed,
            "alerts_raised": self._alerts_raised,
            "current_severity": self._current_severity.value,
            "reference_size": len(self._reference_scores) if self._reference_scores is not None else 0,
            "current_window_size": len(self._current_scores),
            "latest_psi": latest.psi_score if latest else None,
            "latest_ks": latest.ks_statistic if latest else None,
            "latest_js_divergence": latest.js_divergence if latest else None,
            "latest_recommendation": latest.recommendation if latest else None,
            "drifting_features_count": len(drifting),
            "drifting_features": [
                {"feature": f.feature_name, "psi": f.psi_score, "severity": f.severity.value}
                for f in drifting[:10]
            ],
            "score_history": [
                {"timestamp": r.timestamp, "psi": r.psi_score, "severity": r.severity.value}
                for r in list(self._reports)[-20:]
            ],
        }

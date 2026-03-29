"""
PayFlow — Isolation Forest Anomaly Detector
==============================================
Unsupervised anomaly detection using scikit-learn's Isolation Forest
algorithm.  Operates on the FeatureEngine's 30-dim (or 36-dim with
network features) float32 vectors.

Isolation Forest isolates anomalies by randomly selecting a feature and
a split value between the minimum and maximum of the selected feature.
Anomalous points require fewer splits to isolate, yielding lower anomaly
scores.

Integration:
    detector = IsolationDetector()
    detector.fit(feature_matrix)                 # train on accumulated data
    result   = detector.score(feature_vector)     # per-transaction scoring
    bulk     = detector.score_batch(matrix)        # batch scoring

Memory Budget:
    ~50 MB for 100 estimators with max_samples=2048 — well within budget.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import NamedTuple

import numpy as np

logger = logging.getLogger(__name__)

# Lazy import to avoid import-time sklearn overhead
_IsolationForest = None


def _get_isolation_forest():
    global _IsolationForest
    if _IsolationForest is None:
        from sklearn.ensemble import IsolationForest
        _IsolationForest = IsolationForest
    return _IsolationForest


# ── Results ─────────────────────────────────────────────────────────────────

class AnomalyScore(NamedTuple):
    """Score for a single transaction."""
    anomaly_score: float     # ∈ [-1.0, 0.0] lower = more anomalous
    is_anomaly: bool         # True if classified as outlier
    normalized_score: float  # ∈ [0.0, 1.0] higher = more anomalous


@dataclass
class IsolationDetectorMetrics:
    """Performance counters."""
    fit_count: int = 0
    last_fit_samples: int = 0
    last_fit_ms: float = 0.0
    total_scored: int = 0
    total_anomalies: int = 0
    _start_time: float = field(default_factory=time.monotonic)

    @property
    def anomaly_rate(self) -> float:
        if self.total_scored == 0:
            return 0.0
        return self.total_anomalies / self.total_scored

    def snapshot(self) -> dict:
        return {
            "fit_count": self.fit_count,
            "last_fit_samples": self.last_fit_samples,
            "last_fit_ms": round(self.last_fit_ms, 2),
            "total_scored": self.total_scored,
            "total_anomalies": self.total_anomalies,
            "anomaly_rate": round(self.anomaly_rate, 4),
            "uptime_sec": round(time.monotonic() - self._start_time, 1),
        }


# ── Isolation Forest Detector ──────────────────────────────────────────────

class IsolationDetector:
    """
    Unsupervised anomaly detector using Isolation Forest.

    Parameters
    ----------
    n_estimators : int
        Number of isolation trees (default 100).
    contamination : float
        Expected proportion of anomalies in training data (default 0.05).
    max_samples : int
        Max samples per tree (default 2048 for memory efficiency).
    random_state : int
        Seed for reproducibility.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        contamination: float = 0.05,
        max_samples: int = 2048,
        random_state: int = 42,
    ) -> None:
        self._n_estimators = n_estimators
        self._contamination = contamination
        self._max_samples = max_samples
        self._random_state = random_state
        self._model = None
        self._fitted = False
        self.metrics = IsolationDetectorMetrics()

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    def fit(self, features: np.ndarray) -> None:
        """
        Fit the Isolation Forest on accumulated feature data.

        Parameters
        ----------
        features : np.ndarray
            Shape (N, D) float32 feature matrix from FeatureEngine.
        """
        if features.shape[0] < 10:
            logger.warning("IsolationDetector: too few samples (%d), skipping fit",
                           features.shape[0])
            return

        IsoForest = _get_isolation_forest()
        t0 = time.monotonic()

        effective_max = min(self._max_samples, features.shape[0])
        self._model = IsoForest(
            n_estimators=self._n_estimators,
            contamination=self._contamination,
            max_samples=effective_max,
            random_state=self._random_state,
            n_jobs=1,  # single-threaded to avoid GIL contention with async
        )
        self._model.fit(features)
        self._fitted = True

        elapsed_ms = (time.monotonic() - t0) * 1000
        self.metrics.fit_count += 1
        self.metrics.last_fit_samples = features.shape[0]
        self.metrics.last_fit_ms = elapsed_ms
        logger.info(
            "IsolationDetector fitted on %d samples (%d features) in %.1f ms",
            features.shape[0], features.shape[1], elapsed_ms,
        )

    def score(self, features: np.ndarray) -> AnomalyScore:
        """
        Score a single sample or 1-row array.

        Returns
        -------
        AnomalyScore with anomaly_score (raw), is_anomaly flag, and
        normalized_score ∈ [0.0, 1.0] where 1.0 = most anomalous.
        """
        if not self._fitted:
            return AnomalyScore(anomaly_score=0.0, is_anomaly=False, normalized_score=0.0)

        x = features.reshape(1, -1) if features.ndim == 1 else features[:1]
        raw_score = float(self._model.score_samples(x)[0])
        prediction = int(self._model.predict(x)[0])
        is_anomaly = prediction == -1

        # Normalize: sklearn scores are negative (more negative = more anomalous)
        # Map roughly to [0, 1] using offset from decision threshold
        offset = float(self._model.offset_)
        normalized = max(0.0, min(1.0, -(raw_score - offset) / abs(offset) if offset != 0 else 0.0))

        self.metrics.total_scored += 1
        if is_anomaly:
            self.metrics.total_anomalies += 1

        return AnomalyScore(
            anomaly_score=raw_score,
            is_anomaly=is_anomaly,
            normalized_score=round(normalized, 6),
        )

    def score_batch(self, features: np.ndarray) -> list[AnomalyScore]:
        """
        Score a batch of samples.

        Parameters
        ----------
        features : np.ndarray
            Shape (N, D) float32 matrix.

        Returns
        -------
        list[AnomalyScore] — one per row.
        """
        if not self._fitted:
            return [AnomalyScore(0.0, False, 0.0)] * features.shape[0]

        raw_scores = self._model.score_samples(features)
        predictions = self._model.predict(features)
        offset = float(self._model.offset_)

        results: list[AnomalyScore] = []
        for raw, pred in zip(raw_scores, predictions):
            is_anomaly = int(pred) == -1
            normalized = max(0.0, min(1.0, -(raw - offset) / abs(offset) if offset != 0 else 0.0))
            results.append(AnomalyScore(
                anomaly_score=float(raw),
                is_anomaly=is_anomaly,
                normalized_score=round(float(normalized), 6),
            ))
            self.metrics.total_scored += 1
            if is_anomaly:
                self.metrics.total_anomalies += 1

        return results

    def snapshot(self) -> dict:
        """State snapshot for dashboard / debugging."""
        return {
            "fitted": self._fitted,
            "config": {
                "n_estimators": self._n_estimators,
                "contamination": self._contamination,
                "max_samples": self._max_samples,
            },
            "metrics": self.metrics.snapshot(),
        }

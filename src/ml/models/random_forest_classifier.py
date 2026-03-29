"""
PayFlow — Random Forest Fraud Classifier
==========================================
Ensemble diversity member alongside XGBoost. Uses scikit-learn
RandomForestClassifier for CPU-based fraud probability estimation.

Architecture::

    FeatureEngine (30-dim float32)
         ↓
    RandomForestFraudClassifier.predict(features)
         ↓
    risk_scores (float64, 0.0–1.0)

The Random Forest provides an independent vote that is averaged with
XGBoost scores in the orchestrator for improved ensemble robustness
(mitigates single-model overfitting on adversarial fraud patterns).

RBI compliance note: RBI FMS documentation explicitly lists Random Forest
as a recommended classifier alongside Gradient Boosting / Neural Networks.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import NamedTuple

import numpy as np

logger = logging.getLogger(__name__)


# ── Result Types ─────────────────────────────────────────────────────────────

class RFPredictionResult(NamedTuple):
    """Per-batch inference output."""
    risk_scores: np.ndarray       # (N,) float64 — P(fraud)
    predicted_labels: np.ndarray  # (N,) int — 0/1 at 0.5
    inference_ms: float


@dataclass
class RFValidationMetrics:
    """Hold-out evaluation metrics."""
    aucpr: float = 0.0
    auc_roc: float = 0.0
    f1: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    confusion_matrix: list[list[int]] = field(default_factory=list)
    eval_samples: int = 0

    def summary(self) -> dict:
        return {
            "aucpr": round(self.aucpr, 4),
            "auc_roc": round(self.auc_roc, 4),
            "f1": round(self.f1, 4),
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "eval_samples": self.eval_samples,
        }


@dataclass
class RFTrainingMetrics:
    """Training run diagnostics."""
    n_train: int = 0
    n_eval: int = 0
    n_estimators: int = 0
    best_aucpr: float = 0.0
    class_weight_mode: str = "balanced"
    training_ms: float = 0.0
    oob_score: float = 0.0

    def summary(self) -> dict:
        return {
            "n_train": self.n_train,
            "n_eval": self.n_eval,
            "n_estimators": self.n_estimators,
            "best_aucpr": round(self.best_aucpr, 4),
            "class_weight": self.class_weight_mode,
            "training_ms": round(self.training_ms, 1),
            "oob_score": round(self.oob_score, 4),
        }


# ── Random Forest Classifier ────────────────────────────────────────────────

class RandomForestFraudClassifier:
    """
    Scikit-learn Random Forest wrapper for fraud classification.

    Handles class imbalance via ``class_weight='balanced'`` which
    auto-adjusts weights inversely proportional to class frequencies.

    Usage::

        rf = RandomForestFraudClassifier()
        metrics = rf.train(X_train, y_train, X_eval, y_eval)
        result = rf.predict(X_new)
    """

    N_ESTIMATORS = 300
    MAX_DEPTH = 12
    MIN_SAMPLES_SPLIT = 5
    MIN_SAMPLES_LEAF = 2

    def __init__(self) -> None:
        self._model = None
        self._is_fitted: bool = False
        self._training_metrics: RFTrainingMetrics | None = None
        self._recent_scores: list[float] = []
        self._MAX_RECENT = 2000

    # ── Training ─────────────────────────────────────────────────────────

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_eval: np.ndarray | None = None,
        y_eval: np.ndarray | None = None,
        feature_names: list[str] | None = None,
    ) -> RFTrainingMetrics:
        """
        Fit the Random Forest on labelled feature data.

        Args:
            X_train: (N, D) float32 feature array
            y_train: (N,) binary labels (0=legit, 1=fraud)
            X_eval:  optional hold-out features
            y_eval:  optional hold-out labels
            feature_names: optional column names

        Returns:
            RFTrainingMetrics with diagnostics
        """
        from sklearn.ensemble import RandomForestClassifier

        y_bin_train = (y_train > 0).astype(np.int8)
        y_bin_eval = (y_eval > 0).astype(np.int8) if y_eval is not None else None

        metrics = RFTrainingMetrics(
            n_train=X_train.shape[0],
            n_eval=X_eval.shape[0] if X_eval is not None else 0,
            n_estimators=self.N_ESTIMATORS,
            class_weight_mode="balanced",
        )

        clf = RandomForestClassifier(
            n_estimators=self.N_ESTIMATORS,
            max_depth=self.MAX_DEPTH,
            min_samples_split=self.MIN_SAMPLES_SPLIT,
            min_samples_leaf=self.MIN_SAMPLES_LEAF,
            class_weight="balanced",
            oob_score=True,
            n_jobs=-1,
            random_state=42,
        )

        t0 = time.perf_counter()
        clf.fit(X_train, y_bin_train)
        metrics.training_ms = (time.perf_counter() - t0) * 1000

        self._model = clf
        self._is_fitted = True
        metrics.oob_score = clf.oob_score_

        # Evaluate on hold-out if provided
        if X_eval is not None and y_bin_eval is not None:
            from sklearn.metrics import average_precision_score
            proba = clf.predict_proba(X_eval)[:, 1]
            metrics.best_aucpr = float(average_precision_score(y_bin_eval, proba))

        self._training_metrics = metrics
        logger.info("RandomForest training complete: %s", metrics.summary())
        return metrics

    # ── Inference ────────────────────────────────────────────────────────

    def predict(self, X: np.ndarray) -> RFPredictionResult:
        """
        Compute fraud probabilities for a feature batch.

        Args:
            X: (N, D) float32 array from FeatureEngine

        Returns:
            RFPredictionResult with risk_scores, predicted_labels, timing
        """
        if not self._is_fitted:
            raise RuntimeError("RandomForest not trained. Call train() first.")

        t0 = time.perf_counter()
        proba = self._model.predict_proba(X)[:, 1]
        elapsed = (time.perf_counter() - t0) * 1000
        labels = (proba >= 0.5).astype(np.int8)

        self._recent_scores.extend(proba.tolist())
        if len(self._recent_scores) > self._MAX_RECENT:
            self._recent_scores = self._recent_scores[-self._MAX_RECENT:]

        return RFPredictionResult(
            risk_scores=proba,
            predicted_labels=labels,
            inference_ms=elapsed,
        )

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return raw fraud probability scores (N,) float64."""
        if not self._is_fitted:
            raise RuntimeError("RandomForest not trained. Call train() first.")
        return self._model.predict_proba(X)[:, 1]

    # ── Validation ───────────────────────────────────────────────────────

    def validate(
        self, X_eval: np.ndarray, y_eval: np.ndarray,
    ) -> RFValidationMetrics:
        """Full evaluation on a held-out set."""
        from sklearn.metrics import (
            average_precision_score,
            confusion_matrix,
            f1_score,
            precision_score,
            recall_score,
            roc_auc_score,
        )

        y_bin = (y_eval > 0).astype(np.int8)
        proba = self.predict_proba(X_eval)
        preds = (proba >= 0.5).astype(np.int8)

        m = RFValidationMetrics(eval_samples=len(y_bin))
        m.aucpr = float(average_precision_score(y_bin, proba))
        m.auc_roc = float(roc_auc_score(y_bin, proba))
        m.f1 = float(f1_score(y_bin, preds, zero_division=0))
        m.precision = float(precision_score(y_bin, preds, zero_division=0))
        m.recall = float(recall_score(y_bin, preds, zero_division=0))
        m.confusion_matrix = confusion_matrix(y_bin, preds).tolist()

        logger.info("RandomForest validation: %s", m.summary())
        return m

    # ── Persistence ──────────────────────────────────────────────────────

    def save(self, path: str | Path) -> None:
        """Save trained model via joblib."""
        if not self._is_fitted:
            raise RuntimeError("No trained model to save.")
        import joblib
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self._model, str(path))
        logger.info("RandomForest saved to %s", path)

    def load(self, path: str | Path) -> None:
        """Load a previously saved model."""
        import joblib
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")
        self._model = joblib.load(str(path))
        self._is_fitted = True
        logger.info("RandomForest loaded from %s", path)

    # ── Feature Importance ───────────────────────────────────────────────

    def feature_importance(
        self, feature_names: list[str] | None = None,
    ) -> dict[str, float]:
        """Return Gini importances sorted descending."""
        if not self._is_fitted:
            return {}
        importances = self._model.feature_importances_
        names = feature_names or [f"f{i}" for i in range(len(importances))]
        ranked = sorted(
            zip(names, importances), key=lambda x: x[1], reverse=True,
        )
        return {n: float(v) for n, v in ranked}

    # ── Diagnostics ──────────────────────────────────────────────────────

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    @property
    def training_info(self) -> dict | None:
        return self._training_metrics.summary() if self._training_metrics else None

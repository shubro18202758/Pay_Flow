"""
PayFlow — Logistic Regression Fraud Classifier
================================================
Interpretable baseline model for AML compliance.  Uses scikit-learn
LogisticRegression as an audit-friendly alternative to tree-based
ensembles — regulators value the ability to inspect per-feature
coefficients directly.

Architecture::

    FeatureEngine (36-dim float32)
         ↓
    LogisticFraudClassifier.predict(features)
         ↓
    risk_scores (float64, 0.0–1.0)

The model's linear coefficients provide a transparent explanation of
each feature's contribution to the fraud probability, satisfying
RBI/FIU-IND explainability requirements.
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

class LRPredictionResult(NamedTuple):
    """Per-batch inference output."""
    risk_scores: np.ndarray       # (N,) float64 — P(fraud)
    predicted_labels: np.ndarray  # (N,) int — 0/1 at 0.5
    inference_ms: float


@dataclass
class LRValidationMetrics:
    """Hold-out evaluation metrics."""
    aucpr: float = 0.0
    auc_roc: float = 0.0
    f1: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    n_eval: int = 0


@dataclass
class LRTrainingMetrics:
    """Training run summary."""
    n_train: int = 0
    n_eval: int = 0
    best_aucpr: float = 0.0
    auc_roc: float = 0.0
    train_time_s: float = 0.0
    converged: bool = False
    coefficients: list[float] = field(default_factory=list)


# ── Classifier ───────────────────────────────────────────────────────────────

class LogisticFraudClassifier:
    """
    Scikit-learn Logistic Regression wrapper with the same train / predict /
    validate interface used by FraudClassifier and RandomForestFraudClassifier.

    Uses class_weight="balanced" to handle the typical ≈1:200 fraud imbalance,
    and C=0.1 (stronger regularisation) to prevent overfit on sparse signals.
    """

    def __init__(
        self,
        C: float = 0.1,
        max_iter: int = 1000,
        solver: str = "lbfgs",
    ) -> None:
        self._C = C
        self._max_iter = max_iter
        self._solver = solver
        self._model = None
        self._is_fitted = False
        self._feature_names: list[str] = []
        self._training_info: dict = {}

    # ── Properties ────────────────────────────────────────────────────────

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    @property
    def training_info(self) -> dict:
        return dict(self._training_info)

    # ── Training ──────────────────────────────────────────────────────────

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_eval: np.ndarray | None = None,
        y_eval: np.ndarray | None = None,
        feature_names: list[str] | None = None,
    ) -> LRTrainingMetrics:
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import (
            precision_recall_curve,
            roc_auc_score,
        )

        self._feature_names = list(feature_names) if feature_names else []

        t0 = time.monotonic()

        y_bin_train = (y_train > 0).astype(np.int8)

        self._model = LogisticRegression(
            C=self._C,
            class_weight="balanced",
            max_iter=self._max_iter,
            solver=self._solver,
            random_state=42,
        )
        self._model.fit(X_train, y_bin_train)
        elapsed = time.monotonic() - t0

        self._is_fitted = True

        metrics = LRTrainingMetrics(
            n_train=X_train.shape[0],
            train_time_s=round(elapsed, 3),
            converged=bool(self._model.n_iter_[0] < self._max_iter),
            coefficients=self._model.coef_[0].tolist(),
        )

        if X_eval is not None and y_eval is not None:
            metrics.n_eval = X_eval.shape[0]
            y_bin_eval = (y_eval > 0).astype(np.int8)
            probs = self._model.predict_proba(X_eval)[:, 1]
            if len(np.unique(y_bin_eval)) > 1:
                precision_vals, recall_vals, _ = precision_recall_curve(y_bin_eval, probs)
                metrics.best_aucpr = float(-np.trapezoid(precision_vals, recall_vals))
                metrics.auc_roc = float(roc_auc_score(y_bin_eval, probs))

        self._training_info = {
            "model": "LogisticRegression",
            "C": self._C,
            "solver": self._solver,
            "n_train": metrics.n_train,
            "aucpr": round(metrics.best_aucpr, 4),
            "auc_roc": round(metrics.auc_roc, 4),
            "converged": metrics.converged,
            "train_time_s": metrics.train_time_s,
        }
        logger.info(
            "LogisticRegression fitted: %d samples, AUCPR=%.4f, converged=%s",
            metrics.n_train, metrics.best_aucpr, metrics.converged,
        )
        return metrics

    # ── Inference ─────────────────────────────────────────────────────────

    def predict(self, features: np.ndarray) -> LRPredictionResult:
        if not self._is_fitted or self._model is None:
            raise RuntimeError("LogisticFraudClassifier not fitted — call train() first")

        t0 = time.perf_counter()
        probs = self._model.predict_proba(features)[:, 1]
        labels = (probs >= 0.5).astype(int)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        return LRPredictionResult(
            risk_scores=probs,
            predicted_labels=labels,
            inference_ms=round(elapsed_ms, 3),
        )

    # ── Validation ────────────────────────────────────────────────────────

    def validate(self, X: np.ndarray, y: np.ndarray) -> LRValidationMetrics:
        if not self._is_fitted or self._model is None:
            raise RuntimeError("LogisticFraudClassifier not fitted")

        from sklearn.metrics import (
            precision_recall_curve,
            f1_score,
            precision_score,
            recall_score,
            roc_auc_score,
        )

        probs = self._model.predict_proba(X)[:, 1]
        y_bin = (y > 0).astype(np.int8)
        preds = (probs >= 0.5).astype(int)
        p_vals, r_vals, _ = precision_recall_curve(y_bin, probs)

        return LRValidationMetrics(
            aucpr=float(-np.trapezoid(p_vals, r_vals)),
            auc_roc=float(roc_auc_score(y_bin, probs)),
            f1=float(f1_score(y_bin, preds, zero_division=0)),
            precision=float(precision_score(y_bin, preds, zero_division=0)),
            recall=float(recall_score(y_bin, preds, zero_division=0)),
            n_eval=X.shape[0],
        )

    # ── Feature Importance (coefficients) ─────────────────────────────────

    def feature_importance(self) -> dict[str, float]:
        """Return feature → absolute coefficient mapping."""
        if not self._is_fitted or self._model is None:
            return {}

        coeffs = self._model.coef_[0]
        names = self._feature_names or [f"f{i}" for i in range(len(coeffs))]
        return {n: round(float(abs(c)), 6) for n, c in zip(names, coeffs)}

    # ── Persistence ───────────────────────────────────────────────────────

    def save(self, path: str | Path) -> None:
        import joblib
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self._model, path)
        logger.info("LogisticRegression model saved → %s", path)

    def load(self, path: str | Path) -> None:
        import joblib
        self._model = joblib.load(path)
        self._is_fitted = True
        logger.info("LogisticRegression model loaded ← %s", path)

    # ── Snapshot ──────────────────────────────────────────────────────────

    def snapshot(self) -> dict:
        return {
            "model": "LogisticRegression",
            "is_fitted": self._is_fitted,
            "C": self._C,
            **self._training_info,
        }

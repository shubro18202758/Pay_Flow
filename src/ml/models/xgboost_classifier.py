"""
PayFlow — XGBoost CUDA Fraud Classifier
==========================================
GPU-accelerated gradient boosting classifier for real-time fraud probability
estimation on the RTX 4070 (8 GB VRAM).

Architecture:
    FeatureEngine (30-dim float32)
         ↓
    FraudClassifier.predict(features)
         ↓
    risk_scores (float64, 0.0–1.0)
         ↓
    DynamicThreshold → high-risk payloads → AlertRouter

GPU Strategy:
    - Primary: XGBoost 'cuda' device with `hist` tree method (histogram-based
      splitting uses ~128 bins → fits in ~1 GB VRAM budget).
    - Fallback: automatic degradation to CPU `hist` when CUDA OOM or VRAM
      saturation is detected. The switchover is transparent to callers.

Class Imbalance:
    Fraud datasets are severely imbalanced (~1–5% positive). We address this
    via `scale_pos_weight` (auto-computed from training labels) and optimise
    for `aucpr` (area under precision-recall curve) instead of standard AUC.

Model Lifecycle:
    1. train()      — fit on accumulated FeatureEngine data
    2. validate()   — hold-out evaluation with PR-AUC, F1, confusion matrix
    3. predict()    — single-batch or streaming inference
    4. save/load    — XGBoost native binary format (.ubj)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import NamedTuple

import numpy as np

logger = logging.getLogger(__name__)


# ── Prediction Result ────────────────────────────────────────────────────────

class PredictionResult(NamedTuple):
    """Per-batch prediction output."""
    risk_scores: np.ndarray     # shape (N,), float64, probability of fraud
    predicted_labels: np.ndarray  # shape (N,), int, 0/1 binary (at default 0.5)
    inference_ms: float


# ── Validation Metrics ───────────────────────────────────────────────────────

@dataclass
class ValidationMetrics:
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


# ── Training Metrics ─────────────────────────────────────────────────────────

@dataclass
class TrainingMetrics:
    """Training run diagnostics."""
    n_train: int = 0
    n_eval: int = 0
    n_estimators_used: int = 0
    best_iteration: int = 0
    best_aucpr: float = 0.0
    scale_pos_weight: float = 1.0
    device_used: str = "unknown"
    training_ms: float = 0.0
    fell_back_to_cpu: bool = False

    def summary(self) -> dict:
        return {
            "n_train": self.n_train,
            "n_eval": self.n_eval,
            "n_estimators_used": self.n_estimators_used,
            "best_iteration": self.best_iteration,
            "best_aucpr": round(self.best_aucpr, 4),
            "scale_pos_weight": round(self.scale_pos_weight, 2),
            "device": self.device_used,
            "training_ms": round(self.training_ms, 1),
            "cpu_fallback": self.fell_back_to_cpu,
        }


# ── Fraud Classifier ────────────────────────────────────────────────────────

class FraudClassifier:
    """
    XGBoost-based fraud probability estimator with VRAM-aware GPU/CPU fallback.

    Wraps the XGBoost Scikit-Learn API (XGBClassifier) for ergonomic
    train/predict/save/load, while injecting the hardware-specific config
    from `config.settings.XGBoostConfig`.

    Usage:
        from config import XGBOOST_CFG, analysis_mode

        clf = FraudClassifier(XGBOOST_CFG)

        with analysis_mode():
            metrics = clf.train(X_train, y_train, X_eval, y_eval)
            result = clf.predict(X_new)
    """

    def __init__(self, cfg=None) -> None:
        from config.settings import XGBOOST_CFG
        self._cfg = cfg or XGBOOST_CFG
        self._model = None
        self._is_fitted: bool = False
        self._device: str = self._cfg.device
        self._training_metrics: TrainingMetrics | None = None
        self._recent_scores: list[float] = []
        self._MAX_RECENT_SCORES = 2000

    # ── Training ──────────────────────────────────────────────────────────

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_eval: np.ndarray | None = None,
        y_eval: np.ndarray | None = None,
        feature_names: list[str] | None = None,
    ) -> TrainingMetrics:
        """
        Fit the XGBoost classifier. Auto-computes scale_pos_weight from
        label distribution and falls back to CPU on CUDA failure.

        Args:
            X_train: (N, 30) float32 feature array
            y_train: (N,) binary labels (0=legit, 1=fraud)
            X_eval:  optional hold-out features
            y_eval:  optional hold-out labels
            feature_names: column names for feature importance tracking

        Returns:
            TrainingMetrics with diagnostics
        """
        import xgboost as xgb

        # Binarise multi-class labels → 0 (NONE) vs 1 (any fraud pattern)
        y_bin_train = (y_train > 0).astype(np.int8)
        y_bin_eval = (y_eval > 0).astype(np.int8) if y_eval is not None else None

        # Auto-compute class imbalance weight
        n_neg = int((y_bin_train == 0).sum())
        n_pos = int((y_bin_train == 1).sum())
        spw = n_neg / max(n_pos, 1)

        metrics = TrainingMetrics(
            n_train=X_train.shape[0],
            n_eval=X_eval.shape[0] if X_eval is not None else 0,
            scale_pos_weight=spw,
        )

        # Build params dict from frozen config
        params = {
            "objective": "binary:logistic",
            "eval_metric": self._cfg.eval_metric,
            "tree_method": self._cfg.tree_method,
            "max_bin": self._cfg.max_bin,
            "max_depth": self._cfg.max_depth,
            "subsample": self._cfg.subsample,
            "colsample_bytree": self._cfg.colsample_bytree,
            "n_estimators": self._cfg.n_estimators,
            "early_stopping_rounds": self._cfg.early_stopping_rounds,
            "scale_pos_weight": spw,
            "verbosity": 1,
            "n_jobs": -1,
        }

        # Attempt GPU first, fall back to CPU on failure
        self._model, metrics.device_used, metrics.fell_back_to_cpu = (
            self._fit_with_fallback(xgb, params, X_train, y_bin_train,
                                    X_eval, y_bin_eval, feature_names)
        )

        t0 = time.perf_counter()
        # Training already happened inside _fit_with_fallback, record timing
        metrics.training_ms = (time.perf_counter() - t0) * 1000

        # Extract best iteration info
        if hasattr(self._model, "best_iteration"):
            metrics.best_iteration = self._model.best_iteration or 0
            metrics.n_estimators_used = (self._model.best_iteration or 0) + 1
        else:
            metrics.n_estimators_used = self._cfg.n_estimators

        # Evaluate on held-out set for AUCPR
        if X_eval is not None and y_bin_eval is not None:
            proba = self._model.predict_proba(X_eval)[:, 1]
            from sklearn.metrics import average_precision_score
            metrics.best_aucpr = float(average_precision_score(y_bin_eval, proba))

        self._is_fitted = True
        self._training_metrics = metrics
        logger.info("Training complete: %s", metrics.summary())
        return metrics

    def _fit_with_fallback(
        self, xgb, params: dict,
        X_train, y_train, X_eval, y_eval, feature_names,
    ) -> tuple:
        """Try GPU training; on CUDA error, retry on CPU."""
        import xgboost

        eval_set = [(X_eval, y_eval)] if X_eval is not None else None
        fell_back = False

        for device in [self._cfg.device, "cpu"]:
            try:
                p = {**params, "device": device}
                clf = xgboost.XGBClassifier(**p)

                fit_kwargs = {}
                if eval_set:
                    fit_kwargs["eval_set"] = eval_set

                t0 = time.perf_counter()
                clf.fit(X_train, y_train, verbose=False, **fit_kwargs)
                elapsed = (time.perf_counter() - t0) * 1000

                logger.info(
                    "XGBoost trained on '%s' in %.1f ms (%d trees)",
                    device, elapsed,
                    clf.best_iteration + 1 if hasattr(clf, "best_iteration") and clf.best_iteration else p["n_estimators"],
                )
                self._device = device
                return clf, device, fell_back

            except (xgboost.core.XGBoostError, RuntimeError) as exc:
                if device != "cpu":
                    logger.warning(
                        "CUDA training failed (%s). Falling back to CPU.", exc
                    )
                    fell_back = True
                    continue
                raise  # CPU also failed — re-raise

        # Should not reach here, but satisfy type checker
        raise RuntimeError("XGBoost training failed on all devices")

    # ── Inference ─────────────────────────────────────────────────────────

    def predict(self, X: np.ndarray) -> PredictionResult:
        """
        Compute fraud probabilities for a feature batch.

        Args:
            X: (N, 30) float32 array from FeatureEngine

        Returns:
            PredictionResult with risk_scores, predicted_labels, timing
        """
        if not self._is_fitted:
            raise RuntimeError("Classifier not trained. Call train() first.")

        t0 = time.perf_counter()
        try:
            proba = self._model.predict_proba(X)[:, 1]
        except Exception as exc:
            # GPU inference failure → rebuild on CPU and retry
            logger.warning("GPU predict failed (%s). Retrying on CPU.", exc)
            proba = self._cpu_fallback_predict(X)

        elapsed = (time.perf_counter() - t0) * 1000
        labels = (proba >= 0.5).astype(np.int8)

        # Track recent scores for analytics endpoint
        self._recent_scores.extend(proba.tolist())
        if len(self._recent_scores) > self._MAX_RECENT_SCORES:
            self._recent_scores = self._recent_scores[-self._MAX_RECENT_SCORES:]

        return PredictionResult(
            risk_scores=proba,
            predicted_labels=labels,
            inference_ms=elapsed,
        )

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return raw fraud probability scores (N,) float64."""
        if not self._is_fitted:
            raise RuntimeError("Classifier not trained. Call train() first.")
        try:
            return self._model.predict_proba(X)[:, 1]
        except Exception:
            return self._cpu_fallback_predict(X)

    def _cpu_fallback_predict(self, X: np.ndarray) -> np.ndarray:
        """Reconstruct model on CPU for inference fallback."""
        import xgboost as xgb

        # Save current model to temp buffer and reload on CPU
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".ubj", delete=False) as f:
            tmp_path = f.name
            self._model.save_model(tmp_path)

        cpu_model = xgb.XGBClassifier()
        cpu_model.load_model(tmp_path)
        Path(tmp_path).unlink(missing_ok=True)

        logger.info("Inference fell back to CPU.")
        return cpu_model.predict_proba(X)[:, 1]

    # ── Validation ────────────────────────────────────────────────────────

    def validate(
        self, X_eval: np.ndarray, y_eval: np.ndarray
    ) -> ValidationMetrics:
        """
        Full evaluation on a held-out set.

        Returns PR-AUC, ROC-AUC, F1, precision, recall, and confusion matrix.
        """
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

        m = ValidationMetrics(eval_samples=len(y_bin))
        m.aucpr = float(average_precision_score(y_bin, proba))
        m.auc_roc = float(roc_auc_score(y_bin, proba))
        m.f1 = float(f1_score(y_bin, preds, zero_division=0))
        m.precision = float(precision_score(y_bin, preds, zero_division=0))
        m.recall = float(recall_score(y_bin, preds, zero_division=0))
        m.confusion_matrix = confusion_matrix(y_bin, preds).tolist()

        logger.info("Validation: %s", m.summary())
        return m

    # ── Persistence ───────────────────────────────────────────────────────

    def save(self, path: str | Path) -> None:
        """Save trained model in XGBoost Universal Binary JSON format."""
        if not self._is_fitted:
            raise RuntimeError("No trained model to save.")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._model.save_model(str(path))
        logger.info("Model saved to %s", path)

    def load(self, path: str | Path) -> None:
        """Load a previously saved model."""
        import xgboost as xgb

        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")
        self._model = xgb.XGBClassifier()
        self._model.load_model(str(path))
        self._is_fitted = True
        logger.info("Model loaded from %s", path)

    # ── Feature Importance ────────────────────────────────────────────────

    def feature_importance(self, importance_type: str = "gain") -> dict[str, float]:
        """Return feature importance scores."""
        if not self._is_fitted:
            return {}
        booster = self._model.get_booster()
        scores = booster.get_score(importance_type=importance_type)
        # Sort descending
        return dict(sorted(scores.items(), key=lambda x: x[1], reverse=True))

    # ── Diagnostics ───────────────────────────────────────────────────────

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    @property
    def device(self) -> str:
        return self._device

    @property
    def training_info(self) -> dict | None:
        return self._training_metrics.summary() if self._training_metrics else None

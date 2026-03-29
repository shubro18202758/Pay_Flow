"""
PayFlow — 3-Model Fraud Detection Ensemble Combiner
=====================================================
Combines outputs from XGBoost (primary classifier), Isolation Forest
(anomaly detection), and Autoencoder (behavioral deviation) into a
single fraud probability score using the build-prompt prescribed
weighted formula:

    P_final = 0.5 * P_xgb + 0.3 * P_iso + 0.2 * P_ae

Verdict routing:
    P_final < 0.3   → LEGITIMATE
    P_final < 0.6   → MONITOR
    P_final < 0.8   → INVESTIGATE
    P_final >= 0.8  → BLOCK
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import NamedTuple

import numpy as np

logger = logging.getLogger(__name__)


# ── Configuration ───────────────────────────────────────────────────────────

@dataclass(frozen=True)
class EnsembleWeights:
    """Prescribed ensemble model weights (must sum to 1.0)."""
    xgboost: float = 0.5
    isolation_forest: float = 0.3
    autoencoder: float = 0.2

    def __post_init__(self) -> None:
        total = self.xgboost + self.isolation_forest + self.autoencoder
        if abs(total - 1.0) > 0.001:
            raise ValueError(f"Ensemble weights must sum to 1.0, got {total:.4f}")


# ── Result ──────────────────────────────────────────────────────────────────

class EnsembleResult(NamedTuple):
    """Result of ensemble fraud scoring."""
    p_final: float           # Combined probability [0.0, 1.0]
    p_xgb: float             # XGBoost component
    p_iso: float             # Isolation Forest component
    p_ae: float              # Autoencoder component
    verdict: str             # LEGITIMATE / MONITOR / INVESTIGATE / BLOCK
    model_contributions: dict  # Weighted contribution of each model

    def to_dict(self) -> dict:
        return {
            "p_final": round(self.p_final, 4),
            "components": {
                "xgboost": round(self.p_xgb, 4),
                "isolation_forest": round(self.p_iso, 4),
                "autoencoder": round(self.p_ae, 4),
            },
            "weighted_contributions": {
                k: round(v, 4) for k, v in self.model_contributions.items()
            },
            "verdict": self.verdict,
        }


# ── Ensemble Combiner ──────────────────────────────────────────────────────

class FraudEnsemble:
    """
    3-model ensemble combiner for fraud detection.

    Takes probability outputs from XGBoost, Isolation Forest, and
    Autoencoder and combines them using prescribed weights:
        P_final = 0.5 * P_xgb + 0.3 * P_iso + 0.2 * P_ae
    """

    def __init__(self, weights: EnsembleWeights | None = None) -> None:
        self._weights = weights or EnsembleWeights()
        self._scored_count: int = 0
        self._verdict_counts: dict[str, int] = {
            "LEGITIMATE": 0, "MONITOR": 0,
            "INVESTIGATE": 0, "BLOCK": 0,
        }
        self._start_time = time.monotonic()

    # ── Core Scoring ────────────────────────────────────────────────────

    def score(
        self,
        p_xgb: float,
        p_iso: float,
        p_ae: float,
    ) -> EnsembleResult:
        """
        Combine three model probabilities into a single fraud score.

        Parameters
        ----------
        p_xgb : float
            XGBoost fraud probability [0.0, 1.0].
        p_iso : float
            Isolation Forest anomaly score, normalised to [0.0, 1.0]
            (higher = more anomalous).
        p_ae : float
            Autoencoder reconstruction error, normalised to [0.0, 1.0]
            (higher = more deviation from baseline).
        """
        p_xgb = float(np.clip(p_xgb, 0.0, 1.0))
        p_iso = float(np.clip(p_iso, 0.0, 1.0))
        p_ae = float(np.clip(p_ae, 0.0, 1.0))

        w = self._weights
        p_final = w.xgboost * p_xgb + w.isolation_forest * p_iso + w.autoencoder * p_ae
        p_final = min(1.0, max(0.0, p_final))

        verdict = self._route(p_final)
        self._scored_count += 1
        self._verdict_counts[verdict] += 1

        return EnsembleResult(
            p_final=p_final,
            p_xgb=p_xgb,
            p_iso=p_iso,
            p_ae=p_ae,
            verdict=verdict,
            model_contributions={
                "xgboost": w.xgboost * p_xgb,
                "isolation_forest": w.isolation_forest * p_iso,
                "autoencoder": w.autoencoder * p_ae,
            },
        )

    def score_batch(
        self,
        p_xgb: np.ndarray,
        p_iso: np.ndarray,
        p_ae: np.ndarray,
    ) -> list[EnsembleResult]:
        """Score a batch of probabilities from the three models."""
        results = []
        for x, i, a in zip(p_xgb, p_iso, p_ae):
            results.append(self.score(float(x), float(i), float(a)))
        return results

    # ── Verdict Router ──────────────────────────────────────────────────

    @staticmethod
    def _route(p_final: float) -> str:
        if p_final < 0.3:
            return "LEGITIMATE"
        if p_final < 0.6:
            return "MONITOR"
        if p_final < 0.8:
            return "INVESTIGATE"
        return "BLOCK"

    # ── Diagnostics ─────────────────────────────────────────────────────

    @property
    def scored_count(self) -> int:
        return self._scored_count

    def snapshot(self) -> dict:
        return {
            "scored_count": self._scored_count,
            "verdict_distribution": dict(self._verdict_counts),
            "weights": {
                "xgboost": self._weights.xgboost,
                "isolation_forest": self._weights.isolation_forest,
                "autoencoder": self._weights.autoencoder,
            },
            "uptime_sec": round(time.monotonic() - self._start_time, 1),
        }

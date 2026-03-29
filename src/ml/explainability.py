"""
PayFlow — SHAP Explainability Engine
======================================
Provides per-transaction SHAP-based feature importance explanations
for XGBoost fraud predictions. Enables human-interpretable AI
reasoning with waterfall decompositions and risk factor narratives.

Uses TreeSHAP (exact, polynomial-time) for XGBoost models and
generates structured explanation payloads for the dashboard.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class FeatureContribution:
    """Single feature's contribution to a prediction."""
    feature_name: str
    shap_value: float
    feature_value: float
    direction: str  # "risk_increase" | "risk_decrease"
    rank: int


@dataclass
class TransactionExplanation:
    """Complete SHAP explanation for a single transaction."""
    txn_id: str
    risk_score: float
    base_value: float  # expected value (population mean)
    contributions: list[FeatureContribution]
    top_risk_factors: list[str]
    top_protective_factors: list[str]
    explanation_ms: float
    narrative: str


@dataclass
class ExplainabilityMetrics:
    """Aggregate metrics for the explainability engine."""
    explanations_generated: int = 0
    avg_explanation_ms: float = 0.0
    top_global_features: list[str] = field(default_factory=list)
    last_explanation_time: float = 0.0


class SHAPExplainer:
    """
    SHAP-based explainability engine for XGBoost fraud classifier.

    Uses TreeSHAP for exact, fast feature attributions. Generates
    human-readable narratives explaining why a transaction was
    flagged or cleared.
    """

    FEATURE_DESCRIPTIONS: dict[str, str] = {
        "amount_zscore": "Transaction amount deviation",
        "hour_sin": "Time-of-day pattern",
        "hour_cos": "Time-of-day pattern (cosine)",
        "day_of_week": "Day of week",
        "is_weekend": "Weekend transaction",
        "channel": "Payment channel",
        "sender_type": "Sender account type",
        "receiver_type": "Receiver account type",
        "geo_distance_km": "Geographic distance",
        "geo_deviation": "Location deviation from normal",
        "device_fingerprint_hash": "Device fingerprint",
        "txn_count_1h": "Transactions in last hour",
        "txn_count_24h": "Transactions in last 24 hours",
        "avg_amount_24h_zscore": "Average amount deviation (24h)",
        "max_amount_24h_zscore": "Maximum amount deviation (24h)",
        "unique_receivers_1h": "Unique recipients (1h)",
        "unique_receivers_24h": "Unique recipients (24h)",
        "time_since_last_txn_sec": "Time since last transaction",
        "is_new_receiver": "First-time recipient",
        "is_new_device": "New device detected",
        "sender_degree": "Sender connection count",
        "receiver_degree": "Receiver connection count",
        "sender_in_ratio": "Sender incoming ratio",
        "receiver_out_ratio": "Receiver outgoing ratio",
        "common_neighbors": "Shared connections",
        "shortest_path_length": "Network distance",
        "sender_pagerank": "Sender network importance",
        "receiver_pagerank": "Receiver network importance",
        "sender_clustering": "Sender cluster density",
        "receiver_clustering": "Receiver cluster density",
        "sender_community": "Sender community ID",
        "receiver_community": "Receiver community ID",
        "same_community": "Same community flag",
        "community_cross_ratio": "Cross-community ratio",
        "sender_fraud_neighbor_ratio": "Sender fraud neighbor ratio",
        "receiver_fraud_neighbor_ratio": "Receiver fraud neighbor ratio",
        "ext_round_number": "Round number indicator",
        "ext_cfr_match": "Central Fraud Registry match",
        "ext_prior_fraud_reports": "Prior fraud reports count",
        "ext_cross_border": "Cross-border transaction",
        "ext_pass_through": "Pass-through account pattern",
        "ext_dormant_activation": "Dormant account reactivation",
    }

    def __init__(self, feature_names: Optional[list[str]] = None):
        self._explainer = None
        self._feature_names = feature_names or []
        self._metrics = ExplainabilityMetrics()
        self._global_importance: Optional[np.ndarray] = None

    def attach_model(self, classifier) -> None:
        """Attach a trained XGBoost classifier for SHAP computation."""
        try:
            import shap
            booster = classifier._model
            if booster is None:
                logger.warning("Classifier model is None — SHAP unavailable")
                return
            self._explainer = shap.TreeExplainer(booster)
            logger.info("SHAP TreeExplainer attached to XGBoost model")
        except ImportError:
            logger.info("shap package not installed — using fallback gain-based importance")
            self._explainer = None
            self._attach_fallback(classifier)
        except Exception as e:
            logger.warning("SHAP attach failed: %s — using fallback", e)
            self._attach_fallback(classifier)

    def _attach_fallback(self, classifier) -> None:
        """Use XGBoost's built-in feature importance as fallback."""
        try:
            model = classifier._model
            if model is not None:
                importance = model.get_score(importance_type="gain")
                self._global_importance = importance
        except Exception:
            pass

    def explain_transaction(
        self,
        features: np.ndarray,
        txn_id: str,
        risk_score: float,
        feature_names: Optional[list[str]] = None,
    ) -> TransactionExplanation:
        """
        Generate a SHAP explanation for a single transaction.

        Parameters
        ----------
        features : np.ndarray
            Feature vector (1D or 2D with single row).
        txn_id : str
            Transaction identifier.
        risk_score : float
            ML-predicted risk score.
        feature_names : list[str] | None
            Feature column names.

        Returns
        -------
        TransactionExplanation with ranked contributions and narrative.
        """
        t0 = time.monotonic()
        names = feature_names or self._feature_names

        if features.ndim == 1:
            features = features.reshape(1, -1)

        contributions = []
        base_value = 0.5  # default

        if self._explainer is not None:
            try:
                shap_values = self._explainer.shap_values(features)
                if isinstance(shap_values, list):
                    shap_values = shap_values[1]  # positive class
                shap_row = shap_values[0]
                base_value = float(self._explainer.expected_value)
                if isinstance(base_value, (list, np.ndarray)):
                    base_value = float(base_value[1])

                for i, sv in enumerate(shap_row):
                    fname = names[i] if i < len(names) else f"f{i}"
                    contributions.append(FeatureContribution(
                        feature_name=fname,
                        shap_value=float(sv),
                        feature_value=float(features[0, i]),
                        direction="risk_increase" if sv > 0 else "risk_decrease",
                        rank=0,
                    ))
            except Exception as e:
                logger.debug("SHAP computation failed: %s — using fallback", e)
                contributions = self._fallback_contributions(features, names)
        else:
            contributions = self._fallback_contributions(features, names)

        # Sort by absolute SHAP value descending
        contributions.sort(key=lambda c: abs(c.shap_value), reverse=True)
        for i, c in enumerate(contributions):
            c.rank = i + 1

        top_risk = [
            c.feature_name for c in contributions[:5]
            if c.direction == "risk_increase"
        ]
        top_protective = [
            c.feature_name for c in contributions[:5]
            if c.direction == "risk_decrease"
        ]

        narrative = self._build_narrative(risk_score, contributions[:5])

        elapsed = (time.monotonic() - t0) * 1000
        self._metrics.explanations_generated += 1
        self._metrics.avg_explanation_ms = (
            (self._metrics.avg_explanation_ms * (self._metrics.explanations_generated - 1) + elapsed)
            / self._metrics.explanations_generated
        )
        self._metrics.last_explanation_time = time.time()

        return TransactionExplanation(
            txn_id=txn_id,
            risk_score=risk_score,
            base_value=base_value,
            contributions=contributions[:15],  # top 15 for payload size
            top_risk_factors=top_risk,
            top_protective_factors=top_protective,
            explanation_ms=round(elapsed, 2),
            narrative=narrative,
        )

    def explain_batch(
        self,
        features: np.ndarray,
        txn_ids: list[str],
        risk_scores: np.ndarray,
        feature_names: Optional[list[str]] = None,
    ) -> list[TransactionExplanation]:
        """Explain a batch of transactions."""
        results = []
        for i in range(features.shape[0]):
            exp = self.explain_transaction(
                features[i],
                txn_ids[i],
                float(risk_scores[i]),
                feature_names,
            )
            results.append(exp)
        return results

    def global_feature_importance(
        self,
        features: Optional[np.ndarray] = None,
        feature_names: Optional[list[str]] = None,
    ) -> list[dict]:
        """
        Compute global feature importance across all training data.
        Returns sorted list of {feature, importance, description}.
        """
        names = feature_names or self._feature_names

        if self._explainer is not None and features is not None:
            try:
                import shap
                shap_values = self._explainer.shap_values(features[:500])
                if isinstance(shap_values, list):
                    shap_values = shap_values[1]
                mean_abs = np.abs(shap_values).mean(axis=0)
                result = []
                for i, imp in enumerate(mean_abs):
                    fname = names[i] if i < len(names) else f"f{i}"
                    result.append({
                        "feature": fname,
                        "importance": round(float(imp), 6),
                        "description": self.FEATURE_DESCRIPTIONS.get(fname, fname),
                    })
                result.sort(key=lambda x: x["importance"], reverse=True)
                self._metrics.top_global_features = [
                    r["feature"] for r in result[:10]
                ]
                return result
            except Exception as e:
                logger.debug("Global SHAP failed: %s", e)

        # Fallback: use stored importance
        if self._global_importance:
            result = []
            for fname, imp in self._global_importance.items():
                result.append({
                    "feature": fname,
                    "importance": round(float(imp), 6),
                    "description": self.FEATURE_DESCRIPTIONS.get(fname, fname),
                })
            result.sort(key=lambda x: x["importance"], reverse=True)
            return result

        return []

    def _fallback_contributions(
        self, features: np.ndarray, names: list[str],
    ) -> list[FeatureContribution]:
        """Generate pseudo-contributions from feature values when SHAP unavailable."""
        contributions = []
        for i in range(features.shape[1]):
            fname = names[i] if i < len(names) else f"f{i}"
            val = float(features[0, i])
            # Use normalized value as pseudo-importance
            pseudo_shap = val * 0.01 if abs(val) > 1 else val * 0.1
            contributions.append(FeatureContribution(
                feature_name=fname,
                shap_value=pseudo_shap,
                feature_value=val,
                direction="risk_increase" if pseudo_shap > 0 else "risk_decrease",
                rank=0,
            ))
        return contributions

    def _build_narrative(
        self, risk_score: float, top_contributions: list[FeatureContribution],
    ) -> str:
        """Build a human-readable explanation narrative."""
        if risk_score >= 0.85:
            severity = "CRITICAL risk"
        elif risk_score >= 0.60:
            severity = "HIGH risk"
        elif risk_score >= 0.30:
            severity = "MODERATE risk"
        else:
            severity = "LOW risk"

        risk_factors = [
            self.FEATURE_DESCRIPTIONS.get(c.feature_name, c.feature_name)
            for c in top_contributions if c.direction == "risk_increase"
        ][:3]

        protective = [
            self.FEATURE_DESCRIPTIONS.get(c.feature_name, c.feature_name)
            for c in top_contributions if c.direction == "risk_decrease"
        ][:2]

        parts = [f"This transaction is assessed as {severity} (score: {risk_score:.2f})."]

        if risk_factors:
            parts.append(f"Primary risk drivers: {', '.join(risk_factors)}.")

        if protective:
            parts.append(f"Mitigating factors: {', '.join(protective)}.")

        return " ".join(parts)

    def snapshot(self) -> dict:
        return {
            "explanations_generated": self._metrics.explanations_generated,
            "avg_explanation_ms": round(self._metrics.avg_explanation_ms, 2),
            "shap_available": self._explainer is not None,
            "top_global_features": self._metrics.top_global_features[:10],
        }

"""
PayFlow — Intelligence API Routes
===================================
Endpoints for SHAP explainability, model drift monitoring,
natural language queries, and cross-bank consortium intelligence.

Routes:
    POST /api/v1/intelligence/explain        — SHAP explanation for a transaction
    GET  /api/v1/intelligence/drift          — Model drift status
    POST /api/v1/intelligence/query          — Natural language query (Qwen 3.5)
    GET  /api/v1/intelligence/consortium     — Consortium hub status
    GET  /api/v1/intelligence/consortium/alerts — Query consortium alerts
    POST /api/v1/intelligence/consortium/publish — Publish alert to consortium
    POST /api/v1/intelligence/consortium/check  — Check account in consortium
"""

from __future__ import annotations

from fastapi import APIRouter, Request
from pydantic import BaseModel, Field

from src.api.rbac import require_any_permission, require_permission

router = APIRouter(prefix="/api/v1/intelligence", tags=["intelligence"])


# ── Internal helpers ─────────────────────────────────────────────────────────

def _global_importance_input(orch, limit: int = 500):
    """Collect real feature vectors for global explainability."""
    import numpy as np

    engine = getattr(orch, "_engine", None)
    if not engine:
        return None, None, "none", 0

    cache = getattr(engine, "_feature_cache", {}) or {}
    rows = []
    feature_names = None

    for cached in list(cache.values())[-limit:]:
        if not cached:
            continue
        vector, names = cached
        row = np.asarray(vector, dtype=np.float32).reshape(-1)
        if row.size == 0:
            continue
        rows.append(row)
        if feature_names is None and names:
            feature_names = list(names)

    if rows:
        width = min(row.shape[0] for row in rows)
        matrix = np.vstack([row[:width] for row in rows])
        if feature_names:
            feature_names = feature_names[:width]
        return matrix, feature_names, "feature_cache", matrix.shape[0]

    train_data = engine.get_training_data() if hasattr(engine, "get_training_data") else None
    if train_data is not None and getattr(train_data, "features", None) is not None:
        features = np.asarray(train_data.features, dtype=np.float32)
        if features.ndim == 2 and features.shape[0] > 0:
            try:
                from src.ml.feature_engine import FEATURE_COLUMNS
                feature_names = list(FEATURE_COLUMNS)[:features.shape[1]]
            except Exception:
                feature_names = None
            return features[:limit], feature_names, "training_data", min(features.shape[0], limit)

    return None, None, "none", 0


def _importance_record(importance) -> dict[str, float]:
    if isinstance(importance, dict):
        return {str(name): float(value) for name, value in importance.items()}

    record: dict[str, float] = {}
    for item in importance or []:
        if not isinstance(item, dict):
            continue
        name = item.get("feature")
        value = item.get("importance")
        if name is None or value is None:
            continue
        record[str(name)] = float(value)
    return record


def _drift_payload(detector, report, status: str = "ready") -> dict:
    snapshot = detector.snapshot()
    feature_drift = []
    for fd in detector.get_drifting_features():
        feature_drift.append({
            "feature": fd.feature_name,
            "psi": round(fd.psi_score, 6),
            "severity": fd.severity.name,
        })

    return {
        "status": status,
        "severity": report.severity.name,
        "psi": round(report.psi_score, 6),
        "ks_statistic": round(report.ks_statistic, 6),
        "ks_p_value": round(report.ks_p_value, 6),
        "js_divergence": round(report.js_divergence, 6),
        "reference_size": int(snapshot.get("reference_size") or 0),
        "current_size": report.window_size,
        "recommendation": report.recommendation,
        "feature_drift": feature_drift,
        "snapshot": snapshot,
    }


# ── Request / Response Models ────────────────────────────────────────────────

class ExplainRequest(BaseModel):
    features: list[float] | None = Field(None, min_length=1)
    txn_id: str = Field("unknown", min_length=1)


class NLQueryRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=2000)


class ConsortiumPublishRequest(BaseModel):
    account_id: str
    risk_score: float = Field(..., ge=0.0, le=1.0)
    fraud_type: int = Field(1, ge=1, le=8)
    severity: int = Field(3, ge=1, le=4)


class AccountCheckRequest(BaseModel):
    account_id: str


# ── Explainability ───────────────────────────────────────────────────────────

def _explanation_method(explainer) -> str:
    snapshot = explainer.snapshot() if hasattr(explainer, "snapshot") else {}
    return str(snapshot.get("method") or "model attribution")


def _model_reasoning_payload(
    *,
    classifier,
    explanation,
    verdict: str,
    attribution_method: str,
    domain_controls: list[str],
) -> dict:
    risk_drivers = [
        fc.feature_name
        for fc in explanation.contributions
        if getattr(fc, "direction", "") == "risk_increase"
    ][:5]
    protective_factors = [
        fc.feature_name
        for fc in explanation.contributions
        if getattr(fc, "direction", "") == "risk_decrease"
    ][:5]
    control_count = len(domain_controls)
    risk_score = float(explanation.risk_score)
    return {
        "source": "classifier_attribution_and_union_bank_heuristics",
        "classifier": classifier.__class__.__name__,
        "attribution_method": attribution_method,
        "risk_score": round(risk_score, 6),
        "verdict": verdict,
        "risk_drivers": risk_drivers,
        "protective_factors": protective_factors,
        "heuristic_control_count": control_count,
        "summary": (
            f"{verdict} from {attribution_method}: "
            f"{len(risk_drivers)} risk drivers, "
            f"{len(protective_factors)} mitigating drivers, "
            f"{control_count} Union Bank controls fired."
        ),
    }

@router.post("/explain")
async def explain_transaction(body: ExplainRequest, request: Request):
    """Get SHAP-based explanation for a transaction's risk score."""
    require_permission(request, "explain:view")
    orch = request.app.state.orchestrator
    if not orch or not orch._shap_explainer:
        return {"error": "Explainability engine not available"}

    import numpy as np

    feature_names = None
    feature_source = "request"
    domain_features = {}
    domain_controls = []
    if body.features is not None:
        features = np.array(body.features, dtype=np.float32).reshape(1, -1)
    else:
        engine = getattr(orch, "_engine", None)
        cache = getattr(engine, "_feature_cache", {}) if engine else {}
        cached = cache.get(body.txn_id)
        if not cached:
            return {
                "error": (
                    "Transaction features not found. Run the transaction through "
                    "the pipeline first, then request its explanation."
                ),
                "txn_id": body.txn_id,
            }
        feature_vector, feature_names = cached
        domain_cached = getattr(engine, "_domain_feature_cache", {}).get(body.txn_id)
        if domain_cached:
            domain_features, domain_controls = domain_cached
        features = np.asarray(feature_vector, dtype=np.float32).reshape(1, -1)
        feature_source = "feature_cache"

    classifier = getattr(orch, "_classifier", None)
    if not classifier or not getattr(classifier, "is_fitted", False):
        return {"error": "Classifier not fitted — cannot score explanation", "txn_id": body.txn_id}

    try:
        risk_score = float(classifier.predict_proba(features)[0])
    except Exception as exc:
        return {"error": f"Classifier scoring failed: {exc}", "txn_id": body.txn_id}

    explanation = orch._shap_explainer.explain_transaction(
        features,
        txn_id=body.txn_id,
        risk_score=risk_score,
        feature_names=feature_names,
    )
    if explanation is None:
        return {"error": "Model not fitted — cannot explain"}

    verdict = (
        "FRAUD"
        if explanation.risk_score >= 0.80
        else "SUSPICIOUS"
        if explanation.risk_score >= 0.50
        else "LEGITIMATE"
    )
    attribution_method = _explanation_method(orch._shap_explainer)

    return {
        "txn_id": explanation.txn_id,
        "risk_score": explanation.risk_score,
        "verdict": verdict,
        "narrative": explanation.narrative,
        "attribution_method": attribution_method,
        "base_value": getattr(explanation, "base_value", None),
        "explanation_ms": getattr(explanation, "explanation_ms", None),
        "feature_source": feature_source,
        "feature_count": int(features.shape[1]),
        "domain_feature_count": len(domain_features),
        "domain_features": domain_features,
        "domain_controls": domain_controls,
        "model_reasoning": _model_reasoning_payload(
            classifier=classifier,
            explanation=explanation,
            verdict=verdict,
            attribution_method=attribution_method,
            domain_controls=domain_controls,
        ),
        "top_features": [
            {
                "name": fc.feature_name,
                "description": orch._shap_explainer.FEATURE_DESCRIPTIONS.get(
                    fc.feature_name,
                    fc.feature_name,
                ),
                "value": fc.feature_value,
                "contribution": fc.shap_value,
                "direction": "increases_risk" if fc.direction == "risk_increase" else "decreases_risk",
            }
            for fc in explanation.contributions
        ],
    }


@router.get("/explainability/global")
async def global_feature_importance(request: Request):
    """Get global SHAP feature importance rankings."""
    require_any_permission(request, ("explain:view", "model:feedback", "audit:review"))
    orch = request.app.state.orchestrator
    if not orch or not orch._shap_explainer:
        return {"error": "Explainability engine not available"}

    features, feature_names, feature_source, sample_count = _global_importance_input(orch)
    importance = orch._shap_explainer.global_feature_importance(
        features=features,
        feature_names=feature_names,
    )
    snapshot = orch._shap_explainer.snapshot()
    snapshot.update({
        "feature_source": feature_source,
        "sample_count": sample_count,
        "feature_count": int(features.shape[1]) if features is not None else 0,
        "model_ready": bool(
            getattr(getattr(orch, "_classifier", None), "is_fitted", False)
        ),
    })
    return {
        "feature_importance": _importance_record(importance),
        "feature_details": importance,
        "snapshot": snapshot,
    }


# ── Model Drift ─────────────────────────────────────────────────────────────

@router.get("/drift")
async def drift_status(request: Request):
    """Get current model drift status."""
    require_any_permission(request, ("model:feedback", "risk:view", "system:view", "audit:review"))
    orch = request.app.state.orchestrator
    if not orch or not orch._drift_detector:
        return {"error": "Drift detector not available"}

    detector = orch._drift_detector
    report = detector.check_drift(force=True)
    if report is None:
        latest = detector.get_latest_report()
        if latest is not None:
            return _drift_payload(detector, latest, status="ready")

        snapshot = detector.snapshot()
        reference_size = int(snapshot.get("reference_size") or 0)
        current_size = int(snapshot.get("current_window_size") or 0)
        required_current_size = int(getattr(detector, "_detection_window", 500) // 2)

        base = {
            "severity": "NONE",
            "psi": 0.0,
            "ks_statistic": 0.0,
            "ks_p_value": 1.0,
            "js_divergence": 0.0,
            "reference_size": reference_size,
            "current_size": current_size,
            "required_current_size": required_current_size,
            "feature_drift": [],
            "snapshot": snapshot,
        }

        if reference_size == 0:
            return {
                **base,
                "status": "no_reference",
                "recommendation": "Train the classifier to establish a reference distribution.",
                "message": "No reference distribution has been set by model training.",
            }

        return {
            **base,
            "status": "warming",
            "recommendation": "Collect more live predictions before computing drift statistics.",
            "message": (
                "Reference distribution is ready; live prediction window is still warming."
            ),
        }

    return _drift_payload(detector, report, status="ready")


# ── Natural Language Query ───────────────────────────────────────────────────

@router.post("/query")
async def nl_query(body: NLQueryRequest, request: Request):
    """Ask the system a question in natural language (powered by Qwen 3.5)."""
    require_permission(request, "explain:view")
    orch = request.app.state.orchestrator
    if not orch or not orch._nl_query_engine:
        return {"error": "NL Query engine not available"}

    result = await orch._nl_query_engine.query(body.question)
    return {
        "query": result.query,
        "intent": result.intent,
        "answer": result.answer,
        "sources": result.sources,
        "confidence": result.confidence,
        "processing_ms": result.processing_ms,
        "model_used": result.model_used,
    }


# ── Consortium ───────────────────────────────────────────────────────────────

@router.get("/consortium")
async def consortium_status(request: Request):
    """Get cross-bank consortium hub status."""
    require_any_permission(request, ("cfr:check", "consortium:publish", "audit:review"))
    orch = request.app.state.orchestrator
    if not orch or not orch._consortium_hub:
        return {"error": "Consortium hub not available"}
    return orch._consortium_hub.snapshot()


@router.get("/consortium/alerts")
async def consortium_alerts(
    request: Request,
    fraud_type: int | None = None,
    severity_min: int = 1,
    limit: int = 50,
):
    """Query consortium fraud alerts from peer banks."""
    require_any_permission(request, ("cfr:check", "consortium:publish", "audit:review"))
    orch = request.app.state.orchestrator
    if not orch or not orch._consortium_hub:
        return {"error": "Consortium hub not available"}

    from src.blockchain.consortium import ConsortiumFraudType, AlertSeverity

    ft = ConsortiumFraudType(fraud_type) if fraud_type else None
    alerts = orch._consortium_hub.query_alerts(
        bank_id="UBI",
        fraud_type=ft,
        severity_min=AlertSeverity(severity_min),
        limit=limit,
    )
    return {
        "count": len(alerts),
        "alerts": [a.to_dict() for a in alerts],
    }


@router.post("/consortium/publish")
async def consortium_publish(body: ConsortiumPublishRequest, request: Request):
    """Publish a fraud alert to the consortium."""
    require_permission(request, "consortium:publish")
    orch = request.app.state.orchestrator
    if not orch or not orch._consortium_hub:
        return {"error": "Consortium hub not available"}

    alert = orch._consortium_hub.publish_alert(
        bank_id="UBI",
        account_id=body.account_id,
        risk_score=body.risk_score,
        fraud_type=body.fraud_type,
        severity=body.severity,
    )
    if alert is None:
        return {"error": "Alert not published — risk below threshold or invalid proof"}
    return {"status": "published", "alert": alert.to_dict()}


@router.post("/consortium/check")
async def consortium_check(body: AccountCheckRequest, request: Request):
    """Check if an account appears in consortium fraud alerts."""
    require_permission(request, "cfr:check")
    orch = request.app.state.orchestrator
    if not orch or not orch._consortium_hub:
        return {"error": "Consortium hub not available"}

    alerts = orch._consortium_hub.check_account(body.account_id)
    return {
        "account_id": body.account_id,
        "flagged": len(alerts) > 0,
        "alert_count": len(alerts),
        "alerts": [a.to_dict() for a in alerts],
    }

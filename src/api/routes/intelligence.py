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

router = APIRouter(prefix="/api/v1/intelligence", tags=["intelligence"])


# ── Request / Response Models ────────────────────────────────────────────────

class ExplainRequest(BaseModel):
    features: list[float] = Field(..., min_length=1)
    txn_id: str = "unknown"


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

@router.post("/explain")
async def explain_transaction(body: ExplainRequest, request: Request):
    """Get SHAP-based explanation for a transaction's risk score."""
    orch = request.app.state.orchestrator
    if not orch or not orch._shap_explainer:
        return {"error": "Explainability engine not available"}

    import numpy as np
    features = np.array(body.features, dtype=np.float32).reshape(1, -1)

    explanation = orch._shap_explainer.explain_transaction(
        features, txn_id=body.txn_id,
    )
    if explanation is None:
        return {"error": "Model not fitted — cannot explain"}

    return {
        "txn_id": explanation.txn_id,
        "risk_score": explanation.risk_score,
        "verdict": explanation.verdict,
        "narrative": explanation.narrative,
        "top_features": [
            {
                "name": fc.feature_name,
                "description": fc.description,
                "value": fc.value,
                "contribution": fc.contribution,
                "direction": fc.direction,
            }
            for fc in explanation.top_contributions
        ],
    }


@router.get("/explainability/global")
async def global_feature_importance(request: Request):
    """Get global SHAP feature importance rankings."""
    orch = request.app.state.orchestrator
    if not orch or not orch._shap_explainer:
        return {"error": "Explainability engine not available"}

    importance = orch._shap_explainer.global_feature_importance()
    return {
        "feature_importance": importance,
        "snapshot": orch._shap_explainer.snapshot(),
    }


# ── Model Drift ─────────────────────────────────────────────────────────────

@router.get("/drift")
async def drift_status(request: Request):
    """Get current model drift status."""
    orch = request.app.state.orchestrator
    if not orch or not orch._drift_detector:
        return {"error": "Drift detector not available"}

    report = orch._drift_detector.check_drift()
    if report is None:
        return {"status": "no_reference", "message": "No reference distribution set"}

    return {
        "severity": report.severity.name,
        "psi": round(report.psi, 6),
        "ks_statistic": round(report.ks_statistic, 6),
        "ks_p_value": round(report.ks_p_value, 6),
        "js_divergence": round(report.js_divergence, 6),
        "reference_size": report.reference_size,
        "current_size": report.current_size,
        "recommendation": report.recommendation,
        "feature_drift": [
            {
                "feature": fd.feature_name,
                "psi": round(fd.psi, 6),
                "severity": fd.severity.name,
            }
            for fd in (report.feature_drift or [])
        ],
        "snapshot": orch._drift_detector.snapshot(),
    }


# ── Natural Language Query ───────────────────────────────────────────────────

@router.post("/query")
async def nl_query(body: NLQueryRequest, request: Request):
    """Ask the system a question in natural language (powered by Qwen 3.5)."""
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

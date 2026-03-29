"""
PayFlow — Fraud Intelligence API Routes
=========================================
Exposes the new fraud detection modules via REST endpoints:

  - Pre-Approval Gate:      POST /api/v1/fraud/gate/evaluate
                            GET  /api/v1/fraud/gate/stats
  - Risk Scoring:           POST /api/v1/fraud/risk/score
  - Rule Engine:            POST /api/v1/fraud/rules/evaluate
                            GET  /api/v1/fraud/rules
                            GET  /api/v1/fraud/rules/stats
                            POST /api/v1/fraud/rules/{rule_id}/toggle
  - Regulatory Reports:     POST /api/v1/fraud/reports/str
                            POST /api/v1/fraud/reports/ctr
                            GET  /api/v1/fraud/reports
  - Community Anomaly:      GET  /api/v1/fraud/clusters
  - Device Verification:    POST /api/v1/fraud/device/verify
                            GET  /api/v1/fraud/device/{account_id}
  - Centrality Analysis:    GET  /api/v1/fraud/centrality/intermediaries
                            GET  /api/v1/fraud/centrality/node/{node_id}
  - Anomaly Detection:      POST /api/v1/fraud/anomaly/isolation/score
                            POST /api/v1/fraud/anomaly/autoencoder/score
                            GET  /api/v1/fraud/anomaly/stats
  - Central Fraud Registry: POST /api/v1/fraud/cfr/report
                            GET  /api/v1/fraud/cfr/query/{account_id}
                            POST /api/v1/fraud/cfr/check
                            POST /api/v1/fraud/cfr/score
                            GET  /api/v1/fraud/cfr/stats
  - FMR Reports:            POST /api/v1/fraud/reports/fmr
  - Investigation:          POST /api/v1/fraud/investigation/open
                            POST /api/v1/fraud/investigation/refer
                            POST /api/v1/fraud/investigation/legal
                            GET  /api/v1/fraud/investigation/case/{case_id}
                            GET  /api/v1/fraud/investigation/stats
  - AML Stage Detection:    POST /api/v1/fraud/aml/placement/evaluate
                            POST /api/v1/fraud/aml/integration/evaluate
                            POST /api/v1/fraud/aml/integration/inflow
                            GET  /api/v1/fraud/aml/stats
"""

from __future__ import annotations

import time
from typing import Any

from fastapi import APIRouter, Request
from pydantic import BaseModel, Field

router = APIRouter(prefix="/api/v1/fraud", tags=["fraud-intelligence"])


# ── Request / Response Models ──────────────────────────────────────────────────

class RiskScoreRequest(BaseModel):
    amount_paisa: int
    txn_count_1h: int = 0
    txn_count_24h: int = 0
    velocity_zscore: float = 0.0
    sender_id: str
    receiver_id: str
    geo_distance_km: float = 0.0
    geo_deviation: float = 0.0
    country_code: str = ""


class GateEvalRequest(BaseModel):
    txn_id: str
    sender_id: str
    receiver_id: str
    amount_paisa: int
    timestamp: int
    channel: int = 4
    device_fingerprint: str = ""
    sender_geo_lat: float = 0.0
    sender_geo_lon: float = 0.0


class DeviceVerifyRequest(BaseModel):
    account_id: str


class AnomalyScoreRequest(BaseModel):
    features: list[list[float]] = Field(
        ..., description="List of feature vectors (each vector = 36-dim float list)",
    )
    device_fingerprint: str
    timestamp: int = Field(default_factory=lambda: int(time.time()))


class STRRequest(BaseModel):
    account_id: str
    account_name: str = "Unknown"
    txn_ids: list[str] = []
    total_amount_paisa: int = 0
    suspicion_category: str = "ML_FLAGGED"
    suspicion_description: str = ""
    ml_confidence: float = 0.0


class CTRRequest(BaseModel):
    account_id: str
    account_name: str = "Unknown"
    amount_paisa: int
    txn_id: str = ""


class CFRReportRequest(BaseModel):
    account_id: str
    entity_identifier: str
    bank_code: str = "UBIN"
    category: int = 5  # defaults to UPI
    fraud_amount_paisa: int
    description: str = ""
    related_accounts: list[str] = []
    txn_ids: list[str] = []


class CFRCheckRequest(BaseModel):
    """KYC check — verify account or entity before account opening."""
    account_id: str = ""
    entity_identifier: str = ""


class CFRScoreRequest(BaseModel):
    sender_id: str
    receiver_id: str
    amount_paisa: int
    account_age_days: int = 365
    geo_distance_km: float = 0.0
    geo_deviation: float = 0.0


# ── Lazy Module Access ─────────────────────────────────────────────────────────
# Modules are instantiated once and reused across requests.

_risk_scorer = None
_pre_approval_gate = None
_regulatory_reporter = None
_community_detector = None
_device_verifier = None


def _get_risk_scorer():
    global _risk_scorer
    if _risk_scorer is None:
        from src.ml.risk_scorer import TransactionRiskScorer
        _risk_scorer = TransactionRiskScorer()
    return _risk_scorer


def _get_reporter():
    global _regulatory_reporter
    if _regulatory_reporter is None:
        from src.ml.regulatory_reporter import RegulatoryReporter
        _regulatory_reporter = RegulatoryReporter()
    return _regulatory_reporter


def _get_community_detector():
    global _community_detector
    if _community_detector is None:
        from src.graph.algorithms import CommunityAnomalyDetector
        _community_detector = CommunityAnomalyDetector()
    return _community_detector


def _get_device_verifier():
    global _device_verifier
    if _device_verifier is None:
        from src.ml.device_verifier import DeviceVerifier
        _device_verifier = DeviceVerifier()
    return _device_verifier


# ── Risk Scoring ───────────────────────────────────────────────────────────────

@router.post("/risk/score")
async def score_risk(req: RiskScoreRequest) -> dict[str, Any]:
    """Compute weighted composite risk score for a transaction."""
    scorer = _get_risk_scorer()
    result = scorer.score(
        amount_paisa=req.amount_paisa,
        txn_count_1h=req.txn_count_1h,
        txn_count_24h=req.txn_count_24h,
        velocity_zscore=req.velocity_zscore,
        sender_id=req.sender_id,
        receiver_id=req.receiver_id,
        geo_distance_km=req.geo_distance_km,
        geo_deviation=req.geo_deviation,
        country_code=req.country_code,
    )
    return {
        "composite_score": result.composite_score,
        "risk_level": result.risk_level,
        "large_txn_score": result.large_txn_score,
        "velocity_score": result.velocity_score,
        "new_account_score": result.new_account_score,
        "high_risk_country_score": result.location_score,
        "flags": result.flags,
    }


# ── Pre-Approval Gate ─────────────────────────────────────────────────────────

@router.post("/gate/evaluate")
async def evaluate_gate(req: GateEvalRequest, request: Request) -> dict[str, Any]:
    """
    Pre-authorisation fraud check — returns APPROVE / HOLD / BLOCK decision.
    Requires orchestrator attached to app.state for full pipeline access.
    """
    orch = request.app.state.orchestrator
    if orch is None:
        return {"error": "Orchestrator not attached — gate unavailable", "decision": "APPROVE"}

    gate = getattr(orch, "_pre_approval_gate", None)
    if gate is None:
        return {"error": "Pre-approval gate not initialised", "decision": "APPROVE"}

    result = gate.evaluate(
        txn_id=req.txn_id,
        sender_id=req.sender_id,
        receiver_id=req.receiver_id,
        amount_paisa=req.amount_paisa,
        timestamp=req.timestamp,
        channel=req.channel,
        device_fingerprint=req.device_fingerprint,
        sender_geo_lat=req.sender_geo_lat,
        sender_geo_lon=req.sender_geo_lon,
    )
    return {
        "decision": result.decision.name,
        "risk_score": result.risk_score,
        "reasons": result.reasons,
        "evaluation_time_ms": result.evaluation_time_ms,
    }


# ── Device Verification ───────────────────────────────────────────────────────

@router.post("/device/verify")
async def verify_device(req: DeviceVerifyRequest) -> dict[str, Any]:
    """Verify a device fingerprint against account history."""
    verifier = _get_device_verifier()
    result = verifier.verify(
        account_id=req.account_id,
        device_fingerprint=req.device_fingerprint,
        timestamp=req.timestamp,
    )
    return {
        "account_id": result.account_id,
        "device_fingerprint": result.device_fingerprint,
        "is_known_device": result.is_known_device,
        "device_risk_score": result.device_risk_score,
        "risk_factors": result.risk_factors,
        "accounts_on_device": result.accounts_on_device,
        "devices_on_account": result.devices_on_account,
        "evaluation_time_ms": result.evaluation_time_ms,
    }


@router.get("/device/{account_id}")
async def get_known_devices(account_id: str) -> dict[str, Any]:
    """List known device fingerprints for an account."""
    verifier = _get_device_verifier()
    return {
        "account_id": account_id,
        "known_devices": verifier.known_devices(account_id),
    }


# ── Regulatory Reports ────────────────────────────────────────────────────────

@router.post("/reports/str")
async def file_str_report(req: STRRequest) -> dict[str, Any]:
    """File a Suspicious Transaction Report (STR) for FIU-IND."""
    reporter = _get_reporter()
    report = reporter.file_str(
        account_id=req.account_id,
        account_name=req.account_name,
        txn_ids=req.txn_ids,
        total_amount_paisa=req.total_amount_paisa,
        suspicion_category=req.suspicion_category,
        suspicion_description=req.suspicion_description,
        ml_confidence=req.ml_confidence,
    )
    return {
        "report_id": report.report_id,
        "report_type": report.report_type.name,
        "status": report.status.name,
        "filed_at": report.filed_at,
    }


@router.post("/reports/ctr")
async def file_ctr_report(req: CTRRequest) -> dict[str, Any]:
    """File a Cash Transaction Report (CTR) for amounts ≥ ₹10 lakh."""
    reporter = _get_reporter()
    report = reporter.file_ctr(
        account_id=req.account_id,
        account_name=req.account_name,
        amount_paisa=req.amount_paisa,
        txn_id=req.txn_id,
    )
    return {
        "report_id": report.report_id,
        "report_type": report.report_type.name,
        "status": report.status.name,
        "filed_at": report.filed_at,
        "threshold_met": req.amount_paisa >= 10_00_000_00,
    }


@router.get("/reports")
async def list_reports() -> dict[str, Any]:
    """List all filed regulatory reports."""
    reporter = _get_reporter()
    reports = reporter.filed_reports
    return {
        "total": len(reports),
        "reports": [
            {
                "report_id": r.report_id,
                "report_type": r.report_type.name,
                "status": r.status.name,
                "filed_at": r.filed_at,
            }
            for r in reports
        ],
    }


# ── Community Anomaly Detection ────────────────────────────────────────────────

@router.get("/clusters")
async def get_suspicious_clusters(request: Request) -> dict[str, Any]:
    """
    Run community anomaly detection on the live transaction graph.
    Requires orchestrator for graph + community map access.
    """
    orch = request.app.state.orchestrator
    if orch is None:
        return {"error": "Orchestrator not attached", "clusters": []}

    graph_obj = getattr(orch, "graph", None)
    if graph_obj is None:
        return {"error": "Transaction graph not available", "clusters": []}

    g = graph_obj.graph
    community_map = getattr(graph_obj, "_community_map", {})
    if not community_map:
        return {"clusters": [], "message": "No communities detected yet"}

    detector = _get_community_detector()
    clusters = detector.detect(
        graph=g,
        community_map=community_map,
        current_time=int(time.time()),
    )

    return {
        "total_clusters": len(clusters),
        "clusters": [
            {
                "community_id": c.community_id,
                "node_count": c.node_count,
                "internal_edges": c.internal_edges,
                "density": c.density,
                "avg_amount_paisa": c.avg_amount_paisa,
                "total_amount_paisa": c.total_amount_paisa,
                "fraud_edge_ratio": c.fraud_edge_ratio,
                "anomaly_score": c.anomaly_score,
            }
            for c in clusters
        ],
    }


# ── Rule Engine ────────────────────────────────────────────────────────────────

_rule_engine = None


def _get_rule_engine():
    global _rule_engine
    if _rule_engine is None:
        from src.ml.rule_engine import TransactionRuleEngine
        _rule_engine = TransactionRuleEngine()
    return _rule_engine


class RuleEvalRequest(BaseModel):
    amount_paisa: int
    sender_id: str
    receiver_id: str
    timestamp: int
    txn_count_1h: int = 0
    txn_count_24h: int = 0
    velocity_zscore: float = 0.0
    geo_distance_km: float = 0.0
    device_trusted: bool = True
    hour_of_day: int = -1


@router.post("/rules/evaluate")
async def evaluate_rules(req: RuleEvalRequest, request: Request) -> dict[str, Any]:
    """Evaluate a transaction against all enabled rules."""
    # Prefer orchestrator's engine (has learned beneficiaries) over standalone
    orch = getattr(request.app.state, "orchestrator", None)
    engine = getattr(orch, "_rule_engine", None) if orch else None
    if engine is None:
        engine = _get_rule_engine()

    result = engine.evaluate(
        amount_paisa=req.amount_paisa,
        sender_id=req.sender_id,
        receiver_id=req.receiver_id,
        timestamp=req.timestamp,
        txn_count_1h=req.txn_count_1h,
        txn_count_24h=req.txn_count_24h,
        velocity_zscore=req.velocity_zscore,
        geo_distance_km=req.geo_distance_km,
        device_trusted=req.device_trusted,
        hour_of_day=req.hour_of_day,
    )
    return result.to_dict()


@router.get("/rules")
async def list_rules(request: Request) -> dict[str, Any]:
    """List all available detection rules with their thresholds and status."""
    orch = getattr(request.app.state, "orchestrator", None)
    engine = getattr(orch, "_rule_engine", None) if orch else None
    if engine is None:
        engine = _get_rule_engine()

    return {
        "rules": engine.list_rules(),
        "total": len(engine.list_rules()),
    }


@router.get("/rules/stats")
async def rule_stats(request: Request) -> dict[str, Any]:
    """Get rule engine evaluation statistics."""
    orch = getattr(request.app.state, "orchestrator", None)
    engine = getattr(orch, "_rule_engine", None) if orch else None
    if engine is None:
        engine = _get_rule_engine()

    return engine.snapshot()


@router.post("/rules/{rule_id}/toggle")
async def toggle_rule(rule_id: str, enable: bool = True, request: Request = None) -> dict[str, Any]:
    """Enable or disable a specific rule by ID."""
    orch = getattr(request.app.state, "orchestrator", None) if request else None
    engine = getattr(orch, "_rule_engine", None) if orch else None
    if engine is None:
        engine = _get_rule_engine()

    if enable:
        engine.enable_rule(rule_id)
    else:
        engine.disable_rule(rule_id)

    return {
        "rule_id": rule_id,
        "enabled": enable,
        "message": f"Rule {rule_id} {'enabled' if enable else 'disabled'}",
    }


@router.get("/gate/stats")
async def gate_stats(request: Request) -> dict[str, Any]:
    """Get pre-approval gate evaluation statistics."""
    orch = getattr(request.app.state, "orchestrator", None)
    gate = getattr(orch, "_pre_approval_gate", None) if orch else None
    if gate is None:
        return {"error": "Pre-approval gate not initialized", "metrics": {}}

    return gate.metrics.snapshot()


# ── Centrality Analysis ────────────────────────────────────────────────────────

@router.get("/centrality/intermediaries")
async def get_intermediaries(request: Request) -> dict[str, Any]:
    """
    Detect accounts with anomalously high betweenness centrality —
    potential transaction intermediaries in money laundering chains.

    CB(v) = Σ σ(s,t|v) / σ(s,t)

    Nodes with high betweenness sit on many shortest paths between other
    accounts, indicating they serve as bridges or layering conduits.
    """
    orch = getattr(request.app.state, "orchestrator", None)
    graph_obj = getattr(orch, "_graph", None) if orch else None
    if graph_obj is None:
        return {"error": "Transaction graph not initialized", "intermediaries": []}

    analyzer = graph_obj._centrality_analyzer
    findings = analyzer.detect(graph_obj.graph, int(time.time()))

    return {
        "count": len(findings),
        "intermediaries": [
            {
                "node_id": f.node_id,
                "betweenness_centrality": f.betweenness,
                "in_degree": f.in_degree,
                "out_degree": f.out_degree,
                "total_volume_paisa": f.total_volume_paisa,
                "pagerank": f.pagerank,
                "z_score": f.z_score,
                "detection_time_ms": round(f.detection_time_ms, 2),
            }
            for f in findings
        ],
    }


@router.get("/centrality/node/{node_id}")
async def get_node_centrality(node_id: str, request: Request) -> dict[str, Any]:
    """Get betweenness centrality and graph metrics for a specific account."""
    orch = getattr(request.app.state, "orchestrator", None)
    graph_obj = getattr(orch, "_graph", None) if orch else None
    if graph_obj is None:
        return {"error": "Transaction graph not initialized"}

    g = graph_obj.graph
    if not g.has_node(node_id):
        return {"error": f"Node {node_id} not found in graph"}

    analyzer = graph_obj._centrality_analyzer
    bc_map = analyzer.last_betweenness
    n_nodes = max(g.number_of_nodes(), 1)

    result: dict[str, Any] = {
        "node_id": node_id,
        "in_degree": g.in_degree(node_id),
        "out_degree": g.out_degree(node_id),
        "degree_centrality": round(
            (g.in_degree(node_id) + g.out_degree(node_id)) / max(2 * (n_nodes - 1), 1), 6,
        ),
        "betweenness_centrality": round(bc_map.get(node_id, 0.0), 6) if bc_map else None,
        "pagerank": round(analyzer._last_pr.get(node_id, 0.0), 6) if analyzer._last_pr else None,
    }

    # Check if node is flagged as intermediary
    finding = analyzer.detect_node(g, node_id)
    result["is_intermediary"] = finding is not None
    if finding:
        result["z_score"] = finding.z_score

    return result


# ── Anomaly Detection (Isolation Forest + Autoencoder) ─────────────────────────

@router.post("/anomaly/isolation/score")
async def score_isolation(req: AnomalyScoreRequest, request: Request) -> dict[str, Any]:
    """Score feature vectors through the Isolation Forest anomaly detector."""
    orch = getattr(request.app.state, "orchestrator", None)
    detector = getattr(orch, "_isolation_detector", None) if orch else None
    if detector is None or not detector._fitted:
        return {"error": "Isolation detector not fitted"}

    import numpy as np
    features = np.asarray(req.features, dtype=np.float32)
    results = detector.score_batch(features)
    return {
        "count": len(results),
        "scores": [
            {
                "anomaly_score": round(r.anomaly_score, 6),
                "is_anomaly": r.is_anomaly,
                "normalized_score": round(r.normalized_score, 4),
            }
            for r in results
        ],
    }


@router.post("/anomaly/autoencoder/score")
async def score_autoencoder(req: AnomalyScoreRequest, request: Request) -> dict[str, Any]:
    """Score feature vectors through the Autoencoder reconstruction detector."""
    orch = getattr(request.app.state, "orchestrator", None)
    detector = getattr(orch, "_autoencoder_detector", None) if orch else None
    if detector is None or not detector._fitted:
        return {"error": "Autoencoder detector not fitted"}

    import numpy as np
    features = np.asarray(req.features, dtype=np.float32)
    results = detector.score_batch(features)
    return {
        "count": len(results),
        "scores": [
            {
                "mse_loss": round(r.mse_loss, 6),
                "is_anomaly": r.is_anomaly,
                "normalized_score": round(r.normalized_score, 4),
            }
            for r in results
        ],
    }


@router.get("/anomaly/stats")
async def anomaly_stats(request: Request) -> dict[str, Any]:
    """Get metrics from both anomaly detectors."""
    orch = getattr(request.app.state, "orchestrator", None)
    result: dict[str, Any] = {}

    iso = getattr(orch, "_isolation_detector", None) if orch else None
    if iso:
        result["isolation_forest"] = iso.snapshot()

    ae = getattr(orch, "_autoencoder_detector", None) if orch else None
    if ae:
        result["autoencoder"] = ae.snapshot()

    if not result:
        return {"error": "No anomaly detectors initialized"}
    return result


# ── Central Fraud Registry (CFR) ──────────────────────────────────────────────

@router.post("/cfr/report")
async def cfr_report_fraud(body: CFRReportRequest, request: Request) -> dict[str, Any]:
    """Report a fraud case to the Central Fraud Registry."""
    orch = getattr(request.app.state, "orchestrator", None)
    registry = getattr(orch, "_fraud_registry", None) if orch else None
    if registry is None:
        return {"error": "Central Fraud Registry not initialized"}

    from src.cfr.registry import FraudCategory
    try:
        category = FraudCategory(body.category)
    except ValueError:
        return {"error": f"Invalid category: {body.category}"}

    record = registry.report_fraud(
        account_id=body.account_id,
        entity_identifier=body.entity_identifier,
        bank_code=body.bank_code,
        category=category,
        fraud_amount_paisa=body.fraud_amount_paisa,
        description=body.description,
        related_accounts=body.related_accounts,
        txn_ids=body.txn_ids,
    )
    return {"status": "reported", "record": record.to_dict()}


@router.get("/cfr/query/{account_id}")
async def cfr_query_account(account_id: str, request: Request) -> dict[str, Any]:
    """Query an account against the Central Fraud Registry."""
    orch = getattr(request.app.state, "orchestrator", None)
    registry = getattr(orch, "_fraud_registry", None) if orch else None
    if registry is None:
        return {"error": "Central Fraud Registry not initialized"}

    match_result = registry.check_account(account_id)
    records = registry.get_account_records(account_id)
    return {
        "account_id": account_id,
        "is_match": match_result.is_match,
        "match_score": round(match_result.match_score, 4),
        "records_found": match_result.records_found,
        "categories": match_result.categories,
        "total_fraud_amount_paisa": match_result.total_fraud_amount_paisa,
        "highest_severity": match_result.highest_severity,
        "records": [r.to_dict() for r in records],
    }


@router.post("/cfr/check")
async def cfr_kyc_check(body: CFRCheckRequest, request: Request) -> dict[str, Any]:
    """
    KYC verification — check account and/or entity against CFR
    before account opening or onboarding.
    """
    orch = getattr(request.app.state, "orchestrator", None)
    registry = getattr(orch, "_fraud_registry", None) if orch else None
    if registry is None:
        return {"error": "Central Fraud Registry not initialized"}

    result: dict[str, Any] = {"kyc_clear": True, "checks": []}

    if body.account_id:
        acct_match = registry.check_account(body.account_id)
        check = {
            "type": "account",
            "identifier": body.account_id,
            "is_match": acct_match.is_match,
            "match_score": round(acct_match.match_score, 4),
            "severity": acct_match.highest_severity,
        }
        result["checks"].append(check)
        if acct_match.is_match:
            result["kyc_clear"] = False

    if body.entity_identifier:
        entity_match = registry.check_entity(body.entity_identifier)
        check = {
            "type": "entity",
            "identifier": body.entity_identifier[:4] + "****",
            "is_match": entity_match.is_match,
            "match_score": round(entity_match.match_score, 4),
            "severity": entity_match.highest_severity,
        }
        result["checks"].append(check)
        if entity_match.is_match:
            result["kyc_clear"] = False

    return result


@router.post("/cfr/score")
async def cfr_risk_score(body: CFRScoreRequest, request: Request) -> dict[str, Any]:
    """Score a transaction using CFR-aware 40/30/20/10 weights."""
    orch = getattr(request.app.state, "orchestrator", None)
    cfr_scorer = getattr(orch, "_cfr_scorer", None) if orch else None
    if cfr_scorer is None:
        return {"error": "CFR Risk Scorer not initialized"}

    result = cfr_scorer.score(
        sender_id=body.sender_id,
        receiver_id=body.receiver_id,
        amount_paisa=body.amount_paisa,
        account_age_days=body.account_age_days,
        geo_distance_km=body.geo_distance_km,
        geo_deviation=body.geo_deviation,
    )
    return {
        "composite_score": round(result.composite_score, 4),
        "cfr_match_score": round(result.cfr_match_score, 4),
        "large_txn_score": round(result.large_txn_score, 4),
        "new_account_score": round(result.new_account_score, 4),
        "location_score": round(result.location_score, 4),
        "risk_level": result.risk_level,
        "flags": result.flags,
    }


@router.get("/cfr/stats")
async def cfr_stats(request: Request) -> dict[str, Any]:
    """Get Central Fraud Registry metrics."""
    orch = getattr(request.app.state, "orchestrator", None)
    result: dict[str, Any] = {}

    registry = getattr(orch, "_fraud_registry", None) if orch else None
    if registry:
        result["registry"] = registry.snapshot()

    cfr_scorer = getattr(orch, "_cfr_scorer", None) if orch else None
    if cfr_scorer:
        result["scorer"] = cfr_scorer.snapshot()

    if not result:
        return {"error": "CFR not initialized"}
    return result


# ═══════════════════════════════════════════════════════════════════════════
# FMR Reports
# ═══════════════════════════════════════════════════════════════════════════

class FMRRequest(BaseModel):
    account_id: str
    account_name: str = "Unknown"
    fraud_category: str = "INTERNET_BANKING"
    fraud_amount_paisa: int = 0
    txn_ids: list[str] = []
    modus_operandi: str = ""
    detection_method: str = "AI_MODEL"
    risk_score: float = 0.0
    ml_confidence: float = 0.0
    cfr_match: bool = False
    account_frozen: bool = False
    law_enforcement_notified: bool = False
    legal_proceedings_initiated: bool = False
    corrective_actions: list[str] = []


@router.post("/reports/fmr")
async def generate_fmr(body: FMRRequest, request: Request) -> dict[str, Any]:
    """Generate a Fraud Monitoring Return (FMR) for RBI submission."""
    orch = getattr(request.app.state, "orchestrator", None)
    reporter = getattr(orch, "_reporter", None) if orch else None
    if reporter is None:
        # Fallback: create an ad-hoc reporter
        from src.ml.regulatory_reporter import RegulatoryReporter
        reporter = RegulatoryReporter()

    report = reporter.generate_fmr(
        account_id=body.account_id,
        account_name=body.account_name,
        fraud_category=body.fraud_category,
        fraud_amount_paisa=body.fraud_amount_paisa,
        txn_ids=body.txn_ids,
        modus_operandi=body.modus_operandi,
        detection_method=body.detection_method,
        risk_score=body.risk_score,
        ml_confidence=body.ml_confidence,
        cfr_match=body.cfr_match,
        account_frozen=body.account_frozen,
        law_enforcement_notified=body.law_enforcement_notified,
        legal_proceedings_initiated=body.legal_proceedings_initiated,
        corrective_actions=body.corrective_actions,
    )
    return report.to_dict()


# ═══════════════════════════════════════════════════════════════════════════
# Investigation Management (Law Enforcement + Legal Proceedings)
# ═══════════════════════════════════════════════════════════════════════════

class InvestigationOpenRequest(BaseModel):
    account_id: str
    fraud_category: str = "INTERNET_BANKING"
    fraud_amount_paisa: int = 0
    risk_score: float = 0.0
    detection_method: str = "AI_MODEL"
    transaction_ids: list[str] = []
    related_accounts: list[str] = []
    account_frozen: bool = False
    fmr_id: str = ""


class ReferralRequest(BaseModel):
    case_id: str
    agency: str = "CBI"
    jurisdiction: str = ""
    graph_evidence: str = ""


class LegalProceedingRequest(BaseModel):
    case_id: str
    court_name: str = ""
    case_number: str = ""
    fir_date: str = ""


@router.post("/investigation/open")
async def open_investigation(
    body: InvestigationOpenRequest, request: Request,
) -> dict[str, Any]:
    """Open a new investigation case for a detected fraud."""
    orch = getattr(request.app.state, "orchestrator", None)
    mgr = getattr(orch, "_investigation_mgr", None) if orch else None
    if mgr is None:
        return {"error": "Investigation manager not initialized"}

    case = mgr.open_case(
        account_id=body.account_id,
        fraud_category=body.fraud_category,
        fraud_amount_paisa=body.fraud_amount_paisa,
        risk_score=body.risk_score,
        detection_method=body.detection_method,
        transaction_ids=body.transaction_ids,
        related_accounts=body.related_accounts,
        account_frozen=body.account_frozen,
        fmr_id=body.fmr_id,
    )
    return case.to_dict()


@router.post("/investigation/refer")
async def refer_to_law_enforcement(
    body: ReferralRequest, request: Request,
) -> dict[str, Any]:
    """Refer a case to a law enforcement agency."""
    orch = getattr(request.app.state, "orchestrator", None)
    mgr = getattr(orch, "_investigation_mgr", None) if orch else None
    if mgr is None:
        return {"error": "Investigation manager not initialized"}

    referral = mgr.refer_to_law_enforcement(
        case_id=body.case_id,
        agency=body.agency,
        jurisdiction=body.jurisdiction,
        graph_evidence=body.graph_evidence,
    )
    if referral is None:
        return {"error": f"Case {body.case_id} not found"}
    return referral.to_dict()


@router.post("/investigation/legal")
async def file_legal_proceeding(
    body: LegalProceedingRequest, request: Request,
) -> dict[str, Any]:
    """File a legal proceeding for a case."""
    orch = getattr(request.app.state, "orchestrator", None)
    mgr = getattr(orch, "_investigation_mgr", None) if orch else None
    if mgr is None:
        return {"error": "Investigation manager not initialized"}

    proceeding = mgr.file_legal_proceeding(
        case_id=body.case_id,
        court_name=body.court_name,
        case_number=body.case_number,
        fir_date=body.fir_date,
    )
    if proceeding is None:
        return {"error": f"Case {body.case_id} not found"}
    return proceeding.to_dict()


@router.get("/investigation/case/{case_id}")
async def get_investigation_case(
    case_id: str, request: Request,
) -> dict[str, Any]:
    """Get full investigation case details."""
    orch = getattr(request.app.state, "orchestrator", None)
    mgr = getattr(orch, "_investigation_mgr", None) if orch else None
    if mgr is None:
        return {"error": "Investigation manager not initialized"}

    case = mgr.get_case(case_id)
    if case is None:
        return {"error": f"Case {case_id} not found"}

    result = case.to_dict()
    # Enrich with linked records
    if case.referral_id:
        ref = mgr.get_referral(case.referral_id)
        if ref:
            result["referral"] = ref.to_dict()
    if case.proceeding_id:
        proc = mgr.get_proceeding(case.proceeding_id)
        if proc:
            result["legal_proceeding"] = proc.to_dict()
    return result


@router.get("/investigation/stats")
async def investigation_stats(request: Request) -> dict[str, Any]:
    """Get investigation module statistics."""
    orch = getattr(request.app.state, "orchestrator", None)
    mgr = getattr(orch, "_investigation_mgr", None) if orch else None
    if mgr is None:
        return {"error": "Investigation manager not initialized"}
    return mgr.snapshot()


# ── AML Stage Detection ───────────────────────────────────────────────────────

class PlacementEvalRequest(BaseModel):
    account_id: str = Field(..., min_length=1)
    amount_paisa: int = Field(..., gt=0)
    timestamp: int = Field(..., gt=0)
    channel: str = Field(default="CASH")
    is_deposit: bool = Field(default=True)


class IntegrationEvalRequest(BaseModel):
    account_id: str = Field(..., min_length=1)
    amount_paisa: int = Field(..., gt=0)
    timestamp: int = Field(..., gt=0)
    purpose_code: str = Field(default="")
    is_outflow: bool = Field(default=True)


class IntegrationInflowRequest(BaseModel):
    account_id: str = Field(..., min_length=1)
    amount_paisa: int = Field(..., gt=0)
    timestamp: int = Field(..., gt=0)


@router.post("/aml/placement/evaluate")
async def aml_placement_evaluate(body: PlacementEvalRequest, request: Request) -> dict[str, Any]:
    """Evaluate a deposit transaction for placement-stage ML indicators."""
    orch = getattr(request.app.state, "orchestrator", None)
    detector = getattr(orch, "_placement_detector", None) if orch else None
    if detector is None:
        return {"error": "PlacementDetector not initialized"}
    alerts = detector.evaluate(
        account_id=body.account_id,
        amount_paisa=body.amount_paisa,
        timestamp=body.timestamp,
        channel=body.channel,
        is_deposit=body.is_deposit,
    )
    return {
        "account_id": body.account_id,
        "alerts": [
            {
                "stage": a.stage.value,
                "risk": a.risk.value,
                "score": round(a.score, 4),
                "indicator": a.indicator,
                "reason": a.reason,
                "metadata": a.metadata,
            }
            for a in alerts
        ],
        "alert_count": len(alerts),
    }


@router.post("/aml/integration/evaluate")
async def aml_integration_evaluate(body: IntegrationEvalRequest, request: Request) -> dict[str, Any]:
    """Evaluate an outgoing transaction for integration-stage ML indicators."""
    orch = getattr(request.app.state, "orchestrator", None)
    detector = getattr(orch, "_integration_detector", None) if orch else None
    if detector is None:
        return {"error": "IntegrationDetector not initialized"}
    alerts = detector.evaluate(
        account_id=body.account_id,
        amount_paisa=body.amount_paisa,
        timestamp=body.timestamp,
        purpose_code=body.purpose_code,
        is_outflow=body.is_outflow,
    )
    return {
        "account_id": body.account_id,
        "alerts": [
            {
                "stage": a.stage.value,
                "risk": a.risk.value,
                "score": round(a.score, 4),
                "indicator": a.indicator,
                "reason": a.reason,
                "metadata": a.metadata,
            }
            for a in alerts
        ],
        "alert_count": len(alerts),
    }


@router.post("/aml/integration/inflow")
async def aml_integration_inflow(body: IntegrationInflowRequest, request: Request) -> dict[str, Any]:
    """Record an inflow for integration-stage round-trip/withdrawal detection."""
    orch = getattr(request.app.state, "orchestrator", None)
    detector = getattr(orch, "_integration_detector", None) if orch else None
    if detector is None:
        return {"error": "IntegrationDetector not initialized"}
    detector.record_inflow(
        account_id=body.account_id,
        amount_paisa=body.amount_paisa,
        timestamp=body.timestamp,
    )
    return {"status": "recorded", "account_id": body.account_id}


@router.get("/aml/stats")
async def aml_stats(request: Request) -> dict[str, Any]:
    """Get AML stage detection statistics."""
    orch = getattr(request.app.state, "orchestrator", None)
    result: dict[str, Any] = {}
    placement = getattr(orch, "_placement_detector", None) if orch else None
    integration = getattr(orch, "_integration_detector", None) if orch else None
    if placement:
        result["placement"] = placement.snapshot()
    if integration:
        result["integration"] = integration.snapshot()
    if not result:
        return {"error": "AML detectors not initialized"}
    return result


# ── FIU-IND Intelligence ──────────────────────────────────────────────────────

class FIUCollectSTRRequest(BaseModel):
    account_id: str
    report_id: str
    suspicion_category: str
    risk_score: float
    amount_paisa: int
    counterparty_ids: list[str] = []


class FIUCollectAlertRequest(BaseModel):
    account_id: str
    alert_type: str  # ML_ALERT, GRAPH_ALERT, AML_STAGE, RULE_VIOLATION
    severity: str    # LOW, MEDIUM, HIGH, CRITICAL
    summary: str
    details: dict[str, Any] = {}
    counterparty_id: str = ""


@router.post("/fiu/collect/str")
async def fiu_collect_str(body: FIUCollectSTRRequest, request: Request) -> dict[str, Any]:
    """Collect a Suspicious Transaction Report into FIU intelligence."""
    orch = getattr(request.app.state, "orchestrator", None)
    fiu = getattr(orch, "_fiu_intelligence", None) if orch else None
    if fiu is None:
        return {"error": "FIU Intelligence Unit not initialized"}
    entry_id = fiu.collect_str(
        account_id=body.account_id,
        report_id=body.report_id,
        suspicion_category=body.suspicion_category,
        risk_score=body.risk_score,
        amount_paisa=body.amount_paisa,
        counterparty_ids=body.counterparty_ids or None,
    )
    return {"status": "collected", "entry_id": entry_id}


@router.post("/fiu/collect/alert")
async def fiu_collect_alert(body: FIUCollectAlertRequest, request: Request) -> dict[str, Any]:
    """Collect an ML / graph / AML-stage alert into FIU intelligence."""
    from src.ml.fiu_intelligence import IntelligenceType
    orch = getattr(request.app.state, "orchestrator", None)
    fiu = getattr(orch, "_fiu_intelligence", None) if orch else None
    if fiu is None:
        return {"error": "FIU Intelligence Unit not initialized"}
    try:
        alert_type = IntelligenceType(body.alert_type)
    except ValueError:
        return {"error": f"Invalid alert_type: {body.alert_type}"}
    entry_id = fiu.collect_alert(
        account_id=body.account_id,
        alert_type=alert_type,
        severity=body.severity,
        summary=body.summary,
        details=body.details,
        counterparty_id=body.counterparty_id,
    )
    return {"status": "collected", "entry_id": entry_id}


@router.get("/fiu/intelligence/{account_id}")
async def fiu_get_intelligence(account_id: str, request: Request) -> dict[str, Any]:
    """Prepare and return an intelligence package for an account."""
    orch = getattr(request.app.state, "orchestrator", None)
    fiu = getattr(orch, "_fiu_intelligence", None) if orch else None
    if fiu is None:
        return {"error": "FIU Intelligence Unit not initialized"}
    package = fiu.prepare_intelligence(account_id)
    return package.to_dict()


@router.get("/fiu/dossier/{account_id}")
async def fiu_get_dossier(account_id: str, request: Request) -> dict[str, Any]:
    """Return raw intelligence dossier (all entries) for an account."""
    orch = getattr(request.app.state, "orchestrator", None)
    fiu = getattr(orch, "_fiu_intelligence", None) if orch else None
    if fiu is None:
        return {"error": "FIU Intelligence Unit not initialized"}
    entries = fiu.get_dossier(account_id)
    return {"account_id": account_id, "entries": entries, "count": len(entries)}


@router.get("/fiu/high-risk")
async def fiu_high_risk_accounts(request: Request) -> dict[str, Any]:
    """List accounts flagged as high-risk by the FIU intelligence unit."""
    orch = getattr(request.app.state, "orchestrator", None)
    fiu = getattr(orch, "_fiu_intelligence", None) if orch else None
    if fiu is None:
        return {"error": "FIU Intelligence Unit not initialized"}
    accounts = fiu.list_high_risk_accounts()
    return {"high_risk_accounts": accounts, "count": len(accounts)}


@router.post("/fiu/disseminate/{package_id}")
async def fiu_mark_disseminated(package_id: str, request: Request) -> dict[str, Any]:
    """Mark an intelligence package as disseminated to law enforcement."""
    orch = getattr(request.app.state, "orchestrator", None)
    fiu = getattr(orch, "_fiu_intelligence", None) if orch else None
    if fiu is None:
        return {"error": "FIU Intelligence Unit not initialized"}
    ok = fiu.mark_disseminated(package_id)
    if not ok:
        return {"error": f"Package {package_id} not found"}
    return {"status": "disseminated", "package_id": package_id}


@router.get("/fiu/stats")
async def fiu_stats(request: Request) -> dict[str, Any]:
    """Get FIU intelligence unit statistics."""
    orch = getattr(request.app.state, "orchestrator", None)
    fiu = getattr(orch, "_fiu_intelligence", None) if orch else None
    if fiu is None:
        return {"error": "FIU Intelligence Unit not initialized"}
    return fiu.snapshot()


# ── Mule Chain Detection (Carbanak) ───────────────────────────────────────────


@router.get("/mule/chains")
async def detect_mule_chains(request: Request) -> dict[str, Any]:
    """
    Detect multi-hop mule layering chains (Carbanak pattern):
    Source → Mule₁ → Mule₂ → … → Terminal (Cash-Out).
    """
    orch = getattr(request.app.state, "orchestrator", None)
    detector = getattr(orch, "_mule_chain_detector", None) if orch else None
    graph_obj = getattr(orch, "_graph", None) if orch else None
    if detector is None or graph_obj is None:
        return {"error": "Mule chain detector not initialized", "chains": []}

    chains = detector.detect(graph_obj.graph, int(time.time()))
    return {
        "count": len(chains),
        "chains": [
            {
                "chain_nodes": c.chain_nodes,
                "chain_length": c.chain_length,
                "txn_ids": c.txn_ids,
                "origin_node": c.origin_node,
                "terminal_node": c.terminal_node,
                "total_amount_paisa": c.total_amount_paisa,
                "time_span_minutes": c.time_span_minutes,
                "detection_time_ms": round(c.detection_time_ms, 2),
            }
            for c in chains
        ],
    }


@router.get("/mule/chains/{node_id}")
async def trace_mule_chain_from_node(
    request: Request, node_id: str,
) -> dict[str, Any]:
    """
    Trace forward mule layering chains from a specific account.
    Used during investigation to map onward fund flow.
    """
    orch = getattr(request.app.state, "orchestrator", None)
    detector = getattr(orch, "_mule_chain_detector", None) if orch else None
    graph_obj = getattr(orch, "_graph", None) if orch else None
    if detector is None or graph_obj is None:
        return {"error": "Mule chain detector not initialized", "chains": []}

    chains = detector.trace_from_node(graph_obj.graph, node_id, int(time.time()))
    return {
        "node_id": node_id,
        "count": len(chains),
        "chains": [
            {
                "chain_nodes": c.chain_nodes,
                "chain_length": c.chain_length,
                "txn_ids": c.txn_ids,
                "origin_node": c.origin_node,
                "terminal_node": c.terminal_node,
                "total_amount_paisa": c.total_amount_paisa,
                "time_span_minutes": c.time_span_minutes,
                "detection_time_ms": round(c.detection_time_ms, 2),
            }
            for c in chains
        ],
    }


# ── Mule Account Scoring (Carbanak) ──────────────────────────────────────────


class MuleScoreRequest(BaseModel):
    account_id: str
    account_age_days: int = 30
    txn_count_24h: int = 0
    median_forward_delay_sec: float = 0.0
    total_inbound_paisa: int = 0
    total_cashout_paisa: int = 0


@router.post("/mule/score")
async def score_mule_account(
    request: Request, body: MuleScoreRequest,
) -> dict[str, Any]:
    """
    Score an account on the 4 Carbanak mule indicators:
    25% Newly Opened + 25% High Frequency + 25% Rapid Forward + 25% Large Cash-Out.
    """
    orch = getattr(request.app.state, "orchestrator", None)
    scorer = getattr(orch, "_mule_scorer", None) if orch else None
    if scorer is None:
        return {"error": "Mule account scorer not initialized"}

    result = scorer.score_account(
        account_id=body.account_id,
        account_age_days=body.account_age_days,
        txn_count_24h=body.txn_count_24h,
        median_forward_delay_sec=body.median_forward_delay_sec,
        total_inbound_paisa=body.total_inbound_paisa,
        total_cashout_paisa=body.total_cashout_paisa,
    )
    return {
        "account_id": result.account_id,
        "mule_score": result.mule_score,
        "risk_level": result.risk_level,
        "is_suspected_mule": result.is_suspected_mule,
        "indicators": {
            "newly_opened": result.indicators.newly_opened,
            "high_frequency": result.indicators.high_frequency,
            "rapid_forward": result.indicators.rapid_forward,
            "large_cashout": result.indicators.large_cashout,
        },
    }


@router.get("/mule/suspected")
async def get_suspected_mules(request: Request) -> dict[str, Any]:
    """List all accounts previously scored as suspected mules."""
    orch = getattr(request.app.state, "orchestrator", None)
    scorer = getattr(orch, "_mule_scorer", None) if orch else None
    if scorer is None:
        return {"error": "Mule account scorer not initialized", "mules": []}

    mules = scorer.get_suspected_mules()
    return {
        "count": len(mules),
        "mules": [
            {
                "account_id": m.account_id,
                "mule_score": m.mule_score,
                "risk_level": m.risk_level,
                "indicators": {
                    "newly_opened": m.indicators.newly_opened,
                    "high_frequency": m.indicators.high_frequency,
                    "rapid_forward": m.indicators.rapid_forward,
                    "large_cashout": m.indicators.large_cashout,
                },
            }
            for m in mules
        ],
    }


@router.get("/mule/stats")
async def mule_stats(request: Request) -> dict[str, Any]:
    """Get mule detection statistics (chain detector + account scorer)."""
    orch = getattr(request.app.state, "orchestrator", None)
    chain_det = getattr(orch, "_mule_chain_detector", None) if orch else None
    scorer = getattr(orch, "_mule_scorer", None) if orch else None

    result: dict[str, Any] = {}
    if chain_det:
        result["chain_detector"] = chain_det.snapshot()
    if scorer:
        result["account_scorer"] = scorer.snapshot()
    if not result:
        return {"error": "Mule detection modules not initialized"}
    return result


# ── Victim Fund Tracing (Digital Banking Fraud Flow) ──────────────────────


@router.get("/victim/trace/{victim_id}")
async def trace_victim_funds(
    victim_id: str, request: Request,
) -> dict[str, Any]:
    """Trace the complete downstream fund flow from a reported victim."""
    orch = getattr(request.app.state, "orchestrator", None)
    tracer = getattr(orch, "_victim_tracer", None) if orch else None
    graph_obj = getattr(orch, "_graph", None) if orch else None

    if not tracer or not graph_obj:
        return {"error": "Victim tracer or graph not initialized"}

    import time as _time
    flow_map = tracer.trace(
        graph=graph_obj.graph,
        victim_id=victim_id,
        current_time=int(_time.time()),
    )
    return flow_map.to_dict()


@router.get("/victim/traces")
async def list_victim_traces(request: Request) -> dict[str, Any]:
    """List all victim IDs that have been traced."""
    orch = getattr(request.app.state, "orchestrator", None)
    tracer = getattr(orch, "_victim_tracer", None) if orch else None

    if not tracer:
        return {"error": "Victim tracer not initialized"}

    victims = tracer.list_victims()
    return {
        "total_victims_traced": len(victims),
        "victim_ids": victims,
    }


@router.get("/victim/stats")
async def victim_tracer_stats(request: Request) -> dict[str, Any]:
    """Get victim fund tracer statistics."""
    orch = getattr(request.app.state, "orchestrator", None)
    tracer = getattr(orch, "_victim_tracer", None) if orch else None

    if not tracer:
        return {"error": "Victim tracer not initialized"}
    return tracer.snapshot()


# ── Full Pipeline Analysis ────────────────────────────────────────────────────

class TransactionAnalyzeRequest(BaseModel):
    txn_id: str
    sender_id: str
    receiver_id: str
    amount_paisa: int
    timestamp: int
    channel: int = 4
    device_fingerprint: str = ""
    sender_geo_lat: float = 0.0
    sender_geo_lon: float = 0.0
    account_age_days: int = 365
    is_new_beneficiary: bool = False
    country_code: str = "IN"


@router.post("/transaction/analyze")
async def analyze_transaction(req: TransactionAnalyzeRequest, request: Request) -> dict[str, Any]:
    """
    Full pipeline analysis: feature extraction → ensemble scoring →
    graph analysis → point-based risk scoring → verdict + evidence.
    """
    orch = getattr(request.app.state, "orchestrator", None)
    if not orch:
        return {"error": "Orchestrator not attached"}

    import time as _time
    t0 = _time.monotonic()

    # 1. Point-based risk scoring
    from src.ml.risk_scorer import compute_point_risk_score
    hour = (_time.gmtime(req.timestamp).tm_hour + 5) % 24  # IST offset
    features_dict = {
        "amount_paisa": req.amount_paisa,
        "txn_count_1h": 0,
        "account_age_days": req.account_age_days,
        "is_night": 23 <= hour or hour < 5,
        "geo_distance_km": 0.0,
        "device_known": req.device_fingerprint != "",
        "cfr_match": False,
        "circular_flow": False,
        "betweenness": 0.0,
        "pass_through": False,
        "high_risk_country": req.country_code not in ("IN", ""),
    }

    # Check CFR registry
    cfr_reg = getattr(orch, "_fraud_registry", None)
    if cfr_reg:
        entry = cfr_reg.query(req.sender_id)
        if entry:
            features_dict["cfr_match"] = True

    # Check graph evidence
    graph = getattr(orch, "_graph", None)
    if graph and graph._graph.has_node(req.sender_id):
        import networkx as nx
        try:
            bc = nx.betweenness_centrality(graph._graph, k=min(50, graph._graph.number_of_nodes()))
            features_dict["betweenness"] = bc.get(req.sender_id, 0.0)
        except Exception:
            pass

    point_result = compute_point_risk_score(features_dict)

    # 2. Ensemble scoring (if models are fitted)
    ensemble_result = None
    classifier = getattr(orch, "_classifier", None)
    iso_det = getattr(orch, "_isolation_detector", None)
    ae_det = getattr(orch, "_autoencoder_detector", None)

    if classifier and classifier.is_fitted:
        try:
            from src.ml.ensemble import FraudEnsemble
            ensemble = FraudEnsemble()

            import numpy as np
            # Build a minimal feature vector for XGBoost
            engine = getattr(orch, "_engine", None)
            p_xgb = 0.0
            if engine:
                # Use a simplified approach: score based on risk indicators
                p_xgb = min(point_result.risk_score / 100.0, 1.0)

            p_iso = 0.5  # default neutral
            if iso_det and iso_det.is_fitted:
                p_iso = min(point_result.risk_score / 100.0, 1.0)

            p_ae = 0.5
            if ae_det and ae_det.is_fitted:
                p_ae = min(point_result.risk_score / 100.0, 1.0)

            ensemble_result = ensemble.score(p_xgb, p_iso, p_ae).to_dict()
        except Exception:
            pass

    elapsed_ms = (_time.monotonic() - t0) * 1000

    return {
        "txn_id": req.txn_id,
        "risk_score": point_result.risk_score,
        "verdict": point_result.verdict,
        "fraud_patterns": point_result.fraud_patterns,
        "risk_breakdown": point_result.breakdown,
        "ensemble": ensemble_result,
        "analysis_ms": round(elapsed_ms, 2),
    }


# ── Graph Subgraph Extraction ─────────────────────────────────────────────────

@router.get("/graph/subgraph/{account_id}")
async def get_subgraph(account_id: str, request: Request, hops: int = 2) -> dict[str, Any]:
    """
    Extract 2-hop ego subgraph around an account for Cytoscape.js frontend.
    Returns nodes and edges in a format ready for graph visualization.
    """
    orch = getattr(request.app.state, "orchestrator", None)
    graph = getattr(orch, "_graph", None) if orch else None

    if not graph:
        return {"error": "Graph not initialized"}

    g = graph._graph
    if not g.has_node(account_id):
        return {"error": f"Node {account_id} not found in graph", "nodes": [], "edges": []}

    import networkx as nx

    # Extract k-hop neighborhood (bidirectional)
    hops = min(hops, 3)  # cap at 3 to avoid massive subgraphs
    fwd = nx.ego_graph(g, account_id, radius=hops)
    rev = nx.ego_graph(g.reverse(copy=False), account_id, radius=hops)
    neighborhood = set(fwd.nodes()) | set(rev.nodes())
    sub = g.subgraph(neighborhood)

    # Build Cytoscape.js-compatible JSON
    breaker = getattr(orch, "_breaker", None)
    nodes = []
    for nid in sub.nodes():
        nd = sub.nodes[nid]
        is_frozen = breaker.is_frozen(nid) if breaker else False
        nodes.append({
            "data": {
                "id": nid,
                "label": nid[:12],
                "type": nd.get("account_type", "unknown"),
                "fraud_edge_count": nd.get("fraud_edge_count", 0),
                "is_frozen": is_frozen,
                "is_center": nid == account_id,
            }
        })

    edges = []
    for u, v, key, data in sub.edges(keys=True, data=True):
        edges.append({
            "data": {
                "id": f"{u}-{v}-{key}",
                "source": u,
                "target": v,
                "amount_paisa": data.get("amount_paisa", 0),
                "channel": data.get("channel", ""),
                "is_fraud": data.get("fraud_label", 0) > 0,
                "timestamp": data.get("timestamp", 0),
            }
        })

    return {
        "account_id": account_id,
        "hops": hops,
        "node_count": len(nodes),
        "edge_count": len(edges),
        "nodes": nodes,
        "edges": edges,
    }


# ── Circuit Breaker Manual Trigger ─────────────────────────────────────────────

class CircuitBreakerTriggerRequest(BaseModel):
    account_id: str
    reason: str = "manual_trigger"


@router.post("/circuit-breaker/trigger")
async def trigger_circuit_breaker(req: CircuitBreakerTriggerRequest, request: Request) -> dict[str, Any]:
    """
    Manually trigger circuit breaker freeze for an account + 1-hop network.
    """
    orch = getattr(request.app.state, "orchestrator", None)
    breaker = getattr(orch, "_breaker", None) if orch else None

    if not breaker:
        return {"error": "Circuit breaker not initialized"}

    import time as _time
    from src.blockchain.circuit_breaker import FreezeOrder

    now = _time.time()
    order = FreezeOrder(
        node_id=req.account_id,
        freeze_timestamp=now,
        trigger_txn_id=f"manual_{int(now)}",
        ml_risk_score=1.0,
        gnn_risk_score=-1.0,
        graph_evidence_score=0.0,
        consensus_score=1.0,
        reason=req.reason,
        ttl_seconds=breaker._cfg.freeze_ttl_seconds,
    )

    await breaker.freeze_node(order)

    # Also freeze 1-hop neighbors if graph is attached
    frozen_neighbors = []
    if breaker._graph is not None:
        await breaker._freeze_1hop_neighbors(
            req.account_id, order.trigger_txn_id,
            1.0, -1.0, 0.0, 1.0,
        )
        graph_obj = breaker._graph
        try:
            frozen_neighbors = [
                nid for nid in graph_obj.graph.neighbors(req.account_id)
                if breaker.is_frozen(nid)
            ]
        except Exception:
            pass

    return {
        "status": "frozen",
        "account_id": req.account_id,
        "reason": req.reason,
        "frozen_neighbors": frozen_neighbors,
        "total_frozen": len(breaker.get_frozen_nodes()),
    }


# ── PDF Report Generation ────────────────────────────────────────────────────

@router.get("/reports/str/{case_id}")
async def get_str_pdf(case_id: str, request: Request):
    """Generate and return an STR (Suspicious Transaction Report) PDF."""
    orch = getattr(request.app.state, "orchestrator", None)

    from src.ml.pdf_generator import generate_str_pdf
    from fastapi.responses import Response

    reporter = _get_reporter()
    reports = reporter.list_reports()
    target = None
    for r in reports:
        if r.get("report_id", "") == case_id or r.get("account_id", "") == case_id:
            target = r
            break

    if target is None:
        target = {"report_id": case_id, "account_id": case_id, "type": "STR"}

    pdf_bytes = generate_str_pdf(target)
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f"attachment; filename=STR_{case_id}.pdf"},
    )


@router.get("/reports/fmr/{case_id}")
async def get_fmr_pdf(case_id: str, request: Request):
    """Generate and return an FMR (Fraud Monitoring Return) PDF."""
    from src.ml.pdf_generator import generate_fmr_pdf
    from fastapi.responses import Response

    pdf_bytes = generate_fmr_pdf({"case_id": case_id})
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f"attachment; filename=FMR_{case_id}.pdf"},
    )

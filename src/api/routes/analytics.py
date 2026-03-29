"""
PayFlow -- Analytics Endpoints
==============================
Aggregated risk analytics, fraud typology breakdown, velocity trends,
and temporal risk concentration data for the enhanced dashboard.

Routes:

    GET  /api/v1/analytics/risk-distribution   — Risk score histogram buckets
    GET  /api/v1/analytics/fraud-typology       — Fraud pattern breakdown counts
    GET  /api/v1/analytics/velocity-trends      — Per-account velocity windows
    GET  /api/v1/analytics/temporal-heatmap     — Risk concentration over time
    GET  /api/v1/analytics/threat-summary       — Aggregate threat level assessment
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict

from fastapi import APIRouter, Query, Request

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/analytics", tags=["analytics"])


@router.get("/risk-distribution")
async def risk_distribution(request: Request):
    """
    Risk score distribution across buckets.
    Returns counts in 10-percentile buckets for dashboard histogram.
    """
    orch = request.app.state.orchestrator
    if orch is None:
        return {"buckets": [], "total": 0}

    classifier = getattr(orch, "_classifier", None)
    if classifier is None:
        return {"buckets": [], "total": 0}

    # Collect recent scores from the classifier's prediction cache
    recent_scores = getattr(classifier, "_recent_scores", [])
    if not recent_scores:
        return {"buckets": [0] * 10, "total": 0}

    buckets = [0] * 10
    for score in recent_scores:
        idx = min(int(score * 10), 9)
        buckets[idx] += 1

    return {
        "buckets": buckets,
        "total": len(recent_scores),
        "mean": sum(recent_scores) / len(recent_scores) if recent_scores else 0,
        "p95": sorted(recent_scores)[int(len(recent_scores) * 0.95)] if len(recent_scores) > 1 else 0,
    }


@router.get("/fraud-typology")
async def fraud_typology_breakdown(request: Request):
    """
    Count of detected fraud patterns by type from the transaction graph.
    """
    orch = request.app.state.orchestrator
    if orch is None:
        return {"typology": {}, "total": 0}

    graph = getattr(orch, "_graph", None)
    if graph is None:
        return {"typology": {}, "total": 0}

    g = graph._graph
    from src.ingestion.schemas import FraudPattern
    label_names = {e.value: e.name for e in FraudPattern}

    counts: dict[str, int] = defaultdict(int)
    total_fraud = 0
    for _, _, data in g.edges(data=True):
        fl = data.get("fraud_label", 0)
        if fl > 0:
            name = label_names.get(fl, f"UNKNOWN_{fl}")
            counts[name] += 1
            total_fraud += 1

    return {
        "typology": dict(counts),
        "total": total_fraud,
    }


@router.get("/velocity-trends")
async def velocity_trends(
    request: Request,
    window_minutes: int = Query(30, ge=5, le=360),
    top_n: int = Query(10, ge=1, le=50),
):
    """
    Top-N accounts by transaction velocity in the specified time window.
    Returns per-account transaction counts and total volume.
    """
    orch = request.app.state.orchestrator
    if orch is None:
        return {"accounts": [], "window_minutes": window_minutes}

    graph = getattr(orch, "_graph", None)
    if graph is None:
        return {"accounts": [], "window_minutes": window_minutes}

    g = graph._graph
    cutoff = time.time() - window_minutes * 60

    account_stats: dict[str, dict] = defaultdict(lambda: {"count": 0, "volume_paisa": 0, "fraud_count": 0})
    for u, v, data in g.edges(data=True):
        ts = data.get("timestamp", 0)
        if ts < cutoff:
            continue
        amount = data.get("amount_paisa", 0)
        fl = data.get("fraud_label", 0)
        for acct in (u, v):
            account_stats[acct]["count"] += 1
            account_stats[acct]["volume_paisa"] += amount
            if fl > 0:
                account_stats[acct]["fraud_count"] += 1

    # Sort by count descending
    sorted_accounts = sorted(account_stats.items(), key=lambda x: x[1]["count"], reverse=True)[:top_n]

    return {
        "accounts": [
            {"account_id": acct, **stats}
            for acct, stats in sorted_accounts
        ],
        "window_minutes": window_minutes,
    }


@router.get("/temporal-heatmap")
async def temporal_heatmap(
    request: Request,
    bucket_seconds: int = Query(60, ge=10, le=600),
    lookback_minutes: int = Query(30, ge=5, le=120),
):
    """
    Temporal risk concentration — transaction volume and fraud density
    bucketed over time for a heatmap visualization.
    """
    orch = request.app.state.orchestrator
    if orch is None:
        return {"buckets": [], "bucket_seconds": bucket_seconds}

    graph = getattr(orch, "_graph", None)
    if graph is None:
        return {"buckets": [], "bucket_seconds": bucket_seconds}

    g = graph._graph
    now = time.time()
    cutoff = now - lookback_minutes * 60

    time_buckets: dict[int, dict] = defaultdict(
        lambda: {"txn_count": 0, "fraud_count": 0, "total_paisa": 0, "max_amount": 0}
    )

    for _, _, data in g.edges(data=True):
        ts = data.get("timestamp", 0)
        if ts < cutoff:
            continue
        bucket_idx = int((ts - cutoff) // bucket_seconds)
        bucket = time_buckets[bucket_idx]
        bucket["txn_count"] += 1
        amount = data.get("amount_paisa", 0)
        bucket["total_paisa"] += amount
        bucket["max_amount"] = max(bucket["max_amount"], amount)
        if data.get("fraud_label", 0) > 0:
            bucket["fraud_count"] += 1

    num_buckets = int((lookback_minutes * 60) / bucket_seconds)
    result = []
    for i in range(num_buckets):
        bucket_data = time_buckets.get(i, {"txn_count": 0, "fraud_count": 0, "total_paisa": 0, "max_amount": 0})
        bucket_start = cutoff + i * bucket_seconds
        result.append({
            "bucket_start": bucket_start,
            "bucket_end": bucket_start + bucket_seconds,
            **bucket_data,
        })

    return {
        "buckets": result,
        "bucket_seconds": bucket_seconds,
        "lookback_minutes": lookback_minutes,
    }


@router.get("/threat-summary")
async def threat_summary(request: Request):
    """
    Aggregate threat level assessment — overall system risk posture
    based on real-time metrics.
    """
    orch = request.app.state.orchestrator
    if orch is None:
        return {"threat_level": "unknown", "indicators": []}

    # Gather all available metrics
    breaker = getattr(orch, "_breaker", None)
    graph = getattr(orch, "_graph", None)
    hw_snap = orch.profiler.snapshot() if hasattr(orch, "profiler") else {}
    sim = getattr(orch, "_threat_engine", None)

    indicators = []
    severity_score = 0.0

    # 1. Frozen nodes count
    frozen_count = len(breaker.get_frozen_nodes()) if breaker else 0
    if frozen_count > 5:
        indicators.append({"signal": "HIGH_FREEZE_RATE", "detail": f"{frozen_count} accounts frozen", "severity": "critical"})
        severity_score += 0.3
    elif frozen_count > 0:
        indicators.append({"signal": "ACTIVE_FREEZES", "detail": f"{frozen_count} accounts frozen", "severity": "high"})
        severity_score += 0.15

    # 2. Active simulated attacks
    active_attacks = 0
    if sim:
        active_attacks = len([s for s in getattr(sim, "_scenarios", {}).values()
                             if getattr(s, "status", "") == "running"])
    if active_attacks > 2:
        indicators.append({"signal": "MULTI_VECTOR_ATTACK", "detail": f"{active_attacks} concurrent attacks", "severity": "critical"})
        severity_score += 0.25
    elif active_attacks > 0:
        indicators.append({"signal": "ACTIVE_ATTACK", "detail": f"{active_attacks} ongoing attack(s)", "severity": "high"})
        severity_score += 0.1

    # 3. VRAM pressure
    vram_used = hw_snap.get("gpu_vram_used_mb", 0)
    vram_total = hw_snap.get("gpu_vram_total_mb", 1)
    vram_pct = vram_used / vram_total * 100 if vram_total > 0 else 0
    if vram_pct > 90:
        indicators.append({"signal": "VRAM_PRESSURE", "detail": f"{vram_pct:.0f}% VRAM in use", "severity": "high"})
        severity_score += 0.1

    # 4. Fraud edge density
    if graph:
        g = graph._graph
        total_edges = g.number_of_edges()
        fraud_edges = sum(1 for _, _, d in g.edges(data=True) if d.get("fraud_label", 0) > 0)
        fraud_ratio = fraud_edges / total_edges if total_edges > 0 else 0
        if fraud_ratio > 0.15:
            indicators.append({"signal": "HIGH_FRAUD_DENSITY", "detail": f"{fraud_ratio:.1%} edges flagged", "severity": "critical"})
            severity_score += 0.2
        elif fraud_ratio > 0.05:
            indicators.append({"signal": "ELEVATED_FRAUD", "detail": f"{fraud_ratio:.1%} edges flagged", "severity": "medium"})
            severity_score += 0.05

    # Determine overall threat level
    if severity_score >= 0.5:
        threat_level = "critical"
    elif severity_score >= 0.25:
        threat_level = "high"
    elif severity_score >= 0.1:
        threat_level = "elevated"
    else:
        threat_level = "normal"

    return {
        "threat_level": threat_level,
        "severity_score": round(severity_score, 2),
        "frozen_count": frozen_count,
        "active_attacks": active_attacks,
        "indicators": indicators,
    }

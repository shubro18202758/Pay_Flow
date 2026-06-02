"""Human analyst escalation queue for HITL fraud investigations.

The InvestigatorAgent posts low-confidence or analyst-required decisions here.
The API normalizes each payload into an analyst case with status, priority,
evidence summary, immutable audit hash, and SSE updates for the frontend.
"""

from __future__ import annotations

import hashlib
import json
import time
import uuid
from typing import Literal

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from src.api.rbac import require_any_permission, require_permission

router = APIRouter(prefix="/api/v1/analyst", tags=["analyst"])

_escalation_store: list[dict] = []
_MAX_ESCALATIONS = 250


class EscalationAck(BaseModel):
    ack_id: str
    status: str


class EscalationDecisionRequest(BaseModel):
    decision: Literal["approve", "reject", "escalate"]
    analyst: str = Field(default="union_bank_analyst", min_length=2, max_length=80)
    reason: str = Field(default="analyst_decision", min_length=2, max_length=240)


def _now() -> float:
    return time.time()


def _audit_hash(record: dict) -> str:
    material = json.dumps(record, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(material.encode()).hexdigest()


def _as_float(payload: dict, key: str, default: float = 0.0) -> float:
    try:
        return float(payload.get(key, default))
    except (TypeError, ValueError):
        return default


def _priority(payload: dict) -> str:
    confidence = _as_float(payload, "agent_confidence", 1.0)
    threshold = _as_float(payload, "confidence_threshold", 0.75)
    ml_score = _as_float(payload, "ml_score")
    gnn_score = _as_float(payload, "gnn_score", -1.0)
    gap = max(0.0, threshold - confidence)
    risk = max(ml_score, gnn_score, 0.0) + gap
    if risk >= 1.45:
        return "critical"
    if risk >= 1.05:
        return "high"
    if risk >= 0.75:
        return "medium"
    return "low"


def _evidence_summary(payload: dict) -> dict:
    graph_context = payload.get("graph_context") if isinstance(payload.get("graph_context"), dict) else {}
    evidence = payload.get("evidence_collected") if isinstance(payload.get("evidence_collected"), dict) else {}
    patterns = graph_context.get("patterns") if isinstance(graph_context.get("patterns"), dict) else {}
    reasoning_trace = payload.get("reasoning_trace") if isinstance(payload.get("reasoning_trace"), list) else []
    return {
        "reasoning_steps": len(reasoning_trace),
        "evidence_keys": sorted(evidence.keys())[:8],
        "graph_available": bool(graph_context.get("available")),
        "subgraph_nodes": (graph_context.get("subgraph") or {}).get("nodes", 0) if isinstance(graph_context.get("subgraph"), dict) else 0,
        "subgraph_edges": (graph_context.get("subgraph") or {}).get("edges", 0) if isinstance(graph_context.get("subgraph"), dict) else 0,
        "mule_network_detected": bool(patterns.get("mule_network_detected")),
        "cycles_found": int(patterns.get("cycles_found", 0) or 0),
    }


async def _publish_analyst_event(event_type: str, case: dict) -> None:
    try:
        from src.api.events import EventBroadcaster

        broadcaster = EventBroadcaster.get()
        await broadcaster.publish("analyst", {
            "type": event_type,
            "case": case,
            "timestamp": _now(),
        })
        if event_type == "escalation_decided":
            node_status = "normal" if case.get("status") == "rejected" else "suspicious"
            resolution = {
                "type": "case_resolution",
                "case": case,
                "txn_id": case.get("txn_id"),
                "node_id": case.get("node_id"),
                "case_id": case.get("case_id"),
                "status": case.get("status"),
                "node_status": node_status,
                "audit_hash": case.get("audit_hash"),
                "decision_history": case.get("decision_history", []),
                "timestamp": _now(),
                "fanout": {
                    "activity_lifecycle": True,
                    "graph_node_status": bool(case.get("node_id")),
                    "query_groups": [
                        "escalations",
                        "recent-blocks",
                        "snapshot",
                        "topology",
                        "verdicts",
                        "case-trace",
                        "risk-distribution",
                        "fraud-typology",
                        "velocity-trends",
                        "temporal-heatmap",
                        "threat-summary",
                        "countermeasure-proposals",
                        "circuit-breaker",
                        "fraud",
                    ],
                },
            }
            await broadcaster.publish("transaction_decision", resolution)
            if case.get("node_id"):
                await broadcaster.publish("graph", {
                    "type": "node_status_changed",
                    "node_id": case["node_id"],
                    "status": node_status,
                    "source": "analyst_decision",
                    "case_id": case.get("case_id"),
                    "txn_id": case.get("txn_id"),
                    "audit_hash": case.get("audit_hash"),
                })
    except Exception:
        pass


def _build_case(ack_id: str, payload: dict) -> dict:
    received_at = _now()
    case = {
        "ack_id": ack_id,
        "case_id": f"HITL-{ack_id[:8].upper()}",
        "escalation_id": str(payload.get("escalation_id") or ack_id),
        "status": "pending_review",
        "priority": _priority(payload),
        "txn_id": str(payload.get("txn_id") or ""),
        "node_id": str(payload.get("node_id") or ""),
        "detected_typology": payload.get("detected_typology"),
        "agent_confidence": _as_float(payload, "agent_confidence"),
        "confidence_threshold": _as_float(payload, "confidence_threshold", 0.0),
        "ml_score": _as_float(payload, "ml_score"),
        "gnn_score": _as_float(payload, "gnn_score", -1.0),
        "recommended_action": str(payload.get("recommended_action") or "ESCALATE_TO_HUMAN"),
        "evidence_summary": _evidence_summary(payload),
        "payload": payload,
        "received_at": received_at,
        "updated_at": received_at,
        "sla_seconds": 3600,
        "analyst": None,
        "analyst_reason": None,
        "decision_history": [],
    }
    case["audit_hash"] = _audit_hash({
        "ack_id": ack_id,
        "payload": payload,
        "received_at": received_at,
    })
    return case


@router.post("/escalation", response_model=EscalationAck)
async def receive_escalation(payload: dict, request: Request = None) -> EscalationAck:
    if request is not None:
        require_any_permission(request, ("case:launch", "case:decide"))
    ack_id = str(uuid.uuid4())
    case = _build_case(ack_id, payload)
    _escalation_store.append(case)
    if len(_escalation_store) > _MAX_ESCALATIONS:
        del _escalation_store[: len(_escalation_store) - _MAX_ESCALATIONS]
    await _publish_analyst_event("escalation_received", case)
    return EscalationAck(ack_id=ack_id, status="received")


@router.get("/escalations")
async def list_escalations(status: str | None = None, limit: int = 100) -> list[dict]:
    cases = _escalation_store
    if status:
        cases = [case for case in cases if case.get("status") == status]
    return list(reversed(cases[-max(1, min(limit, _MAX_ESCALATIONS)):]))


@router.get("/escalations/{ack_id}")
async def get_escalation(ack_id: str) -> dict:
    for case in reversed(_escalation_store):
        if case["ack_id"] == ack_id or case["case_id"] == ack_id:
            return case
    raise HTTPException(404, f"Unknown analyst escalation: {ack_id}")


@router.post("/escalations/{ack_id}/decision")
async def decide_escalation(ack_id: str, body: EscalationDecisionRequest, request: Request = None) -> dict:
    if request is not None:
        require_permission(request, "case:decide")
    case = await get_escalation(ack_id)
    decision_status = {
        "approve": "approved",
        "reject": "rejected",
        "escalate": "escalated",
    }[body.decision]
    now = _now()
    decision = {
        "decision": body.decision,
        "status": decision_status,
        "analyst": body.analyst,
        "reason": body.reason,
        "timestamp": now,
    }
    case["status"] = decision_status
    case["analyst"] = body.analyst
    case["analyst_reason"] = body.reason
    case["updated_at"] = now
    case["decision_history"] = [*case.get("decision_history", []), decision]
    case["audit_hash"] = _audit_hash({
        "ack_id": case["ack_id"],
        "payload": case["payload"],
        "decision_history": case["decision_history"],
    })
    await _publish_analyst_event("escalation_decided", case)
    return case


def clear_store() -> None:
    _escalation_store.clear()

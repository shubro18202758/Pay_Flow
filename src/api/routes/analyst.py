"""
PayFlow -- Human Analyst Escalation API
=========================================
Mock FastAPI endpoint for receiving HITL escalation payloads from the
InvestigatorAgent when confidence falls below threshold on complex
money laundering typologies.

In production, this endpoint would connect to the bank's internal case
management system (e.g., Actimize, NICE, or a custom AML workbench).
For PayFlow's purposes, it stores escalations in-memory for testing
and demonstration.

Routes:
    POST /api/v1/analyst/escalation  — Receive an HITL escalation payload
    GET  /api/v1/analyst/escalations — List all received escalations
"""

from __future__ import annotations

import uuid

from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter(prefix="/api/v1/analyst", tags=["analyst"])

# In-memory store for received escalations
_escalation_store: list[dict] = []


class EscalationAck(BaseModel):
    ack_id: str
    status: str


@router.post("/escalation", response_model=EscalationAck)
async def receive_escalation(payload: dict) -> EscalationAck:
    """
    Receive an HITL escalation payload from the InvestigatorAgent.

    Returns an acknowledgement with a unique ack_id.
    """
    ack_id = str(uuid.uuid4())
    _escalation_store.append({
        "ack_id": ack_id,
        "payload": payload,
    })
    return EscalationAck(ack_id=ack_id, status="received")


@router.get("/escalations")
async def list_escalations() -> list[dict]:
    """List all received HITL escalation payloads (for testing)."""
    return list(_escalation_store)


def clear_store() -> None:
    """Clear the in-memory escalation store (for test cleanup)."""
    _escalation_store.clear()

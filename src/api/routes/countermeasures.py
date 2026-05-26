"""Analyst-gated adaptive countermeasure API routes."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel, Field

from src.simulation import get_event_lab_service

router = APIRouter(prefix="/api/v1/countermeasures", tags=["countermeasures"])


class CountermeasureDecisionRequest(BaseModel):
    analyst: str = Field(default="union_bank_analyst", min_length=2, max_length=80)
    reason: str = Field(default="analyst_decision", min_length=2, max_length=240)


@router.get("/proposals")
async def list_countermeasure_proposals(
    run_id: str | None = Query(default=None),
    status: str | None = Query(default=None),
) -> dict:
    """List analyst-actionable adaptive countermeasure proposals."""
    return get_event_lab_service().list_proposals(run_id=run_id, status=status)


@router.post("/proposals/{proposal_id}/approve")
async def approve_countermeasure(
    proposal_id: str,
    request: Request,
    body: CountermeasureDecisionRequest | None = None,
) -> dict:
    """Approve and execute a proposed adaptive countermeasure."""
    decision = body or CountermeasureDecisionRequest()
    try:
        return await get_event_lab_service().approve_proposal(
            proposal_id=proposal_id,
            orchestrator=request.app.state.orchestrator,
            analyst=decision.analyst,
            reason=decision.reason,
        )
    except KeyError as exc:
        raise HTTPException(404, f"Unknown countermeasure proposal: {proposal_id}") from exc
    except PermissionError as exc:
        raise HTTPException(409, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(503, str(exc)) from exc


@router.post("/proposals/{proposal_id}/reject")
async def reject_countermeasure(
    proposal_id: str,
    body: CountermeasureDecisionRequest | None = None,
) -> dict:
    """Reject a proposed adaptive countermeasure and retain the audit trail."""
    decision = body or CountermeasureDecisionRequest(reason="analyst_rejected")
    try:
        return await get_event_lab_service().reject_proposal(
            proposal_id=proposal_id,
            analyst=decision.analyst,
            reason=decision.reason,
        )
    except KeyError as exc:
        raise HTTPException(404, f"Unknown countermeasure proposal: {proposal_id}") from exc
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc

"""Union Bank role and access policy routes."""

from __future__ import annotations

from fastapi import APIRouter, Request

from src.api.rbac import resolve_role, role_profile
from src.domain.union_bank import (
    OPERATING_UNITS,
    OPERATIONAL_WORKFLOWS,
    ORGANIZATIONAL_HIERARCHY,
    ROLE_POLICIES,
    UNION_BANK_DOMAIN_THRESHOLDS,
    operating_model_for_role,
    role_policy,
)

router = APIRouter(prefix="/api/v1/rbac", tags=["rbac"])


@router.get("/roles")
async def list_roles() -> dict:
    return {
        "roles": [role_profile(role) for role in ROLE_POLICIES],
        "thresholds": UNION_BANK_DOMAIN_THRESHOLDS,
        "operating_units": OPERATING_UNITS,
    }


@router.get("/profile")
async def current_profile(request: Request) -> dict:
    return role_profile(resolve_role(request))


@router.get("/workflows")
async def operational_workflows() -> dict:
    return {"workflows": OPERATIONAL_WORKFLOWS}


@router.get("/operating-model")
async def operating_model(request: Request) -> dict:
    """Return the selected role's practical bank operating model.

    The response is designed for the app shell, not just documentation: it
    tells the UI which real bank unit owns the role, which authority boundaries
    apply, and which regulatory obligations must remain visible.
    """

    return operating_model_for_role(resolve_role(request))


@router.get("/reality-check")
async def reality_check(request: Request) -> dict:
    """Return the selected role's real-world incident handoffs.

    This is intentionally derived from the same role/workflow policy that
    drives frontend gates so the UI cannot drift into a decorative RBAC banner.
    """
    role = resolve_role(request)
    scenario_handoffs = []
    for workflow in OPERATIONAL_WORKFLOWS:
        stages = list(workflow["stages"])
        for index, stage in enumerate(stages):
            if stage["owner"] != role:
                continue
            previous_stage = stages[index - 1] if index > 0 else None
            next_stage = stages[index + 1] if index < len(stages) - 1 else None
            scenario_handoffs.append({
                "workflow_id": workflow["id"],
                "workflow_title": workflow["title"],
                "trigger": workflow["trigger"],
                "step": index + 1,
                "total_steps": len(stages),
                "receives_from": role_policy(previous_stage["owner"]).label if previous_stage else "EFRMS / SIEM trigger",
                "action": stage["action"],
                "hands_to": role_policy(next_stage["owner"]).label if next_stage else "audit / closure",
            })

    return {
        "source": "Indian bank fraud, cyber, AML, FIU, branch and audit operating model",
        "profile": role_profile(role),
        "scenario_handoffs": scenario_handoffs,
        "operating_model": operating_model_for_role(role),
        "organizational_hierarchy": ORGANIZATIONAL_HIERARCHY,
        "floor_level_owner": bool(scenario_handoffs),
        "governance_note": (
            "AI triages risk; human authority remains segmented across SOC, transaction monitoring, AML, "
            "Principal Officer, compliance, investigation, branch operations, risk and audit."
        ),
    }

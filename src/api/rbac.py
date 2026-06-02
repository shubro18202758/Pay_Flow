"""Header-driven RBAC guardrails for the PayFlow prototype.

The frontend sends ``X-Payflow-Role`` from the selected Union Bank role. When
older tests or local tools omit the header, the backend uses the fraud committee
role for compatibility; explicit low-privilege headers are still enforced.
"""

from __future__ import annotations

from collections.abc import Iterable

from fastapi import HTTPException, Request

from src.domain.union_bank import ROLE_POLICIES, has_permission, role_policy

ROLE_HEADER = "x-payflow-role"
LEGACY_COMPAT_ROLE = "fraud_committee"


def resolve_role(request: Request) -> str:
    raw = request.headers.get(ROLE_HEADER) or request.query_params.get("role")
    if not raw:
        return LEGACY_COMPAT_ROLE
    role = raw.strip()
    if role not in ROLE_POLICIES:
        raise HTTPException(
            status_code=403,
            detail={
                "error": "unknown_payflow_role",
                "role": role,
                "allowed_roles": sorted(ROLE_POLICIES),
            },
        )
    return role


def require_permission(request: Request, permission: str) -> str:
    role = resolve_role(request)
    if not has_permission(role, permission):
        policy = role_policy(role)
        raise HTTPException(
            status_code=403,
            detail={
                "error": "payflow_role_forbidden",
                "role": role,
                "role_label": policy.label,
                "required_permission": permission,
                "domain": policy.domain,
            },
        )
    return role


def require_any_permission(request: Request, permissions: Iterable[str]) -> str:
    required = tuple(permissions)
    role = resolve_role(request)
    if any(has_permission(role, permission) for permission in required):
        return role

    policy = role_policy(role)
    raise HTTPException(
        status_code=403,
        detail={
            "error": "payflow_role_forbidden",
            "role": role,
            "role_label": policy.label,
            "required_any_permission": list(required),
            "domain": policy.domain,
        },
    )


def role_profile(role: str | None = None) -> dict:
    policy = role_policy(role)
    return {
        "role": policy.role,
        "label": policy.label,
        "domain": policy.domain,
        "summary": policy.summary,
        "tabs": list(policy.tabs),
        "permissions": sorted(policy.permissions),
        "feature_focus": list(policy.feature_focus),
        "escalation_scope": policy.escalation_scope,
        "shift": policy.shift,
        "reporting_line": policy.reporting_line,
        "decision_authority": policy.decision_authority,
        "tool_stack": list(policy.tool_stack),
        "workflow_steps": list(policy.workflow_steps),
    }

"""Adaptive Event Lab and analyst-gated countermeasure tests."""

from __future__ import annotations

import os
import time
from types import SimpleNamespace

import pytest

os.environ.setdefault("PAYFLOW_CPU_ONLY", "1")


class FakePipeline:
    def __init__(self) -> None:
        self._running = False
        self.events = []

    async def start(self) -> None:
        self._running = True

    async def ingest(self, event) -> None:
        self.events.append(event)


class FakeBreaker:
    def __init__(self) -> None:
        self._cfg = SimpleNamespace(freeze_ttl_seconds=600)
        self._graph = None
        self._agent_listener = None
        self.orders = []

    async def freeze_node(self, order) -> None:
        self.orders.append(order)


class FakeLedger:
    def __init__(self) -> None:
        self.anchors = []

    async def anchor_circuit_breaker(self, action: str, details: dict) -> None:
        self.anchors.append({"action": action, "details": details})


class FakeOrchestrator:
    def __init__(self) -> None:
        self._pipeline = FakePipeline()
        self._breaker = FakeBreaker()
        self._ledger = FakeLedger()


def _seed_high_trust_intel():
    from src.intel import reset_pre_fraud_intel_service
    from src.simulation import reset_event_lab_service

    intel = reset_pre_fraud_intel_service()
    intel.simulate_signal("digital_arrest_mule")
    return reset_event_lab_service()


def test_templates_link_to_active_pre_fraud_playbooks():
    service = _seed_high_trust_intel()

    body = service.templates()

    assert body["templates"]
    assert any(t["linked_playbooks"] for t in body["templates"])
    assert any(t["execution_allowed"] for t in body["templates"])


def test_event_lab_api_routes_cover_preview_launch_and_approval():
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from src.api.routes.countermeasures import router as countermeasure_router
    from src.api.routes.simulation import router as simulation_router

    _seed_high_trust_intel()
    app = FastAPI()
    app.state.orchestrator = FakeOrchestrator()
    app.include_router(simulation_router)
    app.include_router(countermeasure_router)
    client = TestClient(app)

    templates = client.get("/api/v1/simulation/event-lab/templates")
    assert templates.status_code == 200
    template_id = "digital_arrest_chain"

    preview = client.post(
        "/api/v1/simulation/event-lab/preview",
        json={"template_id": template_id, "mode": "chain", "seed": 2026},
    )
    assert preview.status_code == 200
    assert preview.json()["run_preview"]["events"]

    launched = client.post(
        "/api/v1/simulation/event-lab/runs",
        json={"template_id": template_id, "mode": "chain", "seed": 2026, "analyst_required": True},
    )
    assert launched.status_code == 200
    run_id = launched.json()["run_id"]

    explainability = client.get(f"/api/v1/simulation/event-lab/runs/{run_id}/explainability")
    assert explainability.status_code == 200
    explainability_body = explainability.json()
    assert explainability_body["stage_groups"]
    assert explainability_body["evidence_panels"]
    assert explainability_body["authority_matrix"]
    assert explainability_body["runtime"]["proposal_count"] >= 1

    proposals = client.get(f"/api/v1/countermeasures/proposals?run_id={run_id}")
    assert proposals.status_code == 200
    freeze = next(p for p in proposals.json()["proposals"] if p["action"] in {"FREEZE_NODE", "FREEZE_1HOP"})

    approved = client.post(
        f"/api/v1/countermeasures/proposals/{freeze['proposal_id']}/approve",
        json={"analyst": "tester", "reason": "route_verified"},
    )
    assert approved.status_code == 200
    assert approved.json()["status"] == "executed"


def test_preview_generation_is_deterministic():
    service = _seed_high_trust_intel()

    one = service.preview("digital_arrest_chain", mode="chain", seed=2026)
    two = service.preview("digital_arrest_chain", mode="chain", seed=2026)

    assert one["run_preview"]["event_ids"] == two["run_preview"]["event_ids"]
    assert one["run_preview"]["events"]
    assert one["run_preview"]["countermeasure_policy"]["execution_allowed"] is True


@pytest.mark.asyncio
async def test_launch_run_injects_events_and_creates_proposals():
    service = _seed_high_trust_intel()
    orch = FakeOrchestrator()

    run = await service.launch_run(
        orchestrator=orch,
        template_id="digital_arrest_chain",
        mode="chain",
        intensity="demo",
        seed=2026,
    )

    assert run["event_ids"]
    assert len(orch._pipeline.events) == len(run["event_ids"])
    assert run["countermeasure_proposals"]
    assert run["countermeasure_policy"]["authority"] == "analyst_approval_required"
    explainability = service.explainability_response(run["run_id"])
    assert explainability["stage_groups"][0]["group"] == "pre_fraud_intel"
    assert any(panel["key"] == "qwen" for panel in explainability["evidence_panels"])


@pytest.mark.asyncio
async def test_analyst_approval_executes_freeze_and_anchors():
    service = _seed_high_trust_intel()
    orch = FakeOrchestrator()
    run = await service.launch_run(orch, "digital_arrest_chain", mode="chain", seed=2026)
    freeze = next(p for p in run["countermeasure_proposals"] if p["action"] in {"FREEZE_NODE", "FREEZE_1HOP"})

    result = await service.approve_proposal(freeze["proposal_id"], orch, analyst="tester", reason="verified")

    assert result["status"] == "executed"
    assert orch._breaker.orders
    assert orch._ledger.anchors
    assert result["audit_hash"]
    explainability = service.explainability_response(run["run_id"])
    assert explainability["runtime"]["executed_count"] == 1
    assert any(item["status"] == "executed" for item in explainability["proposal_lifecycle"])
    assert any(group["group"] == "execution_audit" and group["completed"] for group in explainability["stage_groups"])


@pytest.mark.asyncio
async def test_evidence_package_includes_event_lab_countermeasure_context():
    from src.api.ps3_case import build_evidence_package

    service = _seed_high_trust_intel()
    orch = FakeOrchestrator()
    run = await service.launch_run(orch, "digital_arrest_chain", mode="chain", seed=2026)
    freeze = next(p for p in run["countermeasure_proposals"] if p["action"] in {"FREEZE_NODE", "FREEZE_1HOP"})
    await service.approve_proposal(freeze["proposal_id"], orch, analyst="tester", reason="verified")

    meta = {
        "primary_case_id": "PS3-EVENT-LAB",
        "scenario_id": "event-lab-demo",
        "scenario": "rapid_layering",
        "scenario_label": "Event Lab Rapid Layering",
        "focus_account_id": "ACC001",
        "focus_txn_id": "TXN001",
        "expected_indicators": ["Adaptive event lab correlation"],
        "typologies": ["LAYERING"],
        "recommended_actions": ["Attach countermeasure context"],
        "transaction_chain": [
            {
                "type": "transaction",
                "txn_id": "TXN001",
                "sender": "ACC001",
                "receiver": "ACC002",
                "amount_paisa": 12500000,
                "channel": "UPI",
                "fraud_label": "LAYERING",
            }
        ],
        "generated_at": time.time(),
    }
    orch._threat_engine = SimpleNamespace(get_ps3_case=lambda case_id: meta if case_id == "PS3-EVENT-LAB" else None)

    package = await build_evidence_package(orch, "PS3-EVENT-LAB")

    assert package["event_lab_run_id"] == run["run_id"]
    assert package["countermeasure_proposals"]
    assert package["executed_actions"]
    assert package["qwen_explanation"]
    assert package["json_payload"]["countermeasure_proposals"]
    assert "Adaptive Event Lab Countermeasures" in package["printable_html"]


@pytest.mark.asyncio
async def test_low_trust_or_missing_intel_cannot_execute_countermeasure():
    from src.intel import reset_pre_fraud_intel_service
    from src.simulation import reset_event_lab_service

    reset_pre_fraud_intel_service()
    service = reset_event_lab_service()
    orch = FakeOrchestrator()
    run = await service.launch_run(orch, "upi_mule_cashout", mode="single", seed=11)
    proposal = run["countermeasure_proposals"][0]

    assert proposal["execution_allowed"] is False
    with pytest.raises(PermissionError):
        await service.approve_proposal(proposal["proposal_id"], orch)

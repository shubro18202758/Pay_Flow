"""PS3 Round 2 workflow tests."""

from __future__ import annotations

import asyncio
import os
import random
from types import SimpleNamespace

os.environ.setdefault("PAYFLOW_CPU_ONLY", "1")


def _make_world():
    from src.ingestion.generators.synthetic_transactions import build_world
    return build_world(num_accounts=120)


def test_ps3_generators_cover_all_required_typologies():
    from src.ingestion.schemas import FraudPattern, Transaction
    from src.ingestion.validators import validate_event
    from src.simulation.attack_generators import PS3_SCENARIO_DETAILS, generate_ps3_scenario

    expected = {
        "rapid_layering": FraudPattern.LAYERING,
        "round_tripping": FraudPattern.ROUND_TRIPPING,
        "structuring": FraudPattern.STRUCTURING,
        "dormant_activation": FraudPattern.DORMANT_ACTIVATION,
        "profile_mismatch": FraudPattern.PROFILE_MISMATCH,
    }
    world = _make_world()

    for idx, (scenario, pattern) in enumerate(expected.items()):
        events = generate_ps3_scenario(
            scenario=scenario,
            world=world,
            rng=random.Random(2026 + idx),
            intensity="demo",
        )
        assert scenario in PS3_SCENARIO_DETAILS
        assert events
        for event in events:
            result = validate_event(event)
            assert result.valid, f"{scenario} event failed validation: {result.errors}"
        txns = [event for event in events if isinstance(event, Transaction)]
        assert txns, f"{scenario} should include transaction events"
        assert {txn.fraud_label for txn in txns} == {pattern}


def test_engine_launches_ps3_case_with_focus_metadata():
    async def _run():
        from config.settings import SimulationConfig
        from src.ingestion.stream_processor import IngestionPipeline
        from src.simulation.threat_engine import ThreatSimulationEngine

        pipeline = IngestionPipeline(batch_size=64, batch_timeout_sec=0.1)
        engine = ThreatSimulationEngine(
            pipeline,
            _make_world(),
            SimulationConfig(default_event_interval_sec=0.01, max_concurrent_attacks=2),
        )
        metadata = await engine.launch_ps3_scenario(
            "rapid_layering",
            intensity="demo",
            seed=2026,
        )
        assert metadata["primary_case_id"].startswith("PS3-")
        assert metadata["focus_account_id"]
        assert metadata["focus_txn_id"]
        assert metadata["expected_indicators"]
        assert engine.get_ps3_case(metadata["primary_case_id"]) is not None
        await engine.stop_all()
        if pipeline._running:
            await pipeline.stop()

    asyncio.run(_run())


def test_ps3_routes_trace_and_evidence_package():
    from config.settings import SimulationConfig
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from src.api.routes.dashboard import router as dashboard_router
    from src.api.routes.fraud import router as fraud_router
    from src.api.routes.simulation import router as simulation_router
    from src.ingestion.stream_processor import IngestionPipeline
    from src.simulation.threat_engine import ThreatSimulationEngine

    pipeline = IngestionPipeline(batch_size=64, batch_timeout_sec=0.1)
    engine = ThreatSimulationEngine(
        pipeline,
        _make_world(),
        SimulationConfig(default_event_interval_sec=0.01, max_concurrent_attacks=2),
    )
    app = FastAPI()
    app.include_router(simulation_router)
    app.include_router(fraud_router)
    app.include_router(dashboard_router)
    app.state.orchestrator = SimpleNamespace(_threat_engine=engine)
    client = TestClient(app)

    scenarios = client.get("/api/v1/simulation/ps3/scenarios")
    assert scenarios.status_code == 200
    assert len(scenarios.json()["scenarios"]) == 5

    launched = client.post(
        "/api/v1/simulation/ps3/launch",
        json={"scenario": "structuring", "intensity": "demo", "seed": 2026},
    )
    assert launched.status_code == 200, launched.text
    launch_data = launched.json()
    case_id = launch_data["primary_case_id"]
    assert launch_data["focus_txn_id"]

    trace = client.get(f"/api/v1/fraud/investigation/case/{case_id}/trace")
    assert trace.status_code == 200
    trace_data = trace.json()
    assert trace_data["case_id"] == case_id
    assert trace_data["timeline"]
    assert "STRUCTURING" in trace_data["ps3_typologies"]

    package = client.post(f"/api/v1/fraud/investigation/case/{case_id}/evidence-package")
    assert package.status_code == 200
    package_data = package.json()
    assert package_data["package_id"].startswith("FIU-")
    assert package_data["json_payload"]["case_id"] == case_id
    assert "PayFlow FIU Evidence Package" in package_data["printable_html"]

    readiness = client.get("/api/v1/readiness/ps3")
    assert readiness.status_code == 200
    readiness_data = readiness.json()
    assert len(readiness_data["requirements"]) == 5
    assert readiness_data["runtime_health"]["qwen_model"]

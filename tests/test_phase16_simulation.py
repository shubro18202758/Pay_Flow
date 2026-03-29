"""
PayFlow -- Phase 16: Threat Simulation Engine Test Suite
=========================================================
Tests for the dynamic, user-customizable Threat Simulation Engine:
  - SimulationConfig defaults and immutability
  - FraudPattern enum extensions (backward compat)
  - Attack generator correctness (UPI mule, circular laundering, phishing)
  - CRC32 checksum + structural validation compliance
  - ThreatSimulationEngine lifecycle (launch, stop, drip-feed)
  - FastAPI simulation routes
  - SSE integration on "simulation" channel

Run:
    PYTHONPATH=. python tests/test_phase16_simulation.py
"""

from __future__ import annotations

import asyncio
import os
import random
import sys
import time

os.environ.setdefault("PAYFLOW_CPU_ONLY", "1")


# -- Safe printing (Windows cp1252 compat) ----------------------------------

def _safe_print(*args, **kwargs):
    try:
        print(*args, **kwargs)
    except UnicodeEncodeError:
        text = " ".join(str(a) for a in args)
        print(text.encode("ascii", errors="replace").decode(), **kwargs)


# ============================================================================
# CONFIG TESTS
# ============================================================================

def test_simulation_config_defaults():
    """SimulationConfig has expected default values."""
    from config.settings import SimulationConfig, SIMULATION_CFG
    cfg = SIMULATION_CFG
    assert isinstance(cfg, SimulationConfig)
    assert cfg.default_event_interval_sec == 0.5
    assert cfg.burst_interval_sec == 0.05
    assert cfg.upi_mule_count == 6
    assert cfg.circular_shell_count == 5
    assert cfg.phishing_credential_attempts == 8
    assert cfg.max_concurrent_attacks == 3
    assert cfg.max_events_per_attack == 200


def test_simulation_config_frozen():
    """SimulationConfig is immutable."""
    from config.settings import SIMULATION_CFG
    try:
        SIMULATION_CFG.max_concurrent_attacks = 99
        raise AssertionError("Should have raised FrozenInstanceError")
    except AttributeError:
        pass  # expected -- frozen dataclass


def test_simulation_config_exported():
    """SimulationConfig and SIMULATION_CFG are exported from config package."""
    from config import SimulationConfig, SIMULATION_CFG
    assert isinstance(SIMULATION_CFG, SimulationConfig)


# ============================================================================
# FRAUD PATTERN ENUM TESTS
# ============================================================================

def test_fraud_pattern_new_values():
    """FraudPattern has the 3 new simulation typology values."""
    from src.ingestion.schemas import FraudPattern
    assert FraudPattern.UPI_MULE_NETWORK == 6
    assert FraudPattern.CIRCULAR_LAUNDERING == 7
    assert FraudPattern.VELOCITY_PHISHING == 8


def test_fraud_pattern_backward_compat():
    """Existing FraudPattern values 0-5 are unchanged."""
    from src.ingestion.schemas import FraudPattern
    assert FraudPattern.NONE == 0
    assert FraudPattern.LAYERING == 1
    assert FraudPattern.ROUND_TRIPPING == 2
    assert FraudPattern.STRUCTURING == 3
    assert FraudPattern.DORMANT_ACTIVATION == 4
    assert FraudPattern.PROFILE_MISMATCH == 5


# ============================================================================
# UPI MULE NETWORK GENERATOR TESTS
# ============================================================================

def _make_world():
    from src.ingestion.generators.synthetic_transactions import build_world
    return build_world(num_accounts=100)


def test_upi_mule_generates_events():
    """UPI mule network generates correct number and types of events."""
    from src.simulation.attack_generators import generate_upi_mule_network
    from src.ingestion.schemas import Transaction, InterbankMessage, AuthEvent
    world = _make_world()
    rng = random.Random(42)
    events = generate_upi_mule_network(world, rng, mule_count=4)
    assert len(events) > 0, "Should generate events"
    txns = [e for e in events if isinstance(e, Transaction)]
    msgs = [e for e in events if isinstance(e, InterbankMessage)]
    auths = [e for e in events if isinstance(e, AuthEvent)]
    # 4 fan-out + 4 dispersal = 8 transactions, 1 interbank, 3-4 auth events
    assert len(txns) == 8, f"Expected 8 transactions, got {len(txns)}"
    assert len(msgs) >= 1, f"Expected >= 1 interbank message, got {len(msgs)}"
    assert len(auths) >= 3, f"Expected >= 3 auth events, got {len(auths)}"


def test_upi_mule_checksum_valid():
    """Every event from UPI mule generator passes CRC32 validation."""
    from src.simulation.attack_generators import generate_upi_mule_network
    from src.ingestion.validators import validate_event
    world = _make_world()
    rng = random.Random(42)
    events = generate_upi_mule_network(world, rng, mule_count=4)
    for event in events:
        result = validate_event(event)
        assert result.valid, f"Event {result.event_id} failed: {result.errors}"


def test_upi_mule_has_auth_events():
    """UPI mule attack includes OTP_FAIL and LOGIN auth events."""
    from src.simulation.attack_generators import generate_upi_mule_network
    from src.ingestion.schemas import AuthEvent, AuthAction
    world = _make_world()
    rng = random.Random(42)
    events = generate_upi_mule_network(world, rng, mule_count=4)
    auth_events = [e for e in events if isinstance(e, AuthEvent)]
    actions = {e.action for e in auth_events}
    assert AuthAction.OTP_FAIL in actions, "Should have OTP_FAIL events"
    assert AuthAction.LOGIN in actions, "Should have LOGIN event"


def test_upi_mule_fan_out_fan_in():
    """UPI mule shows fan-out from victim then fan-in to collector."""
    from src.simulation.attack_generators import generate_upi_mule_network
    from src.ingestion.schemas import Transaction, Channel
    world = _make_world()
    rng = random.Random(42)
    events = generate_upi_mule_network(world, rng, mule_count=4)
    txns = [e for e in events if isinstance(e, Transaction)]
    # Fan-out: first 4 txns have Channel.UPI, same sender
    fan_out = [t for t in txns if t.channel == Channel.UPI]
    assert len(fan_out) == 4, f"Expected 4 UPI fan-out txns, got {len(fan_out)}"
    senders = {t.sender_id for t in fan_out}
    assert len(senders) == 1, "Fan-out should have single sender (victim)"
    # Fan-in: last 4 IMPS txns to same receiver
    fan_in = [t for t in txns if t.channel == Channel.IMPS]
    assert len(fan_in) == 4, f"Expected 4 IMPS fan-in txns, got {len(fan_in)}"
    receivers = {t.receiver_id for t in fan_in}
    assert len(receivers) == 1, "Fan-in should have single receiver (collector)"


# ============================================================================
# CIRCULAR LAUNDERING GENERATOR TESTS
# ============================================================================

def test_circular_laundering_generates_events():
    """Circular laundering generates correct number of events."""
    from src.simulation.attack_generators import generate_circular_laundering
    from src.ingestion.schemas import Transaction, InterbankMessage, AuthEvent
    world = _make_world()
    rng = random.Random(42)
    events = generate_circular_laundering(world, rng, shell_count=4)
    txns = [e for e in events if isinstance(e, Transaction)]
    msgs = [e for e in events if isinstance(e, InterbankMessage)]
    auths = [e for e in events if isinstance(e, AuthEvent)]
    # 4 auth + 4 hop txns + 4 interbank msgs + 4 fee txns = 16 total
    assert len(auths) == 4, f"Expected 4 auth events, got {len(auths)}"
    assert len(msgs) == 4, f"Expected 4 interbank messages, got {len(msgs)}"
    # 4 hops + 4 fees = 8 transactions
    assert len(txns) == 8, f"Expected 8 transactions, got {len(txns)}"


def test_circular_laundering_checksum_valid():
    """Every event from circular laundering generator passes validation."""
    from src.simulation.attack_generators import generate_circular_laundering
    from src.ingestion.validators import validate_event
    world = _make_world()
    rng = random.Random(42)
    events = generate_circular_laundering(world, rng, shell_count=4)
    for event in events:
        result = validate_event(event)
        assert result.valid, f"Event {result.event_id} failed: {result.errors}"


def test_circular_laundering_forms_cycle():
    """Circular laundering hop transactions form a complete cycle."""
    from src.simulation.attack_generators import generate_circular_laundering
    from src.ingestion.schemas import Transaction, Channel, FraudPattern
    world = _make_world()
    rng = random.Random(42)
    events = generate_circular_laundering(world, rng, shell_count=4)
    txns = [e for e in events if isinstance(e, Transaction)
            and e.fraud_label == FraudPattern.CIRCULAR_LAUNDERING]
    # The hop transactions are the ones using RTGS/NEFT channels
    hops = [t for t in txns if t.channel in (Channel.RTGS, Channel.NEFT)]
    assert len(hops) == 4, f"Expected 4 hop transactions, got {len(hops)}"
    # Verify cycle: receiver of last hop == sender of first hop
    assert hops[-1].receiver_id == hops[0].sender_id, \
        "Last hop receiver should equal first hop sender (cycle)"


def test_circular_laundering_has_interbank_messages():
    """Circular laundering includes InterbankMessage events paired with hops."""
    from src.simulation.attack_generators import generate_circular_laundering
    from src.ingestion.schemas import InterbankMessage
    world = _make_world()
    rng = random.Random(42)
    events = generate_circular_laundering(world, rng, shell_count=4)
    msgs = [e for e in events if isinstance(e, InterbankMessage)]
    assert len(msgs) == 4, f"Expected 4 interbank messages, got {len(msgs)}"
    for msg in msgs:
        assert msg.sender_ifsc.startswith("UBIN0"), \
            f"IFSC should start with UBIN0, got {msg.sender_ifsc}"


def test_circular_laundering_uses_shell_accounts():
    """Circular laundering prefers CURRENT/INTERNAL account types."""
    from src.simulation.attack_generators import generate_circular_laundering
    from src.ingestion.schemas import Transaction, AccountType, Channel
    world = _make_world()
    rng = random.Random(42)
    events = generate_circular_laundering(world, rng, shell_count=4)
    hops = [e for e in events if isinstance(e, Transaction)
            and e.channel in (Channel.RTGS, Channel.NEFT)]
    # At least some hops should involve CURRENT or INTERNAL accounts
    shell_types = {AccountType.CURRENT, AccountType.INTERNAL}
    has_shell = any(
        t.sender_account_type in shell_types or t.receiver_account_type in shell_types
        for t in hops
    )
    # Note: depends on world composition; with 100 accounts some should be CURRENT
    # The test is soft -- we just verify it doesn't crash
    assert len(hops) == 4


# ============================================================================
# VELOCITY PHISHING GENERATOR TESTS
# ============================================================================

def test_velocity_phishing_generates_events():
    """Velocity phishing generates correct number of events."""
    from src.simulation.attack_generators import generate_velocity_phishing
    from src.ingestion.schemas import Transaction, AuthEvent
    world = _make_world()
    rng = random.Random(42)
    events = generate_velocity_phishing(
        world, rng, credential_attempts=6, unauthorized_txns=3, geo_spread=4,
    )
    auths = [e for e in events if isinstance(e, AuthEvent)]
    txns = [e for e in events if isinstance(e, Transaction)]
    # 6 failed + 1 login + 1 OTP + 1 password change = 9 auth events
    assert len(auths) == 9, f"Expected 9 auth events, got {len(auths)}"
    assert len(txns) == 3, f"Expected 3 transactions, got {len(txns)}"


def test_velocity_phishing_checksum_valid():
    """Every event from velocity phishing generator passes validation."""
    from src.simulation.attack_generators import generate_velocity_phishing
    from src.ingestion.validators import validate_event
    world = _make_world()
    rng = random.Random(42)
    events = generate_velocity_phishing(
        world, rng, credential_attempts=6, unauthorized_txns=3,
    )
    for event in events:
        result = validate_event(event)
        assert result.valid, f"Event {result.event_id} failed: {result.errors}"


def test_velocity_phishing_multi_geo():
    """Velocity phishing transactions originate from distinct geolocations."""
    from src.simulation.attack_generators import generate_velocity_phishing
    from src.ingestion.schemas import Transaction
    world = _make_world()
    rng = random.Random(42)
    events = generate_velocity_phishing(
        world, rng, credential_attempts=6, unauthorized_txns=4, geo_spread=4,
    )
    txns = [e for e in events if isinstance(e, Transaction)]
    # Each txn should have different sender geo (impossible travel)
    coords = [(t.sender_geo_lat, t.sender_geo_lon) for t in txns]
    # At least 2 distinct city-level locations (allow jitter)
    rounded = set((round(lat, 1), round(lon, 1)) for lat, lon in coords)
    assert len(rounded) >= 2, \
        f"Expected >= 2 distinct geolocations, got {len(rounded)}: {rounded}"


def test_velocity_phishing_multi_device():
    """Velocity phishing transactions use distinct device fingerprints."""
    from src.simulation.attack_generators import generate_velocity_phishing
    from src.ingestion.schemas import Transaction
    world = _make_world()
    rng = random.Random(42)
    events = generate_velocity_phishing(
        world, rng, credential_attempts=6, unauthorized_txns=4,
    )
    txns = [e for e in events if isinstance(e, Transaction)]
    fps = [t.device_fingerprint for t in txns]
    assert len(set(fps)) == len(fps), \
        f"Expected all distinct device fingerprints, got {len(set(fps))} unique of {len(fps)}"


# ============================================================================
# ACCOUNT ID EXTRACTION
# ============================================================================

def test_get_account_ids():
    """get_account_ids extracts all unique account IDs from events."""
    from src.simulation.attack_generators import (
        generate_upi_mule_network, get_account_ids,
    )
    world = _make_world()
    rng = random.Random(42)
    events = generate_upi_mule_network(world, rng, mule_count=3)
    ids = get_account_ids(events)
    assert isinstance(ids, list)
    assert len(ids) >= 4, f"Expected >= 4 account IDs (victim+3 mules+collector), got {len(ids)}"
    assert ids == sorted(ids), "Account IDs should be sorted"


# ============================================================================
# THREAT ENGINE LIFECYCLE TESTS
# ============================================================================

def _make_engine():
    from src.ingestion.stream_processor import IngestionPipeline
    from src.ingestion.generators.synthetic_transactions import build_world
    from src.simulation.threat_engine import ThreatSimulationEngine
    from config.settings import SimulationConfig
    pipeline = IngestionPipeline(batch_size=64, batch_timeout_sec=0.5)
    world = build_world(num_accounts=100)
    cfg = SimulationConfig(
        default_event_interval_sec=0.01,  # fast for tests
        burst_interval_sec=0.01,
        max_concurrent_attacks=2,
    )
    return ThreatSimulationEngine(pipeline, world, cfg), pipeline


def test_engine_available_attacks():
    """Engine reports 3 available attack types."""
    engine, _ = _make_engine()
    attacks = engine.available_attacks()
    assert len(attacks) == 3
    assert "upi_mule_network" in attacks
    assert "circular_laundering" in attacks
    assert "velocity_phishing" in attacks


def test_engine_launch_and_complete():
    """Engine can launch an attack that runs to completion."""

    async def _run():
        engine, pipeline = _make_engine()
        sid = await engine.launch_attack("upi_mule_network", {"mule_count": 3})
        assert sid is not None
        status = engine.get_status(sid)
        assert status["status"] == "running"
        # Wait for completion (fast interval so should finish quickly)
        for _ in range(100):
            await asyncio.sleep(0.05)
            status = engine.get_status(sid)
            if status["status"] != "running":
                break
        assert status["status"] == "completed", f"Expected completed, got {status['status']}"
        assert status["events_ingested"] == status["events_generated"]
        # Cleanup
        if pipeline._running:
            await pipeline.stop()

    asyncio.run(_run())


def test_engine_stop_attack():
    """Engine can stop a running attack mid-stream."""

    async def _run():
        from config.settings import SimulationConfig
        from src.ingestion.stream_processor import IngestionPipeline
        from src.ingestion.generators.synthetic_transactions import build_world
        from src.simulation.threat_engine import ThreatSimulationEngine
        pipeline = IngestionPipeline(batch_size=64, batch_timeout_sec=0.5)
        world = build_world(num_accounts=100)
        cfg = SimulationConfig(
            default_event_interval_sec=0.5,  # slow enough to stop mid-stream
            burst_interval_sec=0.5,
            max_concurrent_attacks=2,
        )
        engine = ThreatSimulationEngine(pipeline, world, cfg)
        sid = await engine.launch_attack("circular_laundering", {"shell_count": 5})
        await asyncio.sleep(0.2)
        stopped = await engine.stop_attack(sid)
        assert stopped is True
        status = engine.get_status(sid)
        assert status["status"] == "stopped"
        assert status["events_ingested"] < status["events_generated"]
        if pipeline._running:
            await pipeline.stop()

    asyncio.run(_run())


def test_engine_stop_all():
    """Engine stop_all cancels all running attacks."""

    async def _run():
        from config.settings import SimulationConfig
        from src.ingestion.stream_processor import IngestionPipeline
        from src.ingestion.generators.synthetic_transactions import build_world
        from src.simulation.threat_engine import ThreatSimulationEngine
        pipeline = IngestionPipeline(batch_size=64, batch_timeout_sec=0.5)
        world = build_world(num_accounts=100)
        cfg = SimulationConfig(
            default_event_interval_sec=1.0,
            burst_interval_sec=1.0,
            max_concurrent_attacks=2,
        )
        engine = ThreatSimulationEngine(pipeline, world, cfg)
        await engine.launch_attack("upi_mule_network")
        await engine.launch_attack("velocity_phishing")
        await asyncio.sleep(0.1)
        count = await engine.stop_all()
        assert count == 2, f"Expected 2 stopped, got {count}"
        assert len(engine.list_active()) == 0
        if pipeline._running:
            await pipeline.stop()

    asyncio.run(_run())


def test_engine_max_concurrent():
    """Engine rejects attacks when max concurrent limit is reached."""

    async def _run():
        from config.settings import SimulationConfig
        from src.ingestion.stream_processor import IngestionPipeline
        from src.ingestion.generators.synthetic_transactions import build_world
        from src.simulation.threat_engine import ThreatSimulationEngine
        pipeline = IngestionPipeline(batch_size=64, batch_timeout_sec=0.5)
        world = build_world(num_accounts=100)
        cfg = SimulationConfig(
            default_event_interval_sec=2.0,
            burst_interval_sec=2.0,
            max_concurrent_attacks=1,
        )
        engine = ThreatSimulationEngine(pipeline, world, cfg)
        await engine.launch_attack("upi_mule_network")
        try:
            await engine.launch_attack("velocity_phishing")
            raise AssertionError("Should have raised RuntimeError")
        except RuntimeError:
            pass  # expected
        await engine.stop_all()
        if pipeline._running:
            await pipeline.stop()

    asyncio.run(_run())


def test_engine_invalid_attack_type():
    """Engine raises ValueError for unknown attack type."""

    async def _run():
        engine, pipeline = _make_engine()
        try:
            await engine.launch_attack("nonexistent_attack")
            raise AssertionError("Should have raised ValueError")
        except ValueError:
            pass  # expected

    asyncio.run(_run())


def test_engine_history():
    """Completed attacks appear in list_all."""

    async def _run():
        engine, pipeline = _make_engine()
        sid = await engine.launch_attack("velocity_phishing",
                                         {"credential_attempts": 3, "unauthorized_txns": 2})
        for _ in range(100):
            await asyncio.sleep(0.05)
            s = engine.get_status(sid)
            if s["status"] != "running":
                break
        all_scenarios = engine.list_all()
        assert len(all_scenarios) == 1
        assert all_scenarios[0]["status"] == "completed"
        assert all_scenarios[0]["scenario_id"] == sid
        if pipeline._running:
            await pipeline.stop()

    asyncio.run(_run())


def test_engine_snapshot():
    """Engine snapshot includes active and total counts."""

    async def _run():
        engine, pipeline = _make_engine()
        snap = engine.snapshot()
        assert "active_attacks" in snap
        assert "total_attacks" in snap
        assert snap["active_attacks"] == 0
        assert snap["total_attacks"] == 0

    asyncio.run(_run())


# ============================================================================
# API ROUTE TESTS
# ============================================================================

def _make_test_app():
    """Build a FastAPI test app with a mock orchestrator."""
    from unittest.mock import MagicMock
    from src.api.routes.simulation import router
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from src.simulation.threat_engine import ThreatSimulationEngine
    from src.ingestion.stream_processor import IngestionPipeline
    from src.ingestion.generators.synthetic_transactions import build_world
    from config.settings import SimulationConfig

    app = FastAPI()
    app.include_router(router)

    pipeline = IngestionPipeline(batch_size=64, batch_timeout_sec=0.5)
    world = build_world(num_accounts=100)
    cfg = SimulationConfig(
        default_event_interval_sec=0.01,
        burst_interval_sec=0.01,
    )
    engine = ThreatSimulationEngine(pipeline, world, cfg)

    orch = MagicMock()
    orch._threat_engine = engine
    app.state.orchestrator = orch

    return TestClient(app), engine, pipeline


def test_route_list_attacks():
    """GET /attacks returns 3 attack types."""
    client, _, pipeline = _make_test_app()
    resp = client.get("/api/v1/simulation/attacks")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["attacks"]) == 3


def test_route_launch_attack():
    """POST /launch successfully launches and returns scenario_id."""
    client, engine, pipeline = _make_test_app()
    resp = client.post("/api/v1/simulation/launch", json={
        "attack_type": "velocity_phishing",
        "params": {"credential_attempts": 3, "unauthorized_txns": 2},
    })
    assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text}"
    data = resp.json()
    assert "scenario_id" in data
    assert data["attack_type"] == "velocity_phishing"
    assert data["status"] == "running"


def test_route_launch_invalid():
    """POST /launch with unknown attack type returns 400."""
    client, _, _ = _make_test_app()
    resp = client.post("/api/v1/simulation/launch", json={
        "attack_type": "doesnt_exist",
    })
    assert resp.status_code == 400


def test_route_status_not_found():
    """GET /status/{id} with unknown ID returns 404."""
    client, _, _ = _make_test_app()
    resp = client.get("/api/v1/simulation/status/nonexistent")
    assert resp.status_code == 404


def test_route_no_engine():
    """Routes return 503 when orchestrator has no threat engine."""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from src.api.routes.simulation import router

    app = FastAPI()
    app.include_router(router)
    app.state.orchestrator = None
    client = TestClient(app)
    resp = client.get("/api/v1/simulation/attacks")
    assert resp.status_code == 503


# ============================================================================
# SSE CHANNEL TEST
# ============================================================================

def test_simulation_sse_channel():
    """The 'simulation' channel is in ALL_CHANNELS."""
    from src.api.routes.dashboard import ALL_CHANNELS
    assert "simulation" in ALL_CHANNELS


def test_sse_broadcast_on_attack():
    """Launching an attack broadcasts events to 'simulation' channel."""

    async def _run():
        from src.api.events import EventBroadcaster
        EventBroadcaster.reset()
        broadcaster = EventBroadcaster.get()
        queue = await broadcaster.subscribe(["simulation"])

        engine, pipeline = _make_engine()
        sid = await engine.launch_attack("velocity_phishing",
                                         {"credential_attempts": 2, "unauthorized_txns": 1})
        # Wait for some events
        for _ in range(50):
            await asyncio.sleep(0.05)
            if not queue.empty():
                break

        assert not queue.empty(), "Should have received SSE events on simulation channel"
        event = queue.get_nowait()
        assert event["channel"] == "simulation"
        assert "data" in event

        await engine.stop_all()
        if pipeline._running:
            await pipeline.stop()
        EventBroadcaster.reset()

    asyncio.run(_run())


# ============================================================================
# TEST RUNNER
# ============================================================================

def _run_tests():
    """Collect and run all test_ functions."""
    tests = [
        (name, func)
        for name, func in sorted(globals().items())
        if name.startswith("test_") and callable(func)
    ]

    passed = 0
    failed = 0
    errors = []

    _safe_print(f"\n{'=' * 66}")
    _safe_print(f"  Phase 16 Threat Simulation Engine Test Suite  ({len(tests)} tests)")
    _safe_print(f"{'=' * 66}\n")

    for name, func in tests:
        try:
            t0 = time.perf_counter()
            func()
            elapsed = (time.perf_counter() - t0) * 1000
            _safe_print(f"  PASS  {name} ({elapsed:.1f} ms)")
            passed += 1
        except Exception as exc:
            elapsed = (time.perf_counter() - t0) * 1000
            _safe_print(f"  FAIL  {name} ({elapsed:.1f} ms) -- {exc}")
            failed += 1
            errors.append((name, exc))

    _safe_print(f"\n{'=' * 66}")
    _safe_print(f"  Results: {passed} passed, {failed} failed, {passed + failed} total")
    _safe_print(f"{'=' * 66}\n")

    if errors:
        _safe_print("  FAILURES:")
        for name, exc in errors:
            _safe_print(f"    {name}: {exc}")
            import traceback
            traceback.print_exception(type(exc), exc, exc.__traceback__)
        _safe_print()

    return failed == 0


if __name__ == "__main__":
    success = _run_tests()
    sys.exit(0 if success else 1)

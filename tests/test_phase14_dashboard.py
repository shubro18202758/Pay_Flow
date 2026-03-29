"""
PayFlow -- Phase 14: Dashboard & Event Broadcasting Test Suite
================================================================
Tests for the real-time web dashboard infrastructure:
  - EventBroadcaster pub/sub singleton
  - Dashboard REST endpoints (FastAPI TestClient)
  - SSE event stream
  - Subsystem event publication hooks

Run:
    PYTHONPATH=. python tests/test_phase14_dashboard.py
"""

from __future__ import annotations

import asyncio
import json
import sys
import time

# ── Safe printing (no encoding errors) ──────────────────────────────────────

def _safe_print(*args, **kwargs):
    try:
        print(*args, **kwargs)
    except UnicodeEncodeError:
        text = " ".join(str(a) for a in args)
        print(text.encode("ascii", errors="replace").decode(), **kwargs)


# ════════════════════════════════════════════════════════════════════════════
# EVENT BROADCASTER TESTS
# ════════════════════════════════════════════════════════════════════════════

def test_broadcaster_singleton():
    """EventBroadcaster.get() returns the same instance."""
    from src.api.events import EventBroadcaster
    EventBroadcaster.reset()
    a = EventBroadcaster.get()
    b = EventBroadcaster.get()
    assert a is b, "Singleton should return same instance"
    EventBroadcaster.reset()


def test_publish_sync_delivers_to_subscriber():
    """publish_sync delivers event to subscribed queue."""

    async def _run():
        from src.api.events import EventBroadcaster
        EventBroadcaster.reset()
        broadcaster = EventBroadcaster.get()
        queue = await broadcaster.subscribe(["test_channel"])

        broadcaster.publish_sync("test_channel", {"msg": "hello"})

        event = queue.get_nowait()
        assert event["channel"] == "test_channel"
        assert event["data"]["msg"] == "hello"
        assert "timestamp" in event
        EventBroadcaster.reset()

    asyncio.run(_run())


def test_publish_sync_backpressure():
    """publish_sync drops events when queue is full."""

    async def _run():
        from src.api.events import EventBroadcaster
        EventBroadcaster.reset()
        broadcaster = EventBroadcaster.get()
        queue = await broadcaster.subscribe(["bp"], max_size=2)

        # Fill the queue
        broadcaster.publish_sync("bp", {"i": 1})
        broadcaster.publish_sync("bp", {"i": 2})
        # This should be silently dropped (no exception)
        broadcaster.publish_sync("bp", {"i": 3})

        assert queue.qsize() == 2, f"Queue should be full at 2, got {queue.qsize()}"
        EventBroadcaster.reset()

    asyncio.run(_run())


def test_publish_no_subscribers():
    """Publishing to a channel with no subscribers is a no-op."""

    async def _run():
        from src.api.events import EventBroadcaster
        EventBroadcaster.reset()
        broadcaster = EventBroadcaster.get()
        # Should not raise
        broadcaster.publish_sync("empty_channel", {"data": "test"})
        await broadcaster.publish("empty_channel", {"data": "test"})
        EventBroadcaster.reset()

    asyncio.run(_run())


def test_subscribe_multiple_channels():
    """A single queue can subscribe to multiple channels."""

    async def _run():
        from src.api.events import EventBroadcaster
        EventBroadcaster.reset()
        broadcaster = EventBroadcaster.get()
        queue = await broadcaster.subscribe(["ch1", "ch2"])

        broadcaster.publish_sync("ch1", {"from": "ch1"})
        broadcaster.publish_sync("ch2", {"from": "ch2"})

        e1 = queue.get_nowait()
        e2 = queue.get_nowait()
        channels = {e1["channel"], e2["channel"]}
        assert channels == {"ch1", "ch2"}, f"Expected both channels, got {channels}"
        EventBroadcaster.reset()

    asyncio.run(_run())


def test_unsubscribe():
    """Unsubscribe removes queue from channels."""

    async def _run():
        from src.api.events import EventBroadcaster
        EventBroadcaster.reset()
        broadcaster = EventBroadcaster.get()
        queue = await broadcaster.subscribe(["unsub_ch"])

        assert broadcaster.subscriber_count("unsub_ch") == 1
        await broadcaster.unsubscribe(queue, ["unsub_ch"])
        assert broadcaster.subscriber_count("unsub_ch") == 0

        # Publishing should not deliver to unsubscribed queue
        broadcaster.publish_sync("unsub_ch", {"data": "orphan"})
        assert queue.empty(), "Queue should be empty after unsubscribe"
        EventBroadcaster.reset()

    asyncio.run(_run())


def test_async_publish():
    """Async publish delivers correctly."""

    async def _run():
        from src.api.events import EventBroadcaster
        EventBroadcaster.reset()
        broadcaster = EventBroadcaster.get()
        queue = await broadcaster.subscribe(["async_ch"])

        await broadcaster.publish("async_ch", {"mode": "async"})

        event = queue.get_nowait()
        assert event["data"]["mode"] == "async"
        EventBroadcaster.reset()

    asyncio.run(_run())


def test_broadcaster_snapshot():
    """Snapshot returns channel subscriber counts."""

    async def _run():
        from src.api.events import EventBroadcaster
        EventBroadcaster.reset()
        broadcaster = EventBroadcaster.get()
        await broadcaster.subscribe(["snap_ch"])
        await broadcaster.subscribe(["snap_ch"])

        snap = broadcaster.snapshot()
        assert snap["snap_ch"] == 2, f"Expected 2 subscribers, got {snap}"
        EventBroadcaster.reset()

    asyncio.run(_run())


# ════════════════════════════════════════════════════════════════════════════
# DASHBOARD CONFIG TESTS
# ════════════════════════════════════════════════════════════════════════════

def test_dashboard_config_defaults():
    """DashboardConfig has correct defaults."""
    from config.settings import DashboardConfig
    cfg = DashboardConfig()
    assert cfg.enable_dashboard is True
    assert cfg.host == "127.0.0.1"
    assert cfg.port == 8000
    assert cfg.max_graph_nodes_display == 500
    assert cfg.max_graph_edges_display == 1000
    assert cfg.sse_keepalive_seconds == 15
    assert cfg.sse_queue_max_size == 256
    assert cfg.system_telemetry_interval_sec == 1.0


def test_dashboard_config_frozen():
    """DashboardConfig is immutable."""
    from config.settings import DashboardConfig
    cfg = DashboardConfig()
    try:
        cfg.port = 9999
        assert False, "Should raise FrozenInstanceError"
    except AttributeError:
        pass  # Expected


def test_dashboard_config_exported():
    """DashboardConfig and DASHBOARD_CFG are exported from config package."""
    from config import DashboardConfig, DASHBOARD_CFG
    assert isinstance(DASHBOARD_CFG, DashboardConfig)


# ════════════════════════════════════════════════════════════════════════════
# APP FACTORY TESTS
# ════════════════════════════════════════════════════════════════════════════

def test_create_app_returns_fastapi():
    """create_app returns a FastAPI instance with routes."""
    from src.api.app import create_app
    app = create_app(orchestrator=None)
    assert app.title == "PayFlow Dashboard"
    route_paths = [r.path for r in app.routes]
    assert "/" in route_paths, f"Missing / route: {route_paths}"


def test_app_stores_orchestrator():
    """App stores orchestrator reference on state."""
    from src.api.app import create_app

    class FakeOrch:
        pass

    orch = FakeOrch()
    app = create_app(orchestrator=orch)
    assert app.state.orchestrator is orch


# ════════════════════════════════════════════════════════════════════════════
# ROUTE TESTS (FastAPI TestClient)
# ════════════════════════════════════════════════════════════════════════════

def _make_test_client():
    """Create a TestClient with a mock orchestrator."""
    from src.api.app import create_app
    from unittest.mock import MagicMock, AsyncMock
    from src.api.events import EventBroadcaster

    EventBroadcaster.reset()

    # Build a minimal mock orchestrator
    mock_orch = MagicMock()
    mock_orch.full_snapshot.return_value = {
        "orchestrator": {"events_ingested": 1000, "events_per_sec": 50.0,
                         "ml_inferences": 500, "alerts_routed": 20, "elapsed_sec": 10.0,
                         "features_extracted": 900},
        "hardware": {"gpu_vram_used_mb": 4000, "gpu_vram_total_mb": 8192},
    }

    # Mock graph with a small networkx graph
    import networkx as nx
    g = nx.MultiDiGraph()
    g.add_node("ACC-001", first_seen=1000, last_seen=2000, txn_count=5)
    g.add_node("ACC-002", first_seen=1000, last_seen=2000, txn_count=3)
    g.add_edge("ACC-001", "ACC-002", key="TXN-001",
               timestamp=2000, amount_paisa=100000, channel=4, fraud_label=0)

    mock_graph = MagicMock()
    mock_graph._graph = g
    mock_orch._graph = mock_graph

    # Mock circuit breaker
    mock_breaker = MagicMock()
    mock_breaker.is_frozen.return_value = False
    mock_breaker.get_frozen_nodes.return_value = []
    mock_breaker.snapshot.return_value = {
        "frozen_count": 0, "pending_alerts": 0,
        "cooldowns_active": 0, "metrics": {}, "config": {},
    }
    mock_breaker._agent_listener = None
    mock_orch._breaker = mock_breaker

    # Mock ledger
    mock_ledger = MagicMock()
    mock_ledger.get_recent_blocks = AsyncMock(return_value=[])
    mock_orch._ledger = mock_ledger

    app = create_app(orchestrator=mock_orch)
    return app, mock_orch


def test_route_dashboard_page():
    """GET / returns 200 with HTML."""
    from fastapi.testclient import TestClient
    app, _ = _make_test_client()
    client = TestClient(app)
    resp = client.get("/")
    assert resp.status_code == 200, f"Expected 200, got {resp.status_code}"
    assert "text/html" in resp.headers.get("content-type", "")
    assert "PayFlow" in resp.text
    from src.api.events import EventBroadcaster
    EventBroadcaster.reset()


def test_route_snapshot():
    """GET /api/v1/snapshot returns orchestrator snapshot."""
    from fastapi.testclient import TestClient
    app, mock_orch = _make_test_client()
    client = TestClient(app)
    resp = client.get("/api/v1/snapshot")
    assert resp.status_code == 200
    data = resp.json()
    assert "orchestrator" in data
    assert data["orchestrator"]["events_ingested"] == 1000
    from src.api.events import EventBroadcaster
    EventBroadcaster.reset()


def test_route_graph_topology():
    """GET /api/v1/graph/topology returns Cytoscape-formatted data."""
    from fastapi.testclient import TestClient
    app, _ = _make_test_client()
    client = TestClient(app)
    resp = client.get("/api/v1/graph/topology?limit=100")
    assert resp.status_code == 200
    data = resp.json()
    assert "nodes" in data
    assert "edges" in data
    assert len(data["nodes"]) == 2, f"Expected 2 nodes, got {len(data['nodes'])}"
    assert len(data["edges"]) == 1, f"Expected 1 edge, got {len(data['edges'])}"
    # Verify Cytoscape.js format
    node = data["nodes"][0]
    assert "data" in node
    assert "id" in node["data"]
    assert "status" in node["data"]
    edge = data["edges"][0]
    assert edge["data"]["source"] in ("ACC-001", "ACC-002")
    assert edge["data"]["target"] in ("ACC-001", "ACC-002")
    from src.api.events import EventBroadcaster
    EventBroadcaster.reset()


def test_route_circuit_breaker_status():
    """GET /api/v1/circuit-breaker/status returns freeze orders."""
    from fastapi.testclient import TestClient
    app, _ = _make_test_client()
    client = TestClient(app)
    resp = client.get("/api/v1/circuit-breaker/status")
    assert resp.status_code == 200
    data = resp.json()
    assert "freeze_orders" in data
    assert "snapshot" in data
    assert isinstance(data["freeze_orders"], list)
    from src.api.events import EventBroadcaster
    EventBroadcaster.reset()


def test_route_agent_verdicts():
    """GET /api/v1/agent/verdicts returns verdict list."""
    from fastapi.testclient import TestClient
    app, _ = _make_test_client()
    client = TestClient(app)
    resp = client.get("/api/v1/agent/verdicts?limit=5")
    assert resp.status_code == 200
    data = resp.json()
    assert "verdicts" in data
    assert isinstance(data["verdicts"], list)
    from src.api.events import EventBroadcaster
    EventBroadcaster.reset()


def test_route_sse_content_type():
    """SSE endpoint returns a StreamingResponse with text/event-stream media type."""
    from src.api.events import EventBroadcaster
    from src.api.routes.dashboard import event_stream
    from fastapi.responses import StreamingResponse

    EventBroadcaster.reset()

    async def _run():
        from unittest.mock import MagicMock, AsyncMock
        mock_request = MagicMock()
        mock_request.is_disconnected = AsyncMock(return_value=True)
        resp = await event_stream(mock_request)
        assert isinstance(resp, StreamingResponse), f"Expected StreamingResponse, got {type(resp)}"
        assert resp.media_type == "text/event-stream", (
            f"Expected text/event-stream, got {resp.media_type}"
        )
        EventBroadcaster.reset()

    asyncio.run(_run())


def test_route_snapshot_no_orchestrator():
    """Endpoints return safe defaults when orchestrator is None."""
    from fastapi.testclient import TestClient
    from src.api.app import create_app
    from src.api.events import EventBroadcaster
    EventBroadcaster.reset()
    app = create_app(orchestrator=None)
    client = TestClient(app)

    resp = client.get("/api/v1/snapshot")
    assert resp.status_code == 200
    assert "error" in resp.json()

    resp = client.get("/api/v1/graph/topology")
    assert resp.status_code == 200
    data = resp.json()
    assert data["nodes"] == []
    assert data["edges"] == []

    resp = client.get("/api/v1/circuit-breaker/status")
    assert resp.status_code == 200
    assert resp.json()["freeze_orders"] == []

    resp = client.get("/api/v1/agent/verdicts")
    assert resp.status_code == 200
    assert resp.json()["verdicts"] == []

    EventBroadcaster.reset()


# ════════════════════════════════════════════════════════════════════════════
# SSE INTEGRATION TESTS
# ════════════════════════════════════════════════════════════════════════════

def test_sse_receives_published_event():
    """SSE stream receives events published via EventBroadcaster."""

    async def _run():
        from src.api.events import EventBroadcaster
        EventBroadcaster.reset()
        broadcaster = EventBroadcaster.get()

        # Subscribe to the same channels as the SSE endpoint
        queue = await broadcaster.subscribe(
            ["graph", "agent", "circuit_breaker", "risk_scores", "system"],
        )

        # Publish a test event
        await broadcaster.publish("agent", {
            "type": "verdict",
            "txn_id": "TXN-test-001",
            "verdict": "FRAUDULENT",
            "confidence": 0.97,
        })

        event = await asyncio.wait_for(queue.get(), timeout=2.0)
        assert event["channel"] == "agent"
        assert event["data"]["type"] == "verdict"
        assert event["data"]["confidence"] == 0.97
        EventBroadcaster.reset()

    asyncio.run(_run())


def test_multiple_subscribers_receive_all_events():
    """Multiple concurrent subscribers each receive all events."""

    async def _run():
        from src.api.events import EventBroadcaster
        EventBroadcaster.reset()
        broadcaster = EventBroadcaster.get()

        q1 = await broadcaster.subscribe(["multi_ch"])
        q2 = await broadcaster.subscribe(["multi_ch"])

        await broadcaster.publish("multi_ch", {"id": 1})

        e1 = q1.get_nowait()
        e2 = q2.get_nowait()
        assert e1["data"]["id"] == 1
        assert e2["data"]["id"] == 1
        EventBroadcaster.reset()

    asyncio.run(_run())


# ════════════════════════════════════════════════════════════════════════════
# TEST RUNNER
# ════════════════════════════════════════════════════════════════════════════

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
    _safe_print(f"  Phase 14 Dashboard Test Suite  ({len(tests)} tests)")
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

    _safe_print(f"\n{'─' * 66}")
    _safe_print(f"  Results: {passed} passed, {failed} failed, {passed + failed} total")
    _safe_print(f"{'─' * 66}\n")

    if errors:
        _safe_print("  FAILURES:")
        for name, exc in errors:
            _safe_print(f"    {name}: {exc}")
        _safe_print()

    return failed == 0


if __name__ == "__main__":
    success = _run_tests()
    sys.exit(0 if success else 1)

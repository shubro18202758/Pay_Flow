"""
PayFlow -- Phase 13 Tests: Agent-Triggered Blockchain Circuit Breaker Listeners
================================================================================
Verifies AgentBreakerConfig defaults/immutability, DeviceBanEntry lifecycle,
reasoning chain SHA-256 hashing, two-tier verdict gate logic (CRITICAL vs
HIGH_SUSPICION), node freeze cascade (sender + receiver + mule), device
fingerprint ban/TTL, routing pause/TTL, ZKP proof generation, immutable
ledger anchoring, execution latency tracking, and metrics snapshots.

Tests:
 1. AgentBreakerConfig defaults match spec
 2. AgentBreakerConfig frozen rejects assignment
 3. DeviceBanEntry construction with all fields
 4. DeviceBanEntry is_expired TTL check
 5. Reasoning hash deterministic (same input -> same SHA-256)
 6. Reasoning hash canonical ordering (key order irrelevant)
 7. CRITICAL verdict triggers full response (freeze + ban + anchor)
 8. Non-critical verdict returns None (SUSPICIOUS + 0.60)
 9. HIGH_SUSPICION triggers routing pause only (no freeze/ban)
10. Borderline confidence below critical triggers routing pause only
11. Freeze sender and receiver via CircuitBreaker
12. Mule cascade freeze includes detected mule nodes
13. Mule cascade cap (max_cascade_nodes) respected
14. Device fingerprint banned on critical verdict
15. Device ban TTL expiry
16. is_device_banned check returns correct boolean
17. Routing pause set populated on critical verdict
18. Routing pause TTL expiry
19. Ledger anchoring of AgentBreakerEvent
20. ZKP proof generated and anchored
21. Execution latency tracked in event and metrics
22. AgentBreakerMetrics snapshot serializable to JSON
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
from dataclasses import FrozenInstanceError, dataclass
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import AGENT_BREAKER_CFG, AgentBreakerConfig
from src.blockchain.agent_breaker import (
    AgentBreakerEvent,
    AgentBreakerMetrics,
    AgentCircuitBreakerListener,
    DeviceBanEntry,
    hash_reasoning_chain,
)
from src.blockchain.circuit_breaker import CircuitBreaker, FreezeOrder


# ── Helpers ──────────────────────────────────────────────────────────────────

def _safe_print(msg: str) -> None:
    """Print with ASCII fallback for Windows cp1252 consoles."""
    try:
        print(msg)
    except UnicodeEncodeError:
        print(msg.encode("ascii", "replace").decode())


passed = 0
failed = 0


def run_test(func):
    global passed, failed
    name = func.__name__
    try:
        if asyncio.iscoroutinefunction(func):
            asyncio.run(func())
        else:
            func()
        _safe_print(f"  PASS  {name}")
        passed += 1
    except Exception as e:
        _safe_print(f"  FAIL  {name}: {e}")
        failed += 1


# ── Mock Verdict ─────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class MockVerdict:
    """Lightweight VerdictPayload mock for testing."""
    txn_id: str = "TXN_AB_001"
    node_id: str = "ACC_SENDER_01"
    verdict: str = "FRAUDULENT"
    confidence: float = 0.97
    fraud_typology: str | None = "LAYERING"
    reasoning_summary: str = "Multi-hop layering detected through rapid fund movement."
    evidence_cited: tuple = ("graph_3hop_subgraph", "velocity_spike")
    recommended_action: str = "FREEZE"
    thinking_steps: int = 3
    tools_used: tuple = ("query_graph_database", "check_velocity")
    total_duration_ms: float = 1200.5


SAMPLE_ALERT = {
    "txn_id": "TXN_AB_001",
    "sender_id": "ACC_SENDER_01",
    "receiver_id": "ACC_RECEIVER_01",
    "timestamp": int(time.time()),
    "risk_score": 0.95,
    "tier": "HIGH",
    "device_fingerprint": "a1b2c3d4e5f60718",
}


# ── Test 1: Config defaults ─────────────────────────────────────────────────

def test_agent_breaker_config_defaults():
    cfg = AgentBreakerConfig()
    assert cfg.critical_confidence_threshold == 0.95, "critical threshold"
    assert cfg.high_suspicion_threshold == 0.80, "high suspicion threshold"
    assert cfg.device_ban_ttl_seconds == 7200, "device ban TTL"
    assert cfg.max_banned_devices == 5000, "max banned devices"
    assert cfg.routing_pause_ttl_seconds == 3600, "routing pause TTL"
    assert cfg.enable_mule_cascade_freeze is True, "mule cascade enabled"
    assert cfg.max_cascade_nodes == 10, "max cascade"
    # Verify singleton matches
    assert AGENT_BREAKER_CFG.critical_confidence_threshold == 0.95


# ── Test 2: Config immutability ──────────────────────────────────────────────

def test_agent_breaker_config_immutability():
    cfg = AgentBreakerConfig()
    try:
        cfg.critical_confidence_threshold = 0.5   # type: ignore[misc]
        raise AssertionError("Should have raised FrozenInstanceError")
    except FrozenInstanceError:
        pass  # expected


# ── Test 3: DeviceBanEntry construction ──────────────────────────────────────

def test_device_ban_entry_construction():
    entry = DeviceBanEntry(
        device_fingerprint="abc123def456",
        banned_at=time.time(),
        trigger_txn_id="TXN_001",
        ttl_seconds=7200,
        reason="Agent critical fraud",
    )
    assert entry.device_fingerprint == "abc123def456"
    assert entry.ttl_seconds == 7200
    d = entry.to_dict()
    assert "device_fingerprint" in d
    assert "banned_at" in d


# ── Test 4: DeviceBanEntry TTL expiry ────────────────────────────────────────

def test_device_ban_expiry():
    # Not expired
    entry_fresh = DeviceBanEntry(
        device_fingerprint="abc123",
        banned_at=time.time(),
        trigger_txn_id="TXN_001",
        ttl_seconds=7200,
        reason="test",
    )
    assert not entry_fresh.is_expired, "fresh entry should not be expired"

    # Expired (banned 2 hours + 1 second ago with 2-hour TTL)
    entry_old = DeviceBanEntry(
        device_fingerprint="abc123",
        banned_at=time.time() - 7201,
        trigger_txn_id="TXN_001",
        ttl_seconds=7200,
        reason="test",
    )
    assert entry_old.is_expired, "old entry should be expired"


# ── Test 5: Reasoning hash deterministic ─────────────────────────────────────

def test_reasoning_hash_deterministic():
    v = MockVerdict()
    h1 = hash_reasoning_chain(v)
    h2 = hash_reasoning_chain(v)
    assert h1 == h2, "same verdict must produce same hash"
    assert len(h1) == 64, "SHA-256 hex digest is 64 chars"
    # All hex chars
    assert all(c in "0123456789abcdef" for c in h1)


# ── Test 6: Reasoning hash canonical ordering ────────────────────────────────

def test_reasoning_hash_canonical_ordering():
    v1 = MockVerdict(
        evidence_cited=("a", "b"),
        tools_used=("z_tool", "a_tool"),
    )
    v2 = MockVerdict(
        evidence_cited=("a", "b"),
        tools_used=("a_tool", "z_tool"),  # different input order
    )
    # tools_used is sorted internally by hash_reasoning_chain, so order doesn't matter
    h1 = hash_reasoning_chain(v1)
    h2 = hash_reasoning_chain(v2)
    assert h1 == h2, "tool order should not affect hash (canonical sort)"


# ── Test 7: CRITICAL verdict triggers full response ──────────────────────────

async def test_critical_verdict_triggers_full_response():
    breaker = CircuitBreaker(config=MagicMock(
        consensus_threshold=0.80,
        freeze_ttl_seconds=3600,
        cooldown_seconds=0,
        max_frozen_nodes=10000,
    ))
    mock_ledger = AsyncMock()
    mock_ledger.anchor_circuit_breaker = AsyncMock()
    mock_ledger.anchor_zkp_proof = AsyncMock()

    listener = AgentCircuitBreakerListener(
        circuit_breaker=breaker,
        audit_ledger=mock_ledger,
        config=AgentBreakerConfig(),
    )

    verdict = MockVerdict(confidence=0.97, verdict="FRAUDULENT", recommended_action="FREEZE")
    event = await listener.on_verdict(verdict, SAMPLE_ALERT)

    assert event is not None, "critical verdict must produce event"
    assert len(event.nodes_frozen) >= 2, "sender + receiver frozen"
    assert len(event.devices_banned) == 1, "device fingerprint banned"
    assert event.devices_banned[0] == "a1b2c3d4e5f60718"
    assert len(event.reasoning_hash) == 64, "SHA-256 hash present"
    assert event.latency_ms > 0, "latency tracked"
    assert listener.metrics.critical_events == 1


# ── Test 8: Non-critical verdict returns None ────────────────────────────────

async def test_non_critical_verdict_returns_none():
    breaker = CircuitBreaker(config=MagicMock(
        consensus_threshold=0.80,
        freeze_ttl_seconds=3600,
    ))
    listener = AgentCircuitBreakerListener(
        circuit_breaker=breaker,
        config=AgentBreakerConfig(),
    )

    verdict = MockVerdict(
        verdict="SUSPICIOUS", confidence=0.60, recommended_action="MONITOR",
    )
    event = await listener.on_verdict(verdict, SAMPLE_ALERT)
    assert event is None, "non-critical verdict should return None"
    assert listener.metrics.critical_events == 0
    assert listener.metrics.high_suspicion_events == 0


# ── Test 9: HIGH_SUSPICION triggers routing pause only ───────────────────────

async def test_high_suspicion_triggers_routing_pause_only():
    breaker = CircuitBreaker(config=MagicMock(
        consensus_threshold=0.80,
        freeze_ttl_seconds=3600,
    ))
    listener = AgentCircuitBreakerListener(
        circuit_breaker=breaker,
        config=AgentBreakerConfig(),
    )

    verdict = MockVerdict(
        verdict="FRAUDULENT", confidence=0.85, recommended_action="ESCALATE",
    )
    event = await listener.on_verdict(verdict, SAMPLE_ALERT)

    assert event is not None, "high suspicion must produce event"
    assert len(event.nodes_frozen) == 0, "no freeze on high suspicion"
    assert len(event.devices_banned) == 0, "no device ban on high suspicion"
    assert len(event.routing_paused_nodes) >= 2, "routing paused for sender + receiver"
    assert listener.metrics.high_suspicion_events == 1
    assert listener.metrics.critical_events == 0


# ── Test 10: Borderline confidence below critical ────────────────────────────

async def test_borderline_confidence_below_critical():
    breaker = CircuitBreaker(config=MagicMock(
        consensus_threshold=0.80,
        freeze_ttl_seconds=3600,
    ))
    listener = AgentCircuitBreakerListener(
        circuit_breaker=breaker,
        config=AgentBreakerConfig(),
    )

    # 0.94 FRAUDULENT + FREEZE -> just below critical (0.95), but above high suspicion (0.80)
    verdict = MockVerdict(
        verdict="FRAUDULENT", confidence=0.94, recommended_action="FREEZE",
    )
    event = await listener.on_verdict(verdict, SAMPLE_ALERT)

    assert event is not None, "0.94 should still trigger high suspicion"
    assert len(event.nodes_frozen) == 0, "no freeze below critical"
    assert len(event.routing_paused_nodes) >= 2, "routing paused"
    assert listener.metrics.high_suspicion_events == 1


# ── Test 11: Freeze sender and receiver ──────────────────────────────────────

async def test_freeze_sender_and_receiver():
    breaker = CircuitBreaker(config=MagicMock(
        consensus_threshold=0.80,
        freeze_ttl_seconds=3600,
        cooldown_seconds=0,
        max_frozen_nodes=10000,
    ))
    mock_ledger = AsyncMock()
    mock_ledger.anchor_circuit_breaker = AsyncMock()
    mock_ledger.anchor_zkp_proof = AsyncMock()

    listener = AgentCircuitBreakerListener(
        circuit_breaker=breaker,
        audit_ledger=mock_ledger,
        config=AgentBreakerConfig(),
    )

    verdict = MockVerdict(confidence=0.97)
    event = await listener.on_verdict(verdict, SAMPLE_ALERT)

    assert breaker.is_frozen("ACC_SENDER_01"), "sender should be frozen"
    assert breaker.is_frozen("ACC_RECEIVER_01"), "receiver should be frozen"
    assert "ACC_SENDER_01" in event.nodes_frozen
    assert "ACC_RECEIVER_01" in event.nodes_frozen


# ── Test 12: Mule cascade freeze ────────────────────────────────────────────

async def test_mule_cascade_freeze():
    breaker = CircuitBreaker(config=MagicMock(
        consensus_threshold=0.80,
        freeze_ttl_seconds=3600,
        cooldown_seconds=0,
        max_frozen_nodes=10000,
    ))
    mock_ledger = AsyncMock()
    mock_ledger.anchor_circuit_breaker = AsyncMock()
    mock_ledger.anchor_zkp_proof = AsyncMock()

    # Mock transaction graph that returns mule nodes
    mock_graph = MagicMock()
    mock_graph.detect_mule_around = MagicMock(side_effect=[
        ["MULE_NODE_A"],    # mule around sender
        ["MULE_NODE_B"],    # mule around receiver
    ])

    listener = AgentCircuitBreakerListener(
        circuit_breaker=breaker,
        audit_ledger=mock_ledger,
        transaction_graph=mock_graph,
        config=AgentBreakerConfig(),
    )

    verdict = MockVerdict(confidence=0.97)
    event = await listener.on_verdict(verdict, SAMPLE_ALERT)

    assert "MULE_NODE_A" in event.nodes_frozen, "mule A should be frozen"
    assert "MULE_NODE_B" in event.nodes_frozen, "mule B should be frozen"
    assert breaker.is_frozen("MULE_NODE_A")
    assert breaker.is_frozen("MULE_NODE_B")
    assert len(event.nodes_frozen) == 4  # sender + receiver + 2 mules


# ── Test 13: Mule cascade cap ───────────────────────────────────────────────

async def test_mule_cascade_cap():
    breaker = CircuitBreaker(config=MagicMock(
        consensus_threshold=0.80,
        freeze_ttl_seconds=3600,
        cooldown_seconds=0,
        max_frozen_nodes=10000,
    ))
    mock_ledger = AsyncMock()
    mock_ledger.anchor_circuit_breaker = AsyncMock()
    mock_ledger.anchor_zkp_proof = AsyncMock()

    # Return many mule nodes - more than max_cascade_nodes
    mock_graph = MagicMock()
    many_mules = [f"MULE_{i}" for i in range(20)]
    mock_graph.detect_mule_around = MagicMock(side_effect=[
        many_mules,   # sender returns 20 mules
        [],           # receiver returns none
    ])

    cfg = AgentBreakerConfig(max_cascade_nodes=3)
    listener = AgentCircuitBreakerListener(
        circuit_breaker=breaker,
        audit_ledger=mock_ledger,
        transaction_graph=mock_graph,
        config=cfg,
    )

    verdict = MockVerdict(confidence=0.97)
    event = await listener.on_verdict(verdict, SAMPLE_ALERT)

    # sender + receiver + max 3 mule cascade = max 5
    mule_nodes = [n for n in event.nodes_frozen if n.startswith("MULE_")]
    assert len(mule_nodes) <= 3, f"mule cascade capped at 3, got {len(mule_nodes)}"


# ── Test 14: Device fingerprint banned ───────────────────────────────────────

async def test_device_fingerprint_banned():
    breaker = CircuitBreaker(config=MagicMock(
        consensus_threshold=0.80,
        freeze_ttl_seconds=3600,
        cooldown_seconds=0,
        max_frozen_nodes=10000,
    ))
    mock_ledger = AsyncMock()
    mock_ledger.anchor_circuit_breaker = AsyncMock()
    mock_ledger.anchor_zkp_proof = AsyncMock()

    listener = AgentCircuitBreakerListener(
        circuit_breaker=breaker,
        audit_ledger=mock_ledger,
        config=AgentBreakerConfig(),
    )

    verdict = MockVerdict(confidence=0.97)
    await listener.on_verdict(verdict, SAMPLE_ALERT)

    assert listener.is_device_banned("a1b2c3d4e5f60718"), "device should be banned"
    assert not listener.is_device_banned("zzzzzzzzzzzzzzzz"), "unknown device not banned"
    assert listener.metrics.devices_banned == 1


# ── Test 15: Device ban TTL expiry ───────────────────────────────────────────

async def test_device_ban_ttl_expiry():
    breaker = CircuitBreaker(config=MagicMock(
        consensus_threshold=0.80,
        freeze_ttl_seconds=3600,
        cooldown_seconds=0,
        max_frozen_nodes=10000,
    ))
    mock_ledger = AsyncMock()
    mock_ledger.anchor_circuit_breaker = AsyncMock()
    mock_ledger.anchor_zkp_proof = AsyncMock()

    # Short TTL for test (1 second)
    cfg = AgentBreakerConfig(device_ban_ttl_seconds=1)
    listener = AgentCircuitBreakerListener(
        circuit_breaker=breaker,
        audit_ledger=mock_ledger,
        config=cfg,
    )

    verdict = MockVerdict(confidence=0.97)
    await listener.on_verdict(verdict, SAMPLE_ALERT)

    assert listener.is_device_banned("a1b2c3d4e5f60718"), "initially banned"

    # Manually expire the ban by adjusting the entry's banned_at
    fp = "a1b2c3d4e5f60718"
    old_entry = listener._banned_devices[fp]
    listener._banned_devices[fp] = DeviceBanEntry(
        device_fingerprint=fp,
        banned_at=time.time() - 2,  # 2 seconds ago, TTL is 1
        trigger_txn_id=old_entry.trigger_txn_id,
        ttl_seconds=1,
        reason=old_entry.reason,
    )
    assert not listener.is_device_banned(fp), "should be expired now"


# ── Test 16: is_device_banned check ──────────────────────────────────────────

def test_is_device_banned_check():
    breaker = CircuitBreaker(config=MagicMock(
        consensus_threshold=0.80,
        freeze_ttl_seconds=3600,
    ))
    listener = AgentCircuitBreakerListener(
        circuit_breaker=breaker,
        config=AgentBreakerConfig(),
    )

    # No devices banned initially
    assert not listener.is_device_banned("abc123")

    # Manually add a ban entry
    listener._banned_devices["abc123"] = DeviceBanEntry(
        device_fingerprint="abc123",
        banned_at=time.time(),
        trigger_txn_id="TXN_001",
        ttl_seconds=7200,
        reason="test",
    )
    assert listener.is_device_banned("abc123"), "should be banned"
    assert not listener.is_device_banned("def456"), "other device not banned"


# ── Test 17: Routing pause set populated ─────────────────────────────────────

async def test_routing_pause_set():
    breaker = CircuitBreaker(config=MagicMock(
        consensus_threshold=0.80,
        freeze_ttl_seconds=3600,
        cooldown_seconds=0,
        max_frozen_nodes=10000,
    ))
    mock_ledger = AsyncMock()
    mock_ledger.anchor_circuit_breaker = AsyncMock()
    mock_ledger.anchor_zkp_proof = AsyncMock()

    listener = AgentCircuitBreakerListener(
        circuit_breaker=breaker,
        audit_ledger=mock_ledger,
        config=AgentBreakerConfig(),
    )

    verdict = MockVerdict(confidence=0.97)
    event = await listener.on_verdict(verdict, SAMPLE_ALERT)

    assert listener.is_routing_paused("ACC_SENDER_01"), "sender routing paused"
    assert listener.is_routing_paused("ACC_RECEIVER_01"), "receiver routing paused"
    assert not listener.is_routing_paused("ACC_UNRELATED"), "unrelated not paused"


# ── Test 18: Routing pause TTL expiry ────────────────────────────────────────

def test_routing_pause_ttl_expiry():
    breaker = CircuitBreaker(config=MagicMock(
        consensus_threshold=0.80,
        freeze_ttl_seconds=3600,
    ))
    listener = AgentCircuitBreakerListener(
        circuit_breaker=breaker,
        config=AgentBreakerConfig(routing_pause_ttl_seconds=1),
    )

    # Manually set a routing pause that has already expired
    listener._routing_paused_nodes["ACC_EXPIRED"] = time.time() - 1
    assert not listener.is_routing_paused("ACC_EXPIRED"), "should be expired"

    # Set one that's still active
    listener._routing_paused_nodes["ACC_ACTIVE"] = time.time() + 3600
    assert listener.is_routing_paused("ACC_ACTIVE"), "should still be paused"


# ── Test 19: Ledger anchoring of AgentBreakerEvent ──────────────────────────

async def test_ledger_anchoring_event():
    breaker = CircuitBreaker(config=MagicMock(
        consensus_threshold=0.80,
        freeze_ttl_seconds=3600,
        cooldown_seconds=0,
        max_frozen_nodes=10000,
    ))
    mock_ledger = AsyncMock()
    mock_ledger.anchor_circuit_breaker = AsyncMock()
    mock_ledger.anchor_zkp_proof = AsyncMock()

    listener = AgentCircuitBreakerListener(
        circuit_breaker=breaker,
        audit_ledger=mock_ledger,
        config=AgentBreakerConfig(),
    )

    verdict = MockVerdict(confidence=0.97)
    event = await listener.on_verdict(verdict, SAMPLE_ALERT)

    # Verify ledger was called with the right action
    mock_ledger.anchor_circuit_breaker.assert_called_once()
    call_args = mock_ledger.anchor_circuit_breaker.call_args
    assert call_args[1]["action"] == "agent_critical_freeze"
    details = call_args[1]["details"]
    assert "event_id" in details
    assert "reasoning_hash" in details
    assert "nodes_frozen" in details
    assert len(details["reasoning_hash"]) == 64


# ── Test 20: ZKP proof generated and anchored ───────────────────────────────

async def test_zkp_proof_generated_and_anchored():
    breaker = CircuitBreaker(config=MagicMock(
        consensus_threshold=0.80,
        freeze_ttl_seconds=3600,
        cooldown_seconds=0,
        max_frozen_nodes=10000,
    ))
    mock_ledger = AsyncMock()
    mock_ledger.anchor_circuit_breaker = AsyncMock()
    mock_ledger.anchor_zkp_proof = AsyncMock()

    listener = AgentCircuitBreakerListener(
        circuit_breaker=breaker,
        audit_ledger=mock_ledger,
        config=AgentBreakerConfig(),
    )

    verdict = MockVerdict(confidence=0.97)
    event = await listener.on_verdict(verdict, SAMPLE_ALERT)

    # ZKP proof should have been anchored
    assert mock_ledger.anchor_zkp_proof.called, "ZKP proof should be anchored"
    assert event.zkp_proof_id is not None, "ZKP proof ID should be set"
    assert "agent_confidence >= 0.95" in event.zkp_proof_id


# ── Test 21: Execution latency tracked ──────────────────────────────────────

async def test_execution_latency_tracked():
    breaker = CircuitBreaker(config=MagicMock(
        consensus_threshold=0.80,
        freeze_ttl_seconds=3600,
        cooldown_seconds=0,
        max_frozen_nodes=10000,
    ))
    mock_ledger = AsyncMock()
    mock_ledger.anchor_circuit_breaker = AsyncMock()
    mock_ledger.anchor_zkp_proof = AsyncMock()

    listener = AgentCircuitBreakerListener(
        circuit_breaker=breaker,
        audit_ledger=mock_ledger,
        config=AgentBreakerConfig(),
    )

    verdict = MockVerdict(confidence=0.97)
    event = await listener.on_verdict(verdict, SAMPLE_ALERT)

    assert event.latency_ms > 0, "latency_ms must be positive"
    assert listener.metrics.total_latency_ms > 0, "total latency tracked in metrics"
    assert listener.metrics.critical_events == 1


# ── Test 22: Metrics snapshot serializable ───────────────────────────────────

async def test_agent_breaker_metrics_snapshot():
    breaker = CircuitBreaker(config=MagicMock(
        consensus_threshold=0.80,
        freeze_ttl_seconds=3600,
        cooldown_seconds=0,
        max_frozen_nodes=10000,
    ))
    mock_ledger = AsyncMock()
    mock_ledger.anchor_circuit_breaker = AsyncMock()
    mock_ledger.anchor_zkp_proof = AsyncMock()

    listener = AgentCircuitBreakerListener(
        circuit_breaker=breaker,
        audit_ledger=mock_ledger,
        config=AgentBreakerConfig(),
    )

    # Trigger a critical event
    verdict = MockVerdict(confidence=0.97)
    await listener.on_verdict(verdict, SAMPLE_ALERT)

    # Check metrics
    snap = listener.metrics.snapshot()
    serialized = json.dumps(snap)
    assert isinstance(serialized, str), "snapshot must be JSON serializable"

    parsed = json.loads(serialized)
    assert parsed["critical_events"] == 1
    assert parsed["nodes_frozen_by_agent"] >= 2
    assert parsed["devices_banned"] == 1

    # Check full snapshot
    full_snap = listener.snapshot()
    assert "metrics" in full_snap
    assert "config" in full_snap
    assert full_snap["config"]["critical_confidence_threshold"] == 0.95


# ── Runner ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    _safe_print("\n" + "=" * 72)
    _safe_print("PayFlow Phase 13: Agent-Triggered Blockchain Circuit Breaker")
    _safe_print("=" * 72 + "\n")

    tests = [
        test_agent_breaker_config_defaults,
        test_agent_breaker_config_immutability,
        test_device_ban_entry_construction,
        test_device_ban_expiry,
        test_reasoning_hash_deterministic,
        test_reasoning_hash_canonical_ordering,
        test_critical_verdict_triggers_full_response,
        test_non_critical_verdict_returns_none,
        test_high_suspicion_triggers_routing_pause_only,
        test_borderline_confidence_below_critical,
        test_freeze_sender_and_receiver,
        test_mule_cascade_freeze,
        test_mule_cascade_cap,
        test_device_fingerprint_banned,
        test_device_ban_ttl_expiry,
        test_is_device_banned_check,
        test_routing_pause_set,
        test_routing_pause_ttl_expiry,
        test_ledger_anchoring_event,
        test_zkp_proof_generated_and_anchored,
        test_execution_latency_tracked,
        test_agent_breaker_metrics_snapshot,
    ]

    for test_fn in tests:
        run_test(test_fn)

    _safe_print(f"\n{'=' * 72}")
    _safe_print(f"Results: {passed} passed, {failed} failed, {passed + failed} total")
    _safe_print(f"{'=' * 72}\n")

    if failed:
        sys.exit(1)

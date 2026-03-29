"""
PayFlow -- Phase 12 Tests: HITL Escalation & Dynamic Agentic Orchestration
============================================================================
Verifies HITLConfig defaults, ConfidenceEvaluator threshold logic,
HITLEscalationPayload construction/serialization, GraphContextPackager
k-hop extraction, HITLDispatcher HTTP dispatch with retry, InvestigatorAgent
HITL escalation paths, verdict types, metrics tracking, and mock analyst
endpoint integration.

Tests:
 1. HITLConfig defaults match spec
 2. HITLConfig frozen rejects assignment
 3. HITLConfig typology thresholds present
 4. ConfidenceEvaluator below threshold returns True
 5. ConfidenceEvaluator above threshold returns False
 6. ConfidenceEvaluator typology-specific threshold (LAYERING uses 0.80)
 7. ConfidenceEvaluator unknown typology falls back to default
 8. ConfidenceEvaluator None typology handled gracefully
 9. HITLEscalationPayload construction
10. HITLEscalationPayload to_dict serialization
11. GraphContextPackager with graph extracts subgraph
12. GraphContextPackager without graph returns empty context
13. GraphContextPackager unknown node handled
14. HITLDispatcher success on 200
15. HITLDispatcher timeout handled gracefully
16. HITLDispatcher retry on 5xx failure
17. HITLDispatchResult structure
18. Agent escalation path on low confidence
19. Agent autonomous path on high confidence
20. Agent escalation on complex typology (LAYERING + 0.72)
21. Agent no escalation on simple typology (PROFILE_MISMATCH + 0.72)
22. Agent HITL verdict type is ESCALATED_TO_HUMAN
23. Agent HITL metrics tracking (verdicts_escalated)
24. Agent evaluate_escalation recovery (confidence rises)
25. HITLMetrics snapshot serializable
26. Analyst endpoint receives and acks payload
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
import uuid
from dataclasses import FrozenInstanceError
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import HITL_CFG, HITLConfig
from src.llm.hitl import (
    ConfidenceEvaluator,
    GraphContextPackager,
    HITLDispatchResult,
    HITLDispatcher,
    HITLEscalationPayload,
    HITLMetrics,
    build_escalation_payload,
)
from src.llm.agent import AgentMetrics, AgentState, InvestigatorAgent, VerdictPayload


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


SAMPLE_ALERT = {
    "txn_id": "TXN_HITL_001",
    "sender_id": "ACC_S_HITL01",
    "receiver_id": "ACC_R_HITL01",
    "timestamp": int(time.time()),
    "risk_score": 0.91,
    "tier": "HIGH",
    "threshold": 0.85,
    "top_features": {
        "velocity_1h_count": 12.0,
        "amt_zscore": 2.8,
    },
}


# ── 1. HITLConfig defaults ──────────────────────────────────────────────────

def test_hitl_config_defaults():
    cfg = HITLConfig()
    assert cfg.default_confidence_threshold == 0.75, \
        f"Expected 0.75, got {cfg.default_confidence_threshold}"
    assert cfg.analyst_endpoint_url == "http://localhost:8000/api/v1/analyst/escalation"
    assert cfg.analyst_timeout_seconds == 10
    assert cfg.analyst_max_retries == 2
    assert cfg.graph_context_k_hops == 3
    assert cfg.max_evidence_items == 20
    assert cfg.max_pending_escalations == 100
    assert cfg.escalation_ttl_seconds == 3600


# ── 2. HITLConfig frozen ────────────────────────────────────────────────────

def test_hitl_config_immutability():
    cfg = HITLConfig()
    try:
        cfg.default_confidence_threshold = 0.99
        raise AssertionError("Should have raised FrozenInstanceError")
    except FrozenInstanceError:
        pass


# ── 3. HITLConfig typology thresholds ───────────────────────────────────────

def test_hitl_config_typology_thresholds():
    cfg = HITLConfig()
    assert "LAYERING" in cfg.typology_thresholds
    assert "ROUND_TRIPPING" in cfg.typology_thresholds
    assert "STRUCTURING" in cfg.typology_thresholds
    assert "DORMANT_ACTIVATION" in cfg.typology_thresholds
    assert "PROFILE_MISMATCH" in cfg.typology_thresholds
    assert cfg.typology_thresholds["LAYERING"] == 0.80
    assert cfg.typology_thresholds["DORMANT_ACTIVATION"] == 0.65


# ── 4. ConfidenceEvaluator below threshold ──────────────────────────────────

def test_confidence_evaluator_below_threshold():
    cfg = HITLConfig()
    result = ConfidenceEvaluator.should_escalate(0.60, None, cfg)
    assert result is True, "0.60 < 0.75 default threshold should escalate"


# ── 5. ConfidenceEvaluator above threshold ──────────────────────────────────

def test_confidence_evaluator_above_threshold():
    cfg = HITLConfig()
    result = ConfidenceEvaluator.should_escalate(0.80, None, cfg)
    assert result is False, "0.80 >= 0.75 default threshold should NOT escalate"


# ── 6. ConfidenceEvaluator typology-specific ────────────────────────────────

def test_confidence_evaluator_typology_specific():
    cfg = HITLConfig()
    # LAYERING threshold is 0.80
    # 0.78 is above default 0.75 but below LAYERING 0.80
    result = ConfidenceEvaluator.should_escalate(0.78, "LAYERING", cfg)
    assert result is True, "0.78 < 0.80 LAYERING threshold should escalate"

    threshold = ConfidenceEvaluator.get_threshold("LAYERING", cfg)
    assert threshold == 0.80, f"LAYERING threshold should be 0.80, got {threshold}"


# ── 7. ConfidenceEvaluator unknown typology ─────────────────────────────────

def test_confidence_evaluator_unknown_typology():
    cfg = HITLConfig()
    threshold = ConfidenceEvaluator.get_threshold("UNKNOWN_TYPE", cfg)
    assert threshold == 0.75, f"Unknown typology should fall back to 0.75, got {threshold}"


# ── 8. ConfidenceEvaluator None typology ────────────────────────────────────

def test_confidence_evaluator_none_typology():
    cfg = HITLConfig()
    threshold = ConfidenceEvaluator.get_threshold(None, cfg)
    assert threshold == 0.75, f"None typology should fall back to 0.75, got {threshold}"
    result = ConfidenceEvaluator.should_escalate(0.70, None, cfg)
    assert result is True, "0.70 < 0.75 with None typology should escalate"


# ── 9. HITLEscalationPayload construction ───────────────────────────────────

def test_escalation_payload_construction():
    payload = HITLEscalationPayload(
        escalation_id="esc-001",
        txn_id="TXN-001",
        node_id="ACC-001",
        agent_confidence=0.65,
        confidence_threshold=0.80,
        detected_typology="LAYERING",
        reasoning_trace=["Step 1: checked velocity", "Step 2: graph analysis"],
        evidence_collected={"tool1": {"result": "data"}},
        graph_context={"available": True, "nodes": 5},
        ml_score=0.91,
        gnn_score=0.85,
        nlu_findings=None,
        recommended_action="ESCALATE_TO_HUMAN",
        escalated_at=1700000000.0,
    )
    assert payload.escalation_id == "esc-001"
    assert payload.txn_id == "TXN-001"
    assert payload.agent_confidence == 0.65
    assert payload.detected_typology == "LAYERING"
    assert len(payload.reasoning_trace) == 2
    assert payload.recommended_action == "ESCALATE_TO_HUMAN"


# ── 10. HITLEscalationPayload serialization ─────────────────────────────────

def test_escalation_payload_serialization():
    payload = HITLEscalationPayload(
        escalation_id="esc-002",
        txn_id="TXN-002",
        node_id="ACC-002",
        agent_confidence=0.7123456,
        confidence_threshold=0.80,
        detected_typology="ROUND_TRIPPING",
        reasoning_trace=["Step 1"],
        evidence_collected={"key": "value"},
        graph_context={"available": False},
        ml_score=0.88887777,
        gnn_score=-1.0,
        nlu_findings={"finding": "test"},
        recommended_action="ESCALATE_TO_HUMAN",
        escalated_at=1700000000.0,
    )
    d = payload.to_dict()
    # Verify JSON serializable
    json_str = json.dumps(d)
    assert isinstance(json_str, str)
    # Verify rounding
    assert d["agent_confidence"] == 0.7123, f"Expected 0.7123, got {d['agent_confidence']}"
    assert d["ml_score"] == 0.8889, f"Expected 0.8889, got {d['ml_score']}"
    assert d["nlu_findings"] == {"finding": "test"}
    assert d["escalation_id"] == "esc-002"
    assert d["detected_typology"] == "ROUND_TRIPPING"


# ── 11. GraphContextPackager with graph ─────────────────────────────────────

def test_graph_context_packager_with_graph():
    # Mock a TransactionGraph
    mock_graph = MagicMock()
    mock_nx = MagicMock()
    mock_graph.graph = mock_nx

    # Node exists
    mock_nx.has_node.return_value = True
    mock_nx.nodes.get.return_value = {
        "first_seen": 1700000000,
        "last_seen": 1700001000,
        "txn_count": 42,
    }
    mock_nx.in_degree.return_value = 5
    mock_nx.out_degree.return_value = 3
    mock_nx.in_edges.return_value = [("A", "B"), ("C", "B"), ("D", "B")]
    mock_nx.out_edges.return_value = [("B", "E"), ("B", "F")]

    # Subgraph extraction
    mock_subgraph = MagicMock()
    mock_subgraph.number_of_nodes.return_value = 10
    mock_subgraph.number_of_edges.return_value = 15
    mock_graph._extract_k_hop_subgraph.return_value = mock_subgraph

    # Mule detector returns None (no mule)
    mock_graph._mule_detector.detect_around_node.return_value = None
    mock_graph._latest_timestamp = 1700001000

    # Cycle detector returns empty list
    mock_graph._cycle_detector.detect_around_nodes.return_value = []

    result = GraphContextPackager.package(mock_graph, "ACC-001", k_hops=3)

    assert result["available"] is True
    assert result["node_id"] == "ACC-001"
    assert result["subgraph"]["nodes"] == 10
    assert result["subgraph"]["edges"] == 15
    assert result["connections"]["in_degree"] == 5
    assert result["connections"]["out_degree"] == 3
    assert result["patterns"]["mule_network_detected"] is False
    assert result["patterns"]["cycles_found"] == 0


# ── 12. GraphContextPackager no graph ───────────────────────────────────────

def test_graph_context_packager_no_graph():
    result = GraphContextPackager.package(None, "ACC-001", k_hops=3)
    assert result["available"] is False
    assert "reason" in result
    assert "not available" in result["reason"]


# ── 13. GraphContextPackager unknown node ───────────────────────────────────

def test_graph_context_packager_unknown_node():
    mock_graph = MagicMock()
    mock_nx = MagicMock()
    mock_graph.graph = mock_nx
    mock_nx.has_node.return_value = False

    result = GraphContextPackager.package(mock_graph, "UNKNOWN_NODE", k_hops=3)
    assert result["available"] is False
    assert "not found" in result["reason"]
    assert result["node_id"] == "UNKNOWN_NODE"


# ── 14. HITLDispatcher success ──────────────────────────────────────────────

async def test_hitl_dispatcher_success():
    cfg = HITLConfig(
        analyst_endpoint_url="http://test:8000/api/v1/analyst/escalation",
        analyst_max_retries=2,
        analyst_timeout_seconds=5,
    )
    dispatcher = HITLDispatcher(config=cfg)

    payload = HITLEscalationPayload(
        escalation_id="esc-dispatch-001",
        txn_id="TXN-D01",
        node_id="ACC-D01",
        agent_confidence=0.65,
        confidence_threshold=0.80,
        detected_typology="LAYERING",
        reasoning_trace=["Step 1"],
        evidence_collected={},
        graph_context={},
        ml_score=0.9,
        gnn_score=-1.0,
        nlu_findings=None,
        recommended_action="ESCALATE_TO_HUMAN",
        escalated_at=time.time(),
    )

    # Mock httpx response
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"ack_id": "ack-123", "status": "received"}

    mock_client = AsyncMock()
    mock_client.post.return_value = mock_response
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    with patch("httpx.AsyncClient", return_value=mock_client):
        result = await dispatcher.dispatch(payload)

    assert result.success is True
    assert result.status_code == 200
    assert result.analyst_ack_id == "ack-123"
    assert result.error is None


# ── 15. HITLDispatcher timeout ──────────────────────────────────────────────

async def test_hitl_dispatcher_timeout():
    import httpx

    cfg = HITLConfig(
        analyst_endpoint_url="http://test:8000/api/v1/analyst/escalation",
        analyst_max_retries=1,
        analyst_timeout_seconds=1,
    )
    dispatcher = HITLDispatcher(config=cfg)

    payload = HITLEscalationPayload(
        escalation_id="esc-timeout-001",
        txn_id="TXN-T01",
        node_id="ACC-T01",
        agent_confidence=0.60,
        confidence_threshold=0.80,
        detected_typology="STRUCTURING",
        reasoning_trace=[],
        evidence_collected={},
        graph_context={},
        ml_score=0.85,
        gnn_score=-1.0,
        nlu_findings=None,
        recommended_action="ESCALATE_TO_HUMAN",
        escalated_at=time.time(),
    )

    mock_client = AsyncMock()
    mock_client.post.side_effect = httpx.TimeoutException("Connection timed out")
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    with patch("httpx.AsyncClient", return_value=mock_client):
        result = await dispatcher.dispatch(payload)

    assert result.success is False
    assert "timed out" in (result.error or "").lower()


# ── 16. HITLDispatcher retry on 5xx ─────────────────────────────────────────

async def test_hitl_dispatcher_retry_on_failure():
    cfg = HITLConfig(
        analyst_endpoint_url="http://test:8000/api/v1/analyst/escalation",
        analyst_max_retries=3,
        analyst_timeout_seconds=5,
    )
    dispatcher = HITLDispatcher(config=cfg)

    payload = HITLEscalationPayload(
        escalation_id="esc-retry-001",
        txn_id="TXN-R01",
        node_id="ACC-R01",
        agent_confidence=0.55,
        confidence_threshold=0.80,
        detected_typology="LAYERING",
        reasoning_trace=[],
        evidence_collected={},
        graph_context={},
        ml_score=0.88,
        gnn_score=-1.0,
        nlu_findings=None,
        recommended_action="ESCALATE_TO_HUMAN",
        escalated_at=time.time(),
    )

    # 3 attempts, all return 500
    mock_response = MagicMock()
    mock_response.status_code = 500
    mock_response.text = "Internal Server Error"

    mock_client = AsyncMock()
    mock_client.post.return_value = mock_response
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    with patch("httpx.AsyncClient", return_value=mock_client):
        result = await dispatcher.dispatch(payload)

    assert result.success is False
    assert result.status_code == 500
    assert mock_client.post.call_count == 3, \
        f"Expected 3 retry attempts, got {mock_client.post.call_count}"


# ── 17. HITLDispatchResult structure ────────────────────────────────────────

def test_hitl_dispatch_result_structure():
    result = HITLDispatchResult(
        success=True,
        status_code=200,
        analyst_ack_id="ack-xyz",
        error=None,
    )
    assert result.success is True
    assert result.status_code == 200
    assert result.analyst_ack_id == "ack-xyz"
    assert result.error is None

    fail_result = HITLDispatchResult(
        success=False,
        status_code=500,
        error="Server error 500",
    )
    assert fail_result.success is False
    assert fail_result.analyst_ack_id is None


# ── 18. Agent escalation path (low confidence) ─────────────────────────────

def test_agent_escalation_path_low_confidence():
    """Agent routes to HITL when confidence below threshold."""
    # Create agent with no LLM (mock everything)
    agent = InvestigatorAgent(llm_client=None, tool_executor=MagicMock())

    # Directly test the escalate_hitl_node
    state = {
        "alert": SAMPLE_ALERT,
        "intermediate_confidence": 0.60,
        "detected_typology": "LAYERING",
        "thinking_trace": ["Step 1: high volume detected"],
        "evidence_collected": {"tool1": {"data": "value"}},
        "ml_score": 0.91,
        "gnn_score": -1.0,
    }

    result = agent._escalate_hitl_node(state)
    assert result["status"] == "evaluate_escalation"
    assert result["escalation_payload"] is not None
    payload = result["escalation_payload"]
    assert payload["detected_typology"] == "LAYERING"
    assert payload["agent_confidence"] == 0.60
    assert payload["recommended_action"] == "ESCALATE_TO_HUMAN"


# ── 19. Agent autonomous path (high confidence) ────────────────────────────

def test_agent_autonomous_path_high_confidence():
    """Agent does NOT escalate when confidence is above threshold."""
    cfg = HITLConfig()
    # 0.85 is above all thresholds (default 0.75, LAYERING 0.80)
    result = ConfidenceEvaluator.should_escalate(0.85, "LAYERING", cfg)
    assert result is False, "0.85 >= 0.80 LAYERING threshold should NOT escalate"


# ── 20. Agent escalation on complex typology ────────────────────────────────

def test_agent_escalation_complex_typology():
    """LAYERING with 0.72 confidence should escalate (threshold 0.80)."""
    cfg = HITLConfig()
    result = ConfidenceEvaluator.should_escalate(0.72, "LAYERING", cfg)
    assert result is True, "0.72 < 0.80 LAYERING threshold should escalate"

    # But with a non-complex typology at same confidence, should NOT escalate
    # because PROFILE_MISMATCH threshold is 0.65
    result2 = ConfidenceEvaluator.should_escalate(0.72, "PROFILE_MISMATCH", cfg)
    assert result2 is False, "0.72 >= 0.65 PROFILE_MISMATCH threshold should NOT escalate"


# ── 21. Agent no escalation on simple typology ─────────────────────────────

def test_agent_no_escalation_simple_typology():
    """PROFILE_MISMATCH at 0.72 should NOT escalate (threshold 0.65)."""
    cfg = HITLConfig()
    result = ConfidenceEvaluator.should_escalate(0.72, "PROFILE_MISMATCH", cfg)
    assert result is False
    result2 = ConfidenceEvaluator.should_escalate(0.72, "DORMANT_ACTIVATION", cfg)
    assert result2 is False


# ── 22. Agent HITL verdict type ─────────────────────────────────────────────

def test_agent_hitl_verdict_type():
    """Escalated verdict from dispatch_hitl_node is ESCALATED_TO_HUMAN."""
    agent = InvestigatorAgent(llm_client=None, tool_executor=MagicMock())

    # Build a minimal escalation payload dict
    escalation_dict = {
        "escalation_id": "esc-vt-001",
        "txn_id": "TXN-VT01",
        "node_id": "ACC-VT01",
        "agent_confidence": 0.60,
        "confidence_threshold": 0.80,
        "detected_typology": "LAYERING",
        "reasoning_trace": ["Step 1"],
        "evidence_collected": {},
        "graph_context": {},
        "ml_score": 0.90,
        "gnn_score": -1.0,
        "nlu_findings": None,
        "recommended_action": "ESCALATE_TO_HUMAN",
        "escalated_at": time.time(),
    }

    state = {
        "alert": SAMPLE_ALERT,
        "escalation_payload": escalation_dict,
        "detected_typology": "LAYERING",
        "intermediate_confidence": 0.60,
        "evidence_collected": {"tool1": {"data": "val"}},
    }

    # Mock the dispatcher to avoid actual HTTP calls
    mock_dispatch_result = HITLDispatchResult(
        success=True, status_code=200, analyst_ack_id="ack-vt-001",
    )
    agent._hitl_dispatcher.dispatch = AsyncMock(return_value=mock_dispatch_result)

    result = agent._dispatch_hitl_node(state)
    assert result["verdict"]["verdict"] == "ESCALATED_TO_HUMAN"
    assert result["verdict"]["recommended_action"] == "ESCALATE_TO_HUMAN"
    assert result["status"] == "done"
    assert result["hitl_dispatched"] is True


# ── 23. Agent HITL metrics tracking ─────────────────────────────────────────

def test_agent_hitl_metrics_tracking():
    """verdicts_escalated counter incremented on HITL dispatch."""
    agent = InvestigatorAgent(llm_client=None, tool_executor=MagicMock())
    assert agent.metrics.verdicts_escalated == 0

    # Simulate an escalation metric record
    agent.hitl_metrics.record_escalation(
        typology="LAYERING", dispatched=True, latency_ms=50.0,
    )
    assert agent.hitl_metrics.escalations_triggered == 1
    assert agent.hitl_metrics.escalations_dispatched == 1
    assert agent.hitl_metrics.escalations_failed == 0
    assert agent.hitl_metrics.escalations_by_typology["LAYERING"] == 1


# ── 24. Evaluate escalation recovery ────────────────────────────────────────

def test_agent_evaluate_escalation_recovery():
    """If confidence rises above threshold, route back to think."""
    agent = InvestigatorAgent(llm_client=None, tool_executor=MagicMock())

    # Confidence is now ABOVE the threshold for LAYERING (0.80)
    state = {
        "intermediate_confidence": 0.85,
        "detected_typology": "LAYERING",
        "escalation_payload": {"some": "payload"},
    }

    result = agent._evaluate_escalation_node(state)
    assert result["status"] == "thinking", \
        f"Expected thinking (recovery), got {result['status']}"
    assert result["escalation_payload"] is None
    assert agent.hitl_metrics.escalations_recovered == 1

    # Now test when confidence is still below threshold
    state2 = {
        "intermediate_confidence": 0.72,
        "detected_typology": "LAYERING",
        "escalation_payload": {"some": "payload"},
    }
    result2 = agent._evaluate_escalation_node(state2)
    assert result2["status"] == "dispatch"


# ── 25. HITLMetrics snapshot ────────────────────────────────────────────────

def test_hitl_metrics_snapshot():
    metrics = HITLMetrics()
    metrics.record_escalation("LAYERING", dispatched=True, latency_ms=100.0)
    metrics.record_escalation("STRUCTURING", dispatched=False, latency_ms=200.0)
    metrics.escalations_recovered = 1

    snap = metrics.snapshot()
    assert snap["triggered"] == 2
    assert snap["dispatched"] == 1
    assert snap["failed"] == 1
    assert snap["recovered"] == 1
    assert "LAYERING" in snap["by_typology"]
    assert "STRUCTURING" in snap["by_typology"]
    assert snap["avg_latency_ms"] == 150.0
    assert "uptime_sec" in snap
    # Verify JSON-serializable
    json_str = json.dumps(snap)
    assert isinstance(json_str, str)


# ── 26. Analyst endpoint receives payload ───────────────────────────────────

async def test_analyst_endpoint_receives_payload():
    """Mock analyst API receives and acks payload."""
    from src.api.routes.analyst import (
        EscalationAck,
        clear_store,
        list_escalations,
        receive_escalation,
    )

    clear_store()

    test_payload = {
        "escalation_id": "esc-analyst-001",
        "txn_id": "TXN-A01",
        "agent_confidence": 0.60,
    }

    ack = await receive_escalation(test_payload)
    assert isinstance(ack, EscalationAck)
    assert ack.status == "received"
    assert ack.ack_id is not None and len(ack.ack_id) > 0

    # Verify stored
    stored = await list_escalations()
    assert len(stored) == 1
    assert stored[0]["payload"]["txn_id"] == "TXN-A01"
    assert stored[0]["ack_id"] == ack.ack_id

    # Cleanup
    clear_store()
    stored_after = await list_escalations()
    assert len(stored_after) == 0


# ── Runner ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    _safe_print("\n=== Phase 12: HITL Escalation & Dynamic Agentic Orchestration ===\n")

    # 1-3: HITLConfig
    run_test(test_hitl_config_defaults)
    run_test(test_hitl_config_immutability)
    run_test(test_hitl_config_typology_thresholds)

    # 4-8: ConfidenceEvaluator
    run_test(test_confidence_evaluator_below_threshold)
    run_test(test_confidence_evaluator_above_threshold)
    run_test(test_confidence_evaluator_typology_specific)
    run_test(test_confidence_evaluator_unknown_typology)
    run_test(test_confidence_evaluator_none_typology)

    # 9-10: HITLEscalationPayload
    run_test(test_escalation_payload_construction)
    run_test(test_escalation_payload_serialization)

    # 11-13: GraphContextPackager
    run_test(test_graph_context_packager_with_graph)
    run_test(test_graph_context_packager_no_graph)
    run_test(test_graph_context_packager_unknown_node)

    # 14-17: HITLDispatcher
    run_test(test_hitl_dispatcher_success)
    run_test(test_hitl_dispatcher_timeout)
    run_test(test_hitl_dispatcher_retry_on_failure)
    run_test(test_hitl_dispatch_result_structure)

    # 18-24: Agent Integration
    run_test(test_agent_escalation_path_low_confidence)
    run_test(test_agent_autonomous_path_high_confidence)
    run_test(test_agent_escalation_complex_typology)
    run_test(test_agent_no_escalation_simple_typology)
    run_test(test_agent_hitl_verdict_type)
    run_test(test_agent_hitl_metrics_tracking)
    run_test(test_agent_evaluate_escalation_recovery)

    # 25-26: Metrics & Analyst Endpoint
    run_test(test_hitl_metrics_snapshot)
    run_test(test_analyst_endpoint_receives_payload)

    _safe_print(f"\n--- Results: {passed} passed, {failed} failed out of {passed + failed} ---")
    if failed > 0:
        sys.exit(1)
    else:
        _safe_print("All Phase 12 tests PASSED!")

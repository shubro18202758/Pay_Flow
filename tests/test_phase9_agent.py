"""
PayFlow -- Phase 9 Tests: LangGraph Investigator Agent
========================================================
Verifies tool schemas, tool executor dispatch, agent state management,
verdict extraction, prompt building, LangGraph routing, and ledger anchoring.
"""

from __future__ import annotations

import asyncio
import json
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import InvestigatorAgentConfig, LedgerConfig
from src.blockchain.ledger import AuditLedger
from src.blockchain.models import EventType
from src.llm.agent import AgentMetrics, AgentState, InvestigatorAgent, VerdictPayload
from src.llm.prompts import (
    COT_ACTIVATION_PREFIX,
    INVESTIGATOR_SYSTEM_PROMPT,
    VERDICT_SCHEMA,
    build_cot_prompt,
    build_investigation_prompt,
    build_verdict_prompt,
)
from src.llm.tools import TOOL_SCHEMAS, ToolCall, ToolExecutor, ToolResult


# ── Helpers ──────────────────────────────────────────────────────────────────

def _safe_print(msg: str) -> None:
    """Print with ASCII fallback for Windows cp1252 consoles."""
    try:
        print(msg)
    except UnicodeEncodeError:
        print(msg.encode("ascii", "replace").decode())


def make_ledger_config(tmp_dir: Path, **overrides) -> LedgerConfig:
    defaults = {
        "db_path": tmp_dir / "test_ledger.db",
        "key_dir": tmp_dir / "keys",
        "checkpoint_interval": 100,
        "enable_signing": True,
    }
    defaults.update(overrides)
    return LedgerConfig(**defaults)


SAMPLE_ALERT = {
    "txn_id": "TXN_AGENT_001",
    "sender_id": "ACC_S001",
    "receiver_id": "ACC_R001",
    "timestamp": int(time.time()),
    "risk_score": 0.9321,
    "tier": "HIGH",
    "threshold": 0.85,
    "top_features": {
        "velocity_1h_count": 14.0,
        "amt_zscore": 3.21,
        "geo_distance_km": 842.5,
    },
}


SAMPLE_VERDICT_JSON = json.dumps({
    "verdict": "FRAUDULENT",
    "confidence": 0.92,
    "fraud_typology": "layering",
    "reasoning_summary": "High velocity transfers through intermediary accounts with rapid chain traversal.",
    "evidence_cited": [
        "velocity_1h_count: 14.0",
        "mule_network_detected: true",
        "cycle_length: 3",
    ],
    "recommended_action": "FREEZE",
})


# ── Test 1: Tool Schemas Validation ─────────────────────────────────────────

async def test_tool_schemas_valid():
    """Validate all 5 TOOL_SCHEMAS are well-formed OpenAI-compatible."""
    assert len(TOOL_SCHEMAS) == 5, f"Expected 5 schemas, got {len(TOOL_SCHEMAS)}"

    expected_names = {
        "query_graph_database",
        "get_ml_feature_analysis",
        "read_audit_logs",
        "check_node_freeze_status",
        "analyze_unstructured_data",
    }
    actual_names = set()

    for schema in TOOL_SCHEMAS:
        # Top-level structure
        assert schema["type"] == "function", f"Schema type must be 'function'"
        assert "function" in schema, "Missing 'function' key"

        fn = schema["function"]
        assert "name" in fn, "Function must have 'name'"
        assert "description" in fn, "Function must have 'description'"
        assert "parameters" in fn, "Function must have 'parameters'"

        # Parameters structure
        params = fn["parameters"]
        assert params["type"] == "object", "Parameters type must be 'object'"
        assert "properties" in params, "Parameters must have 'properties'"
        assert "required" in params, "Parameters must have 'required'"

        actual_names.add(fn["name"])

    assert actual_names == expected_names, (
        f"Schema names mismatch: {actual_names} != {expected_names}"
    )
    _safe_print("  [PASS] test_tool_schemas_valid")


# ── Test 2: ToolExecutor — Graph Query ──────────────────────────────────────

async def test_tool_executor_graph_query():
    """ToolExecutor.execute() with query_graph_database returns subgraph stats."""
    # Mock TransactionGraph
    mock_graph = MagicMock()
    mock_nx = MagicMock()
    mock_graph.graph = mock_nx

    mock_nx.has_node.return_value = True
    mock_nx.nodes.get.return_value = {
        "first_seen": 1000, "last_seen": 2000, "txn_count": 15,
    }
    mock_nx.in_degree.return_value = 5
    mock_nx.out_degree.return_value = 3
    mock_nx.in_edges.return_value = [("A", "B"), ("C", "B"), ("D", "B")]
    mock_nx.out_edges.return_value = [("B", "E"), ("B", "F")]

    # Mock subgraph extraction
    mock_subgraph = MagicMock()
    mock_subgraph.number_of_nodes.return_value = 8
    mock_subgraph.number_of_edges.return_value = 12
    mock_graph._extract_k_hop_subgraph.return_value = mock_subgraph
    mock_graph._latest_timestamp = time.time()

    # Mock mule detector
    mock_graph._mule_detector = MagicMock()
    mock_graph._mule_detector.detect_around_node.return_value = None

    # Mock cycle detector
    mock_graph._cycle_detector = MagicMock()
    mock_graph._cycle_detector.detect_around_nodes.return_value = []

    executor = ToolExecutor(transaction_graph=mock_graph)
    tc = ToolCall(name="query_graph_database", arguments={"node_id": "ACC001", "k_hops": 2})
    result = await executor.execute(tc)

    assert result.success, f"Expected success, got error: {result.error}"
    assert result.tool_name == "query_graph_database"
    assert result.data["node_id"] == "ACC001"
    assert result.data["found"] is True
    assert result.data["subgraph"]["nodes"] == 8
    assert result.data["subgraph"]["edges"] == 12
    assert "_execution_ms" in result.data

    _safe_print("  [PASS] test_tool_executor_graph_query")


# ── Test 3: ToolExecutor — ML Features ──────────────────────────────────────

async def test_tool_executor_ml_features():
    """ToolExecutor.execute() with get_ml_feature_analysis returns ranked features."""
    import numpy as np

    mock_engine = MagicMock()
    features = np.array([0.1, 0.9, 0.3, 0.7, 0.5], dtype=np.float32)
    names = ["feat_a", "feat_b", "feat_c", "feat_d", "feat_e"]
    mock_engine._feature_cache = {"TXN001": (features, names)}

    executor = ToolExecutor(feature_engine=mock_engine)
    tc = ToolCall(name="get_ml_feature_analysis", arguments={"txn_id": "TXN001", "top_k": 3})
    result = await executor.execute(tc)

    assert result.success, f"Expected success, got error: {result.error}"
    assert result.data["txn_id"] == "TXN001"
    assert result.data["feature_dim"] == 5
    assert len(result.data["top_features"]) == 3
    # Top feature by absolute value should be feat_b (0.9)
    assert "feat_b" in result.data["top_features"]

    _safe_print("  [PASS] test_tool_executor_ml_features")


# ── Test 4: ToolExecutor — Audit Logs ───────────────────────────────────────

async def test_tool_executor_audit_logs():
    """ToolExecutor.execute() with read_audit_logs filters by event type."""
    mock_ledger = AsyncMock()
    mock_ledger.get_recent_blocks = AsyncMock(return_value=[
        {
            "index": 5,
            "timestamp": time.time(),
            "event_type": "ALERT",
            "payload": {"txn_id": "TXN001", "sender_id": "ACC_S001"},
            "prev_hash": "abc",
            "block_hash": "def",
            "has_signature": True,
            "merkle_root": None,
        },
    ])

    executor = ToolExecutor(audit_ledger=mock_ledger)
    tc = ToolCall(
        name="read_audit_logs",
        arguments={"event_type": "ALERT", "node_id": "ACC_S001", "limit": 10},
    )
    result = await executor.execute(tc)

    assert result.success, f"Expected success, got error: {result.error}"
    assert result.data["entries_count"] >= 1
    assert result.data["filter"]["event_type"] == "ALERT"
    assert result.data["filter"]["node_id"] == "ACC_S001"

    _safe_print("  [PASS] test_tool_executor_audit_logs")


# ── Test 5: ToolExecutor — Freeze Status ────────────────────────────────────

async def test_tool_executor_freeze_status():
    """ToolExecutor.execute() with check_node_freeze_status returns details."""
    mock_breaker = MagicMock()
    mock_breaker.is_frozen.return_value = True

    freeze_order = MagicMock()
    freeze_order.trigger_txn_id = "TXN_FREEZE_001"
    freeze_order.ml_risk_score = 0.95
    freeze_order.gnn_risk_score = 0.88
    freeze_order.consensus_score = 0.915
    freeze_order.reason = "Multi-model consensus freeze"
    freeze_order.freeze_timestamp = time.time() - 100
    freeze_order.ttl_seconds = 3600

    mock_breaker._frozen_nodes = {"ACC_FROZEN": freeze_order}

    executor = ToolExecutor(circuit_breaker=mock_breaker)
    tc = ToolCall(name="check_node_freeze_status", arguments={"node_id": "ACC_FROZEN"})
    result = await executor.execute(tc)

    assert result.success, f"Expected success, got error: {result.error}"
    assert result.data["is_frozen"] is True
    assert "freeze_details" in result.data
    details = result.data["freeze_details"]
    assert details["trigger_txn_id"] == "TXN_FREEZE_001"
    assert details["ttl_remaining_seconds"] > 3400  # ~3500 remaining

    _safe_print("  [PASS] test_tool_executor_freeze_status")


# ── Test 6: ToolExecutor — Unknown Tool ─────────────────────────────────────

async def test_tool_executor_unknown_tool():
    """Unknown tool name returns error ToolResult with success=False."""
    executor = ToolExecutor()
    tc = ToolCall(name="nonexistent_tool", arguments={"key": "value"})
    result = await executor.execute(tc)

    assert not result.success, "Expected failure for unknown tool"
    assert result.error is not None
    assert "Unknown tool" in result.error
    assert result.data == {}

    _safe_print("  [PASS] test_tool_executor_unknown_tool")


# ── Test 7: AgentState Initialization ───────────────────────────────────────

async def test_agent_state_initialization():
    """AgentState created with correct default fields for TypedDict."""
    state: AgentState = {
        "alert": SAMPLE_ALERT,
        "ml_score": 0.93,
        "gnn_score": -1.0,
        "messages": [{"role": "system", "content": "test"}],
        "thinking_trace": [],
        "tool_calls_made": [],
        "evidence_collected": {},
        "iteration": 0,
        "max_iterations": 5,
        "verdict": None,
        "status": "thinking",
    }

    assert state["alert"]["txn_id"] == "TXN_AGENT_001"
    assert state["ml_score"] == 0.93
    assert state["gnn_score"] == -1.0
    assert state["iteration"] == 0
    assert state["status"] == "thinking"
    assert state["verdict"] is None
    assert len(state["thinking_trace"]) == 0
    assert len(state["tool_calls_made"]) == 0

    _safe_print("  [PASS] test_agent_state_initialization")


# ── Test 8: VerdictPayload Structure ────────────────────────────────────────

async def test_verdict_payload_structure():
    """VerdictPayload fields validated, serialization via to_dict() works."""
    vp = VerdictPayload(
        txn_id="TXN_VP_001",
        node_id="ACC_VP_S",
        verdict="FRAUDULENT",
        confidence=0.9234,
        fraud_typology="layering",
        reasoning_summary="Evidence of rapid multi-hop transfers.",
        evidence_cited=["velocity_1h_count: 14.0", "mule_detected: true"],
        recommended_action="FREEZE",
        thinking_steps=3,
        tools_used=["query_graph_database", "get_ml_feature_analysis"],
        total_duration_ms=1234.56,
    )

    assert vp.txn_id == "TXN_VP_001"
    assert vp.verdict == "FRAUDULENT"
    assert vp.confidence == 0.9234
    assert vp.fraud_typology == "layering"
    assert len(vp.evidence_cited) == 2
    assert vp.recommended_action == "FREEZE"

    # Test serialization
    d = vp.to_dict()
    assert d["txn_id"] == "TXN_VP_001"
    assert d["confidence"] == 0.9234  # rounded to 4 decimals
    assert d["total_duration_ms"] == 1234.56
    assert d["thinking_steps"] == 3
    assert len(d["tools_used"]) == 2

    # Test immutability (frozen dataclass)
    try:
        vp.verdict = "LEGITIMATE"
        assert False, "Should not be able to modify frozen dataclass"
    except AttributeError:
        pass  # expected

    _safe_print("  [PASS] test_verdict_payload_structure")


# ── Test 9: CoT Prompt Building ─────────────────────────────────────────────

async def test_cot_prompt_building():
    """Prompt builders format alerts and thinking traces correctly."""
    # Test build_investigation_prompt
    prompt = build_investigation_prompt(SAMPLE_ALERT, context={"gnn_score": 0.78})
    assert "TXN_AGENT_001" in prompt
    assert "ACC_S001" in prompt
    assert "ACC_R001" in prompt
    assert "0.9321" in prompt
    assert "velocity_1h_count" in prompt
    assert "gnn_score" in prompt

    # Test build_cot_prompt
    cot = build_cot_prompt(
        thinking_so_far="Step 1: High velocity transfers detected.",
        new_evidence='{"mule_detected": true}',
    )
    assert "/think" in cot
    assert "EVIDENCE UPDATE" in cot
    assert "mule_detected" in cot
    assert "High velocity transfers" in cot

    # Test build_verdict_prompt
    verdict_prompt = build_verdict_prompt("Step 1: Analysis. Step 2: Conclusion.")
    assert "FINAL VERDICT" in verdict_prompt
    assert "Step 1: Analysis" in verdict_prompt
    assert '"verdict"' in verdict_prompt  # schema embedded

    _safe_print("  [PASS] test_cot_prompt_building")


# ── Test 10: Agent Think Node (Mocked LLM) ─────────────────────────────────

async def test_agent_think_node_mock():
    """_think_node updates state with CoT trace and parses tool calls."""
    cfg = InvestigatorAgentConfig(max_iterations=5)
    agent = InvestigatorAgent(llm_client=None, tool_executor=ToolExecutor(), config=cfg)

    # Mock the _call_llm to return a response with tool calls
    agent._call_llm = MagicMock(return_value={
        "content": "I need to check the graph structure for this account.",
        "tool_calls": [
            {"name": "query_graph_database", "arguments": {"node_id": "ACC_S001", "k_hops": 3}},
        ],
    })

    state: AgentState = {
        "alert": SAMPLE_ALERT,
        "ml_score": 0.93,
        "gnn_score": -1.0,
        "messages": [{"role": "system", "content": INVESTIGATOR_SYSTEM_PROMPT}],
        "thinking_trace": [],
        "tool_calls_made": [],
        "evidence_collected": {},
        "iteration": 0,
        "max_iterations": 5,
        "verdict": None,
        "status": "thinking",
    }

    result = agent._think_node(state)

    assert result["iteration"] == 1, f"Iteration should increment to 1, got {result['iteration']}"
    assert result["status"] == "calling_tools", f"Status should be 'calling_tools', got {result['status']}"
    assert len(result["thinking_trace"]) == 1
    assert "graph structure" in result["thinking_trace"][0]
    assert len(result["tool_calls_made"]) == 1
    assert result["tool_calls_made"][0]["name"] == "query_graph_database"

    _safe_print("  [PASS] test_agent_think_node_mock")


# ── Test 11: Agent Routing Logic ────────────────────────────────────────────

async def test_agent_tool_routing_mock():
    """_should_continue routes correctly based on state.status values."""
    from langgraph.graph import END

    cfg = InvestigatorAgentConfig()
    agent = InvestigatorAgent(llm_client=None, tool_executor=ToolExecutor(), config=cfg)

    # calling_tools -> execute_tools
    assert agent._should_continue({"status": "calling_tools"}) == "execute_tools"

    # verdict -> verdict
    assert agent._should_continue({"status": "verdict"}) == "verdict"

    # done -> END
    assert agent._should_continue({"status": "done"}) == END

    # thinking -> think (default)
    assert agent._should_continue({"status": "thinking"}) == "think"

    # unknown -> think (fallback)
    assert agent._should_continue({"status": "unknown_status"}) == "think"

    # empty -> think (default when missing)
    assert agent._should_continue({}) == "think"

    _safe_print("  [PASS] test_agent_tool_routing_mock")


# ── Test 12: Agent Full Loop (Mocked LLM) ──────────────────────────────────

async def test_agent_full_loop_mock():
    """Full investigate() with mocked LLM runs think-act-observe loop."""
    cfg = InvestigatorAgentConfig(max_iterations=3)
    agent = InvestigatorAgent(llm_client=None, tool_executor=ToolExecutor(), config=cfg)

    call_count = 0

    def mock_call_llm(messages, tools=None):
        nonlocal call_count
        call_count += 1

        if call_count == 1:
            # First think: request a tool call
            return {
                "content": "Let me investigate this transaction.",
                "tool_calls": [
                    {"name": "query_graph_database", "arguments": {"node_id": "ACC_S001"}},
                ],
            }
        elif call_count == 2:
            # Second think (after tools return): provide verdict signal
            return {
                "content": f'Based on evidence:\n```json\n{SAMPLE_VERDICT_JSON}\n```',
                "tool_calls": [],
            }
        else:
            # Verdict extraction fallback
            return {
                "content": SAMPLE_VERDICT_JSON,
                "tool_calls": [],
            }

    agent._call_llm = mock_call_llm

    @dataclass
    class MockAlert:
        txn_id: str = "TXN_LOOP_001"
        sender_id: str = "ACC_S001"
        receiver_id: str = "ACC_R001"
        risk_score: float = 0.93

        def to_dict(self):
            return {
                "txn_id": self.txn_id,
                "sender_id": self.sender_id,
                "receiver_id": self.receiver_id,
                "risk_score": self.risk_score,
                "tier": "HIGH",
                "threshold": 0.85,
                "top_features": {},
            }

    verdict = await agent.investigate(MockAlert(), ml_score=0.93, gnn_score=0.78)

    assert isinstance(verdict, VerdictPayload)
    assert verdict.txn_id == "TXN_LOOP_001"
    assert verdict.verdict in ("FRAUDULENT", "SUSPICIOUS", "LEGITIMATE")
    assert 0.0 <= verdict.confidence <= 1.0
    assert verdict.recommended_action in ("FREEZE", "ESCALATE", "MONITOR", "CLEAR")
    assert verdict.thinking_steps >= 1
    assert verdict.total_duration_ms > 0

    # Metrics updated
    assert agent.metrics.investigations_started == 1
    assert agent.metrics.investigations_completed == 1

    _safe_print("  [PASS] test_agent_full_loop_mock")


# ── Test 13: Max Iterations Guard ───────────────────────────────────────────

async def test_agent_max_iterations_guard():
    """Agent stops at max_iterations and forces a verdict."""
    cfg = InvestigatorAgentConfig(max_iterations=2)
    agent = InvestigatorAgent(llm_client=None, tool_executor=ToolExecutor(), config=cfg)

    def mock_call_llm(messages, tools=None):
        # Always return thinking content with no tool calls and no verdict signal
        return {
            "content": "I need to continue thinking about this case.",
            "tool_calls": [],
        }

    agent._call_llm = mock_call_llm

    verdict = await agent.investigate(SAMPLE_ALERT, ml_score=0.93)

    assert isinstance(verdict, VerdictPayload)
    # Should default to SUSPICIOUS when max iterations hit without clear verdict
    assert verdict.verdict in ("FRAUDULENT", "SUSPICIOUS", "LEGITIMATE")
    assert verdict.recommended_action in ("FREEZE", "ESCALATE", "MONITOR", "CLEAR")
    # Should have iterated at most max_iterations times
    assert verdict.thinking_steps <= cfg.max_iterations + 1  # +1 for possible verdict extraction step

    _safe_print("  [PASS] test_agent_max_iterations_guard")


# ── Test 14: Verdict Ledger Anchoring ───────────────────────────────────────

async def test_agent_verdict_ledger_anchoring():
    """Verdict anchored to AuditLedger via anchor_agent_verdict()."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        lcfg = make_ledger_config(tmp_path)
        ledger = AuditLedger(config=lcfg)
        await ledger.open()

        try:
            initial_head = ledger.head_index

            # Create a VerdictPayload and anchor it
            vp = VerdictPayload(
                txn_id="TXN_ANCHOR_001",
                node_id="ACC_ANCHOR_S",
                verdict="FRAUDULENT",
                confidence=0.95,
                fraud_typology="mule_network",
                reasoning_summary="Star topology detected with 7 distinct senders.",
                evidence_cited=["in_degree: 7", "mule_network_detected: true"],
                recommended_action="FREEZE",
                thinking_steps=3,
                tools_used=["query_graph_database"],
                total_duration_ms=987.65,
            )

            block = await ledger.anchor_agent_verdict(vp)

            # Verify block was created
            assert block.index == initial_head + 1
            assert block.event_type == EventType.AGENT_VERDICT
            assert block.payload["txn_id"] == "TXN_ANCHOR_001"
            assert block.payload["verdict"] == "FRAUDULENT"
            assert block.payload["_source"] == "investigator_agent"
            assert block.payload["fraud_typology"] == "mule_network"

            # Verify ledger metrics updated
            assert ledger.metrics.agent_verdicts_anchored == 1

            # Verify chain integrity after anchoring
            result = await ledger.verify_chain()
            assert result.valid, f"Chain verification failed: {result.error_message}"

        finally:
            await ledger.close()

    _safe_print("  [PASS] test_agent_verdict_ledger_anchoring")


# ── Runner ──────────────────────────────────────────────────────────────────

async def main():
    _safe_print("\n" + "=" * 70)
    _safe_print("  PayFlow Phase 9 Tests: LangGraph Investigator Agent")
    _safe_print("=" * 70 + "\n")

    tests = [
        test_tool_schemas_valid,
        test_tool_executor_graph_query,
        test_tool_executor_ml_features,
        test_tool_executor_audit_logs,
        test_tool_executor_freeze_status,
        test_tool_executor_unknown_tool,
        test_agent_state_initialization,
        test_verdict_payload_structure,
        test_cot_prompt_building,
        test_agent_think_node_mock,
        test_agent_tool_routing_mock,
        test_agent_full_loop_mock,
        test_agent_max_iterations_guard,
        test_agent_verdict_ledger_anchoring,
    ]

    passed = 0
    failed = 0

    for test_fn in tests:
        try:
            await test_fn()
            passed += 1
        except Exception as exc:
            _safe_print(f"  [FAIL] {test_fn.__name__}: {exc}")
            import traceback
            traceback.print_exc()
            failed += 1

    _safe_print(f"\n{'=' * 70}")
    _safe_print(f"  Results: {passed} passed, {failed} failed, {len(tests)} total")
    _safe_print(f"{'=' * 70}\n")

    if failed:
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())

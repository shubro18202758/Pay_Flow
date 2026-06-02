"""
PayFlow -- LangGraph Investigator Agent
=========================================
Autonomous fraud investigation agent powered by Qwen 3.5 4B via Ollama,
orchestrated through a LangGraph state-machine graph.

The agent implements an **analyze-act-observe** loop:

1. **ANALYZE**: Build a concise, evidence-grounded rationale from current evidence.
2. **ACT** (Tool Calls): Invoke PayFlow subsystem tools to gather evidence.
3. **OBSERVE** (Evidence Integration): Incorporate tool results into the
   investigation trace.
4. **DECIDE** (Verdict): Issue a structured fraud verdict when confident
   or when max iterations are exhausted.

State Machine::

    ┌──────────┐
    │  START   │
    └────┬─────┘
         │
         ▼
    ┌──────────┐     "tools"      ┌──────────────┐
    │ ANALYZE  │ ──────────────> │ EXECUTE_TOOLS │
    │ TRACE    │ <────────────── │              │
    └────┬─────┘    "think"      └──────────────┘
         │
         │ "verdict"
         ▼
    ┌──────────┐
    │ VERDICT  │
    └────┬─────┘
         │
         ▼
    ┌──────────┐
    │   END    │
    └──────────┘

Integration::

    agent = InvestigatorAgent(llm_client=llm, tool_executor=executor)
    router.register_agent_consumer(agent.on_alert)

Dependencies:
    - langgraph (LangGraph state machine)
    - ollama (Qwen 3.5 4B inference via PayFlowLLM)
    - src.llm.tools (ToolExecutor, ToolCall, ToolResult)
    - src.llm.prompts (system prompts, evidence-rationale templates)
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Literal, TypedDict

from langgraph.graph import END, StateGraph

from src.llm.prompts import (
    INVESTIGATOR_SYSTEM_PROMPT,
    VERDICT_SCHEMA,
    build_cot_prompt,
    build_investigation_prompt,
    build_verdict_prompt,
)
from src.llm.tools import TOOL_SCHEMAS, ToolCall, ToolExecutor, ToolResult
from src.llm.unstructured_prompts import build_consensus_injection_prompt

logger = logging.getLogger(__name__)


# ── Agent State ──────────────────────────────────────────────────────────────

class AgentState(TypedDict, total=False):
    """
    LangGraph state flowing through the investigation graph.

    Uses TypedDict for LangGraph compatibility (requires dict-like state).
    """
    alert: dict                          # Original AlertPayload.to_dict()
    ml_score: float                      # Risk score from ML pipeline
    gnn_score: float                     # GNN topology score (-1 if unavailable)
    messages: list[dict]                 # Full conversation history
    thinking_trace: list[str]            # Evidence-rationale steps
    tool_calls_made: list[dict]          # History of tool invocations
    evidence_collected: dict             # Accumulated evidence from tools
    iteration: int                       # Current reasoning loop count
    max_iterations: int                  # Cap to prevent infinite loops
    verdict: dict | None                 # Final structured verdict
    status: str                          # "thinking" | "calling_tools" | "verdict" | "escalate" | "evaluate_escalation" | "dispatch" | "done"
    # ── HITL Escalation State ──
    escalation_payload: dict | None      # Packaged HITL payload
    hitl_dispatched: bool                # Whether escalation was sent
    intermediate_confidence: float       # Running confidence estimate
    detected_typology: str | None        # Typology detected during reasoning
    _pending_tool_calls: list[ToolCall]  # Tool calls to execute in the next node
    _t0: float                           # Investigation start time for events


# ── Verdict Payload ──────────────────────────────────────────────────────────

@dataclass(frozen=True)
class VerdictPayload:
    """
    Structured verdict from the Investigator Agent.

    Immutable record anchored to the audit ledger after every investigation.
    """
    txn_id: str
    node_id: str
    verdict: str                         # "FRAUDULENT" | "SUSPICIOUS" | "LEGITIMATE" | "ESCALATED_TO_HUMAN"
    confidence: float                    # 0.0 - 1.0
    fraud_typology: str | None           # e.g., "layering", "round_tripping"
    reasoning_summary: str               # Concise evidence-rationale summary
    evidence_cited: list[str]            # Specific evidence items cited
    recommended_action: str              # "FREEZE" | "ESCALATE" | "MONITOR" | "CLEAR" | "ESCALATE_TO_HUMAN"
    thinking_steps: int                  # Number of rationale iterations
    tools_used: list[str]               # Names of tools invoked
    total_duration_ms: float
    confidence_source: str = "qwen_json"
    llm_parse_status: str = "parsed"
    model_used: str | None = None

    def to_dict(self) -> dict:
        return {
            "txn_id": self.txn_id,
            "node_id": self.node_id,
            "verdict": self.verdict,
            "confidence": round(self.confidence, 4),
            "fraud_typology": self.fraud_typology,
            "reasoning_summary": self.reasoning_summary,
            "evidence_cited": self.evidence_cited,
            "recommended_action": self.recommended_action,
            "thinking_steps": self.thinking_steps,
            "tools_used": self.tools_used,
            "total_duration_ms": round(self.total_duration_ms, 2),
            "confidence_source": self.confidence_source,
            "llm_parse_status": self.llm_parse_status,
            "model_used": self.model_used,
        }


# ── Agent Metrics ────────────────────────────────────────────────────────────

@dataclass
class AgentMetrics:
    """Runtime performance counters for the investigator agent."""
    investigations_started: int = 0
    investigations_completed: int = 0
    verdicts_fraudulent: int = 0
    verdicts_suspicious: int = 0
    verdicts_legitimate: int = 0
    verdicts_escalated: int = 0
    agent_breaker_triggered: int = 0
    total_tool_calls: int = 0
    total_thinking_steps: int = 0
    total_investigation_ms: float = 0.0
    _start_time: float = field(default_factory=time.monotonic)

    def snapshot(self) -> dict:
        return {
            "started": self.investigations_started,
            "completed": self.investigations_completed,
            "verdicts": {
                "fraudulent": self.verdicts_fraudulent,
                "suspicious": self.verdicts_suspicious,
                "legitimate": self.verdicts_legitimate,
                "escalated": self.verdicts_escalated,
            },
            "agent_breaker_triggered": self.agent_breaker_triggered,
            "total_tool_calls": self.total_tool_calls,
            "total_thinking_steps": self.total_thinking_steps,
            "avg_investigation_ms": round(
                self.total_investigation_ms / max(self.investigations_completed, 1), 2,
            ),
            "uptime_sec": round(time.monotonic() - self._start_time, 1),
        }


# ── Investigator Agent ───────────────────────────────────────────────────────

class InvestigatorAgent:
    """
    LangGraph-powered fraud investigation agent using Qwen 3.5 4B.

    Implements an analyze-act-observe loop where the LLM evaluates
    fraud typologies through concise evidence rationales,
    calls tools to gather evidence from PayFlow subsystems, and issues
    a structured verdict.

    Usage::

        agent = InvestigatorAgent(
            llm_client=PayFlowLLM(),
            tool_executor=ToolExecutor(graph, engine, ledger, breaker),
        )
        verdict = await agent.investigate(alert_payload, ml_score=0.92)
    """

    def __init__(
        self,
        llm_client=None,
        tool_executor: ToolExecutor | None = None,
        audit_ledger=None,
        unstructured_agent=None,
        transaction_graph=None,
        hitl_config=None,
        agent_breaker_listener=None,
        config=None,
    ) -> None:
        from config.settings import HITL_CFG, INVESTIGATOR_CFG
        self._cfg = config or INVESTIGATOR_CFG
        self._hitl_cfg = hitl_config or HITL_CFG

        self._llm = llm_client
        self._tools = tool_executor or ToolExecutor()
        self._audit_ledger = audit_ledger
        self._unstructured_agent = unstructured_agent
        self._transaction_graph = transaction_graph
        self._agent_breaker_listener = agent_breaker_listener
        self._graph = self._build_graph()
        self.metrics = AgentMetrics()
        self._investigation_records: dict[str, dict] = {}  # txn_id → full record
        self._tool_evidence_cache: dict[str, dict] = {}

        # HITL components (lazy-initialized)
        from src.llm.hitl import (
            ConfidenceEvaluator,
            GraphContextPackager,
            HITLDispatcher,
            HITLMetrics,
        )
        self._confidence_evaluator = ConfidenceEvaluator()
        self._graph_packager = GraphContextPackager()
        self._hitl_dispatcher = HITLDispatcher(config=self._hitl_cfg)
        self.hitl_metrics = HITLMetrics()

    def _graph_context(self, gnn_score: float) -> dict:
        """Expose optional GNN state without letting -1 become risk evidence."""
        try:
            score = float(gnn_score)
        except (TypeError, ValueError):
            score = -1.0
        if score >= 0.0:
            return {
                "gnn_score": round(score, 4),
                "graph_model_status": "available",
            }
        return {
            "gnn_score": None,
            "graph_model_status": "unavailable",
            "graph_evidence_rule": (
                "Do not cite -1/null GNN as risk. Use graph tool pattern "
                "fields such as mule_network_detected, cycles_found, degrees, "
                "and subgraph size."
            ),
        }

    def _summarize_tool_result(self, result_or_name, payload: dict | None = None) -> dict:
        """
        Compact tool output before sending it back to a 4B model.

        The full evidence remains in state for audit. The prompt receives only
        fields that influence the verdict, preventing old ledger payloads from
        crowding out the transaction evidence.
        """
        if isinstance(result_or_name, ToolResult):
            tool_name = result_or_name.tool_name
            success = result_or_name.success
            data = result_or_name.data or {}
            error = result_or_name.error
        else:
            tool_name = str(result_or_name)
            raw = payload or {}
            success = bool(raw.get("success", True))
            data = raw.get("data", raw)
            error = raw.get("error")

        summary: dict[str, Any] = {
            "tool_name": tool_name,
            "success": success,
        }
        if error:
            summary["error"] = error

        if tool_name == "read_audit_logs":
            entries = data.get("entries", []) if isinstance(data, dict) else []
            summary["data"] = {
                "entries_count": data.get("entries_count", len(entries)) if isinstance(data, dict) else len(entries),
                "filter": data.get("filter", {}) if isinstance(data, dict) else {},
                "recent_entries": [
                    {
                        "index": entry.get("index"),
                        "timestamp": entry.get("timestamp"),
                        "event_type": entry.get("event_type"),
                        "block_hash": entry.get("block_hash"),
                        "payload_keys": sorted((entry.get("payload") or {}).keys())[:12],
                    }
                    for entry in entries[:5]
                    if isinstance(entry, dict)
                ],
            }
            return summary

        if tool_name == "query_graph_database" and isinstance(data, dict):
            summary["data"] = {
                "node_id": data.get("node_id"),
                "found": data.get("found"),
                "subgraph": data.get("subgraph", {}),
                "connections": data.get("connections", {}),
                "patterns": data.get("patterns", {}),
                "message": data.get("message"),
            }
            return summary

        if tool_name == "get_ml_feature_analysis" and isinstance(data, dict):
            top_features = data.get("top_features", {})
            summary["data"] = {
                "txn_id": data.get("txn_id"),
                "feature_dim": data.get("feature_dim"),
                "top_features": dict(list(top_features.items())[:12])
                if isinstance(top_features, dict)
                else top_features,
                "error": data.get("error"),
            }
            return summary

        if tool_name == "check_node_freeze_status" and isinstance(data, dict):
            summary["data"] = {
                "node_id": data.get("node_id"),
                "is_frozen": data.get("is_frozen"),
                "freeze_details": data.get("freeze_details"),
                "error": data.get("error"),
            }
            return summary

        summary["data"] = data
        return summary

    def _summarize_evidence_for_prompt(self, evidence: dict) -> dict:
        return {
            name: self._summarize_tool_result(name, payload)
            for name, payload in evidence.items()
            if isinstance(payload, dict)
        }

    def _public_trace_content(self, content: str) -> str:
        """
        Convert a raw model response into an auditable dashboard trace.

        Qwen responses may contain JSON verdict drafts, tool-call prose, or
        longer analysis. Public trace entries must stay evidence-facing: no
        hidden chain-of-thought and no speculative "AI said so" text.
        """
        text = str(content or "").strip()
        if not text:
            return ""

        payload = text
        if "```json" in payload:
            payload = payload.split("```json", 1)[1].split("```", 1)[0]
        elif payload.startswith("```") and "```" in payload[3:]:
            payload = payload.split("```", 1)[1].split("```", 1)[0]

        try:
            parsed = json.loads(payload.strip())
        except (TypeError, ValueError):
            parsed = None

        if isinstance(parsed, dict):
            parts: list[str] = []
            summary = parsed.get("reasoning_summary") or parsed.get("summary")
            if summary:
                parts.append(str(summary).strip())
            verdict = parsed.get("verdict")
            action = parsed.get("recommended_action")
            if verdict or action:
                parts.append(
                    "Draft verdict: "
                    + " / ".join(str(value).strip() for value in (verdict, action) if value)
                )
            evidence = parsed.get("evidence_cited")
            if isinstance(evidence, list) and evidence:
                parts.append("Evidence cited: " + ", ".join(str(item) for item in evidence[:4]))
            if parts:
                return " ".join(parts)[:500]

        public_lines: list[str] = []
        blocked_prefixes = (
            "thought:",
            "chain-of-thought",
            "hidden reasoning",
            "let me think",
            "i think",
        )
        for raw_line in text.replace("\r", "\n").split("\n"):
            line = raw_line.strip().strip("- ")
            if not line or line.startswith("```"):
                continue
            lowered = line.lower()
            if lowered.startswith(blocked_prefixes):
                continue
            public_lines.append(line)
            if sum(len(item) for item in public_lines) >= 500:
                break

        public = " ".join(public_lines).strip()
        return public[:500] if public else "Evidence-rationale update recorded."

    def _evidence_from_tool_result_messages(self, messages: list[dict]) -> dict:
        """
        Reconstruct compact evidence from TOOL RESULTS messages if LangGraph
        drops custom state fields before the final record is materialized.
        """
        evidence: dict[str, dict] = {}
        for msg in messages:
            content = msg.get("content", "") if isinstance(msg, dict) else ""
            if "## TOOL RESULTS" not in content:
                continue
            payload = content
            if "```json" in payload:
                payload = payload.split("```json", 1)[1].split("```", 1)[0]
            elif "```" in payload:
                payload = payload.split("```", 1)[1].split("```", 1)[0]
            try:
                parsed = json.loads(payload.strip())
            except (TypeError, ValueError):
                continue
            if isinstance(parsed, dict):
                parsed = [parsed]
            if not isinstance(parsed, list):
                continue
            for item in parsed:
                if not isinstance(item, dict):
                    continue
                name = item.get("tool_name")
                if name:
                    evidence[str(name)] = item
        return evidence

    async def _replay_tool_evidence(self, tool_calls: list[dict]) -> dict:
        """
        Last-resort evidence capture for already requested read-only tools.

        This should rarely run; it protects investigation records if graph state
        propagation loses the tool results after the LLM has requested tools.
        """
        replayable = {
            "query_graph_database",
            "get_ml_feature_analysis",
            "read_audit_logs",
            "check_node_freeze_status",
        }
        evidence: dict[str, dict] = {}
        for raw in tool_calls:
            if not isinstance(raw, dict):
                continue
            name = raw.get("name")
            args = raw.get("arguments", {})
            if name not in replayable:
                continue
            if not isinstance(args, dict):
                args = {}
            result = await self._tools.execute(ToolCall(name=name, arguments=args))
            evidence[result.tool_name] = result.to_dict()
        return evidence

    # ── LangGraph Node Functions ──────────────────────────────────────────

    def _think_node(self, state: AgentState) -> dict:
        """
        ANALYZE node: evidence-rationale step.

        Calls the LLM with the full conversation history and the evidence
        rationale prefix. Parses the response for either tool calls
        or a verdict signal.
        """
        iteration = state.get("iteration", 0)
        messages = list(state.get("messages", []))
        thinking_trace = list(state.get("thinking_trace", []))
        txn_id = state.get("alert", {}).get("txn_id", "unknown")
        evidence = state.get("evidence_collected", {}) or self._tool_evidence_cache.get(txn_id, {})

        # Build the prompt for this iteration
        if iteration == 0:
            # First iteration: investigation prompt
            context = self._graph_context(state.get("gnn_score", -1.0))
            try:
                from src.intel import get_pre_fraud_intel_service

                context["pre_fraud_intelligence"] = (
                    get_pre_fraud_intel_service().active_context_for_ai()
                )
            except Exception:
                pass
            user_msg = build_investigation_prompt(
                state["alert"],
                context=context,
            )
        else:
            # Continuation: evidence-rationale prompt with accumulated evidence
            evidence_summary = json.dumps(
                self._summarize_evidence_for_prompt(evidence),
                indent=2,
                default=str,
            )
            thinking_summary = "\n".join(
                f"Step {i + 1}: {t}" for i, t in enumerate(thinking_trace)
            )
            user_msg = build_cot_prompt(thinking_summary, evidence_summary)

        messages.append({"role": "user", "content": user_msg})

        # Call LLM
        response = self._call_llm(messages, tools=TOOL_SCHEMAS)

        # Parse response
        content = response.get("content", "")
        tool_calls = self._parse_tool_calls(response)

        # Update public evidence trace. The raw assistant response remains in
        # messages for JSON/tool parsing, but dashboard/HITL surfaces only see
        # this bounded evidence-rationale text.
        if content:
            public_content = self._public_trace_content(content)
            thinking_trace.append(public_content)
            # Broadcast investigation trace step to dashboard (best-effort)
            try:
                from src.api.events import EventBroadcaster
                EventBroadcaster.get().publish_sync("agent", {
                    "type": "thinking_step",
                    "txn_id": state.get("alert", {}).get("txn_id", "?"),
                    "iteration": iteration,
                    "max_iterations": state.get("max_iterations", 5),
                    "content": public_content,
                    "elapsed_ms": round((time.perf_counter() - state.get("_t0", time.perf_counter())) * 1000, 1) if "_t0" in state else 0,
                })
            except Exception:
                pass

        messages.append({"role": "assistant", "content": content})

        # Determine next status
        if tool_calls:
            if iteration + 1 >= state.get("max_iterations", self._cfg.max_iterations):
                status = "verdict"
                tool_calls = []
            else:
                status = "calling_tools"
        elif self._has_verdict_signal(content):
            # Extract intermediate confidence and typology for HITL check
            interim = self._extract_verdict(content)
            if interim is not None:
                conf = interim.get("confidence", 0.5)
                typo = interim.get("fraud_typology")
                if self._confidence_evaluator.should_escalate(
                    conf, typo, self._hitl_cfg,
                ):
                    status = "escalate"
                    return {
                        "messages": messages,
                        "thinking_trace": thinking_trace,
                        "evidence_collected": evidence,
                        "iteration": iteration + 1,
                        "status": status,
                        "intermediate_confidence": conf,
                        "detected_typology": typo,
                        "tool_calls_made": list(state.get("tool_calls_made", [])) + [
                            tc.to_dict() for tc in tool_calls
                        ],
                        "_pending_tool_calls": [],
                    }
            status = "verdict"
        elif iteration + 1 >= state.get("max_iterations", self._cfg.max_iterations):
            status = "verdict"  # force verdict at max iterations
        else:
            status = "thinking"

        return {
            "messages": messages,
            "thinking_trace": thinking_trace,
            "evidence_collected": evidence,
            "iteration": iteration + 1,
            "status": status,
            "tool_calls_made": list(state.get("tool_calls_made", [])) + [
                tc.to_dict() for tc in tool_calls
            ],
            "_pending_tool_calls": [tc for tc in tool_calls],
        }

    def _execute_tools_node(self, state: AgentState) -> dict:
        """
        EXECUTE_TOOLS node: Dispatch pending tool calls.

        Runs all pending tool calls concurrently and merges results
        into the evidence collection.
        """
        import asyncio

        pending: list[ToolCall] = state.get("_pending_tool_calls", [])
        evidence = dict(state.get("evidence_collected", {}))
        messages = list(state.get("messages", []))

        if not pending:
            return {
                "evidence_collected": evidence,
                "status": "thinking",
                "_pending_tool_calls": [],
            }

        # Execute tools (sync wrapper for LangGraph node)
        results: list[ToolResult] = []
        for tc in pending:
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # We're inside an async context — create a task
                    import concurrent.futures
                    with concurrent.futures.ThreadPoolExecutor() as pool:
                        result = pool.submit(
                            asyncio.run, self._tools.execute(tc)
                        ).result()
                else:
                    result = asyncio.run(self._tools.execute(tc))
            except RuntimeError:
                result = asyncio.run(self._tools.execute(tc))
            results.append(result)

        # Merge results into evidence
        for result in results:
            evidence[result.tool_name] = result.to_dict()
        txn_id = state.get("alert", {}).get("txn_id", "unknown")
        self._tool_evidence_cache[txn_id] = evidence

        # Broadcast tool calls to dashboard (best-effort)
        try:
            from src.api.events import EventBroadcaster
            broadcaster = EventBroadcaster.get()
            for result in results:
                broadcaster.publish_sync("agent", {
                    "type": "tool_call",
                    "txn_id": state.get("alert", {}).get("txn_id", "?"),
                    "iteration": state.get("iteration", 0),
                    "tool_name": result.tool_name,
                    "success": result.success,
                    "duration_ms": result.data.get("_execution_ms", 0)
                    if isinstance(result.data, dict)
                    else 0,
                    "output_summary": json.dumps(
                        self._summarize_tool_result(result),
                        default=str,
                    )[:300],
                })
        except Exception:
            pass

        # Add tool results as a message for the LLM context
        tool_summary = json.dumps(
            [self._summarize_tool_result(r) for r in results],
            indent=2,
            default=str,
        )
        messages.append({
            "role": "user",
            "content": f"## TOOL RESULTS\n\n```json\n{tool_summary}\n```",
        })

        return {
            "messages": messages,
            "evidence_collected": evidence,
            "status": "thinking",
            "_pending_tool_calls": [],
        }

    def _evidence_fallback_verdict(self, state: AgentState, reason: str) -> dict:
        """
        Deterministic fallback used only when Qwen does not return parseable JSON.

        It is deliberately marked as fallback so dashboards and audit consumers
        do not confuse it with a direct Qwen confidence value.
        """
        alert = state.get("alert", {})
        txn_id = alert.get("txn_id", "unknown")
        evidence = state.get("evidence_collected", {}) or self._tool_evidence_cache.get(txn_id, {})

        def as_float(value, default=0.0) -> float:
            try:
                return float(value)
            except (TypeError, ValueError):
                return default

        ml_score = as_float(alert.get("risk_score"), as_float(state.get("ml_score"), 0.0))
        threshold = as_float(alert.get("threshold"), 0.0)
        top_features = alert.get("top_features", {})
        if not isinstance(top_features, dict):
            top_features = {}

        graph_payload = evidence.get("query_graph_database", {})
        graph_data = graph_payload.get("data", {}) if isinstance(graph_payload, dict) else {}
        patterns = graph_data.get("patterns", {}) if isinstance(graph_data, dict) else {}
        connections = graph_data.get("connections", {}) if isinstance(graph_data, dict) else {}
        subgraph = graph_data.get("subgraph", {}) if isinstance(graph_data, dict) else {}

        mule_detected = bool(patterns.get("mule_network_detected"))
        cycles_found = int(as_float(patterns.get("cycles_found"), 0))
        distinct_senders = int(as_float(connections.get("distinct_senders"), 0))
        in_degree = int(as_float(connections.get("in_degree"), 0))
        subgraph_edges = int(as_float(subgraph.get("edges"), 0))

        feature_items = sorted(
            top_features.items(),
            key=lambda item: abs(as_float(item[1], 0.0)),
            reverse=True,
        )
        feature_names = {str(name).lower() for name, _ in feature_items[:8]}
        velocity_signal = any("vel_" in name or "velocity" in name for name in feature_names)
        geo_signal = any("geo" in name for name in feature_names)
        text_signal = any("txt_" in name or "edit" in name for name in feature_names)

        graph_signal = mule_detected or cycles_found > 0 or distinct_senders >= 5 or in_degree >= 5
        behavior_signal = velocity_signal or geo_signal or text_signal

        if mule_detected or distinct_senders >= 5 or in_degree >= 5:
            typology = "mule_network"
        elif cycles_found > 0:
            typology = "round_tripping"
        elif velocity_signal and ml_score >= 0.85:
            typology = "profile_behavior_mismatch"
        else:
            typology = None

        evidence_strength = 0.0
        if graph_signal:
            evidence_strength += 0.12
        if behavior_signal:
            evidence_strength += 0.08
        if subgraph_edges >= 10:
            evidence_strength += 0.03

        if ml_score >= max(threshold, 0.95) and (graph_signal or behavior_signal):
            verdict = "FRAUDULENT"
            action = "FREEZE"
            confidence = min(0.91, max(0.82, ml_score * 0.78 + evidence_strength))
        elif ml_score >= max(threshold, 0.70):
            verdict = "SUSPICIOUS"
            action = "ESCALATE"
            confidence = min(0.79, max(0.62, ml_score * 0.70 + evidence_strength))
        else:
            verdict = "LEGITIMATE"
            action = "MONITOR"
            confidence = min(0.72, max(0.55, 1.0 - ml_score))

        evidence_cited = [
            f"ML risk score: {ml_score:.4f}",
        ]
        if threshold > 0:
            evidence_cited.append(f"Dynamic threshold: {threshold:.4f}")
        if mule_detected:
            evidence_cited.append("Graph pattern: mule_network_detected=true")
        if cycles_found:
            evidence_cited.append(f"Graph pattern: cycles_found={cycles_found}")
        if distinct_senders:
            evidence_cited.append(f"Graph distinct_senders: {distinct_senders}")
        for name, value in feature_items[:4]:
            evidence_cited.append(f"{name}: {value}")

        graph_summary = (
            f"graph signals mule={mule_detected}, cycles={cycles_found}, "
            f"distinct_senders={distinct_senders}, subgraph_edges={subgraph_edges}"
        )
        reasoning = (
            f"Qwen did not return parseable verdict JSON ({reason}); the system "
            f"used bounded ML and graph evidence for an auditable fallback. "
            f"Risk score {ml_score:.4f} with {graph_summary} supports "
            f"{verdict.lower()} handling."
        )

        return {
            "verdict": verdict,
            "confidence": round(confidence, 4),
            "fraud_typology": typology,
            "reasoning_summary": reasoning,
            "evidence_cited": evidence_cited,
            "recommended_action": action,
            "confidence_source": "deterministic_evidence_fallback",
            "llm_parse_status": reason,
        }

    def _verdict_node(self, state: AgentState) -> dict:
        """
        VERDICT node: Extract final structured verdict.

        Either parses the verdict from the last LLM response, or forces
        a verdict extraction by sending the full reasoning trace to the LLM
        with a verdict-specific prompt.
        """
        messages = list(state.get("messages", []))
        thinking_trace = state.get("thinking_trace", [])
        txn_id = state.get("alert", {}).get("txn_id", "unknown")
        evidence = (
            state.get("evidence_collected", {})
            or self._tool_evidence_cache.get(txn_id, {})
        )

        # Try to extract verdict from the last assistant message
        last_content = ""
        for msg in reversed(messages):
            if msg.get("role") == "assistant":
                last_content = msg.get("content", "")
                break

        verdict = self._extract_verdict(last_content)
        parse_status = "parsed_from_reasoning"

        if verdict is None:
            # Force verdict extraction
            full_reasoning = "\n".join(
                f"Step {i + 1}: {t}" for i, t in enumerate(thinking_trace)
            )
            verdict_prompt = build_verdict_prompt(
                full_reasoning,
                alert_payload=state.get("alert", {}),
                evidence_summary=self._summarize_evidence_for_prompt(
                    evidence,
                ),
            )
            messages.append({"role": "user", "content": verdict_prompt})

            response = self._call_llm(messages, tools=None)
            content = response.get("content", "")
            messages.append({"role": "assistant", "content": content})
            verdict = self._extract_verdict(content)
            parse_status = "parsed_from_forced_json"

        # Fallback verdict if extraction still fails
        if verdict is None:
            verdict = self._evidence_fallback_verdict(
                state,
                "fallback_unparseable_json",
            )
        else:
            verdict.setdefault("confidence_source", "qwen_json")
            verdict.setdefault("llm_parse_status", parse_status)

        return {
            "messages": messages,
            "evidence_collected": evidence,
            "verdict": verdict,
            "status": "done",
        }

    # ── Routing Logic ────────────────────────────────────────────────────

    def _should_continue(self, state: AgentState) -> str:
        """
        Conditional edge: determine the next node based on current status.

        Returns:
            "execute_tools" | "think" | "verdict" | "escalate_hitl" |
            "evaluate_escalation" | "dispatch_hitl" | END
        """
        status = state.get("status", "thinking")

        if status == "calling_tools":
            return "execute_tools"
        elif status == "verdict":
            return "verdict"
        elif status == "escalate":
            return "escalate_hitl"
        elif status == "evaluate_escalation":
            return "evaluate_escalation"
        elif status == "dispatch":
            return "dispatch_hitl"
        elif status == "done":
            return END
        else:
            return "think"

    # ── HITL Escalation Nodes ────────────────────────────────────────────

    def _escalate_hitl_node(self, state: AgentState) -> dict:
        """
        ESCALATE_HITL node: Package graph context and build escalation payload.

        Triggered when the agent's confidence on a complex typology falls
        below the configured threshold. Prepares the full evidence package
        for human analyst review.
        """
        from src.llm.hitl import build_escalation_payload

        confidence = state.get("intermediate_confidence", 0.5)
        typology = state.get("detected_typology")

        # Package graph context
        graph_context = self._graph_packager.package(
            self._transaction_graph,
            state.get("alert", {}).get("sender_id", "unknown"),
            k_hops=self._hitl_cfg.graph_context_k_hops,
        )

        # Build the escalation payload
        threshold = self._confidence_evaluator.get_threshold(
            typology, self._hitl_cfg,
        )
        payload = build_escalation_payload(
            state=state,
            graph_context=graph_context,
            confidence=confidence,
            threshold=threshold,
            typology=typology,
        )

        logger.info(
            "HITL escalation prepared: txn=%s confidence=%.2f "
            "threshold=%.2f typology=%s",
            payload.txn_id, confidence, threshold, typology,
        )

        return {
            "escalation_payload": payload.to_dict(),
            "status": "evaluate_escalation",
        }

    def _evaluate_escalation_node(self, state: AgentState) -> dict:
        """
        EVALUATE_ESCALATION node: Second-chance confidence gate.

        Re-checks whether escalation is still necessary. If accumulated
        evidence has pushed confidence above the threshold (e.g., from
        tool results gathered during packaging), routes back to think
        instead of dispatching to the human analyst.
        """
        confidence = state.get("intermediate_confidence", 0.5)
        typology = state.get("detected_typology")

        if not self._confidence_evaluator.should_escalate(
            confidence, typology, self._hitl_cfg,
        ):
            # Confidence recovered — return to thinking
            self.hitl_metrics.escalations_recovered += 1
            logger.info(
                "HITL escalation recovered: confidence=%.2f now above "
                "threshold — returning to think",
                confidence,
            )
            return {
                "status": "thinking",
                "escalation_payload": None,
            }

        # Confirm escalation
        return {"status": "dispatch"}

    def _dispatch_hitl_node(self, state: AgentState) -> dict:
        """
        DISPATCH_HITL node: Send escalation payload to human analyst API.

        Creates an ESCALATED_TO_HUMAN verdict and dispatches the payload
        via HTTP POST. Records metrics and anchors to audit ledger.
        """
        import asyncio

        escalation_dict = state.get("escalation_payload", {})
        typology = state.get("detected_typology")
        confidence = state.get("intermediate_confidence", 0.5)

        # Dispatch via HTTP (sync wrapper for LangGraph node)
        dispatch_result = None
        from src.llm.hitl import HITLEscalationPayload
        try:
            # Reconstruct payload for dispatch
            payload = HITLEscalationPayload(
                escalation_id=escalation_dict.get("escalation_id", "unknown"),
                txn_id=escalation_dict.get("txn_id", "unknown"),
                node_id=escalation_dict.get("node_id", "unknown"),
                agent_confidence=escalation_dict.get("agent_confidence", 0.0),
                confidence_threshold=escalation_dict.get("confidence_threshold", 0.0),
                detected_typology=escalation_dict.get("detected_typology"),
                reasoning_trace=escalation_dict.get("reasoning_trace", []),
                evidence_collected=escalation_dict.get("evidence_collected", {}),
                graph_context=escalation_dict.get("graph_context", {}),
                ml_score=escalation_dict.get("ml_score", 0.0),
                gnn_score=escalation_dict.get("gnn_score", -1.0),
                nlu_findings=escalation_dict.get("nlu_findings"),
                recommended_action="ESCALATE_TO_HUMAN",
                escalated_at=escalation_dict.get("escalated_at", time.time()),
            )

            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    import concurrent.futures
                    with concurrent.futures.ThreadPoolExecutor() as pool:
                        dispatch_result = pool.submit(
                            asyncio.run,
                            self._hitl_dispatcher.dispatch(payload),
                        ).result()
                else:
                    dispatch_result = asyncio.run(
                        self._hitl_dispatcher.dispatch(payload),
                    )
            except RuntimeError:
                dispatch_result = asyncio.run(
                    self._hitl_dispatcher.dispatch(payload),
                )

        except Exception as exc:
            logger.error("HITL dispatch failed: %s", exc)

        # Record metrics
        dispatched = (
            dispatch_result is not None and dispatch_result.success
        )
        self.hitl_metrics.record_escalation(
            typology=typology,
            dispatched=dispatched,
            latency_ms=0.0,
        )

        # Build ESCALATED_TO_HUMAN verdict
        verdict = {
            "verdict": "ESCALATED_TO_HUMAN",
            "confidence": confidence,
            "fraud_typology": typology,
            "reasoning_summary": (
                f"Investigation escalated to human analyst. "
                f"Agent confidence ({confidence:.2f}) below threshold "
                f"for typology {typology or 'unknown'}."
            ),
            "evidence_cited": list(state.get("evidence_collected", {}).keys()),
            "recommended_action": "ESCALATE_TO_HUMAN",
        }

        logger.info(
            "HITL dispatch complete: txn=%s dispatched=%s",
            escalation_dict.get("txn_id", "unknown"), dispatched,
        )

        return {
            "verdict": verdict,
            "hitl_dispatched": dispatched,
            "status": "done",
        }

    # ── Graph Construction ───────────────────────────────────────────────

    def _build_graph(self) -> StateGraph:
        """
        Construct the LangGraph state machine.

        Nodes: think, execute_tools, verdict, escalate_hitl,
               evaluate_escalation, dispatch_hitl
        Edges: conditional routing based on agent status
        """
        graph = StateGraph(AgentState)

        # Add nodes
        graph.add_node("think", self._think_node)
        graph.add_node("execute_tools", self._execute_tools_node)
        graph.add_node("verdict", self._verdict_node)
        graph.add_node("escalate_hitl", self._escalate_hitl_node)
        graph.add_node("evaluate_escalation", self._evaluate_escalation_node)
        graph.add_node("dispatch_hitl", self._dispatch_hitl_node)

        # Entry point
        graph.set_entry_point("think")

        # Conditional edges from think node
        graph.add_conditional_edges(
            "think",
            self._should_continue,
            {
                "execute_tools": "execute_tools",
                "verdict": "verdict",
                "think": "think",
                "escalate_hitl": "escalate_hitl",
                "evaluate_escalation": "evaluate_escalation",
                "dispatch_hitl": "dispatch_hitl",
                END: END,
            },
        )

        # Execute_tools always returns to think
        graph.add_edge("execute_tools", "think")

        # Verdict goes to END
        graph.add_edge("verdict", END)

        # HITL escalation path: escalate → evaluate → dispatch or back to think
        graph.add_edge("escalate_hitl", "evaluate_escalation")
        graph.add_conditional_edges(
            "evaluate_escalation",
            self._should_continue,
            {
                "think": "think",
                "dispatch_hitl": "dispatch_hitl",
                END: END,
            },
        )
        graph.add_edge("dispatch_hitl", END)

        return graph.compile()

    # ── Public Interface ─────────────────────────────────────────────────

    async def investigate(
        self,
        alert_payload,
        ml_score: float = 0.0,
        gnn_score: float = -1.0,
    ) -> VerdictPayload:
        """
        Run a full investigation on a flagged transaction.

        This is the main entry point called by the AlertRouter consumer.

        Args:
            alert_payload: AlertPayload object (must have ``.to_dict()``).
            ml_score: ML risk score for the transaction.
            gnn_score: GNN topology risk score (-1.0 if unavailable).

        Returns:
            VerdictPayload with the structured fraud verdict.
        """
        t0 = time.perf_counter()
        self.metrics.investigations_started += 1

        alert_dict = (
            alert_payload.to_dict()
            if hasattr(alert_payload, "to_dict")
            else alert_payload
        )

        # Initialize state
        initial_state: AgentState = {
            "alert": alert_dict,
            "ml_score": ml_score,
            "gnn_score": gnn_score,
            "messages": [
                {"role": "system", "content": INVESTIGATOR_SYSTEM_PROMPT},
            ],
            "thinking_trace": [],
            "tool_calls_made": [],
            "evidence_collected": {},
            "iteration": 0,
            "max_iterations": self._cfg.max_iterations,
            "verdict": None,
            "status": "thinking",
            # HITL state
            "escalation_payload": None,
            "hitl_dispatched": False,
            "intermediate_confidence": 0.0,
            "detected_typology": None,
            "_pending_tool_calls": [],
            "_t0": t0,
        }

        # Run the LangGraph state machine in a thread — keeps the asyncio
        # event loop free so uvicorn can serve dashboard requests during
        # the blocking Ollama inference.
        final_state = await asyncio.to_thread(self._graph.invoke, initial_state)

        elapsed = (time.perf_counter() - t0) * 1000

        # Build VerdictPayload
        verdict_dict = final_state.get("verdict", {})
        tool_names = list({
            tc.get("name", "unknown")
            for tc in final_state.get("tool_calls_made", [])
        })

        # ── NLU Sub-Agent Consensus Integration ──────────────────────────
        # Run the unstructured data analysis sub-agent and fold its
        # qualitative findings into the verdict before finalisation.
        nlu_result = None
        nlu_risk_modifier = 0.0
        if self._unstructured_agent is not None:
            nlu_result = await self._run_nlu_consensus(
                alert_dict, final_state,
            )
            if nlu_result is not None:
                nlu_risk_modifier = nlu_result.get("overall_risk_modifier", 0.0)
                tool_names.append("analyze_unstructured_data")

        # Apply NLU risk modifier to confidence
        raw_confidence = verdict_dict.get("confidence", 0.5)
        adjusted_confidence = max(0.0, min(1.0, raw_confidence + nlu_risk_modifier))

        # If NLU found critical findings, escalate verdict if still SUSPICIOUS
        nlu_escalated = False
        raw_verdict = verdict_dict.get("verdict", "SUSPICIOUS")
        if (
            nlu_result is not None
            and nlu_result.get("has_critical_findings", False)
            and raw_verdict == "SUSPICIOUS"
            and adjusted_confidence >= 0.7
        ):
            raw_verdict = "FRAUDULENT"
            nlu_escalated = True

        # Enrich reasoning summary with NLU findings
        reasoning = verdict_dict.get(
            "reasoning_summary", "Investigation completed."
        )
        if nlu_result and nlu_result.get("findings_count", 0) > 0:
            reasoning += (
                f" NLU sub-agent detected {nlu_result['findings_count']} "
                f"semantic anomalies (SE={nlu_result.get('social_engineering_score', 0):.2f}, "
                f"LA={nlu_result.get('linguistic_anomaly_score', 0):.2f}, "
                f"DA={nlu_result.get('device_anomaly_score', 0):.2f})."
            )
            if nlu_escalated:
                reasoning += " Verdict escalated to FRAUDULENT based on critical NLU findings."

        # Enrich evidence cited with NLU findings
        evidence_cited = list(verdict_dict.get("evidence_cited", []))
        if nlu_result:
            for finding in nlu_result.get("findings", [])[:5]:
                evidence_cited.append(
                    f"NLU:{finding.get('anomaly_type', 'unknown')}"
                    f"[{finding.get('confidence', 'LOW')}]"
                )

        confidence_source = verdict_dict.get("confidence_source", "qwen_json")
        llm_parse_status = verdict_dict.get("llm_parse_status", "parsed")
        recommended_action = verdict_dict.get("recommended_action", "ESCALATE")
        has_cited_evidence = any(str(item).strip() for item in evidence_cited)
        if raw_verdict in {"FRAUDULENT", "SUSPICIOUS"} and not has_cited_evidence:
            raw_verdict = "SUSPICIOUS"
            adjusted_confidence = min(adjusted_confidence, 0.69)
            recommended_action = "ESCALATE"
            confidence_source = "qwen_json_evidence_capped"
            llm_parse_status = f"{llm_parse_status}|missing_evidence_capped"
            reasoning += (
                " Qwen did not cite specific evidence, so the decision was capped "
                "to suspicious analyst review instead of autonomous enforcement."
            )

        verdict = VerdictPayload(
            txn_id=alert_dict.get("txn_id", "unknown"),
            node_id=alert_dict.get("sender_id", "unknown"),
            verdict=raw_verdict,
            confidence=adjusted_confidence,
            fraud_typology=verdict_dict.get("fraud_typology"),
            reasoning_summary=reasoning,
            evidence_cited=evidence_cited,
            recommended_action=recommended_action,
            thinking_steps=final_state.get("iteration", 0),
            tools_used=tool_names,
            total_duration_ms=elapsed,
            confidence_source=confidence_source,
            llm_parse_status=llm_parse_status,
            model_used=(
                getattr(self._llm, "_resolved_model", None)
                or getattr(self._llm, "_model", None)
            ),
        )

        # Update metrics
        self.metrics.investigations_completed += 1
        self.metrics.total_investigation_ms += elapsed
        self.metrics.total_tool_calls += len(final_state.get("tool_calls_made", []))
        self.metrics.total_thinking_steps += verdict.thinking_steps

        if verdict.verdict == "FRAUDULENT":
            self.metrics.verdicts_fraudulent += 1
        elif verdict.verdict == "SUSPICIOUS":
            self.metrics.verdicts_suspicious += 1
        elif verdict.verdict == "ESCALATED_TO_HUMAN":
            self.metrics.verdicts_escalated += 1
        else:
            self.metrics.verdicts_legitimate += 1

        # Anchor verdict to audit ledger
        if self._audit_ledger is not None:
            try:
                await self._audit_ledger.anchor_agent_verdict(verdict)
            except Exception as exc:
                logger.warning(
                    "Ledger anchoring failed for agent verdict %s: %s",
                    verdict.txn_id, exc,
                )

        # Broadcast verdict to dashboard (best-effort)
        try:
            from src.api.events import EventBroadcaster
            await EventBroadcaster.get().publish("agent", {
                "type": "verdict",
                "txn_id": verdict.txn_id,
                "node_id": verdict.node_id,
                "verdict": verdict.verdict,
                "confidence": verdict.confidence,
                "fraud_typology": verdict.fraud_typology,
                "reasoning_summary": verdict.reasoning_summary,
                "evidence_cited": verdict.evidence_cited,
                "recommended_action": verdict.recommended_action,
                "thinking_steps": verdict.thinking_steps,
                "tools_used": verdict.tools_used,
                "total_duration_ms": verdict.total_duration_ms,
                "confidence_source": verdict.confidence_source,
                "llm_parse_status": verdict.llm_parse_status,
                "model_used": verdict.model_used,
                "nlu_findings_count": nlu_result.get("findings_count", 0) if nlu_result else 0,
                "nlu_escalated": nlu_escalated,
            })
        except Exception:
            pass

        # Agent circuit breaker listener — trigger defensive actions
        if self._agent_breaker_listener is not None:
            try:
                breaker_event = await self._agent_breaker_listener.on_verdict(
                    verdict, alert_dict,
                )
                if breaker_event is not None:
                    self.metrics.agent_breaker_triggered += 1
            except Exception as exc:
                logger.warning(
                    "Agent breaker listener failed for %s: %s",
                    verdict.txn_id, exc,
                )

        logger.info(
            "Investigation complete: txn=%s verdict=%s confidence=%.2f "
            "typology=%s action=%s steps=%d tools=%d (%.1f ms)",
            verdict.txn_id, verdict.verdict, verdict.confidence,
            verdict.fraud_typology, verdict.recommended_action,
            verdict.thinking_steps, len(tool_names), elapsed,
        )

        cached_evidence = self._tool_evidence_cache.pop(verdict.txn_id, {})
        evidence_for_record = (
            final_state.get("evidence_collected", {})
            or cached_evidence
            or self._evidence_from_tool_result_messages(final_state.get("messages", []))
        )
        if not evidence_for_record and final_state.get("tool_calls_made"):
            evidence_for_record = await self._replay_tool_evidence(
                final_state.get("tool_calls_made", [])
            )

        # Store investigation record for API retrieval
        self._investigation_records[verdict.txn_id] = {
            "txn_id": verdict.txn_id,
            "node_id": verdict.node_id,
            "verdict": verdict.to_dict(),
            "iterations": final_state.get("iteration", 0),
            "thinking_trace": final_state.get("thinking_trace", []),
            "tool_calls": final_state.get("tool_calls_made", []),
            "evidence_collected": {
                k: v for k, v in evidence_for_record.items()
            },
            "nlu_findings": nlu_result if nlu_result else None,
            "nlu_escalated": nlu_escalated,
            "total_duration_ms": round(elapsed, 2),
            "timestamp": time.time(),
        }

        return verdict

    async def on_alert(self, payload) -> None:
        """
        AlertRouter consumer protocol — receives HIGH-tier alerts.

        Conforms to the ``AgentConsumer`` callback signature.
        """
        verdict = await self.investigate(
            alert_payload=payload,
            ml_score=payload.risk_score,
        )
        return verdict

    def get_investigation(self, txn_id: str) -> dict | None:
        """Return the full investigation record for a transaction, or None."""
        return self._investigation_records.get(txn_id)

    # ── LLM Interaction ──────────────────────────────────────────────────

    def _call_llm(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
    ) -> dict:
        """
        Call the Qwen 3.5 4B model via the PayFlowLLM client.

        Returns a dict with ``content`` (str) and optionally ``tool_calls``
        (list of dicts with ``name`` and ``arguments``).
        """
        if self._llm is None:
            # No LLM client — return empty response for testing
            return {"content": "", "tool_calls": []}

        try:
            from config.settings import OLLAMA_CFG
            last_user = next(
                (
                    str(message.get("content", ""))
                    for message in reversed(messages)
                    if message.get("role") == "user"
                ),
                "",
            )
            is_verdict_request = tools is None and "FINAL VERDICT REQUIRED" in last_user
            temperature = (
                OLLAMA_CFG.verdict_temperature
                if is_verdict_request
                else self._cfg.thinking_temperature
            )
            max_tokens = (
                min(self._cfg.max_verdict_tokens, OLLAMA_CFG.agent_max_tokens)
                if is_verdict_request
                else min(self._cfg.max_thinking_tokens, OLLAMA_CFG.agent_max_tokens)
            )
            result = self._llm.chat(
                messages,
                tools=tools,
                temperature=temperature,
                max_tokens=max_tokens,
                num_ctx=OLLAMA_CFG.agent_num_ctx,
                timeout=OLLAMA_CFG.request_timeout_sec,
                response_format=VERDICT_SCHEMA if is_verdict_request else None,
            )
            return {
                "content": result.get("content", "") or "",
                "tool_calls": result.get("tool_calls", []),
            }

        except Exception as exc:
            logger.error("LLM call failed: %s", exc)
            return {"content": f"[LLM Error: {exc}]", "tool_calls": []}

    def _parse_tool_calls(self, response: dict) -> list[ToolCall]:
        """Parse tool calls from an LLM response dict."""
        raw_calls = response.get("tool_calls", [])
        parsed = []
        for tc in raw_calls:
            name = tc.get("name", "")
            args = tc.get("arguments", {})
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {}
            if name:
                parsed.append(ToolCall(name=name, arguments=args))
        return parsed

    def _has_verdict_signal(self, content: str) -> bool:
        """Check if the LLM response contains a verdict JSON object."""
        if not content:
            return False
        verdict_markers = [
            '"verdict"',
            "verdict:",
            "FINAL_VERDICT",
            '"FRAUDULENT"',
            '"SUSPICIOUS"',
            '"LEGITIMATE"',
            "FRAUDULENT",
            "SUSPICIOUS",
            "LEGITIMATE",
        ]
        return any(marker in content for marker in verdict_markers)

    def _coerce_confidence(self, value: Any) -> float:
        """Parse confidence from numeric, percent-string, or coarse labels."""
        if isinstance(value, str):
            raw = value.strip().lower()
            label_map = {"high": 0.85, "medium": 0.65, "low": 0.4}
            if raw in label_map:
                return label_map[raw]
            raw = raw.rstrip("%")
            confidence = float(raw)
        else:
            confidence = float(value)
        if confidence > 1.0 and confidence <= 100.0:
            confidence /= 100.0
        return max(0.0, min(1.0, confidence))

    def _normalize_verdict_dict(self, parsed: dict) -> dict | None:
        """Normalize common Qwen JSON variants into the audit schema."""
        if not isinstance(parsed, dict):
            return None

        for key in ("final_verdict", "verdict_payload", "decision"):
            nested = parsed.get(key)
            if isinstance(nested, dict):
                parsed = nested
                break

        raw_verdict = str(parsed.get("verdict", parsed.get("classification", ""))).upper()
        if raw_verdict not in ("FRAUDULENT", "SUSPICIOUS", "LEGITIMATE"):
            return None

        raw_confidence = parsed.get(
            "confidence",
            parsed.get("confidence_score", parsed.get("certainty", None)),
        )
        if raw_confidence is None:
            return None
        confidence = self._coerce_confidence(raw_confidence)

        action = str(
            parsed.get(
                "recommended_action",
                parsed.get("action", parsed.get("recommendation", "ESCALATE")),
            )
        ).upper()
        if "FREEZE" in action or "BLOCK" in action:
            action = "FREEZE"
        elif "CLEAR" in action or "APPROVE" in action:
            action = "CLEAR"
        elif "MONITOR" in action:
            action = "MONITOR"
        elif "ESCALATE" in action or "REVIEW" in action:
            action = "ESCALATE"
        else:
            action = "ESCALATE"

        evidence = parsed.get("evidence_cited", parsed.get("evidence", []))
        if isinstance(evidence, str):
            evidence = [evidence]
        elif not isinstance(evidence, list):
            evidence = []

        typology = parsed.get("fraud_typology", parsed.get("typology"))
        if raw_verdict == "LEGITIMATE":
            typology = None

        return {
            "verdict": raw_verdict,
            "confidence": confidence,
            "fraud_typology": typology,
            "reasoning_summary": str(
                parsed.get(
                    "reasoning_summary",
                    parsed.get("reasoning", parsed.get("summary", "No summary provided.")),
                )
            ),
            "evidence_cited": [str(item) for item in evidence[:10]],
            "recommended_action": action,
        }

    def _extract_verdict(self, content: str) -> dict | None:
        """
        Extract a structured verdict JSON from the LLM response content.

        Attempts to parse JSON from the content, looking for the verdict
        schema fields. Returns None if extraction fails.
        """
        if not content:
            return None

        # Try to find JSON block in the content
        json_candidates = []

        # Look for ```json ... ``` blocks
        import re
        json_blocks = re.findall(r"```(?:json)?\s*\n?(.*?)\n?\s*```", content, re.DOTALL)
        json_candidates.extend(json_blocks)

        # Look for bare JSON objects
        brace_depth = 0
        start = -1
        for i, ch in enumerate(content):
            if ch == "{":
                if brace_depth == 0:
                    start = i
                brace_depth += 1
            elif ch == "}":
                brace_depth -= 1
                if brace_depth == 0 and start >= 0:
                    json_candidates.append(content[start:i + 1])
                    start = -1

        for candidate in json_candidates:
            try:
                parsed = json.loads(candidate.strip())
                if isinstance(parsed, list):
                    parsed = next((item for item in parsed if isinstance(item, dict)), None)
                normalized = self._normalize_verdict_dict(parsed)
                if normalized is not None:
                    return normalized
            except (json.JSONDecodeError, ValueError, TypeError):
                continue

        # Last resort for Markdown/plain-text answers that include all fields.
        verdict_match = re.search(r"\b(FRAUDULENT|SUSPICIOUS|LEGITIMATE)\b", content, re.I)
        confidence_match = re.search(
            r"confidence(?:\s*score)?\s*[:=-]\s*([0-9]+(?:\.[0-9]+)?%?)",
            content,
            re.I,
        )
        action_match = re.search(r"\b(FREEZE|ESCALATE|MONITOR|CLEAR)\b", content, re.I)
        if verdict_match and confidence_match and action_match:
            try:
                return {
                    "verdict": verdict_match.group(1).upper(),
                    "confidence": self._coerce_confidence(confidence_match.group(1)),
                    "fraud_typology": None,
                    "reasoning_summary": content[:500],
                    "evidence_cited": [],
                    "recommended_action": action_match.group(1).upper(),
                }
            except (ValueError, TypeError):
                return None

        return None

    # ── NLU Sub-Agent Consensus ─────────────────────────────────────────

    async def _run_nlu_consensus(
        self,
        alert_dict: dict,
        final_state: dict,
    ) -> dict | None:
        """
        Run the NLU sub-agent on the unstructured data associated with a
        transaction and return the analysis result as a dict.

        Extracts textual fields from the alert payload, builds an
        UnstructuredPayload, invokes the sub-agent analysis, and returns
        the structured result for integration into the verdict consensus.

        Returns None if the sub-agent is unavailable or analysis fails.
        """
        from src.llm.unstructured_models import UnstructuredPayload

        try:
            # Extract identifiers
            txn_id = alert_dict.get("txn_id", "unknown")
            sender_id = alert_dict.get("sender_id", "unknown")
            receiver_id = alert_dict.get("receiver_id", "unknown")

            # Extract optional unstructured text fields from alert
            payload = UnstructuredPayload(
                txn_id=txn_id,
                sender_id=sender_id,
                receiver_id=receiver_id,
                sender_name=alert_dict.get("sender_name"),
                receiver_name=alert_dict.get("receiver_name"),
                remittance_info=alert_dict.get("remittance_info"),
                email_subject=alert_dict.get("email_subject"),
                email_sender=alert_dict.get("email_sender"),
                email_body_snippet=alert_dict.get("email_body_snippet"),
                user_agent=alert_dict.get("user_agent"),
                device_fingerprint=alert_dict.get("device_fingerprint"),
                previous_device_fingerprint=alert_dict.get(
                    "previous_device_fingerprint",
                ),
                ip_geo=alert_dict.get("ip_geo"),
                swift_message=alert_dict.get("swift_message"),
                neft_narration=alert_dict.get("neft_narration"),
            )

            # Skip if no textual data present
            if not payload.has_textual_data:
                logger.debug(
                    "NLU consensus skipped for %s: no textual data", txn_id,
                )
                return None

            # Run analysis
            result = await self._unstructured_agent.analyze(payload)
            result_dict = result.to_dict()

            # Attach has_critical_findings for the verdict escalation logic
            result_dict["has_critical_findings"] = result.has_critical_findings

            logger.info(
                "NLU consensus for %s: %d findings, risk_mod=%+.3f, "
                "critical=%s",
                txn_id,
                result_dict.get("findings_count", 0),
                result_dict.get("overall_risk_modifier", 0.0),
                result_dict["has_critical_findings"],
            )

            return result_dict

        except Exception as exc:
            logger.warning(
                "NLU sub-agent consensus failed for %s: %s",
                alert_dict.get("txn_id", "unknown"), exc,
            )
            return None

    # ── Diagnostics ──────────────────────────────────────────────────────

    def snapshot(self) -> dict:
        """Full agent state for monitoring dashboards."""
        return {
            "metrics": self.metrics.snapshot(),
            "hitl_metrics": self.hitl_metrics.snapshot(),
            "config": {
                "max_iterations": self._cfg.max_iterations,
                "thinking_temperature": self._cfg.thinking_temperature,
                "verdict_temperature": self._cfg.verdict_temperature,
                "enable_investigation_trace": self._cfg.enable_cot_trace,
            },
            "tools_available": [
                s["function"]["name"] for s in TOOL_SCHEMAS
            ],
        }

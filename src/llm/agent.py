"""
PayFlow -- LangGraph Investigator Agent
=========================================
Autonomous fraud investigation agent powered by Qwen 3.5 9B via Ollama,
orchestrated through a LangGraph state-machine graph.

The agent implements a **think-act-observe** loop:

1. **THINK** (Chain-of-Thought): Reason step-by-step about current evidence.
   The ``/think`` prefix activates Qwen's internal reasoning mode.
2. **ACT** (Tool Calls): Invoke PayFlow subsystem tools to gather evidence.
3. **OBSERVE** (Evidence Integration): Incorporate tool results into the
   reasoning trace.
4. **DECIDE** (Verdict): Issue a structured fraud verdict when confident
   or when max iterations are exhausted.

State Machine::

    ┌──────────┐
    │  START   │
    └────┬─────┘
         │
         ▼
    ┌──────────┐     "tools"      ┌──────────────┐
    │  THINK   │ ──────────────> │ EXECUTE_TOOLS │
    │  (CoT)   │ <────────────── │              │
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
    - ollama (Qwen 3.5 9B inference via PayFlowLLM)
    - src.llm.tools (ToolExecutor, ToolCall, ToolResult)
    - src.llm.prompts (system prompts, CoT templates)
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
    COT_ACTIVATION_PREFIX,
    INVESTIGATOR_SYSTEM_PROMPT,
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
    thinking_trace: list[str]            # Chain-of-thought steps
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
    reasoning_summary: str               # Condensed CoT summary
    evidence_cited: list[str]            # Specific evidence items cited
    recommended_action: str              # "FREEZE" | "ESCALATE" | "MONITOR" | "CLEAR" | "ESCALATE_TO_HUMAN"
    thinking_steps: int                  # Number of CoT iterations
    tools_used: list[str]               # Names of tools invoked
    total_duration_ms: float

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
    LangGraph-powered fraud investigation agent using Qwen 3.5 9B.

    Implements a think-act-observe loop where the LLM reasons through
    fraud typologies step-by-step (Chain-of-Thought / Thinking Mode),
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

    # ── LangGraph Node Functions ──────────────────────────────────────────

    def _think_node(self, state: AgentState) -> dict:
        """
        THINK node: Chain-of-Thought reasoning step.

        Calls the LLM with the full conversation history and the CoT
        activation prefix. Parses the response for either tool calls
        or a verdict signal.
        """
        iteration = state.get("iteration", 0)
        messages = list(state.get("messages", []))
        thinking_trace = list(state.get("thinking_trace", []))
        evidence = state.get("evidence_collected", {})

        # Build the prompt for this iteration
        if iteration == 0:
            # First iteration: investigation prompt
            user_msg = build_investigation_prompt(
                state["alert"],
                context={"gnn_score": state.get("gnn_score", -1.0)},
            )
        else:
            # Continuation: CoT prompt with accumulated evidence
            evidence_summary = json.dumps(evidence, indent=2, default=str)
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

        # Update thinking trace
        if content:
            thinking_trace.append(content[:500])  # cap per-step trace length
            # Broadcast CoT step to dashboard (best-effort)
            try:
                from src.api.events import EventBroadcaster
                EventBroadcaster.get().publish_sync("agent", {
                    "type": "thinking_step",
                    "txn_id": state.get("alert", {}).get("txn_id", "?"),
                    "iteration": iteration,
                    "max_iterations": state.get("max_iterations", 5),
                    "content": content[:500],
                    "elapsed_ms": round((time.perf_counter() - state.get("_t0", time.perf_counter())) * 1000, 1) if "_t0" in state else 0,
                })
            except Exception:
                pass

        messages.append({"role": "assistant", "content": content})

        # Determine next status
        if tool_calls:
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
            return {"status": "thinking", "_pending_tool_calls": []}

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
                    "duration_ms": getattr(result, "duration_ms", 0),
                    "output_summary": str(result.output)[:300] if hasattr(result, "output") else "",
                })
        except Exception:
            pass

        # Add tool results as a message for the LLM context
        tool_summary = json.dumps(
            [r.to_dict() for r in results], indent=2, default=str,
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

    def _verdict_node(self, state: AgentState) -> dict:
        """
        VERDICT node: Extract final structured verdict.

        Either parses the verdict from the last LLM response, or forces
        a verdict extraction by sending the full reasoning trace to the LLM
        with a verdict-specific prompt.
        """
        messages = list(state.get("messages", []))
        thinking_trace = state.get("thinking_trace", [])

        # Try to extract verdict from the last assistant message
        last_content = ""
        for msg in reversed(messages):
            if msg.get("role") == "assistant":
                last_content = msg.get("content", "")
                break

        verdict = self._extract_verdict(last_content)

        if verdict is None:
            # Force verdict extraction
            full_reasoning = "\n".join(
                f"Step {i + 1}: {t}" for i, t in enumerate(thinking_trace)
            )
            verdict_prompt = build_verdict_prompt(full_reasoning)
            messages.append({"role": "user", "content": verdict_prompt})

            response = self._call_llm(messages, tools=None)
            content = response.get("content", "")
            messages.append({"role": "assistant", "content": content})
            verdict = self._extract_verdict(content)

        # Fallback verdict if extraction still fails
        if verdict is None:
            verdict = {
                "verdict": "SUSPICIOUS",
                "confidence": 0.5,
                "fraud_typology": None,
                "reasoning_summary": "Unable to reach definitive conclusion within iteration limit.",
                "evidence_cited": [],
                "recommended_action": "ESCALATE",
            }

        return {
            "messages": messages,
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

        verdict = VerdictPayload(
            txn_id=alert_dict.get("txn_id", "unknown"),
            node_id=alert_dict.get("sender_id", "unknown"),
            verdict=raw_verdict,
            confidence=adjusted_confidence,
            fraud_typology=verdict_dict.get("fraud_typology"),
            reasoning_summary=reasoning,
            evidence_cited=evidence_cited,
            recommended_action=verdict_dict.get("recommended_action", "ESCALATE"),
            thinking_steps=final_state.get("iteration", 0),
            tools_used=tool_names,
            total_duration_ms=elapsed,
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

        # Store investigation record for API retrieval
        self._investigation_records[verdict.txn_id] = {
            "txn_id": verdict.txn_id,
            "node_id": verdict.node_id,
            "verdict": verdict.to_dict(),
            "iterations": final_state.get("iteration", 0),
            "thinking_trace": final_state.get("thinking_trace", []),
            "tool_calls": final_state.get("tool_calls_made", []),
            "evidence_collected": {
                k: v for k, v in final_state.get("evidence_collected", {}).items()
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
        Call the Qwen 3.5 9B model via the PayFlowLLM client.

        Returns a dict with ``content`` (str) and optionally ``tool_calls``
        (list of dicts with ``name`` and ``arguments``).
        """
        if self._llm is None:
            # No LLM client — return empty response for testing
            return {"content": "", "tool_calls": []}

        try:
            from config.settings import OLLAMA_CFG
            from config.vram_manager import assistant_mode
            import httpx

            model = self._llm._ensure_model_available()

            payload: dict[str, Any] = {
                "model": model,
                "messages": messages,
                "stream": False,
                "options": {
                    "temperature": self._cfg.thinking_temperature,
                    "num_predict": self._cfg.max_thinking_tokens,
                    "num_ctx": OLLAMA_CFG.num_ctx,
                },
            }

            if tools:
                payload["tools"] = tools

            with assistant_mode():
                response = httpx.post(
                    f"{OLLAMA_CFG.base_url}/api/chat",
                    json=payload,
                    timeout=120.0,
                )
                response.raise_for_status()
                response_data = response.json()

            message = response_data.get("message", {}) if isinstance(response_data, dict) else {}
            content = message.get("content", "") if isinstance(message, dict) else ""
            raw_tool_calls = message.get("tool_calls", []) if isinstance(message, dict) else []

            result: dict[str, Any] = {
                "content": content or "",
                "tool_calls": [],
            }

            for tc in raw_tool_calls:
                if not isinstance(tc, dict):
                    continue
                function_data = tc.get("function", {})
                if not isinstance(function_data, dict):
                    continue
                arguments = function_data.get("arguments", {})
                if isinstance(arguments, str):
                    try:
                        arguments = json.loads(arguments)
                    except json.JSONDecodeError:
                        arguments = {}
                result["tool_calls"].append({
                    "name": function_data.get("name", ""),
                    "arguments": arguments,
                })

            return result

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
        verdict_markers = ['"verdict"', '"FRAUDULENT"', '"SUSPICIOUS"', '"LEGITIMATE"']
        return any(marker in content for marker in verdict_markers)

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

        # Try to parse each candidate
        required_keys = {"verdict", "confidence", "recommended_action"}

        for candidate in json_candidates:
            try:
                parsed = json.loads(candidate.strip())
                if isinstance(parsed, dict) and required_keys.issubset(parsed.keys()):
                    # Normalize and validate
                    verdict = parsed.get("verdict", "SUSPICIOUS")
                    if verdict not in ("FRAUDULENT", "SUSPICIOUS", "LEGITIMATE"):
                        verdict = "SUSPICIOUS"

                    confidence = float(parsed.get("confidence", 0.5))
                    confidence = max(0.0, min(1.0, confidence))

                    action = parsed.get("recommended_action", "ESCALATE")
                    if action not in ("FREEZE", "ESCALATE", "MONITOR", "CLEAR"):
                        action = "ESCALATE"

                    return {
                        "verdict": verdict,
                        "confidence": confidence,
                        "fraud_typology": parsed.get("fraud_typology"),
                        "reasoning_summary": parsed.get(
                            "reasoning_summary", "No summary provided."
                        ),
                        "evidence_cited": parsed.get("evidence_cited", []),
                        "recommended_action": action,
                    }
            except (json.JSONDecodeError, ValueError, TypeError):
                continue

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
                "enable_cot_trace": self._cfg.enable_cot_trace,
            },
            "tools_available": [
                s["function"]["name"] for s in TOOL_SCHEMAS
            ],
        }

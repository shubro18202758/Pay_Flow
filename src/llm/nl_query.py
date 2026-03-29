"""
PayFlow — Natural Language Query Interface
=============================================
Enables analysts to query the fraud detection system using natural
language, powered by Qwen 3.5 9b. Translates questions into
structured API calls and returns contextual responses.

Supports queries like:
  - "Show me the top 5 riskiest accounts"
  - "What fraud patterns were detected in the last hour?"
  - "Explain why account ACC_0042 was frozen"
  - "How many SWIFT heist attempts have been blocked?"
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class NLQueryResult:
    """Result of a natural language query."""
    query: str
    intent: str
    answer: str
    data: dict
    sources: list[str]
    confidence: float
    processing_ms: float
    model_used: str


class NLQueryEngine:
    """
    Natural language query engine using Qwen 3.5 via Ollama.

    Interprets analyst questions, routes to appropriate data sources,
    and generates contextual responses.
    """

    SYSTEM_PROMPT = """You are PayFlow Intelligence Analyst, an AI assistant for Union Bank of India's fraud detection system.
You answer questions about fraud detection, account risk, transaction patterns, and system status.

You have access to the following data sources:
1. Transaction Graph - network of accounts and transactions with fraud labels
2. Circuit Breaker - frozen accounts and enforcement actions
3. ML Models - XGBoost, IsolationForest, Autoencoder risk scores
4. Audit Ledger - blockchain-anchored cryptographic audit trail
5. Central Fraud Registry (CFR) - cross-bank fraud intelligence
6. Agent Verdicts - AI agent investigation results
7. System Metrics - hardware, pipeline, model performance

When answering:
- Be precise with numbers and account IDs
- Reference specific fraud patterns (UPI_MULE, CIRCULAR_LAUNDERING, VELOCITY_PHISHING, SWIFT_HEIST, etc.)
- Use INR amounts (₹) for currency
- Cite evidence from the system when available
- If data is unavailable, say so clearly

Respond in a structured format with clear sections."""

    INTENT_CLASSIFIER_PROMPT = """Classify the user's query intent. Return ONLY a JSON object:
{
    "intent": "one of: risk_query, account_lookup, fraud_patterns, system_status, explanation, statistics, recommendation, general",
    "entities": {"account_id": "...", "time_range": "...", "fraud_type": "...", "limit": N},
    "data_sources": ["graph", "circuit_breaker", "ml_models", "ledger", "cfr", "verdicts", "metrics"]
}

User query: """

    def __init__(self, llm_client=None, orchestrator=None):
        self._llm = llm_client
        self._orchestrator = orchestrator
        self._query_count: int = 0
        self._avg_response_ms: float = 0.0

    def attach_llm(self, llm_client) -> None:
        self._llm = llm_client

    def attach_orchestrator(self, orchestrator) -> None:
        self._orchestrator = orchestrator

    async def query(self, question: str) -> NLQueryResult:
        """Process a natural language query and return structured results."""
        t0 = time.monotonic()

        # Classify intent
        intent_info = await self._classify_intent(question)
        intent = intent_info.get("intent", "general")
        entities = intent_info.get("entities", {})
        data_sources = intent_info.get("data_sources", [])

        # Gather context from data sources
        context = await self._gather_context(intent, entities, data_sources)

        # Generate answer using LLM
        answer = await self._generate_answer(question, intent, context)

        elapsed = (time.monotonic() - t0) * 1000
        self._query_count += 1
        self._avg_response_ms = (
            (self._avg_response_ms * (self._query_count - 1) + elapsed) / self._query_count
        )

        return NLQueryResult(
            query=question,
            intent=intent,
            answer=answer,
            data=context,
            sources=data_sources,
            confidence=0.85 if self._llm else 0.5,
            processing_ms=round(elapsed, 2),
            model_used="qwen3.5:9b-q4_K_M" if self._llm else "fallback",
        )

    async def _classify_intent(self, question: str) -> dict:
        """Use LLM to classify query intent, with fallback heuristics."""
        if self._llm:
            try:
                prompt = self.INTENT_CLASSIFIER_PROMPT + question
                response = await self._llm.generate(prompt, temperature=0.1, max_tokens=256)
                text = response.strip()
                # Extract JSON from response
                start = text.find("{")
                end = text.rfind("}") + 1
                if start >= 0 and end > start:
                    return json.loads(text[start:end])
            except Exception as e:
                logger.debug("Intent classification via LLM failed: %s", e)

        # Fallback heuristic classification
        q = question.lower()
        if any(w in q for w in ["risk", "score", "dangerous", "suspicious"]):
            return {"intent": "risk_query", "entities": {}, "data_sources": ["ml_models", "graph"]}
        if any(w in q for w in ["account", "acc_", "frozen", "freeze"]):
            return {"intent": "account_lookup", "entities": {}, "data_sources": ["graph", "circuit_breaker"]}
        if any(w in q for w in ["pattern", "mule", "laundering", "phishing", "swift"]):
            return {"intent": "fraud_patterns", "entities": {}, "data_sources": ["graph", "verdicts"]}
        if any(w in q for w in ["status", "health", "gpu", "vram", "cpu", "pipeline"]):
            return {"intent": "system_status", "entities": {}, "data_sources": ["metrics"]}
        if any(w in q for w in ["why", "explain", "reason", "because"]):
            return {"intent": "explanation", "entities": {}, "data_sources": ["verdicts", "ml_models"]}
        if any(w in q for w in ["how many", "count", "total", "statistics"]):
            return {"intent": "statistics", "entities": {}, "data_sources": ["graph", "metrics"]}
        return {"intent": "general", "entities": {}, "data_sources": ["metrics"]}

    async def _gather_context(
        self, intent: str, entities: dict, data_sources: list[str],
    ) -> dict:
        """Gather relevant data from system components."""
        context: dict[str, Any] = {}

        if not self._orchestrator:
            return context

        orch = self._orchestrator

        if "metrics" in data_sources or intent == "system_status":
            try:
                context["system_snapshot"] = orch.full_snapshot()
            except Exception:
                pass

        if "graph" in data_sources:
            graph = getattr(orch, "_graph", None)
            if graph:
                try:
                    g = graph._graph
                    context["graph_summary"] = {
                        "nodes": g.number_of_nodes(),
                        "edges": g.number_of_edges(),
                        "fraud_edges": sum(
                            1 for _, _, d in g.edges(data=True)
                            if d.get("fraud_label", 0) > 0
                        ),
                    }
                    # Top risky nodes
                    node_risks = []
                    for node in list(g.nodes())[:500]:
                        data = g.nodes[node]
                        risk = data.get("risk_score", 0.0)
                        if risk > 0.5:
                            node_risks.append({"id": node, "risk": round(risk, 4)})
                    node_risks.sort(key=lambda x: x["risk"], reverse=True)
                    context["top_risky_nodes"] = node_risks[:20]
                except Exception:
                    pass

        if "circuit_breaker" in data_sources:
            breaker = getattr(orch, "_breaker", None)
            if breaker:
                try:
                    context["circuit_breaker"] = breaker.snapshot()
                except Exception:
                    pass

        if "ml_models" in data_sources:
            # Gather model performance info
            classifier = getattr(orch, "_classifier", None)
            if classifier and classifier.is_fitted:
                try:
                    context["ml_models"] = {
                        "xgboost": {"fitted": True, "device": classifier.device},
                    }
                except Exception:
                    pass

            drift = getattr(orch, "_drift_detector", None)
            if drift:
                context["model_drift"] = drift.snapshot()

        if "verdicts" in data_sources:
            agent = getattr(orch, "_agent", None)
            if agent:
                try:
                    snap = agent.snapshot()
                    context["agent_verdicts"] = {
                        "total": snap.get("total_investigations", 0),
                        "recent": snap.get("recent_verdicts", [])[:10],
                    }
                except Exception:
                    pass

        if "cfr" in data_sources:
            cfr = getattr(orch, "_fraud_registry", None)
            if cfr:
                try:
                    context["cfr"] = cfr.snapshot()
                except Exception:
                    pass

        return context

    async def _generate_answer(self, question: str, intent: str, context: dict) -> str:
        """Generate a natural language answer using LLM or structured fallback."""
        if self._llm:
            try:
                # Build context summary for LLM
                context_str = json.dumps(context, indent=2, default=str)[:4000]
                prompt = (
                    f"{self.SYSTEM_PROMPT}\n\n"
                    f"System Context:\n{context_str}\n\n"
                    f"User Question: {question}\n\n"
                    f"Provide a clear, data-driven answer:"
                )
                response = await self._llm.generate(
                    prompt, temperature=0.3, max_tokens=1024,
                )
                return response.strip()
            except Exception as e:
                logger.debug("LLM answer generation failed: %s", e)

        # Structured fallback
        return self._fallback_answer(intent, context)

    def _fallback_answer(self, intent: str, context: dict) -> str:
        """Generate a structured answer without LLM."""
        parts = []

        if intent == "system_status":
            snap = context.get("system_snapshot", {})
            orch = snap.get("orchestrator", {})
            hw = snap.get("hardware", {})
            parts.append(f"System Status Summary:")
            parts.append(f"• Events Ingested: {orch.get('events_ingested', 0):,}")
            parts.append(f"• ML Inferences: {orch.get('ml_inferences', 0):,}")
            parts.append(f"• Alerts Routed: {orch.get('alerts_routed', 0):,}")
            parts.append(f"• GPU VRAM: {hw.get('gpu_vram_used_mb', 0):.0f}/{hw.get('gpu_vram_total_mb', 0):.0f} MB")
            parts.append(f"• CPU: {hw.get('cpu_utilization_pct', 0):.1f}%")
            parts.append(f"• LLM TPS: {hw.get('llm_tps', 0):.1f} tokens/sec")

        elif intent == "risk_query":
            top = context.get("top_risky_nodes", [])
            parts.append(f"Top {len(top)} Risky Accounts:")
            for node in top[:10]:
                parts.append(f"• {node['id']}: Risk Score {node['risk']:.4f}")

        elif intent == "fraud_patterns":
            graph_sum = context.get("graph_summary", {})
            parts.append(f"Fraud Pattern Summary:")
            parts.append(f"• Total Fraud Edges: {graph_sum.get('fraud_edges', 0)}")
            parts.append(f"• Total Nodes: {graph_sum.get('nodes', 0)}")

        elif intent == "statistics":
            snap = context.get("system_snapshot", {})
            cb = context.get("circuit_breaker", {})
            parts.append(f"System Statistics:")
            orch = snap.get("orchestrator", {})
            parts.append(f"• Events Processed: {orch.get('events_ingested', 0):,}")
            parts.append(f"• Alerts Generated: {orch.get('alerts_routed', 0):,}")
            parts.append(f"• Frozen Accounts: {cb.get('frozen_count', 0)}")

        else:
            parts.append("Query processed. Key system data retrieved from available sources.")
            snap = context.get("system_snapshot", {})
            if snap:
                orch = snap.get("orchestrator", {})
                parts.append(f"Current pipeline: {orch.get('events_ingested', 0):,} events processed.")

        return "\n".join(parts)

    def snapshot(self) -> dict:
        return {
            "queries_processed": self._query_count,
            "avg_response_ms": round(self._avg_response_ms, 2),
            "llm_available": self._llm is not None,
            "orchestrator_attached": self._orchestrator is not None,
        }

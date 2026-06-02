"""
PayFlow — Natural Language Query Interface
=============================================
Enables analysts to query the fraud detection system using natural
language, powered by Qwen 3.5 4B. Translates questions into
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
import re
import time
import asyncio
from dataclasses import dataclass
from typing import Any, Optional

from config.settings import OLLAMA_CFG

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
8. Pre-Fraud Intel Radar - external OSINT/SOCMINT trends and guarded adaptive playbooks
9. LLM Runtime - the Ollama-hosted Qwen model that writes analyst-facing explanations

When answering:
- Be precise with numbers and account IDs
- Reference specific fraud patterns (UPI_MULE, CIRCULAR_LAUNDERING, VELOCITY_PHISHING, SWIFT_HEIST, etc.)
- Use INR amounts (₹) for currency
- Cite evidence from the system when available
- If data is unavailable, say so clearly
- If the question asks which AI/LLM/model is answering, use LLM Runtime; do not confuse it with ML Models such as XGBoost.
- Never state or imply that Qwen, the LLM Runtime, or the AI assistant has decision authority.
- Decision authority belongs only to PayFlow rules, XGBoost/ML risk scoring, transaction graph evidence, circuit breaker enforcement, audit ledger evidence, and analyst approval gates.

Respond in a structured format with clear sections."""

    INTENT_CLASSIFIER_PROMPT = """Classify the user's query intent. Return ONLY a JSON object:
{
    "intent": "one of: risk_query, account_lookup, fraud_patterns, system_status, explanation, statistics, recommendation, general",
    "entities": {"account_id": "...", "time_range": "...", "fraud_type": "...", "limit": N},
    "data_sources": ["graph", "circuit_breaker", "ml_models", "ledger", "cfr", "verdicts", "metrics", "pre_fraud_intel"]
}

User query: """

    _VALID_INTENTS = {
        "risk_query",
        "account_lookup",
        "fraud_patterns",
        "system_status",
        "explanation",
        "statistics",
        "recommendation",
        "general",
    }

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
        if self._is_model_identity_query(question):
            intent_info = {
                "intent": "system_status",
                "entities": self._extract_entities(question),
                "data_sources": ["metrics"],
            }
        else:
            intent_info = await self._classify_intent(question)
        intent = intent_info.get("intent", "general")
        entities = intent_info.get("entities", {})
        data_sources = intent_info.get("data_sources", [])

        # Gather context from data sources
        context = await self._gather_context(intent, entities, data_sources)
        context.setdefault("llm_runtime", self._llm_runtime_context())

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
            model_used=(
                getattr(self._llm, "_resolved_model", None)
                or getattr(self._llm, "_model", OLLAMA_CFG.model)
            ) if self._llm else "fallback",
        )

    def _extract_entities(self, question: str) -> dict:
        """Cheap entity extraction before asking the LLM classifier."""
        entities: dict[str, Any] = {}
        account_match = re.search(r"\bACC[_-]?[A-Z0-9]+\b", question, flags=re.IGNORECASE)
        if account_match:
            entities["account_id"] = account_match.group(0).upper().replace("-", "_")

        limit_match = re.search(r"\b(?:top|first|latest|last)\s+(\d{1,3})\b", question, flags=re.IGNORECASE)
        if limit_match:
            entities["limit"] = min(int(limit_match.group(1)), 100)

        q = question.lower()
        for fraud_type in (
            "upi_mule",
            "mule",
            "circular_laundering",
            "laundering",
            "velocity_phishing",
            "phishing",
            "swift_heist",
            "swift",
            "structuring",
            "round_tripping",
        ):
            if fraud_type in q:
                entities["fraud_type"] = fraud_type.upper()
                break
        return entities

    def _heuristic_intent(self, question: str) -> dict:
        """Fast deterministic routing for common dashboard questions."""
        q = question.lower()
        entities = self._extract_entities(question)
        if any(w in q for w in ["risk", "score", "dangerous", "suspicious"]):
            return {"intent": "risk_query", "entities": entities, "data_sources": ["ml_models", "graph"]}
        if any(w in q for w in ["account", "acc_", "frozen", "freeze"]):
            return {"intent": "account_lookup", "entities": entities, "data_sources": ["graph", "circuit_breaker", "ml_models"]}
        if any(w in q for w in ["latest", "trend", "osint", "socmint", "digital arrest", "kyc", "loan app", "public signal"]):
            return {"intent": "fraud_patterns", "entities": entities, "data_sources": ["pre_fraud_intel", "graph", "verdicts"]}
        if any(w in q for w in ["pattern", "mule", "laundering", "phishing", "swift", "structuring", "round trip"]):
            return {"intent": "fraud_patterns", "entities": entities, "data_sources": ["graph", "verdicts", "pre_fraud_intel"]}
        if any(w in q for w in ["status", "health", "gpu", "vram", "cpu", "pipeline", "ollama", "qwen"]):
            return {"intent": "system_status", "entities": entities, "data_sources": ["metrics"]}
        if any(w in q for w in ["why", "explain", "reason", "because"]):
            return {"intent": "explanation", "entities": entities, "data_sources": ["verdicts", "ml_models", "graph"]}
        if any(w in q for w in ["how many", "count", "total", "statistics", "stats"]):
            return {"intent": "statistics", "entities": entities, "data_sources": ["graph", "metrics", "circuit_breaker"]}
        if any(w in q for w in ["recommend", "next action", "what should", "mitigate"]):
            return {"intent": "recommendation", "entities": entities, "data_sources": ["verdicts", "ml_models", "circuit_breaker", "pre_fraud_intel"]}
        return {"intent": "general", "entities": entities, "data_sources": ["metrics"]}

    def _llm_runtime_context(self) -> dict[str, Any]:
        """Expose the answer-generation model separately from fraud classifiers."""
        model = (
            getattr(self._llm, "_resolved_model", None)
            or getattr(self._llm, "_model", None)
            or OLLAMA_CFG.model
        ) if self._llm else "fallback"
        return {
            "assistant_model": model,
            "provider": "Ollama" if self._llm else "structured_fallback",
            "role": "bounded analyst-facing explanation and query layer",
            "strict_model_family": OLLAMA_CFG.strict_model_family,
            "decision_authority": "none; the LLM is advisory and explanation-only",
            "authoritative_decision_layers": [
                "rules",
                "XGBoost risk model",
                "transaction graph",
                "circuit breaker",
                "audit ledger",
                "analyst approval gates",
            ],
        }

    def _is_model_identity_query(self, question: str) -> bool:
        """Detect direct questions about the model/runtime so answers stay deterministic."""
        q = question.lower()
        model_terms = ("model", "llm", "qwen", "ai", "assistant", "runtime")
        action_terms = ("which", "what", "who", "using", "powered", "answering")
        return any(term in q for term in model_terms) and any(term in q for term in action_terms)

    def _model_identity_answer(self, context: dict) -> str:
        """Return a fixed runtime disclosure with authority boundaries."""
        runtime = context.get("llm_runtime", {})
        model = runtime.get("assistant_model") or OLLAMA_CFG.model
        provider = runtime.get("provider") or "Ollama"
        authority = runtime.get("authoritative_decision_layers") or [
            "PayFlow rules",
            "XGBoost/ML risk scoring",
            "transaction graph evidence",
            "circuit breaker enforcement",
            "audit ledger evidence",
            "analyst approval gates",
        ]
        return "\n".join([
            f"- Analyst-facing explanations are generated by {model} on {provider}.",
            "- Qwen is a bounded explanation and query copilot; it has no decision authority.",
            "- Authoritative decisions remain with " + ", ".join(authority) + ".",
        ])

    async def _classify_intent(self, question: str) -> dict:
        """Use LLM to classify query intent, with fallback heuristics."""
        heuristic = self._heuristic_intent(question)
        if heuristic["intent"] != "general" or not self._llm:
            return heuristic

        if self._llm:
            try:
                prompt = self.INTENT_CLASSIFIER_PROMPT + question
                if hasattr(self._llm, "chat"):
                    response_data = await asyncio.to_thread(
                        self._llm.chat,
                        [{"role": "user", "content": prompt}],
                        temperature=OLLAMA_CFG.intent_temperature,
                        max_tokens=OLLAMA_CFG.intent_max_tokens,
                        num_ctx=OLLAMA_CFG.num_ctx_status,
                        response_format="json",
                    )
                    response = str(response_data.get("content", ""))
                else:
                    response = await self._llm.generate(
                        prompt,
                        temperature=OLLAMA_CFG.intent_temperature,
                        max_tokens=OLLAMA_CFG.intent_max_tokens,
                        num_ctx=OLLAMA_CFG.num_ctx_status,
                    )
                text = response.strip()
                # Extract JSON from response
                start = text.find("{")
                end = text.rfind("}") + 1
                if start >= 0 and end > start:
                    parsed = json.loads(text[start:end])
                    if not isinstance(parsed, dict):
                        return heuristic
                    intent = str(parsed.get("intent", "general"))
                    if intent not in self._VALID_INTENTS:
                        return heuristic
                    entities = parsed.get("entities", {})
                    if not isinstance(entities, dict):
                        entities = {}
                    parsed["entities"] = {**heuristic.get("entities", {}), **entities}
                    sources = parsed.get("data_sources", [])
                    parsed["data_sources"] = sources if isinstance(sources, list) and sources else heuristic["data_sources"]
                    return parsed
            except Exception as e:
                logger.debug("Intent classification via LLM failed: %s", e)

        return heuristic

    async def _gather_context(
        self, intent: str, entities: dict, data_sources: list[str],
    ) -> dict:
        """Gather relevant data from system components."""
        context: dict[str, Any] = {}

        if not self._orchestrator:
            return context

        orch = self._orchestrator

        if "pre_fraud_intel" in data_sources:
            try:
                from src.intel import get_pre_fraud_intel_service

                context["pre_fraud_intelligence"] = (
                    get_pre_fraud_intel_service().active_context_for_ai()
                )
            except Exception:
                context["pre_fraud_intelligence"] = {
                    "active_playbooks": [],
                    "top_trends": [],
                    "guardrail": "Pre-fraud intelligence context unavailable.",
                }

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
        if self._is_model_identity_query(question):
            return self._model_identity_answer(context)

        if self._llm:
            try:
                fast_intents = {"system_status", "statistics", "risk_query", "account_lookup"}
                context_limit = min(OLLAMA_CFG.context_chars, 4000) if intent in fast_intents else OLLAMA_CFG.context_chars
                answer_max_tokens = (
                    min(OLLAMA_CFG.answer_max_tokens, 256)
                    if intent in fast_intents
                    else OLLAMA_CFG.answer_max_tokens
                )
                answer_num_ctx = (
                    OLLAMA_CFG.num_ctx_status
                    if intent in fast_intents
                    else OLLAMA_CFG.num_ctx_interactive
                )
                # Build context summary for LLM
                context_str = json.dumps(context, indent=2, default=str)[:context_limit]
                length_instruction = (
                    "For this routine dashboard query, answer in at most five compact bullets. "
                    if intent in fast_intents
                    else ""
                )
                prompt = (
                    f"{self.SYSTEM_PROMPT}\n\n"
                    f"System Context:\n{context_str}\n\n"
                    f"User Question: {question}\n\n"
                    "Use only the provided system context. If data is missing, say exactly what is unavailable. "
                    f"{length_instruction}"
                    "Do not use Markdown tables, bold markers, headings, or asterisks; use plain text bullets. "
                    "Keep the answer concise, operational, and evidence-grounded.\n\n"
                    "Provide a clear, data-driven answer:"
                )
                response = await self._llm.generate(
                    prompt,
                    temperature=OLLAMA_CFG.answer_temperature,
                    max_tokens=answer_max_tokens,
                    num_ctx=answer_num_ctx,
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
            intel = context.get("pre_fraud_intelligence", {})
            parts.append(f"Fraud Pattern Summary:")
            parts.append(f"• Total Fraud Edges: {graph_sum.get('fraud_edges', 0)}")
            parts.append(f"• Total Nodes: {graph_sum.get('nodes', 0)}")
            trends = intel.get("top_trends", [])
            if trends:
                parts.append("Pre-Fraud Intel Radar:")
                for trend in trends[:3]:
                    parts.append(
                        f"• {trend.get('title', 'External trend')} "
                        f"(trust {trend.get('trust_score', 0)})"
                    )

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

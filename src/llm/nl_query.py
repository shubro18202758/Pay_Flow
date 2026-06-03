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
from pathlib import Path
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
- Distinguish the user-editable copilot query box from internal model prompts. The global copilot query box is editable; internal prompt templates are not exposed to analysts.

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
        self._doc_context_cache: list[dict[str, Any]] | None = None

    def attach_llm(self, llm_client) -> None:
        self._llm = llm_client

    def attach_orchestrator(self, orchestrator) -> None:
        self._orchestrator = orchestrator

    async def query(
        self,
        question: str,
        *,
        role: str | None = None,
        surface: str | None = None,
        active_tab: str | None = None,
        conversation: list[dict[str, str]] | None = None,
    ) -> NLQueryResult:
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
        context.setdefault(
            "prototype_context",
            self._prototype_context(
                role=role,
                surface=surface,
                active_tab=active_tab,
                conversation=conversation or [],
            ),
        )

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
            sources=list(dict.fromkeys(["prototype_context", *data_sources])),
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
        if any(w in q for w in [
            "search",
            "find",
            "where",
            "page",
            "tab",
            "feature",
            "payflow",
            "prototype",
            "proof of concept",
            "poc",
            "demo",
            "readme",
            "architecture",
            "run locally",
            "local host",
            "localhost",
            "coolify",
            "deployment",
            "rbac",
            "role",
            "union bank",
            "ps3",
            "fund flow",
            "event lab",
            "adaptive event",
            "custom event",
            "fraud event",
            "create event",
            "autonomous report",
        ]):
            entities["_prototype_query"] = True
            return {
                "intent": "general",
                "entities": entities,
                "data_sources": ["metrics", "graph", "verdicts", "pre_fraud_intel"],
            }
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

    def _prototype_context(
        self,
        *,
        role: str | None,
        surface: str | None,
        active_tab: str | None,
        conversation: list[dict[str, str]],
    ) -> dict[str, Any]:
        """Curated PayFlow knowledge base for the global Qwen search/chat layer."""
        try:
            from src.domain.union_bank import (
                ROLE_POLICIES,
                REGULATORY_OBLIGATIONS,
                UNION_BANK_DOMAIN_THRESHOLDS,
                operating_model_for_role,
                role_policy,
            )

            policy = role_policy(role)
            role_profile = {
                "role": policy.role,
                "label": policy.label,
                "domain": policy.domain,
                "summary": policy.summary,
                "tabs": list(policy.tabs),
                "permissions_count": len(policy.permissions),
                "feature_focus": list(policy.feature_focus),
                "decision_authority": policy.decision_authority,
                "escalation_scope": policy.escalation_scope,
                "reporting_line": policy.reporting_line,
                "tool_stack": list(policy.tool_stack),
            }
            role_catalog = [
                {
                    "role": item.role,
                    "label": item.label,
                    "domain": item.domain,
                    "tabs": list(item.tabs),
                    "decision_authority": item.decision_authority,
                }
                for item in ROLE_POLICIES.values()
            ]
            operating_model = operating_model_for_role(policy.role)
            regulatory = list(REGULATORY_OBLIGATIONS)
            thresholds = dict(UNION_BANK_DOMAIN_THRESHOLDS)
        except Exception:
            role_profile = {"role": role or "fraud_analyst", "label": role or "Fraud Analyst"}
            role_catalog = []
            operating_model = {}
            regulatory = []
            thresholds = {}

        return {
            "surface": surface or "global_search_chat",
            "active_tab": active_tab,
            "conversation_tail": conversation[-6:],
            "project": {
                "name": "PayFlow",
                "team": "Team Aryabhata U6TZX1",
                "domain": "Union Bank of India fraud operations and digital-payment fund-flow intelligence",
                "problem_statement": (
                    "PS3: Tracking of Funds within Bank for Fraud Detection. PayFlow maps and visualizes "
                    "end-to-end movement of funds across accounts, branches, products and channels, then uses "
                    "graph analytics, ML risk scoring, heuristics and Qwen explanations to support investigators."
                ),
                "primary_users": [
                    "Fraud Analyst",
                    "SOC Analyst",
                    "SOC L2 Incident Responder",
                    "Threat Hunter",
                    "Transaction Officer",
                    "EFRMS Specialist",
                    "Branch Operations",
                    "AML Analyst",
                    "Compliance Officer",
                    "Principal Officer / MLRO",
                    "Fraud Investigator",
                    "Fraud Committee",
                    "Risk Analyst",
                    "Data Scientist",
                    "Internal Audit",
                    "System Admin",
                ],
                "live_url": "https://u6tzx1.apps.ideahackathon.com/",
                "local_runtime": "FastAPI serves the built React app on the configured PORT, normally http://localhost:8000/.",
            },
            "navigation": [
                {
                    "tab": "pre-fraud-intel",
                    "purpose": "External OSINT/SOCMINT signal fusion, media preview, trend clusters and adaptive playbooks.",
                },
                {
                    "tab": "overview",
                    "purpose": "Fund-flow graph, mule networks, risk distribution, live activity and graph metrics.",
                },
                {
                    "tab": "threat-sim",
                    "purpose": "Adaptive Event Lab for custom fraud-event generation, pipeline visibility, countermeasures and autonomous report output.",
                },
                {
                    "tab": "investigations",
                    "purpose": "Case trace, timeline, graph path, evidence package and analyst decision workflow.",
                },
                {
                    "tab": "intelligence",
                    "purpose": "Model explanations, Qwen query panel, drift, consortium/CFR intelligence and integrity views.",
                },
                {
                    "tab": "analytics",
                    "purpose": "Live charts for risk distribution, typologies, velocity trends, heatmaps and threat summaries.",
                },
                {
                    "tab": "compliance",
                    "purpose": "RBI/FIU/CFR/AML reporting controls, STR/FMR evidence and regulatory handoff surfaces.",
                },
                {
                    "tab": "system",
                    "purpose": "Runtime, SSE, GPU/CPU, health, RBAC, pipeline and operations telemetry.",
                },
            ],
            "core_capabilities": {
                "fund_flow_tracking": "NetworkX transaction graph with account nodes, transaction edges, mule/layering/cycle evidence and live topology.",
                "risk_scoring": "Feature engine plus XGBoost-style classifier, drift monitoring and explainability contributions.",
                "heuristics": "Union Bank domain thresholds for UPI mule splits, CTR/FIU thresholds, RBI fraud reporting and digital-channel controls.",
                "qwen_ai_core": (
                    "Ollama-hosted qwen3.5:4b provides bounded analyst explanations, global search answers, event-lab narratives "
                    "and NL query responses; it does not approve freezes, filings or fraud verdicts."
                ),
                "rbac": "Header-driven X-Payflow-Role guards tabs, API permissions, write actions and role-aware UI controls.",
                "event_lab": "Custom scenario templates generate transactions/auth/interbank events, inject them into pipeline stages and fan out to reports, graphs and countermeasures.",
                "audit": "Append-only ledger and evidence package hashes preserve investigation and decision provenance.",
                "deployment": "Docker/Nixpacks-compatible FastAPI single-port app for Coolify with qwen3.5:4b kept as the target Ollama model.",
                "global_search_copilot": (
                    "The landing-page search field seeds the PayFlow Qwen Search/Copilot overlay. "
                    "The overlay input remains editable after opening, can be cleared with Reset, and only the query submitted with Ask is sent to the backend."
                ),
            },
            "task_shortcuts": {
                "create_custom_fraud_event": (
                    "Open the Adaptive Event Lab tab (internal tab id: threat-sim), pick or configure an Event Chain Creator template, "
                    "preview the generated chain, launch it into the pipeline, then watch Live Processing Pipeline, Pipeline Transparency, "
                    "countermeasure proposals and the autonomous report after verdict completion."
                ),
                "inspect_fund_flow_graph": "Open Fund-Flow Overview (internal tab id: overview) for the live 3D/network graph, mule chains, topology and graph statistics.",
                "ask_qwen": "Use this global PayFlow Qwen Search/Copilot overlay or the Intelligence tab NL query panel.",
                "package_evidence": "Open Investigator Workbench (internal tab id: investigations) after a case exists, then generate evidence packages from case trace.",
                "review_reports": "Use Compliance/FIU Reporting when the selected role has regulatory permissions.",
                "edit_copilot_query": "After the overlay opens from landing search or Ctrl+K, type directly in the bottom prompt box; the seed text is not a locked audit record until Ask is submitted.",
            },
            "role_context": role_profile,
            "role_catalog": role_catalog,
            "operating_model": operating_model,
            "regulatory_context": {
                "obligations": regulatory,
                "thresholds": thresholds,
                "guardrail": "AI output is evidence narration only; Union Bank role gates and policy decide customer-impacting actions.",
            },
            "documentation_digest": self._documentation_digest(),
        }

    def _documentation_digest(self) -> list[dict[str, Any]]:
        """Index key local deliverables without flooding every prompt."""
        if self._doc_context_cache is not None:
            return self._doc_context_cache

        root = Path(__file__).resolve().parents[2]
        candidates = [
            ("README.md", "repository_readme"),
            ("artifacts/payflow-idea-round2-problem-solution-brief.md", "round2_problem_solution_brief"),
            ("artifacts/payflow-d3-labelled-architecture-diagram.md", "d3_architecture_diagram"),
            ("artifacts/payflow-d3-technical-architecture-document.md", "d3_architecture_document"),
        ]
        digest: list[dict[str, Any]] = []
        for relative, kind in candidates:
            path = root / relative
            if not path.exists():
                continue
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue
            headings = re.findall(r"^#{1,4}\s+(.+)$", text, flags=re.MULTILINE)[:16]
            excerpt = re.sub(r"\s+", " ", text).strip()[:900]
            digest.append({
                "path": relative,
                "kind": kind,
                "headings": headings,
                "excerpt": excerpt,
            })

        self._doc_context_cache = digest
        return digest

    async def _classify_intent(self, question: str) -> dict:
        """Use LLM to classify query intent, with fallback heuristics."""
        heuristic = self._heuristic_intent(question)
        if heuristic.get("entities", {}).get("_prototype_query"):
            return heuristic
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

    def _prompt_context(self, context: dict, *, compact: bool) -> dict:
        """Keep Qwen prompts bounded and ordered by answer value."""
        if not compact:
            return context

        snap = context.get("system_snapshot", {})
        orch = snap.get("orchestrator", {}) if isinstance(snap, dict) else {}
        hw = snap.get("hardware", {}) if isinstance(snap, dict) else {}
        pipeline = snap.get("pipeline", {}) if isinstance(snap, dict) else {}
        graph = context.get("graph_summary", {})
        verdicts = context.get("agent_verdicts", {})
        intel = context.get("pre_fraud_intelligence", {})

        compact_context: dict[str, Any] = {
            "prototype_context": context.get("prototype_context", {}),
            "llm_runtime": context.get("llm_runtime", {}),
            "system_metrics": {
                "events_ingested": orch.get("events_ingested"),
                "ml_inferences": orch.get("ml_inferences"),
                "alerts_routed": orch.get("alerts_routed"),
                "throughput_eps": orch.get("events_per_sec"),
                "gpu_vram_used_mb": hw.get("gpu_vram_used_mb"),
                "gpu_vram_total_mb": hw.get("gpu_vram_total_mb"),
                "gpu_utilization_pct": hw.get("gpu_utilization_pct"),
                "cpu_utilization_pct": hw.get("cpu_utilization_pct"),
                "llm_tps": hw.get("llm_tps"),
                "pipeline": pipeline,
            },
            "graph_summary": graph,
            "top_risky_nodes": context.get("top_risky_nodes", [])[:8],
            "ml_models": context.get("ml_models", {}),
            "model_drift": context.get("model_drift", {}),
            "circuit_breaker": context.get("circuit_breaker", {}),
            "agent_verdicts": {
                "total": verdicts.get("total") if isinstance(verdicts, dict) else None,
                "recent": (verdicts.get("recent", []) if isinstance(verdicts, dict) else [])[:3],
            },
            "pre_fraud_intelligence": {
                "active_playbooks": (intel.get("active_playbooks", []) if isinstance(intel, dict) else [])[:4],
                "top_trends": (intel.get("top_trends", []) if isinstance(intel, dict) else [])[:4],
                "guardrail": intel.get("guardrail") if isinstance(intel, dict) else None,
            },
        }
        return compact_context

    async def _generate_answer(self, question: str, intent: str, context: dict) -> str:
        """Generate a natural language answer using LLM or structured fallback."""
        if self._is_model_identity_query(question):
            return self._model_identity_answer(context)

        if self._llm:
            try:
                fast_intents = {"system_status", "statistics", "risk_query", "account_lookup"}
                prototype = context.get("prototype_context", {})
                surface = str(prototype.get("surface") or "").lower()
                is_global_copilot = "copilot" in surface or "global_search" in surface
                compact_prompt = is_global_copilot or intent in fast_intents
                context_limit = (
                    min(OLLAMA_CFG.context_chars, 1800)
                    if is_global_copilot
                    else min(OLLAMA_CFG.context_chars, 4000)
                    if intent in fast_intents
                    else OLLAMA_CFG.context_chars
                )
                answer_max_tokens = (
                    min(OLLAMA_CFG.answer_max_tokens, 180)
                    if is_global_copilot
                    else
                    min(OLLAMA_CFG.answer_max_tokens, 256)
                    if intent in fast_intents
                    else OLLAMA_CFG.answer_max_tokens
                )
                answer_num_ctx = (
                    OLLAMA_CFG.num_ctx_status
                    if is_global_copilot
                    else
                    OLLAMA_CFG.num_ctx_status
                    if intent in fast_intents
                    else OLLAMA_CFG.num_ctx_interactive
                )
                # Build context summary for LLM
                prompt_context = self._prompt_context(context, compact=compact_prompt)
                context_str = json.dumps(prompt_context, indent=2, default=str)[:context_limit]
                length_instruction = (
                    "For this global copilot query, answer in at most six compact bullets or two short paragraphs. "
                    if is_global_copilot
                    else
                    "For this routine dashboard query, answer in at most five compact bullets. "
                    if intent in fast_intents
                    else ""
                )
                valid_tabs = ", ".join(
                    f"{item.get('tab')} ({item.get('purpose')})"
                    for item in prototype.get("navigation", [])
                )
                navigation_guardrail = (
                    "Valid PayFlow tabs are: "
                    f"{valid_tabs}. "
                    "Never invent page names, controls, models, databases, or workflow steps outside the provided context. "
                    if valid_tabs
                    else ""
                )
                copilot_ui_guardrail = (
                    "Critical UI fact: the landing-page search text only seeds the PayFlow Qwen Search/Copilot overlay; "
                    "the bottom copilot query box remains editable after opening, Reset clears it, and only pressing Ask sends the current text to the backend. "
                    "Do not describe the analyst-facing query box as immutable or read-only. "
                    "Internal model prompt templates are separate from this user-editable query input. "
                )
                prompt = (
                    f"{self.SYSTEM_PROMPT}\n\n"
                    f"{copilot_ui_guardrail}\n"
                    f"System Context:\n{context_str}\n\n"
                    f"User Question: {question}\n\n"
                    "Use only the provided system context. If data is missing, say exactly what is unavailable. "
                    "For prototype navigation questions, tell the user which PayFlow page, tab, role, or control to use. "
                    "For search or copilot UI behavior questions, follow prototype_context.core_capabilities.global_search_copilot exactly. "
                    "For proof-of-concept questions, ground the answer in PS3, Union Bank operating context, and the documented prototype features. "
                    f"{navigation_guardrail}"
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
                return self._clean_llm_answer(response)
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
            prototype = context.get("prototype_context", {})
            project = prototype.get("project", {})
            nav = prototype.get("navigation", [])
            parts.append(f"{project.get('name', 'PayFlow')} is the PS3 fund-flow fraud intelligence prototype for Union Bank operations.")
            if nav:
                parts.append("Use the main tabs this way:")
                for item in nav[:5]:
                    parts.append(f"• {item.get('tab')}: {item.get('purpose')}")
            snap = context.get("system_snapshot", {})
            if snap:
                orch = snap.get("orchestrator", {})
                parts.append(f"Current pipeline: {orch.get('events_ingested', 0):,} events processed.")

        return "\n".join(parts)

    def _clean_llm_answer(self, response: str) -> str:
        """Normalize common markdown artifacts before UI rendering."""
        text = response.strip().replace("**", "")
        text = re.sub(r"(?m)^\s*\*\s+", "- ", text)
        text = re.sub(r"(?m)^(\s*)\*\s{2,}", r"\1- ", text)
        return text

    def snapshot(self) -> dict:
        return {
            "queries_processed": self._query_count,
            "avg_response_ms": round(self._avg_response_ms, 2),
            "llm_available": self._llm is not None,
            "orchestrator_attached": self._orchestrator is not None,
        }

"""
PayFlow -- Investigator Agent Tool Schemas & Executor
======================================================
OpenAI-compatible function-calling schemas and a ToolExecutor that dispatches
tool calls to live PayFlow subsystems (TransactionGraph, FeatureEngine,
AuditLedger, CircuitBreaker).

The schemas follow the OpenAI ``tools`` format so that Ollama's Qwen 3.5 9B
model can invoke them via its native function-calling capability.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


# ── Data Structures ──────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ToolCall:
    """A single tool invocation parsed from the LLM response."""
    name: str
    arguments: dict

    def to_dict(self) -> dict:
        return {"name": self.name, "arguments": self.arguments}


@dataclass(frozen=True)
class ToolResult:
    """Result of executing a tool call against a PayFlow subsystem."""
    tool_name: str
    success: bool
    data: dict
    error: str | None = None

    def to_dict(self) -> dict:
        result = {
            "tool_name": self.tool_name,
            "success": self.success,
            "data": self.data,
        }
        if self.error:
            result["error"] = self.error
        return result


# ── Tool Schemas (OpenAI-compatible) ─────────────────────────────────────────

TOOL_SCHEMAS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "query_graph_database",
            "description": (
                "Query the transaction graph for a node's neighborhood, "
                "connections, and structural patterns. Returns k-hop subgraph "
                "statistics, mule network findings, and cycle detection results."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "node_id": {
                        "type": "string",
                        "description": "Account ID to investigate",
                    },
                    "k_hops": {
                        "type": "integer",
                        "description": "Neighborhood radius (1-5)",
                        "default": 3,
                    },
                    "include_patterns": {
                        "type": "boolean",
                        "description": "Run mule and cycle detection on the subgraph",
                        "default": True,
                    },
                },
                "required": ["node_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_ml_feature_analysis",
            "description": (
                "Retrieve the full 30-dimensional ML feature vector for a "
                "transaction, including velocity metrics, behavioral deviation "
                "scores, and text anomaly indicators. Returns feature names and "
                "values ranked by absolute importance."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "txn_id": {
                        "type": "string",
                        "description": "Transaction ID to analyze",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of top features to return",
                        "default": 10,
                    },
                },
                "required": ["txn_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_audit_logs",
            "description": (
                "Read recent blockchain audit ledger entries filtered by event "
                "type. Returns tamper-evident log entries with block hashes, "
                "timestamps, and payloads."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "event_type": {
                        "type": "string",
                        "enum": [
                            "ALERT", "INVESTIGATION", "CIRCUIT_BREAKER",
                            "ZKP_VERIFICATION", "MODEL_UPDATE", "SYSTEM_STATE",
                        ],
                        "description": "Filter by event category",
                    },
                    "node_id": {
                        "type": "string",
                        "description": "Optional: filter entries involving this account ID",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of entries to return",
                        "default": 20,
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_node_freeze_status",
            "description": (
                "Check if an account is currently frozen by the circuit breaker. "
                "Returns freeze order details including consensus scores and "
                "TTL remaining."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "node_id": {
                        "type": "string",
                        "description": "Account ID to check",
                    },
                },
                "required": ["node_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "analyze_unstructured_data",
            "description": (
                "Invoke the NLU sub-agent to analyze unstructured textual data "
                "associated with a transaction — email metadata, user-agent "
                "strings, device fingerprints, beneficiary names, and interbank "
                "messaging text (SWIFT/NEFT). Returns semantic anomaly findings "
                "including social engineering indicators, linguistic "
                "inconsistencies, and device/channel anomalies."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "txn_id": {
                        "type": "string",
                        "description": "Transaction ID whose unstructured data to analyze",
                    },
                    "sender_id": {
                        "type": "string",
                        "description": "Sender account ID",
                    },
                    "receiver_id": {
                        "type": "string",
                        "description": "Receiver account ID",
                    },
                },
                "required": ["txn_id", "sender_id", "receiver_id"],
            },
        },
    },
]


# ── Tool Executor ────────────────────────────────────────────────────────────

class ToolExecutor:
    """
    Dispatches tool calls to live PayFlow subsystems.

    All subsystem references are optional — if a subsystem is not available,
    the executor returns a graceful error ToolResult rather than crashing.

    Usage::

        executor = ToolExecutor(
            transaction_graph=graph,
            feature_engine=engine,
            audit_ledger=ledger,
            circuit_breaker=breaker,
        )
        result = await executor.execute(ToolCall("query_graph_database", {"node_id": "ACC001"}))
    """

    def __init__(
        self,
        transaction_graph=None,
        feature_engine=None,
        audit_ledger=None,
        circuit_breaker=None,
        unstructured_agent=None,
    ) -> None:
        self._graph = transaction_graph
        self._feature_engine = feature_engine
        self._ledger = audit_ledger
        self._breaker = circuit_breaker
        self._unstructured_agent = unstructured_agent
        self._dispatch = {
            "query_graph_database": self._query_graph_database,
            "get_ml_feature_analysis": self._get_ml_feature_analysis,
            "read_audit_logs": self._read_audit_logs,
            "check_node_freeze_status": self._check_node_freeze_status,
            "analyze_unstructured_data": self._analyze_unstructured_data,
        }

    async def execute(self, tool_call: ToolCall) -> ToolResult:
        """
        Execute a single tool call and return the result.

        Unknown tool names or execution errors are returned as failed
        ToolResults — never raised as exceptions.
        """
        handler = self._dispatch.get(tool_call.name)
        if handler is None:
            return ToolResult(
                tool_name=tool_call.name,
                success=False,
                data={},
                error=f"Unknown tool: {tool_call.name}",
            )

        try:
            t0 = time.perf_counter()
            data = await handler(tool_call.arguments)
            elapsed = (time.perf_counter() - t0) * 1000
            data["_execution_ms"] = round(elapsed, 2)
            return ToolResult(
                tool_name=tool_call.name,
                success=True,
                data=data,
            )
        except Exception as exc:
            logger.warning("Tool %s failed: %s", tool_call.name, exc)
            return ToolResult(
                tool_name=tool_call.name,
                success=False,
                data={},
                error=str(exc),
            )

    # ── Tool Implementations ──────────────────────────────────────────────

    async def _query_graph_database(self, args: dict) -> dict:
        """Query the transaction graph for node neighborhood and patterns."""
        if self._graph is None:
            return {"error": "TransactionGraph not available", "node_id": args.get("node_id")}

        node_id = args["node_id"]
        k_hops = args.get("k_hops", 3)
        include_patterns = args.get("include_patterns", True)

        g = self._graph.graph
        if not g.has_node(node_id):
            return {
                "node_id": node_id,
                "found": False,
                "message": f"Node {node_id} not found in transaction graph",
            }

        # Extract k-hop subgraph
        subgraph = self._graph._extract_k_hop_subgraph([node_id], min(k_hops, 5))
        node_data = g.nodes.get(node_id, {})

        result = {
            "node_id": node_id,
            "found": True,
            "node_attributes": {
                "first_seen": node_data.get("first_seen", 0),
                "last_seen": node_data.get("last_seen", 0),
                "txn_count": node_data.get("txn_count", 0),
            },
            "subgraph": {
                "nodes": subgraph.number_of_nodes(),
                "edges": subgraph.number_of_edges(),
                "k_hops": k_hops,
            },
            "connections": {
                "in_degree": g.in_degree(node_id),
                "out_degree": g.out_degree(node_id),
                "distinct_senders": len(set(u for u, _ in g.in_edges(node_id))),
                "distinct_receivers": len(set(v for _, v in g.out_edges(node_id))),
            },
        }

        if include_patterns:
            # Mule detection around node
            mule_finding = self._graph._mule_detector.detect_around_node(
                subgraph, node_id, self._graph._latest_timestamp,
            )
            # Cycle detection around node
            cycles = self._graph._cycle_detector.detect_around_nodes(
                subgraph, [node_id], self._graph._latest_timestamp, k_hops=k_hops,
            )
            result["patterns"] = {
                "mule_network_detected": mule_finding is not None,
                "mule_distinct_senders": mule_finding.distinct_senders if mule_finding else 0,
                "cycles_found": len(cycles),
                "cycle_lengths": [c.cycle_length for c in cycles[:5]],
            }

        return result

    async def _get_ml_feature_analysis(self, args: dict) -> dict:
        """Retrieve ML feature vector for a transaction."""
        txn_id = args["txn_id"]
        top_k = args.get("top_k", 10)

        if self._feature_engine is None:
            return {"txn_id": txn_id, "error": "FeatureEngine not available"}

        # Look up cached features if available
        if hasattr(self._feature_engine, "_feature_cache"):
            cached = self._feature_engine._feature_cache.get(txn_id)
            if cached is not None:
                features, names = cached
                indices = np.argsort(np.abs(features))[::-1][:top_k]
                ranked = {names[i]: round(float(features[i]), 4) for i in indices}
                return {
                    "txn_id": txn_id,
                    "feature_dim": len(features),
                    "top_features": ranked,
                }

        return {"txn_id": txn_id, "error": "Features not found in cache"}

    async def _read_audit_logs(self, args: dict) -> dict:
        """Read recent audit ledger entries."""
        if self._ledger is None:
            return {"error": "AuditLedger not available"}

        from src.blockchain.models import EventType

        limit = args.get("limit", 20)
        event_type_str = args.get("event_type")
        node_id = args.get("node_id")

        event_type = None
        if event_type_str:
            try:
                event_type = EventType[event_type_str]
            except KeyError:
                return {"error": f"Unknown event type: {event_type_str}"}

        entries = await self._ledger.get_recent_blocks(
            event_type=event_type, limit=limit,
        )

        # Filter by node_id if specified
        if node_id:
            filtered = []
            for entry in entries:
                payload = entry.get("payload", {})
                payload_str = json.dumps(payload)
                if node_id in payload_str:
                    filtered.append(entry)
            entries = filtered

        return {
            "entries_count": len(entries),
            "filter": {
                "event_type": event_type_str,
                "node_id": node_id,
                "limit": limit,
            },
            "entries": entries[:limit],
        }

    async def _check_node_freeze_status(self, args: dict) -> dict:
        """Check circuit breaker freeze status for a node."""
        node_id = args["node_id"]

        if self._breaker is None:
            return {"node_id": node_id, "error": "CircuitBreaker not available"}

        is_frozen = self._breaker.is_frozen(node_id)
        result = {
            "node_id": node_id,
            "is_frozen": is_frozen,
        }

        if is_frozen:
            order = self._breaker._frozen_nodes.get(node_id)
            if order:
                ttl_remaining = max(
                    0,
                    order.ttl_seconds - (time.time() - order.freeze_timestamp),
                )
                result["freeze_details"] = {
                    "trigger_txn_id": order.trigger_txn_id,
                    "ml_risk_score": round(order.ml_risk_score, 4),
                    "gnn_risk_score": round(order.gnn_risk_score, 4),
                    "consensus_score": round(order.consensus_score, 4),
                    "reason": order.reason,
                    "freeze_timestamp": order.freeze_timestamp,
                    "ttl_remaining_seconds": round(ttl_remaining, 1),
                }

        return result

    async def _analyze_unstructured_data(self, args: dict) -> dict:
        """Invoke the NLU sub-agent for unstructured data analysis."""
        txn_id = args.get("txn_id", "unknown")
        sender_id = args.get("sender_id", "unknown")
        receiver_id = args.get("receiver_id", "unknown")

        if self._unstructured_agent is None:
            return {
                "txn_id": txn_id,
                "error": "UnstructuredAnalysisAgent not available",
            }

        from src.llm.unstructured_models import UnstructuredPayload

        # Build payload from available data — look up supplementary fields
        # from the unstructured data store if available
        payload_kwargs: dict = {
            "txn_id": txn_id,
            "sender_id": sender_id,
            "receiver_id": receiver_id,
        }

        # If the unstructured agent has a data store, query it
        if hasattr(self._unstructured_agent, "_data_store"):
            store = self._unstructured_agent._data_store
            if store and txn_id in store:
                stored = store[txn_id]
                if isinstance(stored, dict):
                    for key in (
                        "sender_name", "receiver_name", "remittance_info",
                        "email_subject", "email_sender", "email_body_snippet",
                        "user_agent", "device_fingerprint",
                        "previous_device_fingerprint", "ip_geo",
                        "swift_message", "neft_narration",
                    ):
                        if key in stored:
                            payload_kwargs[key] = stored[key]

        payload = UnstructuredPayload(**payload_kwargs)
        result = await self._unstructured_agent.analyze(payload)
        return result.to_dict()

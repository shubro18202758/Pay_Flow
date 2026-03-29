"""
PayFlow — Transaction Graph Builder
=======================================
Dynamic directed multi-graph construction from streaming transaction data
with parallel pattern detection.

Architecture:
    IngestionPipeline                        AlertRouter
         │                                        │
    EventBatch                              AlertPayload
         │                                        │
         ▼                                        ▼
    TransactionGraph.ingest()         TransactionGraph.investigate()
         │                                        │
         ├── nx.MultiDiGraph                     ├── k-hop subgraph extraction
         │   (nodes = accounts,                  ├── MuleDetector.detect_around_node()
         │    edges = transactions)              └── CycleDetector.detect_around_nodes()
         │                                        │
         └── periodic scan_patterns()            └── InvestigationResult
              ├── MuleDetector.detect()
              └── CycleDetector.detect()

Graph Data Model:
    Nodes: account IDs (sender_id / receiver_id strings)
        Attributes: first_seen (int), last_seen (int), txn_count (int)
    Edges: directed, keyed by txn_id
        Attributes: timestamp (int), amount_paisa (int), channel (int),
                    fraud_label (int)

Memory Estimate:
    Node : ~200 B × 100 K nodes  ≈  20 MB
    Edge : ~500 B × 1 M edges    ≈ 500 MB
    Worst-case total: ~520 MB CPU RAM (VRAM not consumed).

Temporal Pruning:
    Edges older than ``edge_ttl_seconds`` (default 7 days) are pruned
    periodically to bound memory growth (matches VelocityTracker retention).

Parallel Processing:
    Heavy algorithmic scans (mule / cycle detection) are offloaded to the
    default thread-pool via ``asyncio.to_thread()`` so the event loop and
    ingestion pipeline are never blocked.  An ``asyncio.Lock`` serialises
    graph access defensively.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import NamedTuple

import networkx as nx
from networkx.algorithms.community import greedy_modularity_communities

from config.settings import FRAUD_THRESHOLDS
from src.graph.algorithms import (
    CentralityAnalyzer,
    CycleDetector,
    IntermediaryNode,
    MuleDetector,
    MuleNetwork,
    TransactionCycle,
)
from src.ingestion.schemas import EventBatch, Transaction
from src.ml.models.alert_router import AlertPayload

logger = logging.getLogger(__name__)

_7_DAYS = 7 * 24 * 3600  # 604 800 seconds


# ── Configuration ───────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class GraphConfig:
    """Tunable parameters for the transaction graph."""
    edge_ttl_seconds: int = _7_DAYS           # prune edges older than 7 days
    prune_interval_batches: int = 100         # run pruning every N batches
    scan_interval_batches: int = 50           # run full pattern scan every N batches
    min_distinct_senders_mule: int = 5        # mule detection convergence threshold
    k_hop_investigation: int = 3              # subgraph extraction radius
    max_cycle_results: int = 100              # cap on cycle detection results


# ── Investigation Result ────────────────────────────────────────────────────────

class InvestigationResult(NamedTuple):
    """Result of investigating an AlertPayload through graph analysis."""
    txn_id: str
    sender_id: str
    receiver_id: str
    subgraph_nodes: int
    subgraph_edges: int
    mule_findings: list[MuleNetwork]
    cycle_findings: list[TransactionCycle]
    centrality_findings: list[IntermediaryNode]
    investigation_ms: float
    gnn_risk_score: float = -1.0   # -1.0 = GNN not available / not run


# ── Metrics ─────────────────────────────────────────────────────────────────────

@dataclass
class GraphMetrics:
    """Transaction graph health and performance counters."""
    nodes: int = 0
    edges: int = 0
    batches_ingested: int = 0
    transactions_added: int = 0
    edges_pruned: int = 0
    mule_detections: int = 0
    cycle_detections: int = 0
    centrality_detections: int = 0
    scan_runs: int = 0
    investigations_completed: int = 0
    total_scan_ms: float = 0.0
    total_investigation_ms: float = 0.0
    _start_time: float = field(default_factory=time.monotonic)

    @property
    def uptime_sec(self) -> float:
        return time.monotonic() - self._start_time

    def snapshot(self) -> dict:
        return {
            "nodes": self.nodes,
            "edges": self.edges,
            "batches": self.batches_ingested,
            "txns_added": self.transactions_added,
            "pruned": self.edges_pruned,
            "mule_detections": self.mule_detections,
            "cycle_detections": self.cycle_detections,
            "centrality_detections": self.centrality_detections,
            "scans": self.scan_runs,
            "investigations": self.investigations_completed,
            "avg_scan_ms": round(
                self.total_scan_ms / max(self.scan_runs, 1), 2,
            ),
            "uptime_sec": round(self.uptime_sec, 1),
        }


# ── Transaction Graph ───────────────────────────────────────────────────────────

class TransactionGraph:
    """
    Dynamic directed multi-graph built from streaming transactions.

    Dual-role integration:
        1. Pipeline Consumer : ``pipeline.add_consumer(graph.ingest)``
        2. Alert Investigator: ``router.register_graph_consumer(graph.investigate)``

    Usage:
        graph = TransactionGraph()

        # Wire to ingestion pipeline
        pipeline.add_consumer(graph.ingest)

        # Wire to alert router
        router.register_graph_consumer(graph.investigate)

        # On-demand pattern scan
        mules, cycles = await graph.scan_patterns()
    """

    def __init__(
        self,
        config: GraphConfig | None = None,
        gnn_scorer=None,
        audit_ledger=None,
        circuit_breaker=None,
    ) -> None:
        self._cfg = config or GraphConfig()
        self._graph = nx.MultiDiGraph()
        self._mule_detector = MuleDetector(
            min_distinct_senders=self._cfg.min_distinct_senders_mule,
        )
        self._cycle_detector = CycleDetector()
        self._centrality_analyzer = CentralityAnalyzer()
        self._gnn_scorer = gnn_scorer          # optional GNNScorer instance
        self._audit_ledger = audit_ledger      # optional AuditLedger instance
        self._circuit_breaker = circuit_breaker  # optional CircuitBreaker instance
        self.metrics = GraphMetrics()
        self._latest_timestamp: int = 0
        self._lock = asyncio.Lock()
        self._community_map: dict[str, int] = {}
        self._last_community_detect: float = 0

    # ── Pipeline Consumer Interface (EventBatch) ─────────────────────────

    async def ingest(self, batch: EventBatch) -> None:
        """
        Async consumer callable — conforms to the
        ``Consumer = Callable[[EventBatch], Coroutine[…]]`` protocol.

        Adds all transactions from the batch to the graph.  Periodically
        prunes stale edges and runs pattern detection scans (offloaded
        to a background thread).
        """
        if not batch.transactions:
            return

        async with self._lock:
            # Graph mutations are fast — run synchronously
            self._add_transactions(batch.transactions)
            self.metrics.batches_ingested += 1

            # Broadcast graph update to dashboard (best-effort)
            try:
                from src.api.events import EventBroadcaster
                from src.ingestion.schemas import FraudPattern
                fraud_label_names = {e.value: e.name for e in FraudPattern}
                community_map = self._community_map  # use cached, don't recompute per-batch
                g = self._graph
                seen_nodes: set[str] = set()
                cyto_nodes = []
                cyto_edges = []
                for txn in batch.transactions:
                    for nid in (txn.sender_id, txn.receiver_id):
                        if nid not in seen_nodes and g.has_node(nid):
                            seen_nodes.add(nid)
                            nd = g.nodes[nid]
                            # Compute per-node enrichment inline
                            fec = 0
                            tvp = 0
                            sk: set = set()
                            for _, _, k, d in g.out_edges(nid, keys=True, data=True):
                                sk.add(k)
                                if d.get("fraud_label", 0) > 0:
                                    fec += 1
                                tvp += d.get("amount_paisa", 0)
                            for _, _, k, d in g.in_edges(nid, keys=True, data=True):
                                if k not in sk:
                                    if d.get("fraud_label", 0) > 0:
                                        fec += 1
                                    tvp += d.get("amount_paisa", 0)

                            status = "normal"
                            if self._circuit_breaker and self._circuit_breaker.is_frozen(nid):
                                status = "frozen"

                            cyto_nodes.append({"data": {
                                "id": nid,
                                "txn_count": nd.get("txn_count", 1),
                                "status": status,
                                "account_type": nd.get("account_type", "UNKNOWN"),
                                "community_id": community_map.get(nid, -1),
                                "fraud_edge_count": fec,
                                "total_volume_paisa": tvp,
                                "first_seen": nd.get("first_seen", 0),
                                "last_seen": nd.get("last_seen", 0),
                            }})
                    fl = int(txn.fraud_label)
                    cyto_edges.append({"data": {
                        "id": txn.txn_id,
                        "source": txn.sender_id,
                        "target": txn.receiver_id,
                        "amount_paisa": txn.amount_paisa,
                        "channel": int(txn.channel),
                        "fraud_label": fl,
                        "fraud_label_name": fraud_label_names.get(fl, "NONE"),
                        "timestamp": txn.timestamp,
                        "device_fingerprint": txn.device_fingerprint,
                    }})
                await EventBroadcaster.get().publish("graph", {
                    "type": "batch_update",
                    "nodes": cyto_nodes,
                    "edges": cyto_edges,
                })
            except Exception:
                pass

            # Periodic pruning
            if self.metrics.batches_ingested % self._cfg.prune_interval_batches == 0:
                pruned = self._prune_stale_edges()
                self.metrics.edges_pruned += pruned

            # Periodic pattern scan — offload to thread pool
            if self.metrics.batches_ingested % self._cfg.scan_interval_batches == 0:
                mules, cycles, intermediaries = await asyncio.to_thread(self._scan_sync)
                self.metrics.mule_detections += len(mules)
                self.metrics.cycle_detections += len(cycles)
                self.metrics.centrality_detections += len(intermediaries)
                self.metrics.scan_runs += 1
                if mules or cycles or intermediaries:
                    logger.info(
                        "Pattern scan [batch %d]: %d mule networks, %d cycles, "
                        "%d intermediary nodes",
                        self.metrics.batches_ingested, len(mules), len(cycles),
                        len(intermediaries),
                    )

            # Refresh live counters
            self.metrics.nodes = self._graph.number_of_nodes()
            self.metrics.edges = self._graph.number_of_edges()

    def _add_transactions(self, transactions: list[Transaction]) -> None:
        """Synchronous graph mutation.  Adds nodes and edges from transactions."""
        g = self._graph
        for txn in transactions:
            ts = txn.timestamp
            if ts > self._latest_timestamp:
                self._latest_timestamp = ts

            # Add / update sender node
            if g.has_node(txn.sender_id):
                nd = g.nodes[txn.sender_id]
                nd["last_seen"] = ts
                nd["txn_count"] = nd.get("txn_count", 0) + 1
            else:
                acct_type = txn.sender_account_type.name if hasattr(txn.sender_account_type, 'name') else str(txn.sender_account_type)
                g.add_node(txn.sender_id, first_seen=ts, last_seen=ts, txn_count=1, account_type=acct_type)

            # Add / update receiver node
            if g.has_node(txn.receiver_id):
                nd = g.nodes[txn.receiver_id]
                nd["last_seen"] = ts
                nd["txn_count"] = nd.get("txn_count", 0) + 1
            else:
                acct_type = txn.receiver_account_type.name if hasattr(txn.receiver_account_type, 'name') else str(txn.receiver_account_type)
                g.add_node(txn.receiver_id, first_seen=ts, last_seen=ts, txn_count=1, account_type=acct_type)

            # Directed edge keyed by txn_id
            g.add_edge(
                txn.sender_id,
                txn.receiver_id,
                key=txn.txn_id,
                timestamp=ts,
                amount_paisa=txn.amount_paisa,
                channel=int(txn.channel),
                fraud_label=int(txn.fraud_label),
                device_fingerprint=txn.device_fingerprint,
            )

        self.metrics.transactions_added += len(transactions)

    # ── Alert Investigation Interface (AlertPayload) ─────────────────────

    async def investigate(self, payload: AlertPayload) -> None:
        """
        Async callback — conforms to the ``GraphConsumer`` protocol.

        Extracts a k-hop subgraph around the sender and receiver of the
        flagged transaction, runs both pattern detectors, and logs findings.
        """
        async with self._lock:
            result = await asyncio.to_thread(self._investigate_sync, payload)

        self.metrics.investigations_completed += 1
        self.metrics.total_investigation_ms += result.investigation_ms

        # Broadcast graph investigation completion to frontend (per-txn stage)
        try:
            from src.api.events import EventBroadcaster
            await EventBroadcaster.get().publish("pipeline", {
                "type": "stage_complete",
                "stage": "graph_investigated",
                "txn_id": payload.txn_id,
                "duration_ms": round(result.investigation_ms, 1),
                "subgraph_nodes": result.subgraph_nodes,
                "subgraph_edges": result.subgraph_edges,
                "mule_findings": len(result.mule_findings),
                "cycle_findings": len(result.cycle_findings),
                "centrality_findings": len(result.centrality_findings),
                "gnn_risk_score": round(result.gnn_risk_score, 4),
            })
        except Exception:
            pass

        # Anchor investigation result to audit ledger (non-blocking)
        if self._audit_ledger is not None:
            try:
                await self._audit_ledger.anchor_investigation(result)
            except Exception as exc:
                logger.warning(
                    "Ledger anchoring failed for investigation %s: %s",
                    payload.txn_id, exc,
                )

        # Feed circuit breaker with investigation results (non-blocking)
        if self._circuit_breaker is not None:
            try:
                await self._circuit_breaker.on_investigation(result)
            except Exception as exc:
                logger.warning(
                    "Circuit breaker feed failed for investigation %s: %s",
                    payload.txn_id, exc,
                )

        if result.mule_findings or result.cycle_findings or result.centrality_findings:
            logger.warning(
                "Investigation %s: %d mule patterns, %d cycles, %d intermediaries "
                "in %d-node subgraph (%.1f ms)",
                payload.txn_id,
                len(result.mule_findings),
                len(result.cycle_findings),
                len(result.centrality_findings),
                result.subgraph_nodes,
                result.investigation_ms,
            )

    def _investigate_sync(self, payload: AlertPayload) -> InvestigationResult:
        """Synchronous investigation logic — offloaded via ``asyncio.to_thread()``."""
        t0 = time.perf_counter()

        center_nodes = [payload.sender_id, payload.receiver_id]
        subgraph = self._extract_k_hop_subgraph(
            center_nodes, self._cfg.k_hop_investigation,
        )

        # Mule check on both sender and receiver
        mules: list[MuleNetwork] = []
        for nid in center_nodes:
            if subgraph.has_node(nid):
                finding = self._mule_detector.detect_around_node(
                    subgraph, nid, payload.timestamp,
                )
                if finding is not None:
                    mules.append(finding)

        # Cycle check in the local subgraph
        cycles = self._cycle_detector.detect_around_nodes(
            subgraph, center_nodes, payload.timestamp,
            k_hops=self._cfg.k_hop_investigation,
        )

        # Centrality check on sender and receiver (from cached full-graph data)
        centrality_findings: list[IntermediaryNode] = []
        for nid in center_nodes:
            finding = self._centrality_analyzer.detect_node(self._graph, nid)
            if finding is not None:
                centrality_findings.append(finding)

        # GNN topological risk score (optional)
        gnn_score = -1.0
        if self._gnn_scorer is not None and self._gnn_scorer.is_loaded:
            try:
                gnn_result = self._gnn_scorer.score_subgraph(
                    subgraph, center_nodes, payload.timestamp,
                )
                gnn_score = gnn_result.risk_score
                logger.debug(
                    "GNN score for %s: %.4f (%d nodes, %d edges, %.1f ms)",
                    payload.txn_id, gnn_score, gnn_result.node_count,
                    gnn_result.edge_count, gnn_result.inference_ms,
                )
            except Exception as exc:
                logger.warning("GNN scoring failed for %s: %s", payload.txn_id, exc)

        elapsed = (time.perf_counter() - t0) * 1000
        return InvestigationResult(
            txn_id=payload.txn_id,
            sender_id=payload.sender_id,
            receiver_id=payload.receiver_id,
            subgraph_nodes=subgraph.number_of_nodes(),
            subgraph_edges=subgraph.number_of_edges(),
            mule_findings=mules,
            cycle_findings=cycles,
            centrality_findings=centrality_findings,
            investigation_ms=elapsed,
            gnn_risk_score=gnn_score,
        )

    # ── Public Mule Detection API ──────────────────────────────────────

    def detect_mule_around(self, node_id: str) -> list[str]:
        """
        Public API for mule detection around a single node.

        Used by the agent circuit breaker listener for cascade freezing —
        identifies suspected mule accounts adjacent to a confirmed fraud node.

        Args:
            node_id: Account ID to check for mule-star topology.

        Returns:
            List of mule node IDs detected (empty if none or node unknown).
        """
        if not self._graph.has_node(node_id):
            return []
        try:
            result = self._mule_detector.detect_around_node(
                self._graph, node_id, self._latest_timestamp,
            )
            if result is None:
                return []
            return [result.mule_node]
        except Exception as exc:
            logger.warning(
                "Mule detection around %s failed: %s", node_id, exc,
            )
            return []

    # ── Pattern Scanning ─────────────────────────────────────────────────

    async def scan_patterns(
        self,
    ) -> tuple[list[MuleNetwork], list[TransactionCycle], list[IntermediaryNode]]:
        """
        Run a full-graph pattern scan.  Offloaded to the thread pool so
        the event loop is never blocked.

        Returns:
            (mule_findings, cycle_findings, centrality_findings)
        """
        async with self._lock:
            return await asyncio.to_thread(self._scan_sync)

    def _scan_sync(
        self,
    ) -> tuple[list[MuleNetwork], list[TransactionCycle], list[IntermediaryNode]]:
        """Synchronous scan — called via ``asyncio.to_thread()``."""
        t0 = time.perf_counter()

        mules = self._mule_detector.detect(self._graph, self._latest_timestamp)
        cycles = self._cycle_detector.detect(
            self._graph, self._latest_timestamp,
            max_results=self._cfg.max_cycle_results,
        )
        intermediaries = self._centrality_analyzer.detect(
            self._graph, self._latest_timestamp,
        )

        elapsed = (time.perf_counter() - t0) * 1000
        self.metrics.total_scan_ms += elapsed
        return mules, cycles, intermediaries

    # ── Temporal Pruning ─────────────────────────────────────────────────

    def _prune_stale_edges(self) -> int:
        """
        Remove edges older than ``edge_ttl_seconds`` and orphaned nodes.

        Returns:
            Number of edges removed.
        """
        cutoff = self._latest_timestamp - self._cfg.edge_ttl_seconds
        stale: list[tuple[str, str, str]] = []

        for u, v, key, data in self._graph.edges(keys=True, data=True):
            if data.get("timestamp", 0) < cutoff:
                stale.append((u, v, key))

        for u, v, key in stale:
            self._graph.remove_edge(u, v, key=key)

        # Remove orphaned nodes (no remaining edges)
        orphans = [n for n in self._graph.nodes() if self._graph.degree(n) == 0]
        self._graph.remove_nodes_from(orphans)

        if stale:
            logger.debug(
                "Pruned %d stale edges and %d orphan nodes", len(stale), len(orphans),
            )
        return len(stale)

    # ── Subgraph Extraction ──────────────────────────────────────────────

    def _extract_k_hop_subgraph(
        self,
        center_nodes: list[str],
        k: int,
    ) -> nx.MultiDiGraph:
        """
        Extract a bidirectional k-hop neighbourhood around the given nodes.

        Returns a *copy* (not a view) of the subgraph so that algorithm
        execution is safe from concurrent graph mutation.
        """
        neighbourhood: set[str] = set()
        for node in center_nodes:
            if not self._graph.has_node(node):
                continue
            # Forward k-hop (outgoing edges)
            fwd = nx.ego_graph(self._graph, node, radius=k)
            neighbourhood.update(fwd.nodes())
            # Backward k-hop (incoming edges)
            rev = nx.ego_graph(self._graph.reverse(copy=False), node, radius=k)
            neighbourhood.update(rev.nodes())

        return self._graph.subgraph(neighbourhood).copy()

    # ── Community Detection & Node Enrichment ──────────────────────────

    def detect_communities(self) -> dict[str, int]:
        """Detect communities using greedy modularity. Throttled to max once per 30s."""
        now = time.monotonic()
        if now - self._last_community_detect < 30.0 and self._community_map:
            return self._community_map

        g = self._graph
        if g.number_of_nodes() < 2:
            return {}

        try:
            undirected = g.to_undirected(as_view=True)
            # greedy_modularity_communities returns a list of frozensets
            communities = greedy_modularity_communities(undirected, best_n=12, cutoff=1)
            self._community_map = {}
            for idx, comm in enumerate(communities):
                for node in comm:
                    self._community_map[node] = idx
            self._last_community_detect = now
        except Exception as exc:
            logger.warning("Community detection failed: %s", exc)

        return self._community_map

    def get_node_enrichment(self, node_id: str) -> dict:
        """Get enriched data for a node: fraud edge count, total volume, etc."""
        g = self._graph
        if not g.has_node(node_id):
            return {}
        nd = g.nodes[node_id]

        fraud_edge_count = 0
        total_volume_paisa = 0
        seen_keys: set = set()

        for u, v, key, data in g.edges(node_id, keys=True, data=True):
            seen_keys.add(key)
            if data.get("fraud_label", 0) > 0:
                fraud_edge_count += 1
            total_volume_paisa += data.get("amount_paisa", 0)
        for u, v, key, data in g.in_edges(node_id, keys=True, data=True):
            if key not in seen_keys:
                if data.get("fraud_label", 0) > 0:
                    fraud_edge_count += 1
                total_volume_paisa += data.get("amount_paisa", 0)

        return {
            "fraud_edge_count": fraud_edge_count,
            "total_volume_paisa": total_volume_paisa,
            "first_seen": nd.get("first_seen", 0),
            "last_seen": nd.get("last_seen", 0),
        }

    # ── Diagnostics ──────────────────────────────────────────────────────

    @property
    def graph(self) -> nx.MultiDiGraph:
        """Direct graph access for testing and advanced queries."""
        return self._graph

    def snapshot(self) -> dict:
        """Full graph state for monitoring dashboards."""
        return {
            "graph": {
                "nodes": self._graph.number_of_nodes(),
                "edges": self._graph.number_of_edges(),
                "latest_timestamp": self._latest_timestamp,
            },
            "metrics": self.metrics.snapshot(),
            "config": {
                "edge_ttl_sec": self._cfg.edge_ttl_seconds,
                "prune_interval": self._cfg.prune_interval_batches,
                "scan_interval": self._cfg.scan_interval_batches,
            },
        }

"""
PayFlow — Graph Pattern Detection Algorithms
================================================
Pure CPU graph algorithms for structural fraud pattern identification
on the transaction multi-digraph.

Supported Patterns:

    1. Money Mule Networks (Star Topology)
       ───────────────────────────────────
       Multiple distinct senders converge on a single "mule" node within
       a tight temporal window.  Detection scans node in-degree from
       unique sources and flags nodes exceeding a configurable threshold.

           S₁ ──┐
           S₂ ──┤
           S₃ ──┼──▶ MULE ──▶ Beneficiary
           S₄ ──┤
           S₅ ──┘

    2. Circular Transaction Patterns (Cycle Detection)
       ────────────────────────────────────────────────
       Funds cycle back to the originator through a chain of accounts
       within a rapid temporal window (round-tripping / layering):

           A ──▶ B ──▶ C ──▶ A     (length-3 cycle)

       Uses Johnson's algorithm (via ``nx.simple_cycles``) with a
       ``length_bound`` to cap DFS depth (default 10), operating on a
       temporally-filtered projection of the multi-digraph.

    3. Suspicious Community Clusters
       ──────────────────────────────
       Dense subgraphs scored via density, volume, fraud-edge ratio
       and temporal burst rate.

    4. Centrality-Based Intermediary Detection
       ────────────────────────────────────────
       Accounts with statistically anomalous betweenness centrality —
       CB(v) = Σ σ(s,t|v) / σ(s,t) — are flagged as potential
       transaction intermediaries (bridge nodes / layering conduits).

Pipeline Position:
    TransactionGraph (builder.py)
         │
         ├── MuleDetector.detect(graph, …)       → list[MuleNetwork]
         ├── CycleDetector.detect(graph, …)      → list[TransactionCycle]
         ├── CommunityAnomalyDetector.detect(…)  → list[SuspiciousCluster]
         └── CentralityAnalyzer.detect(graph, …) → list[IntermediaryNode]

Algorithm Complexity:
    - Mule detection    : O(V + E_window) per scan (single pass)
    - Cycle detection   : O((V + E) · C) where C = cycles found, bounded by
      length_bound and max_results cap
    - Community scoring : O(V + E) per community
    - Centrality analysis: O(V·E) exact, O(k·(V+E)) approximate (k ≤ 200)
    - Memory: algorithms operate on nx views/subgraphs, minimal copies.
"""

from __future__ import annotations

import logging
import time
from typing import NamedTuple

import networkx as nx

from config.settings import FRAUD_THRESHOLDS

logger = logging.getLogger(__name__)


# ── Result Types ────────────────────────────────────────────────────────────────

class IntermediaryNode(NamedTuple):
    """Account flagged as a potential transaction intermediary via centrality."""
    node_id: str                      # account ID
    betweenness: float                # betweenness centrality ∈ [0, 1]
    in_degree: int
    out_degree: int
    total_volume_paisa: int           # total (in + out) transaction volume
    pagerank: float                   # PageRank score
    z_score: float                    # standard deviations above mean betweenness
    detection_time_ms: float


class MuleNetwork(NamedTuple):
    """Detected money mule star topology."""
    mule_node: str                    # suspected mule account ID
    distinct_senders: int             # number of unique incoming senders
    total_inbound_paisa: int          # sum of inbound amounts in window
    sender_ids: list[str]             # feeder account IDs
    edge_timestamps: list[int]        # timestamps of inbound edges
    txn_ids: list[str]                # transaction IDs forming the star
    detection_time_ms: float


class TransactionCycle(NamedTuple):
    """Detected circular transaction pattern (round-tripping)."""
    cycle_nodes: list[str]            # ordered account IDs in the cycle
    cycle_length: int                 # number of hops
    txn_ids: list[str]                # transaction IDs forming the cycle
    timestamps: list[int]             # edge timestamps in cycle order
    total_amount_paisa: int           # sum of amounts along the cycle
    time_span_minutes: float          # (max_ts − min_ts) / 60
    detection_time_ms: float


# ── Money Mule Detector ────────────────────────────────────────────────────────

class MuleDetector:
    """
    Detects money mule star topologies: nodes with unusually high
    in-degree from DISTINCT senders within a temporal window.

    Usage:
        detector = MuleDetector(min_distinct_senders=5)
        findings = detector.detect(graph, current_time=int(time.time()))
    """

    def __init__(
        self,
        min_distinct_senders: int = 5,
        window_seconds: int = FRAUD_THRESHOLDS.rapid_layering_window_minutes * 60,
    ) -> None:
        self._min_senders = min_distinct_senders
        self._window_sec = window_seconds

    def detect(
        self,
        graph: nx.MultiDiGraph,
        current_time: int,
    ) -> list[MuleNetwork]:
        """
        Full-graph scan for mule star patterns within the time window.

        Args:
            graph: transaction multi-digraph
            current_time: Unix epoch reference; edges within
                          [current_time − window, current_time] are considered.

        Returns:
            List of detected MuleNetwork results.
        """
        t0 = time.perf_counter()
        cutoff = current_time - self._window_sec
        results: list[MuleNetwork] = []

        for node in graph.nodes():
            # Fast pre-filter: skip nodes whose total in-degree is below threshold
            if graph.in_degree(node) < self._min_senders:
                continue

            finding = self._check_node(graph, node, cutoff, t0)
            if finding is not None:
                results.append(finding)

        elapsed = (time.perf_counter() - t0) * 1000
        if results:
            logger.info(
                "Mule scan: %d networks found in %.1f ms", len(results), elapsed,
            )
        return results

    def detect_around_node(
        self,
        graph: nx.MultiDiGraph,
        node_id: str,
        current_time: int,
    ) -> MuleNetwork | None:
        """
        Targeted single-node check.  Used during AlertRouter investigation
        to inspect the sender/receiver of a flagged transaction.
        """
        if not graph.has_node(node_id):
            return None
        cutoff = current_time - self._window_sec
        return self._check_node(graph, node_id, cutoff, time.perf_counter())

    # ── internals ───────────────────────────────────────────────────────

    def _check_node(
        self,
        graph: nx.MultiDiGraph,
        node: str,
        cutoff: int,
        t0: float,
    ) -> MuleNetwork | None:
        senders: set[str] = set()
        total_paisa = 0
        timestamps: list[int] = []
        txn_ids: list[str] = []

        for src, _, key, data in graph.in_edges(node, data=True, keys=True):
            ts = data.get("timestamp", 0)
            if ts < cutoff:
                continue
            senders.add(src)
            total_paisa += data.get("amount_paisa", 0)
            timestamps.append(ts)
            txn_ids.append(key)

        if len(senders) < self._min_senders:
            return None

        return MuleNetwork(
            mule_node=node,
            distinct_senders=len(senders),
            total_inbound_paisa=total_paisa,
            sender_ids=sorted(senders),
            edge_timestamps=timestamps,
            txn_ids=txn_ids,
            detection_time_ms=(time.perf_counter() - t0) * 1000,
        )


# ── Circular Transaction Detector ──────────────────────────────────────────────

class CycleDetector:
    """
    Detects circular transaction patterns (round-tripping) using bounded
    DFS cycle enumeration with temporal filtering.

    Algorithm:
        1. Project the multi-digraph onto a simple DiGraph containing only
           edges within the temporal window (collapse parallel edges,
           keeping the most recent).
        2. Run ``nx.simple_cycles(digraph, length_bound=max_cycle_length)``.
        3. Map each node-level cycle back to specific transaction edges
           in the original multi-digraph.
        4. Filter: keep only cycles whose full time span ≤ window.

    Usage:
        detector = CycleDetector()
        cycles = detector.detect(graph, current_time=int(time.time()))
    """

    def __init__(
        self,
        max_cycle_length: int = FRAUD_THRESHOLDS.max_cycle_length,
        window_seconds: int = FRAUD_THRESHOLDS.rapid_layering_window_minutes * 60,
    ) -> None:
        self._max_len = max_cycle_length
        self._window_sec = window_seconds

    def detect(
        self,
        graph: nx.MultiDiGraph,
        current_time: int,
        max_results: int = 100,
    ) -> list[TransactionCycle]:
        """
        Find all cycles of length ≤ max_cycle_length in the temporal window.

        Args:
            graph: transaction multi-digraph
            current_time: Unix epoch reference
            max_results: cap on returned cycles to prevent runaway enumeration

        Returns:
            List of TransactionCycle results sorted by time_span ascending.
        """
        t0 = time.perf_counter()
        cutoff = current_time - self._window_sec

        # Step 1: build temporal simple DiGraph (collapse multi-edges)
        temporal = self._build_temporal_digraph(graph, cutoff)

        if temporal.number_of_edges() == 0:
            return []

        # Step 2: enumerate cycles (Johnson's algorithm with length bound)
        results: list[TransactionCycle] = []
        for cycle_nodes in nx.simple_cycles(temporal, length_bound=self._max_len):
            if len(results) >= max_results:
                break

            cycle_result = self._resolve_cycle(graph, cycle_nodes, cutoff, t0)
            if cycle_result is not None:
                results.append(cycle_result)

        # Sort tightest (smallest time span) first
        results.sort(key=lambda c: c.time_span_minutes)

        elapsed = (time.perf_counter() - t0) * 1000
        if results:
            logger.info(
                "Cycle scan: %d cycles found (len %d–%d) in %.1f ms",
                len(results),
                min(c.cycle_length for c in results),
                max(c.cycle_length for c in results),
                elapsed,
            )
        return results

    def detect_around_nodes(
        self,
        graph: nx.MultiDiGraph,
        node_ids: list[str],
        current_time: int,
        k_hops: int = 3,
    ) -> list[TransactionCycle]:
        """
        Extract a k-hop subgraph around the given nodes and run cycle
        detection on that subgraph.  Used for targeted AlertRouter
        investigation.
        """
        neighborhood: set[str] = set()
        for node in node_ids:
            if not graph.has_node(node):
                continue
            fwd = nx.ego_graph(graph, node, radius=k_hops)
            neighborhood.update(fwd.nodes())
            rev = nx.ego_graph(graph.reverse(copy=False), node, radius=k_hops)
            neighborhood.update(rev.nodes())

        if not neighborhood:
            return []

        subgraph = graph.subgraph(neighborhood)
        return self.detect(subgraph, current_time)

    # ── internals ───────────────────────────────────────────────────────

    @staticmethod
    def _build_temporal_digraph(
        graph: nx.MultiDiGraph,
        cutoff: int,
    ) -> nx.DiGraph:
        """
        Collapse multi-edges into a simple DiGraph containing only edges
        within the time window.  For each (u, v) pair, keep the most
        recent edge (highest timestamp ≥ cutoff).
        """
        temporal = nx.DiGraph()

        for u, v, key, data in graph.edges(keys=True, data=True):
            ts = data.get("timestamp", 0)
            if ts < cutoff:
                continue

            if temporal.has_edge(u, v):
                if ts > temporal[u][v].get("timestamp", 0):
                    temporal[u][v].update(
                        timestamp=ts, txn_id=key,
                        amount_paisa=data.get("amount_paisa", 0),
                    )
            else:
                temporal.add_edge(
                    u, v,
                    timestamp=ts, txn_id=key,
                    amount_paisa=data.get("amount_paisa", 0),
                )

        return temporal

    def _resolve_cycle(
        self,
        graph: nx.MultiDiGraph,
        cycle_nodes: list[str],
        cutoff: int,
        t0: float,
    ) -> TransactionCycle | None:
        """
        Map a node-level cycle back to specific transaction edges in the
        original multi-digraph.  Selects the most recent in-window edge
        for each hop.  Returns None if any hop has no valid edge.
        """
        txn_ids: list[str] = []
        timestamps: list[int] = []
        total_paisa = 0

        n = len(cycle_nodes)
        for i in range(n):
            u = cycle_nodes[i]
            v = cycle_nodes[(i + 1) % n]

            # Find the most recent in-window edge for this hop
            best_ts = -1
            best_key = None
            best_amt = 0
            if graph.has_node(u) and graph.has_node(v):
                edges = graph.get_edge_data(u, v)
                if edges:
                    for key, data in edges.items():
                        ts = data.get("timestamp", 0)
                        if ts >= cutoff and ts > best_ts:
                            best_ts = ts
                            best_key = key
                            best_amt = data.get("amount_paisa", 0)

            if best_key is None:
                return None  # broken hop — skip this cycle

            txn_ids.append(best_key)
            timestamps.append(best_ts)
            total_paisa += best_amt

        # Temporal constraint: entire cycle must fit within the window
        span_sec = max(timestamps) - min(timestamps)
        span_min = span_sec / 60.0
        if span_sec > self._window_sec:
            return None

        return TransactionCycle(
            cycle_nodes=cycle_nodes,
            cycle_length=n,
            txn_ids=txn_ids,
            timestamps=timestamps,
            total_amount_paisa=total_paisa,
            time_span_minutes=round(span_min, 2),
            detection_time_ms=(time.perf_counter() - t0) * 1000,
        )


# ── Suspicious Community / Cluster Anomaly Scoring ─────────────────────────────

class SuspiciousCluster(NamedTuple):
    """Detected anomalous community / cluster."""
    community_id: int
    node_ids: list[str]
    node_count: int
    internal_edges: int
    density: float                  # internal_edges / max_possible
    avg_amount_paisa: float
    total_amount_paisa: int
    fraud_edge_ratio: float         # fraction of fraud-labelled edges
    anomaly_score: float            # ∈ [0.0, 1.0]
    detection_time_ms: float


class CommunityAnomalyDetector:
    """
    Detects anomalous dense subgraphs (suspicious clusters) that may
    represent coordinated fraud rings, layering networks, or collusive
    account groups.

    Scoring factors:
      1. Internal density — tightly connected clusters are unusual for
         genuine P2P payments (legitimate networks are sparse).
      2. Transaction volume — high aggregate volume within a small cluster
         indicates possible layering.
      3. Temporal concentration — many edges within a short window.
      4. Fraud edge ratio — fraction of edges with non-zero fraud labels
         (from ML model predictions or ground truth).

    Algorithm:
      1. Detect communities via greedy modularity optimisation (NetworkX).
      2. For each community, compute density + volume + temporal stats.
      3. Score = weighted combination of anomaly signals.
      4. Return communities scoring above threshold.

    Usage:
        detector = CommunityAnomalyDetector(min_score=0.4)
        clusters = detector.detect(graph, community_map, current_time)
    """

    def __init__(
        self,
        min_score: float = 0.35,
        min_cluster_size: int = 3,
        max_cluster_size: int = 200,
    ) -> None:
        self._min_score = min_score
        self._min_size = min_cluster_size
        self._max_size = max_cluster_size

    def detect(
        self,
        graph: nx.MultiDiGraph,
        community_map: dict[str, int],
        current_time: int,
        window_seconds: int = FRAUD_THRESHOLDS.rapid_layering_window_minutes * 60,
    ) -> list[SuspiciousCluster]:
        """
        Score all communities and return those exceeding the anomaly threshold.

        Parameters
        ----------
        graph : nx.MultiDiGraph
            The transaction graph.
        community_map : dict[str, int]
            Mapping of node_id → community_id (from greedy modularity).
        current_time : int
            Unix epoch reference for temporal windowing.
        window_seconds : int
            Temporal window for edge relevance.
        """
        t0 = time.perf_counter()
        cutoff = current_time - window_seconds

        # Group nodes by community
        communities: dict[int, list[str]] = {}
        for node, cid in community_map.items():
            if graph.has_node(node):
                communities.setdefault(cid, []).append(node)

        results: list[SuspiciousCluster] = []

        for cid, nodes in communities.items():
            n = len(nodes)
            if n < self._min_size or n > self._max_size:
                continue

            cluster = self._score_community(graph, cid, nodes, cutoff, t0)
            if cluster is not None and cluster.anomaly_score >= self._min_score:
                results.append(cluster)

        # Sort by anomaly score descending
        results.sort(key=lambda c: c.anomaly_score, reverse=True)

        elapsed = (time.perf_counter() - t0) * 1000
        if results:
            logger.info(
                "Community anomaly scan: %d suspicious clusters (of %d total) "
                "in %.1f ms",
                len(results), len(communities), elapsed,
            )
        return results

    def _score_community(
        self,
        graph: nx.MultiDiGraph,
        cid: int,
        nodes: list[str],
        cutoff: int,
        t0: float,
    ) -> SuspiciousCluster | None:
        """Compute anomaly score for a single community."""
        node_set = set(nodes)
        n = len(nodes)

        # Count internal edges (within community) and their properties
        internal_edges = 0
        total_paisa = 0
        fraud_edges = 0
        total_edges = 0
        timestamps: list[int] = []

        for u, v, data in graph.edges(data=True):
            if u in node_set and v in node_set:
                ts = data.get("timestamp", 0)
                if ts >= cutoff:
                    internal_edges += 1
                    total_paisa += data.get("amount_paisa", 0)
                    if data.get("fraud_label", 0) > 0:
                        fraud_edges += 1
                    timestamps.append(ts)
                total_edges += 1

        if internal_edges < 2:
            return None

        # Density: internal_edges / max_possible_directed_edges
        max_edges = n * (n - 1)  # directed graph
        density = internal_edges / max_edges if max_edges > 0 else 0.0

        # Average transaction amount
        avg_amount = total_paisa / internal_edges if internal_edges > 0 else 0.0

        # Fraud edge ratio
        fraud_ratio = fraud_edges / internal_edges if internal_edges > 0 else 0.0

        # Temporal concentration (edges per hour within window)
        if timestamps:
            span = max(timestamps) - min(timestamps)
            hours = max(span / 3600, 1.0)
            temporal_rate = internal_edges / hours
        else:
            temporal_rate = 0.0

        # Anomaly score: weighted combination
        #   density_signal:  dense clusters in financial networks are unusual
        #   volume_signal:   high total volume in small group
        #   fraud_signal:    already-flagged edges
        #   temporal_signal: burst of activity
        density_signal = min(1.0, density * 5.0)  # density > 0.2 → max
        volume_signal = min(1.0, total_paisa / (50_000_00 * n))  # ₹50K/node avg
        fraud_signal = min(1.0, fraud_ratio * 2.0)  # >50% fraud → max
        temporal_signal = min(1.0, temporal_rate / 20.0)  # >20 edges/hour → max

        anomaly_score = (
            0.30 * density_signal
            + 0.25 * volume_signal
            + 0.25 * fraud_signal
            + 0.20 * temporal_signal
        )

        return SuspiciousCluster(
            community_id=cid,
            node_ids=nodes,
            node_count=n,
            internal_edges=internal_edges,
            density=round(density, 4),
            avg_amount_paisa=round(avg_amount, 2),
            total_amount_paisa=total_paisa,
            fraud_edge_ratio=round(fraud_ratio, 4),
            anomaly_score=round(anomaly_score, 4),
            detection_time_ms=(time.perf_counter() - t0) * 1000,
        )


# ── Centrality-Based Intermediary Detection ────────────────────────────────────

class CentralityAnalyzer:
    """
    Detects accounts acting as transaction intermediaries using
    betweenness centrality — CB(v) = Σ σ(s,t|v) / σ(s,t).

    High betweenness centrality indicates a node that lies on many
    shortest paths between other accounts.  In banking fraud, these
    nodes often represent:
      • Money mule accounts channelling funds between layers
      • Bridge accounts connecting otherwise disjoint fraud rings
      • Intermediaries in round-trip layering schemes

    Key difference from MuleDetector: MuleDetector finds star topologies
    (one node with many distinct incoming senders).  CentralityAnalyzer
    finds *bridge* nodes that may have moderate degree but sit on critical
    flow paths — a complementary signal.

    Algorithm:
      1. Compute betweenness centrality for all nodes (approximate for
         graphs > 500 nodes, using k pivot samples).
      2. Flag nodes whose betweenness exceeds ``min_z_score`` standard
         deviations above the network mean.
      3. Return flagged nodes sorted by betweenness descending.

    Usage:
        analyzer = CentralityAnalyzer(min_z_score=2.5)
        findings = analyzer.detect(graph, current_time)
    """

    def __init__(
        self,
        min_z_score: float = 2.5,
        min_betweenness: float = 0.01,
        k_approx: int = 200,
    ) -> None:
        self._min_z = min_z_score
        self._min_bc = min_betweenness
        self._k_approx = k_approx
        # Cache for reuse by feature engine
        self._last_bc: dict[str, float] = {}
        self._last_pr: dict[str, float] = {}

    def detect(
        self,
        graph: nx.MultiDiGraph,
        current_time: int,
        max_results: int = 50,
    ) -> list[IntermediaryNode]:
        """
        Identify nodes with statistically anomalous betweenness centrality.

        Parameters
        ----------
        graph : nx.MultiDiGraph
        current_time : int
            Unix epoch (used for logging; centrality is topology-based).
        max_results : int
            Cap on returned findings.

        Returns
        -------
        list[IntermediaryNode]
            Nodes sorted by betweenness centrality descending.
        """
        t0 = time.perf_counter()
        n = graph.number_of_nodes()
        if n < 3:
            return []

        # Compute betweenness centrality (approximate for large graphs)
        try:
            if n > 500:
                k = min(self._k_approx, n)
                bc = nx.betweenness_centrality(
                    graph, k=k, normalized=True, weight=None,
                )
            else:
                bc = nx.betweenness_centrality(
                    graph, normalized=True, weight=None,
                )
        except nx.NetworkXError:
            return []

        self._last_bc = bc

        # Compute PageRank for enrichment
        try:
            pr = nx.pagerank(graph, alpha=0.85, max_iter=50, tol=1e-4)
        except (nx.NetworkXError, nx.PowerIterationFailedConvergence):
            pr = {}
        self._last_pr = pr

        # Statistical thresholding — z-score above mean
        values = list(bc.values())
        if not values:
            return []
        mean_bc = sum(values) / len(values)
        var_bc = sum((v - mean_bc) ** 2 for v in values) / len(values)
        std_bc = var_bc ** 0.5

        results: list[IntermediaryNode] = []

        for node_id, node_bc in bc.items():
            if node_bc < self._min_bc:
                continue

            z = (node_bc - mean_bc) / std_bc if std_bc > 1e-12 else 0.0
            if z < self._min_z:
                continue

            # Compute total volume through this node
            total_vol = 0
            for _, _, data in graph.edges(node_id, data=True):
                total_vol += data.get("amount_paisa", 0)
            for _, _, data in graph.in_edges(node_id, data=True):
                total_vol += data.get("amount_paisa", 0)

            results.append(IntermediaryNode(
                node_id=node_id,
                betweenness=round(node_bc, 6),
                in_degree=graph.in_degree(node_id),
                out_degree=graph.out_degree(node_id),
                total_volume_paisa=total_vol,
                pagerank=round(pr.get(node_id, 0.0), 6),
                z_score=round(z, 2),
                detection_time_ms=(time.perf_counter() - t0) * 1000,
            ))

        results.sort(key=lambda x: x.betweenness, reverse=True)
        results = results[:max_results]

        elapsed = (time.perf_counter() - t0) * 1000
        if results:
            logger.info(
                "Centrality analysis: %d intermediary nodes flagged "
                "(z>%.1f, bc>%.3f) in %.1f ms",
                len(results), self._min_z, self._min_bc, elapsed,
            )
        return results

    def detect_node(
        self,
        graph: nx.MultiDiGraph,
        node_id: str,
    ) -> IntermediaryNode | None:
        """
        Check a single node against cached centrality data.

        Returns the IntermediaryNode if the node exceeds thresholds,
        or None if it does not (or if no cached data is available).
        """
        if not self._last_bc or node_id not in self._last_bc:
            return None

        t0 = time.perf_counter()
        node_bc = self._last_bc[node_id]
        if node_bc < self._min_bc:
            return None

        values = list(self._last_bc.values())
        mean_bc = sum(values) / len(values)
        var_bc = sum((v - mean_bc) ** 2 for v in values) / len(values)
        std_bc = var_bc ** 0.5
        z = (node_bc - mean_bc) / std_bc if std_bc > 1e-12 else 0.0

        if z < self._min_z:
            return None

        total_vol = 0
        if graph.has_node(node_id):
            for _, _, data in graph.edges(node_id, data=True):
                total_vol += data.get("amount_paisa", 0)
            for _, _, data in graph.in_edges(node_id, data=True):
                total_vol += data.get("amount_paisa", 0)

        return IntermediaryNode(
            node_id=node_id,
            betweenness=round(node_bc, 6),
            in_degree=graph.in_degree(node_id) if graph.has_node(node_id) else 0,
            out_degree=graph.out_degree(node_id) if graph.has_node(node_id) else 0,
            total_volume_paisa=total_vol,
            pagerank=round(self._last_pr.get(node_id, 0.0), 6),
            z_score=round(z, 2),
            detection_time_ms=(time.perf_counter() - t0) * 1000,
        )

    @property
    def last_betweenness(self) -> dict[str, float]:
        """Cached betweenness centrality map from last ``detect()`` run."""
        return self._last_bc

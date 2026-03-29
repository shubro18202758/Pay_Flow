"""
PayFlow — Multi-Hop Mule Chain Detection (Carbanak Pattern)
=============================================================
Detects linear layering chains of the form:

    Source ──▶ Mule₁ ──▶ Mule₂ ──▶ … ──▶ Terminal (Cash-Out)

Unlike the star-topology ``MuleDetector`` (many-to-one convergence), this
module traces *forward paths* through the transaction graph where each hop
moves funds onward within a tight temporal window — the signature pattern
used by the Carbanak banking fraud network (2013–2018).

Detection Strategy:
    1. Identify **source candidates**: nodes that received large inbound
       value and forwarded most of it onward (pass-through ratio ≥ 80 %).
    2. Run bounded forward DFS from each source, following edges within
       the temporal window, pruning branches where:
       a.  a node has already been visited (prevents cycles),
       b.  the chain exceeds ``max_chain_length``,
       c.  the edge timestamp falls outside the time window.
    3. A chain is **reported** when it reaches a **terminal node** — an
       account that received funds but has little or no further outbound
       activity in the same window (likely a cash-out endpoint).

Complexity:
    O(V + E_window) per DFS tree, bounded by ``max_chain_length``.

Pipeline Position:
    TransactionGraph (builder.py)
         └── MuleChainDetector.detect(graph, …) → list[MuleChain]
"""

from __future__ import annotations

import logging
import time
from typing import NamedTuple

import networkx as nx

from config.settings import FRAUD_THRESHOLDS

logger = logging.getLogger(__name__)


# ── Result Types ───────────────────────────────────────────────────────────────


class MuleChain(NamedTuple):
    """A linear layering chain detected in the transaction graph."""

    chain_nodes: list[str]          # ordered account IDs: [source, m1, m2, …, terminal]
    chain_length: int               # number of hops (len(chain_nodes) - 1)
    txn_ids: list[str]              # transaction ID per hop
    timestamps: list[int]           # timestamp per hop
    total_amount_paisa: int         # cumulative value moved along the chain
    time_span_minutes: float        # first-hop to last-hop time span
    origin_node: str                # first node (fund source)
    terminal_node: str              # last node (suspected cash-out)
    detection_time_ms: float


# ── Detector ───────────────────────────────────────────────────────────────────


class MuleChainDetector:
    """
    Detect multi-hop linear mule chains (Carbanak-style layering).

    Usage::

        detector = MuleChainDetector()
        chains = detector.detect(graph, current_time=int(time.time()))
    """

    def __init__(
        self,
        max_chain_length: int = FRAUD_THRESHOLDS.max_cycle_length,
        min_chain_length: int = FRAUD_THRESHOLDS.min_layering_hops,
        window_seconds: int = FRAUD_THRESHOLDS.rapid_layering_window_minutes * 60,
        pass_through_ratio: float = 0.80,
    ) -> None:
        """
        Args:
            max_chain_length: Maximum hops to trace forward.
            min_chain_length: Minimum hops for a chain to be reported.
            window_seconds:   Only edges within this window are considered.
            pass_through_ratio: Fraction of inbound value that must be forwarded
                                for a node to be classified as a pass-through.
        """
        self._max_len = max_chain_length
        self._min_len = min_chain_length
        self._window = window_seconds
        self._pass_ratio = pass_through_ratio

    # ── Public API ─────────────────────────────────────────────────────────

    def detect(
        self,
        graph: nx.MultiDiGraph,
        current_time: int,
        max_results: int = 50,
    ) -> list[MuleChain]:
        """
        Full-graph scan for multi-hop layering chains.

        Returns:
            Chains sorted by chain_length descending (longest first).
        """
        t0 = time.perf_counter()
        cutoff = current_time - self._window

        sources = self._find_source_candidates(graph, cutoff)
        if not sources:
            return []

        results: list[MuleChain] = []
        seen_terminals: set[str] = set()

        for src in sources:
            if len(results) >= max_results:
                break
            chains = self._trace_forward(graph, src, cutoff, t0)
            for chain in chains:
                # Deduplicate: skip chains ending at an already-reported terminal
                if chain.terminal_node in seen_terminals:
                    continue
                seen_terminals.add(chain.terminal_node)
                results.append(chain)
                if len(results) >= max_results:
                    break

        results.sort(key=lambda c: c.chain_length, reverse=True)

        elapsed = (time.perf_counter() - t0) * 1000
        if results:
            logger.info(
                "Chain scan: %d chains found (len %d–%d) in %.1f ms",
                len(results),
                min(c.chain_length for c in results),
                max(c.chain_length for c in results),
                elapsed,
            )
        return results

    def trace_from_node(
        self,
        graph: nx.MultiDiGraph,
        node_id: str,
        current_time: int,
    ) -> list[MuleChain]:
        """
        Targeted: trace all forward chains originating from a specific node.
        Used during investigation to map onward fund flow from a
        flagged account.
        """
        if not graph.has_node(node_id):
            return []
        cutoff = current_time - self._window
        return self._trace_forward(graph, node_id, cutoff, time.perf_counter())

    def snapshot(self) -> dict:
        return {
            "max_chain_length": self._max_len,
            "min_chain_length": self._min_len,
            "window_seconds": self._window,
            "pass_through_ratio": self._pass_ratio,
        }

    # ── Internals ──────────────────────────────────────────────────────────

    def _find_source_candidates(
        self,
        graph: nx.MultiDiGraph,
        cutoff: int,
    ) -> list[str]:
        """
        Identify nodes that act as pass-through conduits: they receive
        substantial inbound value and forward ≥ pass_through_ratio onward
        within the time window.
        """
        candidates: list[tuple[str, int]] = []

        for node in graph.nodes():
            in_total = 0
            out_total = 0

            for _, _, data in graph.in_edges(node, data=True):
                if data.get("timestamp", 0) >= cutoff:
                    in_total += data.get("amount_paisa", 0)

            if in_total == 0:
                continue

            for _, _, data in graph.out_edges(node, data=True):
                if data.get("timestamp", 0) >= cutoff:
                    out_total += data.get("amount_paisa", 0)

            if out_total == 0:
                continue

            ratio = out_total / in_total if in_total > 0 else 0.0
            if ratio >= self._pass_ratio:
                candidates.append((node, in_total))

        # Return sorted by inbound volume descending (biggest movers first)
        candidates.sort(key=lambda x: x[1], reverse=True)
        return [c[0] for c in candidates[:200]]

    def _trace_forward(
        self,
        graph: nx.MultiDiGraph,
        start: str,
        cutoff: int,
        t0: float,
    ) -> list[MuleChain]:
        """
        Bounded forward DFS from *start*, collecting all linear paths
        that reach a terminal node (one with no further outbound edges
        in the window, or whose outbound ratio drops below threshold).
        """
        results: list[MuleChain] = []

        # Stack: (current_node, path_nodes, path_txn_ids, path_timestamps, path_amount, visited)
        stack: list[tuple[str, list[str], list[str], list[int], int, set[str]]] = [
            (start, [start], [], [], 0, {start}),
        ]

        while stack:
            node, path, txn_ids, timestamps, total_amt, visited = stack.pop()

            # Enumerate outbound edges within the time window
            children: list[tuple[str, str, int, int]] = []  # (target, txn_id, ts, amt)
            for _, target, key, data in graph.out_edges(node, keys=True, data=True):
                ts = data.get("timestamp", 0)
                if ts >= cutoff and target not in visited:
                    children.append((target, key, ts, data.get("amount_paisa", 0)))

            # Temporal ordering: follow edges chronologically
            children.sort(key=lambda x: x[2])

            if not children:
                # Terminal node — report chain if long enough
                if len(path) - 1 >= self._min_len:
                    span_sec = max(timestamps) - min(timestamps) if timestamps else 0
                    results.append(MuleChain(
                        chain_nodes=list(path),
                        chain_length=len(path) - 1,
                        txn_ids=list(txn_ids),
                        timestamps=list(timestamps),
                        total_amount_paisa=total_amt,
                        time_span_minutes=round(span_sec / 60.0, 2),
                        origin_node=path[0],
                        terminal_node=path[-1],
                        detection_time_ms=(time.perf_counter() - t0) * 1000,
                    ))
                continue

            # Check if max depth reached — report and stop
            if len(path) - 1 >= self._max_len:
                if len(path) - 1 >= self._min_len:
                    span_sec = max(timestamps) - min(timestamps) if timestamps else 0
                    results.append(MuleChain(
                        chain_nodes=list(path),
                        chain_length=len(path) - 1,
                        txn_ids=list(txn_ids),
                        timestamps=list(timestamps),
                        total_amount_paisa=total_amt,
                        time_span_minutes=round(span_sec / 60.0, 2),
                        origin_node=path[0],
                        terminal_node=path[-1],
                        detection_time_ms=(time.perf_counter() - t0) * 1000,
                    ))
                continue

            for target, txn_id, ts, amt in children:
                stack.append((
                    target,
                    path + [target],
                    txn_ids + [txn_id],
                    timestamps + [ts],
                    total_amt + amt,
                    visited | {target},
                ))

        return results

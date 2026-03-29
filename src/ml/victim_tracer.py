"""
PayFlow — Victim Fund Tracer (Digital Banking Fraud Flow Mapper)
==================================================================
Given a reported victim account, traces the complete fraud flow through
the transaction graph:

    Victim ──▶ Fraud Collector ──▶ Mule₁ ──▶ Mule₂ ──▶ … ──▶ Cash-Out

This implements the phishing / scam fraud model described in digital
banking fraud case studies, where:

    1. A victim is socially engineered (phishing, fake support, scam).
    2. The victim transfers funds to a **fraud collector** account.
    3. The fraud collector distributes funds across **mule accounts**.
    4. Mules forward funds onward until reaching **terminal nodes**
       (cash withdrawal, crypto conversion, etc.).

Unlike ``MuleChainDetector`` (which scans the full graph for pass-through
chains), this module starts from a *known victim* and maps the complete
downstream fund flow, producing a structured ``FraudFlowMap``.

Pipeline Position:
    Victim report / alert
         │
         └── VictimFundTracer.trace(graph, victim_id)
                  │
                  ├── immediate_receivers   (fraud collector accounts)
                  ├── mule_layers           (layered mule tiers)
                  ├── terminal_nodes        (cash-out endpoints)
                  └── FraudFlowMap          (complete case snapshot)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import NamedTuple

import networkx as nx

from config.settings import FRAUD_THRESHOLDS

logger = logging.getLogger(__name__)


# ── Result Types ───────────────────────────────────────────────────────────────


class FundFlowHop(NamedTuple):
    """A single hop in the fund flow from victim to cash-out."""

    from_node: str
    to_node: str
    txn_id: str
    amount_paisa: int
    timestamp: int
    layer: int  # 0 = victim→collector, 1 = collector→mule1, …


@dataclass
class FraudFlowMap:
    """
    Complete victim-centric fraud flow map.

    Captures the full downstream path of victim funds through
    fraud collectors, mule layers, and terminal cash-out nodes.
    """

    victim_id: str
    total_victim_loss_paisa: int = 0
    immediate_receivers: list[str] = field(default_factory=list)
    mule_layers: dict[int, list[str]] = field(default_factory=dict)
    terminal_nodes: list[str] = field(default_factory=list)
    all_hops: list[FundFlowHop] = field(default_factory=list)
    total_nodes_involved: int = 0
    max_depth: int = 0
    time_span_minutes: float = 0.0
    detection_time_ms: float = 0.0

    def to_dict(self) -> dict:
        return {
            "victim_id": self.victim_id,
            "total_victim_loss_paisa": self.total_victim_loss_paisa,
            "total_victim_loss_inr": round(self.total_victim_loss_paisa / 100, 2),
            "immediate_receivers": self.immediate_receivers,
            "mule_layers": {
                str(k): v for k, v in sorted(self.mule_layers.items())
            },
            "terminal_nodes": self.terminal_nodes,
            "total_nodes_involved": self.total_nodes_involved,
            "max_depth": self.max_depth,
            "time_span_minutes": self.time_span_minutes,
            "hop_count": len(self.all_hops),
            "hops": [
                {
                    "from": h.from_node,
                    "to": h.to_node,
                    "txn_id": h.txn_id,
                    "amount_paisa": h.amount_paisa,
                    "timestamp": h.timestamp,
                    "layer": h.layer,
                }
                for h in self.all_hops
            ],
            "detection_time_ms": round(self.detection_time_ms, 2),
        }


# ── Tracer ─────────────────────────────────────────────────────────────────────


class VictimFundTracer:
    """
    Trace the complete downstream fund flow from a reported victim account.

    Usage::

        tracer = VictimFundTracer()
        flow_map = tracer.trace(graph, victim_id="ACC-VICTIM-001",
                                current_time=int(time.time()))
        if flow_map.terminal_nodes:
            # Freeze terminal accounts, file investigation
            ...
    """

    def __init__(
        self,
        max_depth: int = FRAUD_THRESHOLDS.max_cycle_length,
        window_seconds: int = FRAUD_THRESHOLDS.rapid_layering_window_minutes * 60,
    ) -> None:
        self._max_depth = max_depth
        self._window = window_seconds
        self._traces: dict[str, FraudFlowMap] = {}

    # ── Public API ─────────────────────────────────────────────────────────

    def trace(
        self,
        graph: nx.MultiDiGraph,
        victim_id: str,
        current_time: int,
    ) -> FraudFlowMap:
        """
        Trace all downstream fund flows from a victim account.

        Performs a BFS traversal from the victim, recording every
        outbound hop within the temporal window. Nodes are classified
        into layers:
            - Layer 0: victim → immediate receivers (fraud collectors)
            - Layer 1+: onward mule hops
            - Terminal: nodes with no further outbound edges in window

        Args:
            graph:        Transaction multi-digraph.
            victim_id:    Account ID of the reported victim.
            current_time: Unix epoch reference.

        Returns:
            FraudFlowMap with full downstream tracing.
        """
        t0 = time.perf_counter()
        cutoff = current_time - self._window

        if not graph.has_node(victim_id):
            empty = FraudFlowMap(victim_id=victim_id)
            empty.detection_time_ms = (time.perf_counter() - t0) * 1000
            return empty

        flow_map = FraudFlowMap(victim_id=victim_id)
        visited: set[str] = {victim_id}
        all_timestamps: list[int] = []

        # BFS queue: (node, layer_depth)
        queue: list[tuple[str, int]] = [(victim_id, 0)]
        head = 0

        while head < len(queue):
            node, layer = queue[head]
            head += 1

            if layer >= self._max_depth:
                continue

            # Collect outbound edges within the temporal window
            outbound: list[tuple[str, str, int, int]] = []
            for _, target, key, data in graph.out_edges(node, keys=True, data=True):
                ts = data.get("timestamp", 0)
                if ts >= cutoff:
                    outbound.append((
                        target, key, data.get("amount_paisa", 0), ts,
                    ))

            if not outbound and layer > 0:
                # Terminal node — no further outbound in window
                if node not in flow_map.terminal_nodes:
                    flow_map.terminal_nodes.append(node)
                continue

            for target, txn_id, amount, ts in outbound:
                hop = FundFlowHop(
                    from_node=node,
                    to_node=target,
                    txn_id=txn_id,
                    amount_paisa=amount,
                    timestamp=ts,
                    layer=layer,
                )
                flow_map.all_hops.append(hop)
                all_timestamps.append(ts)

                # Track victim's total outbound at layer 0
                if layer == 0:
                    flow_map.total_victim_loss_paisa += amount
                    if target not in flow_map.immediate_receivers:
                        flow_map.immediate_receivers.append(target)

                # Classify node into mule layers (layer 1+)
                if layer >= 1:
                    layer_nodes = flow_map.mule_layers.setdefault(layer, [])
                    if node not in layer_nodes:
                        layer_nodes.append(node)

                # Enqueue target if not already visited
                if target not in visited:
                    visited.add(target)
                    queue.append((target, layer + 1))

        # Classify final-layer targets as terminals if they had no outbound
        # (already handled above via the BFS terminal check)

        # Also classify targets at the deepest visited layer as terminals
        # if they weren't already enqueued further
        for hop in flow_map.all_hops:
            if hop.to_node not in visited:
                continue
            # Check if this target has any outbound in the queue results
            has_outbound = any(
                h.from_node == hop.to_node for h in flow_map.all_hops
            )
            if not has_outbound and hop.to_node not in flow_map.terminal_nodes:
                if hop.to_node != victim_id:
                    flow_map.terminal_nodes.append(hop.to_node)

        # Compute summary stats
        flow_map.total_nodes_involved = len(visited)
        flow_map.max_depth = max(
            (h.layer for h in flow_map.all_hops), default=0,
        )
        if all_timestamps:
            span_sec = max(all_timestamps) - min(all_timestamps)
            flow_map.time_span_minutes = round(span_sec / 60.0, 2)

        flow_map.detection_time_ms = (time.perf_counter() - t0) * 1000

        # Cache result
        self._traces[victim_id] = flow_map

        if flow_map.all_hops:
            logger.info(
                "Victim trace %s: %d hops, %d nodes, %d terminals, "
                "loss=₹%.2f, %.1f ms",
                victim_id,
                len(flow_map.all_hops),
                flow_map.total_nodes_involved,
                len(flow_map.terminal_nodes),
                flow_map.total_victim_loss_paisa / 100,
                flow_map.detection_time_ms,
            )

        return flow_map

    def get_trace(self, victim_id: str) -> FraudFlowMap | None:
        """Retrieve a previously computed trace."""
        return self._traces.get(victim_id)

    def list_victims(self) -> list[str]:
        """Return all victim IDs that have been traced."""
        return list(self._traces.keys())

    def snapshot(self) -> dict:
        return {
            "max_depth": self._max_depth,
            "window_seconds": self._window,
            "total_traces": len(self._traces),
            "total_victim_loss_paisa": sum(
                t.total_victim_loss_paisa for t in self._traces.values()
            ),
        }

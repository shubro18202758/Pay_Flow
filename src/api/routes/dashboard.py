"""
PayFlow -- Dashboard Routes & SSE Event Stream
================================================
Serves the single-page real-time dashboard and provides REST endpoints
for initial data hydration plus a unified Server-Sent Events (SSE) stream
for live updates.

Routes:

    GET  /                              — Dashboard HTML page
    GET  /api/v1/snapshot               — Full system snapshot (JSON)
    GET  /api/v1/graph/topology         — Cytoscape.js-formatted graph data
    GET  /api/v1/circuit-breaker/status — Active freeze orders + metrics
    GET  /api/v1/agent/verdicts         — Recent agent verdicts from ledger
    GET  /api/v1/stream/events          — Unified SSE event stream
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path

from fastapi import APIRouter, Query, Request
from fastapi.responses import HTMLResponse, StreamingResponse

from src.api.events import EventBroadcaster

logger = logging.getLogger(__name__)

router = APIRouter(tags=["dashboard"])

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_FRONTEND_DIST = _PROJECT_ROOT / "frontend" / "app" / "dist"


# ── HTML Page ─────────────────────────────────────────────────────────────────

@router.get("/dashboard", response_class=HTMLResponse)
async def dashboard_page(request: Request):
    """Serve the React SPA Intelligence Center dashboard."""
    spa_index = _FRONTEND_DIST / "index.html"
    if spa_index.exists():
        return HTMLResponse(spa_index.read_text(encoding="utf-8"))
    # Fallback to Jinja2 template if SPA not built
    templates = request.app.state.templates
    return templates.TemplateResponse("dashboard.html", {"request": request})


@router.get("/prototype", response_class=HTMLResponse)
async def prototype_page(request: Request):
    """Serve the lightweight Cytoscape prototype dashboard."""
    templates = request.app.state.templates
    return templates.TemplateResponse("dashboard.html", {"request": request})


# ── REST: Initial Data Hydration ──────────────────────────────────────────────

@router.get("/api/v1/snapshot")
async def full_snapshot(request: Request):
    """Full system snapshot — one-shot JSON for initial page load."""
    orch = request.app.state.orchestrator
    if orch is None:
        return {"error": "orchestrator not initialized"}
    return orch.full_snapshot()


@router.get("/api/v1/graph/topology")
async def graph_topology(
    request: Request,
    limit: int = Query(500, ge=1, le=5000),
):
    """
    Current graph topology serialised for Cytoscape.js.

    Returns the *limit* most-recent edges and all connected nodes.
    Each node includes a ``status`` field (normal / frozen / paused)
    derived from the circuit breaker and agent breaker listener.
    """
    orch = request.app.state.orchestrator
    if orch is None:
        return {"nodes": [], "edges": []}

    graph = getattr(orch, "_graph", None)
    if graph is None:
        return {"nodes": [], "edges": []}

    g = graph._graph  # nx.MultiDiGraph

    # Collect edges sorted by recency, capped at limit
    all_edges = []
    for u, v, key, data in g.edges(keys=True, data=True):
        all_edges.append((u, v, key, data))
    all_edges.sort(key=lambda e: e[3].get("timestamp", 0), reverse=True)
    recent_edges = all_edges[:limit]

    # Collect connected node IDs
    node_ids: set[str] = set()
    for u, v, _, _ in recent_edges:
        node_ids.add(u)
        node_ids.add(v)

    # Determine node status from circuit breaker + agent listener
    breaker = getattr(orch, "_breaker", None)
    agent_listener = (
        getattr(breaker, "_agent_listener", None) if breaker else None
    )

    # Get enrichment data from graph
    community_map = graph.detect_communities() if hasattr(graph, 'detect_communities') else {}

    # Build account type lookup from world state if available
    world = getattr(orch, "_world", None)
    account_type_map: dict[str, str] = {}
    if world is not None:
        for acct in getattr(world, "accounts", []):
            account_type_map[acct.account_id] = acct.account_type.name if hasattr(acct.account_type, 'name') else str(acct.account_type)

    cyto_nodes = []
    for nid in node_ids:
        nd = g.nodes.get(nid, {})
        status = "normal"
        if breaker and breaker.is_frozen(nid):
            status = "frozen"
        elif agent_listener and agent_listener.is_routing_paused(nid):
            status = "paused"

        # Compute per-node enrichment
        fraud_edge_count = 0
        total_volume_paisa = 0
        seen_keys: set = set()
        for _, _, key, data in g.out_edges(nid, keys=True, data=True):
            seen_keys.add(key)
            if data.get("fraud_label", 0) > 0:
                fraud_edge_count += 1
            total_volume_paisa += data.get("amount_paisa", 0)
        for _, _, key, data in g.in_edges(nid, keys=True, data=True):
            if key not in seen_keys:
                if data.get("fraud_label", 0) > 0:
                    fraud_edge_count += 1
                total_volume_paisa += data.get("amount_paisa", 0)

        cyto_nodes.append({
            "data": {
                "id": nid,
                "txn_count": nd.get("txn_count", 0),
                "status": status,
                "account_type": account_type_map.get(nid, "UNKNOWN"),
                "community_id": community_map.get(nid, -1),
                "fraud_edge_count": fraud_edge_count,
                "total_volume_paisa": total_volume_paisa,
                "first_seen": nd.get("first_seen", 0),
                "last_seen": nd.get("last_seen", 0),
            }
        })

    # Import FraudPattern for label names
    from src.ingestion.schemas import FraudPattern
    fraud_label_names = {e.value: e.name for e in FraudPattern}

    cyto_edges = []
    for u, v, key, data in recent_edges:
        fl = data.get("fraud_label", 0)
        cyto_edges.append({
            "data": {
                "id": key,
                "source": u,
                "target": v,
                "amount_paisa": data.get("amount_paisa", 0),
                "channel": data.get("channel", 0),
                "fraud_label": fl,
                "fraud_label_name": fraud_label_names.get(fl, "NONE"),
                "timestamp": data.get("timestamp", 0),
                "device_fingerprint": data.get("device_fingerprint", ""),
            }
        })

    return {"nodes": cyto_nodes, "edges": cyto_edges}


@router.get("/api/v1/circuit-breaker/status")
async def circuit_breaker_status(request: Request):
    """Current freeze orders, device bans, and routing pauses."""
    orch = request.app.state.orchestrator
    breaker = getattr(orch, "_breaker", None) if orch else None
    if breaker is None:
        return {"freeze_orders": [], "snapshot": {}}

    orders = [fo.to_dict() for fo in breaker.get_frozen_nodes()]
    snap = breaker.snapshot()

    # Include agent breaker state if available
    agent_listener = getattr(breaker, "_agent_listener", None)
    agent_snap = agent_listener.snapshot() if agent_listener else {}

    return {
        "freeze_orders": orders,
        "snapshot": snap,
        "agent_breaker": agent_snap,
    }


@router.get("/api/v1/agent/verdicts")
async def agent_verdicts(
    request: Request,
    limit: int = Query(20, ge=1, le=200),
):
    """Recent agent verdicts retrieved from the audit ledger."""
    orch = request.app.state.orchestrator
    ledger = getattr(orch, "_ledger", None) if orch else None
    if ledger is None:
        return {"verdicts": []}

    from src.blockchain.models import EventType

    blocks = await ledger.get_recent_blocks(
        event_type=EventType.AGENT_VERDICT, limit=limit,
    )
    return {"verdicts": blocks}


@router.get("/api/v1/agent/investigation/{txn_id}")
async def agent_investigation(request: Request, txn_id: str):
    """Full investigation record for a specific transaction."""
    orch = request.app.state.orchestrator
    agent = getattr(orch, "_investigator", None) if orch else None
    if agent is None:
        return {"error": "Investigator agent not initialized", "txn_id": txn_id}

    record = agent.get_investigation(txn_id) if hasattr(agent, 'get_investigation') else None
    if record is None:
        return {"error": "No investigation record found", "txn_id": txn_id}

    return record


@router.get("/api/v1/blockchain/recent-blocks")
async def recent_blocks(
    request: Request,
    limit: int = Query(50, ge=1, le=500),
):
    """Recent blocks from the audit ledger for the Cryptographic Audit Trail."""
    orch = request.app.state.orchestrator
    ledger = getattr(orch, "_ledger", None) if orch else None
    if ledger is None:
        return {"blocks": [], "stats": None}

    blocks = await ledger.get_recent_blocks(limit=limit)
    stats = await ledger.get_stats()

    return {
        "blocks": blocks,
        "stats": {
            "total_blocks": stats.total_blocks,
            "latest_index": stats.latest_index,
            "latest_hash": stats.latest_hash,
            "latest_timestamp": stats.latest_timestamp,
            "checkpoints": stats.checkpoints,
            "db_size_bytes": stats.db_size_bytes,
        },
    }


# ── SSE: Unified Event Stream ─────────────────────────────────────────────────

ALL_CHANNELS = ["graph", "agent", "circuit_breaker", "risk_scores", "system", "simulation", "pipeline"]


@router.get("/api/v1/stream/events")
async def event_stream(request: Request):
    """
    Server-Sent Events stream carrying all dashboard event channels.

    Each event is a JSON object::

        {
            "channel": "graph" | "agent" | "circuit_breaker" | ...,
            "timestamp": <float>,
            "data": { "type": "...", ... }
        }

    A keepalive comment is sent every 15 seconds to prevent proxy
    timeouts and browser reconnects.
    """
    broadcaster = EventBroadcaster.get()
    queue = await broadcaster.subscribe(ALL_CHANNELS)

    async def generate():
        try:
            while True:
                if await request.is_disconnected():
                    break
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=15.0)
                    yield f"data: {json.dumps(event, default=str)}\n\n"
                except asyncio.TimeoutError:
                    yield ": keepalive\n\n"
        finally:
            await broadcaster.unsubscribe(queue, ALL_CHANNELS)

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )

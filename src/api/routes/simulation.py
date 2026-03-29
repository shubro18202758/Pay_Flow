"""
PayFlow -- Threat Simulation API Routes
=========================================
REST endpoints for controlling the Threat Simulation Engine during
live hackathon demos. Allows launching, stopping, and monitoring
attack scenarios.

Routes:

    GET   /api/v1/simulation/attacks           -- list available attack types
    POST  /api/v1/simulation/launch            -- launch an attack scenario
    POST  /api/v1/simulation/stop/{id}         -- stop a running attack
    POST  /api/v1/simulation/stop-all          -- stop all running attacks
    GET   /api/v1/simulation/status/{id}       -- get scenario status
    GET   /api/v1/simulation/active            -- list active scenarios
    GET   /api/v1/simulation/history           -- list all scenarios
"""

from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/simulation", tags=["simulation"])


# -- Request / Response models ---------------------------------------------

class LaunchRequest(BaseModel):
    attack_type: str
    params: Optional[dict] = None


class LaunchResponse(BaseModel):
    scenario_id: str
    attack_type: str
    attack_label: str
    status: str
    events_generated: int
    message: str


class StopResponse(BaseModel):
    scenario_id: str
    stopped: bool
    message: str


class StopAllResponse(BaseModel):
    stopped_count: int
    message: str


class InjectEventRequest(BaseModel):
    event_type: str  # "transaction" | "auth" | "interbank"
    # Transaction fields
    sender_id: Optional[str] = None
    receiver_id: Optional[str] = None
    amount_inr: Optional[float] = None
    channel: Optional[str] = None
    fraud_label: int = 0
    sender_account_type: Optional[str] = None
    receiver_account_type: Optional[str] = None
    # Auth fields
    account_id: Optional[str] = None
    action: Optional[str] = None
    ip_address: Optional[str] = None
    success: Optional[bool] = None
    # Interbank fields
    sender_ifsc: Optional[str] = None
    receiver_ifsc: Optional[str] = None
    message_type: Optional[str] = None
    currency_code: int = 356
    priority: int = 0
    # Common
    device_fingerprint: Optional[str] = None
    geo_lat: Optional[float] = None
    geo_lon: Optional[float] = None


# -- Helpers ---------------------------------------------------------------

def _get_engine(request: Request):
    """Extract the ThreatSimulationEngine from the orchestrator."""
    orch = request.app.state.orchestrator
    if orch is None:
        raise HTTPException(503, "Orchestrator not initialized")
    engine = getattr(orch, "_threat_engine", None)
    if engine is None:
        raise HTTPException(503, "Threat simulation engine not initialized")
    return engine


# -- Routes ----------------------------------------------------------------

@router.get("/attacks")
async def list_attacks(request: Request):
    """List available attack typologies with full parameter schemas."""
    engine = _get_engine(request)
    return {"attacks": engine.available_attacks_detailed()}


@router.post("/launch", response_model=LaunchResponse)
async def launch_attack(request: Request, body: LaunchRequest):
    """Launch a named attack scenario."""
    engine = _get_engine(request)

    try:
        scenario_id = await engine.launch_attack(body.attack_type, body.params)
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    except RuntimeError as exc:
        raise HTTPException(409, str(exc))

    status = engine.get_status(scenario_id)
    return LaunchResponse(
        scenario_id=scenario_id,
        attack_type=body.attack_type,
        attack_label=status["attack_label"],
        status=status["status"],
        events_generated=status["events_generated"],
        message=f"Attack '{status['attack_label']}' launched with {status['events_generated']} events",
    )


@router.post("/stop/{scenario_id}", response_model=StopResponse)
async def stop_attack(request: Request, scenario_id: str):
    """Stop a running attack by scenario ID."""
    engine = _get_engine(request)
    stopped = await engine.stop_attack(scenario_id)
    if not stopped:
        raise HTTPException(404, f"Scenario '{scenario_id}' not found or not running")
    return StopResponse(
        scenario_id=scenario_id,
        stopped=True,
        message=f"Scenario '{scenario_id}' stopped",
    )


@router.post("/stop-all", response_model=StopAllResponse)
async def stop_all(request: Request):
    """Stop all running attack scenarios."""
    engine = _get_engine(request)
    count = await engine.stop_all()
    return StopAllResponse(
        stopped_count=count,
        message=f"Stopped {count} running scenario(s)",
    )


@router.get("/status/{scenario_id}")
async def scenario_status(request: Request, scenario_id: str):
    """Get detailed status of a specific scenario."""
    engine = _get_engine(request)
    status = engine.get_status(scenario_id)
    if status is None:
        raise HTTPException(404, f"Scenario '{scenario_id}' not found")
    return status


@router.get("/active")
async def list_active(request: Request):
    """List all currently running attack scenarios."""
    engine = _get_engine(request)
    return {"scenarios": engine.list_active()}


@router.get("/history")
async def list_history(request: Request):
    """List all scenarios (active, completed, stopped, error)."""
    engine = _get_engine(request)
    return {"scenarios": engine.list_all()}


@router.get("/enums")
async def list_enums():
    """Return all schema enum values for dynamic form generation."""
    from src.ingestion.schemas import Channel, AccountType, AuthAction, FraudPattern

    def enum_to_list(enum_cls):
        return [{"value": e.value, "name": e.name} for e in enum_cls]

    return {
        "channels": enum_to_list(Channel),
        "account_types": enum_to_list(AccountType),
        "auth_actions": enum_to_list(AuthAction),
        "fraud_patterns": enum_to_list(FraudPattern),
        "message_types": ["MT103", "MT202", "N01", "N02", "N06"],
    }


@router.post("/inject-event")
async def inject_event(request: Request, body: InjectEventRequest):
    """Inject a custom event into the pipeline for real-time processing."""
    import time as _time
    import random as _random
    import hashlib as _hashlib
    from src.ingestion.schemas import (
        Transaction, AuthEvent, InterbankMessage,
        Channel, AccountType, AuthAction, FraudPattern,
    )
    from src.ingestion.validators import (
        compute_transaction_checksum,
        compute_auth_checksum,
        compute_interbank_checksum,
    )

    orch = request.app.state.orchestrator
    if orch is None:
        raise HTTPException(503, "Orchestrator not initialized")
    pipeline = getattr(orch, "_pipeline", None)
    if pipeline is None:
        raise HTTPException(503, "Pipeline not initialized")

    ts = int(_time.time())
    fp = body.device_fingerprint or _hashlib.sha256(f"custom-{ts}-{_random.randint(0,999999)}".encode()).hexdigest()[:16]
    geo_lat = body.geo_lat if body.geo_lat is not None else round(_random.uniform(8.0, 35.0), 6)
    geo_lon = body.geo_lon if body.geo_lon is not None else round(_random.uniform(69.0, 97.0), 6)

    def _hex_id(prefix: str) -> str:
        return f"{prefix}{_hashlib.sha256(f'{ts}-{_random.randint(0,999999)}'.encode()).hexdigest()[:12].upper()}"

    if body.event_type == "transaction":
        if not body.sender_id or not body.receiver_id or not body.amount_inr:
            raise HTTPException(400, "Transaction requires sender_id, receiver_id, and amount_inr")

        channel_val = Channel[body.channel.upper()] if body.channel else Channel.UPI
        amount_paisa = int(body.amount_inr * 100)
        txn_id = _hex_id("TXN")
        sender_acct_type = AccountType[body.sender_account_type.upper()] if body.sender_account_type else AccountType.SAVINGS
        receiver_acct_type = AccountType[body.receiver_account_type.upper()] if body.receiver_account_type else AccountType.SAVINGS
        fraud = FraudPattern(body.fraud_label)

        cs = compute_transaction_checksum(txn_id, ts, body.sender_id, body.receiver_id, amount_paisa, int(channel_val))

        event = Transaction(
            txn_id=txn_id, timestamp=ts,
            sender_id=body.sender_id, receiver_id=body.receiver_id,
            amount_paisa=amount_paisa, channel=channel_val,
            sender_branch=body.sender_id[:4] if len(body.sender_id) >= 4 else "0001",
            receiver_branch=body.receiver_id[:4] if len(body.receiver_id) >= 4 else "0001",
            sender_geo_lat=geo_lat, sender_geo_lon=geo_lon,
            receiver_geo_lat=round(geo_lat + _random.uniform(-0.5, 0.5), 6),
            receiver_geo_lon=round(geo_lon + _random.uniform(-0.5, 0.5), 6),
            device_fingerprint=fp,
            sender_account_type=sender_acct_type,
            receiver_account_type=receiver_acct_type,
            checksum=cs, fraud_label=fraud,
        )
        await pipeline.ingest(event)

        summary = {"type": "transaction", "txn_id": txn_id, "sender": body.sender_id, "receiver": body.receiver_id, "amount_paisa": amount_paisa, "channel": channel_val.name, "fraud_label": fraud.name}

    elif body.event_type == "auth":
        if not body.account_id:
            raise HTTPException(400, "Auth event requires account_id")

        action_val = AuthAction[body.action.upper()] if body.action else AuthAction.LOGIN
        ip = body.ip_address or f"182.{_random.randint(0,255)}.{_random.randint(0,255)}.{_random.randint(1,254)}"
        success = body.success if body.success is not None else True
        eid = _hex_id("AUTH")

        cs = compute_auth_checksum(eid, ts, body.account_id, int(action_val), ip)

        event = AuthEvent(
            event_id=eid, timestamp=ts, account_id=body.account_id,
            action=action_val, ip_address=ip,
            geo_lat=geo_lat, geo_lon=geo_lon,
            device_fingerprint=fp,
            user_agent_hash=_hashlib.sha256(f"CustomAgent/{ts}".encode()).hexdigest()[:16],
            success=success, checksum=cs,
        )
        await pipeline.ingest(event)

        summary = {"type": "auth", "event_id": eid, "account": body.account_id, "action": action_val.name, "success": success, "ip": ip}

    elif body.event_type == "interbank":
        if not body.sender_ifsc or not body.receiver_ifsc:
            raise HTTPException(400, "Interbank message requires sender_ifsc and receiver_ifsc")

        channel_val = Channel[body.channel.upper()] if body.channel else Channel.NEFT
        amount_paisa = int((body.amount_inr or 100000) * 100)
        msg_id = _hex_id("MSG")
        sender_acct = body.sender_id or f"SHELL{_random.randint(1000000000, 9999999999)}"
        receiver_acct = body.receiver_id or f"SHELL{_random.randint(1000000000, 9999999999)}"
        msg_type = body.message_type or "N06"

        cs = compute_interbank_checksum(msg_id, ts, body.sender_ifsc, body.receiver_ifsc, amount_paisa, int(channel_val))

        event = InterbankMessage(
            msg_id=msg_id, timestamp=ts,
            sender_ifsc=body.sender_ifsc, receiver_ifsc=body.receiver_ifsc,
            sender_account=sender_acct, receiver_account=receiver_acct,
            amount_paisa=amount_paisa, currency_code=body.currency_code,
            channel=channel_val, message_type=msg_type,
            sender_geo_lat=geo_lat, sender_geo_lon=geo_lon,
            device_fingerprint=fp, priority=body.priority, checksum=cs,
        )
        await pipeline.ingest(event)

        summary = {"type": "interbank", "msg_id": msg_id, "sender_ifsc": body.sender_ifsc, "receiver_ifsc": body.receiver_ifsc, "amount_paisa": amount_paisa, "channel": channel_val.name}

    else:
        raise HTTPException(400, f"Unknown event_type: {body.event_type}. Must be 'transaction', 'auth', or 'interbank'.")

    # Broadcast to SSE
    try:
        from src.api.events import EventBroadcaster
        await EventBroadcaster.get().publish("simulation", {
            "type": "custom_event_injected",
            "event": summary,
            "timestamp": ts,
        })
    except Exception:
        pass

    return {"status": "injected", "event": summary, "timestamp": ts}

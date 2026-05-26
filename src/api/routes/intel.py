"""Pre-fraud intelligence API routes."""

from __future__ import annotations

from fastapi import APIRouter, Query
from pydantic import BaseModel, Field

from src.intel import get_pre_fraud_intel_service

router = APIRouter(prefix="/api/v1/intel", tags=["pre-fraud-intelligence"])


class RefreshRequest(BaseModel):
    seed: int | None = None


class SimulateSignalRequest(BaseModel):
    scenario: str = Field(default="digital_arrest_mule", min_length=3, max_length=80)


async def _publish_intel_event(event_type: str, payload: dict) -> None:
    try:
        from src.api.events import EventBroadcaster

        await EventBroadcaster.get().publish("intel", {"type": event_type, **payload})
    except Exception:
        pass


@router.get("/sources")
async def list_intel_sources() -> dict:
    """Return configured OSINT/SOCMINT source tiers and trust policy."""
    service = get_pre_fraud_intel_service()
    return service.list_sources()


@router.post("/refresh")
async def refresh_pre_fraud_intel(body: RefreshRequest | None = None) -> dict:
    """Run a bounded prototype refresh across enabled public-source adapters."""
    service = get_pre_fraud_intel_service()
    result = service.refresh(seed=body.seed if body else None)
    await _publish_intel_event(
        "refresh_completed",
        {
            "signals_added": result["signals_added"],
            "trend_count": len(result["trends"]),
            "active_playbooks": result["tuning_status"]["active_playbooks"],
        },
    )
    return result


@router.get("/signals")
async def list_intel_signals(
    typology: str | None = None,
    region: str | None = None,
    source_tier: str | None = None,
    min_trust: float | None = Query(default=None, ge=0.0, le=1.0),
    since: float | None = Query(default=None, ge=0.0),
) -> dict:
    """Return external threat signals with filters for analyst triage."""
    service = get_pre_fraud_intel_service()
    return service.list_signals(
        typology=typology,
        region=region,
        source_tier=source_tier,
        min_trust=min_trust,
        since=since,
    )


@router.get("/trends")
async def list_intel_trends() -> dict:
    """Return clustered emerging fraud patterns."""
    service = get_pre_fraud_intel_service()
    return service.list_trends()


@router.get("/playbooks")
async def list_intel_playbooks() -> dict:
    """Return adaptive Qwen/watchlist/risk playbooks and promotion state."""
    service = get_pre_fraud_intel_service()
    return service.list_playbooks()


@router.get("/cockpit")
async def intel_cockpit() -> dict:
    """Return the visual cockpit dataset for preventive OSINT/SOCMINT fusion."""
    service = get_pre_fraud_intel_service()
    return service.cockpit()


@router.get("/media")
async def intel_media() -> dict:
    """Return source media previews with provenance and fallback status."""
    service = get_pre_fraud_intel_service()
    return service.media()


@router.get("/tuning/status")
async def pre_fraud_tuning_status() -> dict:
    """Return guarded auto-tuning state."""
    service = get_pre_fraud_intel_service()
    return service.tuning_status()


@router.post("/simulate-signal")
async def simulate_intel_signal(body: SimulateSignalRequest) -> dict:
    """Inject a deterministic judge-demo external fraud signal."""
    service = get_pre_fraud_intel_service()
    result = service.simulate_signal(body.scenario)
    await _publish_intel_event(
        "signal_simulated",
        {
            "scenario": result["scenario"],
            "signal_id": result["signal"]["signal_id"],
            "typologies": result["signal"]["typologies"],
            "active_playbooks": result["tuning_status"]["active_playbooks"],
        },
    )
    return result

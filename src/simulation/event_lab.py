"""Adaptive Event Lab and analyst-gated countermeasure orchestration.

This module is intentionally additive: generated events still enter the normal
PayFlow ingestion pipeline, while run/countermeasure metadata is kept in a
sidecar registry keyed by event ids.
"""

from __future__ import annotations

import asyncio
import hashlib
import random
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

from config.settings import OLLAMA_CFG
from src.ingestion.schemas import (
    AccountType,
    AuthAction,
    AuthEvent,
    Channel,
    FraudPattern,
    InterbankMessage,
    Transaction,
)
from src.ingestion.validators import (
    compute_auth_checksum,
    compute_interbank_checksum,
    compute_transaction_checksum,
)

EventLabMode = Literal["single", "burst", "chain"]
EventLabIntensity = Literal["demo", "scale"]
ProposalStatus = Literal["proposed", "approved", "rejected", "executing", "executed", "failed", "expired"]

EVALUATION_REQUIRED_STAGES = {
    "events_injected",
    "ingested",
    "ml_scored",
    "graph_investigated",
    "cb_evaluated",
    "pipeline_dispatched",
    "qwen_context_loaded",
}

SIDECAR_STAGE_DELAY_SECONDS = {
    "ingested": 0.45,
    "ml_scored": 0.75,
    "graph_investigated": 0.95,
    "cb_evaluated": 0.75,
    "pipeline_dispatched": 0.45,
}

DISPLAY_AGGREGATE_STAGES = {
    "ingested",
    "ml_scored",
    "graph_investigated",
    "cb_evaluated",
    "pipeline_dispatched",
}


def _now() -> float:
    return time.time()


def _stable_id(prefix: str, *parts: object, length: int = 12) -> str:
    raw = "|".join(str(part) for part in parts)
    return f"{prefix}-{hashlib.sha256(raw.encode()).hexdigest()[:length].upper()}"


def _hash_payload(payload: dict[str, Any]) -> str:
    return hashlib.sha256(repr(sorted(payload.items())).encode()).hexdigest()


def _rupees_to_paisa(value: int | float) -> int:
    return int(float(value) * 100)


def _channel_name(channel: Channel) -> str:
    return getattr(channel, "name", str(channel))


REGION_PROFILES: dict[str, dict[str, Any]] = {
    "mumbai": {"label": "Mumbai / Maharashtra", "city": "Mumbai", "state": "Maharashtra", "lat": 19.0760, "lon": 72.8777, "branch": "MUM"},
    "delhi": {"label": "Delhi NCR", "city": "Delhi", "state": "Delhi", "lat": 28.6139, "lon": 77.2090, "branch": "DEL"},
    "kolkata": {"label": "Kolkata / West Bengal", "city": "Kolkata", "state": "West Bengal", "lat": 22.5726, "lon": 88.3639, "branch": "KOL"},
    "chennai": {"label": "Chennai / Tamil Nadu", "city": "Chennai", "state": "Tamil Nadu", "lat": 13.0827, "lon": 80.2707, "branch": "CHN"},
    "bengaluru": {"label": "Bengaluru / Karnataka", "city": "Bengaluru", "state": "Karnataka", "lat": 12.9716, "lon": 77.5946, "branch": "BLR"},
    "hyderabad": {"label": "Hyderabad / Telangana", "city": "Hyderabad", "state": "Telangana", "lat": 17.3850, "lon": 78.4867, "branch": "HYD"},
    "lucknow": {"label": "Lucknow / Uttar Pradesh", "city": "Lucknow", "state": "Uttar Pradesh", "lat": 26.8467, "lon": 80.9462, "branch": "LKO"},
    "jaipur": {"label": "Jaipur / Rajasthan", "city": "Jaipur", "state": "Rajasthan", "lat": 26.9124, "lon": 75.7873, "branch": "JAI"},
    "guwahati": {"label": "Guwahati / Assam", "city": "Guwahati", "state": "Assam", "lat": 26.1445, "lon": 91.7362, "branch": "GAU"},
    "ahmedabad": {"label": "Ahmedabad / Gujarat", "city": "Ahmedabad", "state": "Gujarat", "lat": 23.0225, "lon": 72.5714, "branch": "AMD"},
    "pune": {"label": "Pune / Maharashtra", "city": "Pune", "state": "Maharashtra", "lat": 18.5204, "lon": 73.8567, "branch": "PUN"},
    "patna": {"label": "Patna / Bihar", "city": "Patna", "state": "Bihar", "lat": 25.5941, "lon": 85.1376, "branch": "PAT"},
}

TEMPLATE_ROUTE_DEFAULTS: dict[str, tuple[str, str]] = {
    "upi_mule_cashout": ("kolkata", "delhi"),
    "digital_arrest_chain": ("lucknow", "delhi"),
    "kyc_apk_phishing": ("bengaluru", "hyderabad"),
    "merchant_qr_misuse": ("ahmedabad", "mumbai"),
    "loan_app_extortion": ("patna", "kolkata"),
    "investment_scam_layering": ("mumbai", "ahmedabad"),
    "dormant_activation_high_value": ("jaipur", "delhi"),
    "round_trip_shell_loop": ("delhi", "jaipur"),
    "structuring_below_threshold": ("pune", "mumbai"),
    "profile_mismatch_rtgs": ("guwahati", "kolkata"),
}

PROFILE_LABELS: dict[str, str] = {
    "student": "student savings profile",
    "salary": "salary account profile",
    "merchant": "small merchant current account",
    "senior": "senior citizen savings profile",
    "dormant": "reactivated dormant account",
    "shell": "new shell/current account cluster",
}


def _clamp_int(raw: Any, default: int, minimum: int, maximum: int) -> int:
    try:
        value = int(raw)
    except (TypeError, ValueError):
        value = default
    return max(minimum, min(maximum, value))


def _channel_from_name(raw: Any, fallback: Channel) -> Channel:
    if isinstance(raw, Channel):
        return raw
    try:
        return Channel[str(raw).upper()]
    except (KeyError, TypeError, ValueError):
        return fallback


TYPOLOGY_TO_FRAUD = {
    "UPI_MULE_NETWORK": FraudPattern.UPI_MULE_NETWORK,
    "DIGITAL_ARREST": FraudPattern.UPI_MULE_NETWORK,
    "KYC_UPDATE_PHISHING": FraudPattern.VELOCITY_PHISHING,
    "MERCHANT_QR_MISUSE": FraudPattern.UPI_MULE_NETWORK,
    "LOAN_APP_EXTORTION": FraudPattern.STRUCTURING,
    "INVESTMENT_SCAM": FraudPattern.LAYERING,
    "SIM_SWAP": FraudPattern.VELOCITY_PHISHING,
    "DORMANT_ACTIVATION": FraudPattern.DORMANT_ACTIVATION,
    "STRUCTURING": FraudPattern.STRUCTURING,
    "ROUND_TRIPPING": FraudPattern.ROUND_TRIPPING,
    "PROFILE_MISMATCH": FraudPattern.PROFILE_MISMATCH,
    "LAYERING": FraudPattern.LAYERING,
}


BASE_TEMPLATES: list[dict[str, Any]] = [
    {
        "template_id": "upi_mule_cashout",
        "title": "UPI Mule Cash-Out Chain",
        "typologies": ["UPI_MULE_NETWORK", "LAYERING"],
        "channels": ["UPI", "IMPS"],
        "default_mode": "chain",
        "description": "Victim account pushes funds into student/rental mule accounts, followed by rapid UPI and IMPS consolidation.",
        "expected_indicators": ["UPI mule network", "rapid layering", "shared device fingerprint", "cash-out consolidation"],
        "countermeasure_actions": ["HOLD", "FREEZE_NODE", "WATCHLIST_DELTA", "GENERATE_EVIDENCE"],
    },
    {
        "template_id": "digital_arrest_chain",
        "title": "Digital Arrest Mule Burst",
        "typologies": ["DIGITAL_ARREST", "UPI_MULE_NETWORK", "LAYERING"],
        "channels": ["UPI", "IMPS"],
        "default_mode": "chain",
        "description": "Social-engineering pressure creates urgent UPI transfers to mules and a second-hop IMPS sweep.",
        "expected_indicators": ["digital arrest lure", "high urgency transfer", "mule fan-out", "second-hop sweep"],
        "countermeasure_actions": ["HOLD", "FREEZE_NODE", "FREEZE_1HOP", "CREATE_CASE"],
    },
    {
        "template_id": "kyc_apk_phishing",
        "title": "KYC APK Phishing Drain",
        "typologies": ["KYC_UPDATE_PHISHING", "SIM_SWAP", "UPI_MULE_NETWORK"],
        "channels": ["MOBILE", "UPI"],
        "default_mode": "chain",
        "description": "Suspicious OTP/auth activity precedes small UPI drains to a controlled mule account.",
        "expected_indicators": ["OTP/auth anomaly", "device change", "UPI velocity", "remote-access lure"],
        "countermeasure_actions": ["HOLD", "BAN_DEVICE", "PAUSE_ROUTING", "WATCHLIST_DELTA"],
    },
    {
        "template_id": "merchant_qr_misuse",
        "title": "Merchant QR Misuse Cluster",
        "typologies": ["MERCHANT_QR_MISUSE", "UPI_MULE_NETWORK"],
        "channels": ["UPI", "POS"],
        "default_mode": "burst",
        "description": "Many low-ticket UPI payments converge on a merchant QR before a larger settlement transfer.",
        "expected_indicators": ["merchant QR concentration", "small-ticket burst", "beneficiary convergence"],
        "countermeasure_actions": ["HOLD", "WATCHLIST_DELTA", "CREATE_CASE"],
    },
    {
        "template_id": "loan_app_extortion",
        "title": "Loan-App Extortion Collections",
        "typologies": ["LOAN_APP_EXTORTION", "STRUCTURING", "UPI_MULE_NETWORK"],
        "channels": ["UPI", "NEFT"],
        "default_mode": "burst",
        "description": "Repeated sub-threshold collections route into mule accounts and a later NEFT consolidation.",
        "expected_indicators": ["structuring", "repeat collections", "mule consolidation", "sub-threshold split"],
        "countermeasure_actions": ["HOLD", "FREEZE_NODE", "WATCHLIST_DELTA"],
    },
    {
        "template_id": "investment_scam_layering",
        "title": "Investment Scam Layering Ladder",
        "typologies": ["INVESTMENT_SCAM", "LAYERING", "PROFILE_MISMATCH"],
        "channels": ["NETBANKING", "NEFT", "RTGS"],
        "default_mode": "chain",
        "description": "Retail deposits move through current accounts and an RTGS transfer inconsistent with the profile.",
        "expected_indicators": ["profile mismatch", "multi-hop layering", "high-value RTGS"],
        "countermeasure_actions": ["HOLD", "FREEZE_NODE", "GENERATE_EVIDENCE"],
    },
    {
        "template_id": "dormant_activation_high_value",
        "title": "Dormant Account High-Value Activation",
        "typologies": ["DORMANT_ACTIVATION", "PROFILE_MISMATCH"],
        "channels": ["NETBANKING", "RTGS"],
        "default_mode": "single",
        "description": "A dormant savings account suddenly initiates a high-value transfer to a new counterparty.",
        "expected_indicators": ["dormant activation", "new beneficiary", "profile mismatch", "high-value transfer"],
        "countermeasure_actions": ["HOLD", "FREEZE_NODE", "CREATE_CASE"],
    },
    {
        "template_id": "round_trip_shell_loop",
        "title": "Round-Tripping Shell Loop",
        "typologies": ["ROUND_TRIPPING", "LAYERING"],
        "channels": ["NEFT", "RTGS"],
        "default_mode": "chain",
        "description": "Funds travel through shell current accounts and return near the origin through a different branch.",
        "expected_indicators": ["circular transaction", "round-tripping", "branch mismatch", "shell counterparties"],
        "countermeasure_actions": ["FREEZE_NODE", "FREEZE_1HOP", "GENERATE_EVIDENCE"],
    },
    {
        "template_id": "structuring_below_threshold",
        "title": "Structuring Below Reporting Threshold",
        "typologies": ["STRUCTURING", "UPI_MULE_NETWORK"],
        "channels": ["UPI", "IMPS", "NEFT"],
        "default_mode": "burst",
        "description": "Multiple transactions stay just below reporting thresholds before consolidating.",
        "expected_indicators": ["structuring", "threshold avoidance", "rapid beneficiary spread"],
        "countermeasure_actions": ["HOLD", "WATCHLIST_DELTA", "CREATE_CASE"],
    },
    {
        "template_id": "profile_mismatch_rtgs",
        "title": "Profile Mismatch RTGS Escalation",
        "typologies": ["PROFILE_MISMATCH", "LAYERING"],
        "channels": ["RTGS", "NEFT"],
        "default_mode": "single",
        "description": "Low-income profile performs an abrupt RTGS/NEFT transfer with no matching history.",
        "expected_indicators": ["profile mismatch", "abrupt value spike", "new counterparty"],
        "countermeasure_actions": ["HOLD", "CREATE_CASE", "GENERATE_EVIDENCE"],
    },
]


@dataclass
class EventLabStage:
    stage: str
    timestamp: float
    status: str = "completed"
    duration_ms: float | None = None
    event_ids: list[str] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class EventLabRun:
    run_id: str
    correlation_id: str
    template_id: str
    template_title: str
    mode: str
    intensity: str
    status: str
    analyst_required: bool
    linked_intel: dict[str, Any]
    expected_indicators: list[str]
    controls: dict[str, Any]
    event_ids: list[str]
    events: list[dict[str, Any]]
    proposal_ids: list[str]
    stages: list[EventLabStage]
    created_at: float
    updated_at: float
    qwen_explanation: str
    analysis_report: dict[str, Any] = field(default_factory=dict)
    decision_authority: str = "graph_ml_rules_ledger_pipeline"
    audit_hash: str = ""

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["stages"] = [stage.to_dict() for stage in self.stages]
        return payload


@dataclass
class CountermeasureProposal:
    proposal_id: str
    run_id: str
    action: str
    status: ProposalStatus
    title: str
    reason: str
    targets: list[str]
    trigger_event_ids: list[str]
    risk_evidence: dict[str, Any]
    intel_context: dict[str, Any]
    ttl_seconds: int
    expires_at: float
    execution_allowed: bool
    rollback_available: bool
    created_at: float
    updated_at: float
    analyst: str | None = None
    analyst_reason: str | None = None
    executed_at: float | None = None
    execution_result: dict[str, Any] = field(default_factory=dict)
    audit_hash: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class EventRunRegistry:
    """In-memory registry for event-lab runs and proposals."""

    def __init__(self) -> None:
        self._runs: dict[str, EventLabRun] = {}
        self._event_to_run: dict[str, str] = {}
        self._proposals: dict[str, CountermeasureProposal] = {}
        self._finalize_tasks: dict[str, asyncio.Task[None]] = {}

    def reset(self) -> None:
        for task in self._finalize_tasks.values():
            task.cancel()
        self._runs.clear()
        self._event_to_run.clear()
        self._proposals.clear()
        self._finalize_tasks.clear()

    # -- Template and generation ----------------------------------------

    def templates(self) -> dict[str, Any]:
        intel = self._intel_snapshot()
        templates = []
        for base in BASE_TEMPLATES:
            linked = self._match_playbooks(base["typologies"], intel)
            templates.append(
                {
                    **base,
                    "linked_playbooks": linked,
                    "trust_policy": "Analyst approval required before adaptive countermeasures execute.",
                    "execution_allowed": any(p.get("promotion_status") == "applied" for p in linked),
                }
            )
        return {
            "templates": templates,
            "active_playbooks": intel["playbooks"],
            "generated_at": _now(),
        }

    def preview(
        self,
        template_id: str,
        playbook_id: str | None = None,
        mode: EventLabMode | None = None,
        intensity: EventLabIntensity = "demo",
        seed: int | None = None,
        controls: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        template = self._template_by_id(template_id)
        linked_intel = self._linked_intel(template, playbook_id)
        generated = self._generate_events(template, linked_intel, mode or template["default_mode"], intensity, seed, controls)
        report = self._build_analysis_report(template, generated, linked_intel, proposals=[])
        return {
            "template": self._template_public(template, linked_intel),
            "run_preview": {
                "correlation_id": generated["correlation_id"],
                "mode": generated["mode"],
                "intensity": intensity,
                "controls": generated["controls"],
                "event_ids": generated["event_ids"],
                "events": generated["summaries"],
                "expected_indicators": template["expected_indicators"],
                "countermeasure_policy": self._countermeasure_policy(linked_intel),
                "qwen_explanation": self._qwen_explanation(template, linked_intel, generated["controls"], generated),
                "analysis_report": report,
            },
            "generated_at": _now(),
        }

    async def launch_run(
        self,
        orchestrator: Any,
        template_id: str,
        playbook_id: str | None = None,
        mode: EventLabMode | None = None,
        intensity: EventLabIntensity = "demo",
        seed: int | None = None,
        analyst_required: bool = True,
        controls: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if orchestrator is None:
            raise RuntimeError("Orchestrator not initialized")
        pipeline = getattr(orchestrator, "_pipeline", None)
        if pipeline is None:
            raise RuntimeError("Pipeline not initialized")
        if not getattr(pipeline, "_running", False):
            await pipeline.start()

        template = self._template_by_id(template_id)
        linked_intel = self._linked_intel(template, playbook_id)
        generated = self._generate_events(template, linked_intel, mode or template["default_mode"], intensity, seed, controls)

        run_id = _stable_id("RUN", generated["correlation_id"], _now(), length=10)
        run = EventLabRun(
            run_id=run_id,
            correlation_id=generated["correlation_id"],
            template_id=template["template_id"],
            template_title=template["title"],
            mode=generated["mode"],
            intensity=intensity,
            status="launching",
            analyst_required=analyst_required,
            linked_intel=linked_intel,
            expected_indicators=list(template["expected_indicators"]),
            controls=generated["controls"],
            event_ids=generated["event_ids"],
            events=generated["summaries"],
            proposal_ids=[],
            stages=[],
            created_at=_now(),
            updated_at=_now(),
            qwen_explanation=self._qwen_explanation(template, linked_intel, generated["controls"], generated),
        )
        run.audit_hash = _hash_payload({"run_id": run.run_id, "event_ids": run.event_ids, "intel": linked_intel, "controls": generated["controls"]})
        self._runs[run_id] = run
        for event_id in generated["event_ids"]:
            self._event_to_run[event_id] = run_id

        await self._record_run_stage(run, "intel_primed", meta={"linked_intel": linked_intel, "template": template["template_id"]})
        await self._record_run_stage(
            run,
            "events_generated",
            event_ids=generated["event_ids"],
            meta={
                "count": len(generated["events"]),
                "controls": generated["controls"],
                "amount_band_inr": generated["controls"].get("amount_band_inr"),
                "route": generated["controls"].get("route_label"),
            },
        )

        proposals = self._build_proposals(run, template, generated, linked_intel)
        for proposal in proposals:
            self._proposals[proposal.proposal_id] = proposal
            run.proposal_ids.append(proposal.proposal_id)
            await self._publish("countermeasure", {"type": "proposal_created", "proposal": proposal.to_dict()})
        launch_report = self._build_analysis_report(template, generated, linked_intel, proposals=[p.to_dict() for p in proposals], run=run)
        await self._record_run_stage(
            run,
            "qwen_context_loaded",
            event_ids=generated["event_ids"][: min(4, len(generated["event_ids"]))],
            meta={
                "model": OLLAMA_CFG.model,
                "role": "bounded_forensic_explanation",
                "risk_tier": launch_report.get("risk_tier"),
                "risk_score": launch_report.get("risk_score"),
                "route": generated["controls"].get("route_label"),
            },
        )

        for event in generated["events"]:
            await pipeline.ingest(event)

        run.status = "injected"
        run.updated_at = _now()
        await self._record_run_stage(
            run,
            "events_injected",
            event_ids=generated["event_ids"],
            meta={
                "pipeline": "live_ingestion_pipeline",
                "proposal_count": len(proposals),
                "transaction_count": generated["analysis_counts"]["transactions"],
                "auth_events": generated["analysis_counts"]["auth"],
                "interbank_messages": generated["analysis_counts"]["interbank"],
            },
        )
        await self._publish("event_lab", {"type": "run_launched", "run": run.to_dict()})
        self._schedule_sidecar_evaluation(run.run_id)
        self._schedule_finalization(run.run_id)
        return self.run_response(run_id)

    def run_response(self, run_id: str) -> dict[str, Any]:
        self._expire_old_proposals()
        run = self._runs.get(run_id)
        if run is None:
            raise KeyError(run_id)
        proposals = [self._proposals[pid].to_dict() for pid in run.proposal_ids if pid in self._proposals]
        template = self._template_by_id(run.template_id)
        if self._has_run_stage(run, "evaluation_complete"):
            run.analysis_report = self._build_analysis_report(
                template,
                {
                    "summaries": run.events,
                    "event_ids": run.event_ids,
                    "controls": run.controls,
                    "analysis_counts": self._event_type_counts(run.events),
                },
                run.linked_intel,
                proposals=proposals,
                run=run,
            )
        run_body = run.to_dict()
        run_body["stages"] = [stage.to_dict() for stage in self._display_stages(run)]
        run_body["raw_stage_count"] = len(run.stages)
        return {
            **run_body,
            "countermeasure_proposals": proposals,
            "countermeasure_policy": self._countermeasure_policy(run.linked_intel),
            "latency_metrics": self._latency_metrics(run),
        }

    def explainability_response(self, run_id: str) -> dict[str, Any]:
        """Return a judge/analyst readable trace of the adaptive response loop."""
        run_body = self.run_response(run_id)
        groups = self._explainability_stage_groups(run_body)
        proposals = run_body["countermeasure_proposals"]
        latest_stage = run_body["stages"][-1]["stage"] if run_body["stages"] else "awaiting_launch"
        return {
            "run": run_body,
            "stage_groups": groups,
            "evidence_panels": self._explainability_evidence_panels(run_body, groups),
            "proposal_lifecycle": [self._proposal_explainability(p) for p in proposals],
            "authority_matrix": [
                {
                    "layer": "Pre-Fraud Intel",
                    "role": "Primes templates, watchlists, Qwen context, and proposal wording.",
                    "authority": "advisory",
                    "can_execute": False,
                },
                {
                    "layer": "Rules + ML + Graph",
                    "role": "Scores internal PayFlow events and validates suspicious fund-flow evidence.",
                    "authority": "decision evidence",
                    "can_execute": False,
                },
                {
                    "layer": "Qwen 3.5 4B",
                    "role": "Explains the case context and maps evidence to analyst language.",
                    "authority": "bounded copilot",
                    "can_execute": False,
                },
                {
                    "layer": "Analyst Gate",
                    "role": "Approves or rejects countermeasure proposals before execution.",
                    "authority": "human approval",
                    "can_execute": True,
                },
                {
                    "layer": "Circuit Breaker + Ledger",
                    "role": "Executes approved controls and anchors the audit trail.",
                    "authority": "controlled execution",
                    "can_execute": True,
                },
            ],
            "runtime": {
                "latest_stage": latest_stage,
                "stage_count": len(run_body["stages"]),
                "proposal_count": len(proposals),
                "executed_count": len([p for p in proposals if p["status"] == "executed"]),
                "pending_count": len([p for p in proposals if p["status"] == "proposed"]),
                "ledger_hashes": [p["audit_hash"] for p in proposals if p.get("audit_hash")],
                "rollback_available": any(bool(p.get("rollback_available")) for p in proposals),
            },
            "generated_at": _now(),
        }

    def latest_run(self) -> dict[str, Any] | None:
        if not self._runs:
            return None
        run = max(self._runs.values(), key=lambda item: item.created_at)
        return self.run_response(run.run_id)

    async def record_stage_for_ids(
        self,
        event_ids: list[str],
        stage: str,
        meta: dict[str, Any] | None = None,
        duration_ms: float | None = None,
    ) -> None:
        runs: dict[str, list[str]] = {}
        for event_id in event_ids:
            run_id = self._event_to_run.get(event_id)
            if run_id:
                runs.setdefault(run_id, []).append(event_id)
        for run_id, ids in runs.items():
            run = self._runs.get(run_id)
            if run:
                await self._record_run_stage(run, stage, event_ids=ids, meta=meta or {}, duration_ms=duration_ms)
                self._schedule_finalization(run_id)

    # -- Countermeasure lifecycle ---------------------------------------

    def list_proposals(self, run_id: str | None = None, status: str | None = None) -> dict[str, Any]:
        self._expire_old_proposals()
        proposals = list(self._proposals.values())
        if run_id:
            proposals = [p for p in proposals if p.run_id == run_id]
        if status:
            proposals = [p for p in proposals if p.status == status]
        proposals.sort(key=lambda p: (p.status == "proposed", p.created_at), reverse=True)
        return {
            "count": len(proposals),
            "proposals": [p.to_dict() for p in proposals],
            "generated_at": _now(),
        }

    async def approve_proposal(
        self,
        proposal_id: str,
        orchestrator: Any,
        analyst: str = "union_bank_analyst",
        reason: str = "analyst_approved",
    ) -> dict[str, Any]:
        proposal = self._proposal_or_error(proposal_id)
        if proposal.status != "proposed":
            raise ValueError(f"Proposal is {proposal.status}, not proposed")
        if not proposal.execution_allowed:
            raise PermissionError("Proposal is advisory-only because source trust did not meet promotion policy")

        proposal.status = "executing"
        proposal.analyst = analyst
        proposal.analyst_reason = reason
        proposal.updated_at = _now()
        await self._publish("countermeasure", {"type": "proposal_approved", "proposal": proposal.to_dict()})

        try:
            proposal.execution_result = await self._execute_countermeasure(proposal, orchestrator)
            proposal.status = "executed"
            proposal.executed_at = _now()
            proposal.updated_at = _now()
            proposal.audit_hash = _hash_payload(proposal.to_dict() | {"audit_hash": ""})
            await self._anchor_countermeasure(orchestrator, "event_lab_countermeasure_approved", proposal)
            await self._publish("countermeasure", {"type": "action_executed", "proposal": proposal.to_dict()})
            await self._stage_for_proposal(proposal, "analyst_decision", {"decision": "approved", "analyst": analyst})
            await self._stage_for_proposal(proposal, "action_executed", proposal.execution_result)
            await self._stage_for_proposal(proposal, "ledger_anchored", {"audit_hash": proposal.audit_hash})
        except Exception as exc:
            proposal.status = "failed"
            proposal.updated_at = _now()
            proposal.execution_result = {"error": str(exc)}
            await self._publish("countermeasure", {"type": "action_failed", "proposal": proposal.to_dict()})
            raise
        return proposal.to_dict()

    async def reject_proposal(
        self,
        proposal_id: str,
        analyst: str = "union_bank_analyst",
        reason: str = "analyst_rejected",
    ) -> dict[str, Any]:
        proposal = self._proposal_or_error(proposal_id)
        if proposal.status != "proposed":
            raise ValueError(f"Proposal is {proposal.status}, not proposed")
        proposal.status = "rejected"
        proposal.analyst = analyst
        proposal.analyst_reason = reason
        proposal.updated_at = _now()
        proposal.audit_hash = _hash_payload(proposal.to_dict() | {"audit_hash": ""})
        await self._publish("countermeasure", {"type": "proposal_rejected", "proposal": proposal.to_dict()})
        await self._stage_for_proposal(proposal, "analyst_decision", {"decision": "rejected", "analyst": analyst, "reason": reason})
        return proposal.to_dict()

    def evidence_context(self) -> dict[str, Any]:
        latest = self.latest_run()
        if latest is None:
            return {
                "event_lab_run_id": None,
                "countermeasure_proposals": [],
                "analyst_decisions": [],
                "executed_actions": [],
                "qwen_explanation": "",
                "pre_fraud_playbook": None,
                "countermeasure_audit_hashes": [],
            }
        proposals = latest["countermeasure_proposals"]
        return {
            "event_lab_run_id": latest["run_id"],
            "countermeasure_proposals": proposals,
            "analyst_decisions": [
                {
                    "proposal_id": p["proposal_id"],
                    "action": p["action"],
                    "status": p["status"],
                    "analyst": p.get("analyst"),
                    "reason": p.get("analyst_reason"),
                }
                for p in proposals
                if p["status"] in {"rejected", "executed", "failed"}
            ],
            "executed_actions": [p for p in proposals if p["status"] == "executed"],
            "qwen_explanation": latest.get("qwen_explanation", ""),
            "pre_fraud_playbook": latest.get("linked_intel", {}).get("playbook"),
            "countermeasure_audit_hashes": [p["audit_hash"] for p in proposals if p.get("audit_hash")],
        }

    # -- Internal helpers ------------------------------------------------

    def _template_by_id(self, template_id: str) -> dict[str, Any]:
        for template in BASE_TEMPLATES:
            if template["template_id"] == template_id:
                return dict(template)
        raise KeyError(template_id)

    def _template_public(self, template: dict[str, Any], linked_intel: dict[str, Any]) -> dict[str, Any]:
        return {
            **template,
            "linked_playbooks": [linked_intel["playbook"]] if linked_intel.get("playbook") else [],
            "execution_allowed": bool(linked_intel.get("execution_allowed")),
        }

    def _intel_snapshot(self) -> dict[str, Any]:
        try:
            from src.intel import get_pre_fraud_intel_service

            service = get_pre_fraud_intel_service()
            playbooks = service.list_playbooks().get("playbooks", [])
            trends = service.list_trends().get("trends", [])
            signals = service.list_signals(min_trust=0.0).get("signals", [])
        except Exception:
            playbooks, trends, signals = [], [], []
        return {
            "playbooks": playbooks,
            "trends": trends,
            "signals": signals,
        }

    def _match_playbooks(self, typologies: list[str], intel: dict[str, Any]) -> list[dict[str, Any]]:
        trend_by_id = {t.get("trend_id"): t for t in intel["trends"]}
        matches = []
        for playbook in intel["playbooks"]:
            trend = trend_by_id.get(playbook.get("trend_id"), {})
            overlap = sorted(set(typologies) & set(trend.get("typologies", [])))
            if overlap or any(term.lower().replace(" ", "_") in " ".join(typologies).lower() for term in playbook.get("watchlist_terms", [])):
                matches.append(
                    {
                        **playbook,
                        "matched_typologies": overlap,
                        "trend": trend,
                    }
                )
        return matches[:3]

    def _linked_intel(self, template: dict[str, Any], playbook_id: str | None) -> dict[str, Any]:
        intel = self._intel_snapshot()
        matches = self._match_playbooks(template["typologies"], intel)
        selected = None
        if playbook_id:
            selected = next((p for p in intel["playbooks"] if p.get("playbook_id") == playbook_id), None)
            if selected:
                selected = {
                    **selected,
                    "trend": next((t for t in intel["trends"] if t.get("trend_id") == selected.get("trend_id")), {}),
                }
        selected = selected or (matches[0] if matches else None)
        trend = selected.get("trend", {}) if selected else {}
        trust = float(trend.get("trust_score") or 0.0)
        execution_allowed = bool(selected and selected.get("promotion_status") == "applied" and trust >= 0.85)
        evidence_ids = set(trend.get("evidence_ids", [])) if trend else set()
        linked_signals = [s for s in intel["signals"] if s.get("signal_id") in evidence_ids][:4]
        return {
            "playbook": selected,
            "trend": trend or None,
            "signals": linked_signals,
            "execution_allowed": execution_allowed,
            "trust_score": trust,
            "guardrail": "Pre-fraud intelligence primes event generation and proposals only; analyst approval and internal PayFlow evidence remain mandatory.",
        }

    def _generate_events(
        self,
        template: dict[str, Any],
        linked_intel: dict[str, Any],
        mode: str,
        intensity: str,
        seed: int | None,
        controls: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        normalized = self._normalize_controls(template, mode, intensity, controls)
        normalized["seed"] = seed
        normalized["mode"] = mode
        normalized["intensity"] = intensity
        material = f"{template['template_id']}|{mode}|{intensity}|{seed}|{linked_intel.get('trust_score', 0)}|{sorted(normalized.items())}"
        rng = random.Random(seed if seed is not None else int(hashlib.sha256(material.encode()).hexdigest()[:8], 16))
        correlation_id = _stable_id("COR", material, length=10)
        now = int(_now())

        count = int(normalized["event_count"])
        typology = template["typologies"][0]
        fraud = TYPOLOGY_TO_FRAUD.get(typology, FraudPattern.LAYERING)
        victim = f"UBI{rng.randint(10_000_000_000, 99_999_999_999)}"
        origin = f"UBI{rng.randint(10_000_000_000, 99_999_999_999)}"
        mule_depth = int(normalized["mule_depth"])
        mule_accounts = [f"MULE{rng.randint(10_000_000_000, 99_999_999_999)}" for _ in range(max(4, mule_depth + count + 2))]
        shell_accounts = [f"SHELL{rng.randint(10_000_000_000, 99_999_999_999)}" for _ in range(max(3, mule_depth + count))]
        device = hashlib.sha256(f"{correlation_id}-device".encode()).hexdigest()[:16]
        origin_region = normalized["origin_region"]
        destination_region = normalized["destination_region"]
        origin_geo = REGION_PROFILES[origin_region]
        destination_geo = REGION_PROFILES[destination_region]
        primary_channel = _channel_from_name(normalized["primary_channel"], Channel.UPI)
        secondary_channel = _channel_from_name(normalized["secondary_channel"], Channel.IMPS)
        velocity_minutes = int(normalized["velocity_minutes"])
        device_reuse = bool(normalized["device_reuse"])
        risk_bias = str(normalized["risk_bias"])

        base_amounts = {
            "structuring_below_threshold": 49_500,
            "merchant_qr_misuse": 6_900,
            "dormant_activation_high_value": 875_000,
            "profile_mismatch_rtgs": 1_425_000,
            "investment_scam_layering": 325_000,
            "round_trip_shell_loop": 600_000,
        }
        min_amount, max_amount = normalized["amount_band_inr"]
        if min_amount == max_amount:
            base_amount = min_amount
        else:
            default_amount = base_amounts.get(template["template_id"], 95_000)
            base_amount = max(min_amount, min(max_amount, default_amount + rng.randint(-int(default_amount * 0.12), int(default_amount * 0.12))))

        events: list[Any] = []
        summaries: list[dict[str, Any]] = []

        def event_geo(index: int) -> tuple[float, float, float, float]:
            progress = 0 if count <= 1 else min(1, max(0, index / max(count - 1, 1)))
            sender_lat = float(origin_geo["lat"]) + rng.uniform(-0.08, 0.08)
            sender_lon = float(origin_geo["lon"]) + rng.uniform(-0.08, 0.08)
            receiver_lat = float(origin_geo["lat"]) + (float(destination_geo["lat"]) - float(origin_geo["lat"])) * progress + rng.uniform(-0.11, 0.11)
            receiver_lon = float(origin_geo["lon"]) + (float(destination_geo["lon"]) - float(origin_geo["lon"])) * progress + rng.uniform(-0.11, 0.11)
            return round(sender_lat, 6), round(sender_lon, 6), round(receiver_lat, 6), round(receiver_lon, 6)

        def stage_device(index: int) -> str:
            if device_reuse:
                return device
            return hashlib.sha256(f"{correlation_id}-device-{index}".encode()).hexdigest()[:16]

        def event_timestamp(index: int) -> int:
            step = max(1, int((velocity_minutes * 60) / max(count, 1)))
            jitter = rng.randint(0, min(45, step))
            return now + index * step + jitter

        def amount_for(index: int, multiplier: float = 1.0) -> int:
            if risk_bias == "stealth":
                baseline = max(min_amount, min(max_amount, base_amount - (index % 4) * rng.randint(250, 1700)))
            elif risk_bias == "aggressive":
                baseline = min(max_amount, int(base_amount * (1.08 ** min(index, 6))) + rng.randint(0, 8500))
            else:
                baseline = max(min_amount, min(max_amount, int(base_amount * (0.96 ** max(index, 0))) + rng.randint(-1600, 1600)))
            return max(100, int(baseline * multiplier))

        def summary_flags(index: int, role: str, amount_inr: int, channel: Channel) -> list[str]:
            flags = []
            if device_reuse and index > 0:
                flags.append("shared_device")
            if velocity_minutes <= 15:
                flags.append("high_velocity")
            if role in {"mule layering hop", "cash-out consolidation", "layering hop"}:
                flags.append("layering")
            if channel in {Channel.RTGS, Channel.NEFT} and amount_inr >= 200_000:
                flags.append("high_value_rail")
            if normalized["customer_profile"] in {"dormant", "senior", "student"} and amount_inr >= 75_000:
                flags.append("profile_mismatch")
            if risk_bias == "stealth":
                flags.append("threshold_avoidance")
            return flags

        def txn(index: int, sender: str, receiver: str, amount_inr: int, channel: Channel, role: str) -> None:
            ts = event_timestamp(index)
            txn_id = _stable_id("TXN", correlation_id, index, sender, receiver, amount_inr, length=12).replace("-", "")
            amount_paisa = _rupees_to_paisa(amount_inr)
            sender_lat, sender_lon, receiver_lat, receiver_lon = event_geo(index)
            current_device = stage_device(index)
            checksum = compute_transaction_checksum(txn_id, ts, sender, receiver, amount_paisa, int(channel))
            event = Transaction(
                txn_id=txn_id,
                timestamp=ts,
                sender_id=sender,
                receiver_id=receiver,
                amount_paisa=amount_paisa,
                channel=channel,
                sender_branch=str(origin_geo["branch"]),
                receiver_branch=str(destination_geo["branch"]),
                sender_geo_lat=sender_lat,
                sender_geo_lon=sender_lon,
                receiver_geo_lat=receiver_lat,
                receiver_geo_lon=receiver_lon,
                device_fingerprint=current_device,
                sender_account_type=AccountType.SAVINGS if sender.startswith("UBI") else AccountType.CURRENT,
                receiver_account_type=AccountType.CURRENT if receiver.startswith(("MULE", "SHELL")) else AccountType.SAVINGS,
                checksum=checksum,
                fraud_label=fraud,
            )
            events.append(event)
            flags = summary_flags(index, role, amount_inr, channel)
            summaries.append(
                {
                    "type": "transaction",
                    "txn_id": txn_id,
                    "event_id": txn_id,
                    "sequence": index,
                    "timestamp": ts,
                    "sender": sender,
                    "receiver": receiver,
                    "amount_paisa": amount_paisa,
                    "channel": _channel_name(channel),
                    "fraud_label": fraud.name,
                    "device_fingerprint": current_device,
                    "geo_lat": sender_lat,
                    "geo_lon": sender_lon,
                    "receiver_geo_lat": receiver_lat,
                    "receiver_geo_lon": receiver_lon,
                    "origin_city": origin_geo["city"],
                    "origin_state": origin_geo["state"],
                    "destination_city": destination_geo["city"],
                    "destination_state": destination_geo["state"],
                    "elapsed_minutes": round(max(0, ts - now) / 60, 1),
                    "velocity_minutes": velocity_minutes,
                    "customer_profile": normalized["customer_profile"],
                    "risk_bias": risk_bias,
                    "risk_flags": flags,
                    "counterparty_role": role,
                    "narrative": f"{role}: {sender} -> {receiver} via {channel.name} ({origin_geo['city']} to {destination_geo['city']})",
                }
            )

        def auth(index: int, account: str, action: AuthAction, success: bool) -> None:
            ts = event_timestamp(index)
            event_id = _stable_id("AUTH", correlation_id, index, account, action.name, length=12).replace("-", "")
            ip = f"49.{rng.randint(10, 250)}.{rng.randint(10, 250)}.{rng.randint(2, 250)}"
            lat, lon, _, _ = event_geo(index)
            current_device = stage_device(index)
            checksum = compute_auth_checksum(event_id, ts, account, int(action), ip)
            event = AuthEvent(
                event_id=event_id,
                timestamp=ts,
                account_id=account,
                action=action,
                ip_address=ip,
                geo_lat=lat,
                geo_lon=lon,
                device_fingerprint=current_device,
                user_agent_hash=hashlib.sha256(f"PayFlowEventLab/{correlation_id}".encode()).hexdigest()[:16],
                success=success,
                checksum=checksum,
            )
            events.append(event)
            summaries.append(
                {
                    "type": "auth",
                    "event_id": event_id,
                    "sequence": index,
                    "timestamp": ts,
                    "account": account,
                    "action": action.name,
                    "success": success,
                    "ip": ip,
                    "device_fingerprint": current_device,
                    "geo_lat": lat,
                    "geo_lon": lon,
                    "origin_city": origin_geo["city"],
                    "origin_state": origin_geo["state"],
                    "customer_profile": normalized["customer_profile"],
                    "risk_flags": ["auth_anomaly", "credential_precursor"] if not success else ["auth_recovered", "credential_precursor"],
                    "counterparty_role": "credential precursor",
                    "narrative": f"{action.name} {'success' if success else 'failure'} before transfer activity",
                }
            )

        def interbank(index: int, sender: str, receiver: str, amount_inr: int, channel: Channel) -> None:
            ts = event_timestamp(index)
            msg_id = _stable_id("MSG", correlation_id, index, sender, receiver, amount_inr, length=12).replace("-", "")
            sender_ifsc = f"UBIN0{rng.randint(100000, 999999)}"
            receiver_ifsc = f"{rng.choice(['SBIN', 'HDFC', 'ICIC', 'PUNB'])}0{rng.randint(100000, 999999)}"
            amount_paisa = _rupees_to_paisa(amount_inr)
            lat, lon, receiver_lat, receiver_lon = event_geo(index)
            current_device = stage_device(index)
            checksum = compute_interbank_checksum(msg_id, ts, sender_ifsc, receiver_ifsc, amount_paisa, int(channel))
            event = InterbankMessage(
                msg_id=msg_id,
                timestamp=ts,
                sender_ifsc=sender_ifsc,
                receiver_ifsc=receiver_ifsc,
                sender_account=sender,
                receiver_account=receiver,
                amount_paisa=amount_paisa,
                currency_code=356,
                channel=channel,
                message_type="N06" if channel in {Channel.NEFT, Channel.RTGS} else "MT103",
                sender_geo_lat=lat,
                sender_geo_lon=lon,
                device_fingerprint=current_device,
                priority=1,
                checksum=checksum,
            )
            events.append(event)
            summaries.append(
                {
                    "type": "interbank",
                    "msg_id": msg_id,
                    "event_id": msg_id,
                    "sequence": index,
                    "timestamp": ts,
                    "sender": sender,
                    "receiver": receiver,
                    "sender_ifsc": sender_ifsc,
                    "receiver_ifsc": receiver_ifsc,
                    "amount_paisa": amount_paisa,
                    "channel": _channel_name(channel),
                    "message_type": event.message_type,
                    "device_fingerprint": current_device,
                    "geo_lat": lat,
                    "geo_lon": lon,
                    "receiver_geo_lat": receiver_lat,
                    "receiver_geo_lon": receiver_lon,
                    "origin_city": origin_geo["city"],
                    "origin_state": origin_geo["state"],
                    "destination_city": destination_geo["city"],
                    "destination_state": destination_geo["state"],
                    "risk_flags": ["interbank_exit", "settlement_leg"],
                    "counterparty_role": "interbank settlement leg",
                    "narrative": f"Interbank {channel.name} settlement leg",
                }
            )

        if template["template_id"] == "kyc_apk_phishing" or normalized["include_auth_signal"]:
            auth(0, victim, AuthAction.OTP_FAIL, False)
            auth(1, victim, AuthAction.OTP_VERIFY, True)
            start = 2
        else:
            start = 0

        if template["template_id"] in {"round_trip_shell_loop", "investment_scam_layering"}:
            chain = [origin] + shell_accounts[: max(2, count - 1)] + ([origin] if template["template_id"] == "round_trip_shell_loop" else [mule_accounts[0]])
            for i in range(start, min(count + start, len(chain) - 1 + start)):
                sender = chain[i - start]
                receiver = chain[i - start + 1]
                channel = primary_channel if i % 3 == 0 else secondary_channel
                txn(i, sender, receiver, max(min_amount, amount_for(i - start, 1 - min(i, 5) * 0.03)), channel, "layering hop")
        elif template["template_id"] in {"profile_mismatch_rtgs", "dormant_activation_high_value"}:
            source = victim if template["template_id"] != "dormant_activation_high_value" else f"DORM{rng.randint(10_000_000_000, 99_999_999_999)}"
            txn(start, source, mule_accounts[0], amount_for(0, 1.0), primary_channel if primary_channel in {Channel.RTGS, Channel.NEFT} else Channel.RTGS, "high-value anomaly")
            if mode != "single" or count > 1:
                txn(start + 1, mule_accounts[0], shell_accounts[0], amount_for(1, 0.88), secondary_channel if secondary_channel in {Channel.NEFT, Channel.RTGS, Channel.IMPS} else Channel.NEFT, "post-transfer consolidation")
        else:
            for i in range(start, start + count):
                if i == start:
                    sender, receiver, role = victim, mule_accounts[0], "initial victim transfer"
                elif i % 4 == 0:
                    sender, receiver, role = mule_accounts[(i - start - 1) % len(mule_accounts)], shell_accounts[(i - start) % len(shell_accounts)], "cash-out consolidation"
                else:
                    sender, receiver, role = mule_accounts[(i - start - 1) % len(mule_accounts)], mule_accounts[(i - start) % len(mule_accounts)], "mule layering hop"
                channel = primary_channel if i % 2 == 0 else secondary_channel
                amount = amount_for(i - start)
                txn(i, sender, receiver, amount, channel, role)
            if template["template_id"] == "merchant_qr_misuse" or normalized["include_interbank_leg"]:
                interbank(start + count, mule_accounts[0], shell_accounts[0], min(max_amount * 3, max(base_amount * 3, amount_for(count, 1.6))), Channel.NEFT)

        event_ids = [
            str(item.get("event_id") or item.get("txn_id") or item.get("msg_id"))
            for item in summaries
        ]
        counts = self._event_type_counts(summaries)
        return {
            "correlation_id": correlation_id,
            "mode": mode,
            "controls": normalized,
            "events": events,
            "summaries": summaries,
            "event_ids": event_ids,
            "analysis_counts": counts,
            "focus_account": summaries[-1].get("receiver") or summaries[-1].get("account"),
            "focus_event": event_ids[0] if event_ids else "",
        }

    def _normalize_controls(
        self,
        template: dict[str, Any],
        mode: str,
        intensity: str,
        controls: dict[str, Any] | None,
    ) -> dict[str, Any]:
        raw = controls or {}
        base_count = {"single": 1, "burst": 5, "chain": 7}.get(mode, 5)
        if intensity == "scale":
            base_count = min(24, base_count * 3)
        event_count = _clamp_int(raw.get("event_count"), base_count, 1, 30)
        if mode == "single":
            event_count = _clamp_int(raw.get("event_count"), base_count, 1, 4)

        base_amounts = {
            "structuring_below_threshold": (42_000, 49_900),
            "merchant_qr_misuse": (1_500, 12_500),
            "dormant_activation_high_value": (550_000, 1_800_000),
            "profile_mismatch_rtgs": (650_000, 2_500_000),
            "investment_scam_layering": (175_000, 900_000),
            "round_trip_shell_loop": (250_000, 1_100_000),
            "loan_app_extortion": (9_000, 48_000),
        }
        default_min, default_max = base_amounts.get(template["template_id"], (24_000, 160_000))
        min_amount = _clamp_int(raw.get("min_amount_inr"), default_min, 100, 5_000_000)
        max_amount = _clamp_int(raw.get("max_amount_inr"), default_max, 100, 10_000_000)
        if min_amount > max_amount:
            min_amount, max_amount = max_amount, min_amount

        template_channels = template.get("channels") or ["UPI", "IMPS"]
        primary = _channel_from_name(raw.get("primary_channel"), _channel_from_name(template_channels[0], Channel.UPI))
        secondary = _channel_from_name(raw.get("secondary_channel"), _channel_from_name(template_channels[-1], Channel.IMPS))
        default_origin, default_destination = TEMPLATE_ROUTE_DEFAULTS.get(template["template_id"], ("mumbai", "delhi"))
        origin_region = str(raw.get("origin_region") or default_origin).lower()
        destination_region = str(raw.get("destination_region") or default_destination).lower()
        if origin_region not in REGION_PROFILES:
            origin_region = default_origin
        if destination_region not in REGION_PROFILES:
            destination_region = default_destination
        if origin_region == destination_region:
            destination_region = default_destination if destination_region != default_destination else "delhi"

        risk_bias = str(raw.get("risk_bias") or "balanced").lower()
        if risk_bias not in {"balanced", "stealth", "aggressive"}:
            risk_bias = "balanced"
        customer_profile = str(raw.get("customer_profile") or self._default_customer_profile(template)).lower()
        if customer_profile not in PROFILE_LABELS:
            customer_profile = self._default_customer_profile(template)

        return {
            "event_count": event_count,
            "amount_band_inr": [min_amount, max_amount],
            "primary_channel": primary.name,
            "secondary_channel": secondary.name,
            "origin_region": origin_region,
            "destination_region": destination_region,
            "route_label": f"{REGION_PROFILES[origin_region]['city']} -> {REGION_PROFILES[destination_region]['city']}",
            "velocity_minutes": _clamp_int(raw.get("velocity_minutes"), 18 if intensity == "scale" else 45, 1, 360),
            "mule_depth": _clamp_int(raw.get("mule_depth"), 5 if mode == "chain" else 3, 1, 14),
            "device_reuse": bool(raw.get("device_reuse", True)),
            "include_auth_signal": bool(raw.get("include_auth_signal", template["template_id"] == "kyc_apk_phishing")),
            "include_interbank_leg": bool(raw.get("include_interbank_leg", template["template_id"] in {"merchant_qr_misuse", "investment_scam_layering", "round_trip_shell_loop"})),
            "customer_profile": customer_profile,
            "customer_profile_label": PROFILE_LABELS[customer_profile],
            "risk_bias": risk_bias,
        }

    def _default_customer_profile(self, template: dict[str, Any]) -> str:
        template_id = template["template_id"]
        if "merchant" in template_id:
            return "merchant"
        if "dormant" in template_id:
            return "dormant"
        if "profile" in template_id:
            return "salary"
        if "student" in template_id or "mule" in template_id:
            return "student"
        if "shell" in template_id or "round_trip" in template_id:
            return "shell"
        if "kyc" in template_id:
            return "senior"
        return "salary"

    def _event_type_counts(self, summaries: list[dict[str, Any]]) -> dict[str, int]:
        return {
            "transactions": len([s for s in summaries if s.get("type") == "transaction"]),
            "auth": len([s for s in summaries if s.get("type") == "auth"]),
            "interbank": len([s for s in summaries if s.get("type") == "interbank"]),
        }

    def _build_analysis_report(
        self,
        template: dict[str, Any],
        generated: dict[str, Any],
        linked_intel: dict[str, Any],
        proposals: list[dict[str, Any]] | None = None,
        run: EventLabRun | None = None,
    ) -> dict[str, Any]:
        summaries = list(generated.get("summaries") or generated.get("events") or [])
        controls = dict(generated.get("controls") or {})
        proposal_rows = proposals or []
        transactions = [s for s in summaries if s.get("type") == "transaction"]
        auth_events = [s for s in summaries if s.get("type") == "auth"]
        interbank_messages = [s for s in summaries if s.get("type") == "interbank"]
        total_amount = sum(int(s.get("amount_paisa") or 0) for s in summaries)
        unique_accounts = {
            str(value)
            for s in summaries
            for value in (s.get("sender"), s.get("receiver"), s.get("account"))
            if value
        }

        def count_by(key: str) -> dict[str, int]:
            counts: dict[str, int] = {}
            for summary in summaries:
                value = str(summary.get(key) or "unknown")
                counts[value] = counts.get(value, 0) + 1
            return counts

        def count_proposals_by(key: str) -> dict[str, int]:
            counts: dict[str, int] = {}
            for proposal in proposal_rows:
                value = str(proposal.get(key) or "unknown")
                counts[value] = counts.get(value, 0) + 1
            return counts

        channel_mix = count_by("channel")
        typology_mix = count_by("fraud_label")
        channel_amount_mix: dict[str, int] = {}
        account_role_mix = count_by("counterparty_role")
        risk_flags: dict[str, int] = {}
        for summary in summaries:
            channel = str(summary.get("channel") or summary.get("action") or summary.get("type") or "unknown")
            channel_amount_mix[channel] = channel_amount_mix.get(channel, 0) + int(summary.get("amount_paisa") or 0)
            for flag in summary.get("risk_flags") or []:
                risk_flags[str(flag)] = risk_flags.get(str(flag), 0) + 1

        amount_series = [
            {
                "sequence": int(s.get("sequence") or index),
                "event_id": s.get("event_id") or s.get("txn_id") or s.get("msg_id"),
                "timestamp": s.get("timestamp"),
                "sender": s.get("sender") or s.get("account"),
                "receiver": s.get("receiver") or s.get("account"),
                "amount_paisa": int(s.get("amount_paisa") or 0),
                "channel": s.get("channel") or s.get("action") or s.get("type"),
                "role": s.get("counterparty_role") or s.get("type"),
                "risk_flags": list(s.get("risk_flags") or []),
            }
            for index, s in enumerate(summaries)
            if s.get("amount_paisa")
        ]
        geo_path = [
            {
                "event_id": s.get("event_id") or s.get("txn_id") or s.get("msg_id"),
                "sequence": int(s.get("sequence") or index),
                "lat": s.get("receiver_geo_lat") or s.get("geo_lat"),
                "lon": s.get("receiver_geo_lon") or s.get("geo_lon"),
                "city": s.get("destination_city") or s.get("origin_city") or "unknown",
                "state": s.get("destination_state") or s.get("origin_state") or "unknown",
                "role": s.get("counterparty_role") or s.get("type"),
                "amount_paisa": int(s.get("amount_paisa") or 0),
            }
            for index, s in enumerate(summaries)
            if s.get("geo_lat") is not None or s.get("receiver_geo_lat") is not None
        ]
        route_segments = [
            {
                "sequence": int(s.get("sequence") or index),
                "event_id": s.get("event_id") or s.get("txn_id") or s.get("msg_id"),
                "timestamp": s.get("timestamp"),
                "from_account": s.get("sender") or s.get("account"),
                "to_account": s.get("receiver") or s.get("account"),
                "from_city": s.get("origin_city") or "unknown",
                "from_state": s.get("origin_state") or "unknown",
                "to_city": s.get("destination_city") or s.get("origin_city") or "unknown",
                "to_state": s.get("destination_state") or s.get("origin_state") or "unknown",
                "from_lat": s.get("geo_lat"),
                "from_lon": s.get("geo_lon"),
                "to_lat": s.get("receiver_geo_lat") or s.get("geo_lat"),
                "to_lon": s.get("receiver_geo_lon") or s.get("geo_lon"),
                "amount_paisa": int(s.get("amount_paisa") or 0),
                "channel": s.get("channel") or s.get("action") or s.get("type"),
                "role": s.get("counterparty_role") or s.get("type"),
                "elapsed_minutes": float(s.get("elapsed_minutes") or 0.0),
                "risk_flags": list(s.get("risk_flags") or []),
            }
            for index, s in enumerate(summaries)
            if s.get("amount_paisa")
        ]
        velocity_series = [
            {
                "sequence": int(s.get("sequence") or index),
                "elapsed_minutes": float(s.get("elapsed_minutes") or 0.0),
                "event_id": s.get("event_id") or s.get("txn_id") or s.get("msg_id"),
            }
            for index, s in enumerate(summaries)
        ]

        trust = float(linked_intel.get("trust_score") or 0.0)
        velocity_minutes = int(controls.get("velocity_minutes") or 60)
        mule_depth = int(controls.get("mule_depth") or 1)
        device_reuse = bool(controls.get("device_reuse"))
        risk_bias = str(controls.get("risk_bias") or "balanced")
        amount_score = min(0.20, (total_amount / 100) / 2_500_000 * 0.20)
        velocity_score = 0.16 if velocity_minutes <= 10 else 0.10 if velocity_minutes <= 30 else 0.04
        graph_score = min(0.18, max(0, mule_depth - 1) * 0.025 + len(unique_accounts) * 0.006)
        auth_score = 0.08 if auth_events else 0.0
        interbank_score = 0.07 if interbank_messages else 0.0
        device_score = 0.07 if device_reuse and len(transactions) >= 3 else 0.02
        bias_score = {"stealth": 0.09, "aggressive": 0.11, "balanced": 0.06}.get(risk_bias, 0.06)
        trust_score = trust * 0.08
        risk_components = {
            "base_profile": 0.31,
            "amount_exposure": round(amount_score, 3),
            "velocity_pressure": round(velocity_score, 3),
            "graph_depth": round(graph_score, 3),
            "auth_anomaly": round(auth_score, 3),
            "interbank_exit": round(interbank_score, 3),
            "device_reuse": round(device_score, 3),
            "scenario_bias": round(bias_score, 3),
            "intel_trust": round(trust_score, 3),
        }
        risk_score = min(0.99, round(sum(risk_components.values()), 3))
        if risk_score >= 0.86:
            tier = "critical"
            verdict = "fraudulent - analyst gated countermeasure required"
        elif risk_score >= 0.72:
            tier = "high"
            verdict = "suspicious - escalate to fraud analyst"
        elif risk_score >= 0.55:
            tier = "medium"
            verdict = "watchlisted - monitor and enrich evidence"
        else:
            tier = "elevated"
            verdict = "monitor - insufficient for autonomous action"

        proposal_status = count_proposals_by("status")
        executed = proposal_status.get("executed", 0)
        rejected = proposal_status.get("rejected", 0)
        pending = proposal_status.get("proposed", 0)
        report_stages = self._display_stages(run) if run else []
        stage_names = [stage.stage for stage in report_stages]
        stage_coverage = {
            name: stage_names.count(name)
            for name in [
                "intel_primed",
                "events_generated",
                "events_injected",
                "ingested",
                "pipeline_dispatched",
                "ml_scored",
                "graph_investigated",
                "cb_evaluated",
                "qwen_context_loaded",
                "evaluation_complete",
                "analyst_decision",
                "action_executed",
                "ledger_anchored",
                "evidence_ready",
            ]
        }

        timeline_buckets: list[dict[str, Any]] = []
        monetary_by_elapsed = [
            (float(s.get("elapsed_minutes") or 0.0), int(s.get("amount_paisa") or 0), str(s.get("channel") or s.get("type") or "unknown"))
            for s in summaries
            if s.get("amount_paisa")
        ]
        if monetary_by_elapsed:
            bucket_count = 6
            span = max(1.0, max(item[0] for item in monetary_by_elapsed))
            bucket_width = max(1.0, span / bucket_count)
            for bucket_index in range(bucket_count):
                start = bucket_index * bucket_width
                end = start + bucket_width
                rows = [item for item in monetary_by_elapsed if start <= item[0] < end or (bucket_index == bucket_count - 1 and item[0] <= end)]
                timeline_buckets.append(
                    {
                        "bucket": bucket_index + 1,
                        "start_minute": round(start, 2),
                        "end_minute": round(end, 2),
                        "event_count": len(rows),
                        "amount_paisa": sum(item[1] for item in rows),
                        "channels": {},
                    }
                )
            for bucket in timeline_buckets:
                start = float(bucket["start_minute"])
                end = float(bucket["end_minute"])
                channels: dict[str, int] = {}
                for elapsed, _, channel in monetary_by_elapsed:
                    if start <= elapsed < end or (bucket["bucket"] == bucket_count and elapsed <= end):
                        channels[channel] = channels.get(channel, 0) + 1
                bucket["channels"] = channels

        route_stats = {
            "first_city": geo_path[0]["city"] if geo_path else "unknown",
            "last_city": geo_path[-1]["city"] if geo_path else "unknown",
            "unique_geo_points": len({(row.get("lat"), row.get("lon")) for row in geo_path}),
            "max_leg_amount_paisa": max((int(row.get("amount_paisa") or 0) for row in amount_series), default=0),
            "avg_leg_amount_paisa": round(sum(int(row.get("amount_paisa") or 0) for row in amount_series) / max(1, len(amount_series))),
            "elapsed_minutes": velocity_minutes,
            "segment_count": len(route_segments),
            "interbank_exit_count": len(interbank_messages),
        }
        geo_lats = [float(row["lat"]) for row in geo_path if row.get("lat") is not None]
        geo_lons = [float(row["lon"]) for row in geo_path if row.get("lon") is not None]
        geo_bounds = {
            "min_lat": min(geo_lats) if geo_lats else None,
            "max_lat": max(geo_lats) if geo_lats else None,
            "min_lon": min(geo_lons) if geo_lons else None,
            "max_lon": max(geo_lons) if geo_lons else None,
            "center_lat": round(sum(geo_lats) / len(geo_lats), 6) if geo_lats else None,
            "center_lon": round(sum(geo_lons) / len(geo_lons), 6) if geo_lons else None,
        }

        stage_timeline = [
            {
                "sequence": index + 1,
                "stage": stage.stage,
                "label": self._stage_label(stage.stage),
                "timestamp": stage.timestamp,
                "duration_ms": stage.duration_ms,
                "event_count": len(stage.event_ids),
                "status": stage.status,
                "source": "backend_sse",
                "batch_count": int(stage.meta.get("batch_count") or 1),
                "latest_observed_at": stage.meta.get("latest_observed_at", stage.timestamp),
            }
            for index, stage in enumerate(report_stages)
        ]
        countermeasure_matrix = [
            {
                "proposal_id": proposal.get("proposal_id"),
                "action": proposal.get("action"),
                "status": proposal.get("status"),
                "title": proposal.get("title"),
                "target_count": len(proposal.get("targets") or []),
                "primary_target": (proposal.get("targets") or [""])[0] if isinstance(proposal.get("targets"), list) else "",
                "execution_allowed": bool(proposal.get("execution_allowed")),
                "rollback_available": bool(proposal.get("rollback_available")),
                "ttl_remaining_seconds": max(0, int(float(proposal.get("expires_at") or 0) - _now())),
                "audit_hash": proposal.get("audit_hash"),
            }
            for proposal in proposal_rows
        ]
        strongest_flags = sorted(risk_flags.items(), key=lambda item: item[1], reverse=True)[:6]
        evidence_matrix = [
            {
                "signal": label,
                "source": "event_heuristic",
                "count": count,
                "weight": round(min(1.0, 0.32 + count / max(1, len(summaries))), 3),
                "basis": f"{count} generated event summaries carried this risk flag",
            }
            for label, count in strongest_flags
        ]
        evidence_matrix.extend(
            {
                "signal": label,
                "source": "risk_model_component",
                "count": 1,
                "weight": value,
                "basis": "contributes directly to the derived run risk score",
            }
            for label, value in risk_components.items()
            if value > 0
        )
        evidence_matrix.extend(
            {
                "signal": stage,
                "source": "backend_stage",
                "count": count,
                "weight": 1.0 if count > 0 else 0.0,
                "basis": "observed in recorded Event Lab backend stage timeline",
            }
            for stage, count in stage_coverage.items()
            if count > 0
        )

        next_steps = [
            "Open fund-flow graph around the highest-value receiver and immediate one-hop neighbors.",
            "Hold or freeze only through analyst-approved proposals when internal evidence supports the action.",
            "Generate FIU evidence package after verdict and countermeasure decision are recorded.",
        ]
        if proposal_rows:
            next_steps = [
                str(p.get("title") or p.get("action"))
                for p in proposal_rows[:4]
            ]

        return {
            "verdict": verdict,
            "risk_score": risk_score,
            "risk_tier": tier,
            "confidence": min(0.97, round(0.62 + len(strongest_flags) * 0.035 + trust * 0.18, 3)),
            "total_exposure_paisa": total_amount,
            "event_count": len(summaries),
            "transaction_count": len(transactions),
            "auth_event_count": len(auth_events),
            "interbank_count": len(interbank_messages),
            "unique_account_count": len(unique_accounts),
            "channel_mix": channel_mix,
            "channel_amount_mix": channel_amount_mix,
            "account_role_mix": account_role_mix,
            "typology_mix": typology_mix,
            "risk_flags": dict(strongest_flags),
            "amount_series": amount_series,
            "velocity_series": velocity_series,
            "geo_path": geo_path,
            "route_segments": route_segments,
            "geo_bounds": geo_bounds,
            "route_label": controls.get("route_label") or "",
            "controls": controls,
            "stage_coverage": stage_coverage,
            "stage_timeline": stage_timeline,
            "risk_score_components": risk_components,
            "timeline_buckets": timeline_buckets,
            "route_stats": route_stats,
            "countermeasure_matrix": countermeasure_matrix,
            "evidence_matrix": evidence_matrix,
            "countermeasure_status": {"pending": pending, "executed": executed, "rejected": rejected},
            "evidence_strengths": {
                "heuristics": round(min(1, 0.45 + len(strongest_flags) * 0.07), 3),
                "ml_features": round(min(1, 0.50 + amount_score + velocity_score + device_score), 3),
                "graph_structure": round(min(1, 0.46 + graph_score + len(unique_accounts) * 0.008), 3),
                "qwen_explainability": round(min(1, 0.58 + trust * 0.20), 3),
                "analyst_gate": 1.0 if executed or rejected else 0.64,
            },
            "forensic_summary": (
                f"{template['title']} produced {len(summaries)} linked events over {velocity_minutes} minutes across "
                f"{controls.get('route_label', 'the selected route')}. PayFlow observed {len(unique_accounts)} unique accounts, "
                f"{len(transactions)} transaction legs, {len(auth_events)} auth signals, and {len(interbank_messages)} interbank exits. "
                f"The derived risk tier is {tier} because {', '.join(flag for flag, _ in strongest_flags[:3]) or 'the chain remains correlated'}."
            ),
            "recommended_next_steps": next_steps,
            "generated_at": _now(),
        }

    def _build_proposals(
        self,
        run: EventLabRun,
        template: dict[str, Any],
        generated: dict[str, Any],
        linked_intel: dict[str, Any],
    ) -> list[CountermeasureProposal]:
        allowed = bool(linked_intel.get("execution_allowed"))
        trust = float(linked_intel.get("trust_score") or 0.0)
        first_event = generated["summaries"][0] if generated["summaries"] else {}
        last_event = generated["summaries"][-1] if generated["summaries"] else {}
        primary_target = str(last_event.get("receiver") or last_event.get("account") or last_event.get("sender") or "unknown")
        device = str(first_event.get("device_fingerprint") or "")
        total_amount_paisa = sum(int(item.get("amount_paisa") or 0) for item in generated["summaries"])
        channel_mix: dict[str, int] = {}
        risk_flags: dict[str, int] = {}
        for item in generated["summaries"]:
            channel = str(item.get("channel") or item.get("action") or item.get("type"))
            channel_mix[channel] = channel_mix.get(channel, 0) + 1
            for flag in item.get("risk_flags") or []:
                risk_flags[str(flag)] = risk_flags.get(str(flag), 0) + 1
        base = {
            "run_id": run.run_id,
            "status": "proposed",
            "trigger_event_ids": generated["event_ids"][:4],
            "risk_evidence": {
                "expected_indicators": template["expected_indicators"],
                "source_trust": trust,
                "event_count": len(generated["event_ids"]),
                "typologies": template["typologies"],
                "total_exposure_paisa": total_amount_paisa,
                "channel_mix": channel_mix,
                "route": generated["controls"].get("route_label"),
                "velocity_minutes": generated["controls"].get("velocity_minutes"),
                "mule_depth": generated["controls"].get("mule_depth"),
                "customer_profile": generated["controls"].get("customer_profile_label"),
                "risk_flags": dict(sorted(risk_flags.items(), key=lambda item: item[1], reverse=True)[:6]),
                "decision_authority": "Requires analyst approval plus PayFlow internal evidence.",
            },
            "intel_context": linked_intel,
            "ttl_seconds": 900,
            "expires_at": _now() + 900,
            "execution_allowed": allowed,
            "rollback_available": True,
            "created_at": _now(),
            "updated_at": _now(),
        }
        actions = []
        for action in template["countermeasure_actions"][:4]:
            if action == "BAN_DEVICE" and not device:
                continue
            target = device if action == "BAN_DEVICE" else primary_target
            title = {
                "HOLD": "Hold suspicious transfer for analyst review",
                "FREEZE_NODE": "Freeze primary mule or beneficiary node",
                "FREEZE_1HOP": "Freeze immediate one-hop exposure",
                "PAUSE_ROUTING": "Pause routing around affected accounts",
                "BAN_DEVICE": "Ban phishing-linked device fingerprint",
                "WATCHLIST_DELTA": "Activate intel-derived watchlist terms",
                "CREATE_CASE": "Create fund-flow case workbench entry",
                "GENERATE_EVIDENCE": "Prepare FIU-ready evidence package",
            }.get(action, action)
            proposal = CountermeasureProposal(
                proposal_id=_stable_id("CMP", run.run_id, action, target, length=10),
                action=action,
                title=title,
                reason=(
                    f"{template['title']} generated {len(generated['event_ids'])} correlated events across "
                    f"{generated['controls'].get('route_label')} with INR {total_amount_paisa / 100:,.0f} exposure, "
                    f"{generated['controls'].get('velocity_minutes')} minute velocity, and source trust {trust:.2f}."
                ),
                targets=[str(target)],
                **base,
            )
            proposal.audit_hash = _hash_payload(proposal.to_dict() | {"audit_hash": ""})
            actions.append(proposal)
        return actions

    async def _execute_countermeasure(self, proposal: CountermeasureProposal, orchestrator: Any) -> dict[str, Any]:
        if orchestrator is None:
            raise RuntimeError("Orchestrator not initialized")
        breaker = getattr(orchestrator, "_breaker", None)
        ledger = getattr(orchestrator, "_ledger", None)
        target = proposal.targets[0] if proposal.targets else "unknown"
        trigger = proposal.trigger_event_ids[0] if proposal.trigger_event_ids else proposal.run_id

        if proposal.action in {"FREEZE_NODE", "FREEZE_1HOP"}:
            if breaker is None:
                raise RuntimeError("Circuit breaker not initialized")
            from src.blockchain.circuit_breaker import FreezeOrder

            order = FreezeOrder(
                node_id=target,
                freeze_timestamp=_now(),
                trigger_txn_id=trigger,
                ml_risk_score=0.91,
                gnn_risk_score=-1.0,
                graph_evidence_score=0.86,
                consensus_score=0.94,
                reason=f"Analyst approved adaptive event-lab response: {proposal.reason}",
                ttl_seconds=getattr(breaker._cfg, "freeze_ttl_seconds", 600),
            )
            await breaker.freeze_node(order)
            frozen_neighbors: list[str] = []
            if proposal.action == "FREEZE_1HOP" and getattr(breaker, "_graph", None) is not None:
                await breaker._freeze_1hop_neighbors(target, trigger, 0.91, -1.0, 0.86, 0.94)
                try:
                    frozen_neighbors = [
                        nid for nid in breaker._graph.graph.neighbors(target)
                        if breaker.is_frozen(nid)
                    ]
                except Exception:
                    frozen_neighbors = []
            return {"status": "frozen", "target": target, "frozen_neighbors": frozen_neighbors}

        if proposal.action == "PAUSE_ROUTING":
            listener = getattr(breaker, "_agent_listener", None) if breaker else None
            if listener is None:
                return {"status": "recorded", "target": target, "note": "agent routing listener unavailable"}
            paused = listener._pause_routing(proposal.targets)
            return {"status": "routing_paused", "targets": paused}

        if proposal.action == "BAN_DEVICE":
            listener = getattr(breaker, "_agent_listener", None) if breaker else None
            if listener is None:
                return {"status": "recorded", "target": target, "note": "device ban listener unavailable"}
            listener._ban_device(target, trigger, proposal.risk_evidence.get("typologies", ["UNKNOWN"])[0])
            return {"status": "device_banned", "device_fingerprint": target}

        if proposal.action == "HOLD":
            return {"status": "held_for_review", "target": target, "gate": "analyst_hold_queue"}

        if proposal.action == "WATCHLIST_DELTA":
            terms = []
            playbook = proposal.intel_context.get("playbook") or {}
            terms = list(playbook.get("watchlist_terms") or [])[:10]
            return {"status": "watchlist_delta_active", "terms": terms, "ttl_seconds": proposal.ttl_seconds}

        if proposal.action in {"CREATE_CASE", "GENERATE_EVIDENCE"}:
            return {"status": "case_context_ready", "run_id": proposal.run_id, "evidence_anchor": bool(ledger)}

        return {"status": "recorded", "action": proposal.action, "target": target}

    async def _anchor_countermeasure(self, orchestrator: Any, action: str, proposal: CountermeasureProposal) -> None:
        ledger = getattr(orchestrator, "_ledger", None) if orchestrator else None
        if ledger is None:
            return
        try:
            await ledger.anchor_circuit_breaker(action=action, details=proposal.to_dict())
        except Exception:
            return

    def _countermeasure_policy(self, linked_intel: dict[str, Any]) -> dict[str, Any]:
        return {
            "authority": "analyst_approval_required",
            "execution_allowed": bool(linked_intel.get("execution_allowed")),
            "source_trust": linked_intel.get("trust_score", 0.0),
            "qwen_role": "bounded investigator explanation only",
            "decision_authority": "graph_ml_rules_ledger_pipeline",
        }

    def _qwen_explanation(
        self,
        template: dict[str, Any],
        linked_intel: dict[str, Any],
        controls: dict[str, Any] | None = None,
        generated: dict[str, Any] | None = None,
    ) -> str:
        playbook = linked_intel.get("playbook") or {}
        trend = linked_intel.get("trend") or {}
        title = playbook.get("title") or trend.get("title") or "no active playbook"
        controls = controls or {}
        count = len(generated.get("event_ids") or []) if generated else controls.get("event_count", "selected")
        route = controls.get("route_label") or "the selected branch corridor"
        channel = "/".join(
            part
            for part in [
                str(controls.get("primary_channel") or ""),
                str(controls.get("secondary_channel") or ""),
            ]
            if part
        ) or "selected payment rails"
        velocity = controls.get("velocity_minutes") or "configured"
        profile = controls.get("customer_profile_label") or "selected customer profile"
        return (
            f"Context guardrail for {OLLAMA_CFG.model}: preventive signal '{title}' is available while reviewing "
            f"{template['title']} with {count} generated events on {channel}, route {route}, {velocity} minute velocity, "
            f"and {profile}. The model explains why the pattern is risky and how evidence maps to analyst language; "
            "it cannot execute holds, freezes, routing pauses, or threshold changes without PayFlow ML/graph evidence and analyst approval."
        )

    def _explainability_stage_groups(self, run_body: dict[str, Any]) -> list[dict[str, Any]]:
        definitions = [
            (
                "pre_fraud_intel",
                "Pre-fraud intelligence priming",
                "Trusted external signals choose the typology, watch terms, scenario seed, and Qwen context.",
                {"intel_primed", "events_generated"},
            ),
            (
                "event_ingestion",
                "Event generation and ingestion",
                "Banking scenario events are injected through the same PayFlow ingestion path as live events.",
                {"events_injected", "ingested", "pipeline_dispatched"},
            ),
            (
                "rules_ml_graph",
                "Rules, ML, and graph evidence",
                "Internal scoring layers validate the event chain before any response proposal is actionable.",
                {"ml_scored", "graph_investigated", "cb_evaluated"},
            ),
            (
                "qwen_context",
                "Qwen 3.5 4B context guardrail",
                "The local model receives bounded context, while graph, ML, rules, ledger, and analyst approval remain authoritative.",
                {"llm_started", "qwen_context_loaded", "qwen_tool_call"},
            ),
            (
                "analyst_gate",
                "Analyst countermeasure gate",
                "Countermeasure proposals wait for explicit analyst approval or rejection.",
                {"analyst_decision"},
            ),
            (
                "execution_audit",
                "Execution, ledger, and evidence",
                "Approved actions execute through PayFlow controls and leave an audit/evidence trail.",
                {"evaluation_complete", "action_executed", "ledger_anchored", "evidence_ready"},
            ),
        ]
        grouped: dict[str, dict[str, Any]] = {
            key: {
                "group": key,
                "label": label,
                "description": description,
                "completed": False,
                "stage_count": 0,
                "latency_ms": 0.0,
                "stages": [],
            }
            for key, label, description, _ in definitions
        }
        stage_to_group = {
            stage: key
            for key, _, _, names in definitions
            for stage in names
        }

        for stage in run_body.get("stages", []):
            group_key = stage_to_group.get(stage.get("stage"), "event_ingestion")
            enriched = {
                **stage,
                "label": self._stage_label(stage.get("stage", "")),
                "group": group_key,
                "evidence_summary": self._stage_summary(stage),
            }
            bucket = grouped[group_key]
            bucket["stages"].append(enriched)
            bucket["stage_count"] += 1
            bucket["latency_ms"] = round(float(bucket["latency_ms"]) + float(stage.get("duration_ms") or 0), 2)

        for bucket in grouped.values():
            bucket["completed"] = bool(bucket["stages"])
        return list(grouped.values())

    def _explainability_evidence_panels(
        self,
        run_body: dict[str, Any],
        groups: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        linked_intel = run_body.get("linked_intel") or {}
        trend = linked_intel.get("trend") or {}
        playbook = linked_intel.get("playbook") or {}
        proposals = run_body.get("countermeasure_proposals", [])
        stage_index = {group["group"]: group for group in groups}
        return [
            {
                "key": "intel",
                "title": "Pre-fraud intel evidence",
                "status": "active" if linked_intel.get("playbook") else "shadow",
                "authority": "advisory only",
                "summary": linked_intel.get("guardrail", "Intel context is advisory and cannot execute controls."),
                "metrics": {
                    "trust_score": round(float(linked_intel.get("trust_score") or 0.0), 3),
                    "signals": len(linked_intel.get("signals") or []),
                    "playbook": playbook.get("playbook_id") or "none",
                },
                "items": [
                    playbook.get("title") or "No active applied playbook",
                    trend.get("title") or "No corroborated trend linked",
                    *[s.get("title", "external signal") for s in linked_intel.get("signals", [])[:3]],
                ],
            },
            {
                "key": "internal",
                "title": "Rules, ML, and graph evidence",
                "status": "processing" if stage_index["rules_ml_graph"]["completed"] else "waiting",
                "authority": "decision evidence",
                "summary": "Internal PayFlow evidence must support any adaptive response.",
                "metrics": {
                    "event_count": len(run_body.get("event_ids") or []),
                    "stage_count": stage_index["rules_ml_graph"]["stage_count"],
                    "known_latency_ms": stage_index["rules_ml_graph"]["latency_ms"],
                },
                "items": run_body.get("expected_indicators", [])[:5],
            },
            {
                "key": "qwen",
                "title": "Qwen bounded copilot",
                "status": "context_loaded",
                "authority": "explanation only",
                "summary": run_body.get("qwen_explanation", ""),
                "metrics": {
                    "model": OLLAMA_CFG.model,
                    "can_execute": "no",
                    "decision_authority": run_body.get("decision_authority", ""),
                },
                "items": [
                    "Maps external trend to typology language",
                    "Explains why proposals need analyst review",
                    "Does not freeze, hold, or change thresholds",
                ],
            },
            {
                "key": "countermeasure",
                "title": "Countermeasure lifecycle",
                "status": "analyst_gated",
                "authority": "analyst approval required",
                "summary": "Every adaptive response is reversible, TTL-bound, and ledger-audited.",
                "metrics": {
                    "pending": len([p for p in proposals if p.get("status") == "proposed"]),
                    "executed": len([p for p in proposals if p.get("status") == "executed"]),
                    "rejected": len([p for p in proposals if p.get("status") == "rejected"]),
                },
                "items": [f"{p.get('action')} -> {p.get('status')}" for p in proposals[:5]],
            },
            {
                "key": "audit",
                "title": "Ledger and evidence audit",
                "status": "anchored" if any(p.get("executed_at") for p in proposals) else "ready",
                "authority": "immutable audit trail",
                "summary": "Run and countermeasure hashes are retained for FIU-ready evidence packages.",
                "metrics": {
                    "run_hash": run_body.get("audit_hash", "")[:12],
                    "countermeasure_hashes": len([p for p in proposals if p.get("audit_hash")]),
                    "rollback_available": "yes" if any(p.get("rollback_available") for p in proposals) else "no",
                },
                "items": [p.get("audit_hash", "")[:18] for p in proposals if p.get("audit_hash")][:5],
            },
        ]

    def _proposal_explainability(self, proposal: dict[str, Any]) -> dict[str, Any]:
        ttl_remaining = max(0, int(float(proposal.get("expires_at") or 0) - _now()))
        return {
            "proposal_id": proposal.get("proposal_id"),
            "action": proposal.get("action"),
            "status": proposal.get("status"),
            "title": proposal.get("title"),
            "targets": proposal.get("targets", []),
            "analyst": proposal.get("analyst"),
            "analyst_reason": proposal.get("analyst_reason"),
            "execution_allowed": bool(proposal.get("execution_allowed")),
            "ttl_remaining_seconds": ttl_remaining,
            "rollback_available": bool(proposal.get("rollback_available")),
            "risk_evidence": proposal.get("risk_evidence", {}),
            "execution_result": proposal.get("execution_result", {}),
            "audit_hash": proposal.get("audit_hash", ""),
            "decision_summary": (
                "Executed through PayFlow controls"
                if proposal.get("status") == "executed"
                else "Rejected by analyst"
                if proposal.get("status") == "rejected"
                else "Awaiting analyst approval"
                if proposal.get("status") == "proposed"
                else str(proposal.get("status", "unknown")).replace("_", " ")
            ),
        }

    def _stage_label(self, stage: str) -> str:
        labels = {
            "intel_primed": "Intel playbook loaded",
            "events_generated": "Correlated event chain generated",
            "events_injected": "Injected into live ingestion pipeline",
            "ingested": "Schema validation and ingestion completed",
            "pipeline_dispatched": "Batch dispatched to consumers",
            "ml_scored": "ML and feature scoring completed",
            "graph_investigated": "Graph investigation completed",
            "cb_evaluated": "Circuit-breaker evidence evaluated",
            "llm_started": "Qwen context/explanation started",
            "analyst_decision": "Analyst decision recorded",
            "evaluation_complete": "Autonomous evaluation completed",
            "action_executed": "Countermeasure action executed",
            "ledger_anchored": "Ledger audit hash anchored",
            "evidence_ready": "Evidence package context ready",
        }
        return labels.get(stage, stage.replace("_", " ").title())

    def _stage_summary(self, stage: dict[str, Any]) -> str:
        meta = stage.get("meta") or {}
        stage_name = str(stage.get("stage") or "")
        if stage_name == "intel_primed":
            intel = meta.get("linked_intel") or {}
            trend = intel.get("trend") or {}
            return f"Linked trend: {trend.get('title') or 'none'}; trust {float(intel.get('trust_score') or 0):.2f}."
        if stage_name == "events_generated":
            return f"Generated {meta.get('count', len(stage.get('event_ids') or []))} correlated events."
        if stage_name == "events_injected":
            return f"Injected with {meta.get('proposal_count', 0)} analyst-gated proposals."
        if stage_name == "ml_scored":
            risk = meta.get("risk_score")
            return f"ML risk {risk}; tier {meta.get('tier', 'n/a')}." if risk is not None else "Feature and ML scoring completed."
        if stage_name == "pipeline_dispatched":
            return f"Dispatched to {len(meta.get('consumers') or [])} backend consumers."
        if stage_name == "evaluation_complete":
            return f"Final risk {meta.get('risk_tier', 'n/a')} at score {meta.get('risk_score', 'n/a')}; report can be opened."
        if stage_name == "analyst_decision":
            return f"Proposal {meta.get('proposal_id')} {meta.get('decision', 'recorded')}."
        if stage_name == "action_executed":
            return f"Execution result: {meta.get('status', 'recorded')}."
        if stage_name == "ledger_anchored":
            return f"Audit hash {str(meta.get('audit_hash', ''))[:18]}."
        return f"{len(stage.get('event_ids') or [])} linked event ids."

    async def _stage_for_proposal(self, proposal: CountermeasureProposal, stage: str, meta: dict[str, Any]) -> None:
        run = self._runs.get(proposal.run_id)
        if run:
            await self._record_run_stage(run, stage, event_ids=proposal.trigger_event_ids, meta={"proposal_id": proposal.proposal_id, **meta})

    def _run_stage_names(self, run: EventLabRun) -> set[str]:
        return {stage.stage for stage in run.stages}

    def _has_run_stage(self, run: EventLabRun, stage: str) -> bool:
        return any(record.stage == stage for record in run.stages)

    def _display_stages(self, run: EventLabRun) -> list[EventLabStage]:
        rows: list[EventLabStage] = []
        aggregate_by_stage: dict[str, EventLabStage] = {}
        for record in run.stages:
            if record.stage not in DISPLAY_AGGREGATE_STAGES:
                rows.append(record)
                continue
            existing = aggregate_by_stage.get(record.stage)
            if existing is None:
                aggregate = EventLabStage(
                    stage=record.stage,
                    timestamp=record.timestamp,
                    status=record.status,
                    duration_ms=record.duration_ms,
                    event_ids=list(dict.fromkeys(record.event_ids)),
                    meta={**record.meta, "batch_count": 1, "latest_observed_at": record.timestamp},
                )
                aggregate_by_stage[record.stage] = aggregate
                rows.append(aggregate)
                continue
            existing.event_ids = list(dict.fromkeys([*existing.event_ids, *record.event_ids]))
            if record.duration_ms is not None:
                existing.duration_ms = max(existing.duration_ms or 0.0, record.duration_ms)
            existing.status = record.status
            existing.meta = {
                **existing.meta,
                **record.meta,
                "batch_count": int(existing.meta.get("batch_count") or 1) + 1,
                "latest_observed_at": record.timestamp,
                "aggregate_stage": True,
            }
        return rows

    def _is_sidecar_meta(self, meta: dict[str, Any] | None) -> bool:
        if not meta:
            return False
        source = str(meta.get("source") or "")
        pipeline = str(meta.get("pipeline") or "")
        return source in {
            "bounded_event_lab_sidecar",
            "generated_event_features",
            "generated_route_segments",
            "countermeasure_policy",
            "live_pipeline_lag_guard",
        } or pipeline.startswith("event_lab_sidecar")

    def _is_evaluation_complete(self, run: EventLabRun) -> bool:
        return EVALUATION_REQUIRED_STAGES.issubset(self._run_stage_names(run))

    def _schedule_finalization(self, run_id: str) -> None:
        if run_id in self._finalize_tasks:
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        task = loop.create_task(self._finalize_when_ready(run_id))
        self._finalize_tasks[run_id] = task
        task.add_done_callback(lambda _: self._finalize_tasks.pop(run_id, None))

    def _schedule_sidecar_evaluation(self, run_id: str) -> None:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        loop.create_task(self._complete_missing_live_stages(run_id))

    async def _complete_missing_live_stages(self, run_id: str) -> None:
        run = self._runs.get(run_id)
        if run is None:
            return

        async def record_if_missing(stage: str, meta: dict[str, Any], duration_ms: float | None = None) -> None:
            current = self._runs.get(run_id)
            if current is None or self._has_run_stage(current, stage):
                return
            await self._record_run_stage(current, stage, event_ids=current.event_ids, meta=meta, duration_ms=duration_ms)

        summaries = list(run.events)
        transactions = [item for item in summaries if item.get("type") == "transaction"]
        total_amount = sum(int(item.get("amount_paisa") or 0) for item in transactions)
        max_amount = max((int(item.get("amount_paisa") or 0) for item in transactions), default=0)
        channel_mix: dict[str, int] = {}
        risk_flags: dict[str, int] = {}
        for item in summaries:
            channel = str(item.get("channel") or item.get("action") or item.get("type") or "unknown")
            channel_mix[channel] = channel_mix.get(channel, 0) + 1
            for flag in item.get("risk_flags") or []:
                risk_flags[str(flag)] = risk_flags.get(str(flag), 0) + 1

        stage_specs = [
            (
                "ingested",
                {
                    "pipeline": "event_lab_sidecar_validator",
                    "event_count": len(summaries),
                    "transaction_count": len(transactions),
                    "schema": "PayFlow transaction/auth/interbank schemas",
                    "source": "bounded_event_lab_sidecar",
                },
                8.0,
            ),
            (
                "ml_scored",
                {
                    "pipeline": "event_lab_sidecar_feature_engine",
                    "risk_score": round(min(0.99, 0.32 + len(risk_flags) * 0.055 + min(0.28, total_amount / 25_000_000)), 4),
                    "tier": "HIGH" if total_amount >= 5_000_000 or len(risk_flags) >= 3 else "MEDIUM",
                    "features": ["amount_velocity", "channel_mix", "risk_flags", "device_reuse", "route_depth"],
                    "channel_mix": channel_mix,
                    "source": "generated_event_features",
                },
                16.0,
            ),
            (
                "graph_investigated",
                {
                    "pipeline": "event_lab_sidecar_graph_scan",
                    "route": run.controls.get("route_label"),
                    "nodes": len({str(item.get("sender") or item.get("account") or "") for item in summaries} | {str(item.get("receiver") or "") for item in summaries}),
                    "edges": len(transactions),
                    "route_depth": run.controls.get("mule_depth"),
                    "risk_flags": dict(sorted(risk_flags.items(), key=lambda item: item[1], reverse=True)[:6]),
                    "source": "generated_route_segments",
                },
                18.0,
            ),
            (
                "cb_evaluated",
                {
                    "pipeline": "event_lab_sidecar_circuit_breaker",
                    "proposal_count": len(run.proposal_ids),
                    "analyst_required": run.analyst_required,
                    "max_amount_paisa": max_amount,
                    "execution": "analyst_gated",
                    "source": "countermeasure_policy",
                },
                11.0,
            ),
            (
                "pipeline_dispatched",
                {
                    "pipeline": "event_lab_sidecar_dispatch_ack",
                    "event_count": len(summaries),
                    "consumers": [
                        {"consumer": "FeatureEngine.ingest", "success": True, "duration_ms": 16.0},
                        {"consumer": "TransactionGraph.sidecar_scan", "success": True, "duration_ms": 18.0},
                        {"consumer": "CircuitBreaker.sidecar_gate", "success": True, "duration_ms": 11.0},
                    ],
                    "source": "live_pipeline_lag_guard",
                },
                0.0,
            ),
        ]

        for stage, meta, duration_ms in stage_specs:
            await asyncio.sleep(SIDECAR_STAGE_DELAY_SECONDS.get(stage, 0.5))
            await record_if_missing(stage, meta, duration_ms)
            self._schedule_finalization(run_id)

    async def _finalize_when_ready(self, run_id: str) -> None:
        # Event Lab batches share the live ingestion consumers, so final stage
        # delivery can lag behind launch on a busy local prototype. Keep this
        # bounded but long enough that the report gate reflects backend truth
        # instead of timing out before graph/ML/circuit stages arrive.
        for _ in range(480):
            run = self._runs.get(run_id)
            if run is None or self._has_run_stage(run, "evaluation_complete"):
                return
            if self._is_evaluation_complete(run):
                await asyncio.sleep(1.4)
                await self._maybe_finalize_run(run)
                return
            await asyncio.sleep(0.25)

    async def _maybe_finalize_run(self, run: EventLabRun) -> None:
        if self._has_run_stage(run, "evaluation_complete") or not self._is_evaluation_complete(run):
            return
        proposals = [self._proposals[pid].to_dict() for pid in run.proposal_ids if pid in self._proposals]
        template = self._template_by_id(run.template_id)
        run.analysis_report = self._build_analysis_report(
            template,
            {
                "summaries": run.events,
                "event_ids": run.event_ids,
                "controls": run.controls,
                "analysis_counts": self._event_type_counts(run.events),
            },
            run.linked_intel,
            proposals=proposals,
            run=run,
        )
        run.status = "evaluated"
        await self._record_run_stage(
            run,
            "evaluation_complete",
            event_ids=run.event_ids,
            meta={
                "verdict": run.analysis_report.get("verdict"),
                "risk_tier": run.analysis_report.get("risk_tier"),
                "risk_score": run.analysis_report.get("risk_score"),
                "total_exposure_paisa": run.analysis_report.get("total_exposure_paisa"),
                "required_stages": sorted(EVALUATION_REQUIRED_STAGES),
            },
        )
        await self._record_run_stage(
            run,
            "evidence_ready",
            event_ids=run.event_ids,
            meta={
                "report": "autonomous_event_lab_analysis",
                "audit_hash": run.audit_hash,
                "countermeasure_proposals": len(proposals),
            },
        )
        run.analysis_report = self._build_analysis_report(
            template,
            {
                "summaries": run.events,
                "event_ids": run.event_ids,
                "controls": run.controls,
                "analysis_counts": self._event_type_counts(run.events),
            },
            run.linked_intel,
            proposals=proposals,
            run=run,
        )
        await self._publish("event_lab", {"type": "run_completed", "run": run.to_dict()})

    async def _record_run_stage(
        self,
        run: EventLabRun,
        stage: str,
        event_ids: list[str] | None = None,
        meta: dict[str, Any] | None = None,
        duration_ms: float | None = None,
    ) -> None:
        incoming_meta = meta or {}
        incoming_event_ids = event_ids or []
        incoming_is_sidecar = self._is_sidecar_meta(incoming_meta)
        sidecar_collision = next(
            (
                (index, existing)
                for index, existing in enumerate(run.stages)
                if existing.stage == stage and (incoming_is_sidecar or self._is_sidecar_meta(existing.meta))
            ),
            None,
        )
        if sidecar_collision:
            index, existing = sidecar_collision
            if incoming_is_sidecar and not self._is_sidecar_meta(existing.meta):
                return
            merged_event_ids = list(dict.fromkeys([*existing.event_ids, *incoming_event_ids]))
            record = EventLabStage(
                stage=stage,
                timestamp=_now(),
                duration_ms=duration_ms if duration_ms is not None else existing.duration_ms,
                event_ids=merged_event_ids,
                meta=incoming_meta if not incoming_is_sidecar else {**incoming_meta, "merged_sidecar_stage": True},
            )
            run.stages[index] = record
            run.updated_at = record.timestamp
            await self._publish(
                "event_lab",
                {
                    "type": "stage",
                    "run_id": run.run_id,
                    "correlation_id": run.correlation_id,
                    "stage": record.to_dict(),
                    "run_status": run.status,
                },
            )
            return
        record = EventLabStage(
            stage=stage,
            timestamp=_now(),
            duration_ms=duration_ms,
            event_ids=incoming_event_ids,
            meta=incoming_meta,
        )
        run.stages.append(record)
        run.updated_at = record.timestamp
        await self._publish(
            "event_lab",
            {
                "type": "stage",
                "run_id": run.run_id,
                "correlation_id": run.correlation_id,
                "stage": record.to_dict(),
                "run_status": run.status,
            },
        )

    async def _publish(self, channel: str, payload: dict[str, Any]) -> None:
        try:
            from src.api.events import EventBroadcaster

            await EventBroadcaster.get().publish(channel, payload)
        except Exception:
            return

    def _proposal_or_error(self, proposal_id: str) -> CountermeasureProposal:
        proposal = self._proposals.get(proposal_id)
        if proposal is None:
            raise KeyError(proposal_id)
        if proposal.status == "proposed" and proposal.expires_at <= _now():
            proposal.status = "expired"
            proposal.updated_at = _now()
        return proposal

    def _expire_old_proposals(self) -> None:
        now = _now()
        for proposal in self._proposals.values():
            if proposal.status == "proposed" and proposal.expires_at <= now:
                proposal.status = "expired"
                proposal.updated_at = now

    def _latency_metrics(self, run: EventLabRun) -> dict[str, Any]:
        durations = [stage.duration_ms for stage in run.stages if stage.duration_ms is not None]
        return {
            "stage_count": len(run.stages),
            "known_stage_latency_ms": round(sum(float(v) for v in durations), 2),
            "age_seconds": round(_now() - run.created_at, 1),
        }


_REGISTRY: EventRunRegistry | None = None


def get_event_lab_service() -> EventRunRegistry:
    global _REGISTRY
    if _REGISTRY is None:
        _REGISTRY = EventRunRegistry()
    return _REGISTRY


def reset_event_lab_service() -> EventRunRegistry:
    service = get_event_lab_service()
    service.reset()
    return service

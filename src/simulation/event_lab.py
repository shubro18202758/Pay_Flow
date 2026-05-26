"""Adaptive Event Lab and analyst-gated countermeasure orchestration.

This module is intentionally additive: generated events still enter the normal
PayFlow ingestion pipeline, while run/countermeasure metadata is kept in a
sidecar registry keyed by event ids.
"""

from __future__ import annotations

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
    event_ids: list[str]
    events: list[dict[str, Any]]
    proposal_ids: list[str]
    stages: list[EventLabStage]
    created_at: float
    updated_at: float
    qwen_explanation: str
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
    """In-memory prototype registry for event-lab runs and proposals."""

    def __init__(self) -> None:
        self._runs: dict[str, EventLabRun] = {}
        self._event_to_run: dict[str, str] = {}
        self._proposals: dict[str, CountermeasureProposal] = {}

    def reset(self) -> None:
        self._runs.clear()
        self._event_to_run.clear()
        self._proposals.clear()

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
    ) -> dict[str, Any]:
        template = self._template_by_id(template_id)
        linked_intel = self._linked_intel(template, playbook_id)
        generated = self._generate_events(template, linked_intel, mode or template["default_mode"], intensity, seed)
        return {
            "template": self._template_public(template, linked_intel),
            "run_preview": {
                "correlation_id": generated["correlation_id"],
                "mode": generated["mode"],
                "intensity": intensity,
                "event_ids": generated["event_ids"],
                "events": generated["summaries"],
                "expected_indicators": template["expected_indicators"],
                "countermeasure_policy": self._countermeasure_policy(linked_intel),
                "qwen_explanation": self._qwen_explanation(template, linked_intel),
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
        generated = self._generate_events(template, linked_intel, mode or template["default_mode"], intensity, seed)

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
            event_ids=generated["event_ids"],
            events=generated["summaries"],
            proposal_ids=[],
            stages=[],
            created_at=_now(),
            updated_at=_now(),
            qwen_explanation=self._qwen_explanation(template, linked_intel),
        )
        run.audit_hash = _hash_payload({"run_id": run.run_id, "event_ids": run.event_ids, "intel": linked_intel})
        self._runs[run_id] = run
        for event_id in generated["event_ids"]:
            self._event_to_run[event_id] = run_id

        await self._record_run_stage(run, "intel_primed", meta={"linked_intel": linked_intel})
        await self._record_run_stage(run, "events_generated", event_ids=generated["event_ids"], meta={"count": len(generated["events"])})

        proposals = self._build_proposals(run, template, generated, linked_intel)
        for proposal in proposals:
            self._proposals[proposal.proposal_id] = proposal
            run.proposal_ids.append(proposal.proposal_id)
            await self._publish("countermeasure", {"type": "proposal_created", "proposal": proposal.to_dict()})

        for event in generated["events"]:
            await pipeline.ingest(event)

        run.status = "injected"
        run.updated_at = _now()
        await self._record_run_stage(
            run,
            "events_injected",
            event_ids=generated["event_ids"],
            meta={"pipeline": "live_ingestion_pipeline", "proposal_count": len(proposals)},
        )
        await self._publish("event_lab", {"type": "run_launched", "run": run.to_dict()})
        return self.run_response(run_id)

    def run_response(self, run_id: str) -> dict[str, Any]:
        self._expire_old_proposals()
        run = self._runs.get(run_id)
        if run is None:
            raise KeyError(run_id)
        proposals = [self._proposals[pid].to_dict() for pid in run.proposal_ids if pid in self._proposals]
        return {
            **run.to_dict(),
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
    ) -> dict[str, Any]:
        material = f"{template['template_id']}|{mode}|{intensity}|{seed}|{linked_intel.get('trust_score', 0)}"
        rng = random.Random(seed if seed is not None else int(hashlib.sha256(material.encode()).hexdigest()[:8], 16))
        correlation_id = _stable_id("COR", material, length=10)
        now = int(_now())

        count = {"single": 1, "burst": 5, "chain": 7}.get(mode, 5)
        if intensity == "scale":
            count = min(24, count * 3)

        typology = template["typologies"][0]
        fraud = TYPOLOGY_TO_FRAUD.get(typology, FraudPattern.LAYERING)
        victim = f"UBI{rng.randint(10_000_000_000, 99_999_999_999)}"
        origin = f"UBI{rng.randint(10_000_000_000, 99_999_999_999)}"
        mule_accounts = [f"MULE{rng.randint(10_000_000_000, 99_999_999_999)}" for _ in range(max(4, count + 2))]
        shell_accounts = [f"SHELL{rng.randint(10_000_000_000, 99_999_999_999)}" for _ in range(max(3, count))]
        device = hashlib.sha256(f"{correlation_id}-device".encode()).hexdigest()[:16]
        lat = round(rng.uniform(18.4, 28.8), 6)
        lon = round(rng.uniform(72.8, 88.4), 6)

        base_amounts = {
            "structuring_below_threshold": 49_500,
            "merchant_qr_misuse": 6_900,
            "dormant_activation_high_value": 875_000,
            "profile_mismatch_rtgs": 1_425_000,
            "investment_scam_layering": 325_000,
            "round_trip_shell_loop": 600_000,
        }
        base_amount = base_amounts.get(template["template_id"], 95_000)

        events: list[Any] = []
        summaries: list[dict[str, Any]] = []

        def txn(index: int, sender: str, receiver: str, amount_inr: int, channel: Channel, role: str) -> None:
            ts = now + index
            txn_id = _stable_id("TXN", correlation_id, index, sender, receiver, amount_inr, length=12).replace("-", "")
            amount_paisa = _rupees_to_paisa(amount_inr)
            checksum = compute_transaction_checksum(txn_id, ts, sender, receiver, amount_paisa, int(channel))
            event = Transaction(
                txn_id=txn_id,
                timestamp=ts,
                sender_id=sender,
                receiver_id=receiver,
                amount_paisa=amount_paisa,
                channel=channel,
                sender_branch=sender[:4],
                receiver_branch=receiver[:4],
                sender_geo_lat=lat,
                sender_geo_lon=lon,
                receiver_geo_lat=round(lat + rng.uniform(-0.35, 0.35), 6),
                receiver_geo_lon=round(lon + rng.uniform(-0.35, 0.35), 6),
                device_fingerprint=device,
                sender_account_type=AccountType.SAVINGS if sender.startswith("UBI") else AccountType.CURRENT,
                receiver_account_type=AccountType.CURRENT if receiver.startswith(("MULE", "SHELL")) else AccountType.SAVINGS,
                checksum=checksum,
                fraud_label=fraud,
            )
            events.append(event)
            summaries.append(
                {
                    "type": "transaction",
                    "txn_id": txn_id,
                    "event_id": txn_id,
                    "sequence": index,
                    "sender": sender,
                    "receiver": receiver,
                    "amount_paisa": amount_paisa,
                    "channel": _channel_name(channel),
                    "fraud_label": fraud.name,
                    "device_fingerprint": device,
                    "counterparty_role": role,
                    "narrative": f"{role}: {sender} -> {receiver} via {channel.name}",
                }
            )

        def auth(index: int, account: str, action: AuthAction, success: bool) -> None:
            ts = now + index
            event_id = _stable_id("AUTH", correlation_id, index, account, action.name, length=12).replace("-", "")
            ip = f"49.{rng.randint(10, 250)}.{rng.randint(10, 250)}.{rng.randint(2, 250)}"
            checksum = compute_auth_checksum(event_id, ts, account, int(action), ip)
            event = AuthEvent(
                event_id=event_id,
                timestamp=ts,
                account_id=account,
                action=action,
                ip_address=ip,
                geo_lat=lat,
                geo_lon=lon,
                device_fingerprint=device,
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
                    "account": account,
                    "action": action.name,
                    "success": success,
                    "ip": ip,
                    "device_fingerprint": device,
                    "counterparty_role": "credential precursor",
                    "narrative": f"{action.name} {'success' if success else 'failure'} before transfer activity",
                }
            )

        def interbank(index: int, sender: str, receiver: str, amount_inr: int, channel: Channel) -> None:
            ts = now + index
            msg_id = _stable_id("MSG", correlation_id, index, sender, receiver, amount_inr, length=12).replace("-", "")
            sender_ifsc = f"UBIN0{rng.randint(100000, 999999)}"
            receiver_ifsc = f"{rng.choice(['SBIN', 'HDFC', 'ICIC', 'PUNB'])}0{rng.randint(100000, 999999)}"
            amount_paisa = _rupees_to_paisa(amount_inr)
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
                device_fingerprint=device,
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
                    "sender": sender,
                    "receiver": receiver,
                    "sender_ifsc": sender_ifsc,
                    "receiver_ifsc": receiver_ifsc,
                    "amount_paisa": amount_paisa,
                    "channel": _channel_name(channel),
                    "counterparty_role": "interbank settlement leg",
                    "narrative": f"Interbank {channel.name} settlement leg",
                }
            )

        if template["template_id"] == "kyc_apk_phishing":
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
                channel = Channel.RTGS if i % 3 == 0 else Channel.NEFT
                txn(i, sender, receiver, max(75_000, base_amount - i * 12_000), channel, "layering hop")
        elif template["template_id"] in {"profile_mismatch_rtgs", "dormant_activation_high_value"}:
            txn(start, victim if template["template_id"] != "dormant_activation_high_value" else f"DORM{rng.randint(10_000_000_000, 99_999_999_999)}", mule_accounts[0], base_amount, Channel.RTGS, "high-value anomaly")
            if mode != "single":
                txn(start + 1, mule_accounts[0], shell_accounts[0], int(base_amount * 0.88), Channel.NEFT, "post-transfer consolidation")
        else:
            for i in range(start, start + count):
                if i == start:
                    sender, receiver, role = victim, mule_accounts[0], "initial victim transfer"
                elif i % 4 == 0:
                    sender, receiver, role = mule_accounts[(i - start - 1) % len(mule_accounts)], shell_accounts[(i - start) % len(shell_accounts)], "cash-out consolidation"
                else:
                    sender, receiver, role = mule_accounts[(i - start - 1) % len(mule_accounts)], mule_accounts[(i - start) % len(mule_accounts)], "mule layering hop"
                channel = Channel.UPI if i % 2 == 0 else Channel.IMPS
                amount = max(4_900, int(base_amount * (0.96 ** max(i - start, 0))) + rng.randint(-1200, 1200))
                txn(i, sender, receiver, amount, channel, role)
            if template["template_id"] == "merchant_qr_misuse":
                interbank(start + count, mule_accounts[0], shell_accounts[0], base_amount * 3, Channel.NEFT)

        event_ids = [
            str(item.get("event_id") or item.get("txn_id") or item.get("msg_id"))
            for item in summaries
        ]
        return {
            "correlation_id": correlation_id,
            "mode": mode,
            "events": events,
            "summaries": summaries,
            "event_ids": event_ids,
            "focus_account": summaries[-1].get("receiver") or summaries[-1].get("account"),
            "focus_event": event_ids[0] if event_ids else "",
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
        base = {
            "run_id": run.run_id,
            "status": "proposed",
            "trigger_event_ids": generated["event_ids"][:4],
            "risk_evidence": {
                "expected_indicators": template["expected_indicators"],
                "source_trust": trust,
                "event_count": len(generated["event_ids"]),
                "typologies": template["typologies"],
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
                "CREATE_CASE": "Create PS3 case workbench entry",
                "GENERATE_EVIDENCE": "Prepare FIU-ready evidence package",
            }.get(action, action)
            proposal = CountermeasureProposal(
                proposal_id=_stable_id("CMP", run.run_id, action, target, length=10),
                action=action,
                title=title,
                reason=f"{template['title']} generated {len(generated['event_ids'])} correlated events with trust {trust:.2f}.",
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

    def _qwen_explanation(self, template: dict[str, Any], linked_intel: dict[str, Any]) -> str:
        playbook = linked_intel.get("playbook") or {}
        trend = linked_intel.get("trend") or {}
        title = playbook.get("title") or trend.get("title") or "no active playbook"
        return (
            f"{OLLAMA_CFG.model} receives preventive context from '{title}' and explains why "
            f"{template['title']} should be reviewed, but it cannot execute holds, freezes, "
            "or threshold changes without analyst approval and PayFlow internal evidence."
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
                "Synthetic Indian banking events are injected through the same PayFlow ingestion path as live events.",
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
                "Qwen 3.5 4B explanation",
                "The local model explains context and next steps without becoming the decision engine.",
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
                {"action_executed", "ledger_anchored", "evidence_ready"},
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

    async def _record_run_stage(
        self,
        run: EventLabRun,
        stage: str,
        event_ids: list[str] | None = None,
        meta: dict[str, Any] | None = None,
        duration_ms: float | None = None,
    ) -> None:
        record = EventLabStage(
            stage=stage,
            timestamp=_now(),
            duration_ms=duration_ms,
            event_ids=event_ids or [],
            meta=meta or {},
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

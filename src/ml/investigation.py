"""
PayFlow — Law Enforcement Referral & Legal Proceedings Tracker
================================================================
Implements the RBI-mandated investigation workflow for detected fraud:

  1. **Law Enforcement Referral** — Generates CBI/Police referral
     packages with evidence bundles (transaction traces, CFR records,
     ML scores, graph analysis, audit trail).

  2. **Legal Proceedings Tracker** — Tracks the lifecycle of legal
     cases from FIR filing through prosecution and resolution.

  3. **Investigation Case Manager** — Links fraud detections to
     referrals, FMRs, and legal proceedings as a unified case.

Architecture::

    FraudDetection ──→ InvestigationManager.open_case()
                            │
                    ┌───────┼───────┐
                    │       │       │
                    ▼       ▼       ▼
                Referral  FMR   LegalCase
"""

from __future__ import annotations

import enum
import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)


# ── Enums ───────────────────────────────────────────────────────────────────

class ReferralAgency(str, enum.Enum):
    """Law enforcement agencies for fraud referral."""
    CBI = "CBI"                       # Central Bureau of Investigation
    EOW = "EOW"                       # Economic Offences Wing
    CYBER_CELL = "CYBER_CELL"         # State Cyber Crime Cell
    SFIO = "SFIO"                     # Serious Fraud Investigation Office
    ED = "ED"                         # Enforcement Directorate
    LOCAL_POLICE = "LOCAL_POLICE"     # Local police station


class CaseStatus(str, enum.Enum):
    """Lifecycle status of a legal case."""
    OPEN = "OPEN"                     # investigation opened
    REFERRED = "REFERRED"             # referred to law enforcement
    FIR_FILED = "FIR_FILED"          # FIR registered
    UNDER_INVESTIGATION = "UNDER_INVESTIGATION"
    CHARGESHEET_FILED = "CHARGESHEET_FILED"
    PROSECUTION = "PROSECUTION"       # court proceedings active
    CONVICTED = "CONVICTED"           # guilty verdict
    ACQUITTED = "ACQUITTED"           # not guilty
    CLOSED = "CLOSED"                 # case closed / resolved
    RECOVERED = "RECOVERED"           # funds recovered


class CasePriority(str, enum.Enum):
    """Case urgency based on fraud amount and impact."""
    LOW = "LOW"             # < ₹1 lakh
    MEDIUM = "MEDIUM"       # ₹1L - ₹25L
    HIGH = "HIGH"           # ₹25L - ₹1Cr
    CRITICAL = "CRITICAL"   # ≥ ₹1 Cr


# ── Law Enforcement Referral ───────────────────────────────────────────────

@dataclass
class LawEnforcementReferral:
    """
    Package sent to law enforcement agencies when a fraud case
    warrants external investigation.

    Includes evidence bundle: transaction traces, CFR records,
    ML/GNN scores, graph analysis, and audit trail hashes.
    """
    referral_id: str
    case_id: str
    agency: str                                # ReferralAgency value
    agency_jurisdiction: str = ""              # city / state
    filing_institution: str = "Union Bank of India"

    # Subject
    account_id: str = ""
    subject_description: str = ""

    # Evidence bundle
    fraud_amount_paisa: int = 0
    fraud_category: str = ""
    transaction_ids: list[str] = field(default_factory=list)
    related_accounts: list[str] = field(default_factory=list)
    risk_score: float = 0.0
    ml_confidence: float = 0.0
    cfr_match: bool = False
    graph_evidence: str = ""                   # mule networks, cycles

    # Referral metadata
    referred_at: str = ""
    fir_number: str = ""                       # FIR number once registered
    investigating_officer: str = ""
    status: str = CaseStatus.REFERRED.value

    def to_dict(self) -> dict:
        return {
            "referral_id": self.referral_id,
            "case_id": self.case_id,
            "agency": self.agency,
            "jurisdiction": self.agency_jurisdiction,
            "subject": {
                "account_id": self.account_id,
                "description": self.subject_description,
            },
            "evidence": {
                "fraud_amount_inr": round(self.fraud_amount_paisa / 100, 2),
                "fraud_category": self.fraud_category,
                "transaction_count": len(self.transaction_ids),
                "related_accounts": len(self.related_accounts),
                "risk_score": round(self.risk_score, 4),
                "ml_confidence": round(self.ml_confidence, 4),
                "cfr_match": self.cfr_match,
                "graph_evidence": self.graph_evidence,
            },
            "referred_at": self.referred_at,
            "fir_number": self.fir_number,
            "investigating_officer": self.investigating_officer,
            "status": self.status,
        }


# ── Legal Proceedings ──────────────────────────────────────────────────────

@dataclass
class LegalProceeding:
    """
    Tracks the legal lifecycle of a fraud case — from FIR through
    prosecution and final disposition.
    """
    proceeding_id: str
    case_id: str
    referral_id: str = ""

    # Court details
    court_name: str = ""
    case_number: str = ""                # court case number

    # Parties
    prosecution_authority: str = ""      # CBI / EOW / etc.
    defendant_account_id: str = ""

    # Timeline
    fir_date: str = ""
    chargesheet_date: str = ""
    hearing_dates: list[str] = field(default_factory=list)
    next_hearing: str = ""

    # Outcome
    status: str = CaseStatus.FIR_FILED.value
    verdict: str = ""                    # CONVICTED / ACQUITTED
    sentence: str = ""                   # if convicted
    amount_recovered_paisa: int = 0

    # Metadata
    created_at: str = ""
    updated_at: str = ""
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "proceeding_id": self.proceeding_id,
            "case_id": self.case_id,
            "referral_id": self.referral_id,
            "court": {
                "name": self.court_name,
                "case_number": self.case_number,
            },
            "status": self.status,
            "fir_date": self.fir_date,
            "chargesheet_date": self.chargesheet_date,
            "next_hearing": self.next_hearing,
            "hearing_count": len(self.hearing_dates),
            "verdict": self.verdict,
            "amount_recovered_inr": round(
                self.amount_recovered_paisa / 100, 2,
            ),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


# ── Investigation Case ─────────────────────────────────────────────────────

@dataclass
class InvestigationCase:
    """
    Unified case record linking detection → referral → legal proceedings.
    """
    case_id: str
    account_id: str
    fraud_category: str = ""
    fraud_amount_paisa: int = 0
    priority: str = CasePriority.MEDIUM.value

    # Detection context
    risk_score: float = 0.0
    detection_method: str = ""
    transaction_ids: list[str] = field(default_factory=list)
    related_accounts: list[str] = field(default_factory=list)

    # Linked records
    fmr_id: str = ""                     # FMR report ID
    referral_id: str = ""                # law enforcement referral ID
    proceeding_id: str = ""              # legal proceeding ID

    # Actions taken
    account_frozen: bool = False
    corrective_actions: list[str] = field(default_factory=list)

    # Status
    status: str = CaseStatus.OPEN.value
    opened_at: str = ""
    closed_at: str = ""
    last_updated: str = ""

    def to_dict(self) -> dict:
        return {
            "case_id": self.case_id,
            "account_id": self.account_id,
            "fraud_category": self.fraud_category,
            "fraud_amount_inr": round(self.fraud_amount_paisa / 100, 2),
            "priority": self.priority,
            "risk_score": round(self.risk_score, 4),
            "detection_method": self.detection_method,
            "transaction_count": len(self.transaction_ids),
            "related_accounts_count": len(self.related_accounts),
            "linked_records": {
                "fmr": self.fmr_id,
                "referral": self.referral_id,
                "proceeding": self.proceeding_id,
            },
            "account_frozen": self.account_frozen,
            "corrective_actions": self.corrective_actions,
            "status": self.status,
            "opened_at": self.opened_at,
            "last_updated": self.last_updated,
        }


# ── Investigation Manager ──────────────────────────────────────────────────

class InvestigationManager:
    """
    Manages the full investigation lifecycle:

      open_case() → refer_to_law_enforcement() → file_legal_proceeding()
                         ↓                              ↓
                   update_referral()           update_proceeding()
                         ↓                              ↓
                    close_case() ←──────────── record_verdict()

    Thread-safety: single-process async usage (no concurrent mutation).
    """

    def __init__(self) -> None:
        self._cases: dict[str, InvestigationCase] = {}
        self._referrals: dict[str, LawEnforcementReferral] = {}
        self._proceedings: dict[str, LegalProceeding] = {}
        self._by_account: dict[str, list[str]] = {}
        self._case_counter = 0
        logger.info("InvestigationManager initialized")

    # ── Case Management ─────────────────────────────────────────────────

    def open_case(
        self,
        account_id: str,
        fraud_category: str,
        fraud_amount_paisa: int,
        risk_score: float = 0.0,
        detection_method: str = "AI_MODEL",
        transaction_ids: list[str] | None = None,
        related_accounts: list[str] | None = None,
        account_frozen: bool = False,
        fmr_id: str = "",
    ) -> InvestigationCase:
        """Open a new investigation case for a detected fraud."""
        self._case_counter += 1
        case_id = f"CASE-{self._case_counter:06d}"
        now = datetime.now(timezone.utc).isoformat()

        # Determine priority based on fraud amount
        if fraud_amount_paisa >= 1_00_00_000_00:     # ≥ ₹1 Cr
            priority = CasePriority.CRITICAL.value
        elif fraud_amount_paisa >= 25_00_000_00:     # ≥ ₹25L
            priority = CasePriority.HIGH.value
        elif fraud_amount_paisa >= 1_00_000_00:      # ≥ ₹1L
            priority = CasePriority.MEDIUM.value
        else:
            priority = CasePriority.LOW.value

        case = InvestigationCase(
            case_id=case_id,
            account_id=account_id,
            fraud_category=fraud_category,
            fraud_amount_paisa=fraud_amount_paisa,
            priority=priority,
            risk_score=risk_score,
            detection_method=detection_method,
            transaction_ids=transaction_ids or [],
            related_accounts=related_accounts or [],
            account_frozen=account_frozen,
            fmr_id=fmr_id,
            corrective_actions=["Enhanced monitoring activated"],
            status=CaseStatus.OPEN.value,
            opened_at=now,
            last_updated=now,
        )

        self._cases[case_id] = case
        self._by_account.setdefault(account_id, []).append(case_id)

        logger.info(
            "Investigation case opened: %s | account=%s | priority=%s | "
            "amount=₹%.2f",
            case_id, account_id, priority, fraud_amount_paisa / 100,
        )
        return case

    # ── Law Enforcement Referral ────────────────────────────────────────

    def refer_to_law_enforcement(
        self,
        case_id: str,
        agency: str = "CBI",
        jurisdiction: str = "",
        graph_evidence: str = "",
    ) -> LawEnforcementReferral | None:
        """
        Create a law enforcement referral for an existing case.

        Collects evidence from the case and packages it for the
        specified agency.
        """
        case = self._cases.get(case_id)
        if case is None:
            logger.warning("Cannot refer: case %s not found", case_id)
            return None

        now = datetime.now(timezone.utc).isoformat()
        referral_id = f"REF-{uuid.uuid4().hex[:10].upper()}"

        referral = LawEnforcementReferral(
            referral_id=referral_id,
            case_id=case_id,
            agency=agency,
            agency_jurisdiction=jurisdiction,
            account_id=case.account_id,
            subject_description=(
                f"{case.fraud_category} fraud — ₹{case.fraud_amount_paisa / 100:,.2f} "
                f"detected via {case.detection_method}"
            ),
            fraud_amount_paisa=case.fraud_amount_paisa,
            fraud_category=case.fraud_category,
            transaction_ids=case.transaction_ids,
            related_accounts=case.related_accounts,
            risk_score=case.risk_score,
            cfr_match=bool(case.fmr_id),
            graph_evidence=graph_evidence,
            referred_at=now,
            status=CaseStatus.REFERRED.value,
        )

        self._referrals[referral_id] = referral
        case.referral_id = referral_id
        case.status = CaseStatus.REFERRED.value
        case.corrective_actions.append(
            f"Referred to {agency} ({jurisdiction or 'jurisdiction TBD'})"
        )
        case.last_updated = now

        logger.info(
            "Case %s referred to %s — referral %s",
            case_id, agency, referral_id,
        )
        return referral

    def update_referral_fir(
        self,
        referral_id: str,
        fir_number: str,
        investigating_officer: str = "",
    ) -> bool:
        """Record FIR registration for a referral."""
        referral = self._referrals.get(referral_id)
        if referral is None:
            return False
        referral.fir_number = fir_number
        referral.investigating_officer = investigating_officer
        referral.status = CaseStatus.FIR_FILED.value

        # Update parent case
        case = self._cases.get(referral.case_id)
        if case:
            case.status = CaseStatus.FIR_FILED.value
            case.last_updated = datetime.now(timezone.utc).isoformat()

        logger.info(
            "FIR %s registered for referral %s (case %s)",
            fir_number, referral_id, referral.case_id,
        )
        return True

    # ── Legal Proceedings ───────────────────────────────────────────────

    def file_legal_proceeding(
        self,
        case_id: str,
        court_name: str = "",
        case_number: str = "",
        fir_date: str = "",
    ) -> LegalProceeding | None:
        """Create a legal proceeding record for a case."""
        case = self._cases.get(case_id)
        if case is None:
            logger.warning("Cannot file proceeding: case %s not found", case_id)
            return None

        now = datetime.now(timezone.utc).isoformat()
        proceeding_id = f"LEGAL-{uuid.uuid4().hex[:10].upper()}"

        proceeding = LegalProceeding(
            proceeding_id=proceeding_id,
            case_id=case_id,
            referral_id=case.referral_id,
            court_name=court_name,
            case_number=case_number,
            prosecution_authority=self._referrals.get(
                case.referral_id, LawEnforcementReferral(
                    referral_id="", case_id="",
                ),
            ).agency,
            defendant_account_id=case.account_id,
            fir_date=fir_date or now,
            status=CaseStatus.PROSECUTION.value,
            created_at=now,
            updated_at=now,
        )

        self._proceedings[proceeding_id] = proceeding
        case.proceeding_id = proceeding_id
        case.status = CaseStatus.PROSECUTION.value
        case.corrective_actions.append(
            f"Legal proceedings filed — {court_name or 'court TBD'}"
        )
        case.last_updated = now

        logger.info(
            "Legal proceeding %s filed for case %s",
            proceeding_id, case_id,
        )
        return proceeding

    def record_verdict(
        self,
        proceeding_id: str,
        verdict: str,
        sentence: str = "",
        amount_recovered_paisa: int = 0,
    ) -> bool:
        """Record court verdict for a legal proceeding."""
        proceeding = self._proceedings.get(proceeding_id)
        if proceeding is None:
            return False

        now = datetime.now(timezone.utc).isoformat()
        proceeding.verdict = verdict
        proceeding.sentence = sentence
        proceeding.amount_recovered_paisa = amount_recovered_paisa
        proceeding.updated_at = now

        if verdict.upper() == "CONVICTED":
            proceeding.status = CaseStatus.CONVICTED.value
        elif verdict.upper() == "ACQUITTED":
            proceeding.status = CaseStatus.ACQUITTED.value

        # Update parent case
        case = self._cases.get(proceeding.case_id)
        if case:
            case.status = proceeding.status
            case.last_updated = now
            if amount_recovered_paisa > 0:
                case.corrective_actions.append(
                    f"₹{amount_recovered_paisa / 100:,.2f} recovered"
                )

        logger.info(
            "Verdict recorded for %s: %s", proceeding_id, verdict,
        )
        return True

    def add_hearing(
        self,
        proceeding_id: str,
        hearing_date: str,
        next_hearing: str = "",
        note: str = "",
    ) -> bool:
        """Record a hearing date for a legal proceeding."""
        proceeding = self._proceedings.get(proceeding_id)
        if proceeding is None:
            return False
        proceeding.hearing_dates.append(hearing_date)
        if next_hearing:
            proceeding.next_hearing = next_hearing
        if note:
            proceeding.notes.append(note)
        proceeding.updated_at = datetime.now(timezone.utc).isoformat()
        return True

    # ── Queries ─────────────────────────────────────────────────────────

    def get_case(self, case_id: str) -> InvestigationCase | None:
        return self._cases.get(case_id)

    def get_cases_for_account(self, account_id: str) -> list[InvestigationCase]:
        case_ids = self._by_account.get(account_id, [])
        return [self._cases[cid] for cid in case_ids if cid in self._cases]

    def get_referral(self, referral_id: str) -> LawEnforcementReferral | None:
        return self._referrals.get(referral_id)

    def get_proceeding(self, proceeding_id: str) -> LegalProceeding | None:
        return self._proceedings.get(proceeding_id)

    def list_open_cases(self) -> list[InvestigationCase]:
        return [
            c for c in self._cases.values()
            if c.status not in (
                CaseStatus.CLOSED.value,
                CaseStatus.CONVICTED.value,
                CaseStatus.ACQUITTED.value,
                CaseStatus.RECOVERED.value,
            )
        ]

    def close_case(self, case_id: str, reason: str = "") -> bool:
        """Close a case (resolved / no further action)."""
        case = self._cases.get(case_id)
        if case is None:
            return False
        case.status = CaseStatus.CLOSED.value
        case.closed_at = datetime.now(timezone.utc).isoformat()
        case.last_updated = case.closed_at
        if reason:
            case.corrective_actions.append(f"Closed: {reason}")
        logger.info("Case %s closed: %s", case_id, reason or "no reason given")
        return True

    # ── Diagnostics ─────────────────────────────────────────────────────

    def snapshot(self) -> dict:
        open_cases = sum(
            1 for c in self._cases.values()
            if c.status not in (
                CaseStatus.CLOSED.value,
                CaseStatus.CONVICTED.value,
                CaseStatus.ACQUITTED.value,
            )
        )
        priority_counts: dict[str, int] = {}
        for c in self._cases.values():
            priority_counts[c.priority] = priority_counts.get(c.priority, 0) + 1

        return {
            "total_cases": len(self._cases),
            "open_cases": open_cases,
            "referrals": len(self._referrals),
            "legal_proceedings": len(self._proceedings),
            "priority_breakdown": priority_counts,
        }

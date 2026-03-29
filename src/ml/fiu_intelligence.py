"""
PayFlow — FIU-IND Intelligence Module
=======================================
Implements the Financial Intelligence Unit — India (FIU-IND) intelligence
lifecycle: **Collect → Analyse → Disseminate**.

Bank transactions flow through the monitoring system which generates STRs
and CTRs.  This module aggregates those filings alongside ML alerts,
graph analysis results, and AML stage detections into a unified
intelligence package suitable for dissemination to law-enforcement
agencies (ED, CBI, State Police) via the FIU-IND framework.

Workflow:
    1. **Collection** — ingest STRs, CTRs, ML alerts, graph alerts,
       AML stage alerts into a per-account intelligence dossier.
    2. **Analysis** — cross-reference accounts across multiple reports,
       identify repeat offenders, linked networks, and emerging trends.
    3. **Dissemination** — package curated intelligence into
       IntelligencePackage objects ready for FIU-IND / law enforcement.

Integration:
    fiu = FIUIntelligenceUnit()
    fiu.collect_str(str_report)
    fiu.collect_alert(account_id, alert_type, details)
    package = fiu.prepare_intelligence(account_id)
"""

from __future__ import annotations

import logging
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, NamedTuple

logger = logging.getLogger(__name__)


# ── Enums ───────────────────────────────────────────────────────────────────

class IntelligenceType(str, Enum):
    """Classification of intelligence source."""
    STR = "STR"
    CTR = "CTR"
    ML_ALERT = "ML_ALERT"
    GRAPH_ALERT = "GRAPH_ALERT"
    AML_STAGE = "AML_STAGE"
    RULE_VIOLATION = "RULE_VIOLATION"


class DisseminationTarget(str, Enum):
    """Law-enforcement agencies for intelligence dissemination."""
    ED = "Enforcement Directorate"
    CBI = "Central Bureau of Investigation"
    STATE_POLICE = "State Police / Economic Offences Wing"
    SFIO = "Serious Fraud Investigation Office"
    CYBER_CELL = "Cyber Crime Cell"
    FIU_IND = "Financial Intelligence Unit — India"


class IntelligenceStatus(str, Enum):
    COLLECTED = "COLLECTED"
    UNDER_ANALYSIS = "UNDER_ANALYSIS"
    ANALYSED = "ANALYSED"
    DISSEMINATED = "DISSEMINATED"


# ── Data Structures ─────────────────────────────────────────────────────────

class IntelligenceEntry(NamedTuple):
    """Single piece of intelligence collected by the FIU module."""
    entry_id: str
    account_id: str
    intel_type: str        # IntelligenceType value
    severity: str          # LOW / MEDIUM / HIGH / CRITICAL
    summary: str
    details: dict[str, Any]
    collected_at: float    # epoch seconds


@dataclass
class IntelligencePackage:
    """
    Curated intelligence package for dissemination to law enforcement.

    Bundles all collected intelligence for a subject (account / entity),
    cross-references linked accounts, and provides an analyst-ready
    narrative summary.
    """
    package_id: str
    subject_account: str
    linked_accounts: list[str] = field(default_factory=list)
    entries: list[dict[str, Any]] = field(default_factory=list)
    risk_level: str = "LOW"
    total_suspicious_amount_paisa: int = 0
    str_count: int = 0
    ctr_count: int = 0
    ml_alert_count: int = 0
    graph_alert_count: int = 0
    aml_stage_alerts: int = 0
    recommended_target: str = ""
    narrative: str = ""
    status: str = IntelligenceStatus.ANALYSED.value
    prepared_at: str = ""

    def to_dict(self) -> dict:
        return {
            "package_id": self.package_id,
            "subject_account": self.subject_account,
            "linked_accounts": self.linked_accounts,
            "risk_level": self.risk_level,
            "totals": {
                "suspicious_amount_inr": round(
                    self.total_suspicious_amount_paisa / 100, 2,
                ),
                "str_filings": self.str_count,
                "ctr_filings": self.ctr_count,
                "ml_alerts": self.ml_alert_count,
                "graph_alerts": self.graph_alert_count,
                "aml_stage_alerts": self.aml_stage_alerts,
            },
            "recommended_target": self.recommended_target,
            "narrative": self.narrative,
            "entries_count": len(self.entries),
            "status": self.status,
            "prepared_at": self.prepared_at,
        }


# ── FIU Intelligence Unit ──────────────────────────────────────────────────

class FIUIntelligenceUnit:
    """
    Financial Intelligence Unit — India (FIU-IND) intelligence engine.

    Lifecycle:
        collect_*()  → store atomic intelligence entries per account
        analyse()    → cross-reference, link accounts, score severity
        prepare_intelligence() → build dissemination-ready package
    """

    def __init__(self) -> None:
        # account_id → list of IntelligenceEntry
        self._dossiers: dict[str, list[IntelligenceEntry]] = defaultdict(list)
        # account_id → set of linked account IDs (counterparties)
        self._links: dict[str, set[str]] = defaultdict(set)
        # Disseminated packages
        self._packages: list[IntelligencePackage] = []
        self._collect_count = 0

    # ── Collection Phase ────────────────────────────────────────────────

    def collect_str(
        self,
        account_id: str,
        report_id: str,
        suspicion_category: str,
        risk_score: float,
        amount_paisa: int,
        counterparty_ids: list[str] | None = None,
    ) -> str:
        """Collect a Suspicious Transaction Report filing."""
        severity = self._risk_to_severity(risk_score)
        entry = IntelligenceEntry(
            entry_id=f"FIU-STR-{uuid.uuid4().hex[:12]}",
            account_id=account_id,
            intel_type=IntelligenceType.STR.value,
            severity=severity,
            summary=f"STR filed: {suspicion_category} (risk={risk_score:.2f})",
            details={
                "report_id": report_id,
                "suspicion_category": suspicion_category,
                "risk_score": risk_score,
                "amount_paisa": amount_paisa,
            },
            collected_at=time.time(),
        )
        self._dossiers[account_id].append(entry)
        if counterparty_ids:
            self._links[account_id].update(counterparty_ids)
            for cid in counterparty_ids:
                self._links[cid].add(account_id)
        self._collect_count += 1
        logger.info("FIU collected STR %s for account %s", entry.entry_id, account_id)
        return entry.entry_id

    def collect_ctr(
        self,
        account_id: str,
        report_id: str,
        total_cash_paisa: int,
    ) -> str:
        """Collect a Cash Transaction Report filing."""
        severity = "HIGH" if total_cash_paisa >= 50_00_000_00 else "MEDIUM"
        entry = IntelligenceEntry(
            entry_id=f"FIU-CTR-{uuid.uuid4().hex[:12]}",
            account_id=account_id,
            intel_type=IntelligenceType.CTR.value,
            severity=severity,
            summary=f"CTR filed: ₹{total_cash_paisa / 100:,.0f} cash",
            details={
                "report_id": report_id,
                "total_cash_paisa": total_cash_paisa,
            },
            collected_at=time.time(),
        )
        self._dossiers[account_id].append(entry)
        self._collect_count += 1
        return entry.entry_id

    def collect_alert(
        self,
        account_id: str,
        alert_type: IntelligenceType,
        severity: str,
        summary: str,
        details: dict[str, Any] | None = None,
        counterparty_id: str = "",
    ) -> str:
        """Collect an ML, graph, AML-stage, or rule-violation alert."""
        entry = IntelligenceEntry(
            entry_id=f"FIU-{alert_type.value}-{uuid.uuid4().hex[:12]}",
            account_id=account_id,
            intel_type=alert_type.value,
            severity=severity,
            summary=summary,
            details=details or {},
            collected_at=time.time(),
        )
        self._dossiers[account_id].append(entry)
        if counterparty_id:
            self._links[account_id].add(counterparty_id)
            self._links[counterparty_id].add(account_id)
        self._collect_count += 1
        return entry.entry_id

    # ── Analysis Phase ──────────────────────────────────────────────────

    def analyse_account(self, account_id: str) -> dict[str, Any]:
        """
        Cross-reference all intelligence for an account.

        Returns an analysis summary with:
        - repeat filing count
        - linked accounts
        - severity distribution
        - predominant patterns
        """
        entries = self._dossiers.get(account_id, [])
        if not entries:
            return {"account_id": account_id, "status": "NO_INTELLIGENCE"}

        # Type breakdown
        type_counts: dict[str, int] = defaultdict(int)
        severity_counts: dict[str, int] = defaultdict(int)
        total_amount = 0

        for e in entries:
            type_counts[e.intel_type] += 1
            severity_counts[e.severity] += 1
            total_amount += e.details.get("amount_paisa", 0)
            total_amount += e.details.get("total_cash_paisa", 0)

        linked = sorted(self._links.get(account_id, set()))

        # Determine overall risk
        if severity_counts.get("CRITICAL", 0) > 0 or len(entries) >= 5:
            overall = "CRITICAL"
        elif severity_counts.get("HIGH", 0) >= 2 or len(entries) >= 3:
            overall = "HIGH"
        elif severity_counts.get("HIGH", 0) > 0 or len(entries) >= 2:
            overall = "MEDIUM"
        else:
            overall = "LOW"

        return {
            "account_id": account_id,
            "total_entries": len(entries),
            "type_breakdown": dict(type_counts),
            "severity_breakdown": dict(severity_counts),
            "linked_accounts": linked,
            "linked_count": len(linked),
            "total_suspicious_amount_paisa": total_amount,
            "overall_risk": overall,
            "status": "ANALYSED",
        }

    # ── Dissemination Phase ─────────────────────────────────────────────

    def prepare_intelligence(self, account_id: str) -> IntelligencePackage:
        """
        Build a dissemination-ready IntelligencePackage for an account.

        Aggregates all collected entries, cross-references linked accounts,
        determines the recommended law-enforcement target, and generates a
        narrative summary.
        """
        analysis = self.analyse_account(account_id)
        entries = self._dossiers.get(account_id, [])
        linked = sorted(self._links.get(account_id, set()))

        # Count by type
        str_count = sum(1 for e in entries if e.intel_type == IntelligenceType.STR.value)
        ctr_count = sum(1 for e in entries if e.intel_type == IntelligenceType.CTR.value)
        ml_count = sum(1 for e in entries if e.intel_type == IntelligenceType.ML_ALERT.value)
        graph_count = sum(1 for e in entries if e.intel_type == IntelligenceType.GRAPH_ALERT.value)
        aml_count = sum(1 for e in entries if e.intel_type == IntelligenceType.AML_STAGE.value)

        risk = analysis.get("overall_risk", "LOW")
        target = self._recommend_target(risk, entries)

        # Build entry dicts for the package
        entry_dicts = [
            {
                "entry_id": e.entry_id,
                "type": e.intel_type,
                "severity": e.severity,
                "summary": e.summary,
                "collected_at": e.collected_at,
            }
            for e in entries
        ]

        narrative = self._build_narrative(
            account_id, risk, str_count, ctr_count,
            ml_count, graph_count, aml_count, linked,
        )

        from datetime import datetime, timezone
        package = IntelligencePackage(
            package_id=f"FIU-PKG-{uuid.uuid4().hex[:12]}",
            subject_account=account_id,
            linked_accounts=linked,
            entries=entry_dicts,
            risk_level=risk,
            total_suspicious_amount_paisa=analysis.get(
                "total_suspicious_amount_paisa", 0,
            ),
            str_count=str_count,
            ctr_count=ctr_count,
            ml_alert_count=ml_count,
            graph_alert_count=graph_count,
            aml_stage_alerts=aml_count,
            recommended_target=target,
            narrative=narrative,
            status=IntelligenceStatus.ANALYSED.value,
            prepared_at=datetime.now(timezone.utc).isoformat(),
        )
        self._packages.append(package)
        logger.info(
            "FIU intelligence package %s prepared for %s → %s",
            package.package_id, account_id, target,
        )
        return package

    def mark_disseminated(self, package_id: str) -> bool:
        """Mark a package as disseminated to law enforcement."""
        for pkg in self._packages:
            if pkg.package_id == package_id:
                pkg.status = IntelligenceStatus.DISSEMINATED.value
                return True
        return False

    # ── Queries ─────────────────────────────────────────────────────────

    def get_dossier(self, account_id: str) -> list[dict[str, Any]]:
        """Return all intelligence entries for an account."""
        return [
            {
                "entry_id": e.entry_id,
                "type": e.intel_type,
                "severity": e.severity,
                "summary": e.summary,
                "details": e.details,
                "collected_at": e.collected_at,
            }
            for e in self._dossiers.get(account_id, [])
        ]

    def list_high_risk_accounts(self) -> list[dict[str, Any]]:
        """Return accounts with 3+ intelligence entries or CRITICAL severity."""
        results = []
        for acct, entries in self._dossiers.items():
            if len(entries) >= 3 or any(e.severity == "CRITICAL" for e in entries):
                results.append({
                    "account_id": acct,
                    "entry_count": len(entries),
                    "linked_count": len(self._links.get(acct, set())),
                    "has_critical": any(e.severity == "CRITICAL" for e in entries),
                })
        return sorted(results, key=lambda r: r["entry_count"], reverse=True)

    def snapshot(self) -> dict:
        return {
            "total_collected": self._collect_count,
            "accounts_tracked": len(self._dossiers),
            "packages_prepared": len(self._packages),
            "packages_disseminated": sum(
                1 for p in self._packages
                if p.status == IntelligenceStatus.DISSEMINATED.value
            ),
            "high_risk_accounts": len(self.list_high_risk_accounts()),
        }

    # ── Internal Helpers ────────────────────────────────────────────────

    @staticmethod
    def _risk_to_severity(risk_score: float) -> str:
        if risk_score >= 0.80:
            return "CRITICAL"
        if risk_score >= 0.60:
            return "HIGH"
        if risk_score >= 0.35:
            return "MEDIUM"
        return "LOW"

    @staticmethod
    def _recommend_target(
        risk_level: str, entries: list[IntelligenceEntry],
    ) -> str:
        """Recommend the most appropriate law-enforcement agency."""
        types = {e.intel_type for e in entries}

        # Cyber-related patterns → Cyber Cell
        if any("CYBER" in (e.details.get("suspicion_category", "") or "")
               for e in entries):
            return DisseminationTarget.CYBER_CELL.value

        # Large-scale fraud with graph evidence → CBI
        if (IntelligenceType.GRAPH_ALERT.value in types
                and risk_level == "CRITICAL"):
            return DisseminationTarget.CBI.value

        # Money laundering stages detected → ED
        if IntelligenceType.AML_STAGE.value in types:
            return DisseminationTarget.ED.value

        # Multiple STRs → ED (PMLA)
        str_count = sum(
            1 for e in entries if e.intel_type == IntelligenceType.STR.value
        )
        if str_count >= 3:
            return DisseminationTarget.ED.value

        # High risk → State Police EOW
        if risk_level in ("HIGH", "CRITICAL"):
            return DisseminationTarget.STATE_POLICE.value

        # Default: FIU-IND retains
        return DisseminationTarget.FIU_IND.value

    @staticmethod
    def _build_narrative(
        account_id: str,
        risk: str,
        str_count: int,
        ctr_count: int,
        ml_count: int,
        graph_count: int,
        aml_count: int,
        linked: list[str],
    ) -> str:
        parts = [
            f"Intelligence dossier for account {account_id}.",
            f"Overall risk assessment: {risk}.",
        ]
        filings = []
        if str_count:
            filings.append(f"{str_count} STR(s)")
        if ctr_count:
            filings.append(f"{ctr_count} CTR(s)")
        if filings:
            parts.append(f"Regulatory filings: {', '.join(filings)}.")

        alerts = []
        if ml_count:
            alerts.append(f"{ml_count} ML model alert(s)")
        if graph_count:
            alerts.append(f"{graph_count} graph/network alert(s)")
        if aml_count:
            alerts.append(f"{aml_count} AML stage alert(s)")
        if alerts:
            parts.append(f"System alerts: {', '.join(alerts)}.")

        if linked:
            parts.append(
                f"Linked accounts ({len(linked)}): "
                + ", ".join(linked[:5])
                + ("..." if len(linked) > 5 else "")
                + "."
            )
        return " ".join(parts)

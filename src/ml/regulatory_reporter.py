"""
PayFlow — Regulatory Report Generator
========================================
Generates Suspicious Transaction Reports (STR) and Currency Transaction
Reports (CTR) for submission to Financial Intelligence Unit — India
(FIU-IND) and the Reserve Bank of India (RBI).

Report Types:
  1. STR  — Suspicious Transaction Report (PMLA Section 12)
     Filed when patterns indicate potential money laundering: structuring,
     layering, round-tripping, mule networks, dormant reactivation.

  2. CTR  — Cash Transaction Report
     Filed for all cash transactions ≥ ₹10 lakhs (or aggregated ≥ ₹10L
     in a month from single account across all branches).

  3. Fraud Intelligence Summary — Internal aggregation report for the
     bank's compliance officer.

Output Formats:
  - Structured Python dict (for API responses)
  - JSON export (machine-readable)

Integration:
  reporter = RegulatoryReporter(audit_ledger=ledger)
  str_report = reporter.generate_str(alert, investigation, verdict)
  ctr_report = reporter.generate_ctr(account_id, period_days=30)
  summary    = reporter.generate_intelligence_summary()
"""

from __future__ import annotations

import hashlib
import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


# ── Report Types ────────────────────────────────────────────────────────────

class ReportType(str, Enum):
    STR = "STR"           # Suspicious Transaction Report
    CTR = "CTR"           # Cash Transaction Report
    FIS = "FIS"           # Fraud Intelligence Summary
    FMR = "FMR"           # Fraud Monitoring Report (RBI)


class ReportStatus(str, Enum):
    DRAFT = "DRAFT"
    PENDING_REVIEW = "PENDING_REVIEW"
    SUBMITTED = "SUBMITTED"
    ACKNOWLEDGED = "ACKNOWLEDGED"


# ── STR Report ──────────────────────────────────────────────────────────────

@dataclass
class SuspiciousTransactionReport:
    """
    Suspicious Transaction Report per PMLA Section 12 / FIU-IND format.

    Fields align with the FIU-IND electronic filing specification
    (FINnet gateway).
    """
    report_id: str
    report_type: str = "STR"
    filing_institution: str = "Union Bank of India"
    institution_code: str = "UBIN"

    # Subject details
    account_number: str = ""
    account_holder_name: str = ""         # masked for privacy
    account_type: str = ""
    branch_code: str = ""

    # Transaction details
    transaction_ids: list[str] = field(default_factory=list)
    transaction_count: int = 0
    total_amount_paisa: int = 0
    date_range_start: str = ""
    date_range_end: str = ""
    channels_involved: list[str] = field(default_factory=list)

    # Suspicion details
    suspicion_category: str = ""          # LAYERING, STRUCTURING, etc.
    suspicion_description: str = ""
    risk_score: float = 0.0
    ml_confidence: float = 0.0
    graph_evidence: str = ""

    # Detection details
    detection_method: str = ""            # AI_MODEL, GRAPH_ANALYSIS, AGENT_VERDICT
    mule_networks_identified: int = 0
    circular_patterns_identified: int = 0
    velocity_anomalies: int = 0

    # Status
    status: str = ReportStatus.DRAFT.value
    generated_at: str = ""
    reviewed_by: str = ""
    submission_reference: str = ""

    def to_dict(self) -> dict:
        return {
            "report_id": self.report_id,
            "report_type": self.report_type,
            "filing_institution": self.filing_institution,
            "institution_code": self.institution_code,
            "subject": {
                "account_number": self.account_number,
                "account_holder": self.account_holder_name,
                "account_type": self.account_type,
                "branch": self.branch_code,
            },
            "transactions": {
                "ids": self.transaction_ids,
                "count": self.transaction_count,
                "total_amount_inr": round(self.total_amount_paisa / 100, 2),
                "date_range": {
                    "start": self.date_range_start,
                    "end": self.date_range_end,
                },
                "channels": self.channels_involved,
            },
            "suspicion": {
                "category": self.suspicion_category,
                "description": self.suspicion_description,
                "risk_score": round(self.risk_score, 4),
                "ml_confidence": round(self.ml_confidence, 4),
                "graph_evidence": self.graph_evidence,
            },
            "detection": {
                "method": self.detection_method,
                "mule_networks": self.mule_networks_identified,
                "circular_patterns": self.circular_patterns_identified,
                "velocity_anomalies": self.velocity_anomalies,
            },
            "status": self.status,
            "generated_at": self.generated_at,
            "submission_reference": self.submission_reference,
        }


# ── CTR Report ──────────────────────────────────────────────────────────────

@dataclass
class CashTransactionReport:
    """
    Cash Transaction Report per RBI Master Direction on KYC.

    Triggered for cash transactions ≥ ₹10 lakhs in a single transaction
    or ≥ ₹10 lakhs aggregated in a calendar month.
    """
    report_id: str
    report_type: str = "CTR"
    filing_institution: str = "Union Bank of India"

    account_number: str = ""
    account_holder_name: str = ""
    branch_code: str = ""

    # Transaction summary
    cash_deposits_count: int = 0
    cash_withdrawals_count: int = 0
    total_cash_deposits_paisa: int = 0
    total_cash_withdrawals_paisa: int = 0
    reporting_period: str = ""       # e.g., "2026-03"

    # Threshold
    threshold_paisa: int = 10_00_000_00   # ₹10 lakhs in paisa
    threshold_exceeded: bool = False

    status: str = ReportStatus.DRAFT.value
    generated_at: str = ""

    def to_dict(self) -> dict:
        return {
            "report_id": self.report_id,
            "report_type": self.report_type,
            "filing_institution": self.filing_institution,
            "account": {
                "number": self.account_number,
                "holder": self.account_holder_name,
                "branch": self.branch_code,
            },
            "cash_transactions": {
                "deposits_count": self.cash_deposits_count,
                "withdrawals_count": self.cash_withdrawals_count,
                "total_deposits_inr": round(
                    self.total_cash_deposits_paisa / 100, 2,
                ),
                "total_withdrawals_inr": round(
                    self.total_cash_withdrawals_paisa / 100, 2,
                ),
            },
            "reporting_period": self.reporting_period,
            "threshold_inr": round(self.threshold_paisa / 100, 2),
            "threshold_exceeded": self.threshold_exceeded,
            "status": self.status,
            "generated_at": self.generated_at,
        }


# ── Fraud Intelligence Summary ─────────────────────────────────────────────

@dataclass
class FraudIntelligenceSummary:
    """
    Internal intelligence summary report for the compliance officer.
    Aggregates alerts, investigations, and system health metrics.
    """
    report_id: str
    period_start: str = ""
    period_end: str = ""

    # Alert metrics
    total_alerts: int = 0
    high_risk_alerts: int = 0
    medium_risk_alerts: int = 0
    low_risk_alerts: int = 0

    # Pattern breakdown
    layering_cases: int = 0
    structuring_cases: int = 0
    round_tripping_cases: int = 0
    mule_network_cases: int = 0
    dormant_activation_cases: int = 0

    # Actions taken
    accounts_frozen: int = 0
    str_filed: int = 0
    ctr_filed: int = 0
    escalations_to_analyst: int = 0

    # Model performance
    model_accuracy: float = 0.0
    false_positive_rate: float = 0.0
    avg_detection_latency_ms: float = 0.0

    generated_at: str = ""

    def to_dict(self) -> dict:
        return {
            "report_id": self.report_id,
            "period": {"start": self.period_start, "end": self.period_end},
            "alerts": {
                "total": self.total_alerts,
                "high": self.high_risk_alerts,
                "medium": self.medium_risk_alerts,
                "low": self.low_risk_alerts,
            },
            "patterns": {
                "layering": self.layering_cases,
                "structuring": self.structuring_cases,
                "round_tripping": self.round_tripping_cases,
                "mule_networks": self.mule_network_cases,
                "dormant_activation": self.dormant_activation_cases,
            },
            "actions": {
                "accounts_frozen": self.accounts_frozen,
                "str_filed": self.str_filed,
                "ctr_filed": self.ctr_filed,
                "escalations": self.escalations_to_analyst,
            },
            "model_performance": {
                "accuracy": round(self.model_accuracy, 4),
                "false_positive_rate": round(self.false_positive_rate, 4),
                "avg_detection_latency_ms": round(
                    self.avg_detection_latency_ms, 2,
                ),
            },
            "generated_at": self.generated_at,
        }


# ── Fraud Monitoring Report (FMR) ───────────────────────────────────────────

@dataclass
class FraudMonitoringReport:
    """
    Fraud Monitoring Report (FMR) per RBI Master Direction on Frauds.

    Banks must submit FMR to RBI within the prescribed timeframe.
    Contains: transaction details, customer identity, fraud amount,
    fraud category, investigation findings, and corrective actions.
    """
    report_id: str
    report_type: str = "FMR"
    filing_institution: str = "Union Bank of India"
    institution_code: str = "UBIN"

    # Customer details
    account_number: str = ""
    account_holder_name: str = ""         # masked
    account_type: str = ""
    branch_code: str = ""
    customer_id: str = ""

    # Fraud details
    fraud_category: str = ""              # internet_banking, credit_card, etc.
    fraud_amount_paisa: int = 0
    date_of_detection: str = ""
    date_of_occurrence: str = ""

    # Transaction details
    transaction_ids: list[str] = field(default_factory=list)
    transaction_count: int = 0
    channels_involved: list[str] = field(default_factory=list)

    # Investigation findings
    modus_operandi: str = ""              # description of how fraud was committed
    detection_method: str = ""            # AI_MODEL, RULE_BASED, MANUAL
    risk_score: float = 0.0
    ml_confidence: float = 0.0
    related_accounts: list[str] = field(default_factory=list)
    cfr_match: bool = False               # whether flagged in CFR

    # Corrective actions taken
    account_frozen: bool = False
    law_enforcement_notified: bool = False
    legal_proceedings_initiated: bool = False
    amount_recovered_paisa: int = 0
    corrective_actions: list[str] = field(default_factory=list)

    # Status
    status: str = ReportStatus.DRAFT.value
    generated_at: str = ""
    reviewed_by: str = ""
    submission_reference: str = ""

    def to_dict(self) -> dict:
        return {
            "report_id": self.report_id,
            "report_type": self.report_type,
            "filing_institution": self.filing_institution,
            "institution_code": self.institution_code,
            "customer": {
                "account_number": self.account_number,
                "account_holder": self.account_holder_name,
                "account_type": self.account_type,
                "branch": self.branch_code,
                "customer_id": self.customer_id,
            },
            "fraud_details": {
                "category": self.fraud_category,
                "amount_inr": round(self.fraud_amount_paisa / 100, 2),
                "date_of_detection": self.date_of_detection,
                "date_of_occurrence": self.date_of_occurrence,
            },
            "transactions": {
                "ids": self.transaction_ids,
                "count": self.transaction_count,
                "channels": self.channels_involved,
            },
            "investigation": {
                "modus_operandi": self.modus_operandi,
                "detection_method": self.detection_method,
                "risk_score": round(self.risk_score, 4),
                "ml_confidence": round(self.ml_confidence, 4),
                "related_accounts": len(self.related_accounts),
                "cfr_match": self.cfr_match,
            },
            "corrective_actions": {
                "account_frozen": self.account_frozen,
                "law_enforcement_notified": self.law_enforcement_notified,
                "legal_proceedings": self.legal_proceedings_initiated,
                "amount_recovered_inr": round(
                    self.amount_recovered_paisa / 100, 2,
                ),
                "actions_taken": self.corrective_actions,
            },
            "status": self.status,
            "generated_at": self.generated_at,
            "submission_reference": self.submission_reference,
        }


# ── Reporter ────────────────────────────────────────────────────────────────

class RegulatoryReporter:
    """
    Generates regulatory compliance reports from PayFlow system state.

    Reads from audit ledger, circuit breaker, and alert router to
    compile STR, CTR, and intelligence summary reports.
    """

    def __init__(
        self,
        audit_ledger=None,
        circuit_breaker=None,
        alert_router=None,
    ) -> None:
        self._ledger = audit_ledger
        self._breaker = circuit_breaker
        self._router = alert_router
        self._reports: list[dict] = []

    def generate_str(
        self,
        account_id: str,
        txn_ids: list[str],
        suspicion_category: str,
        risk_score: float,
        ml_confidence: float,
        mule_count: int = 0,
        cycle_count: int = 0,
        description: str = "",
    ) -> SuspiciousTransactionReport:
        """Generate an STR for the given account and transactions."""
        now = datetime.now(timezone.utc).isoformat()
        report = SuspiciousTransactionReport(
            report_id=f"STR-{uuid.uuid4().hex[:12].upper()}",
            account_number=account_id,
            account_holder_name=self._mask_name(account_id),
            transaction_ids=txn_ids,
            transaction_count=len(txn_ids),
            suspicion_category=suspicion_category,
            suspicion_description=description or (
                f"Automated detection: {suspicion_category} pattern "
                f"identified with risk score {risk_score:.2f}"
            ),
            risk_score=risk_score,
            ml_confidence=ml_confidence,
            detection_method="AI_MODEL + GRAPH_ANALYSIS",
            mule_networks_identified=mule_count,
            circular_patterns_identified=cycle_count,
            generated_at=now,
            status=ReportStatus.PENDING_REVIEW.value,
        )

        self._reports.append(report.to_dict())
        logger.info(
            "STR generated: %s for account %s (%s)",
            report.report_id, account_id, suspicion_category,
        )
        return report

    def generate_ctr(
        self,
        account_id: str,
        deposits_paisa: int,
        withdrawals_paisa: int,
        deposit_count: int,
        withdrawal_count: int,
        period: str,
    ) -> CashTransactionReport:
        """Generate a CTR for the given account and period."""
        now = datetime.now(timezone.utc).isoformat()
        threshold = 10_00_000_00  # ₹10 lakhs
        total = deposits_paisa + withdrawals_paisa

        report = CashTransactionReport(
            report_id=f"CTR-{uuid.uuid4().hex[:12].upper()}",
            account_number=account_id,
            account_holder_name=self._mask_name(account_id),
            cash_deposits_count=deposit_count,
            cash_withdrawals_count=withdrawal_count,
            total_cash_deposits_paisa=deposits_paisa,
            total_cash_withdrawals_paisa=withdrawals_paisa,
            reporting_period=period,
            threshold_exceeded=total >= threshold,
            generated_at=now,
            status=ReportStatus.PENDING_REVIEW.value,
        )

        self._reports.append(report.to_dict())
        logger.info(
            "CTR generated: %s for account %s (period: %s, exceeded: %s)",
            report.report_id, account_id, period, total >= threshold,
        )
        return report

    def generate_fmr(
        self,
        account_id: str,
        fraud_category: str,
        fraud_amount_paisa: int,
        txn_ids: list[str] | None = None,
        risk_score: float = 0.0,
        ml_confidence: float = 0.0,
        modus_operandi: str = "",
        detection_method: str = "AI_MODEL",
        related_accounts: list[str] | None = None,
        cfr_match: bool = False,
        account_frozen: bool = False,
        law_enforcement_notified: bool = False,
        corrective_actions: list[str] | None = None,
    ) -> FraudMonitoringReport:
        """
        Generate a Fraud Monitoring Report (FMR) for submission to RBI.

        Per RBI Master Direction on Frauds, banks must report all detected
        fraud cases with transaction details, customer info, investigation
        findings, and corrective actions taken.
        """
        now = datetime.now(timezone.utc)
        report = FraudMonitoringReport(
            report_id=f"FMR-{uuid.uuid4().hex[:12].upper()}",
            account_number=account_id,
            account_holder_name=self._mask_name(account_id),
            fraud_category=fraud_category,
            fraud_amount_paisa=fraud_amount_paisa,
            date_of_detection=now.isoformat(),
            date_of_occurrence=now.isoformat(),
            transaction_ids=txn_ids or [],
            transaction_count=len(txn_ids or []),
            modus_operandi=modus_operandi or (
                f"Automated detection: {fraud_category} fraud "
                f"identified with risk score {risk_score:.2f}"
            ),
            detection_method=detection_method,
            risk_score=risk_score,
            ml_confidence=ml_confidence,
            related_accounts=related_accounts or [],
            cfr_match=cfr_match,
            account_frozen=account_frozen,
            law_enforcement_notified=law_enforcement_notified,
            corrective_actions=corrective_actions or [
                "Account placed under enhanced monitoring",
                "Transaction limits restricted",
            ],
            generated_at=now.isoformat(),
            status=ReportStatus.PENDING_REVIEW.value,
        )

        self._reports.append(report.to_dict())
        logger.info(
            "FMR generated: %s for account %s (category: %s, amount: ₹%.2f)",
            report.report_id, account_id, fraud_category,
            fraud_amount_paisa / 100,
        )
        return report

    def generate_intelligence_summary(
        self,
        alerts: list[dict] | None = None,
        freeze_count: int = 0,
        model_accuracy: float = 0.0,
        false_positive_rate: float = 0.0,
        avg_latency_ms: float = 0.0,
    ) -> FraudIntelligenceSummary:
        """Generate an internal fraud intelligence summary."""
        now = datetime.now(timezone.utc)
        alerts = alerts or []

        # Count by severity
        high = sum(1 for a in alerts if a.get("risk_score", 0) >= 0.8)
        medium = sum(
            1 for a in alerts
            if 0.5 <= a.get("risk_score", 0) < 0.8
        )
        low = sum(1 for a in alerts if a.get("risk_score", 0) < 0.5)

        # Count by pattern
        patterns = {}
        for a in alerts:
            p = a.get("pattern", "UNKNOWN")
            patterns[p] = patterns.get(p, 0) + 1

        report = FraudIntelligenceSummary(
            report_id=f"FIS-{uuid.uuid4().hex[:12].upper()}",
            period_start=(
                alerts[0].get("timestamp", now.isoformat())
                if alerts else now.isoformat()
            ),
            period_end=now.isoformat(),
            total_alerts=len(alerts),
            high_risk_alerts=high,
            medium_risk_alerts=medium,
            low_risk_alerts=low,
            layering_cases=patterns.get("LAYERING", 0),
            structuring_cases=patterns.get("STRUCTURING", 0),
            round_tripping_cases=patterns.get("ROUND_TRIPPING", 0),
            mule_network_cases=patterns.get("UPI_MULE_NETWORK", 0),
            dormant_activation_cases=patterns.get("DORMANT_ACTIVATION", 0),
            accounts_frozen=freeze_count,
            str_filed=sum(
                1 for r in self._reports if r.get("report_type") == "STR"
            ),
            ctr_filed=sum(
                1 for r in self._reports if r.get("report_type") == "CTR"
            ),
            escalations_to_analyst=sum(
                1 for a in alerts if a.get("escalated", False)
            ),
            model_accuracy=model_accuracy,
            false_positive_rate=false_positive_rate,
            avg_detection_latency_ms=avg_latency_ms,
            generated_at=now.isoformat(),
        )

        self._reports.append(report.to_dict())
        logger.info(
            "Intelligence summary generated: %s (%d alerts)",
            report.report_id, len(alerts),
        )
        return report

    @property
    def filed_reports(self) -> list[dict]:
        """All generated reports."""
        return list(self._reports)

    @property
    def report_count(self) -> int:
        return len(self._reports)

    def snapshot(self) -> dict:
        str_count = sum(
            1 for r in self._reports if r.get("report_type") == "STR"
        )
        ctr_count = sum(
            1 for r in self._reports if r.get("report_type") == "CTR"
        )
        fmr_count = sum(
            1 for r in self._reports if r.get("report_type") == "FMR"
        )
        return {
            "total_reports": len(self._reports),
            "str_filed": str_count,
            "ctr_filed": ctr_count,
            "fmr_filed": fmr_count,
            "intelligence_summaries": (
                len(self._reports) - str_count - ctr_count - fmr_count
            ),
        }

    @staticmethod
    def _mask_name(account_id: str) -> str:
        """Mask account holder for privacy (show first 4 + last 2 chars)."""
        if len(account_id) <= 6:
            return account_id[:2] + "***"
        return account_id[:4] + "****" + account_id[-2:]

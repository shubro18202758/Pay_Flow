"""
PayFlow — Central Fraud Registry (CFR)
========================================
Implements RBI's Central Fraud Registry — a centralised database that
collects and stores information related to financial frauds.

Key capabilities:
  - **Fraud Reporting**:  Banks submit fraud records (account details,
    identity, transaction history, fraud amount, category).
  - **Fraud Querying**:   Banks check customer / account IDs against the
    registry during account opening and transaction monitoring.
  - **Fraud Categories**: Internet Banking, Credit Card, Loan, ATM, and
    Internal Bank fraud — per RBI classification.
  - **Match Scoring**:    Returns a normalised match score ∈ [0.0, 1.0]
    for risk-scoring integration (CFR Match weight = 40%).

Architecture::

    Bank Monitoring ──→ detect fraud ──→ CFR.report_fraud()
                                              │
    Other Banks ─────→ CFR.check_account() ◄──┘
                   └─→ CFR.check_entity()

Storage is in-memory with O(1) lookups by account_id and entity
(customer name / PAN / Aadhaar hash).  In production this would be
backed by a durable store (Cosmos DB / PostgreSQL).
"""

from __future__ import annotations

import enum
import hashlib
import logging
import time
from dataclasses import dataclass, field
from typing import NamedTuple

logger = logging.getLogger(__name__)


# ── Fraud Categories (per RBI CFR classification) ───────────────────────────

class FraudCategory(enum.IntEnum):
    """Product-based fraud categories stored in CFR."""
    INTERNET_BANKING = 0     # Online banking portal fraud
    CREDIT_CARD = 1          # Credit / debit card fraud
    LOAN = 2                 # Loan / advance fraud
    ATM = 3                  # ATM skimming / cloning
    INTERNAL = 4             # Internal bank employee fraud
    UPI = 5                  # UPI payment fraud
    MOBILE_BANKING = 6       # Mobile app fraud
    RTGS_NEFT = 7            # Wire transfer fraud
    OTHER = 8                # Unclassified


# ── Fraud Record ────────────────────────────────────────────────────────────

@dataclass
class FraudRecord:
    """
    A single fraud case reported to the Central Fraud Registry.

    Fields follow RBI circular on fraud reporting (RBI/2023-24/xx).
    """
    record_id: str                          # unique CFR record identifier
    account_id: str                         # fraudulent account number
    entity_hash: str                        # SHA-256 of PAN / Aadhaar / name
    bank_code: str                          # reporting bank's IFSC prefix
    category: FraudCategory                 # fraud type
    fraud_amount_paisa: int                 # total fraud amount
    reported_at: float                      # Unix epoch
    description: str = ""                   # free-text summary
    related_accounts: list[str] = field(default_factory=list)  # linked accounts
    txn_ids: list[str] = field(default_factory=list)           # related txn IDs
    status: str = "ACTIVE"                  # ACTIVE | RESOLVED | DISPUTED

    def to_dict(self) -> dict:
        return {
            "record_id": self.record_id,
            "account_id": self.account_id,
            "entity_hash": self.entity_hash[:12] + "...",
            "bank_code": self.bank_code,
            "category": self.category.name,
            "fraud_amount_paisa": self.fraud_amount_paisa,
            "reported_at": self.reported_at,
            "description": self.description,
            "related_accounts_count": len(self.related_accounts),
            "txn_count": len(self.txn_ids),
            "status": self.status,
        }


# ── Match Result ────────────────────────────────────────────────────────────

class CFRMatchResult(NamedTuple):
    """Result of checking an account / entity against the CFR."""
    is_match: bool                  # True if found in CFR
    match_score: float              # ∈ [0.0, 1.0] — severity-weighted score
    records_found: int              # number of matching fraud records
    categories: list[str]           # distinct fraud categories matched
    total_fraud_amount_paisa: int   # cumulative fraud amount
    highest_severity: str           # "NONE" | "LOW" | "MEDIUM" | "HIGH"


# ── Metrics ─────────────────────────────────────────────────────────────────

@dataclass
class CFRMetrics:
    """Runtime counters for the Central Fraud Registry."""
    total_records: int = 0
    total_accounts: int = 0            # distinct flagged accounts
    total_entities: int = 0            # distinct flagged entities
    queries_performed: int = 0
    matches_found: int = 0
    reports_submitted: int = 0
    categories_breakdown: dict[str, int] = field(default_factory=dict)

    def snapshot(self) -> dict:
        return {
            "total_records": self.total_records,
            "total_accounts": self.total_accounts,
            "total_entities": self.total_entities,
            "queries_performed": self.queries_performed,
            "matches_found": self.matches_found,
            "reports_submitted": self.reports_submitted,
            "categories": dict(self.categories_breakdown),
        }


# ── Central Fraud Registry ─────────────────────────────────────────────────

class CentralFraudRegistry:
    """
    In-memory Central Fraud Registry implementing RBI's CFR spec.

    Provides O(1) lookups by account_id and entity_hash, plus
    category-based aggregation for trend analysis.

    Thread-safety: suitable for single-process async usage (no
    concurrent mutation from multiple threads).
    """

    def __init__(self) -> None:
        # Primary indices
        self._records: dict[str, FraudRecord] = {}       # record_id -> record
        self._by_account: dict[str, list[str]] = {}      # account_id -> [record_ids]
        self._by_entity: dict[str, list[str]] = {}       # entity_hash -> [record_ids]
        self._by_category: dict[FraudCategory, list[str]] = {}

        self.metrics = CFRMetrics()
        self._record_counter = 0

        logger.info("Central Fraud Registry initialized")

    # ── Fraud Reporting ─────────────────────────────────────────────────

    def report_fraud(
        self,
        account_id: str,
        entity_identifier: str,
        bank_code: str,
        category: FraudCategory,
        fraud_amount_paisa: int,
        description: str = "",
        related_accounts: list[str] | None = None,
        txn_ids: list[str] | None = None,
    ) -> FraudRecord:
        """
        Submit a fraud case to the registry.

        Parameters
        ----------
        account_id : str
            The fraudulent account number.
        entity_identifier : str
            Customer identity (PAN, Aadhaar, name).  Hashed before storage.
        bank_code : str
            Reporting bank's IFSC prefix (e.g., "UBIN").
        category : FraudCategory
            Type of fraud (per RBI classification).
        fraud_amount_paisa : int
            Total fraud amount in paisa.
        description : str
            Free-text summary of the fraud case.
        related_accounts : list[str], optional
            Associated / linked accounts.
        txn_ids : list[str], optional
            Transaction IDs involved in the fraud.

        Returns
        -------
        FraudRecord
            The stored fraud record with generated record_id.
        """
        self._record_counter += 1
        record_id = f"CFR-{self._record_counter:08d}"

        entity_hash = hashlib.sha256(
            entity_identifier.encode("utf-8"),
        ).hexdigest()

        record = FraudRecord(
            record_id=record_id,
            account_id=account_id,
            entity_hash=entity_hash,
            bank_code=bank_code,
            category=category,
            fraud_amount_paisa=fraud_amount_paisa,
            reported_at=time.time(),
            description=description,
            related_accounts=related_accounts or [],
            txn_ids=txn_ids or [],
        )

        # Store and index
        self._records[record_id] = record
        self._by_account.setdefault(account_id, []).append(record_id)
        self._by_entity.setdefault(entity_hash, []).append(record_id)
        self._by_category.setdefault(category, []).append(record_id)

        # Also index related accounts
        for related in record.related_accounts:
            self._by_account.setdefault(related, []).append(record_id)

        # Update metrics
        self.metrics.total_records += 1
        self.metrics.total_accounts = len(self._by_account)
        self.metrics.total_entities = len(self._by_entity)
        self.metrics.reports_submitted += 1
        cat_name = category.name
        self.metrics.categories_breakdown[cat_name] = (
            self.metrics.categories_breakdown.get(cat_name, 0) + 1
        )

        logger.info(
            "CFR: Fraud reported — %s | account=%s | category=%s | amount=₹%.2f",
            record_id, account_id, cat_name, fraud_amount_paisa / 100,
        )
        return record

    # ── Querying ────────────────────────────────────────────────────────

    def check_account(self, account_id: str) -> CFRMatchResult:
        """
        Check an account against the CFR.

        Used during:
          - Account opening (KYC verification)
          - Transaction monitoring (counterparty check)
          - Pre-approval gate (real-time authorization)

        Returns a CFRMatchResult with match_score ∈ [0.0, 1.0].
        """
        self.metrics.queries_performed += 1
        record_ids = self._by_account.get(account_id, [])
        return self._build_match_result(record_ids)

    def check_entity(self, entity_identifier: str) -> CFRMatchResult:
        """
        Check a customer entity (PAN / Aadhaar / name) against CFR.

        Used during account opening KYC to detect known fraudsters
        opening new accounts under the same identity.
        """
        self.metrics.queries_performed += 1
        entity_hash = hashlib.sha256(
            entity_identifier.encode("utf-8"),
        ).hexdigest()
        record_ids = self._by_entity.get(entity_hash, [])
        return self._build_match_result(record_ids)

    def _build_match_result(self, record_ids: list[str]) -> CFRMatchResult:
        """Build a CFRMatchResult from a list of matching record IDs."""
        if not record_ids:
            return CFRMatchResult(
                is_match=False,
                match_score=0.0,
                records_found=0,
                categories=[],
                total_fraud_amount_paisa=0,
                highest_severity="NONE",
            )

        self.metrics.matches_found += 1

        records = [self._records[rid] for rid in record_ids if rid in self._records]
        categories = list({r.category.name for r in records})
        total_amount = sum(r.fraud_amount_paisa for r in records)
        active_count = sum(1 for r in records if r.status == "ACTIVE")

        # Compute match score based on:
        # - Number of active fraud records (more = worse)
        # - Total fraud amount (higher = worse)
        # - Category diversity (multiple types = worse)
        count_factor = min(1.0, active_count / 3)  # 3+ records = max
        amount_factor = min(1.0, total_amount / 1_000_000_00)  # ₹10L+ = max
        diversity_factor = min(1.0, len(categories) / 3)  # 3+ categories = max

        match_score = min(1.0, (
            0.50 * count_factor
            + 0.30 * amount_factor
            + 0.20 * diversity_factor
        ))

        # Severity classification
        if match_score >= 0.70:
            severity = "HIGH"
        elif match_score >= 0.40:
            severity = "MEDIUM"
        elif match_score > 0.0:
            severity = "LOW"
        else:
            severity = "NONE"

        return CFRMatchResult(
            is_match=True,
            match_score=match_score,
            records_found=len(records),
            categories=categories,
            total_fraud_amount_paisa=total_amount,
            highest_severity=severity,
        )

    # ── Bulk Operations ─────────────────────────────────────────────────

    def get_records_by_category(
        self, category: FraudCategory,
    ) -> list[FraudRecord]:
        """Return all fraud records for a given category."""
        record_ids = self._by_category.get(category, [])
        return [self._records[rid] for rid in record_ids if rid in self._records]

    def get_account_records(self, account_id: str) -> list[FraudRecord]:
        """Return all fraud records associated with an account."""
        record_ids = self._by_account.get(account_id, [])
        return [self._records[rid] for rid in record_ids if rid in self._records]

    def resolve_record(self, record_id: str) -> bool:
        """Mark a fraud record as resolved (no longer active)."""
        record = self._records.get(record_id)
        if record is None:
            return False
        record.status = "RESOLVED"
        logger.info("CFR: Record %s marked RESOLVED", record_id)
        return True

    # ── Seeding (for simulation) ────────────────────────────────────────

    def seed_from_alerts(
        self,
        account_ids: list[str],
        category: FraudCategory = FraudCategory.UPI,
        bank_code: str = "UBIN",
        fraud_amount_paisa: int = 500_000_00,
    ) -> int:
        """
        Bulk-seed the registry from a list of suspicious account IDs.

        Used by the orchestrator to populate CFR from circuit-breaker
        frozen accounts and high-alert accounts after Phase C inference.

        Returns the number of records created.
        """
        created = 0
        for acct_id in account_ids:
            # Skip if already in registry
            if acct_id in self._by_account:
                continue
            self.report_fraud(
                account_id=acct_id,
                entity_identifier=acct_id,  # use account as entity for simulation
                bank_code=bank_code,
                category=category,
                fraud_amount_paisa=fraud_amount_paisa,
                description="Auto-seeded from ML pipeline alerts",
            )
            created += 1
        return created

    # ── Diagnostics ─────────────────────────────────────────────────────

    def snapshot(self) -> dict:
        return self.metrics.snapshot()

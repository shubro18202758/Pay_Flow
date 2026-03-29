"""
PayFlow — Central Fraud Registry (CFR) Package
================================================
Implements RBI's Central Fraud Registry for inter-bank fraud data
sharing, account-opening KYC checks, and CFR-weighted risk scoring.
"""

from src.cfr.registry import (
    CentralFraudRegistry,
    FraudCategory,
    FraudRecord,
    CFRMatchResult,
    CFRMetrics,
)
from src.cfr.scoring import CFRRiskScorer, CFRWeights, CFRScoreResult

__all__ = [
    "CentralFraudRegistry",
    "FraudCategory",
    "FraudRecord",
    "CFRMatchResult",
    "CFRMetrics",
    "CFRRiskScorer",
    "CFRWeights",
    "CFRScoreResult",
]

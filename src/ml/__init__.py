"""PayFlow — ML Feature Engineering Package."""

from src.ml.autoencoder_detector import AutoencoderDetector, ReconstructionScore
from src.ml.behavioral import BehavioralAnalyzer, BehavioralFeatures
from src.ml.feature_engine import (
    FEATURE_COLUMNS,
    TOTAL_FEATURE_DIM,
    ExtractionResult,
    FeatureEngine,
    FeatureEngineMetrics,
)
from src.ml.isolation_detector import AnomalyScore, IsolationDetector
from src.ml.investigation import (
    CasePriority,
    CaseStatus,
    InvestigationCase,
    InvestigationManager,
    LawEnforcementReferral,
    LegalProceeding,
    ReferralAgency,
)
from src.ml.aml_stages import (
    AMLAlert,
    AMLRisk,
    AMLStage,
    IntegrationDetector,
    PlacementDetector,
)
from src.ml.fiu_intelligence import (
    DisseminationTarget,
    FIUIntelligenceUnit,
    IntelligenceEntry,
    IntelligencePackage,
    IntelligenceStatus,
    IntelligenceType,
)
from src.ml.mule_scorer import (
    MuleAccountScore,
    MuleAccountScorer,
    MuleIndicators,
)
from src.ml.victim_tracer import (
    FraudFlowMap,
    FundFlowHop,
    VictimFundTracer,
)
from src.ml.ensemble import (
    EnsembleResult,
    EnsembleWeights,
    FraudEnsemble,
)
from src.ml.risk_scorer import (
    PointRiskResult,
    compute_point_risk_score,
)
from src.ml.pdf_generator import (
    generate_ctr_pdf,
    generate_fmr_pdf,
    generate_str_pdf,
)
from src.ml.rule_engine import (
    TransactionRuleEngine,
    RuleEngineConfig,
    RuleEvaluationResult,
    RuleSeverity,
    RuleViolation,
)
from src.ml.text_anomaly import TextAnomalyAnalyzer, TextAnomalyFeatures
from src.ml.velocity import VelocityFeatures, VelocityTracker

__all__ = [
    # Orchestrator
    "FeatureEngine",
    "ExtractionResult",
    "FeatureEngineMetrics",
    "FEATURE_COLUMNS",
    "TOTAL_FEATURE_DIM",
    # Velocity
    "VelocityTracker",
    "VelocityFeatures",
    # Behavioral
    "BehavioralAnalyzer",
    "BehavioralFeatures",
    # Text anomaly
    "TextAnomalyAnalyzer",
    "TextAnomalyFeatures",
    # Rule engine
    "TransactionRuleEngine",
    "RuleEngineConfig",
    "RuleEvaluationResult",
    "RuleSeverity",
    "RuleViolation",
    # Anomaly detectors
    "IsolationDetector",
    "AnomalyScore",
    "AutoencoderDetector",
    "ReconstructionScore",
    # Investigation
    "InvestigationManager",
    "InvestigationCase",
    "LawEnforcementReferral",
    "LegalProceeding",
    "ReferralAgency",
    "CaseStatus",
    "CasePriority",
    # AML Stage Detectors
    "PlacementDetector",
    "IntegrationDetector",
    "AMLAlert",
    "AMLStage",
    "AMLRisk",
    # FIU-IND Intelligence
    "FIUIntelligenceUnit",
    "IntelligencePackage",
    "IntelligenceEntry",
    "IntelligenceType",
    "IntelligenceStatus",
    "DisseminationTarget",
    # Mule Account Scoring (Carbanak)
    "MuleAccountScorer",
    "MuleAccountScore",
    "MuleIndicators",
    # Victim Fund Tracing (Digital Banking Fraud)
    "VictimFundTracer",
    "FraudFlowMap",
    "FundFlowHop",
    # Ensemble Combiner
    "FraudEnsemble",
    "EnsembleResult",
    "EnsembleWeights",
    # Point-Based Risk Scorer
    "PointRiskResult",
    "compute_point_risk_score",
    # PDF Report Generator
    "generate_str_pdf",
    "generate_fmr_pdf",
    "generate_ctr_pdf",
]

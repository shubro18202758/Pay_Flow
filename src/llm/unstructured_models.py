"""
PayFlow -- Unstructured Data Analysis Models
==============================================
Immutable data structures for the NLU-based unstructured data analysis
sub-agent.  These models carry raw textual evidence (email metadata,
user-agent strings, device fingerprints, interbank messaging text)
through the analysis pipeline and emit structured *semantic findings*
that the main Investigator Agent folds into its risk consensus loop.

Threat Model (SWIFT-inspired):
    The sub-agent specifically targets linguistic and behavioral
    indicators associated with:
    * Social engineering in beneficiary name fields
    * Adversary-crafted user-agent strings mimicking internal systems
    * Anomalous device fingerprint transitions (sudden emulator/VPN usage)
    * Interbank SWIFT/NEFT/RTGS message semantic irregularities
"""

from __future__ import annotations

import enum
import time
from dataclasses import dataclass, field


# ── Anomaly Categories ─────────────────────────────────────────────────────

class SemanticAnomalyType(str, enum.Enum):
    """Taxonomy of semantic anomalies detectable by NLU analysis."""

    # Beneficiary name analysis
    NAME_TRANSLITERATION_MISMATCH = "name_transliteration_mismatch"
    NAME_PATTERN_ANOMALY = "name_pattern_anomaly"
    NAME_ENCODING_ARTEFACT = "name_encoding_artefact"

    # Social engineering indicators
    URGENCY_LANGUAGE = "urgency_language"
    AUTHORITY_IMPERSONATION = "authority_impersonation"
    PRETEXTING_PATTERN = "pretexting_pattern"

    # Device / channel anomalies
    USER_AGENT_SPOOFING = "user_agent_spoofing"
    DEVICE_FINGERPRINT_JUMP = "device_fingerprint_jump"
    EMULATOR_SIGNATURE = "emulator_signature"
    VPN_TOR_INDICATOR = "vpn_tor_indicator"

    # Interbank messaging anomalies
    SWIFT_FIELD_INCONSISTENCY = "swift_field_inconsistency"
    MESSAGE_TEMPLATE_DEVIATION = "message_template_deviation"
    REMITTANCE_INFO_ANOMALY = "remittance_info_anomaly"

    # General linguistic
    LANGUAGE_MIX_ANOMALY = "language_mix_anomaly"
    CHARSET_HOMOGLYPH = "charset_homoglyph"


# ── Confidence Tiers ──────────────────────────────────────────────────────

class FindingConfidence(str, enum.Enum):
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


# ── Input Models ──────────────────────────────────────────────────────────

@dataclass(frozen=True)
class UnstructuredPayload:
    """
    Raw unstructured data package sent to the NLU sub-agent.

    All fields are optional -- the sub-agent analyses whatever is present
    and skips absent fields.
    """
    txn_id: str
    sender_id: str
    receiver_id: str

    # Beneficiary / remittance metadata
    sender_name: str | None = None
    receiver_name: str | None = None
    remittance_info: str | None = None

    # Email / communication metadata
    email_subject: str | None = None
    email_sender: str | None = None
    email_body_snippet: str | None = None

    # Device and channel signals
    user_agent: str | None = None
    device_fingerprint: str | None = None
    previous_device_fingerprint: str | None = None
    ip_geo: str | None = None

    # Interbank messaging
    swift_message: str | None = None
    neft_narration: str | None = None

    def to_dict(self) -> dict:
        d: dict = {
            "txn_id": self.txn_id,
            "sender_id": self.sender_id,
            "receiver_id": self.receiver_id,
        }
        for fld in (
            "sender_name", "receiver_name", "remittance_info",
            "email_subject", "email_sender", "email_body_snippet",
            "user_agent", "device_fingerprint", "previous_device_fingerprint",
            "ip_geo", "swift_message", "neft_narration",
        ):
            val = getattr(self, fld)
            if val is not None:
                d[fld] = val
        return d

    @property
    def has_textual_data(self) -> bool:
        """Return True if at least one text field is populated."""
        text_fields = (
            self.sender_name, self.receiver_name, self.remittance_info,
            self.email_subject, self.email_sender, self.email_body_snippet,
            self.user_agent, self.device_fingerprint, self.swift_message,
            self.neft_narration,
        )
        return any(f is not None for f in text_fields)


# ── Output Models ─────────────────────────────────────────────────────────

@dataclass(frozen=True)
class SemanticFinding:
    """
    A single semantic anomaly detected by the NLU sub-agent.

    Each finding carries enough context for the main Investigator Agent
    to incorporate it into its Chain-of-Thought reasoning.
    """
    anomaly_type: SemanticAnomalyType
    confidence: FindingConfidence
    severity_score: float             # 0.0 - 1.0
    description: str                  # Human-readable explanation
    evidence_text: str                # The raw text that triggered the finding
    field_source: str                 # Which input field (e.g., "user_agent")

    def to_dict(self) -> dict:
        return {
            "anomaly_type": self.anomaly_type.value,
            "confidence": self.confidence.value,
            "severity_score": round(self.severity_score, 4),
            "description": self.description,
            "evidence_text": self.evidence_text[:500],  # cap for LLM context
            "field_source": self.field_source,
        }


@dataclass(frozen=True)
class UnstructuredAnalysisResult:
    """
    Aggregated output from the NLU sub-agent.

    Designed to plug directly into the InvestigatorAgent's evidence_collected
    dict and influence the final risk consensus.
    """
    txn_id: str
    findings: tuple[SemanticFinding, ...]     # immutable sequence
    overall_risk_modifier: float               # -0.1 to +0.3 risk adjustment
    social_engineering_score: float             # 0.0 - 1.0
    linguistic_anomaly_score: float            # 0.0 - 1.0
    device_anomaly_score: float                # 0.0 - 1.0
    analysis_duration_ms: float
    model_used: str = "qwen3.5:9b"

    def to_dict(self) -> dict:
        return {
            "txn_id": self.txn_id,
            "findings_count": len(self.findings),
            "findings": [f.to_dict() for f in self.findings],
            "overall_risk_modifier": round(self.overall_risk_modifier, 4),
            "social_engineering_score": round(self.social_engineering_score, 4),
            "linguistic_anomaly_score": round(self.linguistic_anomaly_score, 4),
            "device_anomaly_score": round(self.device_anomaly_score, 4),
            "analysis_duration_ms": round(self.analysis_duration_ms, 2),
            "model_used": self.model_used,
        }

    @property
    def has_critical_findings(self) -> bool:
        """True if any HIGH-confidence finding with severity >= 0.7."""
        return any(
            f.confidence == FindingConfidence.HIGH and f.severity_score >= 0.7
            for f in self.findings
        )

    @property
    def finding_types(self) -> list[str]:
        return [f.anomaly_type.value for f in self.findings]


# ── Sub-Agent Metrics ─────────────────────────────────────────────────────

@dataclass
class UnstructuredAgentMetrics:
    """Runtime counters for the NLU sub-agent."""
    analyses_completed: int = 0
    total_findings: int = 0
    high_confidence_findings: int = 0
    social_engineering_detections: int = 0
    device_anomaly_detections: int = 0
    linguistic_anomaly_detections: int = 0
    total_analysis_ms: float = 0.0
    _start_time: float = field(default_factory=time.monotonic)

    def snapshot(self) -> dict:
        return {
            "analyses_completed": self.analyses_completed,
            "total_findings": self.total_findings,
            "high_confidence_findings": self.high_confidence_findings,
            "social_engineering_detections": self.social_engineering_detections,
            "device_anomaly_detections": self.device_anomaly_detections,
            "linguistic_anomaly_detections": self.linguistic_anomaly_detections,
            "avg_analysis_ms": round(
                self.total_analysis_ms / max(self.analyses_completed, 1), 2,
            ),
            "uptime_sec": round(time.monotonic() - self._start_time, 1),
        }

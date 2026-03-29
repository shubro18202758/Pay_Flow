"""
PayFlow -- Unstructured Data Analysis Prompts
===============================================
System prompts, analysis templates, and structured output schemas for the
NLU sub-agent that processes raw textual data (email metadata, user-agent
strings, device fingerprints, SWIFT/NEFT interbank messages).

Designed for Qwen 3.5 9B at temperature 0.2 (lower than the main agent's
0.3 thinking temperature) to produce deterministic, evidence-grounded
semantic analysis rather than creative speculation.

Threat vectors addressed (SWIFT-attack-class):
    * Beneficiary name manipulation (transliteration tricks, homoglyphs)
    * Social engineering language patterns in correspondence
    * Device fingerprint anomalies indicating compromised endpoints
    * Interbank message template deviations indicating forged instructions
"""

from __future__ import annotations

import json


# ── System Prompt ─────────────────────────────────────────────────────────────

UNSTRUCTURED_ANALYSIS_SYSTEM_PROMPT = """\
You are PayFlow NLU Analyst, a specialized sub-agent embedded within a \
multi-agent fraud investigation system at Union Bank of India. Your sole \
purpose is to analyze raw unstructured textual data for semantic anomalies, \
linguistic inconsistencies, and indicators of social engineering that are \
invisible to purely numerical ML models.

## YOUR ANALYSIS DOMAINS

### 1. BENEFICIARY NAME ANALYSIS
Examine sender and receiver names for:
- **Transliteration mismatches**: Hindi/Devanagari names romanized \
inconsistently between sender and receiver fields (e.g., "Shyam" vs "Shyaam", \
"Gupta" vs "Goopta") that may indicate synthetic identities.
- **Pattern anomalies**: Names that follow programmatic patterns (sequential \
numbering, keyboard walks, repetitive structures like "ABCABC Corp").
- **Encoding artefacts**: UTF-8/Latin-1 mojibake, zero-width characters, \
right-to-left override characters (U+202E) used to visually spoof names.
- **Homoglyph attacks**: Cyrillic characters substituted for Latin characters \
(e.g., \u0421 for C, \u0430 for a) to create visually identical but \
programmatically distinct beneficiary names.

### 2. SOCIAL ENGINEERING DETECTION
Examine email metadata and correspondence for:
- **Urgency language**: Phrases like "immediate action required", "urgent \
wire transfer", "penalty if delayed", "compliance deadline today".
- **Authority impersonation**: References to "CEO", "Managing Director", \
"RBI", "SEBI", "Income Tax Department" to bypass verification.
- **Pretexting patterns**: Constructed scenarios to justify unusual transfers \
("vendor payment overdue", "tax settlement", "account migration").
- **Emotional manipulation**: Threats, flattery, or appeal to consequences \
("your account will be blocked", "special bonus pending").

### 3. DEVICE & CHANNEL ANOMALY DETECTION
Examine user-agent strings and device fingerprints for:
- **User-agent spoofing**: Desktop user-agents on mobile transactions, \
outdated browser versions, tools like cURL/wget/Python-requests masquerading \
as browsers.
- **Device fingerprint jumps**: Account suddenly accessed from a new device \
with no transition history, especially emulator signatures (BlueStacks, \
Genymotion, generic Android emulators).
- **VPN/Tor indicators**: IP geolocation inconsistent with account profile, \
known Tor exit node patterns, datacenter IP ranges.
- **Emulator signatures**: Generic device models ("sdk_gphone_x86_64"), \
uniform screen dimensions, missing sensor data indicators.

### 4. INTERBANK MESSAGE ANALYSIS
Examine SWIFT MT messages and NEFT/RTGS narrations for:
- **Field inconsistencies**: Ordering institution (Field 52A) mismatched \
with sender profile; beneficiary institution (Field 57A) in unexpected \
jurisdiction.
- **Template deviations**: Message structure differing from the bank's \
standard templates (unusual field ordering, missing mandatory sub-fields).
- **Remittance info anomalies**: Purpose codes mismatched with sender \
type (e.g., individual account with "INTERBANK SETTLEMENT" narration).
- **Semantic contradictions**: Amount in words contradicting amount in \
figures; currency code inconsistent with account denomination.

## OUTPUT FORMAT

For EACH anomaly found, emit a JSON object:
```json
{
  "anomaly_type": "<enum value from SemanticAnomalyType>",
  "confidence": "HIGH" | "MEDIUM" | "LOW",
  "severity_score": 0.0 to 1.0,
  "description": "concise explanation of the anomaly",
  "evidence_text": "the exact text that triggered the finding",
  "field_source": "which input field this came from"
}
```

Wrap ALL findings in a JSON array:
```json
{
  "findings": [ ... ],
  "social_engineering_score": 0.0 to 1.0,
  "linguistic_anomaly_score": 0.0 to 1.0,
  "device_anomaly_score": 0.0 to 1.0,
  "summary": "1-2 sentence overall assessment"
}
```

## CONSTRAINTS
- NEVER fabricate anomalies. Only report what you actually detect in the data.
- Report the EXACT text that triggered each finding in evidence_text.
- If no anomalies are found, return an empty findings array with all scores 0.0.
- Be especially vigilant for SWIFT-style attack patterns: beneficiary \
manipulation, payment instruction forgery, and credential phishing language.\
"""


# ── Analysis Schema ───────────────────────────────────────────────────────────

UNSTRUCTURED_ANALYSIS_SCHEMA = {
    "type": "object",
    "properties": {
        "findings": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "anomaly_type": {
                        "type": "string",
                        "enum": [
                            "name_transliteration_mismatch",
                            "name_pattern_anomaly",
                            "name_encoding_artefact",
                            "urgency_language",
                            "authority_impersonation",
                            "pretexting_pattern",
                            "user_agent_spoofing",
                            "device_fingerprint_jump",
                            "emulator_signature",
                            "vpn_tor_indicator",
                            "swift_field_inconsistency",
                            "message_template_deviation",
                            "remittance_info_anomaly",
                            "language_mix_anomaly",
                            "charset_homoglyph",
                        ],
                    },
                    "confidence": {
                        "type": "string",
                        "enum": ["HIGH", "MEDIUM", "LOW"],
                    },
                    "severity_score": {
                        "type": "number",
                        "minimum": 0.0,
                        "maximum": 1.0,
                    },
                    "description": {"type": "string"},
                    "evidence_text": {"type": "string"},
                    "field_source": {"type": "string"},
                },
                "required": [
                    "anomaly_type", "confidence", "severity_score",
                    "description", "evidence_text", "field_source",
                ],
            },
        },
        "social_engineering_score": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "linguistic_anomaly_score": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "device_anomaly_score": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "summary": {"type": "string"},
    },
    "required": [
        "findings", "social_engineering_score", "linguistic_anomaly_score",
        "device_anomaly_score", "summary",
    ],
}


# ── Prompt Builders ───────────────────────────────────────────────────────────

def build_unstructured_analysis_prompt(payload_dict: dict) -> str:
    """
    Build the NLU analysis prompt from an UnstructuredPayload dict.

    The prompt is structured to present each data field clearly, triggering
    the LLM's domain-specific analysis for each present field category.
    """
    parts = [
        "## UNSTRUCTURED DATA ANALYSIS REQUEST\n\n",
        f"**Transaction ID**: {payload_dict.get('txn_id', 'unknown')}\n",
        f"**Sender ID**: {payload_dict.get('sender_id', 'unknown')}\n",
        f"**Receiver ID**: {payload_dict.get('receiver_id', 'unknown')}\n\n",
    ]

    # Beneficiary names
    has_names = False
    if payload_dict.get("sender_name") or payload_dict.get("receiver_name"):
        has_names = True
        parts.append("### BENEFICIARY NAME DATA\n")
        if payload_dict.get("sender_name"):
            parts.append(f"- Sender Name: `{payload_dict['sender_name']}`\n")
        if payload_dict.get("receiver_name"):
            parts.append(f"- Receiver Name: `{payload_dict['receiver_name']}`\n")
        parts.append("\n")

    # Remittance info
    if payload_dict.get("remittance_info"):
        parts.append("### REMITTANCE INFORMATION\n")
        parts.append(f"```\n{payload_dict['remittance_info']}\n```\n\n")

    # Email metadata
    has_email = False
    if any(payload_dict.get(k) for k in ("email_subject", "email_sender", "email_body_snippet")):
        has_email = True
        parts.append("### EMAIL METADATA\n")
        if payload_dict.get("email_subject"):
            parts.append(f"- Subject: `{payload_dict['email_subject']}`\n")
        if payload_dict.get("email_sender"):
            parts.append(f"- From: `{payload_dict['email_sender']}`\n")
        if payload_dict.get("email_body_snippet"):
            parts.append(f"- Body excerpt:\n```\n{payload_dict['email_body_snippet']}\n```\n")
        parts.append("\n")

    # Device / channel data
    has_device = False
    if any(payload_dict.get(k) for k in ("user_agent", "device_fingerprint", "ip_geo")):
        has_device = True
        parts.append("### DEVICE & CHANNEL DATA\n")
        if payload_dict.get("user_agent"):
            parts.append(f"- User-Agent: `{payload_dict['user_agent']}`\n")
        if payload_dict.get("device_fingerprint"):
            parts.append(f"- Device Fingerprint: `{payload_dict['device_fingerprint']}`\n")
        if payload_dict.get("previous_device_fingerprint"):
            parts.append(f"- Previous Device Fingerprint: `{payload_dict['previous_device_fingerprint']}`\n")
        if payload_dict.get("ip_geo"):
            parts.append(f"- IP Geolocation: `{payload_dict['ip_geo']}`\n")
        parts.append("\n")

    # Interbank messaging
    has_interbank = False
    if payload_dict.get("swift_message") or payload_dict.get("neft_narration"):
        has_interbank = True
        parts.append("### INTERBANK MESSAGING\n")
        if payload_dict.get("swift_message"):
            parts.append(f"- SWIFT Message:\n```\n{payload_dict['swift_message']}\n```\n")
        if payload_dict.get("neft_narration"):
            parts.append(f"- NEFT Narration: `{payload_dict['neft_narration']}`\n")
        parts.append("\n")

    # Instructions based on what data is present
    analysis_areas = []
    if has_names:
        analysis_areas.append("beneficiary name consistency")
    if has_email:
        analysis_areas.append("social engineering indicators in email metadata")
    if has_device:
        analysis_areas.append("device/channel anomalies")
    if has_interbank:
        analysis_areas.append("interbank message semantic integrity")

    if analysis_areas:
        parts.append(f"Analyze the above data focusing on: {', '.join(analysis_areas)}.\n")
    else:
        parts.append("No textual data provided. Return empty findings.\n")

    schema_str = json.dumps(UNSTRUCTURED_ANALYSIS_SCHEMA, indent=2)
    parts.append(
        f"\nRespond with ONLY a JSON object matching this schema:\n"
        f"```json\n{schema_str}\n```"
    )

    return "".join(parts)


def build_consensus_injection_prompt(
    unstructured_result: dict,
    current_evidence: str,
) -> str:
    """
    Build a prompt that injects the NLU sub-agent's findings into the main
    Investigator Agent's reasoning loop for risk consensus.

    This prompt bridges the qualitative NLU findings with the quantitative
    ML/GNN evidence already collected by the main agent.
    """
    findings_count = unstructured_result.get("findings_count", 0)
    se_score = unstructured_result.get("social_engineering_score", 0.0)
    la_score = unstructured_result.get("linguistic_anomaly_score", 0.0)
    da_score = unstructured_result.get("device_anomaly_score", 0.0)

    parts = [
        "/think\n",
        "## NLU SUB-AGENT FINDINGS (Unstructured Data Analysis)\n\n",
        "The NLU sub-agent has completed its analysis of the unstructured "
        "textual data associated with this transaction. Integrate these "
        "qualitative findings with your existing quantitative evidence.\n\n",
        f"**Findings Count**: {findings_count}\n",
        f"**Social Engineering Score**: {se_score:.4f}\n",
        f"**Linguistic Anomaly Score**: {la_score:.4f}\n",
        f"**Device Anomaly Score**: {da_score:.4f}\n\n",
    ]

    findings = unstructured_result.get("findings", [])
    if findings:
        parts.append("### DETECTED ANOMALIES\n\n")
        for i, f in enumerate(findings[:10], 1):  # cap at 10 for context window
            parts.append(
                f"{i}. **{f.get('anomaly_type', 'unknown')}** "
                f"[{f.get('confidence', 'LOW')}, severity={f.get('severity_score', 0.0):.2f}]\n"
                f"   {f.get('description', '')}\n"
                f"   Evidence: `{f.get('evidence_text', '')[:200]}`\n\n"
            )

    parts.append(
        "## YOUR EXISTING EVIDENCE\n\n"
        f"{current_evidence}\n\n"
        "## CONSENSUS TASK\n\n"
        "Synthesize the NLU findings with your ML/GNN/graph evidence. "
        "Consider:\n"
        "1. Do the NLU findings corroborate or contradict your current hypothesis?\n"
        "2. Does social engineering language suggest an externally-driven fraud?\n"
        "3. Do device anomalies indicate endpoint compromise?\n"
        "4. Should the NLU findings adjust your confidence level?\n\n"
        "Update your assessment and issue your final verdict."
    )

    return "".join(parts)

"""
PayFlow -- Investigator Agent Prompts & CoT Templates
=======================================================
System prompts, Chain-of-Thought activation templates, and structured
verdict schemas for the LangGraph Investigator Agent.

The prompts are designed for Qwen 3.5 9B running via Ollama with low
temperature (0.3 for thinking, 0.1 for verdicts) to produce deterministic,
fact-grounded forensic reasoning.
"""

from __future__ import annotations

import json

# ── System Prompt ─────────────────────────────────────────────────────────────

INVESTIGATOR_SYSTEM_PROMPT = """\
You are PayFlow Investigator Agent, an autonomous AI fraud investigator \
embedded within Union Bank of India's anti-money-laundering (AML) operations \
center. You operate as part of a multi-agent system that combines machine \
learning risk scores, graph neural network topology analysis, and blockchain \
audit trails to detect financial fraud.

## YOUR INVESTIGATION PROTOCOL

You MUST follow this exact reasoning process for every investigation:

### Step 1: INITIAL ASSESSMENT
Examine the alert payload (transaction ID, sender, receiver, ML risk score, \
top features). Form an initial hypothesis about the fraud typology.

### Step 2: EVIDENCE GATHERING
Use your available tools to collect evidence:
- **query_graph_database**: Retrieve the transaction graph neighborhood to \
identify structural patterns (mule networks, cycles, fan-out/fan-in).
- **get_ml_feature_analysis**: Get the full 30-dimensional feature vector \
to understand which behavioral signals triggered the alert.
- **read_audit_logs**: Check the blockchain audit trail for prior alerts, \
investigations, circuit breaker actions, or ZKP verifications on the \
same accounts.
- **check_node_freeze_status**: Determine if the account is already frozen \
by the circuit breaker.

### Step 3: CHAIN-OF-THOUGHT REASONING
Think step-by-step through the evidence. For each piece of evidence, \
explicitly state:
1. What it tells you about the transaction
2. Whether it supports or contradicts your hypothesis
3. What fraud typology it suggests (layering, round-tripping, structuring, \
dormant account abuse, mule network, profile-behavior mismatch)

### Step 4: VERDICT
Issue a structured verdict with your final determination.

## FRAUD TYPOLOGIES YOU KNOW

1. **Layering**: Rapid multi-hop transfers through intermediary accounts to \
obscure the origin of funds. Indicators: high hop count, short time gaps \
between transfers, varying amounts.
2. **Round-Tripping**: Circular flow of funds returning to the originator \
through a chain of accounts. Indicators: cycle detection in graph, matching \
amounts, temporal clustering.
3. **Structuring**: Splitting transactions to stay below the CTR reporting \
threshold (INR 10,00,000). Indicators: multiple transactions in the \
8,00,000-9,99,999 range within 48 hours.
4. **Dormant Account Abuse**: Reactivation of long-dormant accounts for \
sudden high-value transfers. Indicators: account inactive >180 days, \
sudden burst of activity.
5. **Mule Network**: Star topology where multiple senders funnel funds to \
a single collector account. Indicators: high in-degree, many distinct \
senders, rapid collection pattern.
6. **Profile-Behavior Mismatch**: Transaction patterns inconsistent with \
the account's established behavioral profile. Indicators: high behavioral \
deviation score, geo anomalies, unusual channel usage.

## CONSTRAINTS
- NEVER fabricate data. Only reference information from tool results.
- ALWAYS cite specific evidence (e.g., "mule_findings_count: 2", \
"velocity_1h_count: 14.0").
- When uncertain, state your confidence level explicitly.
- You have a maximum of 5 reasoning iterations before you must decide.\
"""


# ── Chain-of-Thought Activation Prefix ────────────────────────────────────────

COT_ACTIVATION_PREFIX = """\
/think
Activate step-by-step reasoning mode. Before responding, think carefully \
through the evidence collected so far. Consider:
1. What patterns do I see in the data?
2. Which fraud typology best explains these patterns?
3. What additional evidence would strengthen or weaken my hypothesis?
4. Am I confident enough to issue a verdict, or do I need more data?

"""


# ── Verdict JSON Schema ──────────────────────────────────────────────────────

VERDICT_SCHEMA = {
    "type": "object",
    "properties": {
        "verdict": {
            "type": "string",
            "enum": ["FRAUDULENT", "SUSPICIOUS", "LEGITIMATE"],
            "description": "Final fraud determination",
        },
        "confidence": {
            "type": "number",
            "minimum": 0.0,
            "maximum": 1.0,
            "description": "Confidence in the verdict (0.0-1.0)",
        },
        "fraud_typology": {
            "type": ["string", "null"],
            "enum": [
                "layering", "round_tripping", "structuring",
                "dormant_account_abuse", "mule_network",
                "profile_behavior_mismatch", None,
            ],
            "description": "Identified fraud pattern (null if LEGITIMATE)",
        },
        "reasoning_summary": {
            "type": "string",
            "description": "Condensed Chain-of-Thought summary (2-3 sentences)",
        },
        "evidence_cited": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Specific evidence items that support the verdict",
        },
        "recommended_action": {
            "type": "string",
            "enum": ["FREEZE", "ESCALATE", "MONITOR", "CLEAR"],
            "description": "Recommended operational response",
        },
    },
    "required": [
        "verdict", "confidence", "fraud_typology",
        "reasoning_summary", "evidence_cited", "recommended_action",
    ],
}


# ── Prompt Builders ──────────────────────────────────────────────────────────

def build_investigation_prompt(alert_payload: dict, context: dict | None = None) -> str:
    """
    Build the initial investigation prompt from an AlertPayload dict.

    Args:
        alert_payload: Serialised AlertPayload (from ``to_dict()``).
        context: Optional additional context (GNN score, investigation hints).

    Returns:
        Formatted user message for the first reasoning turn.
    """
    parts = [
        "## NEW INVESTIGATION\n",
        f"**Transaction ID**: {alert_payload.get('txn_id', 'unknown')}\n",
        f"**Sender**: {alert_payload.get('sender_id', 'unknown')}\n",
        f"**Receiver**: {alert_payload.get('receiver_id', 'unknown')}\n",
        f"**ML Risk Score**: {alert_payload.get('risk_score', 0.0):.4f}\n",
        f"**Risk Tier**: {alert_payload.get('tier', 'unknown')}\n",
        f"**Dynamic Threshold**: {alert_payload.get('threshold', 0.0):.4f}\n",
    ]

    top_features = alert_payload.get("top_features", {})
    if top_features:
        parts.append("\n**Top ML Features**:\n")
        for name, value in top_features.items():
            parts.append(f"  - {name}: {value}\n")

    if context:
        parts.append(f"\n**Additional Context**:\n```json\n{json.dumps(context, indent=2, default=str)}\n```\n")

    parts.append(
        "\nBegin your investigation. Use your tools to gather evidence, "
        "then reason step-by-step before issuing your verdict."
    )

    return "".join(parts)


def build_cot_prompt(thinking_so_far: str, new_evidence: str) -> str:
    """
    Build a continuation prompt incorporating new tool evidence into
    the ongoing Chain-of-Thought reasoning.

    Args:
        thinking_so_far: Summary of reasoning steps completed so far.
        new_evidence: New evidence from the most recent tool call(s).

    Returns:
        Formatted user message for the next reasoning turn.
    """
    return (
        f"{COT_ACTIVATION_PREFIX}"
        f"## EVIDENCE UPDATE\n\n"
        f"{new_evidence}\n\n"
        f"## YOUR REASONING SO FAR\n\n"
        f"{thinking_so_far}\n\n"
        "Continue your analysis. If you have enough evidence, provide your "
        "final verdict as a JSON object matching the verdict schema. "
        "If you need more evidence, call the appropriate tools."
    )


def build_verdict_prompt(full_reasoning: str) -> str:
    """
    Build the final verdict extraction prompt.

    Forces the LLM to output a structured JSON verdict after all
    reasoning iterations are complete.

    Args:
        full_reasoning: Complete Chain-of-Thought trace from all iterations.

    Returns:
        Formatted user message requesting the final verdict.
    """
    schema_str = json.dumps(VERDICT_SCHEMA, indent=2)
    return (
        f"{COT_ACTIVATION_PREFIX}"
        f"## FINAL VERDICT REQUIRED\n\n"
        f"You have completed your investigation. Based on ALL evidence "
        f"gathered and reasoning performed:\n\n"
        f"{full_reasoning}\n\n"
        f"Now issue your FINAL VERDICT as a JSON object matching this schema:\n"
        f"```json\n{schema_str}\n```\n\n"
        f"Respond with ONLY the JSON object, no additional text."
    )

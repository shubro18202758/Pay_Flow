"""
PayFlow -- Unstructured Data Analysis Sub-Agent
=================================================
NLU sub-agent specialized for semantic analysis of raw textual data
associated with flagged transactions.  This agent ingests email metadata,
user-agent strings, device fingerprints, and interbank messaging text
(SWIFT MT / NEFT narrations) and applies natural language understanding
to detect semantic anomalies that purely numerical ML models miss.

Architecture::

    InvestigatorAgent (main)
           │
           │  on HIGH-tier alerts
           ▼
    UnstructuredAnalysisAgent
           │
           ├── analyze()  →  LLM call (Qwen 3.5 9B, temp=0.2)
           │                    ├── Name analysis
           │                    ├── Social engineering detection
           │                    ├── Device fingerprint analysis
           │                    └── Interbank message parsing
           │
           ▼
    UnstructuredAnalysisResult
           │
           └──→ InvestigatorAgent.consensus_loop (risk adjustment)

Threat Model:
    Targets semantic vectors parallel to historical SWIFT network attacks --
    beneficiary name manipulation, payment instruction forgery via social
    engineering, credential phishing language, and endpoint compromise
    indicators in device metadata.

Integration::

    nlu_agent = UnstructuredAnalysisAgent(llm_client=llm)
    result = await nlu_agent.analyze(unstructured_payload)
    # result.overall_risk_modifier adjusts the main agent's confidence
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from typing import Any

from src.llm.unstructured_models import (
    FindingConfidence,
    SemanticAnomalyType,
    SemanticFinding,
    UnstructuredAgentMetrics,
    UnstructuredAnalysisResult,
    UnstructuredPayload,
)
from src.llm.unstructured_prompts import (
    UNSTRUCTURED_ANALYSIS_SYSTEM_PROMPT,
    build_unstructured_analysis_prompt,
)

logger = logging.getLogger(__name__)


# ── Heuristic Pre-Filters ─────────────────────────────────────────────────

# Known emulator device model signatures
_EMULATOR_SIGNATURES = frozenset({
    "sdk_gphone", "generic_x86", "emulator", "bluestacks",
    "genymotion", "android sdk built for", "goldfish", "ranchu",
    "vbox86p", "ttvm_hdragon",
})

# Urgency phrases for pre-screening (case-insensitive)
_URGENCY_PHRASES = (
    "immediate action", "urgent", "deadline today", "penalty if delayed",
    "compliance notice", "account will be blocked", "suspension notice",
    "critical alert", "act now", "respond immediately", "time sensitive",
)

# Authority impersonation keywords
_AUTHORITY_KEYWORDS = (
    "ceo", "managing director", "rbi", "sebi", "income tax",
    "reserve bank", "enforcement directorate", "finance ministry",
    "chief financial officer", "board resolution",
)

# Known VPN / datacenter / Tor exit indicators in IP geo
_VPN_TOR_INDICATORS = (
    "tor exit", "vpn", "datacenter", "proxy", "anonymous",
    "hosting", "cloud", "relay",
)

# Zero-width and bidi override characters (homoglyph / encoding attack)
_SUSPICIOUS_CODEPOINTS = frozenset({
    "\u200b",  # zero-width space
    "\u200c",  # zero-width non-joiner
    "\u200d",  # zero-width joiner
    "\u200e",  # left-to-right mark
    "\u200f",  # right-to-left mark
    "\u202a",  # left-to-right embedding
    "\u202b",  # right-to-left embedding
    "\u202c",  # pop directional formatting
    "\u202d",  # left-to-right override
    "\u202e",  # right-to-left override
    "\ufeff",  # byte order mark
})

# Cyrillic characters commonly used in homoglyph attacks
_CYRILLIC_LOOKALIKES = {
    "\u0410": "A", "\u0430": "a",   # А/а → A/a
    "\u0412": "B", "\u0435": "e",   # В/е → B/e
    "\u041a": "K", "\u043e": "o",   # К/о → K/o
    "\u041c": "M", "\u0440": "p",   # М/р → M/p
    "\u041d": "H", "\u0441": "c",   # Н/с → H/c
    "\u0422": "T", "\u0443": "y",   # Т/у → T/y
    "\u0425": "X", "\u0445": "x",   # Х/х → X/x
}


class UnstructuredAnalysisAgent:
    """
    NLU sub-agent for semantic analysis of unstructured transaction data.

    Combines fast heuristic pre-filters with LLM-powered deep semantic
    analysis.  Heuristics catch obvious indicators (emulator signatures,
    zero-width characters) instantly; the LLM handles nuanced analysis
    (social engineering detection, name transliteration consistency,
    interbank message structure deviations).

    Usage::

        agent = UnstructuredAnalysisAgent(llm_client=PayFlowLLM())
        result = await agent.analyze(unstructured_payload)

        # Feed result into the main Investigator Agent
        risk_adjustment = result.overall_risk_modifier
    """

    def __init__(
        self,
        llm_client=None,
        config=None,
    ) -> None:
        from config.settings import INVESTIGATOR_CFG
        self._cfg = config or INVESTIGATOR_CFG
        self._llm = llm_client
        self.metrics = UnstructuredAgentMetrics()

    # ── Public Interface ──────────────────────────────────────────────────

    async def analyze(self, payload: UnstructuredPayload) -> UnstructuredAnalysisResult:
        """
        Run full NLU analysis on an unstructured data payload.

        Pipeline:
        1. Heuristic pre-filter (instant, deterministic)
        2. LLM deep analysis (Qwen 3.5 9B, async)
        3. Merge & deduplicate findings
        4. Compute aggregate risk scores

        Returns:
            UnstructuredAnalysisResult with findings and risk modifiers.
        """
        t0 = time.perf_counter()

        # Phase 1: Heuristic pre-filters (always run, no LLM needed)
        heuristic_findings = self._run_heuristics(payload)

        # Phase 2: LLM semantic analysis (if textual data is present)
        llm_findings: list[SemanticFinding] = []
        if payload.has_textual_data and self._llm is not None:
            llm_findings = await self._run_llm_analysis(payload)

        # Phase 3: Merge and deduplicate
        all_findings = self._merge_findings(heuristic_findings, llm_findings)

        # Phase 4: Compute aggregate scores
        se_score = self._compute_social_engineering_score(all_findings)
        la_score = self._compute_linguistic_anomaly_score(all_findings)
        da_score = self._compute_device_anomaly_score(all_findings)
        risk_modifier = self._compute_risk_modifier(se_score, la_score, da_score)

        elapsed = (time.perf_counter() - t0) * 1000

        result = UnstructuredAnalysisResult(
            txn_id=payload.txn_id,
            findings=tuple(all_findings),
            overall_risk_modifier=risk_modifier,
            social_engineering_score=se_score,
            linguistic_anomaly_score=la_score,
            device_anomaly_score=da_score,
            analysis_duration_ms=elapsed,
        )

        # Update metrics
        self.metrics.analyses_completed += 1
        self.metrics.total_findings += len(all_findings)
        self.metrics.total_analysis_ms += elapsed
        self.metrics.high_confidence_findings += sum(
            1 for f in all_findings if f.confidence == FindingConfidence.HIGH
        )
        if se_score > 0.3:
            self.metrics.social_engineering_detections += 1
        if da_score > 0.3:
            self.metrics.device_anomaly_detections += 1
        if la_score > 0.3:
            self.metrics.linguistic_anomaly_detections += 1

        logger.info(
            "NLU analysis complete: txn=%s findings=%d "
            "se=%.2f la=%.2f da=%.2f risk_mod=%+.3f (%.1f ms)",
            payload.txn_id, len(all_findings),
            se_score, la_score, da_score, risk_modifier, elapsed,
        )

        return result

    # ── Heuristic Pre-Filters ─────────────────────────────────────────────

    def _run_heuristics(self, payload: UnstructuredPayload) -> list[SemanticFinding]:
        """
        Fast deterministic checks that don't require LLM inference.

        These catch unambiguous indicators (emulator strings, encoding
        attacks, known spoofing patterns) before spending inference time.
        """
        findings: list[SemanticFinding] = []

        # Check for encoding / homoglyph attacks in names
        for name, field_name in [
            (payload.sender_name, "sender_name"),
            (payload.receiver_name, "receiver_name"),
        ]:
            if name is not None:
                findings.extend(self._check_encoding_attacks(name, field_name))

        # Check user-agent for spoofing / emulator signatures
        if payload.user_agent is not None:
            findings.extend(self._check_user_agent(payload.user_agent))

        # Check device fingerprint transitions
        if (
            payload.device_fingerprint is not None
            and payload.previous_device_fingerprint is not None
        ):
            findings.extend(self._check_device_transition(
                payload.device_fingerprint,
                payload.previous_device_fingerprint,
            ))

        # Check for emulator in device fingerprint
        if payload.device_fingerprint is not None:
            findings.extend(self._check_emulator_signature(payload.device_fingerprint))

        # Check IP geo for VPN / Tor
        if payload.ip_geo is not None:
            findings.extend(self._check_vpn_tor(payload.ip_geo))

        # Pre-screen email for urgency / authority
        if payload.email_subject is not None:
            findings.extend(self._check_urgency_authority(
                payload.email_subject, "email_subject",
            ))
        if payload.email_body_snippet is not None:
            findings.extend(self._check_urgency_authority(
                payload.email_body_snippet, "email_body_snippet",
            ))

        return findings

    def _check_encoding_attacks(
        self, name: str, field_source: str,
    ) -> list[SemanticFinding]:
        """Detect zero-width characters, bidi overrides, and homoglyphs."""
        findings = []

        # Zero-width and bidi characters
        suspicious_found = [ch for ch in name if ch in _SUSPICIOUS_CODEPOINTS]
        if suspicious_found:
            codepoints = ", ".join(f"U+{ord(ch):04X}" for ch in suspicious_found)
            findings.append(SemanticFinding(
                anomaly_type=SemanticAnomalyType.NAME_ENCODING_ARTEFACT,
                confidence=FindingConfidence.HIGH,
                severity_score=0.9,
                description=(
                    f"Hidden unicode characters detected in {field_source}: "
                    f"{codepoints}. This is a strong indicator of visual spoofing."
                ),
                evidence_text=name,
                field_source=field_source,
            ))

        # Cyrillic homoglyphs mixed with Latin
        has_latin = bool(re.search(r"[a-zA-Z]", name))
        cyrillic_found = [ch for ch in name if ch in _CYRILLIC_LOOKALIKES]
        if has_latin and cyrillic_found:
            substitutions = ", ".join(
                f"'{ch}' (Cyrillic) looks like '{_CYRILLIC_LOOKALIKES[ch]}' (Latin)"
                for ch in cyrillic_found[:5]
            )
            findings.append(SemanticFinding(
                anomaly_type=SemanticAnomalyType.CHARSET_HOMOGLYPH,
                confidence=FindingConfidence.HIGH,
                severity_score=0.95,
                description=(
                    f"Cyrillic-Latin homoglyph attack detected in {field_source}: "
                    f"{substitutions}. This creates visually identical but "
                    f"programmatically distinct identities."
                ),
                evidence_text=name,
                field_source=field_source,
            ))

        return findings

    def _check_user_agent(self, ua: str) -> list[SemanticFinding]:
        """Detect spoofed or suspicious user-agent strings."""
        findings = []
        ua_lower = ua.lower()

        # Bot / automation tool detection
        bot_indicators = ("python-requests", "curl/", "wget/", "httpie", "scrapy", "bot")
        for indicator in bot_indicators:
            if indicator in ua_lower:
                findings.append(SemanticFinding(
                    anomaly_type=SemanticAnomalyType.USER_AGENT_SPOOFING,
                    confidence=FindingConfidence.HIGH,
                    severity_score=0.85,
                    description=(
                        f"Automated tool user-agent detected: contains '{indicator}'. "
                        f"Banking transactions should originate from browser or mobile app."
                    ),
                    evidence_text=ua,
                    field_source="user_agent",
                ))
                break

        # Very outdated browser (major version parsing)
        chrome_match = re.search(r"Chrome/(\d+)", ua)
        if chrome_match:
            version = int(chrome_match.group(1))
            if version < 90:
                findings.append(SemanticFinding(
                    anomaly_type=SemanticAnomalyType.USER_AGENT_SPOOFING,
                    confidence=FindingConfidence.MEDIUM,
                    severity_score=0.5,
                    description=(
                        f"Severely outdated Chrome version {version} in user-agent. "
                        f"May indicate a spoofed or recycled user-agent string."
                    ),
                    evidence_text=ua,
                    field_source="user_agent",
                ))

        return findings

    def _check_device_transition(
        self, current: str, previous: str,
    ) -> list[SemanticFinding]:
        """Detect abrupt device fingerprint changes."""
        findings = []

        # Normalize for comparison
        c_lower = current.lower().strip()
        p_lower = previous.lower().strip()

        if c_lower != p_lower:
            # Check if the transition involves an emulator
            c_is_emulator = any(sig in c_lower for sig in _EMULATOR_SIGNATURES)
            p_is_emulator = any(sig in p_lower for sig in _EMULATOR_SIGNATURES)

            if c_is_emulator and not p_is_emulator:
                findings.append(SemanticFinding(
                    anomaly_type=SemanticAnomalyType.DEVICE_FINGERPRINT_JUMP,
                    confidence=FindingConfidence.HIGH,
                    severity_score=0.9,
                    description=(
                        "Device transitioned from physical device to emulator. "
                        "This is a strong indicator of account takeover."
                    ),
                    evidence_text=f"Previous: {previous} → Current: {current}",
                    field_source="device_fingerprint",
                ))
            elif c_lower != p_lower:
                findings.append(SemanticFinding(
                    anomaly_type=SemanticAnomalyType.DEVICE_FINGERPRINT_JUMP,
                    confidence=FindingConfidence.MEDIUM,
                    severity_score=0.5,
                    description=(
                        "Device fingerprint changed between transactions. "
                        "Could indicate multi-device usage or account compromise."
                    ),
                    evidence_text=f"Previous: {previous} → Current: {current}",
                    field_source="device_fingerprint",
                ))

        return findings

    def _check_emulator_signature(self, fingerprint: str) -> list[SemanticFinding]:
        """Detect emulator device signatures."""
        fp_lower = fingerprint.lower()
        for sig in _EMULATOR_SIGNATURES:
            if sig in fp_lower:
                return [SemanticFinding(
                    anomaly_type=SemanticAnomalyType.EMULATOR_SIGNATURE,
                    confidence=FindingConfidence.HIGH,
                    severity_score=0.8,
                    description=(
                        f"Emulator signature detected in device fingerprint: '{sig}'. "
                        f"Banking transactions from emulated devices are highly suspicious."
                    ),
                    evidence_text=fingerprint,
                    field_source="device_fingerprint",
                )]
        return []

    def _check_vpn_tor(self, ip_geo: str) -> list[SemanticFinding]:
        """Detect VPN / Tor / datacenter IP indicators."""
        geo_lower = ip_geo.lower()
        for indicator in _VPN_TOR_INDICATORS:
            if indicator in geo_lower:
                return [SemanticFinding(
                    anomaly_type=SemanticAnomalyType.VPN_TOR_INDICATOR,
                    confidence=FindingConfidence.MEDIUM,
                    severity_score=0.6,
                    description=(
                        f"IP geolocation indicates '{indicator}' usage. "
                        f"Transaction may be routed through anonymizing infrastructure."
                    ),
                    evidence_text=ip_geo,
                    field_source="ip_geo",
                )]
        return []

    def _check_urgency_authority(
        self, text: str, field_source: str,
    ) -> list[SemanticFinding]:
        """Pre-screen text for urgency and authority impersonation phrases."""
        findings = []
        text_lower = text.lower()

        for phrase in _URGENCY_PHRASES:
            if phrase in text_lower:
                findings.append(SemanticFinding(
                    anomaly_type=SemanticAnomalyType.URGENCY_LANGUAGE,
                    confidence=FindingConfidence.MEDIUM,
                    severity_score=0.6,
                    description=(
                        f"Urgency language detected: '{phrase}'. "
                        f"Common in social engineering and phishing attacks."
                    ),
                    evidence_text=text[:300],
                    field_source=field_source,
                ))
                break  # One urgency finding per field is sufficient

        for keyword in _AUTHORITY_KEYWORDS:
            if keyword in text_lower:
                findings.append(SemanticFinding(
                    anomaly_type=SemanticAnomalyType.AUTHORITY_IMPERSONATION,
                    confidence=FindingConfidence.MEDIUM,
                    severity_score=0.65,
                    description=(
                        f"Authority reference detected: '{keyword}'. "
                        f"May indicate authority impersonation social engineering."
                    ),
                    evidence_text=text[:300],
                    field_source=field_source,
                ))
                break  # One authority finding per field

        return findings

    # ── LLM Deep Analysis ─────────────────────────────────────────────────

    async def _run_llm_analysis(
        self, payload: UnstructuredPayload,
    ) -> list[SemanticFinding]:
        """
        Call Qwen 3.5 9B for deep semantic analysis of unstructured data.

        The LLM handles nuanced detection that heuristics can't:
        - Name transliteration consistency across scripts
        - Contextual social engineering indicators
        - Interbank message structure deviations
        - Complex linguistic anomaly patterns
        """
        payload_dict = payload.to_dict()
        user_prompt = build_unstructured_analysis_prompt(payload_dict)

        messages = [
            {"role": "system", "content": UNSTRUCTURED_ANALYSIS_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        response = await asyncio.to_thread(self._call_llm, messages)
        content = response.get("content", "")

        return self._parse_llm_findings(content)

    def _call_llm(self, messages: list[dict]) -> dict:
        """Call Qwen 3.5 9B for NLU analysis."""
        if self._llm is None:
            return {"content": "", "tool_calls": []}

        try:
            from config.settings import OLLAMA_CFG
            from config.vram_manager import assistant_mode

            model = self._llm._ensure_model_available()

            kwargs: dict[str, Any] = {
                "model": model,
                "messages": messages,
                "options": {
                    "temperature": 0.2,  # lower than main agent for precision
                    "num_predict": self._cfg.max_thinking_tokens,
                    "num_ctx": OLLAMA_CFG.num_ctx,
                },
            }

            with assistant_mode():
                response = self._llm._client.chat(**kwargs)

            return {"content": response.message.content or ""}

        except Exception as exc:
            logger.error("NLU LLM call failed: %s", exc)
            return {"content": ""}

    def _parse_llm_findings(self, content: str) -> list[SemanticFinding]:
        """Parse the LLM response into a list of SemanticFinding objects."""
        if not content:
            return []

        # Extract JSON from the response
        parsed = self._extract_json(content)
        if parsed is None:
            logger.warning("Failed to parse NLU LLM response as JSON")
            return []

        raw_findings = parsed.get("findings", [])
        findings = []

        for raw in raw_findings:
            try:
                anomaly_type_str = raw.get("anomaly_type", "")
                try:
                    anomaly_type = SemanticAnomalyType(anomaly_type_str)
                except ValueError:
                    logger.warning("Unknown anomaly_type from LLM: %s", anomaly_type_str)
                    continue

                confidence_str = raw.get("confidence", "LOW")
                try:
                    confidence = FindingConfidence(confidence_str)
                except ValueError:
                    confidence = FindingConfidence.LOW

                severity = float(raw.get("severity_score", 0.5))
                severity = max(0.0, min(1.0, severity))

                findings.append(SemanticFinding(
                    anomaly_type=anomaly_type,
                    confidence=confidence,
                    severity_score=severity,
                    description=str(raw.get("description", "")),
                    evidence_text=str(raw.get("evidence_text", "")),
                    field_source=str(raw.get("field_source", "llm_analysis")),
                ))
            except (KeyError, TypeError, ValueError) as exc:
                logger.warning("Skipping malformed LLM finding: %s — %s", raw, exc)
                continue

        return findings

    def _extract_json(self, content: str) -> dict | None:
        """Extract a JSON object from LLM output."""
        # Try ```json ... ``` blocks first
        json_blocks = re.findall(
            r"```(?:json)?\s*\n?(.*?)\n?\s*```", content, re.DOTALL,
        )
        for block in json_blocks:
            try:
                parsed = json.loads(block.strip())
                if isinstance(parsed, dict):
                    return parsed
            except json.JSONDecodeError:
                continue

        # Try bare JSON objects
        brace_depth = 0
        start = -1
        for i, ch in enumerate(content):
            if ch == "{":
                if brace_depth == 0:
                    start = i
                brace_depth += 1
            elif ch == "}":
                brace_depth -= 1
                if brace_depth == 0 and start >= 0:
                    try:
                        parsed = json.loads(content[start:i + 1])
                        if isinstance(parsed, dict):
                            return parsed
                    except json.JSONDecodeError:
                        pass
                    start = -1

        return None

    # ── Finding Merge & Deduplication ─────────────────────────────────────

    def _merge_findings(
        self,
        heuristic: list[SemanticFinding],
        llm: list[SemanticFinding],
    ) -> list[SemanticFinding]:
        """
        Merge heuristic and LLM findings, deduplicating by anomaly_type
        + field_source.  When both sources report the same anomaly,
        keep the one with higher severity.
        """
        seen: dict[tuple[str, str], SemanticFinding] = {}

        # Heuristics go in first (they are deterministic)
        for f in heuristic:
            key = (f.anomaly_type.value, f.field_source)
            seen[key] = f

        # LLM findings override only if higher severity
        for f in llm:
            key = (f.anomaly_type.value, f.field_source)
            existing = seen.get(key)
            if existing is None or f.severity_score > existing.severity_score:
                seen[key] = f

        return sorted(
            seen.values(),
            key=lambda f: f.severity_score,
            reverse=True,
        )

    # ── Aggregate Score Computation ───────────────────────────────────────

    def _compute_social_engineering_score(
        self, findings: list[SemanticFinding],
    ) -> float:
        """Compute aggregate social engineering risk score."""
        se_types = {
            SemanticAnomalyType.URGENCY_LANGUAGE,
            SemanticAnomalyType.AUTHORITY_IMPERSONATION,
            SemanticAnomalyType.PRETEXTING_PATTERN,
        }
        relevant = [f for f in findings if f.anomaly_type in se_types]
        if not relevant:
            return 0.0
        return min(1.0, sum(f.severity_score for f in relevant) / len(relevant) + 0.1 * len(relevant))

    def _compute_linguistic_anomaly_score(
        self, findings: list[SemanticFinding],
    ) -> float:
        """Compute aggregate linguistic anomaly score."""
        la_types = {
            SemanticAnomalyType.NAME_TRANSLITERATION_MISMATCH,
            SemanticAnomalyType.NAME_PATTERN_ANOMALY,
            SemanticAnomalyType.NAME_ENCODING_ARTEFACT,
            SemanticAnomalyType.CHARSET_HOMOGLYPH,
            SemanticAnomalyType.LANGUAGE_MIX_ANOMALY,
            SemanticAnomalyType.SWIFT_FIELD_INCONSISTENCY,
            SemanticAnomalyType.MESSAGE_TEMPLATE_DEVIATION,
            SemanticAnomalyType.REMITTANCE_INFO_ANOMALY,
        }
        relevant = [f for f in findings if f.anomaly_type in la_types]
        if not relevant:
            return 0.0
        return min(1.0, sum(f.severity_score for f in relevant) / len(relevant) + 0.1 * len(relevant))

    def _compute_device_anomaly_score(
        self, findings: list[SemanticFinding],
    ) -> float:
        """Compute aggregate device/channel anomaly score."""
        da_types = {
            SemanticAnomalyType.USER_AGENT_SPOOFING,
            SemanticAnomalyType.DEVICE_FINGERPRINT_JUMP,
            SemanticAnomalyType.EMULATOR_SIGNATURE,
            SemanticAnomalyType.VPN_TOR_INDICATOR,
        }
        relevant = [f for f in findings if f.anomaly_type in da_types]
        if not relevant:
            return 0.0
        return min(1.0, sum(f.severity_score for f in relevant) / len(relevant) + 0.1 * len(relevant))

    def _compute_risk_modifier(
        self,
        se_score: float,
        la_score: float,
        da_score: float,
    ) -> float:
        """
        Compute the overall risk modifier to inject into the main agent's
        consensus loop.

        Range: -0.05 (no findings, slight reduction) to +0.30 (critical findings).

        Weighting rationale:
        - Social engineering: highest weight because it indicates external threat
          actor involvement (SWIFT attack class)
        - Linguistic anomalies: high weight because name manipulation is a
          primary beneficiary fraud vector
        - Device anomalies: moderate weight (could be legitimate multi-device)
        """
        if se_score == 0.0 and la_score == 0.0 and da_score == 0.0:
            return -0.05  # no findings = slight confidence reduction in fraud

        weighted = (
            se_score * 0.40 +
            la_score * 0.35 +
            da_score * 0.25
        )
        return min(0.30, round(weighted * 0.35, 4))

    # ── Diagnostics ───────────────────────────────────────────────────────

    def snapshot(self) -> dict:
        return {
            "metrics": self.metrics.snapshot(),
            "heuristic_rules": {
                "emulator_signatures": len(_EMULATOR_SIGNATURES),
                "urgency_phrases": len(_URGENCY_PHRASES),
                "authority_keywords": len(_AUTHORITY_KEYWORDS),
                "vpn_tor_indicators": len(_VPN_TOR_INDICATORS),
                "suspicious_codepoints": len(_SUSPICIOUS_CODEPOINTS),
                "cyrillic_lookalikes": len(_CYRILLIC_LOOKALIKES),
            },
        }

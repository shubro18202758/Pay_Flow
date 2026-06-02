"""Union Bank domain policy extracted from the fraud case-study knowledge base.

The document maps PayFlow to Union Bank's EFRMS/SOC operating model, RBI/MoF
governance, KYC/CDD, beneficiary controls, digital-channel safeguards, and
FIU/CBI reporting workflow. This module keeps those rules in code so feature
engineering, LLM evidence, API authorization, and UI access use the same source.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


UNION_BANK_DOMAIN_THRESHOLDS = {
    "rbi_fraud_police_reporting_paisa": 100_000 * 100,      # INR 1 lakh
    "cash_transaction_report_paisa": 1_000_000 * 100,       # INR 10 lakh
    "structuring_watch_floor_paisa": 850_000 * 100,         # 85% of CTR threshold
    "upi_mule_split_reference_paisa": 10_000 * 100,         # common mule split size
    "large_digital_transfer_paisa": 200_000 * 100,          # INR 2 lakh
    "high_geo_deviation_km": 500.0,
}


@dataclass(frozen=True)
class RolePolicy:
    role: str
    label: str
    domain: str
    summary: str
    tabs: tuple[str, ...]
    permissions: frozenset[str]
    feature_focus: tuple[str, ...]
    escalation_scope: str
    shift: str
    reporting_line: str
    decision_authority: str
    tool_stack: tuple[str, ...]
    workflow_steps: tuple[str, ...]


ROLE_POLICIES: dict[str, RolePolicy] = {
    "soc_analyst": RolePolicy(
        role="soc_analyst",
        label="SOC Analyst",
        domain="24x7 EFRMS / Security Operations Centre",
        summary="Monitors live alerts, device/session anomalies, and external fraud signals.",
        tabs=("pre-fraud-intel", "overview", "analytics", "system"),
        permissions=frozenset({
            "ops:view",
            "intel:view",
            "analytics:view",
            "system:view",
            "explain:view",
            "soc:monitor",
        }),
        feature_focus=("device_mfa", "off_hours", "digital_velocity", "soc_queue"),
        escalation_scope="Can triage alerts and raise cases; cannot execute freezes or regulatory filings.",
        shift="24x7 L1 monitoring",
        reporting_line="SOC L2 / Cyber Security Operations",
        decision_authority="Acknowledge EFRMS/SIEM alerts and escalate confirmed account or device anomalies.",
        tool_stack=("EFRMS/Clari5", "SIEM", "UEBA", "Threat Intelligence Platform"),
        workflow_steps=(
            "Correlate device, session, MFA, IP and velocity alerts.",
            "Escalate confirmed suspicious banking activity to Fraud Analyst or Incident Responder.",
            "Keep evidence read-only and preserve event timeline for SOC L2 review.",
        ),
    ),
    "soc_l2_incident_responder": RolePolicy(
        role="soc_l2_incident_responder",
        label="SOC L2 / Incident Responder",
        domain="Cyber Incident Response",
        summary="Contains malware, phishing, endpoint compromise, and payment-channel intrusion signals.",
        tabs=("pre-fraud-intel", "overview", "threat-sim", "investigations", "intelligence", "analytics", "system"),
        permissions=frozenset({
            "ops:view",
            "intel:view",
            "analytics:view",
            "simulation:write",
            "case:view",
            "case:launch",
            "explain:view",
            "soc:monitor",
            "soc:isolate",
            "threat:intel:write",
            "system:view",
        }),
        feature_focus=("phishing_chain", "malware_ioc", "endpoint_isolation", "lateral_movement"),
        escalation_scope="Can isolate compromised endpoints and launch cyber-linked cases; cannot approve banking freezes or FIU filings.",
        shift="24x7 L2 containment",
        reporting_line="CISO / Incident Response Lead",
        decision_authority="Approve endpoint isolation and cyber containment actions before fraud operations act on accounts.",
        tool_stack=("SIEM", "SOAR", "EDR", "Forensics Toolkit", "TIP"),
        workflow_steps=(
            "Validate phishing, malware, PowerShell, or lateral-movement alerts.",
            "Isolate impacted workstation or device session and preserve volatile evidence.",
            "Route banking-loss exposure to Fraud Analyst with cyber timeline attached.",
        ),
    ),
    "threat_hunter": RolePolicy(
        role="threat_hunter",
        label="Threat Hunter",
        domain="Cyber Threat Intelligence",
        summary="Finds mule recruitment, phishing infrastructure, APK campaigns, and OSINT-linked indicators.",
        tabs=("pre-fraud-intel", "overview", "threat-sim", "intelligence", "analytics", "system"),
        permissions=frozenset({
            "ops:view",
            "intel:view",
            "intel:write",
            "analytics:view",
            "simulation:write",
            "explain:view",
            "threat:intel:write",
            "soc:monitor",
            "system:view",
        }),
        feature_focus=("osint_ioc", "apk_campaign", "phishing_domain", "mule_recruitment"),
        escalation_scope="Can publish threat intelligence and test controls; cannot decide customer cases or regulatory filings.",
        shift="Threat-led hunt cycles",
        reporting_line="Cyber Security / SOC L3",
        decision_authority="Promote verified IOCs and playbooks into monitoring queues for SOC and fraud analysts.",
        tool_stack=("TIP", "OSINT", "SIEM", "SOAR", "Sandbox Analysis"),
        workflow_steps=(
            "Collect OSINT and external bank-fraud intelligence.",
            "Map indicators to impacted channels, devices, apps, and accounts.",
            "Publish playbooks for SOC and EFRMS tuning after verification.",
        ),
    ),
    "fraud_analyst": RolePolicy(
        role="fraud_analyst",
        label="Fraud Analyst",
        domain="Transaction Monitoring and Fraud Management Department",
        summary="Investigates fund-flow cases, validates graph evidence, and prepares analyst decisions.",
        tabs=("pre-fraud-intel", "overview", "threat-sim", "investigations", "intelligence", "analytics"),
        permissions=frozenset({
            "ops:view",
            "intel:view",
            "intel:write",
            "analytics:view",
            "simulation:write",
            "case:launch",
            "case:view",
            "case:decide",
            "evidence:package",
            "explain:view",
            "countermeasure:reject",
            "alert:hold",
            "card:hotlist",
        }),
        feature_focus=("velocity", "mule_network", "round_tripping", "profile_mismatch", "evidence_package"),
        escalation_scope="Can decide analyst queue items and package evidence; high-impact freezes need committee approval.",
        shift="24x7 transaction monitoring",
        reporting_line="Fraud Risk Management / Transaction Monitoring Lead",
        decision_authority="Place immediate holds/hotlists within analyst thresholds and package cases for committee or compliance.",
        tool_stack=("EFRMS/Clari5", "Finacle CBS", "Case Management", "Graph Analytics"),
        workflow_steps=(
            "Review UPI/IMPS/cards/net-banking velocity, beneficiary and device signals.",
            "Use graph evidence to validate mule, layering, pass-through, or round-tripping behavior.",
            "Decide analyst-level cases and escalate high-impact freezes or filings.",
        ),
    ),
    "transaction_officer": RolePolicy(
        role="transaction_officer",
        label="Transaction Officer",
        domain="Digital Payments Operations",
        summary="Executes urgent payment holds, card hotlisting, beneficiary checks, and customer-impact triage.",
        tabs=("overview", "investigations", "analytics", "compliance"),
        permissions=frozenset({
            "ops:view",
            "analytics:view",
            "case:view",
            "case:launch",
            "customer:contact",
            "cfr:check",
            "explain:view",
            "alert:hold",
            "card:hotlist",
        }),
        feature_focus=("payment_hold", "card_hotlist", "beneficiary_status", "customer_callback"),
        escalation_scope="Can execute operational holds and hotlisting; cannot tune rules, approve freezes, or file STR/FMR.",
        shift="Customer-impact payment desk",
        reporting_line="Digital Banking Operations / FRM Desk",
        decision_authority="Temporarily hold suspicious transactions and hotlist exposed cards within documented thresholds.",
        tool_stack=("Finacle CBS", "EFRMS/Clari5", "Card Switch", "Case Management"),
        workflow_steps=(
            "Check beneficiary registration, recent payee changes, card exposure, and customer confirmation.",
            "Hold suspect payment or hotlist card when immediate customer protection is required.",
            "Hand confirmed fraud evidence to Fraud Analyst and Compliance Officer.",
        ),
    ),
    "efrms_specialist": RolePolicy(
        role="efrms_specialist",
        label="EFRMS Specialist",
        domain="Fraud Rules and Scenario Tuning",
        summary="Tunes EFRMS scenarios, champion-challenger controls, and false-positive feedback loops.",
        tabs=("pre-fraud-intel", "overview", "threat-sim", "intelligence", "analytics", "system"),
        permissions=frozenset({
            "ops:view",
            "intel:view",
            "intel:write",
            "analytics:view",
            "simulation:write",
            "explain:view",
            "rules:toggle",
            "model:feedback",
            "system:view",
        }),
        feature_focus=("scenario_tuning", "false_positive_feedback", "threshold_drift", "champion_challenger"),
        escalation_scope="Can tune and test rules; cannot decide customer fraud cases or submit regulatory filings.",
        shift="Business-hours tuning with 24x7 emergency support",
        reporting_line="FRM Rules Lead / Analytics Team",
        decision_authority="Promote validated rule changes after simulation and committee-backed emergency approval.",
        tool_stack=("EFRMS/Clari5", "Python Analytics", "Model Monitor", "Case Feedback"),
        workflow_steps=(
            "Review false positives, missed frauds, and analyst feedback.",
            "Simulate rule/risk-threshold changes before rollout.",
            "Deploy or rollback EFRMS scenarios with audit evidence.",
        ),
    ),
    "branch_ops": RolePolicy(
        role="branch_ops",
        label="Branch Operations",
        domain="Branch / Customer Contact",
        summary="Reviews customer context, beneficiary registration, and branch-level remediation evidence.",
        tabs=("overview", "investigations", "compliance"),
        permissions=frozenset({
            "ops:view",
            "case:view",
            "customer:contact",
            "cfr:check",
            "explain:view",
        }),
        feature_focus=("kyc_profile", "beneficiary_prereg", "branch_mismatch", "customer_contact"),
        escalation_scope="Can add customer-contact context and branch observations; cannot file reports or alter controls.",
        shift="Branch business hours / customer callbacks",
        reporting_line="Branch Manager / Regional Operations",
        decision_authority="Confirm customer contact, KYC, and branch observations before central case closure.",
        tool_stack=("Finacle CBS", "KYC/CDD Records", "Case Management", "Customer Contact Log"),
        workflow_steps=(
            "Confirm customer identity, branch relationship, and recent beneficiary changes.",
            "Record customer contact outcome and branch-level remediation evidence.",
            "Escalate disputed or high-value cases to Fraud Analyst and Compliance Officer.",
        ),
    ),
    "aml_analyst": RolePolicy(
        role="aml_analyst",
        label="AML Analyst",
        domain="AML / KYC / Mule Network Monitoring",
        summary="Investigates structuring, layering, mule accounts, CDD gaps, and suspicious transaction drafts.",
        tabs=("overview", "investigations", "intelligence", "analytics", "compliance"),
        permissions=frozenset({
            "ops:view",
            "analytics:view",
            "case:view",
            "evidence:package",
            "cfr:check",
            "explain:view",
            "aml:cdd",
            "aml:str:draft",
        }),
        feature_focus=("structuring", "layering", "mule_network", "kyc_cdd", "watchlist_context"),
        escalation_scope="Can draft AML/STR evidence and CDD notes; Principal Officer or Compliance Officer authorizes filing.",
        shift="AML investigation queue",
        reporting_line="AML Manager / Principal Officer",
        decision_authority="Draft suspicious transaction rationale and recommend account restrictions for approval.",
        tool_stack=("AML Suite", "Watchlists", "Finacle CBS", "Graph Analytics", "Case Management"),
        workflow_steps=(
            "Detect structured deposits, rapid consolidation, and mule-network links.",
            "Review KYC/CDD, occupation, geography, and customer-risk profile.",
            "Draft STR/FIU rationale for Principal Officer authorization.",
        ),
    ),
    "compliance_officer": RolePolicy(
        role="compliance_officer",
        label="Compliance Officer",
        domain="RBI / FIU-IND / CBI Reporting",
        summary="Owns regulatory reporting, fraud registry updates, and FIU-ready evidence review.",
        tabs=("overview", "investigations", "intelligence", "analytics", "compliance"),
        permissions=frozenset({
            "ops:view",
            "analytics:view",
            "case:view",
            "evidence:package",
            "regulatory:file",
            "cfr:report",
            "cfr:check",
            "fiu:disseminate",
            "explain:view",
            "consortium:publish",
            "aml:str:authorize",
            "fraud:fmr:file",
        }),
        feature_focus=("rbi_reporting", "ctr_threshold", "cfr_match", "audit_hash", "fiu_package"),
        escalation_scope="Can file STR/CTR/FMR/CFR artifacts and coordinate external reporting workflows.",
        shift="Regulatory reporting desk",
        reporting_line="Principal Officer / Compliance Head",
        decision_authority="Authorize reportable fraud, STR/CTR/FMR, CFR, DAKSH and external reporting packages.",
        tool_stack=("FIU-IND Portal", "RBI DAKSH", "CFR", "Case Management", "Audit Ledger"),
        workflow_steps=(
            "Validate analyst evidence against RBI/FIU thresholds and audit-chain completeness.",
            "Authorize STR/CTR/FMR/CFR submissions and dissemination packages.",
            "Coordinate CBI/police/NCRP/1930 workflows when thresholds or incident type require it.",
        ),
    ),
    "principal_officer": RolePolicy(
        role="principal_officer",
        label="Principal Officer / MLRO",
        domain="AML Compliance Authority",
        summary="Owns STR authorization, FIU dissemination, AML governance, and suspicious activity sign-off.",
        tabs=("overview", "investigations", "intelligence", "analytics", "compliance"),
        permissions=frozenset({
            "ops:view",
            "analytics:view",
            "case:view",
            "evidence:package",
            "regulatory:file",
            "cfr:check",
            "fiu:disseminate",
            "explain:view",
            "consortium:publish",
            "aml:cdd",
            "aml:str:draft",
            "aml:str:authorize",
        }),
        feature_focus=("str_authorization", "aml_governance", "watchlist_match", "fiu_dissemination"),
        escalation_scope="Can approve AML filings and FIU dissemination; does not directly operate cyber containment.",
        shift="AML sign-off queue",
        reporting_line="Chief Compliance Officer / Board Compliance Committee",
        decision_authority="Authorize STR/FIU submissions and AML governance decisions after analyst review.",
        tool_stack=("AML Suite", "FIU-IND Portal", "Watchlists", "Case Management", "Audit Ledger"),
        workflow_steps=(
            "Review AML Analyst STR draft, CDD evidence, and graph-risk rationale.",
            "Authorize or reject STR/FIU filing with auditable explanation.",
            "Feed governance outcomes back into AML and EFRMS rule review.",
        ),
    ),
    "fraud_investigator": RolePolicy(
        role="fraud_investigator",
        label="Fraud Investigator",
        domain="Fraud Investigation Unit",
        summary="Builds case evidence, root-cause timelines, common-point-of-compromise links, and LEA-ready packs.",
        tabs=("overview", "threat-sim", "investigations", "intelligence", "analytics", "compliance"),
        permissions=frozenset({
            "ops:view",
            "intel:view",
            "analytics:view",
            "simulation:write",
            "case:view",
            "case:launch",
            "case:decide",
            "evidence:package",
            "cfr:check",
            "explain:view",
            "fraud:fmr:file",
        }),
        feature_focus=("case_timeline", "common_point_of_compromise", "lea_package", "root_cause"),
        escalation_scope="Can conclude investigation evidence and FMR packs; freeze/rule decisions still need committee authority.",
        shift="Investigation queue",
        reporting_line="Fraud Investigation Unit Lead",
        decision_authority="Determine investigation findings and prepare law-enforcement/regulatory evidence packs.",
        tool_stack=("Case Management", "Graph Analytics", "Forensics Toolkit", "CFR", "Audit Ledger"),
        workflow_steps=(
            "Correlate events, customer complaint, graph pattern, and channel telemetry.",
            "Identify common-point-of-compromise and linked account/entity clusters.",
            "Package findings for FMR/CFR/LEA workflows with audit trace.",
        ),
    ),
    "fraud_committee": RolePolicy(
        role="fraud_committee",
        label="Fraud Committee",
        domain="Board / Executive Fraud Committee",
        summary="Approves high-impact remediation, account freezes, and enterprise policy overrides.",
        tabs=("pre-fraud-intel", "overview", "threat-sim", "investigations", "intelligence", "analytics", "compliance", "system"),
        permissions=frozenset({
            "ops:view",
            "intel:view",
            "intel:write",
            "analytics:view",
            "simulation:write",
            "case:launch",
            "case:view",
            "case:decide",
            "countermeasure:decide",
            "countermeasure:reject",
            "evidence:package",
            "regulatory:file",
            "cfr:report",
            "cfr:check",
            "fiu:disseminate",
            "rules:toggle",
            "circuit_breaker:trigger",
            "explain:view",
            "consortium:publish",
            "system:view",
            "alert:hold",
            "card:hotlist",
            "aml:cdd",
            "aml:str:draft",
            "aml:str:authorize",
            "soc:isolate",
            "soc:monitor",
            "threat:intel:write",
            "model:feedback",
            "audit:review",
            "risk:view",
            "fraud:fmr:file",
        }),
        feature_focus=("enterprise_risk", "committee_threshold", "freeze_authority", "model_feedback"),
        escalation_scope="Can authorize freezes, rule changes, reporting decisions, and remediation execution.",
        shift="Executive approval board",
        reporting_line="Board-level Risk/Fraud Governance",
        decision_authority="Final authority for high-impact freezes, rule overrides, regulatory packages, and cross-team remediation.",
        tool_stack=("Committee Docket", "EFRMS/Clari5", "Case Management", "RBI/FIU Evidence", "Audit Ledger"),
        workflow_steps=(
            "Review analyst, AML, compliance, cyber, and branch evidence together.",
            "Approve or reject high-impact freezes, emergency rules, and external reporting.",
            "Mandate feedback into EFRMS, SOC playbooks, branch controls, and model governance.",
        ),
    ),
    "risk_analyst": RolePolicy(
        role="risk_analyst",
        label="Risk Analyst",
        domain="Enterprise Risk Management",
        summary="Tracks fraud exposure, thresholds, scenario loss, and portfolio-level control effectiveness.",
        tabs=("overview", "analytics", "compliance", "system"),
        permissions=frozenset({
            "ops:view",
            "analytics:view",
            "case:view",
            "cfr:check",
            "explain:view",
            "risk:view",
            "audit:review",
        }),
        feature_focus=("portfolio_loss", "risk_threshold", "control_effectiveness", "scenario_exposure"),
        escalation_scope="Read-mostly risk oversight; cannot decide cases, freeze accounts, or file reports.",
        shift="Risk review cycles",
        reporting_line="CRO / Enterprise Risk Committee",
        decision_authority="Recommend risk appetite and threshold changes for committee review.",
        tool_stack=("Risk Dashboard", "Analytics Warehouse", "Case Management", "Audit Ledger"),
        workflow_steps=(
            "Track exposure by channel, branch, product, and fraud typology.",
            "Assess false positives, control gaps, and residual risk.",
            "Recommend portfolio controls to committee and EFRMS specialists.",
        ),
    ),
    "data_scientist": RolePolicy(
        role="data_scientist",
        label="Data Scientist / ML Engineer",
        domain="AI / Model Governance",
        summary="Monitors model drift, feature quality, false-positive loops, and Qwen reasoning telemetry.",
        tabs=("overview", "intelligence", "analytics", "system"),
        permissions=frozenset({
            "ops:view",
            "analytics:view",
            "intel:view",
            "system:view",
            "explain:view",
            "model:feedback",
            "audit:review",
        }),
        feature_focus=("model_drift", "feature_quality", "xai_reasoning", "feedback_loop"),
        escalation_scope="Can review and annotate model behavior; cannot operate fraud controls or file external reports.",
        shift="Model monitoring cycles",
        reporting_line="Analytics Head / Model Risk Governance",
        decision_authority="Recommend model/rule recalibration based on measured drift and analyst feedback.",
        tool_stack=("Python Analytics", "Model Monitor", "Qwen Runtime Telemetry", "Feature Store", "Audit Ledger"),
        workflow_steps=(
            "Review drift, feature attribution, and model-derived fraud explanations.",
            "Compare analyst outcomes against model scores and Qwen reasoning traces.",
            "Feed approved changes to EFRMS Specialist and Fraud Committee.",
        ),
    ),
    "internal_audit": RolePolicy(
        role="internal_audit",
        label="Internal Audit",
        domain="Independent Audit and Control Testing",
        summary="Reviews case actions, permission usage, evidence integrity, and regulatory control adherence.",
        tabs=("overview", "investigations", "analytics", "compliance", "system"),
        permissions=frozenset({
            "ops:view",
            "analytics:view",
            "case:view",
            "cfr:check",
            "explain:view",
            "audit:review",
            "system:view",
        }),
        feature_focus=("audit_hash", "segregation_of_duties", "control_testing", "evidence_integrity"),
        escalation_scope="Read-only audit oversight; no operational write, freeze, rule, or filing authority.",
        shift="Periodic and event-triggered audit",
        reporting_line="Internal Audit / Audit Committee",
        decision_authority="Raise audit findings and control exceptions without changing operational state.",
        tool_stack=("Audit Ledger", "Case Management", "RBI/FIU Evidence", "Access Logs"),
        workflow_steps=(
            "Verify role segregation, action approvals, and evidence-chain hashes.",
            "Test whether cases met RBI/FIU/CFR reporting obligations.",
            "Raise audit exceptions for governance remediation.",
        ),
    ),
    "system_admin": RolePolicy(
        role="system_admin",
        label="System Administrator",
        domain="Platform Operations",
        summary="Maintains runtime health, model/service telemetry, and platform controls without case authority.",
        tabs=("overview", "analytics", "intelligence", "system"),
        permissions=frozenset({
            "ops:view",
            "analytics:view",
            "system:view",
            "rules:toggle",
            "explain:view",
        }),
        feature_focus=("runtime_health", "model_drift", "pipeline_latency", "service_integrity"),
        escalation_scope="Can tune platform controls; cannot approve fraud decisions or file external reports.",
        shift="Platform support",
        reporting_line="Technology Operations",
        decision_authority="Maintain service health, deployments, and runtime controls without case authority.",
        tool_stack=("FastAPI", "SSE", "Ollama/Qwen Runtime", "Nixpacks", "Observability Logs"),
        workflow_steps=(
            "Monitor API, SSE, model runtime, and pipeline health.",
            "Apply platform configuration changes with no direct fraud-case authority.",
            "Escalate model or data-quality issues to Data Scientist and EFRMS Specialist.",
        ),
    ),
}


OPERATIONAL_WORKFLOWS: tuple[dict[str, Any], ...] = (
    {
        "id": "upi_remote_access_scam",
        "title": "UPI remote-access scam",
        "trigger": "Fast UPI velocity, new device/MFA context, remote-support or screen-share signal.",
        "stages": (
            {"owner": "soc_analyst", "action": "Detect device/session anomaly and live EFRMS alert."},
            {"owner": "fraud_analyst", "action": "Place analyst hold, inspect beneficiary velocity and mule graph."},
            {"owner": "transaction_officer", "action": "Execute payment hold or card hotlist where customer exposure is active."},
            {"owner": "branch_ops", "action": "Confirm customer contact, KYC, and complaint timeline."},
            {"owner": "compliance_officer", "action": "Prepare 1930/NCRP/DAKSH/FMR package when thresholds are met."},
        ),
    },
    {
        "id": "phishing_malware_intrusion",
        "title": "Phishing or malware-linked banking intrusion",
        "trigger": "Suspicious PowerShell, endpoint anomaly, malicious APK, credential theft, or lateral movement.",
        "stages": (
            {"owner": "threat_hunter", "action": "Verify phishing domain, APK, IOC and external campaign context."},
            {"owner": "soc_l2_incident_responder", "action": "Isolate workstation/device session and preserve forensic evidence."},
            {"owner": "fraud_analyst", "action": "Connect cyber timeline to money movement and impacted accounts."},
            {"owner": "efrms_specialist", "action": "Tune digital-channel scenarios after validated IOC/playbook."},
            {"owner": "fraud_committee", "action": "Approve high-impact containment or emergency rule rollout."},
        ),
    },
    {
        "id": "mule_layering_network",
        "title": "Mule account and layering network",
        "trigger": "Structured deposits, rapid consolidation, pass-through behavior, dormant activation, or CFR match.",
        "stages": (
            {"owner": "aml_analyst", "action": "Perform CDD/KYC review and draft STR rationale."},
            {"owner": "fraud_investigator", "action": "Build graph evidence and connected-account investigation package."},
            {"owner": "principal_officer", "action": "Authorize STR/FIU dissemination after AML review."},
            {"owner": "compliance_officer", "action": "File STR/CTR/FMR/CFR artifacts with audit hash."},
            {"owner": "risk_analyst", "action": "Review portfolio exposure and recommend control tightening."},
        ),
    },
    {
        "id": "atm_card_cloning",
        "title": "ATM skimming or card cloning",
        "trigger": "Impossible travel, repeated card present failures, common ATM point, or cross-border usage.",
        "stages": (
            {"owner": "soc_analyst", "action": "Monitor impossible-travel and device/payment-rail anomalies."},
            {"owner": "transaction_officer", "action": "Hotlist exposed card and protect customer funds."},
            {"owner": "fraud_investigator", "action": "Correlate common point of compromise and linked victims."},
            {"owner": "compliance_officer", "action": "Prepare FMR/CFR and external reporting artifacts."},
            {"owner": "internal_audit", "action": "Validate evidence chain, approvals, and segregation of duties."},
        ),
    },
)

OPERATING_UNITS: tuple[dict[str, Any], ...] = (
    {
        "id": "frm",
        "label": "Fraud Risk Management",
        "mandate": "Sets fraud risk appetite, EWS/RFA policy, and bank-wide fraud governance rather than minute-by-minute alert handling.",
        "tools": ("RBI Master Directions", "Risk appetite", "EFRMS policy", "Committee docket"),
        "authority": "Policy and governance authority; operational execution sits with transaction monitoring, EFRMS, branch, compliance and committee roles.",
        "reporting_line": "Fraud Risk Management Head / Risk Governance",
    },
    {
        "id": "efrms",
        "label": "Enterprise Fraud Risk Management System",
        "mandate": "Operates cross-channel transaction monitoring across ATM, POS, UPI, internet banking and mobile banking.",
        "tools": ("Clari5/EFRMS", "Rule engine", "Scenario tuning", "Case feedback"),
        "authority": "Deploys and rolls back rules after evidence, simulation and governance approval.",
        "reporting_line": "EFRMS Team Lead / FRM Rules Lead",
    },
    {
        "id": "transaction_monitoring",
        "label": "Transaction Monitoring",
        "mandate": "Reviews EFRMS alerts, verifies customer exposure, initiates customer calls, temporary debit holds and card hotlisting.",
        "tools": ("EFRMS/Clari5", "Finacle CBS", "CRM", "Case Management"),
        "authority": "High-speed temporary controls within documented thresholds; high-impact freezes require committee approval.",
        "reporting_line": "EFRMS Team Lead / Shift Manager",
    },
    {
        "id": "soc",
        "label": "Security Operations Centre",
        "mandate": "Monitors SIEM/EDR/network telemetry for unauthorized access, malware, data exfiltration and infrastructure compromise.",
        "tools": ("SIEM", "EDR", "UEBA", "Network monitoring"),
        "authority": "L1/L2 triage and escalation; isolation authority sits with incident response and senior cyber roles.",
        "reporting_line": "SOC Manager / CISO",
    },
    {
        "id": "incident_response",
        "label": "Incident Response",
        "mandate": "Contains confirmed cyber breaches, isolates affected infrastructure, performs volatile forensics and coordinates recovery.",
        "tools": ("SOAR", "EDR", "Forensics toolkit", "Wireshark", "EnCase/FTK"),
        "authority": "High technical containment authority during active incidents; fraud disposition still requires fraud/compliance evidence.",
        "reporting_line": "CISO / Incident Response Lead",
    },
    {
        "id": "aml_kyc",
        "label": "AML / KYC",
        "mandate": "Investigates structuring, mule networks, sanctions risk, CDD gaps and suspicious transaction narratives under PMLA/FATF.",
        "tools": ("AML Suite", "KYC/CDD records", "Watchlists", "FIU-IND FINnet"),
        "authority": "Drafts STR/AML rationale and recommends holds; Principal Officer authorizes FIU filings and tipping-off controls apply.",
        "reporting_line": "AML Head / Principal Officer",
    },
    {
        "id": "principal_officer",
        "label": "Principal Officer / MLRO",
        "mandate": "Authorizes STR/FIU dissemination and owns AML compliance sign-off.",
        "tools": ("FIU-IND portal", "AML Suite", "Watchlists", "Audit ledger"),
        "authority": "Final AML filing authority after analyst CDD and evidence review.",
        "reporting_line": "Chief Compliance Officer / Board Compliance Committee",
    },
    {
        "id": "compliance",
        "label": "Compliance",
        "mandate": "Converts RBI, FIU, PMLA, customer grievance and GRC requirements into operational mandates and reporting checks.",
        "tools": ("GRC platform", "RBI DAKSH", "CFR", "FIU-IND", "Case evidence"),
        "authority": "Can enforce compliance mandates and authorize regulatory packages where policy assigns filing responsibility.",
        "reporting_line": "Chief Compliance Officer",
    },
    {
        "id": "fraud_investigation",
        "label": "Fraud Investigation Unit",
        "mandate": "Runs post-event forensic case work, staff accountability review, LEA/CBI/SFIO liaison and evidence dossiers.",
        "tools": ("Case Management", "Finacle CBS", "OSINT", "Forensics Toolkit", "CCTV/branch evidence"),
        "authority": "High authority to direct investigation and request documents; operational freeze/reporting still follows role gates.",
        "reporting_line": "Fraud Control Head / CVO where internal accountability exists",
    },
    {
        "id": "branch_operations",
        "label": "Branch Operations",
        "mandate": "Supplies customer contact, KYC, beneficiary registration, complaint timeline and local remediation evidence.",
        "tools": ("Finacle CBS", "KYC/CDD records", "Customer contact log", "Branch documents"),
        "authority": "Confirms facts and customer context; central fraud/compliance teams decide case outcome and filings.",
        "reporting_line": "Branch Manager / Regional Operations",
    },
    {
        "id": "risk_management",
        "label": "Risk Management",
        "mandate": "Quantifies operational risk exposure, KRIs, capital impact and control effectiveness for governance committees.",
        "tools": ("GRC platform", "Risk dashboards", "Tableau/PowerBI", "Scenario loss models"),
        "authority": "Advisory risk appetite and control recommendations; no direct case-control authority.",
        "reporting_line": "CRO / Risk Management Committee",
    },
    {
        "id": "data_ai",
        "label": "Data Analytics / AI-ML",
        "mandate": "Builds anomaly models, graph analytics, drift monitoring and empirical threshold recommendations.",
        "tools": ("Python", "Feature store", "Model monitor", "Graph analytics", "MLOps"),
        "authority": "Model design and recommendation authority; production rule/action changes require EFRMS and governance approval.",
        "reporting_line": "Head of Analytics / Model Risk Governance",
    },
    {
        "id": "internal_audit",
        "label": "Internal Audit",
        "mandate": "Independently tests evidence integrity, role segregation, reporting timeliness and control adherence.",
        "tools": ("Audit ledger", "Case files", "RBI/FIU artifacts", "Access logs"),
        "authority": "Raises audit findings independently; cannot alter operational state.",
        "reporting_line": "Audit Committee of the Board",
    },
    {
        "id": "digital_banking_security",
        "label": "Digital Banking Security",
        "mandate": "Secures mobile apps, APIs, fintech integrations, DPI rails and digital banking release posture.",
        "tools": ("VAPT", "WAF", "API security", "Mobile security testing", "Secure SDLC"),
        "authority": "Security design and release-control authority through change management.",
        "reporting_line": "CISO / Cybersecurity Centre of Excellence",
    },
)

ROLE_OPERATING_UNIT_MAP: dict[str, tuple[str, ...]] = {
    "soc_analyst": ("soc", "efrms"),
    "soc_l2_incident_responder": ("incident_response", "soc"),
    "threat_hunter": ("soc", "digital_banking_security"),
    "fraud_analyst": ("transaction_monitoring", "frm", "fraud_investigation"),
    "transaction_officer": ("transaction_monitoring", "branch_operations"),
    "efrms_specialist": ("efrms", "frm", "data_ai"),
    "branch_ops": ("branch_operations", "transaction_monitoring"),
    "aml_analyst": ("aml_kyc", "principal_officer"),
    "compliance_officer": ("compliance", "principal_officer", "internal_audit"),
    "principal_officer": ("principal_officer", "aml_kyc", "compliance"),
    "fraud_investigator": ("fraud_investigation", "compliance", "branch_operations"),
    "fraud_committee": ("frm", "compliance", "risk_management", "internal_audit"),
    "risk_analyst": ("risk_management", "frm"),
    "data_scientist": ("data_ai", "efrms", "risk_management"),
    "internal_audit": ("internal_audit", "compliance"),
    "system_admin": ("digital_banking_security", "data_ai"),
}

ROLE_AUTHORITY_BOUNDARIES: dict[str, tuple[str, ...]] = {
    "fraud_analyst": (
        "Temporary holds/hotlists are allowed within analyst thresholds; enterprise freezes remain committee-gated.",
        "Customer confirmation and branch evidence must be captured before central case closure when account context is disputed.",
        "AI/Qwen explanations are evidence summaries, not autonomous fraud classification authority.",
    ),
    "transaction_officer": (
        "Can protect customer funds through payment hold or card hotlist; cannot file STR/FMR or tune EFRMS rules.",
        "Must hand confirmed fraud evidence to Fraud Analyst and Compliance Officer.",
    ),
    "aml_analyst": (
        "Can draft STR rationale and CDD notes; Principal Officer or Compliance authorizes filing.",
        "No customer tipping-off is permitted during AML investigation.",
    ),
    "principal_officer": (
        "Owns STR/FIU authorization after AML evidence review.",
        "Does not directly operate cyber containment or payment-channel holds.",
    ),
    "compliance_officer": (
        "Regulatory packages require analyst evidence, audit hash and threshold checks before filing.",
        "FMR/CFR/FIU actions are compliance workflows, not model-side effects.",
    ),
    "soc_analyst": (
        "SOC L1/L2 can triage and escalate cyber evidence; fraud-case outcome needs transaction-monitoring and compliance review.",
        "Infrastructure alerts must be correlated with financial ledger movement before customer-impact controls.",
    ),
    "soc_l2_incident_responder": (
        "Can isolate compromised device/session/infrastructure; cannot approve banking freezes or FIU filings.",
        "Cyber timeline must be attached to money-movement evidence for fraud operations.",
    ),
    "efrms_specialist": (
        "Can tune scenarios after evidence and simulation; case decisions remain with fraud/compliance roles.",
        "Emergency rule rollout needs governance approval and rollback evidence.",
    ),
    "branch_ops": (
        "Can validate customer/KYC/beneficiary context; cannot alter fraud controls or regulatory filings.",
        "Branch notes feed central case closure, dispute handling and compliance review.",
    ),
    "fraud_investigator": (
        "Can direct post-event investigation and LEA evidence packs; operational freezes and filings remain gated.",
        "Internal staff accountability requires CVO/legal chain-of-custody discipline.",
    ),
    "risk_analyst": (
        "Advisory only: recommends risk appetite and KRIs without operational write authority.",
        "Portfolio exposure does not itself resolve individual customer cases.",
    ),
    "data_scientist": (
        "Can review drift, features and model behavior; cannot operate fraud controls or file external reports.",
        "Model changes require EFRMS validation and governance sign-off.",
    ),
    "internal_audit": (
        "Independent read-only control testing; cannot change operational state.",
        "Audit findings must test segregation of duties and reporting timelines after the fact.",
    ),
    "system_admin": (
        "Maintains service health and runtime controls without fraud-case authority.",
        "Deployment/runtime changes must not bypass role-gated fraud decisions.",
    ),
    "fraud_committee": (
        "Final approval authority for high-impact freezes, emergency rules and enterprise remediation.",
        "Must review analyst, AML, compliance, cyber and branch evidence together before overriding controls.",
    ),
}

REGULATORY_OBLIGATIONS: tuple[dict[str, str], ...] = (
    {
        "id": "rbi_fmr_14_day",
        "label": "RBI FMR timeline",
        "detail": "Fraud Monitoring Return evidence must be prepared within the mandatory reporting window after fraud classification.",
        "owner_unit": "compliance",
    },
    {
        "id": "fiu_str_ctr",
        "label": "FIU STR/CTR control",
        "detail": "STR/CTR reporting requires AML/CDD rationale and Principal Officer or compliance authorization.",
        "owner_unit": "principal_officer",
    },
    {
        "id": "no_tipping_off",
        "label": "No tipping-off",
        "detail": "AML investigation cannot disclose suspicious-reporting intent to the customer.",
        "owner_unit": "aml_kyc",
    },
    {
        "id": "temporary_vs_enterprise_freeze",
        "label": "Temporary vs enterprise freeze",
        "detail": "Frontline payment holds/hotlists are temporary customer-protection controls; high-impact freezes need committee authority.",
        "owner_unit": "transaction_monitoring",
    },
    {
        "id": "audit_independence",
        "label": "Audit independence",
        "detail": "Internal Audit verifies evidence, access use and reporting adherence without operational write authority.",
        "owner_unit": "internal_audit",
    },
)

ORGANIZATIONAL_HIERARCHY: tuple[dict[str, Any], ...] = (
    {
        "level": "Board Committees",
        "entities": ("SCBMF", "Risk Management Committee", "Audit Committee"),
        "core_function": "Apex fraud governance, large-fraud oversight, audit review and risk appetite approval.",
    },
    {
        "level": "Executive",
        "entities": ("MD & CEO", "CRO", "CISO", "CCO", "CVO", "Principal Officer"),
        "core_function": "Strategic alignment, cyber resilience, statutory compliance and enterprise risk formulation.",
    },
    {
        "level": "Mid-Management",
        "entities": ("Fraud Control Head", "SOC Manager", "AML Head", "Head of Analytics"),
        "core_function": "Operational leadership, resource allocation, queue supervision and rule governance.",
    },
    {
        "level": "Supervisory",
        "entities": ("SOC Shift Leads", "EFRMS Team Leads", "MLROs"),
        "core_function": "Shift management, complex escalation handling and evidence quality control.",
    },
    {
        "level": "Operational Floor",
        "entities": ("Fraud Analysts", "SOC L1/L2", "AML Analysts", "Forensics", "Branch Operations"),
        "core_function": "Real-time monitoring, alert triage, customer contact, CDD review and initial evidence capture.",
    },
)


def role_policy(role: str | None) -> RolePolicy:
    if role and role in ROLE_POLICIES:
        return ROLE_POLICIES[role]
    return ROLE_POLICIES["fraud_analyst"]


def operating_units_for_role(role: str | None) -> list[dict[str, Any]]:
    unit_lookup = {unit["id"]: unit for unit in OPERATING_UNITS}
    return [
        unit_lookup[unit_id]
        for unit_id in ROLE_OPERATING_UNIT_MAP.get(role_policy(role).role, ())
        if unit_id in unit_lookup
    ]


def operating_model_for_role(role: str | None) -> dict[str, Any]:
    policy = role_policy(role)
    primary_units = operating_units_for_role(policy.role)
    workflow_touchpoints: list[dict[str, Any]] = []
    for workflow in OPERATIONAL_WORKFLOWS:
        stages = list(workflow["stages"])
        for index, stage in enumerate(stages):
            if stage["owner"] != policy.role:
                continue
            previous_stage = stages[index - 1] if index > 0 else None
            next_stage = stages[index + 1] if index < len(stages) - 1 else None
            workflow_touchpoints.append({
                "workflow_id": workflow["id"],
                "workflow_title": workflow["title"],
                "trigger": workflow["trigger"],
                "step": index + 1,
                "total_steps": len(stages),
                "receives_from": role_policy(previous_stage["owner"]).label if previous_stage else "EFRMS / SIEM trigger",
                "action": stage["action"],
                "hands_to": role_policy(next_stage["owner"]).label if next_stage else "audit / closure",
            })

    primary_unit_ids = {unit["id"] for unit in primary_units}
    relevant_obligations = [
        item for item in REGULATORY_OBLIGATIONS
        if item["owner_unit"] in primary_unit_ids or policy.role in {"fraud_committee", "compliance_officer"}
    ]
    if policy.role in {"fraud_analyst", "transaction_officer"}:
        relevant_obligations.append(next(item for item in REGULATORY_OBLIGATIONS if item["id"] == "temporary_vs_enterprise_freeze"))
    if policy.role == "internal_audit":
        relevant_obligations.append(next(item for item in REGULATORY_OBLIGATIONS if item["id"] == "audit_independence"))
    obligation_by_id = {item["id"]: item for item in relevant_obligations}

    return {
        "source": "Uploaded Indian Bank Fraud & Cyber Team Structure operating model",
        "role": policy.role,
        "primary_units": primary_units,
        "authority_boundaries": list(ROLE_AUTHORITY_BOUNDARIES.get(policy.role, (
            "This role inherits the standard PayFlow rule: AI is advisory and operational actions remain role-gated.",
        ))),
        "workflow_touchpoints": workflow_touchpoints,
        "regulatory_obligations": list(obligation_by_id.values()),
        "all_operating_units_count": len(OPERATING_UNITS),
        "organizational_hierarchy": ORGANIZATIONAL_HIERARCHY,
        "reality_gap_note": (
            "PayFlow is intentionally a prototype control room: it must show how evidence, authority, and handoffs flow, "
            "but it must not imply that Qwen, charts, or graph scores can replace bank role segregation."
        ),
    }


def has_permission(role: str | None, permission: str) -> bool:
    return permission in role_policy(role).permissions


def _channel_name(txn: Any) -> str:
    channel = getattr(txn, "channel", "")
    return getattr(channel, "name", str(channel)).upper()


def domain_feature_flags(
    txn: Any,
    geo_distance_km: float = 0.0,
    off_hours: bool = False,
    cfr_match: bool = False,
    prior_fraud_reports: float = 0.0,
    pass_through_detected: bool = False,
    dormant_activation: bool = False,
) -> dict[str, float]:
    """Return Union Bank domain-side feature flags for a transaction.

    These flags are intentionally kept as a sidecar to the model tensor so the
    existing trained feature dimension stays stable while downstream XAI, LLM,
    and access-controlled views can use real banking context.
    """

    thresholds = UNION_BANK_DOMAIN_THRESHOLDS
    amount = float(getattr(txn, "amount_paisa", 0) or 0)
    channel = _channel_name(txn)
    sender_branch = str(getattr(txn, "sender_branch", "") or "")
    receiver_branch = str(getattr(txn, "receiver_branch", "") or "")

    digital_channels = {"NETBANKING", "MOBILE", "UPI", "IMPS"}
    real_time_rails = {"UPI", "IMPS", "RTGS"}
    interbank_rails = {"NEFT", "RTGS", "SWIFT"}
    beneficiary_channels = {"NETBANKING", "MOBILE", "UPI", "IMPS", "NEFT", "RTGS", "SWIFT"}

    return {
        "ubi_rbi_reportable_ge_1l": float(amount >= thresholds["rbi_fraud_police_reporting_paisa"]),
        "ubi_ctr_reportable_ge_10l": float(amount >= thresholds["cash_transaction_report_paisa"]),
        "ubi_structuring_watch_near_10l": float(
            thresholds["structuring_watch_floor_paisa"] <= amount < thresholds["cash_transaction_report_paisa"]
        ),
        "ubi_upi_mule_split_amount": float(
            channel == "UPI" and 0 < amount <= thresholds["upi_mule_split_reference_paisa"]
        ),
        "ubi_large_digital_transfer": float(
            channel in digital_channels and amount >= thresholds["large_digital_transfer_paisa"]
        ),
        "ubi_real_time_payment_rail": float(channel in real_time_rails),
        "ubi_interbank_rail": float(channel in interbank_rails),
        "ubi_mfa_expected_channel": float(channel in {"NETBANKING", "MOBILE", "UPI"}),
        "ubi_beneficiary_prereg_expected": float(channel in beneficiary_channels),
        "ubi_cross_branch_transfer": float(bool(sender_branch and receiver_branch and sender_branch != receiver_branch)),
        "ubi_high_geo_deviation": float(geo_distance_km >= thresholds["high_geo_deviation_km"]),
        "ubi_soc_off_hours_review": float(off_hours),
        "ubi_cfr_match": float(cfr_match),
        "ubi_prior_fraud_reports": float(max(0.0, prior_fraud_reports)),
        "ubi_pass_through_pattern": float(pass_through_detected),
        "ubi_dormant_reactivation": float(dormant_activation),
    }


DOMAIN_FEATURE_COLUMNS = tuple(domain_feature_flags(type("_Txn", (), {"amount_paisa": 0, "channel": "", "sender_branch": "", "receiver_branch": ""})()).keys())


def domain_controls_for_flags(flags: dict[str, float]) -> list[str]:
    controls: list[str] = []
    if flags.get("ubi_rbi_reportable_ge_1l", 0) >= 1:
        controls.append("RBI/MoF fraud reporting threshold crossed: prepare police/CBI review if fraud is confirmed.")
    if flags.get("ubi_ctr_reportable_ge_10l", 0) >= 1:
        controls.append("CTR/FIU review threshold crossed: compliance reporting context required.")
    elif flags.get("ubi_structuring_watch_near_10l", 0) >= 1:
        controls.append("Structuring watch: amount sits below INR 10 lakh reporting threshold.")
    if flags.get("ubi_upi_mule_split_amount", 0) >= 1:
        controls.append("UPI mule split signal: small rapid payment requires velocity and network review.")
    if flags.get("ubi_mfa_expected_channel", 0) >= 1:
        controls.append("Digital channel requires MFA/OTP/device-session corroboration.")
    if flags.get("ubi_beneficiary_prereg_expected", 0) >= 1:
        controls.append("Beneficiary pre-registration and recent payee-change checks should be reviewed.")
    if flags.get("ubi_cfr_match", 0) >= 1:
        controls.append("Central Fraud Registry match: escalate related accounts and evidence package.")
    if flags.get("ubi_pass_through_pattern", 0) >= 1:
        controls.append("Pass-through behavior: possible mule/layering account under EFRMS monitoring.")
    if flags.get("ubi_dormant_reactivation", 0) >= 1:
        controls.append("Dormant activation: high-value post-dormancy transfer needs branch/customer verification.")
    return controls

export type PayflowRole =
  | 'soc_analyst'
  | 'soc_l2_incident_responder'
  | 'threat_hunter'
  | 'fraud_analyst'
  | 'transaction_officer'
  | 'efrms_specialist'
  | 'branch_ops'
  | 'aml_analyst'
  | 'compliance_officer'
  | 'principal_officer'
  | 'fraud_investigator'
  | 'fraud_committee'
  | 'risk_analyst'
  | 'data_scientist'
  | 'internal_audit'
  | 'system_admin'

export type Permission =
  | 'ops:view'
  | 'intel:view'
  | 'intel:write'
  | 'analytics:view'
  | 'simulation:write'
  | 'case:launch'
  | 'case:view'
  | 'case:decide'
  | 'countermeasure:decide'
  | 'countermeasure:reject'
  | 'evidence:package'
  | 'regulatory:file'
  | 'cfr:report'
  | 'cfr:check'
  | 'fiu:disseminate'
  | 'rules:toggle'
  | 'circuit_breaker:trigger'
  | 'explain:view'
  | 'consortium:publish'
  | 'system:view'
  | 'customer:contact'
  | 'alert:hold'
  | 'card:hotlist'
  | 'aml:cdd'
  | 'aml:str:draft'
  | 'aml:str:authorize'
  | 'soc:monitor'
  | 'soc:isolate'
  | 'threat:intel:write'
  | 'model:feedback'
  | 'audit:review'
  | 'risk:view'
  | 'fraud:fmr:file'

export interface RolePolicy {
  role: PayflowRole
  label: string
  domain: string
  summary: string
  tabs: string[]
  permissions: Permission[]
  featureFocus: string[]
  escalationScope: string
  shift: string
  reportingLine: string
  decisionAuthority: string
  toolStack: string[]
  workflowSteps: string[]
}

export interface OperationalWorkflowStage {
  owner: PayflowRole
  action: string
}

export interface OperationalWorkflow {
  id: string
  title: string
  trigger: string
  stages: OperationalWorkflowStage[]
}

export const DEFAULT_PAYFLOW_ROLE: PayflowRole = 'fraud_analyst'
const ROLE_STORAGE_KEY = 'payflow.unionBankRole'

export const ROLE_ORDER: PayflowRole[] = [
  'soc_analyst',
  'soc_l2_incident_responder',
  'threat_hunter',
  'fraud_analyst',
  'transaction_officer',
  'efrms_specialist',
  'branch_ops',
  'aml_analyst',
  'compliance_officer',
  'principal_officer',
  'fraud_investigator',
  'fraud_committee',
  'risk_analyst',
  'data_scientist',
  'internal_audit',
  'system_admin',
]

export const ROLE_POLICIES: Record<PayflowRole, RolePolicy> = {
  soc_analyst: {
    role: 'soc_analyst',
    label: 'SOC Analyst',
    domain: '24x7 EFRMS / Security Operations Centre',
    summary: 'Monitors live alerts, device/session anomalies, and external fraud signals.',
    tabs: ['pre-fraud-intel', 'overview', 'analytics', 'system'],
    permissions: ['ops:view', 'intel:view', 'analytics:view', 'system:view', 'explain:view', 'soc:monitor'],
    featureFocus: ['device_mfa', 'off_hours', 'digital_velocity', 'soc_queue'],
    escalationScope: 'Can triage alerts and raise cases; cannot execute freezes or regulatory filings.',
    shift: '24x7 L1 monitoring',
    reportingLine: 'SOC L2 / Cyber Security Operations',
    decisionAuthority: 'Acknowledge EFRMS/SIEM alerts and escalate confirmed account or device anomalies.',
    toolStack: ['EFRMS/Clari5', 'SIEM', 'UEBA', 'Threat Intelligence Platform'],
    workflowSteps: [
      'Correlate device, session, MFA, IP and velocity alerts.',
      'Escalate confirmed suspicious banking activity to Fraud Analyst or Incident Responder.',
      'Keep evidence read-only and preserve event timeline for SOC L2 review.',
    ],
  },
  soc_l2_incident_responder: {
    role: 'soc_l2_incident_responder',
    label: 'SOC L2 / Incident Responder',
    domain: 'Cyber Incident Response',
    summary: 'Contains malware, phishing, endpoint compromise, and payment-channel intrusion signals.',
    tabs: ['pre-fraud-intel', 'overview', 'threat-sim', 'investigations', 'intelligence', 'analytics', 'system'],
    permissions: [
      'ops:view',
      'intel:view',
      'analytics:view',
      'simulation:write',
      'case:view',
      'case:launch',
      'explain:view',
      'soc:monitor',
      'soc:isolate',
      'threat:intel:write',
      'system:view',
    ],
    featureFocus: ['phishing_chain', 'malware_ioc', 'endpoint_isolation', 'lateral_movement'],
    escalationScope: 'Can isolate compromised endpoints and launch cyber-linked cases; cannot approve banking freezes or FIU filings.',
    shift: '24x7 L2 containment',
    reportingLine: 'CISO / Incident Response Lead',
    decisionAuthority: 'Approve endpoint isolation and cyber containment actions before fraud operations act on accounts.',
    toolStack: ['SIEM', 'SOAR', 'EDR', 'Forensics Toolkit', 'TIP'],
    workflowSteps: [
      'Validate phishing, malware, PowerShell, or lateral-movement alerts.',
      'Isolate impacted workstation or device session and preserve volatile evidence.',
      'Route banking-loss exposure to Fraud Analyst with cyber timeline attached.',
    ],
  },
  threat_hunter: {
    role: 'threat_hunter',
    label: 'Threat Hunter',
    domain: 'Cyber Threat Intelligence',
    summary: 'Finds mule recruitment, phishing infrastructure, APK campaigns, and OSINT-linked indicators.',
    tabs: ['pre-fraud-intel', 'overview', 'threat-sim', 'intelligence', 'analytics', 'system'],
    permissions: [
      'ops:view',
      'intel:view',
      'intel:write',
      'analytics:view',
      'simulation:write',
      'explain:view',
      'threat:intel:write',
      'soc:monitor',
      'system:view',
    ],
    featureFocus: ['osint_ioc', 'apk_campaign', 'phishing_domain', 'mule_recruitment'],
    escalationScope: 'Can publish threat intelligence and test controls; cannot decide customer cases or regulatory filings.',
    shift: 'Threat-led hunt cycles',
    reportingLine: 'Cyber Security / SOC L3',
    decisionAuthority: 'Promote verified IOCs and playbooks into monitoring queues for SOC and fraud analysts.',
    toolStack: ['TIP', 'OSINT', 'SIEM', 'SOAR', 'Sandbox Analysis'],
    workflowSteps: [
      'Collect OSINT and external bank-fraud intelligence.',
      'Map indicators to impacted channels, devices, apps, and accounts.',
      'Publish playbooks for SOC and EFRMS tuning after verification.',
    ],
  },
  fraud_analyst: {
    role: 'fraud_analyst',
    label: 'Fraud Analyst',
    domain: 'Transaction Monitoring and Fraud Management Department',
    summary: 'Investigates fund-flow cases, validates graph evidence, and prepares analyst decisions.',
    tabs: ['pre-fraud-intel', 'overview', 'threat-sim', 'investigations', 'intelligence', 'analytics'],
    permissions: [
      'ops:view',
      'intel:view',
      'intel:write',
      'analytics:view',
      'simulation:write',
      'case:launch',
      'case:view',
      'case:decide',
      'evidence:package',
      'explain:view',
      'countermeasure:reject',
      'alert:hold',
      'card:hotlist',
    ],
    featureFocus: ['velocity', 'mule_network', 'round_tripping', 'profile_mismatch', 'evidence_package'],
    escalationScope: 'Can decide analyst queue items and package evidence; high-impact freezes need committee approval.',
    shift: '24x7 transaction monitoring',
    reportingLine: 'Fraud Risk Management / Transaction Monitoring Lead',
    decisionAuthority: 'Place immediate holds/hotlists within analyst thresholds and package cases for committee or compliance.',
    toolStack: ['EFRMS/Clari5', 'Finacle CBS', 'Case Management', 'Graph Analytics'],
    workflowSteps: [
      'Review UPI/IMPS/cards/net-banking velocity, beneficiary and device signals.',
      'Use graph evidence to validate mule, layering, pass-through, or round-tripping behavior.',
      'Decide analyst-level cases and escalate high-impact freezes or filings.',
    ],
  },
  transaction_officer: {
    role: 'transaction_officer',
    label: 'Transaction Officer',
    domain: 'Digital Payments Operations',
    summary: 'Executes urgent payment holds, card hotlisting, beneficiary checks, and customer-impact triage.',
    tabs: ['overview', 'investigations', 'analytics', 'compliance'],
    permissions: [
      'ops:view',
      'analytics:view',
      'case:view',
      'case:launch',
      'customer:contact',
      'cfr:check',
      'explain:view',
      'alert:hold',
      'card:hotlist',
    ],
    featureFocus: ['payment_hold', 'card_hotlist', 'beneficiary_status', 'customer_callback'],
    escalationScope: 'Can execute operational holds and hotlisting; cannot tune rules, approve freezes, or file STR/FMR.',
    shift: 'Customer-impact payment desk',
    reportingLine: 'Digital Banking Operations / FRM Desk',
    decisionAuthority: 'Temporarily hold suspicious transactions and hotlist exposed cards within documented thresholds.',
    toolStack: ['Finacle CBS', 'EFRMS/Clari5', 'Card Switch', 'Case Management'],
    workflowSteps: [
      'Check beneficiary registration, recent payee changes, card exposure, and customer confirmation.',
      'Hold suspect payment or hotlist card when immediate customer protection is required.',
      'Hand confirmed fraud evidence to Fraud Analyst and Compliance Officer.',
    ],
  },
  efrms_specialist: {
    role: 'efrms_specialist',
    label: 'EFRMS Specialist',
    domain: 'Fraud Rules and Scenario Tuning',
    summary: 'Tunes EFRMS scenarios, champion-challenger controls, and false-positive feedback loops.',
    tabs: ['pre-fraud-intel', 'overview', 'threat-sim', 'intelligence', 'analytics', 'system'],
    permissions: [
      'ops:view',
      'intel:view',
      'intel:write',
      'analytics:view',
      'simulation:write',
      'explain:view',
      'rules:toggle',
      'model:feedback',
      'system:view',
    ],
    featureFocus: ['scenario_tuning', 'false_positive_feedback', 'threshold_drift', 'champion_challenger'],
    escalationScope: 'Can tune and test rules; cannot decide customer fraud cases or submit regulatory filings.',
    shift: 'Business-hours tuning with 24x7 emergency support',
    reportingLine: 'FRM Rules Lead / Analytics Team',
    decisionAuthority: 'Promote validated rule changes after simulation and committee-backed emergency approval.',
    toolStack: ['EFRMS/Clari5', 'Python Analytics', 'Model Monitor', 'Case Feedback'],
    workflowSteps: [
      'Review false positives, missed frauds, and analyst feedback.',
      'Simulate rule/risk-threshold changes before rollout.',
      'Deploy or rollback EFRMS scenarios with audit evidence.',
    ],
  },
  branch_ops: {
    role: 'branch_ops',
    label: 'Branch Operations',
    domain: 'Branch / Customer Contact',
    summary: 'Reviews customer context, beneficiary registration, and branch-level remediation evidence.',
    tabs: ['overview', 'investigations', 'compliance'],
    permissions: ['ops:view', 'case:view', 'customer:contact', 'cfr:check', 'explain:view'],
    featureFocus: ['kyc_profile', 'beneficiary_prereg', 'branch_mismatch', 'customer_contact'],
    escalationScope: 'Can add customer-contact context and branch observations; cannot file reports or alter controls.',
    shift: 'Branch business hours / customer callbacks',
    reportingLine: 'Branch Manager / Regional Operations',
    decisionAuthority: 'Confirm customer contact, KYC, and branch observations before central case closure.',
    toolStack: ['Finacle CBS', 'KYC/CDD Records', 'Case Management', 'Customer Contact Log'],
    workflowSteps: [
      'Confirm customer identity, branch relationship, and recent beneficiary changes.',
      'Record customer contact outcome and branch-level remediation evidence.',
      'Escalate disputed or high-value cases to Fraud Analyst and Compliance Officer.',
    ],
  },
  aml_analyst: {
    role: 'aml_analyst',
    label: 'AML Analyst',
    domain: 'AML / KYC / Mule Network Monitoring',
    summary: 'Investigates structuring, layering, mule accounts, CDD gaps, and suspicious transaction drafts.',
    tabs: ['overview', 'investigations', 'intelligence', 'analytics', 'compliance'],
    permissions: [
      'ops:view',
      'analytics:view',
      'case:view',
      'evidence:package',
      'cfr:check',
      'explain:view',
      'aml:cdd',
      'aml:str:draft',
    ],
    featureFocus: ['structuring', 'layering', 'mule_network', 'kyc_cdd', 'watchlist_context'],
    escalationScope: 'Can draft AML/STR evidence and CDD notes; Principal Officer or Compliance Officer authorizes filing.',
    shift: 'AML investigation queue',
    reportingLine: 'AML Manager / Principal Officer',
    decisionAuthority: 'Draft suspicious transaction rationale and recommend account restrictions for approval.',
    toolStack: ['AML Suite', 'Watchlists', 'Finacle CBS', 'Graph Analytics', 'Case Management'],
    workflowSteps: [
      'Detect structured deposits, rapid consolidation, and mule-network links.',
      'Review KYC/CDD, occupation, geography, and customer-risk profile.',
      'Draft STR/FIU rationale for Principal Officer authorization.',
    ],
  },
  compliance_officer: {
    role: 'compliance_officer',
    label: 'Compliance Officer',
    domain: 'RBI / FIU-IND / CBI Reporting',
    summary: 'Owns regulatory reporting, fraud registry updates, and FIU-ready evidence review.',
    tabs: ['overview', 'investigations', 'intelligence', 'analytics', 'compliance'],
    permissions: [
      'ops:view',
      'analytics:view',
      'case:view',
      'evidence:package',
      'regulatory:file',
      'cfr:report',
      'cfr:check',
      'fiu:disseminate',
      'explain:view',
      'consortium:publish',
      'aml:str:authorize',
      'fraud:fmr:file',
    ],
    featureFocus: ['rbi_reporting', 'ctr_threshold', 'cfr_match', 'audit_hash', 'fiu_package'],
    escalationScope: 'Can file STR/CTR/FMR/CFR artifacts and coordinate external reporting workflows.',
    shift: 'Regulatory reporting desk',
    reportingLine: 'Principal Officer / Compliance Head',
    decisionAuthority: 'Authorize reportable fraud, STR/CTR/FMR, CFR, DAKSH and external reporting packages.',
    toolStack: ['FIU-IND Portal', 'RBI DAKSH', 'CFR', 'Case Management', 'Audit Ledger'],
    workflowSteps: [
      'Validate analyst evidence against RBI/FIU thresholds and audit-chain completeness.',
      'Authorize STR/CTR/FMR/CFR submissions and dissemination packages.',
      'Coordinate CBI/police/NCRP/1930 workflows when thresholds or incident type require it.',
    ],
  },
  principal_officer: {
    role: 'principal_officer',
    label: 'Principal Officer / MLRO',
    domain: 'AML Compliance Authority',
    summary: 'Owns STR authorization, FIU dissemination, AML governance, and suspicious activity sign-off.',
    tabs: ['overview', 'investigations', 'intelligence', 'analytics', 'compliance'],
    permissions: [
      'ops:view',
      'analytics:view',
      'case:view',
      'evidence:package',
      'regulatory:file',
      'cfr:check',
      'fiu:disseminate',
      'explain:view',
      'consortium:publish',
      'aml:cdd',
      'aml:str:draft',
      'aml:str:authorize',
    ],
    featureFocus: ['str_authorization', 'aml_governance', 'watchlist_match', 'fiu_dissemination'],
    escalationScope: 'Can approve AML filings and FIU dissemination; does not directly operate cyber containment.',
    shift: 'AML sign-off queue',
    reportingLine: 'Chief Compliance Officer / Board Compliance Committee',
    decisionAuthority: 'Authorize STR/FIU submissions and AML governance decisions after analyst review.',
    toolStack: ['AML Suite', 'FIU-IND Portal', 'Watchlists', 'Case Management', 'Audit Ledger'],
    workflowSteps: [
      'Review AML Analyst STR draft, CDD evidence, and graph-risk rationale.',
      'Authorize or reject STR/FIU filing with auditable explanation.',
      'Feed governance outcomes back into AML and EFRMS rule review.',
    ],
  },
  fraud_investigator: {
    role: 'fraud_investigator',
    label: 'Fraud Investigator',
    domain: 'Fraud Investigation Unit',
    summary: 'Builds case evidence, root-cause timelines, common-point-of-compromise links, and LEA-ready packs.',
    tabs: ['overview', 'threat-sim', 'investigations', 'intelligence', 'analytics', 'compliance'],
    permissions: [
      'ops:view',
      'intel:view',
      'analytics:view',
      'simulation:write',
      'case:view',
      'case:launch',
      'case:decide',
      'evidence:package',
      'cfr:check',
      'explain:view',
      'fraud:fmr:file',
    ],
    featureFocus: ['case_timeline', 'common_point_of_compromise', 'lea_package', 'root_cause'],
    escalationScope: 'Can conclude investigation evidence and FMR packs; freeze/rule decisions still need committee authority.',
    shift: 'Investigation queue',
    reportingLine: 'Fraud Investigation Unit Lead',
    decisionAuthority: 'Determine investigation findings and prepare law-enforcement/regulatory evidence packs.',
    toolStack: ['Case Management', 'Graph Analytics', 'Forensics Toolkit', 'CFR', 'Audit Ledger'],
    workflowSteps: [
      'Correlate events, customer complaint, graph pattern, and channel telemetry.',
      'Identify common-point-of-compromise and linked account/entity clusters.',
      'Package findings for FMR/CFR/LEA workflows with audit trace.',
    ],
  },
  fraud_committee: {
    role: 'fraud_committee',
    label: 'Fraud Committee',
    domain: 'Board / Executive Fraud Committee',
    summary: 'Approves high-impact remediation, account freezes, and enterprise policy overrides.',
    tabs: ['pre-fraud-intel', 'overview', 'threat-sim', 'investigations', 'intelligence', 'analytics', 'compliance', 'system'],
    permissions: [
      'ops:view',
      'intel:view',
      'intel:write',
      'analytics:view',
      'simulation:write',
      'case:launch',
      'case:view',
      'case:decide',
      'countermeasure:decide',
      'countermeasure:reject',
      'evidence:package',
      'regulatory:file',
      'cfr:report',
      'cfr:check',
      'fiu:disseminate',
      'rules:toggle',
      'circuit_breaker:trigger',
      'explain:view',
      'consortium:publish',
      'system:view',
      'alert:hold',
      'card:hotlist',
      'aml:cdd',
      'aml:str:draft',
      'aml:str:authorize',
      'soc:isolate',
      'soc:monitor',
      'threat:intel:write',
      'model:feedback',
      'audit:review',
      'risk:view',
      'fraud:fmr:file',
    ],
    featureFocus: ['enterprise_risk', 'committee_threshold', 'freeze_authority', 'model_feedback'],
    escalationScope: 'Can authorize freezes, rule changes, reporting decisions, and remediation execution.',
    shift: 'Executive approval board',
    reportingLine: 'Board-level Risk/Fraud Governance',
    decisionAuthority: 'Final authority for high-impact freezes, rule overrides, regulatory packages, and cross-team remediation.',
    toolStack: ['Committee Docket', 'EFRMS/Clari5', 'Case Management', 'RBI/FIU Evidence', 'Audit Ledger'],
    workflowSteps: [
      'Review analyst, AML, compliance, cyber, and branch evidence together.',
      'Approve or reject high-impact freezes, emergency rules, and external reporting.',
      'Mandate feedback into EFRMS, SOC playbooks, branch controls, and model governance.',
    ],
  },
  risk_analyst: {
    role: 'risk_analyst',
    label: 'Risk Analyst',
    domain: 'Enterprise Risk Management',
    summary: 'Tracks fraud exposure, thresholds, scenario loss, and portfolio-level control effectiveness.',
    tabs: ['overview', 'analytics', 'compliance', 'system'],
    permissions: ['ops:view', 'analytics:view', 'case:view', 'cfr:check', 'explain:view', 'risk:view', 'audit:review'],
    featureFocus: ['portfolio_loss', 'risk_threshold', 'control_effectiveness', 'scenario_exposure'],
    escalationScope: 'Read-mostly risk oversight; cannot decide cases, freeze accounts, or file reports.',
    shift: 'Risk review cycles',
    reportingLine: 'CRO / Enterprise Risk Committee',
    decisionAuthority: 'Recommend risk appetite and threshold changes for committee review.',
    toolStack: ['Risk Dashboard', 'Analytics Warehouse', 'Case Management', 'Audit Ledger'],
    workflowSteps: [
      'Track exposure by channel, branch, product, and fraud typology.',
      'Assess false positives, control gaps, and residual risk.',
      'Recommend portfolio controls to committee and EFRMS specialists.',
    ],
  },
  data_scientist: {
    role: 'data_scientist',
    label: 'Data Scientist / ML Engineer',
    domain: 'AI / Model Governance',
    summary: 'Monitors model drift, feature quality, false-positive loops, and Qwen reasoning telemetry.',
    tabs: ['overview', 'intelligence', 'analytics', 'system'],
    permissions: ['ops:view', 'analytics:view', 'intel:view', 'system:view', 'explain:view', 'model:feedback', 'audit:review'],
    featureFocus: ['model_drift', 'feature_quality', 'xai_reasoning', 'feedback_loop'],
    escalationScope: 'Can review and annotate model behavior; cannot operate fraud controls or file external reports.',
    shift: 'Model monitoring cycles',
    reportingLine: 'Analytics Head / Model Risk Governance',
    decisionAuthority: 'Recommend model/rule recalibration based on measured drift and analyst feedback.',
    toolStack: ['Python Analytics', 'Model Monitor', 'Qwen Runtime Telemetry', 'Feature Store', 'Audit Ledger'],
    workflowSteps: [
      'Review drift, feature attribution, and model-derived fraud explanations.',
      'Compare analyst outcomes against model scores and Qwen reasoning traces.',
      'Feed approved changes to EFRMS Specialist and Fraud Committee.',
    ],
  },
  internal_audit: {
    role: 'internal_audit',
    label: 'Internal Audit',
    domain: 'Independent Audit and Control Testing',
    summary: 'Reviews case actions, permission usage, evidence integrity, and regulatory control adherence.',
    tabs: ['overview', 'investigations', 'analytics', 'compliance', 'system'],
    permissions: ['ops:view', 'analytics:view', 'case:view', 'cfr:check', 'explain:view', 'audit:review', 'system:view'],
    featureFocus: ['audit_hash', 'segregation_of_duties', 'control_testing', 'evidence_integrity'],
    escalationScope: 'Read-only audit oversight; no operational write, freeze, rule, or filing authority.',
    shift: 'Periodic and event-triggered audit',
    reportingLine: 'Internal Audit / Audit Committee',
    decisionAuthority: 'Raise audit findings and control exceptions without changing operational state.',
    toolStack: ['Audit Ledger', 'Case Management', 'RBI/FIU Evidence', 'Access Logs'],
    workflowSteps: [
      'Verify role segregation, action approvals, and evidence-chain hashes.',
      'Test whether cases met RBI/FIU/CFR reporting obligations.',
      'Raise audit exceptions for governance remediation.',
    ],
  },
  system_admin: {
    role: 'system_admin',
    label: 'System Administrator',
    domain: 'Platform Operations',
    summary: 'Maintains runtime health, model/service telemetry, and platform controls without case authority.',
    tabs: ['overview', 'analytics', 'intelligence', 'system'],
    permissions: ['ops:view', 'analytics:view', 'system:view', 'rules:toggle', 'explain:view'],
    featureFocus: ['runtime_health', 'model_drift', 'pipeline_latency', 'service_integrity'],
    escalationScope: 'Can tune platform controls; cannot approve fraud decisions or file external reports.',
    shift: 'Platform support',
    reportingLine: 'Technology Operations',
    decisionAuthority: 'Maintain service health, deployments, and runtime controls without case authority.',
    toolStack: ['FastAPI', 'SSE', 'Ollama/Qwen Runtime', 'Nixpacks', 'Observability Logs'],
    workflowSteps: [
      'Monitor API, SSE, model runtime, and pipeline health.',
      'Apply platform configuration changes with no direct fraud-case authority.',
      'Escalate model or data-quality issues to Data Scientist and EFRMS Specialist.',
    ],
  },
}

export const OPERATIONAL_WORKFLOWS: OperationalWorkflow[] = [
  {
    id: 'upi_remote_access_scam',
    title: 'UPI remote-access scam',
    trigger: 'Fast UPI velocity, new device/MFA context, remote-support or screen-share signal.',
    stages: [
      { owner: 'soc_analyst', action: 'Detect device/session anomaly and live EFRMS alert.' },
      { owner: 'fraud_analyst', action: 'Place analyst hold, inspect beneficiary velocity and mule graph.' },
      { owner: 'transaction_officer', action: 'Execute payment hold or card hotlist where customer exposure is active.' },
      { owner: 'branch_ops', action: 'Confirm customer contact, KYC, and complaint timeline.' },
      { owner: 'compliance_officer', action: 'Prepare 1930/NCRP/DAKSH/FMR package when thresholds are met.' },
    ],
  },
  {
    id: 'phishing_malware_intrusion',
    title: 'Phishing or malware-linked banking intrusion',
    trigger: 'Suspicious PowerShell, endpoint anomaly, malicious APK, credential theft, or lateral movement.',
    stages: [
      { owner: 'threat_hunter', action: 'Verify phishing domain, APK, IOC and external campaign context.' },
      { owner: 'soc_l2_incident_responder', action: 'Isolate workstation/device session and preserve forensic evidence.' },
      { owner: 'fraud_analyst', action: 'Connect cyber timeline to money movement and impacted accounts.' },
      { owner: 'efrms_specialist', action: 'Tune digital-channel scenarios after validated IOC/playbook.' },
      { owner: 'fraud_committee', action: 'Approve high-impact containment or emergency rule rollout.' },
    ],
  },
  {
    id: 'mule_layering_network',
    title: 'Mule account and layering network',
    trigger: 'Structured deposits, rapid consolidation, pass-through behavior, dormant activation, or CFR match.',
    stages: [
      { owner: 'aml_analyst', action: 'Perform CDD/KYC review and draft STR rationale.' },
      { owner: 'fraud_investigator', action: 'Build graph evidence and connected-account investigation package.' },
      { owner: 'principal_officer', action: 'Authorize STR/FIU dissemination after AML review.' },
      { owner: 'compliance_officer', action: 'File STR/CTR/FMR/CFR artifacts with audit hash.' },
      { owner: 'risk_analyst', action: 'Review portfolio exposure and recommend control tightening.' },
    ],
  },
  {
    id: 'atm_card_cloning',
    title: 'ATM skimming or card cloning',
    trigger: 'Impossible travel, repeated card-present failures, common ATM point, or cross-border usage.',
    stages: [
      { owner: 'soc_analyst', action: 'Monitor impossible-travel and device/payment-rail anomalies.' },
      { owner: 'transaction_officer', action: 'Hotlist exposed card and protect customer funds.' },
      { owner: 'fraud_investigator', action: 'Correlate common point of compromise and linked victims.' },
      { owner: 'compliance_officer', action: 'Prepare FMR/CFR and external reporting artifacts.' },
      { owner: 'internal_audit', action: 'Validate evidence chain, approvals, and segregation of duties.' },
    ],
  },
]

export function rolePolicy(role: PayflowRole | string | null | undefined): RolePolicy {
  return ROLE_POLICIES[(role as PayflowRole) in ROLE_POLICIES ? role as PayflowRole : DEFAULT_PAYFLOW_ROLE]
}

export function hasPermission(role: PayflowRole, permission: Permission): boolean {
  return rolePolicy(role).permissions.includes(permission)
}

export function canAccessTab(role: PayflowRole, tab: string): boolean {
  return rolePolicy(role).tabs.includes(tab)
}

export function defaultTabForRole(role: PayflowRole): string {
  return rolePolicy(role).tabs[0] ?? 'overview'
}

export function getStoredRole(): PayflowRole {
  if (typeof window === 'undefined') return DEFAULT_PAYFLOW_ROLE
  const stored = window.localStorage.getItem(ROLE_STORAGE_KEY)
  return rolePolicy(stored).role
}

export function storeRole(role: PayflowRole) {
  if (typeof window !== 'undefined') {
    window.localStorage.setItem(ROLE_STORAGE_KEY, role)
  }
}

export function roleRequestHeaders(): Record<string, string> {
  return { 'X-Payflow-Role': getStoredRole() }
}

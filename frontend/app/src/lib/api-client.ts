// ============================================================================
// API Client -- Typed fetch wrappers for all PayFlow REST endpoints
// ============================================================================

import type {
  SystemSnapshot,
  GraphTopology,
  CircuitBreakerStatus,
  AgentVerdictsResponse,
  AttackTypesResponse,
  LaunchRequest,
  LaunchResponse,
  PS3ScenariosResponse,
  PS3LaunchRequest,
  PS3LaunchResponse,
  StopResponse,
  StopAllResponse,
  ScenarioStatus,
  ActiveScenariosResponse,
  HistoryResponse,
  Escalation,
  RecentBlocksResponse,
  EnumsResponse,
  InjectEventRequest,
  InjectEventResponse,
  EventLabRequest,
  EventLabRunRequest,
  EventLabTemplatesResponse,
  EventLabPreviewResponse,
  EventLabRunResponse,
  EventLabExplainabilityResponse,
  CountermeasureProposal,
  CountermeasureProposalsResponse,
  InvestigationRecord,
  RiskDistributionResponse,
  FraudTypologyResponse,
  VelocityTrendsResponse,
  TemporalHeatmapResponse,
  ThreatSummaryResponse,
  ExplainResponse,
  GlobalImportanceResponse,
  DriftResponse,
  NLQueryResponse,
  ConsortiumStatusResponse,
  ConsortiumAlertsResponse,
  ConsortiumPublishResponse,
  ConsortiumCheckResponse,
  // Fraud Intelligence types
  RulesListResponse,
  RuleStatsResponse,
  ReportsListResponse,
  GateStatsResponse,
  CFRStatsResponse,
  AMLStatsResponse,
  FIUStatsResponse,
  FIUHighRiskResponse,
  InvestigationStatsResponse,
  CaseTraceResponse,
  EvidencePackageResponse,
  PS3ReadinessResponse,
  IntelSourcesResponse,
  IntelSignalsResponse,
  IntelTrendsResponse,
  IntelPlaybooksResponse,
  IntelTuningStatus,
  IntelRefreshResponse,
  IntelSimulateResponse,
  IntelCockpitResponse,
  IntelMediaResponse,
  MuleChainsResponse,
  MuleStatsResponse,
  SuspectedMulesResponse,
  VictimStatsResponse,
  AnomalyStatsResponse,
  ClustersResponse,
  IntermediariesResponse,
} from './types'

class ApiError extends Error {
  status: number
  constructor(status: number, message: string) {
    super(message)
    this.name = 'ApiError'
    this.status = status
  }
}

async function fetchJson<T>(url: string, init?: RequestInit): Promise<T> {
  const res = await fetch(url, init)
  if (!res.ok) {
    const text = await res.text().catch(() => res.statusText)
    throw new ApiError(res.status, text)
  }
  return res.json() as Promise<T>
}

// -- Dashboard endpoints --

export function fetchSnapshot(): Promise<SystemSnapshot> {
  return fetchJson('/api/v1/snapshot')
}

export function fetchTopology(limit = 500): Promise<GraphTopology> {
  return fetchJson(`/api/v1/graph/topology?limit=${limit}`)
}

export function fetchCircuitBreakerStatus(): Promise<CircuitBreakerStatus> {
  return fetchJson('/api/v1/circuit-breaker/status')
}

export function fetchVerdicts(limit = 20): Promise<AgentVerdictsResponse> {
  return fetchJson(`/api/v1/agent/verdicts?limit=${limit}`)
}

// -- Simulation endpoints --

export function fetchAttackTypes(): Promise<AttackTypesResponse> {
  return fetchJson('/api/v1/simulation/attacks')
}

export function fetchPS3Scenarios(): Promise<PS3ScenariosResponse> {
  return fetchJson('/api/v1/simulation/ps3/scenarios')
}

export function launchPS3Scenario(body: PS3LaunchRequest): Promise<PS3LaunchResponse> {
  return fetchJson('/api/v1/simulation/ps3/launch', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
}

export function launchAttack(body: LaunchRequest): Promise<LaunchResponse> {
  return fetchJson('/api/v1/simulation/launch', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
}

export function stopAttack(scenarioId: string): Promise<StopResponse> {
  return fetchJson(`/api/v1/simulation/stop/${scenarioId}`, { method: 'POST' })
}

export function stopAllAttacks(): Promise<StopAllResponse> {
  return fetchJson('/api/v1/simulation/stop-all', { method: 'POST' })
}

export function fetchScenarioStatus(scenarioId: string): Promise<ScenarioStatus> {
  return fetchJson(`/api/v1/simulation/status/${scenarioId}`)
}

export function fetchActiveScenarios(): Promise<ActiveScenariosResponse> {
  return fetchJson('/api/v1/simulation/active')
}

export function fetchHistory(): Promise<HistoryResponse> {
  return fetchJson('/api/v1/simulation/history')
}

// -- Analyst endpoints --

export function fetchEscalations(): Promise<Escalation[]> {
  return fetchJson('/api/v1/analyst/escalations')
}

// -- Enum / Custom Event endpoints --

export function fetchEnums(): Promise<EnumsResponse> {
  return fetchJson('/api/v1/simulation/enums')
}

export function injectEvent(body: InjectEventRequest): Promise<InjectEventResponse> {
  return fetchJson('/api/v1/simulation/inject-event', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
}

export function fetchEventLabTemplates(): Promise<EventLabTemplatesResponse> {
  return fetchJson('/api/v1/simulation/event-lab/templates')
}

export function previewEventLabRun(body: EventLabRequest): Promise<EventLabPreviewResponse> {
  return fetchJson('/api/v1/simulation/event-lab/preview', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
}

export function createEventLabRun(body: EventLabRunRequest): Promise<EventLabRunResponse> {
  return fetchJson('/api/v1/simulation/event-lab/runs', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
}

export function fetchEventLabRun(runId: string): Promise<EventLabRunResponse> {
  return fetchJson(`/api/v1/simulation/event-lab/runs/${runId}`)
}

export function fetchEventLabExplainability(runId: string): Promise<EventLabExplainabilityResponse> {
  return fetchJson(`/api/v1/simulation/event-lab/runs/${runId}/explainability`)
}

export function fetchCountermeasureProposals(runId?: string, status?: string): Promise<CountermeasureProposalsResponse> {
  const params = new URLSearchParams()
  if (runId) params.set('run_id', runId)
  if (status) params.set('status', status)
  const suffix = params.toString() ? `?${params}` : ''
  return fetchJson(`/api/v1/countermeasures/proposals${suffix}`)
}

export function approveCountermeasure(
  proposalId: string,
  body: { analyst?: string; reason?: string } = {},
): Promise<CountermeasureProposal> {
  return fetchJson(`/api/v1/countermeasures/proposals/${proposalId}/approve`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      analyst: body.analyst ?? 'union_bank_analyst',
      reason: body.reason ?? 'analyst_approved_from_event_lab',
    }),
  })
}

export function rejectCountermeasure(
  proposalId: string,
  body: { analyst?: string; reason?: string } = {},
): Promise<CountermeasureProposal> {
  return fetchJson(`/api/v1/countermeasures/proposals/${proposalId}/reject`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      analyst: body.analyst ?? 'union_bank_analyst',
      reason: body.reason ?? 'analyst_rejected_from_event_lab',
    }),
  })
}

// -- Investigation --

export function fetchInvestigation(txnId: string): Promise<InvestigationRecord> {
  return fetchJson(`/api/v1/agent/investigation/${txnId}`)
}

// -- Blockchain endpoints --

export function fetchRecentBlocks(limit = 50): Promise<RecentBlocksResponse> {
  return fetchJson(`/api/v1/blockchain/recent-blocks?limit=${limit}`)
}

// -- Analytics endpoints --

export function fetchRiskDistribution(): Promise<RiskDistributionResponse> {
  return fetchJson('/api/v1/analytics/risk-distribution')
}

export function fetchFraudTypology(): Promise<FraudTypologyResponse> {
  return fetchJson('/api/v1/analytics/fraud-typology')
}

export function fetchVelocityTrends(windowMinutes = 30, topN = 10): Promise<VelocityTrendsResponse> {
  return fetchJson(`/api/v1/analytics/velocity-trends?window_minutes=${windowMinutes}&top_n=${topN}`)
}

export function fetchTemporalHeatmap(bucketSeconds = 60, lookbackMinutes = 30): Promise<TemporalHeatmapResponse> {
  return fetchJson(`/api/v1/analytics/temporal-heatmap?bucket_seconds=${bucketSeconds}&lookback_minutes=${lookbackMinutes}`)
}

export function fetchThreatSummary(): Promise<ThreatSummaryResponse> {
  return fetchJson('/api/v1/analytics/threat-summary')
}

// -- Intelligence endpoints --

export function fetchExplanation(features: number[], txnId = 'unknown'): Promise<ExplainResponse> {
  return fetchJson('/api/v1/intelligence/explain', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ features, txn_id: txnId }),
  })
}

export function fetchGlobalImportance(): Promise<GlobalImportanceResponse> {
  return fetchJson('/api/v1/intelligence/explainability/global')
}

export function fetchDriftStatus(): Promise<DriftResponse> {
  return fetchJson('/api/v1/intelligence/drift')
}

export function fetchNLQuery(question: string): Promise<NLQueryResponse> {
  return fetchJson('/api/v1/intelligence/query', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ question }),
  })
}

export function fetchConsortiumStatus(): Promise<ConsortiumStatusResponse> {
  return fetchJson('/api/v1/intelligence/consortium')
}

export function fetchConsortiumAlerts(
  fraudType?: number,
  severityMin = 1,
  limit = 50,
): Promise<ConsortiumAlertsResponse> {
  const params = new URLSearchParams()
  if (fraudType != null) params.set('fraud_type', String(fraudType))
  params.set('severity_min', String(severityMin))
  params.set('limit', String(limit))
  return fetchJson(`/api/v1/intelligence/consortium/alerts?${params}`)
}

export function publishConsortiumAlert(body: {
  account_id: string
  risk_score: number
  fraud_type: number
  severity: number
}): Promise<ConsortiumPublishResponse> {
  return fetchJson('/api/v1/intelligence/consortium/publish', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
}

export function checkConsortiumAccount(accountId: string): Promise<ConsortiumCheckResponse> {
  return fetchJson('/api/v1/intelligence/consortium/check', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ account_id: accountId }),
  })
}

// -- Fraud Intelligence endpoints --

// Rule Engine
export function fetchRules(): Promise<RulesListResponse> {
  return fetchJson('/api/v1/fraud/rules')
}

export function fetchRuleStats(): Promise<RuleStatsResponse> {
  return fetchJson('/api/v1/fraud/rules/stats')
}

export function toggleRule(ruleId: string, enabled: boolean): Promise<{ rule_id: string; enabled: boolean }> {
  return fetchJson(`/api/v1/fraud/rules/${ruleId}/toggle`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ enabled }),
  })
}

// Pre-Approval Gate
export function fetchGateStats(): Promise<GateStatsResponse> {
  return fetchJson('/api/v1/fraud/gate/stats')
}

// Regulatory Reports
export function fetchReports(reportType?: string, limit = 50): Promise<ReportsListResponse> {
  const params = new URLSearchParams()
  if (reportType) params.set('report_type', reportType)
  params.set('limit', String(limit))
  return fetchJson(`/api/v1/fraud/reports?${params}`)
}

// CFR-RBI
export function fetchCFRStats(): Promise<CFRStatsResponse> {
  return fetchJson('/api/v1/fraud/cfr/stats')
}

// AML
export function fetchAMLStats(): Promise<AMLStatsResponse> {
  return fetchJson('/api/v1/fraud/aml/stats')
}

// FIU Intelligence
export function fetchFIUStats(): Promise<FIUStatsResponse> {
  return fetchJson('/api/v1/fraud/fiu/stats')
}

export function fetchFIUHighRisk(): Promise<FIUHighRiskResponse> {
  return fetchJson('/api/v1/fraud/fiu/high-risk')
}

// Investigation
export function fetchInvestigationStats(): Promise<InvestigationStatsResponse> {
  return fetchJson('/api/v1/fraud/investigation/stats')
}

export function fetchCaseTrace(caseId: string): Promise<CaseTraceResponse> {
  return fetchJson(`/api/v1/fraud/investigation/case/${caseId}/trace`)
}

export function createEvidencePackage(caseId: string): Promise<EvidencePackageResponse> {
  return fetchJson(`/api/v1/fraud/investigation/case/${caseId}/evidence-package`, {
    method: 'POST',
  })
}

export function fetchPS3Readiness(): Promise<PS3ReadinessResponse> {
  return fetchJson('/api/v1/readiness/ps3')
}

// -- Pre-Fraud Intelligence endpoints --

export function fetchIntelSources(): Promise<IntelSourcesResponse> {
  return fetchJson('/api/v1/intel/sources')
}

export function refreshIntel(seed?: number): Promise<IntelRefreshResponse> {
  return fetchJson('/api/v1/intel/refresh', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(seed == null ? {} : { seed }),
  })
}

export function fetchIntelSignals(params?: {
  typology?: string
  region?: string
  source_tier?: string
  min_trust?: number
  since?: number
}): Promise<IntelSignalsResponse> {
  const qs = new URLSearchParams()
  if (params?.typology) qs.set('typology', params.typology)
  if (params?.region) qs.set('region', params.region)
  if (params?.source_tier) qs.set('source_tier', params.source_tier)
  if (params?.min_trust != null) qs.set('min_trust', String(params.min_trust))
  if (params?.since != null) qs.set('since', String(params.since))
  const suffix = qs.toString() ? `?${qs.toString()}` : ''
  return fetchJson(`/api/v1/intel/signals${suffix}`)
}

export function fetchIntelTrends(): Promise<IntelTrendsResponse> {
  return fetchJson('/api/v1/intel/trends')
}

export function fetchIntelPlaybooks(): Promise<IntelPlaybooksResponse> {
  return fetchJson('/api/v1/intel/playbooks')
}

export function fetchIntelCockpit(): Promise<IntelCockpitResponse> {
  return fetchJson('/api/v1/intel/cockpit')
}

export function fetchIntelMedia(): Promise<IntelMediaResponse> {
  return fetchJson('/api/v1/intel/media')
}

export function fetchIntelTuningStatus(): Promise<IntelTuningStatus> {
  return fetchJson('/api/v1/intel/tuning/status')
}

export function simulateIntelSignal(scenario = 'digital_arrest_mule'): Promise<IntelSimulateResponse> {
  return fetchJson('/api/v1/intel/simulate-signal', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ scenario }),
  })
}

// Mule Detection
export function fetchMuleChains(): Promise<MuleChainsResponse> {
  return fetchJson('/api/v1/fraud/mule/chains')
}

export function fetchMuleStats(): Promise<MuleStatsResponse> {
  return fetchJson('/api/v1/fraud/mule/stats')
}

export function fetchSuspectedMules(threshold = 0.6): Promise<SuspectedMulesResponse> {
  return fetchJson(`/api/v1/fraud/mule/suspected?threshold=${threshold}`)
}

// Victim Fund Tracing
export function fetchVictimStats(): Promise<VictimStatsResponse> {
  return fetchJson('/api/v1/fraud/victim/stats')
}

// Anomaly Detection
export function fetchAnomalyStats(): Promise<AnomalyStatsResponse> {
  return fetchJson('/api/v1/fraud/anomaly/stats')
}

// Community Clusters
export function fetchClusters(): Promise<ClustersResponse> {
  return fetchJson('/api/v1/fraud/clusters')
}

// Centrality Analysis
export function fetchIntermediaries(topN = 20): Promise<IntermediariesResponse> {
  return fetchJson(`/api/v1/fraud/centrality/intermediaries?top_n=${topN}`)
}

export { ApiError }

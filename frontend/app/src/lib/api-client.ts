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

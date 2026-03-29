// ============================================================================
// PayFlow TypeScript Type Definitions
// Mirrors all API response shapes from the Python FastAPI backend
// ============================================================================

// === System Snapshot (GET /api/v1/snapshot) ===

export interface OrchestratorMetrics {
  events_ingested: number
  features_extracted: number
  ml_inferences: number
  alerts_routed: number
  elapsed_sec: number
  events_per_sec: number
}

export interface HardwareSnapshot {
  gpu_vram_used_mb: number
  gpu_vram_total_mb: number
  gpu_vram_free_mb: number
  gpu_utilization_pct: number
  cpu_utilization_pct: number
  llm_tps: number
  llm_tokens_total: number
  load_shed_active: boolean
}

export interface GraphMetrics {
  mule_detections: number
  cycle_detections: number
}

export interface AgentMetrics {
  completed: number
  verdicts: {
    fraudulent: number
    suspicious: number
    legitimate: number
    escalated: number
  }
  agent_breaker_triggered: number
}

export interface SystemSnapshot {
  orchestrator: OrchestratorMetrics
  hardware: HardwareSnapshot
  load_shedder?: Record<string, unknown>
  graph?: {
    metrics: GraphMetrics
    graph: { nodes: number; edges: number }
  }
  circuit_breaker?: Record<string, unknown>
  agent?: { metrics: AgentMetrics }
  threshold?: Record<string, unknown>
  gpu_concurrency?: Record<string, unknown>
  threat_simulation?: ThreatSimulationSnapshot
}

// === Graph Topology (GET /api/v1/graph/topology) ===

export interface CytoNode {
  data: {
    id: string
    txn_count: number
    status: 'normal' | 'frozen' | 'paused' | 'suspicious'
    account_type?: string
    community_id?: number
    fraud_edge_count?: number
    total_volume_paisa?: number
    first_seen?: number
    last_seen?: number
  }
}

export interface CytoEdge {
  data: {
    id: string
    source: string
    target: string
    amount_paisa: number
    channel: number
    fraud_label: number
    fraud_label_name?: string
    timestamp: number
    device_fingerprint: string
  }
}

export interface GraphTopology {
  nodes: CytoNode[]
  edges: CytoEdge[]
}

// === Circuit Breaker (GET /api/v1/circuit-breaker/status) ===

export interface FreezeOrder {
  node_id: string
  consensus_score: number
  reason: string
  freeze_timestamp?: number
  trigger_txn_id?: string
  ml_risk_score?: number
  gnn_risk_score?: number
  graph_evidence_score?: number
  ttl_seconds?: number
}

export interface CircuitBreakerStatus {
  freeze_orders: FreezeOrder[]
  snapshot: {
    frozen_count?: number
    pending_alerts?: number
    metrics?: { consensus_triggers?: number }
    config?: { consensus_threshold?: number }
  }
  agent_breaker: {
    banned_devices?: number
    routing_paused_nodes?: number
  }
}

// === Agent Verdicts (GET /api/v1/agent/verdicts) ===

export interface VerdictPayload {
  verdict: string
  confidence: number
  txn_id: string
  node_id?: string
  recommended_action: string
  fraud_typology?: string
  reasoning_summary?: string
  evidence_cited?: string[]
  thinking_steps?: number
  tools_used?: string[]
  total_duration_ms?: number
}

export interface VerdictBlock {
  index?: number
  timestamp?: number
  event_type?: number
  payload?: VerdictPayload
  previous_hash?: string
  block_hash?: string
}

export interface AgentVerdictsResponse {
  verdicts: VerdictBlock[]
}

// === Simulation (POST/GET /api/v1/simulation/*) ===

export interface LaunchRequest {
  attack_type: string
  params?: Record<string, unknown>
}

export interface LaunchResponse {
  scenario_id: string
  attack_type: string
  attack_label: string
  status: string
  events_generated: number
  message: string
}

export interface StopResponse {
  scenario_id: string
  stopped: boolean
  message: string
}

export interface StopAllResponse {
  stopped_count: number
  message: string
}

export type ScenarioStatusValue = 'running' | 'completed' | 'stopped' | 'error'

export interface ScenarioStatus {
  scenario_id: string
  attack_type: string
  attack_label: string
  status: ScenarioStatusValue
  events_generated: number
  events_ingested: number
  progress_pct: number
  accounts_involved: string[]
  started_at: number
  stopped_at: number | null
  elapsed_sec: number
}

export interface ThreatSimulationSnapshot {
  active_attacks: number
  total_attacks: number
  scenarios: ScenarioStatus[]
}

export interface AttackParamSchema {
  type: string
  default: number
  min: number
  max: number
  step: number
  label: string
  description: string
}

export interface AttackTypeDetail {
  label: string
  description: string
  phases: string[]
  params: Record<string, AttackParamSchema>
}

export interface AttackTypesResponse {
  attacks: Record<string, AttackTypeDetail>
}

// === Custom Event Injection (POST /api/v1/simulation/inject) ===

export interface InjectEventRequest {
  event_type: 'transaction' | 'auth' | 'interbank'
  sender_id?: string
  receiver_id?: string
  amount_inr?: number
  channel?: string
  fraud_label?: number
  sender_account_type?: string
  receiver_account_type?: string
  account_id?: string
  action?: string
  ip_address?: string
  success?: boolean
  sender_ifsc?: string
  receiver_ifsc?: string
  message_type?: string
  currency_code?: number
  priority?: number
  device_fingerprint?: string
  geo_lat?: number
  geo_lon?: number
}

export interface InjectEventResponse {
  status: string
  event: Record<string, unknown>
  timestamp: number
}

// === Enums (GET /api/v1/simulation/enums) ===

export interface EnumValue {
  value: number
  name: string
}

export interface EnumsResponse {
  channels: EnumValue[]
  account_types: EnumValue[]
  auth_actions: EnumValue[]
  fraud_patterns: EnumValue[]
  message_types: string[]
}

export interface ActiveScenariosResponse {
  scenarios: ScenarioStatus[]
}

export interface HistoryResponse {
  scenarios: ScenarioStatus[]
}

// === HITL Escalations (GET /api/v1/analyst/escalations) ===

export interface Escalation {
  ack_id: string
  payload: Record<string, unknown>
}

// === SSE Event Types ===

export type SSEChannel =
  | 'graph'
  | 'agent'
  | 'circuit_breaker'
  | 'risk_scores'
  | 'system'
  | 'simulation'
  | 'pipeline'

export interface SSEEnvelope {
  channel: SSEChannel
  timestamp: number
  data: Record<string, unknown>
}

// -- Graph channel --
export interface SSEGraphBatchUpdate {
  type: 'batch_update'
  nodes: CytoNode[]
  edges: CytoEdge[]
}

export interface SSEGraphNodeStatusChanged {
  type: 'node_status_changed'
  node_id: string
  status: string
}

export type SSEGraphData = SSEGraphBatchUpdate | SSEGraphNodeStatusChanged

// -- Agent channel --
export interface SSEAgentThinking {
  type: 'thinking_step'
  txn_id: string
  iteration: number
  max_iterations?: number
  content: string
  elapsed_ms?: number
}

export interface SSEAgentToolCall {
  type: 'tool_call'
  txn_id: string
  iteration?: number
  tool_name: string
  success: boolean
  duration_ms: number
  output_summary?: string
}

export interface SSEAgentVerdict {
  type: 'verdict'
  txn_id: string
  node_id: string
  verdict: string
  confidence: number
  fraud_typology: string
  reasoning_summary: string
  evidence_cited?: string[]
  recommended_action: string
  thinking_steps: number
  tools_used: string[]
  total_duration_ms: number
  nlu_findings_count?: number
  nlu_escalated?: boolean
}

export type SSEAgentData = SSEAgentThinking | SSEAgentToolCall | SSEAgentVerdict

// -- Circuit Breaker channel --
export interface SSECircuitBreakerEvent {
  type: 'node_frozen' | 'node_unfrozen'
  node_id: string
  order?: Record<string, unknown>
  reason?: string
}

// -- Risk Scores channel --
export interface SSERiskScoreAlert {
  type: 'alert_scored'
  txn_id: string
  risk_score: number
  tier: 'critical' | 'high' | 'medium' | 'low'
  top_features: string[]
}

// -- System channel --
export interface SSESystemTelemetry {
  type: 'telemetry'
  orchestrator: OrchestratorMetrics
  hardware: HardwareSnapshot
  agent?: Record<string, unknown>
  graph?: Record<string, unknown>
  circuit_breaker?: Record<string, unknown>
  threat_simulation?: ThreatSimulationSnapshot
}

export interface SSEGpuPressure {
  type: 'gpu_pressure'
  num_ctx: number
  pressure: string
}

export type SSESystemData = SSESystemTelemetry | SSEGpuPressure

// -- Simulation channel --
export interface SSESimulationAttackEvent {
  type: 'attack_event'
  scenario_id: string
  attack_type: string
  event_index: number
  total_events: number
  progress_pct: number
  event: {
    type: 'transaction' | 'interbank' | 'auth'
    txn_id?: string
    msg_id?: string
    event_id?: string
    sender?: string
    receiver?: string
    amount_paisa?: number
    channel?: string
    fraud_label?: string
    sender_ifsc?: string
    receiver_ifsc?: string
    account?: string
    action?: string
    success?: boolean
    ip?: string
  }
}

export interface SimulationRecentEvent {
  id: string
  scenarioId: string
  attackType: string
  attackLabel: string
  eventType: 'transaction' | 'interbank' | 'auth' | 'unknown'
  progressPct: number
  timestamp: number
  event: SSESimulationAttackEvent['event']
}

export interface SSESimulationLifecycle {
  type: 'simulation_started' | 'simulation_completed' | 'simulation_stopped'
  scenario_id: string
  attack_type: string
  attack_label: string
  events_generated: number
  events_ingested: number
  accounts_involved: string[]
}

export type SSESimulationData = SSESimulationAttackEvent | SSESimulationLifecycle

// -- Agent CoT log entry (for Zustand store) --
export interface AgentLogEntry {
  id: string
  timestamp: number
  type: 'thinking' | 'tool_call' | 'verdict'
  txn_id: string
  data: SSEAgentThinking | SSEAgentToolCall | SSEAgentVerdict
}

// === Fraud Pattern Constants ===
// Uses const object (not enum) due to erasableSyntaxOnly: true

export const FraudPattern = {
  NONE: 0,
  LAYERING: 1,
  ROUND_TRIPPING: 2,
  STRUCTURING: 3,
  DORMANT_ACTIVATION: 4,
  PROFILE_MISMATCH: 5,
  UPI_MULE_NETWORK: 6,
  CIRCULAR_LAUNDERING: 7,
  VELOCITY_PHISHING: 8,
} as const

export type FraudPatternValue = (typeof FraudPattern)[keyof typeof FraudPattern]

export const FRAUD_PATTERN_LABELS: Record<number, string> = {
  0: 'None',
  1: 'Layering',
  2: 'Round-Tripping',
  3: 'Structuring',
  4: 'Dormant Activation',
  5: 'Profile Mismatch',
  6: 'UPI Mule Network',
  7: 'Circular Laundering',
  8: 'Velocity Phishing',
}

// === Pipeline SSE Events ===

export interface SSEPipelineConsumerResult {
  consumer: string
  success: boolean
  duration_ms: number
  error?: string
}

export interface SSEPipelineBatchDispatched {
  type: 'batch_dispatched'
  batch_id: number
  event_count: number
  transactions: number
  auth_events: number
  interbank_messages: number
  consumers: SSEPipelineConsumerResult[]
}

export interface SSEPipelineStageComplete {
  type: 'stage_complete'
  stage: string
  txn_id?: string
  txn_ids?: string[]
  duration_ms?: number
  [key: string]: unknown
}

export type SSEPipelineData = SSEPipelineBatchDispatched | SSEPipelineStageComplete

// === Enum Types (from GET /api/v1/simulation/enums) ===
// (EnumValue and EnumsResponse defined above in the Simulation section)

// === Investigation Record ===

export interface InvestigationRecord {
  txn_id: string
  node_id: string
  verdict: VerdictPayload
  iterations: number
  thinking_trace: string[]
  tool_calls: Record<string, unknown>[]
  evidence_collected: Record<string, unknown>
  nlu_findings: Record<string, unknown> | null
  nlu_escalated: boolean
  total_duration_ms: number
  timestamp: number
}

// === Blockchain Audit Trail (GET /api/v1/blockchain/recent-blocks) ===

export interface LedgerBlock {
  index: number
  timestamp: number
  event_type: string
  payload: Record<string, unknown>
  prev_hash: string
  block_hash: string
  has_signature: boolean
  merkle_root: string | null
}

export interface LedgerStats {
  total_blocks: number
  latest_index: number
  latest_hash: string
  latest_timestamp: number
  checkpoints: number
  db_size_bytes: number
}

export interface RecentBlocksResponse {
  blocks: LedgerBlock[]
  stats: LedgerStats | null
}

// === Analytics Endpoints ===

export interface RiskDistributionResponse {
  buckets: number[]
  total: number
  mean: number
  p95: number
}

export interface FraudTypologyResponse {
  typology: Record<string, number>
  total: number
}

export interface VelocityAccount {
  account_id: string
  count: number
  volume_paisa: number
  fraud_count: number
}

export interface VelocityTrendsResponse {
  accounts: VelocityAccount[]
  window_minutes: number
}

export interface TemporalBucket {
  bucket_start: number
  bucket_end: number
  txn_count: number
  fraud_count: number
  total_paisa: number
  max_amount: number
}

export interface TemporalHeatmapResponse {
  buckets: TemporalBucket[]
  bucket_seconds: number
  lookback_minutes: number
}

export interface ThreatIndicator {
  signal: string
  detail: string
  severity: 'critical' | 'high' | 'medium' | 'low'
}

export interface ThreatSummaryResponse {
  threat_level: 'critical' | 'high' | 'elevated' | 'normal' | 'unknown'
  severity_score: number
  frozen_count: number
  active_attacks: number
  indicators: ThreatIndicator[]
}

// === Intelligence Endpoints ===

// -- SHAP Explainability --

export interface FeatureContribution {
  name: string
  description: string
  value: number
  contribution: number
  direction: 'increases_risk' | 'decreases_risk'
}

export interface ExplainResponse {
  txn_id: string
  risk_score: number
  verdict: string
  narrative: string
  top_features: FeatureContribution[]
}

export interface GlobalImportanceResponse {
  feature_importance: Record<string, number>
  snapshot: Record<string, unknown>
}

// -- Model Drift --

export type DriftSeverity = 'NONE' | 'LOW' | 'MODERATE' | 'HIGH' | 'CRITICAL'

export interface FeatureDrift {
  feature: string
  psi: number
  severity: DriftSeverity
}

export interface DriftResponse {
  severity: DriftSeverity
  psi: number
  ks_statistic: number
  ks_p_value: number
  js_divergence: number
  reference_size: number
  current_size: number
  recommendation: string
  feature_drift: FeatureDrift[]
  snapshot: Record<string, unknown>
}

// -- Natural Language Query --

export interface NLQueryResponse {
  query: string
  intent: string
  answer: string
  sources: string[]
  confidence: number
  processing_ms: number
  model_used: string
}

// -- Consortium Intelligence --

export interface ConsortiumAlertData {
  alert_id: string
  originating_bank: string
  account_hash: string
  fraud_type: string
  severity: string
  risk_score: number
  zkp_verified: boolean
  timestamp: number
  ttl_hours: number
  expired: boolean
}

export interface ConsortiumStatusResponse {
  member_banks: number
  members: Record<string, { joined: number; alerts_published: number; trust_score: number }>
  total_alerts: number
  active_alerts: number
  verified_proofs: number
  rejected_proofs: number
  alerts_by_type: Record<string, number>
}

export interface ConsortiumAlertsResponse {
  count: number
  alerts: ConsortiumAlertData[]
}

export interface ConsortiumPublishResponse {
  status: string
  alert?: ConsortiumAlertData
  error?: string
}

export interface ConsortiumCheckResponse {
  account_id: string
  flagged: boolean
  alert_count: number
  alerts: ConsortiumAlertData[]
}

// === Fraud Intelligence Endpoints ===

// -- Rule Engine --

export interface RuleInfo {
  rule_id: string
  name: string
  description: string
  enabled: boolean
  threshold: number
  category: string
}

export interface RulesListResponse {
  rules: RuleInfo[]
  total: number
}

export interface RuleStatsResponse {
  total_evaluations: number
  rules_triggered: Record<string, number>
  last_evaluation_ms: number
  [key: string]: unknown
}

// -- Regulatory Reports --

export interface RegulatoryReport {
  report_id: string
  report_type: string
  status: string
  filed_at: number
}

export interface ReportsListResponse {
  total: number
  reports: RegulatoryReport[]
}

// -- Pre-Approval Gate --

export interface GateStatsResponse {
  total_evaluations: number
  approved: number
  held: number
  blocked: number
  avg_evaluation_ms: number
  [key: string]: unknown
}

// -- CFR --

export interface CFRStatsResponse {
  registry?: {
    total_records: number
    unique_accounts: number
    total_fraud_amount_paisa: number
    categories: Record<string, number>
    [key: string]: unknown
  }
  scorer?: Record<string, unknown>
  error?: string
}

// -- AML --

export interface AMLStatsResponse {
  placement?: {
    total_evaluations: number
    alerts_raised: number
    [key: string]: unknown
  }
  integration?: {
    total_evaluations: number
    alerts_raised: number
    [key: string]: unknown
  }
  error?: string
}

// -- FIU Intelligence --

export interface FIUStatsResponse {
  total_entries: number
  str_collected: number
  alerts_collected: number
  packages_prepared: number
  packages_disseminated: number
  high_risk_accounts: number
  [key: string]: unknown
}

export interface FIUHighRiskResponse {
  high_risk_accounts: string[]
  count: number
}

// -- Investigation --

export interface InvestigationStatsResponse {
  total_cases: number
  open_cases: number
  referred_cases: number
  legal_proceedings: number
  total_fraud_amount_paisa: number
  [key: string]: unknown
}

// -- Mule Detection --

export interface MuleChain {
  chain_nodes: string[]
  chain_length: number
  txn_ids: string[]
  origin_node: string
  terminal_node: string
  total_amount_paisa: number
  time_span_minutes: number
  detection_time_ms: number
}

export interface MuleChainsResponse {
  count: number
  chains: MuleChain[]
}

export interface MuleStatsResponse {
  chain_detector?: {
    total_chains_detected: number
    total_scans: number
    [key: string]: unknown
  }
  account_scorer?: {
    total_scored: number
    suspected_mules: number
    [key: string]: unknown
  }
  error?: string
}

export interface SuspectedMule {
  account_id: string
  mule_score: number
  risk_level: string
  indicators: {
    newly_opened: number
    high_frequency: number
    rapid_forward: number
    large_cashout: number
  }
}

export interface SuspectedMulesResponse {
  count: number
  mules: SuspectedMule[]
}

// -- Victim Fund Tracing --

export interface VictimStatsResponse {
  total_victims_traced: number
  [key: string]: unknown
}

// -- Anomaly Detection --

export interface AnomalyStatsResponse {
  isolation_forest?: {
    total_scored: number
    anomalies_detected: number
    [key: string]: unknown
  }
  autoencoder?: {
    total_scored: number
    anomalies_detected: number
    [key: string]: unknown
  }
  error?: string
}

// -- Community Clusters --

export interface SuspiciousCluster {
  community_id: number
  node_count: number
  internal_edges: number
  density: number
  avg_amount_paisa: number
  total_amount_paisa: number
  fraud_edge_ratio: number
  anomaly_score: number
}

export interface ClustersResponse {
  total_clusters: number
  clusters: SuspiciousCluster[]
}

// -- Centrality --

export interface CentralityIntermediary {
  node_id: string
  betweenness_centrality: number
  in_degree: number
  out_degree: number
  total_volume_paisa: number
  pagerank: number
  z_score: number
  detection_time_ms: number
}

export interface IntermediariesResponse {
  count: number
  intermediaries: CentralityIntermediary[]
}

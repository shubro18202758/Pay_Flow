// ============================================================================
// Activity Store -- Per-event lifecycle tracking from SSE streams
// ============================================================================

import { create } from 'zustand'
import {
  sanitizeEvidenceList,
  sanitizeOptionalEvidenceText,
  sanitizePublicTraceText,
} from '@/lib/evidence-sanitizer'

const MAX_TRACKED_EVENTS = 200
const MAX_TERMINAL_ENTRIES = 500
const MAX_EVENT_LAB_REPORTS = 50

export type PipelineStage =
  | 'ingested'
  | 'ml_scored'
  | 'graph_investigated'
  | 'cb_evaluated'
  | 'llm_started'
  | 'verdict'
  | 'pipeline_dispatched'

export interface StageDetail {
  stage: PipelineStage
  timestamp: number
  durationMs?: number
  meta?: Record<string, unknown>
}

export interface EventLifecycle {
  txnId: string
  sender: string
  receiver: string
  amountPaisa: number
  fraudLabel: number
  attackLabel: string
  scenarioId: string
  firstSeen: number
  stages: StageDetail[]
  // Populated as stages arrive
  riskScore?: number
  riskTier?: string
  topFeatures?: string[]
  verdict?: string
  confidence?: number
  fraudTypology?: string
  reasoningSummary?: string
  evidenceCited?: string[]
  recommendedAction?: string
  thinkingSteps?: number
  toolsUsed?: string[]
  totalDurationMs?: number | null
  nluFindingsCount?: number
  nluEscalated?: boolean
  confidenceSource?: string
  llmParseStatus?: string
  modelUsed?: string | null
  analystStatus?: string
  analystCaseId?: string
  analystReason?: string
  analystUpdatedAt?: number
  analystAuditHash?: string
  consensusScores?: { ml: number | null; gnn: number | null; graph: number | null; consensus: number | null }
  // Pipeline batch info
  pipelineConsumers?: Array<{ consumer: string; success: boolean; duration_ms: number; error?: string }>
}

export interface EventLabStageActivity {
  stage: string
  timestamp: number
  status?: string
  duration_ms?: number | null
  event_ids?: string[]
  meta?: Record<string, unknown>
}

export interface EventLabRunActivity {
  runId: string
  correlationId: string
  templateTitle: string
  status: string
  eventIds: string[]
  stages: EventLabStageActivity[]
  controls?: Record<string, unknown>
  eventSummaries?: Array<Record<string, unknown>>
  analysisReport?: Record<string, unknown>
  qwenExplanation?: string
  decisionAuthority?: string
  auditHash?: string
}

export interface EventLabReportActivity {
  runId: string
  correlationId: string
  templateTitle: string
  status: string
  riskTier: string
  riskScore: number | null
  verdict: string
  exposurePaisa: number
  eventCount: number
  uniqueAccountCount: number
  routeLabel: string
  generatedAt: number
  auditHash: string
  eventIds: string[]
  report: Record<string, unknown>
  syncTargets: Array<{ key: string; label: string; status: string; count: number }>
}

export interface CountermeasureActivity {
  proposalId: string
  runId: string
  action: string
  status: string
  title: string
  targets: string[]
  executionAllowed: boolean
  auditHash?: string
  updatedAt: number
}

export interface ResolutionFanoutActivity {
  txnId: string
  caseId: string
  nodeId: string
  status: string
  nodeStatus: string
  auditHash: string
  activityLifecycle: boolean
  graphNodeStatus: boolean
  queryGroups: string[]
  updatedAt: number
}

export type BackendTerminalTone = 'info' | 'success' | 'warn' | 'danger' | 'model' | 'muted'

export type BackendTerminalSource =
  | 'event_lab'
  | 'ingest'
  | 'ml'
  | 'heuristic'
  | 'control'
  | 'qwen'
  | 'decision'
  | 'dispatch'
  | 'counter'
  | 'analyst'
  | 'custom'
  | 'runtime'
  | 'system'

export interface BackendTerminalEntry {
  seq: number
  id: string
  timestamp: number
  source: BackendTerminalSource
  tone: BackendTerminalTone
  title: string
  detail?: string
  runId?: string
  txnId?: string
  txnIds?: string[]
  stage?: string
}

interface ActivityState {
  events: Map<string, EventLifecycle>
  orderedIds: string[] // most-recent-first
  trackedEventId: string | null
  eventLabRuns: Record<string, EventLabRunActivity>
  activeEventLabRunId: string | null
  eventLabReports: EventLabReportActivity[]
  countermeasures: CountermeasureActivity[]
  resolutionFanouts: ResolutionFanoutActivity[]
  terminalEntries: BackendTerminalEntry[]
  terminalSeq: number

  // Actions
  setTrackedEventId: (id: string | null) => void
  setActiveEventLabRunId: (id: string | null) => void
  appendTerminalEntry: (entry: Omit<BackendTerminalEntry, 'seq' | 'id'> & { id?: string }) => void
  clearTerminalEntries: () => void
  onGraphBatchUpdate: (edges: Array<{
    data: {
      id: string
      source: string
      target: string
      amount_paisa: number
      fraud_label: number
      timestamp?: number
    }
  }>) => void
  onAgentEvent: (data: {
    type: string
    txn_id: string
    iteration?: number
    max_iterations?: number
    verdict?: string
    confidence?: number
    fraud_typology?: string
    reasoning_summary?: string
    evidence_cited?: string[]
    recommended_action?: string
    thinking_steps?: number
    tools_used?: string[]
    total_duration_ms?: number | null
    nlu_findings_count?: number
    nlu_escalated?: boolean
    confidence_source?: string
    llm_parse_status?: string
    model_used?: string | null
    content?: string
    public_content?: string
    tool_name?: string
    success?: boolean
    duration_ms?: number | null
    output_summary?: string
  }, serverTimestamp?: number) => void
  onCBActivity: (
    data: { type?: string; node_id?: string; order?: Record<string, unknown> },
    txnId?: string,
    serverTimestamp?: number,
  ) => void
  onRiskScoreActivity: (data: {
    txn_id: string
    risk_score: number
    tier: string
    top_features?: string[]
  }, serverTimestamp?: number) => void
  onPipelineStage: (data: {
    type: string
    batch_id: number
    event_count: number
    transactions: number
    auth_events: number
    interbank_messages: number
    consumers: Array<{ consumer: string; success: boolean; duration_ms: number; error?: string }>
    txn_ids?: string[]
  }, serverTimestamp?: number) => void
  onPipelineStageComplete: (data: {
    type: string
    stage: PipelineStage
    txn_id?: string
    txn_ids?: string[]
    duration_ms?: number
    [key: string]: unknown
  }, serverTimestamp?: number) => void
  onEventLabActivity: (data: Record<string, unknown>) => void
  onCountermeasureActivity: (data: Record<string, unknown>) => void
  onAnalystActivity: (data: Record<string, unknown>, serverTimestamp?: number) => void
}

function scoreOrNull(raw: unknown): number | null {
  const n = Number(raw)
  if (!Number.isFinite(n) || n < 0) return null
  return Math.min(1, Math.max(0, n))
}

function timestampOrZero(raw: unknown): number {
  const n = Number(raw)
  return Number.isFinite(n) && n > 0 ? n : 0
}

function nowSeconds(): number {
  return Date.now() / 1000
}

function terminalTimestamp(raw?: unknown): number {
  return timestampOrZero(raw) || nowSeconds()
}

function compactTerminalValue(value: unknown, depth = 0): string {
  if (value == null || value === '') return ''
  if (Array.isArray(value)) {
    const rendered = value.map((item) => compactTerminalValue(item, depth + 1)).filter(Boolean)
    if (rendered.length === 0) return ''
    return rendered.length > 4 ? `${rendered.slice(0, 4).join(', ')} +${rendered.length - 4}` : rendered.join(', ')
  }
  if (typeof value === 'object') {
    if (depth > 0) return JSON.stringify(value).slice(0, 120)
    const record = value as Record<string, unknown>
    const keys = ['title', 'playbook_id', 'trend_id', 'proposal_id', 'status', 'decision', 'risk_score', 'tier']
    const parts = keys
      .filter((key) => record[key] != null && record[key] !== '')
      .map((key) => `${key}=${compactTerminalValue(record[key], depth + 1)}`)
    if (parts.length > 0) return parts.join(', ')
    return JSON.stringify(record).slice(0, 120)
  }
  return String(value)
}

function summarizeTerminalMeta(meta?: Record<string, unknown>): string {
  if (!meta || Object.keys(meta).length === 0) return ''
  const keys = [
    'batch_id',
    'event_count',
    'transactions',
    'auth_events',
    'interbank_messages',
    'count',
    'proposal_count',
    'transaction_count',
    'amount_band_inr',
    'route',
    'velocity_minutes',
    'mule_depth',
    'customer_profile',
    'controls',
    'pipeline',
    'risk_score',
    'risk_tier',
    'verdict',
    'total_exposure_paisa',
    'tier',
    'threshold',
    'top_features',
    'consumers',
    'proposal_id',
    'decision',
    'status',
    'audit_hash',
    'consensus_score',
    'graph_evidence_score',
    'ml_score',
    'gnn_score',
  ]
  const parts: string[] = []
  const linkedIntel = meta.linked_intel as Record<string, unknown> | undefined
  if (linkedIntel) {
    const trend = linkedIntel.trend as Record<string, unknown> | undefined
    const playbook = linkedIntel.playbook as Record<string, unknown> | undefined
    const trust = Number(linkedIntel.trust_score)
    if (trend?.title) parts.push(`trend=${String(trend.title).slice(0, 54)}`)
    if (playbook?.playbook_id) parts.push(`playbook=${String(playbook.playbook_id)}`)
    if (Number.isFinite(trust)) parts.push(`trust=${Math.round(trust * 100)}%`)
  }
  for (const key of keys) {
    if (!(key in meta)) continue
    const value = meta[key]
    if (key === 'consumers' && Array.isArray(value)) {
      parts.push(`consumers=${value.length}`)
    } else if (key === 'audit_hash') {
      parts.push(`audit=${String(value).slice(0, 14)}`)
    } else if (key === 'controls' && value && typeof value === 'object') {
      const record = value as Record<string, unknown>
      const rendered = [
        record.route_label ? `route=${record.route_label}` : '',
        record.primary_channel && record.secondary_channel ? `rails=${record.primary_channel}/${record.secondary_channel}` : '',
        record.event_count ? `events=${record.event_count}` : '',
        record.velocity_minutes ? `velocity=${record.velocity_minutes}m` : '',
      ].filter(Boolean).join(', ')
      if (rendered) parts.push(rendered)
    } else if (key === 'top_features') {
      const rendered = compactTerminalValue(value)
      if (rendered) parts.push(`features=${rendered}`)
    } else {
      const rendered = compactTerminalValue(value)
      if (rendered) parts.push(`${key}=${rendered}`)
    }
  }
  return parts.join(' | ').slice(0, 260)
}

function shortTerminal(value: unknown, left = 12): string {
  const rendered = String(value ?? '')
  if (!rendered) return 'n/a'
  return rendered.length > left + 4 ? `${rendered.slice(0, left)}...` : rendered
}

function txnListSummary(txnIds?: string[]): string {
  if (!txnIds || txnIds.length === 0) return ''
  return `ids=${txnIds.slice(0, 4).map((id) => shortTerminal(id, 10)).join(',')}${txnIds.length > 4 ? ` +${txnIds.length - 4}` : ''}`
}

function countRecordValue(record: Record<string, unknown> | undefined, key: string): number {
  if (!record || typeof record !== 'object') return 0
  const value = Number(record[key])
  return Number.isFinite(value) && value > 0 ? value : 0
}

function eventLabReportActivityFromRun(
  runId: string,
  run: Record<string, unknown>,
  report: Record<string, unknown>,
  eventIds: string[],
  stages: EventLabStageActivity[],
): EventLabReportActivity | null {
  const riskScore = Number(report.risk_score)
  const riskTier = String(report.risk_tier ?? '')
  if (!riskTier || !Number.isFinite(riskScore)) return null
  const coverage = report.stage_coverage && typeof report.stage_coverage === 'object'
    ? report.stage_coverage as Record<string, unknown>
    : {}
  const countermeasures = report.countermeasure_status && typeof report.countermeasure_status === 'object'
    ? report.countermeasure_status as Record<string, unknown>
    : {}
  const evidenceRows = Array.isArray(report.evidence_matrix) ? report.evidence_matrix.length : 0
  const stageRows = Array.isArray(report.stage_timeline) ? report.stage_timeline.length : stages.length
  const amountRows = Array.isArray(report.amount_series) ? report.amount_series.length : 0
  const routeRows = Array.isArray(report.route_segments) ? report.route_segments.length : Array.isArray(report.geo_path) ? report.geo_path.length : 0
  const proposalCount = countRecordValue(countermeasures, 'pending') + countRecordValue(countermeasures, 'executed') + countRecordValue(countermeasures, 'rejected')
  return {
    runId,
    correlationId: String(run.correlation_id ?? ''),
    templateTitle: String(run.template_title ?? 'Adaptive event run'),
    status: String(run.status ?? 'evaluated'),
    riskTier,
    riskScore,
    verdict: String(report.verdict ?? ''),
    exposurePaisa: Number(report.total_exposure_paisa ?? 0),
    eventCount: Number(report.event_count ?? eventIds.length),
    uniqueAccountCount: Number(report.unique_account_count ?? 0),
    routeLabel: String(report.route_label ?? ''),
    generatedAt: timestampOrZero(report.generated_at),
    auditHash: String(run.audit_hash ?? ''),
    eventIds,
    report,
    syncTargets: [
      { key: 'activity', label: 'Activity lifecycle', status: eventIds.length > 0 ? 'synced' : 'pending', count: eventIds.length },
      { key: 'graph', label: 'Fund-flow graph', status: countRecordValue(coverage, 'graph_investigated') > 0 ? 'synced' : 'pending', count: routeRows },
      { key: 'analytics', label: 'Analytics charts', status: amountRows > 0 ? 'synced' : 'pending', count: amountRows },
      { key: 'explainability', label: 'AI explainability', status: countRecordValue(coverage, 'qwen_context_loaded') > 0 ? 'synced' : 'pending', count: evidenceRows },
      { key: 'countermeasure', label: 'Countermeasure gate', status: proposalCount > 0 ? 'synced' : 'pending', count: proposalCount },
      { key: 'audit', label: 'Audit evidence', status: stageRows > 0 ? 'synced' : 'pending', count: stageRows },
    ],
  }
}

function stageTerminalSource(stage: string): BackendTerminalSource {
  if (stage === 'intel_primed' || stage === 'events_generated') return 'event_lab'
  if (stage === 'events_injected' || stage === 'ingested') return 'ingest'
  if (stage === 'pipeline_dispatched') return 'dispatch'
  if (stage === 'ml_scored') return 'ml'
  if (stage === 'graph_investigated') return 'heuristic'
  if (stage === 'cb_evaluated') return 'control'
  if (stage === 'llm_started' || stage === 'qwen_context_loaded' || stage === 'qwen_tool_call') return 'qwen'
  if (stage === 'analyst_decision') return 'analyst'
  if (stage === 'evaluation_complete' || stage === 'action_executed' || stage === 'ledger_anchored' || stage === 'evidence_ready') return 'decision'
  if (stage === 'verdict') return 'decision'
  return 'runtime'
}

function stageTerminalTone(stage: string): BackendTerminalTone {
  if (stage === 'evaluation_complete' || stage === 'action_executed' || stage === 'ledger_anchored' || stage === 'evidence_ready' || stage === 'verdict') {
    return 'success'
  }
  if (stage === 'cb_evaluated' || stage === 'analyst_decision') return 'warn'
  if (stage === 'llm_started' || stage === 'qwen_context_loaded' || stage === 'qwen_tool_call') return 'model'
  if (stage === 'ml_scored' || stage === 'graph_investigated' || stage === 'pipeline_dispatched') return 'info'
  return 'muted'
}

function stageTerminalTitle(stage: string): string {
  const labels: Record<string, string> = {
    intel_primed: 'load pre-fraud intelligence context',
    events_generated: 'generate correlated fraud event chain',
    events_injected: 'inject event chain into live ingestion',
    ingested: 'schema validated and normalized',
    pipeline_dispatched: 'dispatch event batch to backend consumers',
    ml_scored: 'run feature engine and risk scorer',
    graph_investigated: 'scan graph heuristics',
    cb_evaluated: 'evaluate circuit-breaker consensus',
    llm_started: 'invoke bounded Qwen explanation agent',
    qwen_context_loaded: 'load Qwen context window',
    qwen_tool_call: 'execute bounded AI tool call',
    analyst_decision: 'apply analyst decision gate',
    evaluation_complete: 'complete autonomous fraud evaluation',
    action_executed: 'execute approved countermeasure',
    ledger_anchored: 'anchor audit hash to ledger',
    evidence_ready: 'prepare evidence package',
    verdict: 'final verdict recorded',
  }
  return labels[stage] ?? stage.replaceAll('_', ' ')
}

function appendTerminalEntries(
  state: Pick<ActivityState, 'terminalEntries' | 'terminalSeq'>,
  rawEntries: Array<Omit<BackendTerminalEntry, 'seq' | 'id'> & { id?: string }>,
): Pick<ActivityState, 'terminalEntries' | 'terminalSeq'> {
  if (rawEntries.length === 0) return {
    terminalEntries: state.terminalEntries,
    terminalSeq: state.terminalSeq,
  }
  let terminalSeq = state.terminalSeq
  const terminalEntries = [...state.terminalEntries]
  const seen = new Set(terminalEntries.map((entry) => entry.id))
  for (const rawEntry of rawEntries) {
    const nextSeq = terminalSeq + 1
    const timestamp = terminalTimestamp(rawEntry.timestamp)
    const id = rawEntry.id ?? [
      'terminal',
      Math.round(timestamp * 1000),
      rawEntry.source,
      rawEntry.stage ?? '',
      rawEntry.runId ?? '',
      rawEntry.txnId ?? rawEntry.txnIds?.join(',') ?? '',
      nextSeq,
    ].join(':')
    if (seen.has(id)) continue
    terminalSeq = nextSeq
    seen.add(id)
    terminalEntries.push({
      ...rawEntry,
      id,
      seq: terminalSeq,
      timestamp,
    })
  }
  return {
    terminalSeq,
    terminalEntries: terminalEntries.slice(-MAX_TERMINAL_ENTRIES),
  }
}

function addStage(
  lifecycle: EventLifecycle,
  stage: PipelineStage,
  meta?: Record<string, unknown>,
  durationMs?: number,
  timestamp?: number,
) {
  if (lifecycle.stages.some((s) => s.stage === stage)) return
  lifecycle.stages.push({ stage, timestamp: timestampOrZero(timestamp), durationMs, meta })
}

const EVENT_LAB_STAGE_TO_PIPELINE_STAGE: Record<string, PipelineStage> = {
  events_injected: 'ingested',
  ingested: 'ingested',
  ml_scored: 'ml_scored',
  graph_investigated: 'graph_investigated',
  cb_evaluated: 'cb_evaluated',
  qwen_context_loaded: 'llm_started',
  qwen_tool_call: 'llm_started',
  llm_started: 'llm_started',
  pipeline_dispatched: 'pipeline_dispatched',
  evaluation_complete: 'verdict',
  evidence_ready: 'verdict',
  analyst_decision: 'verdict',
  ledger_anchored: 'verdict',
}

function eventLabSummaryId(summary: Record<string, unknown>): string {
  return String(summary.event_id ?? summary.txn_id ?? summary.msg_id ?? '')
}

function eventLabFraudLabel(raw: unknown): number {
  if (typeof raw === 'number' && Number.isFinite(raw)) return raw
  const label = String(raw ?? '').toLowerCase()
  if (!label || label === 'none' || label === 'legitimate' || label === 'clean') return 0
  return 1
}

function pruneEvents(events: Map<string, EventLifecycle>, orderedIds: string[]) {
  while (orderedIds.length > MAX_TRACKED_EVENTS) {
    const removed = orderedIds.pop()
    if (removed) events.delete(removed)
  }
}

function ensureLifecycle(
  events: Map<string, EventLifecycle>,
  orderedIds: string[],
  txnId: string,
  seed: Partial<EventLifecycle> = {},
): EventLifecycle {
  let lifecycle = events.get(txnId)
  if (lifecycle) return lifecycle

  lifecycle = {
    txnId,
    sender: seed.sender ?? '',
    receiver: seed.receiver ?? '',
    amountPaisa: seed.amountPaisa ?? 0,
    fraudLabel: seed.fraudLabel ?? 0,
    attackLabel: seed.attackLabel ?? '',
    scenarioId: seed.scenarioId ?? '',
    firstSeen: timestampOrZero(seed.firstSeen),
    stages: seed.stages ? [...seed.stages] : [],
  }
  events.set(txnId, lifecycle)
  orderedIds.unshift(txnId)
  return lifecycle
}

export const useActivityStore = create<ActivityState>((set) => ({
  events: new Map(),
  orderedIds: [],
  trackedEventId: null,
  eventLabRuns: {},
  activeEventLabRunId: null,
  eventLabReports: [],
  countermeasures: [],
  resolutionFanouts: [],
  terminalEntries: [],
  terminalSeq: 0,

  setTrackedEventId: (id) => set({ trackedEventId: id }),
  setActiveEventLabRunId: (id) => set({ activeEventLabRunId: id }),
  appendTerminalEntry: (entry) => set((state) => appendTerminalEntries(state, [entry])),
  clearTerminalEntries: () => set({ terminalEntries: [], terminalSeq: 0 }),

  onGraphBatchUpdate: (edges) =>
    set((state) => {
      const events = new Map(state.events)
      const orderedIds = [...state.orderedIds]
      let changed = false

      for (const edge of edges) {
        const { id: txnId, source, target, amount_paisa, fraud_label } = edge.data
        if (!txnId) continue

        if (!events.has(txnId)) {
          const observedAt = timestampOrZero(edge.data.timestamp)
          events.set(txnId, {
            txnId,
            sender: source,
            receiver: target,
            amountPaisa: amount_paisa,
            fraudLabel: fraud_label ?? 0,
            attackLabel: '',
            scenarioId: '',
            firstSeen: observedAt,
            stages: [{ stage: 'ingested', timestamp: observedAt }],
          })
          orderedIds.unshift(txnId)
          changed = true
        }
      }

      if (!changed) return {}
      pruneEvents(events, orderedIds)
      return { events, orderedIds }
    }),

  onAgentEvent: (data, serverTimestamp) =>
    set((state) => {
      if (!data.txn_id) return {}
      const events = new Map(state.events)
      const orderedIds = [...state.orderedIds]
      const eventTimestamp = timestampOrZero(serverTimestamp)
      const terminalAt = terminalTimestamp(serverTimestamp)
      const terminalRows: Array<Omit<BackendTerminalEntry, 'seq' | 'id'> & { id?: string }> = []

      const lifecycle = ensureLifecycle(events, orderedIds, data.txn_id, { firstSeen: eventTimestamp })

      if (data.type === 'thinking_step' || data.type === 'tool_call') {
        const publicContent =
          data.type === 'thinking_step'
            ? sanitizePublicTraceText(data.public_content ?? data.content)
            : data.content
        addStage(lifecycle, 'llm_started', {
          type: data.type,
          iteration: data.iteration,
          max_iterations: data.max_iterations,
          tool_name: data.tool_name,
          content: publicContent?.slice(0, 200),
          output_summary: data.output_summary?.slice(0, 200),
        }, undefined, eventTimestamp)
        terminalRows.push({
          id: `agent:${data.txn_id}:${data.type}:${data.iteration ?? ''}:${data.tool_name ?? ''}:${terminalAt}`,
          timestamp: terminalAt,
          source: 'qwen',
          tone: 'model',
          title: data.type === 'tool_call'
            ? `Qwen tool call ${data.tool_name ?? 'tool'}`
            : `Qwen thinking step ${data.iteration ?? 0}/${data.max_iterations ?? '?'}`,
          detail: [
            `txn=${shortTerminal(data.txn_id, 14)}`,
            data.success != null ? `success=${data.success ? 'true' : 'false'}` : '',
            data.duration_ms != null ? `duration=${data.duration_ms}ms` : '',
            publicContent ? `content=${publicContent.slice(0, 180)}` : '',
            data.output_summary ? `output=${data.output_summary.slice(0, 160)}` : '',
          ].filter(Boolean).join(' | '),
          txnId: data.txn_id,
          stage: 'llm_started',
        })
      } else if (data.type === 'verdict') {
        addStage(lifecycle, 'llm_started', undefined, undefined, eventTimestamp)
        addStage(lifecycle, 'verdict', undefined, undefined, eventTimestamp)
        lifecycle.verdict = data.verdict
        lifecycle.confidence = data.confidence
        lifecycle.fraudTypology = data.fraud_typology
        lifecycle.reasoningSummary = sanitizeOptionalEvidenceText(data.reasoning_summary)
        lifecycle.evidenceCited = sanitizeEvidenceList(data.evidence_cited)
        lifecycle.recommendedAction = data.recommended_action
        lifecycle.thinkingSteps = data.thinking_steps
        lifecycle.toolsUsed = data.tools_used
        lifecycle.totalDurationMs = data.total_duration_ms
        lifecycle.nluFindingsCount = data.nlu_findings_count
        lifecycle.nluEscalated = data.nlu_escalated
        lifecycle.confidenceSource = data.confidence_source
        lifecycle.llmParseStatus = data.llm_parse_status
        lifecycle.modelUsed = data.model_used
        terminalRows.push({
          id: `agent:${data.txn_id}:verdict:${data.verdict ?? ''}:${terminalAt}`,
          timestamp: terminalAt,
          source: 'qwen',
          tone: 'model',
          title: `Qwen verdict ${data.verdict ?? 'pending'}`,
          detail: [
            `txn=${shortTerminal(data.txn_id, 14)}`,
            data.confidence != null ? `confidence=${Math.round(data.confidence * 100)}%` : '',
            data.model_used ? `model=${data.model_used}` : '',
            data.recommended_action ? `action=${data.recommended_action}` : '',
            sanitizeOptionalEvidenceText(data.reasoning_summary)?.slice(0, 180)
              ? `reason=${sanitizeOptionalEvidenceText(data.reasoning_summary)?.slice(0, 180)}`
              : '',
          ].filter(Boolean).join(' | '),
          txnId: data.txn_id,
          stage: 'verdict',
        })
      }

      // Bump to top of ordered list
      const idx = orderedIds.indexOf(data.txn_id)
      if (idx > 0) {
        orderedIds.splice(idx, 1)
        orderedIds.unshift(data.txn_id)
      }

      pruneEvents(events, orderedIds)
      return { events, orderedIds, ...appendTerminalEntries(state, terminalRows) }
    }),

  onCBActivity: (data, txnId, serverTimestamp) =>
    set((state) => {
      const nodeId = data.node_id
      if (!nodeId && !txnId) return {}

      const events = new Map(state.events)
      const targetId = txnId ?? nodeId!
      const lifecycle = events.get(targetId)
      if (!lifecycle) return {}
      const terminalAt = terminalTimestamp(serverTimestamp)

      addStage(lifecycle, 'cb_evaluated', {
        type: data.type,
        nodeId: data.node_id,
        ...(data.order ?? {}),
      }, undefined, timestampOrZero(serverTimestamp))

      if (data.order) {
        const order = data.order as Record<string, unknown>
        if (
          order.ml_score !== undefined ||
          order.gnn_score !== undefined ||
          order.graph_score !== undefined ||
          order.graph_evidence_score !== undefined ||
          order.consensus_score !== undefined
        ) {
          lifecycle.consensusScores = {
            ml: scoreOrNull(order.ml_score),
            gnn: scoreOrNull(order.gnn_score),
            graph: scoreOrNull(order.graph_score ?? order.graph_evidence_score),
            consensus: scoreOrNull(order.consensus_score),
          }
        }
      }

      const order = data.order as Record<string, unknown> | undefined
      return {
        events,
        ...appendTerminalEntries(state, [{
          id: `cb:${targetId}:${data.type ?? 'event'}:${terminalAt}`,
          timestamp: terminalAt,
          source: 'control',
          tone: 'warn',
          title: 'circuit breaker consensus evaluated',
          detail: [
            `txn=${shortTerminal(targetId, 14)}`,
            data.type ? `type=${data.type}` : '',
            summarizeTerminalMeta(order),
          ].filter(Boolean).join(' | '),
          txnId: targetId,
          stage: 'cb_evaluated',
        }]),
      }
    }),

  onRiskScoreActivity: (data, serverTimestamp) =>
    set((state) => {
      if (!data.txn_id) return {}
      const events = new Map(state.events)
      const orderedIds = [...state.orderedIds]
      const eventTimestamp = timestampOrZero(serverTimestamp)
      const lifecycle = ensureLifecycle(events, orderedIds, data.txn_id, { firstSeen: eventTimestamp })

      addStage(lifecycle, 'ml_scored', { top_features: data.top_features }, undefined, eventTimestamp)
      lifecycle.riskScore = data.risk_score
      lifecycle.riskTier = data.tier
      lifecycle.topFeatures = data.top_features

      pruneEvents(events, orderedIds)
      return {
        events,
        orderedIds,
        ...appendTerminalEntries(state, [{
          id: `risk:${data.txn_id}:${data.risk_score}:${eventTimestamp || 'live'}`,
          timestamp: eventTimestamp || nowSeconds(),
          source: 'ml',
          tone: 'info',
          title: 'feature engine and risk scorer completed',
          detail: [
            `txn=${shortTerminal(data.txn_id, 14)}`,
            `risk_score=${Number(data.risk_score).toFixed(3)}`,
            `tier=${data.tier}`,
            data.top_features?.length ? `features=${data.top_features.slice(0, 6).join(', ')}` : 'features=pending',
          ].join(' | '),
          txnId: data.txn_id,
          stage: 'ml_scored',
        }]),
      }
    }),

  onPipelineStage: (data, serverTimestamp) =>
    set((state) => {
      const events = new Map(state.events)
      const orderedIds = [...state.orderedIds]
      const eventTimestamp = timestampOrZero(serverTimestamp)
      const targetIds =
        data.txn_ids && data.txn_ids.length > 0
          ? data.txn_ids
          : state.orderedIds.slice(0, Math.min(data.event_count, 20))

      for (const txnId of targetIds) {
        const lifecycle = ensureLifecycle(events, orderedIds, txnId, { firstSeen: eventTimestamp })
        lifecycle.pipelineConsumers = data.consumers
        addStage(lifecycle, 'pipeline_dispatched', {
          batch_id: data.batch_id,
          event_count: data.event_count,
          transactions: data.transactions,
          auth_events: data.auth_events,
          interbank_messages: data.interbank_messages,
          exact_txn_ids: Boolean(data.txn_ids?.length),
        }, undefined, eventTimestamp)
      }
      pruneEvents(events, orderedIds)
      return {
        events,
        orderedIds,
        ...appendTerminalEntries(state, [{
          id: `pipeline:${data.batch_id}:${eventTimestamp || 'live'}`,
          timestamp: eventTimestamp || nowSeconds(),
          source: 'dispatch',
          tone: 'info',
          title: 'dispatch event batch to backend consumers',
          detail: [
            `batch=${data.batch_id}`,
            `events=${data.event_count}`,
            `transactions=${data.transactions}`,
            `auth=${data.auth_events}`,
            `interbank=${data.interbank_messages}`,
            `consumers=${data.consumers.length}`,
            txnListSummary(targetIds),
          ].filter(Boolean).join(' | '),
          txnIds: targetIds,
          stage: 'pipeline_dispatched',
        }]),
      }
    }),

  onPipelineStageComplete: (data, serverTimestamp) =>
    set((state) => {
      const events = new Map(state.events)
      let changed = false
      const eventTimestamp = timestampOrZero(serverTimestamp)

      // Single-txn stage event (graph_investigated, cb_evaluated)
      if (data.txn_id) {
        const orderedIds = [...state.orderedIds]
        const lifecycle = ensureLifecycle(events, orderedIds, data.txn_id, { firstSeen: eventTimestamp })
        const { stage, duration_ms } = data
        const meta = { ...data } as Record<string, unknown>
        delete meta.type
        delete meta.stage
        delete meta.txn_id
        delete meta.duration_ms
        addStage(lifecycle, stage, meta as Record<string, unknown>, duration_ms, eventTimestamp)
        pruneEvents(events, orderedIds)
        return {
          events,
          orderedIds,
          ...appendTerminalEntries(state, [{
            id: `pipeline-complete:${data.txn_id}:${stage}:${eventTimestamp || 'live'}:${duration_ms ?? ''}`,
            timestamp: eventTimestamp || nowSeconds(),
            source: stageTerminalSource(stage),
            tone: stageTerminalTone(stage),
            title: `${stageTerminalTitle(stage)} complete`,
            detail: [
              `txn=${shortTerminal(data.txn_id, 14)}`,
              duration_ms != null ? `duration=${duration_ms}ms` : '',
              summarizeTerminalMeta(meta),
            ].filter(Boolean).join(' | '),
            txnId: data.txn_id,
            stage,
          }]),
        }
      }

      // Batch stage event (ml_scored — txn_ids array)
      if (data.txn_ids && Array.isArray(data.txn_ids)) {
        const orderedIds = [...state.orderedIds]
        for (const txnId of data.txn_ids as string[]) {
          const lifecycle = ensureLifecycle(events, orderedIds, txnId, { firstSeen: eventTimestamp })
          const { stage, duration_ms } = data
          const meta = { ...data } as Record<string, unknown>
          delete meta.type
          delete meta.stage
          delete meta.txn_ids
          delete meta.duration_ms
          addStage(lifecycle, stage, meta as Record<string, unknown>, duration_ms, eventTimestamp)
          changed = true
        }
        if (changed) pruneEvents(events, orderedIds)
        return changed ? {
          events,
          orderedIds,
          ...appendTerminalEntries(state, [{
            id: `pipeline-complete:batch:${data.stage}:${eventTimestamp || 'live'}:${data.txn_ids.join(',')}`,
            timestamp: eventTimestamp || nowSeconds(),
            source: stageTerminalSource(data.stage),
            tone: stageTerminalTone(data.stage),
            title: `${stageTerminalTitle(data.stage)} batch complete`,
            detail: [
              `count=${data.txn_ids.length}`,
              data.duration_ms != null ? `duration=${data.duration_ms}ms` : '',
              txnListSummary(data.txn_ids),
            ].filter(Boolean).join(' | '),
            txnIds: data.txn_ids,
            stage: data.stage,
          }]),
        } : {}
      }

      return changed ? { events } : {}
    }),

  onEventLabActivity: (data) =>
    set((state) => {
      const runs = { ...state.eventLabRuns }
      const events = new Map(state.events)
      const orderedIds = [...state.orderedIds]
      let activeEventLabRunId = state.activeEventLabRunId
      let trackedEventId = state.trackedEventId
      const terminalRows: Array<Omit<BackendTerminalEntry, 'seq' | 'id'> & { id?: string }> = []

      const type = String(data.type ?? '')
      if (type === 'run_launched' && data.run && typeof data.run === 'object') {
        const run = data.run as Record<string, unknown>
        const runId = String(run.run_id ?? '')
        if (!runId) return {}
        const eventIds = Array.isArray(run.event_ids) ? run.event_ids.map(String) : []
        const summaries = Array.isArray(run.events) ? (run.events as Record<string, unknown>[]) : []
        const runStages = Array.isArray(run.stages) ? (run.stages as EventLabStageActivity[]) : []
        const runTimestamp = terminalTimestamp(run.updated_at ?? run.created_at)
        runs[runId] = {
          runId,
          correlationId: String(run.correlation_id ?? ''),
          templateTitle: String(run.template_title ?? 'Adaptive event run'),
          status: String(run.status ?? 'injected'),
          eventIds,
          stages: runStages,
          controls: run.controls && typeof run.controls === 'object' ? run.controls as Record<string, unknown> : {},
          eventSummaries: summaries,
          analysisReport: run.analysis_report && typeof run.analysis_report === 'object' ? run.analysis_report as Record<string, unknown> : {},
          qwenExplanation: run.qwen_explanation ? sanitizePublicTraceText(run.qwen_explanation) : '',
          decisionAuthority: String(run.decision_authority ?? ''),
          auditHash: String(run.audit_hash ?? ''),
        }
        terminalRows.push({
          id: `event-lab:${runId}:run-launched`,
          timestamp: runTimestamp,
          source: 'event_lab',
          tone: 'success',
          title: 'event lab run attached to live pipeline',
          detail: [
            `template=${String(run.template_title ?? 'Adaptive event run')}`,
            `status=${String(run.status ?? 'injected')}`,
            `events=${eventIds.length}`,
            run.analysis_report && typeof run.analysis_report === 'object'
              ? `risk=${String((run.analysis_report as Record<string, unknown>).risk_tier ?? 'n/a')}/${String((run.analysis_report as Record<string, unknown>).risk_score ?? 'n/a')}`
              : '',
            run.correlation_id ? `correlation=${shortTerminal(run.correlation_id, 16)}` : '',
          ].filter(Boolean).join(' | '),
          runId,
          txnIds: eventIds,
          stage: 'run_launched',
        })
        runStages.forEach((runStage) => {
          const stageName = String(runStage.stage ?? 'stage')
          const stageEventIds = Array.isArray(runStage.event_ids) ? runStage.event_ids.map(String) : eventIds
          terminalRows.push({
            id: `event-lab:${runId}:stage:${stageName}:${runStage.timestamp ?? ''}:${stageEventIds.join(',')}`,
            timestamp: terminalTimestamp(runStage.timestamp ?? runTimestamp),
            source: stageTerminalSource(stageName),
            tone: stageTerminalTone(stageName),
            title: stageTerminalTitle(stageName),
            detail: [
              txnListSummary(stageEventIds),
              runStage.duration_ms != null ? `duration=${runStage.duration_ms}ms` : '',
              summarizeTerminalMeta(runStage.meta),
            ].filter(Boolean).join(' | '),
            runId,
            txnIds: stageEventIds,
            stage: stageName,
          })
        })
        if (run.qwen_explanation) {
          terminalRows.push({
            id: `event-lab:${runId}:qwen-explanation`,
            timestamp: runTimestamp + 0.002,
            source: 'qwen',
            tone: 'model',
            title: 'Qwen 3.5 4B context explanation generated',
            detail: sanitizePublicTraceText(run.qwen_explanation).slice(0, 260),
            runId,
            txnIds: eventIds,
            stage: 'qwen_explanation',
          })
        }
        if (run.analysis_report && typeof run.analysis_report === 'object') {
          const report = run.analysis_report as Record<string, unknown>
          terminalRows.push({
            id: `event-lab:${runId}:analysis-report`,
            timestamp: runTimestamp + 0.004,
            source: 'decision',
            tone: 'success',
            title: `autonomous fraud evaluation report: ${String(report.risk_tier ?? 'risk')} risk`,
            detail: [
              `verdict=${String(report.verdict ?? 'pending')}`,
              `exposure=${String(report.total_exposure_paisa ?? 0)}p`,
              `accounts=${String(report.unique_account_count ?? 'n/a')}`,
              `route=${String(report.route_label ?? 'n/a')}`,
            ].join(' | '),
            runId,
            txnIds: eventIds,
            stage: 'analysis_report',
          })
        }

        const firstTrackableSummary = summaries.find((summary) => String(summary.type ?? '') !== 'auth') ?? summaries[0]
        trackedEventId = eventLabSummaryId(firstTrackableSummary ?? {}) || eventIds[0] || trackedEventId

        const knownSummaries = summaries.length > 0
          ? summaries
          : eventIds.map((id, index): Record<string, unknown> => ({ event_id: id, sequence: index }))
        for (const summary of knownSummaries) {
          const eventId = eventLabSummaryId(summary)
          if (!eventId) continue
          const lifecycle = ensureLifecycle(events, orderedIds, eventId, {
            sender: String(summary.sender ?? summary.account ?? ''),
            receiver: String(summary.receiver ?? ''),
            amountPaisa: Number(summary.amount_paisa ?? 0),
            fraudLabel: eventLabFraudLabel(summary.fraud_label),
            attackLabel: String(summary.counterparty_role ?? summary.narrative ?? ''),
            scenarioId: String(run.template_id ?? ''),
            firstSeen: timestampOrZero(summary.timestamp ?? run.updated_at ?? run.created_at),
          })
          for (const runStage of runStages) {
            const mapped = EVENT_LAB_STAGE_TO_PIPELINE_STAGE[String(runStage.stage ?? '')]
            if (!mapped) continue
            const stageEventIds = Array.isArray(runStage.event_ids) ? runStage.event_ids.map(String) : []
            if (stageEventIds.length > 0 && !stageEventIds.includes(eventId)) continue
            addStage(lifecycle, mapped, {
              event_lab_run_id: runId,
              event_lab_stage: runStage.stage,
              ...(runStage.meta ?? {}),
            }, runStage.duration_ms ?? undefined, runStage.timestamp)
          }
        }

        activeEventLabRunId = runId
        pruneEvents(events, orderedIds)
        return {
          eventLabRuns: runs,
          activeEventLabRunId,
          trackedEventId,
          events,
          orderedIds,
          ...appendTerminalEntries(state, terminalRows),
        }
      }

      if (type === 'run_completed' && data.run && typeof data.run === 'object') {
        const run = data.run as Record<string, unknown>
        const runId = String(run.run_id ?? '')
        if (!runId) return {}
        const current = runs[runId]
        const eventIds = Array.isArray(run.event_ids)
          ? run.event_ids.map(String)
          : current?.eventIds ?? []
        const summaries = Array.isArray(run.events)
          ? (run.events as Record<string, unknown>[])
          : current?.eventSummaries ?? []
        const runStages = Array.isArray(run.stages) ? (run.stages as EventLabStageActivity[]) : current?.stages ?? []
        const runTimestamp = terminalTimestamp(run.updated_at ?? run.created_at)
        const report = run.analysis_report && typeof run.analysis_report === 'object'
          ? run.analysis_report as Record<string, unknown>
          : current?.analysisReport ?? {}

        runs[runId] = {
          runId,
          correlationId: String(run.correlation_id ?? current?.correlationId ?? ''),
          templateTitle: String(run.template_title ?? current?.templateTitle ?? 'Adaptive event run'),
          status: String(run.status ?? 'evaluated'),
          eventIds,
          stages: runStages,
          controls: run.controls && typeof run.controls === 'object' ? run.controls as Record<string, unknown> : current?.controls ?? {},
          eventSummaries: summaries,
          analysisReport: report,
          qwenExplanation: run.qwen_explanation ? sanitizePublicTraceText(run.qwen_explanation) : current?.qwenExplanation ?? '',
          decisionAuthority: String(run.decision_authority ?? current?.decisionAuthority ?? ''),
          auditHash: String(run.audit_hash ?? current?.auditHash ?? ''),
        }

        terminalRows.push({
          id: `event-lab:${runId}:run-completed`,
          timestamp: runTimestamp,
          source: 'decision',
          tone: 'success',
          title: 'autonomous fraud evaluation completed',
          detail: [
            `status=${String(run.status ?? 'evaluated')}`,
            `events=${eventIds.length}`,
            `report_unlocked=true`,
            report.risk_tier ? `risk=${String(report.risk_tier)}/${String(report.risk_score ?? 'n/a')}` : '',
            report.verdict ? `verdict=${String(report.verdict)}` : '',
            run.correlation_id ? `correlation=${shortTerminal(run.correlation_id, 16)}` : '',
          ].filter(Boolean).join(' | '),
          runId,
          txnIds: eventIds,
          stage: 'evaluation_complete',
        })

        if (Object.keys(report).length > 0) {
          terminalRows.push({
            id: `event-lab:${runId}:analysis-report`,
            timestamp: runTimestamp + 0.004,
            source: 'decision',
            tone: 'success',
            title: `autonomous fraud evaluation report: ${String(report.risk_tier ?? 'risk')} risk`,
            detail: [
              `verdict=${String(report.verdict ?? 'pending')}`,
              `exposure=${String(report.total_exposure_paisa ?? 0)}p`,
              `accounts=${String(report.unique_account_count ?? 'n/a')}`,
              `route=${String(report.route_label ?? 'n/a')}`,
            ].join(' | '),
            runId,
            txnIds: eventIds,
            stage: 'analysis_report',
          })
        }

        runStages.forEach((runStage) => {
          const stageName = String(runStage.stage ?? 'stage')
          const stageEventIds = Array.isArray(runStage.event_ids) ? runStage.event_ids.map(String) : eventIds
          terminalRows.push({
            id: `event-lab:${runId}:stage:${stageName}:${runStage.timestamp ?? ''}:${stageEventIds.join(',')}`,
            timestamp: terminalTimestamp(runStage.timestamp ?? runTimestamp),
            source: stageTerminalSource(stageName),
            tone: stageTerminalTone(stageName),
            title: stageTerminalTitle(stageName),
            detail: [
              txnListSummary(stageEventIds),
              runStage.duration_ms != null ? `duration=${runStage.duration_ms}ms` : '',
              summarizeTerminalMeta(runStage.meta),
            ].filter(Boolean).join(' | '),
            runId,
            txnIds: stageEventIds,
            stage: stageName,
          })
        })

        const firstTrackableSummary = summaries.find((summary) => String(summary.type ?? '') !== 'auth') ?? summaries[0]
        const reportActivity = eventLabReportActivityFromRun(runId, run, report, eventIds, runStages)
        const eventLabReports = reportActivity
          ? [reportActivity, ...state.eventLabReports.filter((item) => item.runId !== runId)].slice(0, MAX_EVENT_LAB_REPORTS)
          : state.eventLabReports
        let resolutionFanouts = state.resolutionFanouts
        if (reportActivity) {
          const focusId = eventLabSummaryId(firstTrackableSummary ?? {}) || eventIds[0] || runId
          const graphSynced = reportActivity.syncTargets.some((target) => target.key === 'graph' && target.status === 'synced')
          const analysisFanout: ResolutionFanoutActivity = {
            txnId: focusId,
            caseId: `REPORT-${runId}`,
            nodeId: String(firstTrackableSummary?.receiver ?? firstTrackableSummary?.account ?? focusId),
            status: 'analysis_ready',
            nodeStatus: `${reportActivity.riskTier}_risk_report_synced`,
            auditHash: reportActivity.auditHash,
            activityLifecycle: true,
            graphNodeStatus: graphSynced,
            queryGroups: reportActivity.syncTargets
              .filter((target) => target.status === 'synced')
              .map((target) => target.key),
            updatedAt: reportActivity.generatedAt || runTimestamp,
          }
          const rest = state.resolutionFanouts.filter((item) => item.caseId !== analysisFanout.caseId)
          resolutionFanouts = [analysisFanout, ...rest].slice(0, 50)
          terminalRows.push({
            id: `event-lab:${runId}:prototype-sync:${analysisFanout.updatedAt || 'live'}`,
            timestamp: analysisFanout.updatedAt || nowSeconds(),
            source: 'dispatch',
            tone: 'success',
            title: 'completed evaluation synchronized across prototype',
            detail: [
              `report=${shortTerminal(runId, 14)}`,
              `risk=${reportActivity.riskTier}/${reportActivity.riskScore?.toFixed(3) ?? 'n/a'}`,
              `targets=${analysisFanout.queryGroups.join(',') || 'none'}`,
              `events=${eventIds.length}`,
            ].join(' | '),
            runId,
            txnId: focusId,
            txnIds: eventIds,
            stage: 'prototype_sync',
          })
        }
        trackedEventId = eventLabSummaryId(firstTrackableSummary ?? {}) || eventIds[0] || trackedEventId

        const knownSummaries = summaries.length > 0
          ? summaries
          : eventIds.map((id, index): Record<string, unknown> => ({ event_id: id, sequence: index }))
        for (const summary of knownSummaries) {
          const eventId = eventLabSummaryId(summary)
          if (!eventId) continue
          const lifecycle = ensureLifecycle(events, orderedIds, eventId, {
            sender: String(summary.sender ?? summary.account ?? ''),
            receiver: String(summary.receiver ?? ''),
            amountPaisa: Number(summary.amount_paisa ?? 0),
            fraudLabel: eventLabFraudLabel(summary.fraud_label),
            attackLabel: String(summary.counterparty_role ?? summary.narrative ?? ''),
            scenarioId: String(run.template_id ?? ''),
            firstSeen: timestampOrZero(summary.timestamp ?? run.updated_at ?? run.created_at),
          })
          for (const runStage of runStages) {
            const mapped = EVENT_LAB_STAGE_TO_PIPELINE_STAGE[String(runStage.stage ?? '')]
            if (!mapped) continue
            const stageEventIds = Array.isArray(runStage.event_ids) ? runStage.event_ids.map(String) : []
            if (stageEventIds.length > 0 && !stageEventIds.includes(eventId)) continue
            addStage(lifecycle, mapped, {
              event_lab_run_id: runId,
              event_lab_stage: runStage.stage,
              ...(runStage.meta ?? {}),
            }, runStage.duration_ms ?? undefined, runStage.timestamp)
          }
        }

        activeEventLabRunId = runId
        pruneEvents(events, orderedIds)
        return {
          eventLabRuns: runs,
          activeEventLabRunId,
          eventLabReports,
          resolutionFanouts,
          trackedEventId,
          events,
          orderedIds,
          ...appendTerminalEntries(state, terminalRows),
        }
      }

      if (type === 'stage') {
        const runId = String(data.run_id ?? '')
        const stage = data.stage as EventLabStageActivity | undefined
        if (!runId || !stage) return {}
        const current = runs[runId] ?? {
          runId,
          correlationId: String(data.correlation_id ?? ''),
          templateTitle: 'Adaptive event run',
          status: String(data.run_status ?? 'running'),
          eventIds: [],
          stages: [],
          controls: {},
          eventSummaries: [],
          analysisReport: {},
        }
        runs[runId] = {
          ...current,
          status: String(data.run_status ?? current.status),
          stages: [...current.stages, stage].slice(-80),
        }
        const stageName = String(stage.stage ?? 'stage')
        const stageEventIds = Array.isArray(stage.event_ids) ? stage.event_ids.map(String) : []
        terminalRows.push({
          id: `event-lab:${runId}:stage:${stageName}:${stage.timestamp ?? ''}:${stageEventIds.join(',')}`,
          timestamp: terminalTimestamp(stage.timestamp),
          source: stageTerminalSource(stageName),
          tone: stageTerminalTone(stageName),
          title: stageTerminalTitle(stageName),
          detail: [
            txnListSummary(stageEventIds),
            stage.duration_ms != null ? `duration=${stage.duration_ms}ms` : '',
            summarizeTerminalMeta(stage.meta),
          ].filter(Boolean).join(' | '),
          runId,
          txnIds: stageEventIds,
          stage: stageName,
        })
        const mapped = EVENT_LAB_STAGE_TO_PIPELINE_STAGE[String(stage.stage ?? '')]
        if (mapped) {
          for (const eventId of stageEventIds) {
            const lifecycle = ensureLifecycle(events, orderedIds, eventId, { firstSeen: stage.timestamp })
            addStage(lifecycle, mapped, {
              event_lab_run_id: runId,
              event_lab_stage: stage.stage,
              ...(stage.meta ?? {}),
            }, stage.duration_ms ?? undefined, stage.timestamp)
          }
          if (!trackedEventId && stageEventIds[0]) trackedEventId = stageEventIds[0]
          pruneEvents(events, orderedIds)
        }
        activeEventLabRunId = runId
        return {
          eventLabRuns: runs,
          activeEventLabRunId,
          trackedEventId,
          events,
          orderedIds,
          ...appendTerminalEntries(state, terminalRows),
        }
      }

      return {}
    }),

  onCountermeasureActivity: (data) =>
    set((state) => {
      if (!data.proposal || typeof data.proposal !== 'object') return {}
      const proposal = data.proposal as Record<string, unknown>
      const proposalId = String(proposal.proposal_id ?? '')
      if (!proposalId) return {}
      const next: CountermeasureActivity = {
        proposalId,
        runId: String(proposal.run_id ?? ''),
        action: String(proposal.action ?? ''),
        status: String(proposal.status ?? ''),
        title: String(proposal.title ?? proposal.action ?? ''),
        targets: Array.isArray(proposal.targets) ? proposal.targets.map(String) : [],
        executionAllowed: Boolean(proposal.execution_allowed),
        auditHash: String(proposal.audit_hash ?? ''),
        updatedAt: timestampOrZero(proposal.updated_at),
      }
      const rest = state.countermeasures.filter((item) => item.proposalId !== proposalId)
      return {
        countermeasures: [next, ...rest].slice(0, 80),
        ...appendTerminalEntries(state, [{
          id: `countermeasure:${next.proposalId}:${next.status}:${next.updatedAt || 'live'}`,
          timestamp: next.updatedAt || nowSeconds(),
          source: 'counter',
          tone: next.status === 'executed' || next.status === 'approved'
            ? 'success'
            : next.status === 'failed'
              ? 'danger'
              : next.status === 'rejected'
                ? 'muted'
                : 'warn',
          title: `countermeasure ${next.status || 'proposed'}: ${next.action || next.title}`,
          detail: [
            `proposal=${shortTerminal(next.proposalId, 14)}`,
            `targets=${next.targets.map((target) => shortTerminal(target, 10)).join(',') || 'n/a'}`,
            `execution_allowed=${next.executionAllowed ? 'true' : 'false'}`,
            next.auditHash ? `audit=${shortTerminal(next.auditHash, 14)}` : 'audit=pending',
          ].join(' | '),
          runId: next.runId,
          txnIds: next.targets,
          stage: 'countermeasure',
        }]),
      }
    }),

  onAnalystActivity: (data, serverTimestamp) =>
    set((state) => {
      if (!data.case || typeof data.case !== 'object') return {}
      const analystCase = data.case as Record<string, unknown>
      const txnId = String(analystCase.txn_id ?? '')
      if (!txnId) return {}

      const events = new Map(state.events)
      const orderedIds = [...state.orderedIds]
      const receivedAt = timestampOrZero(analystCase.received_at)
      const updatedAt = timestampOrZero(analystCase.updated_at)
      const eventTimestamp = updatedAt || timestampOrZero(serverTimestamp)
      const lifecycle = ensureLifecycle(events, orderedIds, txnId, {
        sender: String(analystCase.node_id ?? ''),
        firstSeen: receivedAt,
      })

      lifecycle.analystStatus = String(analystCase.status ?? '')
      lifecycle.analystCaseId = String(analystCase.case_id ?? analystCase.ack_id ?? '')
      lifecycle.analystReason = String(analystCase.analyst_reason ?? '')
      lifecycle.analystUpdatedAt = updatedAt > 0 ? updatedAt : undefined
      lifecycle.analystAuditHash = String(analystCase.audit_hash ?? '')
      lifecycle.nluEscalated = lifecycle.nluEscalated || lifecycle.analystStatus === 'pending_review'
      const terminalRows: Array<Omit<BackendTerminalEntry, 'seq' | 'id'> & { id?: string }> = [{
        id: `analyst:${txnId}:${lifecycle.analystStatus || 'pending'}:${eventTimestamp || 'live'}`,
        timestamp: eventTimestamp || nowSeconds(),
        source: 'analyst',
        tone: lifecycle.analystStatus === 'approved' || lifecycle.analystStatus === 'resolved' ? 'success' : 'warn',
        title: `analyst decision ${lifecycle.analystStatus || 'pending'}`,
        detail: [
          `txn=${shortTerminal(txnId, 14)}`,
          lifecycle.analystCaseId ? `case=${shortTerminal(lifecycle.analystCaseId, 14)}` : '',
          lifecycle.analystReason ? `reason=${lifecycle.analystReason.slice(0, 160)}` : '',
          lifecycle.analystAuditHash ? `audit=${shortTerminal(lifecycle.analystAuditHash, 14)}` : '',
        ].filter(Boolean).join(' | '),
        txnId,
        stage: 'analyst_decision',
      }]

      if (lifecycle.analystStatus && lifecycle.analystStatus !== 'pending_review') {
        addStage(lifecycle, 'verdict', {
          analyst_status: lifecycle.analystStatus,
          analyst_case_id: lifecycle.analystCaseId,
          analyst_reason: lifecycle.analystReason,
          audit_hash: lifecycle.analystAuditHash,
        }, undefined, eventTimestamp)
      }

      const idx = orderedIds.indexOf(txnId)
      if (idx > 0) {
        orderedIds.splice(idx, 1)
        orderedIds.unshift(txnId)
      }

      pruneEvents(events, orderedIds)
      if (String(data.type ?? '') === 'case_resolution') {
        const fanout = data.fanout && typeof data.fanout === 'object'
          ? data.fanout as Record<string, unknown>
          : {}
        const queryGroups = Array.isArray(fanout.query_groups)
          ? fanout.query_groups.map(String)
          : []
        const resolution: ResolutionFanoutActivity = {
          txnId,
          caseId: String(data.case_id ?? analystCase.case_id ?? analystCase.ack_id ?? ''),
          nodeId: String(data.node_id ?? analystCase.node_id ?? ''),
          status: String(data.status ?? analystCase.status ?? ''),
          nodeStatus: String(data.node_status ?? ''),
          auditHash: String(data.audit_hash ?? analystCase.audit_hash ?? ''),
          activityLifecycle: Boolean(fanout.activity_lifecycle),
          graphNodeStatus: Boolean(fanout.graph_node_status),
          queryGroups,
          updatedAt: timestampOrZero(data.timestamp) || eventTimestamp,
        }
        terminalRows.push({
          id: `resolution-fanout:${resolution.caseId}:${resolution.txnId}:${resolution.updatedAt || 'live'}`,
          timestamp: resolution.updatedAt || nowSeconds(),
          source: 'decision',
          tone: resolution.status === 'approved' || resolution.status === 'resolved' ? 'success' : 'info',
          title: 'resolution fan-out propagated',
          detail: [
            `txn=${shortTerminal(resolution.txnId, 14)}`,
            `case=${shortTerminal(resolution.caseId, 14)}`,
            `node_status=${resolution.nodeStatus || 'n/a'}`,
            `groups=${resolution.queryGroups.join(',') || 'n/a'}`,
            resolution.auditHash ? `audit=${shortTerminal(resolution.auditHash, 14)}` : '',
          ].filter(Boolean).join(' | '),
          txnId: resolution.txnId,
          stage: 'resolution_fanout',
        })
        const rest = state.resolutionFanouts.filter((item) => (
          item.caseId !== resolution.caseId || item.txnId !== resolution.txnId
        ))
        return {
          events,
          orderedIds,
          resolutionFanouts: [resolution, ...rest].slice(0, 50),
          ...appendTerminalEntries(state, terminalRows),
        }
      }

      return { events, orderedIds, ...appendTerminalEntries(state, terminalRows) }
    }),
}))

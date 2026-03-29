// ============================================================================
// Activity Store -- Per-event lifecycle tracking from SSE streams
// ============================================================================

import { create } from 'zustand'

const MAX_TRACKED_EVENTS = 200

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
  totalDurationMs?: number
  nluFindingsCount?: number
  nluEscalated?: boolean
  consensusScores?: { ml: number; gnn: number; graph: number }
  // Pipeline batch info
  pipelineConsumers?: Array<{ consumer: string; success: boolean; duration_ms: number; error?: string }>
}

interface ActivityState {
  events: Map<string, EventLifecycle>
  orderedIds: string[] // most-recent-first
  trackedEventId: string | null

  // Actions
  setTrackedEventId: (id: string | null) => void
  onGraphBatchUpdate: (edges: Array<{
    data: {
      id: string
      source: string
      target: string
      amount_paisa: number
      fraud_label: number
      timestamp: number
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
    total_duration_ms?: number
    nlu_findings_count?: number
    nlu_escalated?: boolean
    content?: string
    tool_name?: string
    success?: boolean
    duration_ms?: number
    output_summary?: string
  }) => void
  onCBActivity: (
    data: { type?: string; node_id?: string; order?: Record<string, unknown> },
    txnId?: string,
  ) => void
  onRiskScoreActivity: (data: {
    txn_id: string
    risk_score: number
    tier: string
    top_features?: string[]
  }) => void
  onPipelineStage: (data: {
    type: string
    batch_id: number
    event_count: number
    transactions: number
    auth_events: number
    interbank_messages: number
    consumers: Array<{ consumer: string; success: boolean; duration_ms: number; error?: string }>
  }) => void
  onPipelineStageComplete: (data: {
    type: string
    stage: PipelineStage
    txn_id?: string
    txn_ids?: string[]
    duration_ms?: number
    [key: string]: unknown
  }) => void
}

function addStage(lifecycle: EventLifecycle, stage: PipelineStage, meta?: Record<string, unknown>, durationMs?: number) {
  if (lifecycle.stages.some((s) => s.stage === stage)) return
  lifecycle.stages.push({ stage, timestamp: Date.now() / 1000, durationMs, meta })
}

function pruneEvents(events: Map<string, EventLifecycle>, orderedIds: string[]) {
  while (orderedIds.length > MAX_TRACKED_EVENTS) {
    const removed = orderedIds.pop()
    if (removed) events.delete(removed)
  }
}

export const useActivityStore = create<ActivityState>((set) => ({
  events: new Map(),
  orderedIds: [],
  trackedEventId: null,

  setTrackedEventId: (id) => set({ trackedEventId: id }),

  onGraphBatchUpdate: (edges) =>
    set((state) => {
      const events = new Map(state.events)
      const orderedIds = [...state.orderedIds]
      let changed = false

      for (const edge of edges) {
        const { id: txnId, source, target, amount_paisa, fraud_label, timestamp } = edge.data
        if (!txnId) continue

        if (!events.has(txnId)) {
          events.set(txnId, {
            txnId,
            sender: source,
            receiver: target,
            amountPaisa: amount_paisa,
            fraudLabel: fraud_label ?? 0,
            attackLabel: '',
            scenarioId: '',
            firstSeen: timestamp || Date.now() / 1000,
            stages: [{ stage: 'ingested', timestamp: timestamp || Date.now() / 1000 }],
          })
          orderedIds.unshift(txnId)
          changed = true
        }
      }

      if (!changed) return {}
      pruneEvents(events, orderedIds)
      return { events, orderedIds }
    }),

  onAgentEvent: (data) =>
    set((state) => {
      if (!data.txn_id) return {}
      const events = new Map(state.events)
      const orderedIds = [...state.orderedIds]

      let lifecycle = events.get(data.txn_id)
      if (!lifecycle) {
        lifecycle = {
          txnId: data.txn_id,
          sender: '',
          receiver: '',
          amountPaisa: 0,
          fraudLabel: 0,
          attackLabel: '',
          scenarioId: '',
          firstSeen: Date.now() / 1000,
          stages: [],
        }
        events.set(data.txn_id, lifecycle)
        orderedIds.unshift(data.txn_id)
      }

      if (data.type === 'thinking_step' || data.type === 'tool_call') {
        addStage(lifecycle, 'llm_started', {
          type: data.type,
          iteration: data.iteration,
          max_iterations: data.max_iterations,
          tool_name: data.tool_name,
          content: data.content?.slice(0, 200),
          output_summary: data.output_summary?.slice(0, 200),
        })
      } else if (data.type === 'verdict') {
        addStage(lifecycle, 'llm_started')
        addStage(lifecycle, 'verdict')
        lifecycle.verdict = data.verdict
        lifecycle.confidence = data.confidence
        lifecycle.fraudTypology = data.fraud_typology
        lifecycle.reasoningSummary = data.reasoning_summary
        lifecycle.evidenceCited = data.evidence_cited
        lifecycle.recommendedAction = data.recommended_action
        lifecycle.thinkingSteps = data.thinking_steps
        lifecycle.toolsUsed = data.tools_used
        lifecycle.totalDurationMs = data.total_duration_ms
        lifecycle.nluFindingsCount = data.nlu_findings_count
        lifecycle.nluEscalated = data.nlu_escalated
      }

      // Bump to top of ordered list
      const idx = orderedIds.indexOf(data.txn_id)
      if (idx > 0) {
        orderedIds.splice(idx, 1)
        orderedIds.unshift(data.txn_id)
      }

      pruneEvents(events, orderedIds)
      return { events, orderedIds }
    }),

  onCBActivity: (data, txnId) =>
    set((state) => {
      const nodeId = data.node_id
      if (!nodeId && !txnId) return {}

      const events = new Map(state.events)
      const targetId = txnId ?? nodeId!
      const lifecycle = events.get(targetId)
      if (!lifecycle) return {}

      addStage(lifecycle, 'cb_evaluated', {
        type: data.type,
        nodeId: data.node_id,
        ...(data.order ?? {}),
      })

      if (data.order) {
        const order = data.order as Record<string, unknown>
        if (order.ml_score !== undefined || order.gnn_score !== undefined) {
          lifecycle.consensusScores = {
            ml: (order.ml_score as number) ?? 0,
            gnn: (order.gnn_score as number) ?? 0,
            graph: (order.graph_score as number) ?? 0,
          }
        }
      }

      return { events }
    }),

  onRiskScoreActivity: (data) =>
    set((state) => {
      if (!data.txn_id) return {}
      const events = new Map(state.events)
      const lifecycle = events.get(data.txn_id)
      if (!lifecycle) return {}

      addStage(lifecycle, 'ml_scored', { top_features: data.top_features })
      lifecycle.riskScore = data.risk_score
      lifecycle.riskTier = data.tier
      lifecycle.topFeatures = data.top_features

      return { events }
    }),

  onPipelineStage: (data) =>
    set((state) => {
      // Pipeline batch dispatched — update tracked events with consumer timing info
      // We don't have specific txn_ids from this event, but we store the latest batch info
      // for display in the pipeline stage bar
      const events = new Map(state.events)
      // Tag the most recent events with pipeline consumer data
      const recentIds = state.orderedIds.slice(0, Math.min(data.event_count, 20))
      for (const txnId of recentIds) {
        const lifecycle = events.get(txnId)
        if (lifecycle && !lifecycle.pipelineConsumers) {
          lifecycle.pipelineConsumers = data.consumers
          addStage(lifecycle, 'pipeline_dispatched', {
            batch_id: data.batch_id,
            event_count: data.event_count,
            transactions: data.transactions,
            auth_events: data.auth_events,
            interbank_messages: data.interbank_messages,
          })
        }
      }
      return { events }
    }),

  onPipelineStageComplete: (data) =>
    set((state) => {
      const events = new Map(state.events)
      let changed = false

      // Single-txn stage event (graph_investigated, cb_evaluated)
      if (data.txn_id) {
        const lifecycle = events.get(data.txn_id)
        if (lifecycle) {
          const { type: _t, stage, txn_id: _tid, duration_ms, ...meta } = data
          addStage(lifecycle, stage, meta as Record<string, unknown>, duration_ms)
          changed = true
        }
      }

      // Batch stage event (ml_scored — txn_ids array)
      if (data.txn_ids && Array.isArray(data.txn_ids)) {
        for (const txnId of data.txn_ids as string[]) {
          const lifecycle = events.get(txnId)
          if (lifecycle) {
            const { type: _t, stage, txn_ids: _tids, duration_ms, ...meta } = data
            addStage(lifecycle, stage, meta as Record<string, unknown>, duration_ms)
            changed = true
          }
        }
      }

      return changed ? { events } : {}
    }),
}))

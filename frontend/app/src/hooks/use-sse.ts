// ============================================================================
// SSE Hook -- Subscribes to EventSource, dispatches by channel to Zustand
// Throttled graph batch processing to prevent render storms
// ============================================================================

import { useEffect, useRef, useCallback } from 'react'
import { useQueryClient } from '@tanstack/react-query'
import { SSEManager } from '@/lib/sse-manager'
import { useDashboardStore } from '@/stores/use-dashboard-store'
import { useSimulationStore } from '@/stores/use-simulation-store'
import { useActivityStore } from '@/stores/use-activity-store'
import { useUIStore } from '@/stores/use-ui-store'
import { useAnalyticsStore } from '@/stores/use-analytics-store'
import type {
  SSEEnvelope,
  SSEGraphBatchUpdate,
  SSEGraphNodeStatusChanged,
  SSEAgentData,
  SSESystemData,
  SSESimulationData,
  SSERiskScoreAlert,
  SSEPipelineData,
  CytoNode,
  CytoEdge,
} from '@/lib/types'

// ---------------------------------------------------------------------------
// Graph-event accumulator — merges rapid SSE graph batches and flushes to the
// Zustand store at most once every GRAPH_FLUSH_MS to prevent a render storm.
// ---------------------------------------------------------------------------
const GRAPH_FLUSH_MS = 2_000
const QUERY_INVALIDATION_THROTTLE_MS = 4_000
const ANALYST_SYNC_QUERY_KEYS = [
  ['escalations'],
  ['recent-blocks'],
  ['snapshot'],
  ['topology'],
  ['verdicts'],
  ['case-trace'],
  ['risk-distribution'],
  ['fraud-typology'],
  ['velocity-trends'],
  ['temporal-heatmap'],
  ['threat-summary'],
] as const
const TRANSACTION_DECISION_SYNC_QUERY_KEYS = [
  ['escalations'],
  ['recent-blocks'],
  ['snapshot'],
  ['topology'],
  ['verdicts'],
  ['case-trace'],
  ['risk-distribution'],
  ['fraud-typology'],
  ['velocity-trends'],
  ['temporal-heatmap'],
  ['threat-summary'],
  ['countermeasure-proposals'],
  ['circuit-breaker'],
  ['fraud'],
] as const
const COUNTERMEASURE_SYNC_QUERY_KEYS = [
  ['countermeasure-proposals'],
  ['event-lab-run'],
  ['event-lab-explainability'],
  ['recent-blocks'],
  ['active-scenarios'],
  ['scenario-history'],
  ['snapshot'],
  ['topology'],
  ['circuit-breaker'],
  ['verdicts'],
  ['case-trace'],
  ['risk-distribution'],
  ['fraud-typology'],
  ['velocity-trends'],
  ['temporal-heatmap'],
  ['threat-summary'],
  ['global-importance'],
  ['drift-status'],
  ['ps3-readiness'],
  ['fraud'],
] as const
const GRAPH_SYNC_QUERY_KEYS = [
  ['topology'],
  ['snapshot'],
  ['risk-distribution'],
  ['fraud-typology'],
  ['velocity-trends'],
  ['temporal-heatmap'],
  ['threat-summary'],
  ['fraud'],
] as const
const RISK_SCORE_SYNC_QUERY_KEYS = [
  ['verdicts'],
  ['escalations'],
  ['snapshot'],
  ['risk-distribution'],
  ['fraud-typology'],
  ['velocity-trends'],
  ['temporal-heatmap'],
  ['threat-summary'],
  ['global-importance'],
  ['drift-status'],
  ['case-trace'],
  ['fraud'],
] as const
const CIRCUIT_BREAKER_SYNC_QUERY_KEYS = [
  ['circuit-breaker'],
  ['snapshot'],
  ['recent-blocks'],
  ['topology'],
  ['case-trace'],
  ['verdicts'],
  ['fraud'],
] as const
const SIMULATION_SYNC_QUERY_KEYS = [
  ['active-scenarios'],
  ['scenario-history'],
  ['snapshot'],
  ['topology'],
  ['verdicts'],
  ['recent-blocks'],
  ['risk-distribution'],
  ['fraud-typology'],
  ['velocity-trends'],
  ['temporal-heatmap'],
  ['threat-summary'],
  ['fraud'],
] as const
const EVENT_LAB_SYNC_QUERY_KEYS = [
  ['event-lab-run'],
  ['event-lab-explainability'],
  ['countermeasure-proposals'],
  ['snapshot'],
  ['topology'],
  ['recent-blocks'],
  ['active-scenarios'],
  ['scenario-history'],
  ['verdicts'],
  ['risk-distribution'],
  ['fraud-typology'],
  ['velocity-trends'],
  ['temporal-heatmap'],
  ['threat-summary'],
  ['fraud'],
] as const

interface PendingGraphBatch {
  nodes: CytoNode[]
  edges: CytoEdge[]
}

export function useSSE() {
  const queryClient = useQueryClient()
  const managerRef = useRef<SSEManager | null>(null)
  const pendingGraphRef = useRef<PendingGraphBatch>({ nodes: [], edges: [] })
  const flushTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  const lastQueryInvalidationRef = useRef<Record<string, number>>({})

  const addGraphElements = useDashboardStore((s) => s.addGraphElements)
  const updateNodeStatus = useDashboardStore((s) => s.updateNodeStatus)
  const addAgentEntry = useDashboardStore((s) => s.addAgentEntry)
  const setSystemTelemetry = useDashboardStore((s) => s.setSystemTelemetry)
  const handleSimulationSSE = useSimulationStore((s) => s.handleSSEEvent)
  const setConnected = useUIStore((s) => s.setConnected)
  const onGraphBatchUpdate = useActivityStore((s) => s.onGraphBatchUpdate)
  const onAgentEvent = useActivityStore((s) => s.onAgentEvent)
  const onCBActivity = useActivityStore((s) => s.onCBActivity)
  const onRiskScoreActivity = useActivityStore((s) => s.onRiskScoreActivity)
  const onPipelineStage = useActivityStore((s) => s.onPipelineStage)
  const onPipelineStageComplete = useActivityStore((s) => s.onPipelineStageComplete)
  const onEventLabActivity = useActivityStore((s) => s.onEventLabActivity)
  const onCountermeasureActivity = useActivityStore((s) => s.onCountermeasureActivity)
  const onAnalystActivity = useActivityStore((s) => s.onAnalystActivity)
  const ingestAnalyticsGraphBatch = useAnalyticsStore((s) => s.ingestGraphBatch)
  const ingestAnalyticsSystemSnapshot = useAnalyticsStore((s) => s.ingestSystemSnapshot)
  const ingestAnalyticsRiskScore = useAnalyticsStore((s) => s.ingestRiskScore)
  const ingestAnalyticsAgentEvent = useAnalyticsStore((s) => s.ingestAgentEvent)

  const invalidateQueryGroup = useCallback(
    (group: string, queryKeys: readonly (readonly unknown[])[]) => {
      const now = Date.now()
      queryKeys.forEach((queryKey) => {
        const key = `${group}:${JSON.stringify(queryKey)}`
        const queryKeyOnly = JSON.stringify(queryKey)
        const lastInvalidated = Math.max(
          lastQueryInvalidationRef.current[key] ?? 0,
          lastQueryInvalidationRef.current[queryKeyOnly] ?? 0,
        )
        if (now - lastInvalidated < QUERY_INVALIDATION_THROTTLE_MS) return
        lastQueryInvalidationRef.current[key] = now
        lastQueryInvalidationRef.current[queryKeyOnly] = now
        void queryClient.invalidateQueries({ queryKey })
      })
    },
    [queryClient],
  )

  // Flush accumulated graph batches to the store in one shot
  const flushGraphBatch = useCallback(() => {
    const pending = pendingGraphRef.current
    if (pending.nodes.length === 0 && pending.edges.length === 0) return
    const merged: SSEGraphBatchUpdate = {
      type: 'batch_update',
      nodes: pending.nodes,
      edges: pending.edges,
    }
    addGraphElements(merged)
    if (pending.edges.length) onGraphBatchUpdate(pending.edges)
    ingestAnalyticsGraphBatch(pending.nodes, pending.edges)
    invalidateQueryGroup('graph', GRAPH_SYNC_QUERY_KEYS)
    pendingGraphRef.current = { nodes: [], edges: [] }
  }, [addGraphElements, ingestAnalyticsGraphBatch, invalidateQueryGroup, onGraphBatchUpdate])

  // Accumulate a graph batch and schedule a flush if one isn't pending
  const enqueueGraphBatch = useCallback(
    (batch: SSEGraphBatchUpdate) => {
      const pending = pendingGraphRef.current
      if (batch.nodes?.length) pending.nodes.push(...batch.nodes)
      if (batch.edges?.length) pending.edges.push(...batch.edges)
      if (!flushTimerRef.current) {
        flushTimerRef.current = setTimeout(() => {
          flushTimerRef.current = null
          flushGraphBatch()
        }, GRAPH_FLUSH_MS)
      }
    },
    [flushGraphBatch],
  )

  useEffect(() => {
    const handleEvent = (envelope: SSEEnvelope) => {
      const { channel, data } = envelope

      switch (channel) {
        case 'graph': {
          const gd = data as unknown
          if ((gd as { type: string }).type === 'batch_update') {
            const batch = gd as SSEGraphBatchUpdate
            enqueueGraphBatch(batch)
          } else if ((gd as { type: string }).type === 'node_status_changed') {
            const nsc = gd as SSEGraphNodeStatusChanged
            updateNodeStatus(nsc.node_id, nsc.status)
          }
          break
        }
        case 'agent': {
          const agentData = data as unknown as SSEAgentData
          addAgentEntry(agentData, envelope.timestamp)
          onAgentEvent(agentData as unknown as Parameters<typeof onAgentEvent>[0], envelope.timestamp)
          ingestAnalyticsAgentEvent(agentData)
          break
        }
        case 'circuit_breaker': {
          // Individual node_frozen/node_unfrozen events — update count incrementally.
          // Aggregate counts are refreshed via system telemetry at 1 Hz.
          const cbd = data as { type?: string; node_id?: string; order?: Record<string, unknown> }
          if (cbd.type === 'node_frozen') {
            const store = useDashboardStore.getState()
            useDashboardStore.setState({ frozenCount: store.frozenCount + 1 })
          } else if (cbd.type === 'node_unfrozen') {
            const store = useDashboardStore.getState()
            useDashboardStore.setState({ frozenCount: Math.max(0, store.frozenCount - 1) })
          }
          onCBActivity(cbd, undefined, envelope.timestamp)
          invalidateQueryGroup('circuit_breaker', CIRCUIT_BREAKER_SYNC_QUERY_KEYS)
          break
        }
        case 'system': {
          const sd = data as unknown as SSESystemData
          if (sd.type === 'telemetry') {
            setSystemTelemetry(data as Record<string, unknown>)
            ingestAnalyticsSystemSnapshot(data as Record<string, unknown>)
          }
          break
        }
        case 'simulation':
          handleSimulationSSE(data as unknown as SSESimulationData, envelope.timestamp)
          invalidateQueryGroup('simulation', SIMULATION_SYNC_QUERY_KEYS)
          break
        case 'risk_scores': {
          const rsd = data as unknown as SSERiskScoreAlert
          if (rsd.type === 'alert_scored' && rsd.txn_id) {
            onRiskScoreActivity(rsd, envelope.timestamp)
            ingestAnalyticsRiskScore(rsd, envelope.timestamp)
            invalidateQueryGroup('risk_scores', RISK_SCORE_SYNC_QUERY_KEYS)
          }
          break
        }
        case 'pipeline': {
          const pd = data as unknown as SSEPipelineData
          if (pd.type === 'batch_dispatched') {
            onPipelineStage(pd, envelope.timestamp)
          } else if (pd.type === 'stage_complete') {
            onPipelineStageComplete(pd as unknown as Parameters<typeof onPipelineStageComplete>[0], envelope.timestamp)
          }
          break
        }
        case 'event_lab':
          onEventLabActivity(data as Record<string, unknown>)
          invalidateQueryGroup('event_lab', EVENT_LAB_SYNC_QUERY_KEYS)
          break
        case 'countermeasure':
          onCountermeasureActivity(data as Record<string, unknown>)
          COUNTERMEASURE_SYNC_QUERY_KEYS.forEach((queryKey) => {
            void queryClient.invalidateQueries({ queryKey })
          })
          break
        case 'analyst':
          onAnalystActivity(data as Record<string, unknown>, envelope.timestamp)
          ANALYST_SYNC_QUERY_KEYS.forEach((queryKey) => {
            void queryClient.invalidateQueries({ queryKey })
          })
          break
        case 'transaction_decision': {
          const decision = data as Record<string, unknown>
          onAnalystActivity(decision, envelope.timestamp)
          const nodeId = typeof decision.node_id === 'string' ? decision.node_id : undefined
          const nodeStatus = typeof decision.node_status === 'string' ? decision.node_status : undefined
          if (nodeId && nodeStatus) {
            updateNodeStatus(nodeId, nodeStatus)
          }
          TRANSACTION_DECISION_SYNC_QUERY_KEYS.forEach((queryKey) => {
            void queryClient.invalidateQueries({ queryKey })
          })
          break
        }
      }
    }

    const manager = new SSEManager(
      '/api/v1/stream/events',
      handleEvent,
      setConnected,
    )
    managerRef.current = manager
    manager.connect()

    return () => {
      manager.disconnect()
      managerRef.current = null
      // Flush any remaining graph data and cancel pending timer
      if (flushTimerRef.current) {
        clearTimeout(flushTimerRef.current)
        flushTimerRef.current = null
      }
      flushGraphBatch()
    }
  }, [
    enqueueGraphBatch,
    updateNodeStatus,
    addAgentEntry,
    setSystemTelemetry,
    handleSimulationSSE,
    setConnected,
    onAgentEvent,
    onCBActivity,
    onRiskScoreActivity,
    onPipelineStage,
    onPipelineStageComplete,
    onEventLabActivity,
    onCountermeasureActivity,
    onAnalystActivity,
    invalidateQueryGroup,
    queryClient,
    ingestAnalyticsGraphBatch,
    ingestAnalyticsSystemSnapshot,
    ingestAnalyticsRiskScore,
    ingestAnalyticsAgentEvent,
    flushGraphBatch,
  ])
}

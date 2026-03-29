// ============================================================================
// SSE Hook -- Subscribes to EventSource, dispatches by channel to Zustand
// Throttled graph batch processing to prevent render storms
// ============================================================================

import { useEffect, useRef, useCallback } from 'react'
import { SSEManager } from '@/lib/sse-manager'
import { useDashboardStore } from '@/stores/use-dashboard-store'
import { useSimulationStore } from '@/stores/use-simulation-store'
import { useActivityStore } from '@/stores/use-activity-store'
import { useUIStore } from '@/stores/use-ui-store'
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

interface PendingGraphBatch {
  nodes: CytoNode[]
  edges: CytoEdge[]
}

export function useSSE() {
  const managerRef = useRef<SSEManager | null>(null)
  const pendingGraphRef = useRef<PendingGraphBatch>({ nodes: [], edges: [] })
  const flushTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null)

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
    pendingGraphRef.current = { nodes: [], edges: [] }
  }, [addGraphElements, onGraphBatchUpdate])

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
          addAgentEntry(agentData)
          onAgentEvent(agentData as unknown as Parameters<typeof onAgentEvent>[0])
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
          onCBActivity(cbd)
          break
        }
        case 'system': {
          const sd = data as unknown as SSESystemData
          if (sd.type === 'telemetry') {
            setSystemTelemetry(data as Record<string, unknown>)
          }
          break
        }
        case 'simulation':
          handleSimulationSSE(data as unknown as SSESimulationData)
          break
        case 'risk_scores': {
          const rsd = data as unknown as SSERiskScoreAlert
          if (rsd.type === 'alert_scored' && rsd.txn_id) {
            onRiskScoreActivity(rsd)
          }
          break
        }
        case 'pipeline': {
          const pd = data as unknown as SSEPipelineData
          if (pd.type === 'batch_dispatched') {
            onPipelineStage(pd)
          } else if (pd.type === 'stage_complete') {
            onPipelineStageComplete(pd as unknown as Parameters<typeof onPipelineStageComplete>[0])
          }
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
    flushGraphBatch,
  ])
}

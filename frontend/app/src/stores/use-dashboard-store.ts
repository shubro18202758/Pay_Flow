// ============================================================================
// Dashboard Store -- Graph elements, system telemetry, agent investigation trace, circuit breaker
// ============================================================================

import { create } from 'zustand'
import type {
  CytoNode,
  CytoEdge,
  OrchestratorMetrics,
  HardwareSnapshot,
  AgentLogEntry,
  SSEAgentData,
  SSEGraphBatchUpdate,
  FreezeOrder,
  GraphMetrics,
  AgentMetrics,
  ThreatSimulationSnapshot,
  VerdictBlock,
} from '@/lib/types'
import {
  sanitizeEvidenceList,
  sanitizeOptionalEvidenceText,
  sanitizePublicTraceText,
} from '@/lib/evidence-sanitizer'

const MAX_EDGES = 900
const MAX_COT_ENTRIES = 500
const RELAYOUT_NODE_THRESHOLD = 300

interface GraphSummary {
  fraudEdges: number
  suspiciousNodes: number
  edgeCount: number
  nodeCount: number
}

interface DashboardState {
  // Graph
  graphNodes: CytoNode[]
  graphEdges: CytoEdge[]
  graphSummary: GraphSummary
  shouldRelayout: boolean

  // System telemetry
  orchestrator: OrchestratorMetrics | null
  hardware: HardwareSnapshot | null
  graphMetrics: GraphMetrics | null
  graphSize: { nodes: number; edges: number } | null
  agentMetrics: AgentMetrics | null
  threatSimulation: ThreatSimulationSnapshot | null

  // Circuit breaker
  freezeOrders: FreezeOrder[]
  frozenCount: number
  pendingAlerts: number
  bannedDevices: number
  routingPausedNodes: number

  // Agent investigation trace log
  agentLog: AgentLogEntry[]

  // Actions
  setInitialTopology: (nodes: CytoNode[], edges: CytoEdge[]) => void
  addGraphElements: (batch: SSEGraphBatchUpdate) => void
  updateNodeStatus: (nodeId: string, status: string) => void
  setSystemTelemetry: (data: Record<string, unknown>) => void
  addAgentEntry: (data: SSEAgentData, serverTimestamp?: number) => void
  hydrateVerdicts: (verdicts: VerdictBlock[]) => void
  setCircuitBreaker: (
    orders: FreezeOrder[],
    snapshot: Record<string, unknown>,
    agentBreaker: Record<string, unknown>,
  ) => void
  clearRelayout: () => void
}

let _cotCounter = 0

function summarizeGraph(nodes: CytoNode[], edges: CytoEdge[]): GraphSummary {
  let fraudEdges = 0
  let suspiciousNodes = 0

  for (const edge of edges) {
    if ((edge.data.fraud_label ?? 0) > 0) fraudEdges += 1
  }

  for (const node of nodes) {
    if (node.data.status === 'suspicious') suspiciousNodes += 1
  }

  return {
    fraudEdges,
    suspiciousNodes,
    edgeCount: edges.length,
    nodeCount: nodes.length,
  }
}

function buildAgentEntryId(data: SSEAgentData, timestamp: number, suffix = ''): string {
  if (data.type === 'thinking_step') {
    return `thinking:${data.txn_id}:${data.iteration}:${suffix || timestamp}`
  }
  if (data.type === 'tool_call') {
    const durationKey =
      typeof data.duration_ms === 'number' && Number.isFinite(data.duration_ms) && data.duration_ms >= 0
        ? Math.round(data.duration_ms)
        : 'duration-na'
    return `tool:${data.txn_id}:${data.tool_name}:${durationKey}:${suffix || timestamp}`
  }
  return `verdict:${data.txn_id}:${data.verdict}:${suffix || timestamp}`
}

function timestampOrZero(raw: unknown): number {
  const n = Number(raw)
  return Number.isFinite(n) && n > 0 ? n : 0
}

function sanitizeAgentData(data: SSEAgentData): SSEAgentData {
  if (data.type === 'thinking_step') {
    const publicContent = sanitizePublicTraceText(data.public_content ?? data.content)
    return {
      ...data,
      content: publicContent,
      public_content: publicContent,
    }
  }
  if (data.type !== 'verdict') return data
  return {
    ...data,
    reasoning_summary: sanitizeOptionalEvidenceText(data.reasoning_summary) ?? '',
    evidence_cited: sanitizeEvidenceList(data.evidence_cited),
    evidence: sanitizeEvidenceList(data.evidence),
  }
}

export const useDashboardStore = create<DashboardState>((set) => ({
  // -- Initial state --
  graphNodes: [],
  graphEdges: [],
  graphSummary: { fraudEdges: 0, suspiciousNodes: 0, edgeCount: 0, nodeCount: 0 },
  shouldRelayout: false,
  orchestrator: null,
  hardware: null,
  graphMetrics: null,
  graphSize: null,
  agentMetrics: null,
  threatSimulation: null,
  freezeOrders: [],
  frozenCount: 0,
  pendingAlerts: 0,
  bannedDevices: 0,
  routingPausedNodes: 0,
  agentLog: [],

  // -- Actions --

  setInitialTopology: (nodes, edges) => {
    const cappedEdges = edges.slice(-MAX_EDGES)
    return set({
      graphNodes: nodes,
      graphEdges: cappedEdges,
      graphSummary: summarizeGraph(nodes, cappedEdges),
      shouldRelayout: nodes.length <= RELAYOUT_NODE_THRESHOLD,
    })
  },

  addGraphElements: (batch) =>
    set((state) => {
      const newNodes = [...state.graphNodes]
      const existingNodeIds = new Set(newNodes.map((n) => n.data.id))

      for (const node of batch.nodes ?? []) {
        if (existingNodeIds.has(node.data.id)) {
          const idx = newNodes.findIndex((n) => n.data.id === node.data.id)
          if (idx >= 0) newNodes[idx] = node
        } else {
          newNodes.push(node)
          existingNodeIds.add(node.data.id)
        }
      }

      let newEdges = [...state.graphEdges, ...(batch.edges ?? [])]

      // Cap edges at MAX_EDGES, remove oldest by timestamp
      if (newEdges.length > MAX_EDGES) {
        newEdges.sort((a, b) => a.data.timestamp - b.data.timestamp)
        newEdges = newEdges.slice(-MAX_EDGES)
      }

      return {
        graphNodes: newNodes,
        graphEdges: newEdges,
        graphSummary: summarizeGraph(newNodes, newEdges),
        shouldRelayout: newNodes.length <= RELAYOUT_NODE_THRESHOLD,
      }
    }),

  updateNodeStatus: (nodeId, status) =>
    set((state) => {
      const graphNodes = state.graphNodes.map((n) =>
        n.data.id === nodeId
          ? { data: { ...n.data, status: status as CytoNode['data']['status'] } }
          : n,
      )
      return {
        graphNodes,
        graphSummary: summarizeGraph(graphNodes, state.graphEdges),
      }
    }),

  setSystemTelemetry: (data) =>
    set({
      orchestrator: (data.orchestrator as OrchestratorMetrics) ?? null,
      hardware: (data.hardware as HardwareSnapshot) ?? null,
      graphMetrics: (data.graph as { metrics: GraphMetrics })?.metrics ?? null,
      graphSize: (data.graph as { graph: { nodes: number; edges: number } })?.graph ?? null,
      agentMetrics: (data.agent as { metrics: AgentMetrics })?.metrics ?? null,
      threatSimulation: (data.threat_simulation as ThreatSimulationSnapshot) ?? null,
      // Circuit breaker aggregate counts from telemetry snapshot
      ...(data.circuit_breaker ? {
        frozenCount: (data.circuit_breaker as Record<string, unknown>).frozen_count as number ?? 0,
        pendingAlerts: (data.circuit_breaker as Record<string, unknown>).pending_alerts as number ?? 0,
      } : {}),
    }),

  addAgentEntry: (data, serverTimestamp) =>
    set((state) => {
      const safeData = sanitizeAgentData(data)
      const timestamp = timestampOrZero(serverTimestamp)
      const entry: AgentLogEntry = {
        id: buildAgentEntryId(safeData, timestamp, `${++_cotCounter}`),
        timestamp,
        txn_id: safeData.txn_id ?? '',
        type:
          safeData.type === 'thinking_step'
            ? 'thinking'
            : safeData.type === 'tool_call'
              ? 'tool_call'
              : 'verdict',
        data: safeData,
      }

      if (state.agentLog.some((existing) => existing.id === entry.id)) {
        return {}
      }

      const log = [...state.agentLog, entry]
      return {
        agentLog: log.length > MAX_COT_ENTRIES ? log.slice(-MAX_COT_ENTRIES) : log,
      }
    }),

  hydrateVerdicts: (verdicts) =>
    set((state) => {
      const existingIds = new Set(state.agentLog.map((entry) => entry.id))
      const hydrated: AgentLogEntry[] = []

      for (const [index, block] of verdicts.entries()) {
        const payload = block.payload
        if (!payload) continue

        const timestamp = timestampOrZero(block.timestamp)
        const verdictData = sanitizeAgentData({
          type: 'verdict' as const,
          txn_id: payload.txn_id,
          node_id: payload.node_id ?? '',
          verdict: payload.verdict,
          confidence: payload.confidence,
          fraud_typology: payload.fraud_typology ?? '',
          reasoning_summary: payload.reasoning_summary ?? '',
          recommended_action: payload.recommended_action,
          thinking_steps: payload.thinking_steps ?? 0,
          tools_used: payload.tools_used ?? [],
          total_duration_ms: payload.total_duration_ms,
          confidence_source: payload.confidence_source,
          llm_parse_status: payload.llm_parse_status,
          model_used: payload.model_used,
        })

        const id = buildAgentEntryId(verdictData, timestamp, `${block.index ?? index}`)
        if (existingIds.has(id)) continue
        existingIds.add(id)

        hydrated.push({
          id,
          timestamp,
          txn_id: payload.txn_id,
          type: 'verdict',
          data: verdictData,
        })
      }

      if (hydrated.length === 0) {
        return {}
      }

      const log = [...state.agentLog, ...hydrated].sort((a, b) => a.timestamp - b.timestamp)
      return {
        agentLog: log.length > MAX_COT_ENTRIES ? log.slice(-MAX_COT_ENTRIES) : log,
      }
    }),

  setCircuitBreaker: (orders, snapshot, agentBreaker) =>
    set({
      freezeOrders: orders,
      frozenCount: (snapshot.frozen_count as number) ?? orders.length,
      pendingAlerts: (snapshot.pending_alerts as number) ?? 0,
      bannedDevices: (agentBreaker.banned_devices as number) ?? 0,
      routingPausedNodes: (agentBreaker.routing_paused_nodes as number) ?? 0,
    }),

  clearRelayout: () => set({ shouldRelayout: false }),
}))

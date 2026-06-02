import { useEffect, useRef } from 'react'
import {
  useActiveScenarios,
  useFraudTypology,
  useRiskDistribution,
  useScenarioHistory,
  useSnapshot,
  useTemporalHeatmap,
  useThreatSummary,
  useTopology,
  useVelocityTrends,
  useVerdicts,
} from '@/hooks/use-api'
import { useDashboardStore } from '@/stores/use-dashboard-store'
import { useSimulationStore } from '@/stores/use-simulation-store'
import { useActivityStore } from '@/stores/use-activity-store'
import { useAnalyticsStore } from '@/stores/use-analytics-store'
import type { SSEAgentVerdict } from '@/lib/types'

export function useDashboardHydration() {
  const hydrateVerdicts = useDashboardStore((s) => s.hydrateVerdicts)
  const setSystemTelemetry = useDashboardStore((s) => s.setSystemTelemetry)
  const setScenarios = useSimulationStore((s) => s.setScenarios)
  const onGraphBatchUpdate = useActivityStore((s) => s.onGraphBatchUpdate)
  const onAgentEvent = useActivityStore((s) => s.onAgentEvent)
  const ingestSystemSnapshot = useAnalyticsStore((s) => s.ingestSystemSnapshot)
  const ingestGraphTopology = useAnalyticsStore((s) => s.ingestGraphTopology)
  const ingestAnalyticsAgentEvent = useAnalyticsStore((s) => s.ingestAgentEvent)
  const hydrateAnalytics = useAnalyticsStore((s) => s.hydrateAnalytics)
  const activityHydratedRef = useRef(false)

  const { data: snapshotData } = useSnapshot()
  const { data: verdictData } = useVerdicts(80)
  const { data: activeData } = useActiveScenarios(true)
  const { data: historyData } = useScenarioHistory()
  const { data: topologyData } = useTopology(300)
  const { data: riskDistribution } = useRiskDistribution()
  const { data: fraudTypology } = useFraudTypology()
  const { data: velocityTrends } = useVelocityTrends()
  const { data: temporalHeatmap } = useTemporalHeatmap()
  const { data: threatSummary } = useThreatSummary()

  useEffect(() => {
    if (snapshotData) {
      setSystemTelemetry(snapshotData as unknown as Record<string, unknown>)
      ingestSystemSnapshot(snapshotData as unknown as Record<string, unknown>)
    }
  }, [snapshotData, setSystemTelemetry, ingestSystemSnapshot])

  useEffect(() => {
    if (verdictData?.verdicts) {
      hydrateVerdicts(verdictData.verdicts)
      verdictData.verdicts.forEach((block) => {
        const payload = block.payload
        if (!payload?.txn_id) return
        const verdictEvent: SSEAgentVerdict = {
          type: 'verdict',
          txn_id: payload.txn_id,
          node_id: payload.node_id ?? '',
          verdict: payload.verdict,
          confidence: payload.confidence,
          fraud_typology: payload.fraud_typology ?? '',
          reasoning_summary: payload.reasoning_summary ?? '',
          evidence_cited: payload.evidence_cited ?? [],
          recommended_action: payload.recommended_action ?? '',
          thinking_steps: payload.thinking_steps ?? 0,
          tools_used: payload.tools_used ?? [],
          total_duration_ms: payload.total_duration_ms ?? undefined,
          confidence_source: payload.confidence_source,
          llm_parse_status: payload.llm_parse_status,
          model_used: payload.model_used,
        }
        onAgentEvent(verdictEvent, block.timestamp)
        ingestAnalyticsAgentEvent(verdictEvent)
      })
    }
  }, [verdictData, hydrateVerdicts, onAgentEvent, ingestAnalyticsAgentEvent])

  useEffect(() => {
    const combined = [
      ...(historyData?.scenarios ?? []),
      ...(activeData?.scenarios ?? []),
    ]

    if (combined.length > 0) {
      setScenarios(combined)
    }
  }, [activeData, historyData, setScenarios])

  // Hydrate activity store from REST topology (one-time, on page load)
  useEffect(() => {
    if (topologyData?.edges && topologyData.edges.length > 0 && !activityHydratedRef.current) {
      activityHydratedRef.current = true
      onGraphBatchUpdate(topologyData.edges)
    }
    if (topologyData?.nodes || topologyData?.edges) {
      ingestGraphTopology(topologyData.nodes ?? [], topologyData.edges ?? [])
    }
  }, [topologyData, onGraphBatchUpdate, ingestGraphTopology])

  useEffect(() => {
    hydrateAnalytics({
      riskDistribution,
      fraudTypology,
      velocityTrends,
      temporalHeatmap,
      threatSummary,
    })
  }, [riskDistribution, fraudTypology, velocityTrends, temporalHeatmap, threatSummary, hydrateAnalytics])
}

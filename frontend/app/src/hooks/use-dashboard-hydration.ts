import { useEffect, useRef } from 'react'
import {
  useActiveScenarios,
  useScenarioHistory,
  useTopology,
  useVerdicts,
} from '@/hooks/use-api'
import { useDashboardStore } from '@/stores/use-dashboard-store'
import { useSimulationStore } from '@/stores/use-simulation-store'
import { useActivityStore } from '@/stores/use-activity-store'

export function useDashboardHydration() {
  const hydrateVerdicts = useDashboardStore((s) => s.hydrateVerdicts)
  const setScenarios = useSimulationStore((s) => s.setScenarios)
  const onGraphBatchUpdate = useActivityStore((s) => s.onGraphBatchUpdate)
  const activityHydratedRef = useRef(false)

  const { data: verdictData } = useVerdicts(80)
  const { data: activeData } = useActiveScenarios(true)
  const { data: historyData } = useScenarioHistory()
  const { data: topologyData } = useTopology(500)

  useEffect(() => {
    if (verdictData?.verdicts) {
      hydrateVerdicts(verdictData.verdicts)
    }
  }, [verdictData, hydrateVerdicts])

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
  }, [topologyData, onGraphBatchUpdate])
}
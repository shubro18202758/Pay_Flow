// ============================================================================
// Simulation Store -- Attack scenarios, simulation SSE events
// ============================================================================

import { create } from 'zustand'
import type {
  ScenarioStatus,
  SSESimulationData,
  SimulationRecentEvent,
} from '@/lib/types'

interface SimulationState {
  scenarios: Map<string, ScenarioStatus>
  recentEvents: SimulationRecentEvent[]
  selectedScenarioId: string | null

  // Actions
  handleSSEEvent: (data: SSESimulationData) => void
  setScenarios: (list: ScenarioStatus[]) => void
  upsertScenario: (scenario: ScenarioStatus) => void
  setSelectedScenario: (scenarioId: string | null) => void
  clearHistory: () => void
}

const MAX_RECENT_EVENTS = 100

export const useSimulationStore = create<SimulationState>((set) => ({
  scenarios: new Map(),
  recentEvents: [],
  selectedScenarioId: null,

  handleSSEEvent: (data) =>
    set((state) => {
      const scenarios = new Map(state.scenarios)

      if (data.type === 'attack_event') {
        const existing = scenarios.get(data.scenario_id)
        if (existing) {
          scenarios.set(data.scenario_id, {
            ...existing,
            events_ingested: data.event_index! + 1,
            progress_pct: data.progress_pct!,
          })
        }

        const recentEvents = [
          ...state.recentEvents,
          {
            id: `${data.scenario_id}:${data.event_index}`,
            scenarioId: data.scenario_id,
            attackType: data.attack_type,
            attackLabel: existing?.attack_label ?? data.attack_type,
            eventType: data.event?.type ?? 'unknown',
            progressPct: data.progress_pct!,
            timestamp: Date.now() / 1000,
            event: data.event,
          },
        ].slice(-MAX_RECENT_EVENTS)

        return {
          scenarios,
          recentEvents,
          selectedScenarioId: state.selectedScenarioId ?? data.scenario_id,
        }
      }

      if (
        data.type === 'simulation_started' ||
        data.type === 'simulation_completed' ||
        data.type === 'simulation_stopped'
      ) {
        const existing = scenarios.get(data.scenario_id)
        const status =
          data.type === 'simulation_started'
            ? 'running'
            : data.type === 'simulation_completed'
              ? 'completed'
              : 'stopped'

        scenarios.set(data.scenario_id, {
          scenario_id: data.scenario_id,
          attack_type: data.attack_type,
          attack_label: data.attack_label ?? existing?.attack_label ?? '',
          status: status as ScenarioStatus['status'],
          events_generated: data.events_generated ?? existing?.events_generated ?? 0,
          events_ingested: data.events_ingested ?? existing?.events_ingested ?? 0,
          progress_pct: existing?.progress_pct ?? 0,
          accounts_involved: data.accounts_involved ?? existing?.accounts_involved ?? [],
          started_at: existing?.started_at ?? Date.now() / 1000,
          stopped_at: status !== 'running' ? Date.now() / 1000 : null,
          elapsed_sec: existing?.elapsed_sec ?? 0,
        })
        return {
          scenarios,
          selectedScenarioId: state.selectedScenarioId ?? data.scenario_id,
        }
      }

      return {}
    }),

  setScenarios: (list) =>
    set((state) => {
      const scenarios = new Map<string, ScenarioStatus>(state.scenarios)
      for (const s of list) {
        scenarios.set(s.scenario_id, s)
      }

      const sorted = Array.from(scenarios.values()).sort(
        (a, b) => ((b.stopped_at ?? b.started_at) - (a.stopped_at ?? a.started_at)),
      )

      return {
        scenarios,
        selectedScenarioId:
          state.selectedScenarioId && scenarios.has(state.selectedScenarioId)
            ? state.selectedScenarioId
            : (sorted[0]?.scenario_id ?? null),
      }
    }),

  upsertScenario: (scenario) =>
    set((state) => {
      const scenarios = new Map(state.scenarios)
      scenarios.set(scenario.scenario_id, scenario)
      return {
        scenarios,
        selectedScenarioId: state.selectedScenarioId ?? scenario.scenario_id,
      }
    }),

  setSelectedScenario: (scenarioId) => set({ selectedScenarioId: scenarioId }),

  clearHistory: () => set({ scenarios: new Map(), recentEvents: [], selectedScenarioId: null }),
}))

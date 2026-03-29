// ============================================================================
// Scenario Monitor -- Active scenarios with progress bars + stop buttons
// ============================================================================

import { useSimulationStore } from '@/stores/use-simulation-store'
import { useStopAttack, useStopAllAttacks } from '@/hooks/use-api'
import { GaugeBar } from '@/components/shared/gauge-bar'
import { cn, fmtDuration } from '@/lib/utils'
import { Activity, Play, Square, OctagonX } from 'lucide-react'

export function ScenarioMonitor() {
  const scenarios = useSimulationStore((s) => s.scenarios)
  const selectedScenarioId = useSimulationStore((s) => s.selectedScenarioId)
  const setSelectedScenario = useSimulationStore((s) => s.setSelectedScenario)
  const stopAttack = useStopAttack()
  const stopAll = useStopAllAttacks()

  const active = Array.from(scenarios.values()).filter((s) => s.status === 'running')

  if (active.length === 0) return null

  return (
    <div className="animate-fade-in bg-bg-elevated/95 border border-border-default rounded-md p-4 backdrop-blur-sm shadow-[inset_0_1px_0_0_rgba(255,255,255,0.03)]">
      {/* header */}
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2">
          <Activity className="w-3.5 h-3.5 text-accent-primary" />
          <span className="text-[10px] font-semibold uppercase tracking-[0.12em] text-text-secondary">
            Active Scenarios
          </span>
          <span className="ml-1 inline-flex items-center justify-center w-4.5 h-4.5 rounded-full bg-accent-primary/15 text-accent-primary text-[9px] font-bold tabular-nums px-1.5 py-0.5">
            {active.length}
          </span>
        </div>
        {active.length > 1 && (
          <button
            onClick={() => void stopAll.mutateAsync()}
            disabled={stopAll.isPending}
            className="flex items-center gap-1 text-[9px] font-bold uppercase tracking-wider text-alert-critical hover:text-alert-critical/80 transition-colors"
          >
            <OctagonX className="w-3 h-3" />
            Stop All
          </button>
        )}
      </div>

      {/* scenario cards */}
      <div className="space-y-2">
        {active.map((s) => (
          <div
            key={s.scenario_id}
            onClick={() => setSelectedScenario(s.scenario_id)}
            className={cn(
              'card-hover w-full rounded-md p-3 text-left transition-all duration-150 cursor-pointer',
              selectedScenarioId === s.scenario_id
                ? 'bg-accent-primary/10 ring-1 ring-accent-primary/50 border border-accent-primary/20'
                : 'bg-bg-overlay/50 border border-border-subtle hover:bg-bg-overlay/80',
            )}
          >
            {/* top row */}
            <div className="flex items-center justify-between mb-2">
              <div className="flex items-center gap-2">
                <Play className="w-3 h-3 text-alert-low fill-alert-low" />
                <span className="text-[10px] font-bold text-text-primary">
                  {s.attack_label}
                </span>
                <span className="text-[9px] text-text-muted font-mono">
                  {s.scenario_id}
                </span>
              </div>
              <button
                type="button"
                onClick={(event) => {
                  event.stopPropagation()
                  void stopAttack.mutateAsync(s.scenario_id)
                }}
                disabled={stopAttack.isPending}
                className="flex items-center gap-1 text-[9px] font-bold uppercase tracking-wider text-alert-high hover:text-alert-critical transition-colors"
              >
                <Square className="w-2.5 h-2.5 fill-current" />
                Stop
              </button>
            </div>

            {/* progress bar */}
            <GaugeBar
              value={s.progress_pct}
              color={s.progress_pct > 80 ? 'low' : 'accent'}
            />

            {/* stats row */}
            <div className="flex items-center gap-3 mt-2 text-[9px] text-text-muted font-mono tabular-nums">
              <span>{s.events_ingested}/{s.events_generated} events</span>
              <span className="text-border-subtle">|</span>
              <span>{fmtDuration(s.elapsed_sec)}</span>
              <span className="text-border-subtle">|</span>
              <span>{s.accounts_involved.length} accounts</span>
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}

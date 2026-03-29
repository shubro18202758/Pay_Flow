// ============================================================================
// Scenario History -- Completed/stopped scenarios table
// ============================================================================

import { useSimulationStore } from '@/stores/use-simulation-store'
import { SeverityBadge } from '@/components/shared/severity-badge'
import { cn, fmtDuration, fmtTimestamp, truncId } from '@/lib/utils'
import { History, Inbox } from 'lucide-react'
import type { ScenarioStatusValue } from '@/lib/types'

const statusMap: Record<ScenarioStatusValue, { severity: 'low' | 'high' | 'critical' | 'medium'; label: string }> = {
  running: { severity: 'medium', label: 'RUNNING' },
  completed: { severity: 'low', label: 'COMPLETE' },
  stopped: { severity: 'high', label: 'STOPPED' },
  error: { severity: 'critical', label: 'ERROR' },
}

export function ScenarioHistory() {
  const scenarios = useSimulationStore((s) => s.scenarios)
  const selectedScenarioId = useSimulationStore((s) => s.selectedScenarioId)
  const setSelectedScenario = useSimulationStore((s) => s.setSelectedScenario)
  const all = Array.from(scenarios.values())
    .filter((s) => s.status !== 'running')
    .sort((a, b) => (b.stopped_at ?? 0) - (a.stopped_at ?? 0))

  return (
    <div className="animate-fade-in bg-bg-elevated/95 border border-border-default rounded-md p-4 backdrop-blur-sm shadow-[inset_0_1px_0_0_rgba(255,255,255,0.03)]">
      {/* header */}
      <div className="flex items-center gap-2 mb-3">
        <History className="w-3.5 h-3.5 text-accent-primary" />
        <span className="text-[10px] font-semibold uppercase tracking-[0.12em] text-text-secondary">
          History
        </span>
        {all.length > 0 && (
          <span className="ml-1 inline-flex items-center justify-center rounded-full bg-bg-overlay px-1.5 py-0.5 text-[9px] font-bold tabular-nums text-text-muted border border-border-subtle">
            {all.length}
          </span>
        )}
      </div>

      {all.length === 0 ? (
        <div className="flex flex-col items-center justify-center min-h-28 rounded-md border border-dashed border-border-subtle bg-bg-overlay/20">
          <Inbox className="w-5 h-5 text-text-muted/50 mb-2" />
          <p className="text-[10px] uppercase tracking-wider text-text-muted font-semibold">
            No completed scenarios yet
          </p>
          <p className="text-[9px] text-text-muted/60 mt-0.5">
            Launch an attack to see results here
          </p>
        </div>
      ) : (
        <div className="rounded-md border border-border-subtle overflow-hidden">
          <table className="w-full text-[10px] font-mono">
            <thead>
              <tr className="text-left text-text-muted bg-bg-overlay/40">
                <th className="py-2 px-3 font-semibold uppercase tracking-wider">ID</th>
                <th className="py-2 px-3 font-semibold uppercase tracking-wider">Attack</th>
                <th className="py-2 px-3 font-semibold uppercase tracking-wider">Status</th>
                <th className="py-2 px-3 font-semibold uppercase tracking-wider">Events</th>
                <th className="py-2 px-3 font-semibold uppercase tracking-wider">Duration</th>
                <th className="py-2 px-3 font-semibold uppercase tracking-wider">Ended</th>
              </tr>
            </thead>
            <tbody>
              {all.map((s, idx) => {
                const sm = statusMap[s.status]
                return (
                  <tr
                    key={s.scenario_id}
                    onClick={() => setSelectedScenario(s.scenario_id)}
                    className={cn(
                      'card-hover border-b border-border-subtle/50 cursor-pointer transition-colors',
                      selectedScenarioId === s.scenario_id
                        ? 'bg-accent-primary/10'
                        : idx % 2 === 0
                          ? 'bg-transparent hover:bg-bg-overlay/40'
                          : 'bg-bg-overlay/15 hover:bg-bg-overlay/40',
                    )}
                  >
                    <td className="py-2 px-3 text-text-primary">{truncId(s.scenario_id, 8)}</td>
                    <td className="py-2 px-3 text-text-secondary">{s.attack_label}</td>
                    <td className="py-2 px-3">
                      <SeverityBadge severity={sm.severity} label={sm.label} />
                    </td>
                    <td className="py-2 px-3 text-text-primary tabular-nums">
                      {s.events_ingested}/{s.events_generated}
                    </td>
                    <td className="py-2 px-3 text-text-muted tabular-nums">{fmtDuration(s.elapsed_sec)}</td>
                    <td className="py-2 px-3 text-text-muted tabular-nums">
                      {s.stopped_at ? fmtTimestamp(s.stopped_at) : '--'}
                    </td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>
      )}
    </div>
  )
}

// ============================================================================
// Signal Briefing -- Threat level, scenario, event, verdict, risk distribution
// ============================================================================

import { useDashboardStore } from '@/stores/use-dashboard-store'
import { useSimulationStore } from '@/stores/use-simulation-store'
import { useThreatSummary, useRiskDistribution } from '@/hooks/use-api'
import { SeverityBadge, verdictToSeverity } from '@/components/shared/severity-badge'
import { fmtPaisa, fmtTimestamp, truncId, cn } from '@/lib/utils'
import { Crosshair, Zap, Scale, ShieldAlert, BarChart3, AlertTriangle } from 'lucide-react'
import type { LucideIcon } from 'lucide-react'
import type { SSEAgentVerdict, ThreatIndicator } from '@/lib/types'

const THREAT_COLORS: Record<string, { bg: string; border: string; text: string; glow: string }> = {
  critical: { bg: 'bg-red-500/15', border: 'border-red-500/40', text: 'text-red-400', glow: 'shadow-red-500/20 shadow-lg' },
  high:     { bg: 'bg-orange-500/15', border: 'border-orange-500/40', text: 'text-orange-400', glow: 'shadow-orange-500/15 shadow-md' },
  elevated: { bg: 'bg-amber-500/15', border: 'border-amber-500/40', text: 'text-amber-400', glow: '' },
  normal:   { bg: 'bg-emerald-500/10', border: 'border-emerald-500/30', text: 'text-emerald-400', glow: '' },
  unknown:  { bg: 'bg-zinc-500/10', border: 'border-zinc-500/30', text: 'text-zinc-400', glow: '' },
}

export function SignalBriefing() {
  const agentLog = useDashboardStore((s) => s.agentLog)
  const scenarios = useSimulationStore((s) => s.scenarios)
  const recentEvents = useSimulationStore((s) => s.recentEvents)
  const selectedScenarioId = useSimulationStore((s) => s.selectedScenarioId)
  const { data: threatData } = useThreatSummary()
  const { data: riskData } = useRiskDistribution()

  const latestVerdict = [...agentLog].reverse().find((entry) => entry.type === 'verdict')
  const selectedScenario = selectedScenarioId ? scenarios.get(selectedScenarioId) ?? null : null
  const currentScenario = selectedScenario ?? Array.from(scenarios.values()).sort(
    (a, b) => ((b.stopped_at ?? b.started_at) - (a.stopped_at ?? a.started_at)),
  )[0] ?? null
  const latestEvent = [...recentEvents].reverse().find(
    (event) => !currentScenario || event.scenarioId === currentScenario.scenario_id,
  ) ?? [...recentEvents].at(-1)

  const threatLevel = threatData?.threat_level ?? 'unknown'
  const colors = THREAT_COLORS[threatLevel] ?? THREAT_COLORS.unknown

  return (
    <div className="p-2.5">
      <div className="mb-2.5 px-1 text-[10px] font-semibold uppercase tracking-[0.12em] text-text-secondary">
        Signal Briefing
      </div>
      <div className="space-y-2">
        {/* Threat Level Indicator */}
        <div className={cn(
          'rounded-md border p-2.5 transition-all duration-300',
          colors.bg, colors.border, colors.glow,
        )}>
          <div className="flex items-center justify-between mb-1.5">
            <div className="flex items-center gap-1.5 text-[9px] uppercase tracking-[0.12em] text-text-muted">
              <ShieldAlert className={cn('w-3.5 h-3.5', colors.text)} />
              <span>Threat Level</span>
            </div>
            <span className={cn('text-[11px] font-bold uppercase tracking-wider', colors.text)}>
              {threatLevel}
            </span>
          </div>
          {threatData?.indicators && threatData.indicators.length > 0 && (
            <div className="space-y-1 mt-1.5">
              {threatData.indicators.slice(0, 3).map((ind: ThreatIndicator, i: number) => (
                <div key={i} className="flex items-center gap-1.5 text-[8px]">
                  <AlertTriangle className={cn('w-2.5 h-2.5 shrink-0',
                    ind.severity === 'critical' ? 'text-red-400' : ind.severity === 'high' ? 'text-orange-400' : 'text-amber-400'
                  )} />
                  <span className="text-text-secondary truncate">{ind.detail}</span>
                </div>
              ))}
            </div>
          )}
          {threatData?.severity_score != null && (
            <div className="mt-2 h-1 rounded-full bg-bg-deep/50 overflow-hidden">
              <div
                className={cn('h-full rounded-full transition-all duration-500',
                  threatLevel === 'critical' ? 'bg-red-500' : threatLevel === 'high' ? 'bg-orange-500' : threatLevel === 'elevated' ? 'bg-amber-500' : 'bg-emerald-500'
                )}
                style={{ width: `${Math.min(threatData.severity_score * 100, 100)}%` }}
              />
            </div>
          )}
        </div>

        {/* Risk Distribution Mini-Chart */}
        {riskData && riskData.total > 0 && (
          <div className="rounded-md border border-border-subtle bg-bg-elevated/70 p-2.5">
            <div className="flex items-center justify-between mb-2 text-[9px] uppercase tracking-[0.12em] text-text-muted">
              <div className="flex items-center gap-1.5">
                <BarChart3 className="w-3 h-3 text-accent-primary/70" />
                <span>Risk Distribution</span>
              </div>
              <span className="font-mono tabular-nums">{riskData.total} scored</span>
            </div>
            <div className="flex items-end gap-0.5 h-6">
              {riskData.buckets.map((count: number, i: number) => {
                const maxVal = Math.max(...riskData.buckets, 1)
                const pct = (count / maxVal) * 100
                const hue = 120 - i * 12 // green → red
                return (
                  <div
                    key={i}
                    className="flex-1 rounded-t-sm transition-all duration-300"
                    style={{
                      height: `${Math.max(pct, 4)}%`,
                      backgroundColor: `hsl(${hue}, 70%, 50%)`,
                      opacity: count > 0 ? 0.8 : 0.2,
                    }}
                    title={`${i * 10}-${(i + 1) * 10}%: ${count} events`}
                  />
                )
              })}
            </div>
            <div className="flex justify-between mt-1 text-[7px] font-mono text-text-muted">
              <span>Low</span>
              <span>Mean: {(riskData.mean * 100).toFixed(1)}%</span>
              <span>High</span>
            </div>
          </div>
        )}

        <BriefingCard
          icon={Crosshair}
          title="Current Scenario"
          body={
            currentScenario
              ? `${currentScenario.attack_label} | ${currentScenario.events_ingested}/${currentScenario.events_generated} events | ${currentScenario.status}`
              : 'No simulation selected yet'
          }
          meta={currentScenario ? truncId(currentScenario.scenario_id, 8) : 'idle'}
        />

        <BriefingCard
          icon={Zap}
          title="Latest Injected Event"
          body={latestEvent ? summarizeEvent(latestEvent) : 'No synthetic event has entered the pipeline yet'}
          meta={latestEvent ? fmtTimestamp(latestEvent.timestamp) : 'awaiting'}
        />

        {/* Latest Verdict -- special card with severity badge */}
        <div className="rounded-md border border-border-subtle bg-bg-elevated/70 p-2.5 card-hover transition-all duration-200">
          <div className="mb-1.5 flex items-center gap-1.5 text-[9px] uppercase tracking-[0.12em] text-text-muted">
            <Scale className="w-3 h-3 text-accent-primary/70" />
            <span>Latest Verdict</span>
          </div>
          {latestVerdict ? (
            <>
              <div className="flex items-center gap-2">
                <SeverityBadge severity={verdictToSeverity((latestVerdict.data as SSEAgentVerdict).verdict)} />
                <span className="text-[10px] font-mono text-text-primary">
                  {truncId((latestVerdict.data as SSEAgentVerdict).txn_id, 10)}
                </span>
                <span className="ml-auto text-[9px] font-mono tabular-nums text-text-muted">
                  {fmtTimestamp(latestVerdict.timestamp)}
                </span>
              </div>
              <div className="mt-1.5 text-[10px] leading-relaxed text-text-secondary">
                {(latestVerdict.data as SSEAgentVerdict).reasoning_summary || 'Verdict emitted with no summary.'}
              </div>
            </>
          ) : (
            <div className="text-[10px] leading-relaxed text-text-muted">
              No agent verdict yet. Once LLM-backed investigations run, this panel will summarize the latest decision and why it was made.
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

function BriefingCard({
  icon: Icon,
  title,
  body,
  meta,
}: {
  icon: LucideIcon
  title: string
  body: string
  meta: string
}) {
  return (
    <div className="rounded-md border border-border-subtle bg-bg-elevated/70 p-2.5 card-hover transition-all duration-200">
      <div className="mb-1.5 flex items-center justify-between gap-2 text-[9px] uppercase tracking-[0.12em] text-text-muted">
        <div className="flex items-center gap-1.5">
          <Icon className="w-3 h-3 text-accent-primary/70" />
          <span>{title}</span>
        </div>
        <span className="font-mono tabular-nums">{meta}</span>
      </div>
      <div className="text-[10px] leading-relaxed text-text-secondary">{body}</div>
    </div>
  )
}

function summarizeEvent(event: ReturnType<typeof useSimulationStore.getState>['recentEvents'][number]): string {
  if (event.eventType === 'transaction') {
    return `${truncId(event.event.sender ?? 'unknown', 10)} -> ${truncId(event.event.receiver ?? 'unknown', 10)} | ${fmtPaisa(event.event.amount_paisa ?? 0)}`
  }
  if (event.eventType === 'auth') {
    return `${truncId(event.event.account ?? 'unknown', 10)} | ${event.event.action ?? 'AUTH'} | ${event.event.success ? 'SUCCESS' : 'FAIL'}`
  }
  if (event.eventType === 'interbank') {
    return `${event.event.sender_ifsc ?? 'unknown'} -> ${event.event.receiver_ifsc ?? 'unknown'} | ${fmtPaisa(event.event.amount_paisa ?? 0)}`
  }
  return `${event.attackLabel} emitted an unknown payload`
}

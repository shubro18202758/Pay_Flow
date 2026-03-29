// ============================================================================
// Scenario Inspector -- API-driven blueprint + accounts + signals
// ============================================================================

import type { ReactNode } from 'react'
import { useSimulationStore } from '@/stores/use-simulation-store'
import { useAttackTypes } from '@/hooks/use-api'
import { SeverityBadge } from '@/components/shared/severity-badge'
import { fmtDuration, fmtNum, truncId } from '@/lib/utils'
import {
  Search,
  Cog,
  Users,
  AlertTriangle,
  Monitor,
  FileSearch,
  ListOrdered,
} from 'lucide-react'
import type { LucideIcon } from 'lucide-react'

export function ScenarioInspector() {
  const scenarios = useSimulationStore((s) => s.scenarios)
  const selectedScenarioId = useSimulationStore((s) => s.selectedScenarioId)
  const { data: attackTypesData } = useAttackTypes()

  const selected = selectedScenarioId ? scenarios.get(selectedScenarioId) ?? null : null
  const fallback = Array.from(scenarios.values()).sort(
    (a, b) => ((b.stopped_at ?? b.started_at) - (a.stopped_at ?? a.started_at)),
  )[0] ?? null
  const scenario = selected ?? fallback

  if (!scenario) {
    return (
      <div className="animate-fade-in bg-bg-elevated/95 border border-border-default rounded-md p-5 backdrop-blur-sm shadow-[inset_0_1px_0_0_rgba(255,255,255,0.03)]">
        <div className="flex items-center gap-2 text-[10px] font-semibold uppercase tracking-[0.12em] text-text-secondary">
          <Search className="w-3.5 h-3.5 text-accent-primary" />
          Scenario Inspector
        </div>
        <div className="mt-6 flex flex-col items-center justify-center py-6">
          <FileSearch className="w-7 h-7 text-text-muted/40 mb-3" />
          <p className="text-[11px] text-text-muted leading-relaxed text-center max-w-xs">
            Launch a scenario to inspect how PayFlow expects that attack to
            manifest across ingestion, graph structure, and downstream
            verdicting.
          </p>
        </div>
      </div>
    )
  }

  // Get enriched attack type info from API
  const attackDetail = attackTypesData?.attacks?.[scenario.attack_type]
  const severity = scenario.status === 'running' ? 'medium' : scenario.status === 'completed' ? 'low' : scenario.status === 'error' ? 'critical' : 'high'

  return (
    <div className="animate-fade-in bg-bg-elevated/95 border border-border-default rounded-md p-5 backdrop-blur-sm shadow-[inset_0_1px_0_0_rgba(255,255,255,0.03)]">
      {/* header */}
      <div className="flex items-start justify-between gap-3">
        <div>
          <div className="flex items-center gap-2 text-[10px] font-semibold uppercase tracking-[0.12em] text-text-secondary">
            <Search className="w-3.5 h-3.5 text-accent-primary" />
            Scenario Inspector
          </div>
          <h3 className="mt-2 text-sm font-semibold text-text-primary">
            {scenario.attack_label}
          </h3>
          <div className="mt-1 text-[10px] font-mono text-text-muted tabular-nums">
            {truncId(scenario.scenario_id, 8)} | {fmtNum(scenario.accounts_involved.length)} accounts | {fmtDuration(scenario.elapsed_sec)}
          </div>
        </div>
        <SeverityBadge severity={severity} label={scenario.status.toUpperCase()} />
      </div>

      {/* metrics row */}
      <div className="mt-4 grid grid-cols-3 gap-2">
        <Metric title="Injected" value={`${scenario.events_ingested}/${scenario.events_generated}`} />
        <Metric title="Progress" value={`${scenario.progress_pct.toFixed(0)}%`} />
        <Metric title="Accounts" value={fmtNum(scenario.accounts_involved.length)} />
      </div>

      {/* API-driven mechanics + phases */}
      <div className="mt-4 grid gap-4 lg:grid-cols-2">
        <Section title="Attack Description" icon={Cog}>
          <div className="border-l-2 border-accent-primary/30 pl-3">
            <p className="text-[11px] leading-relaxed text-text-secondary">
              {attackDetail?.description ?? scenario.attack_label}
            </p>
          </div>
        </Section>

        {attackDetail?.phases && attackDetail.phases.length > 0 && (
          <Section title="Attack Phases" icon={ListOrdered}>
            <div className="space-y-1.5">
              {attackDetail.phases.map((phase, i) => (
                <div key={i} className="flex items-start gap-2 bg-bg-overlay/50 border border-border-subtle rounded-md px-2.5 py-1.5 card-hover">
                  <span className="text-[9px] font-mono font-bold text-accent-primary mt-0.5 shrink-0 w-4">{i + 1}.</span>
                  <span className="text-[10px] text-text-secondary leading-snug">{phase}</span>
                </div>
              ))}
            </div>
          </Section>
        )}
      </div>

      {/* Accounts + Parameters */}
      <div className="mt-4 grid gap-4 lg:grid-cols-2">
        <Section title="Accounts Involved" icon={Users}>
          <div className="flex flex-wrap gap-1.5">
            {scenario.accounts_involved.slice(0, 12).map((accountId) => (
              <span
                key={accountId}
                className="card-hover rounded-md border border-border-subtle bg-bg-overlay px-2 py-1 text-[10px] font-mono text-text-secondary"
              >
                {truncId(accountId, 10)}
              </span>
            ))}
            {scenario.accounts_involved.length > 12 && (
              <span className="rounded-md border border-border-subtle bg-bg-overlay px-2 py-1 text-[10px] text-text-muted">
                +{scenario.accounts_involved.length - 12} more
              </span>
            )}
          </div>
        </Section>

        {attackDetail?.params && Object.keys(attackDetail.params).length > 0 && (
          <Section title="Configurable Parameters" icon={AlertTriangle}>
            <div className="space-y-1.5">
              {Object.entries(attackDetail.params).map(([key, schema]) => (
                <div key={key} className="flex items-center justify-between bg-bg-overlay/50 border border-border-subtle rounded-md px-2.5 py-1.5">
                  <span className="text-[10px] text-text-secondary">{schema.label}</span>
                  <span className="text-[10px] font-mono font-bold text-text-primary tabular-nums">
                    {schema.default}
                  </span>
                </div>
              ))}
            </div>
          </Section>
        )}
      </div>
    </div>
  )
}

function Section({ title, icon: Icon, children }: { title: string; icon: LucideIcon; children: ReactNode }) {
  return (
    <div>
      <div className="mb-2 flex items-center gap-1.5 text-[10px] font-semibold uppercase tracking-[0.12em] text-text-secondary">
        <Icon className="w-3 h-3 text-text-muted" />
        {title}
      </div>
      {children}
    </div>
  )
}

function Metric({ title, value }: { title: string; value: string }) {
  return (
    <div className="rounded-md border border-border-subtle bg-bg-overlay/50 px-3 py-2.5">
      <div className="text-[9px] uppercase tracking-[0.12em] text-text-muted font-semibold">{title}</div>
      <div className="mt-1 text-sm font-semibold text-text-primary font-mono tabular-nums">{value}</div>
    </div>
  )
}

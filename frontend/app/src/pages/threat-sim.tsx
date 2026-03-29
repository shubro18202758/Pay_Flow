// ============================================================================
// Threat Simulation Page -- Attack launcher + Custom events + Monitor + History
// ============================================================================

import { Crosshair, Rocket, Search, ShieldCheck } from 'lucide-react'
import { AttackLauncher } from '@/components/simulation/attack-launcher'
import { CustomEventBuilder } from '@/components/simulation/custom-event-builder'
import { ScenarioMonitor } from '@/components/simulation/scenario-monitor'
import { ScenarioHistory } from '@/components/simulation/scenario-history'
import { ScenarioInspector } from '@/components/simulation/scenario-inspector'
import { SimulationEventTrace } from '@/components/simulation/simulation-event-trace'
import { PipelineMotionVisualizer } from '@/components/simulation/pipeline-motion-visualizer'
import { PipelineTransparency } from '@/components/simulation/pipeline-transparency'
import { useActivityStore } from '@/stores/use-activity-store'

const STEPS = [
  { num: 1, label: 'LAUNCH ATTACK', icon: Rocket },
  { num: 2, label: 'INSPECT EVENTS', icon: Search },
  { num: 3, label: 'VERIFY SIGNALS', icon: ShieldCheck },
] as const

export function ThreatSimPage() {
  const trackedEventId = useActivityStore((s) => s.trackedEventId)

  return (
    <div className="h-full overflow-y-auto p-4 space-y-4">
      {/* ── Hero header ── */}
      <section className="animate-fade-in rounded-md border border-border-default bg-bg-elevated/95 p-5 backdrop-blur-sm shadow-[inset_0_1px_0_0_rgba(255,255,255,0.03)]">
        <div className="flex items-center gap-2 text-[10px] text-accent-primary uppercase tracking-[0.12em] font-semibold">
          <Crosshair className="w-3.5 h-3.5" />
          Threat Simulation Engine
        </div>

        <div className="mt-3 flex flex-col gap-4 lg:flex-row lg:items-end lg:justify-between">
          <div>
            <h2 className="text-lg font-semibold text-text-primary leading-tight">
              Expose the causal path, not just the outcome
            </h2>
            <p className="mt-1.5 max-w-3xl text-[12px] leading-relaxed text-text-secondary">
              Launch a scenario, inject custom events, and follow each event as it enters
              the pipeline. This view explains why the graph, alerts, and verdicts move the way they do.
            </p>
          </div>

          {/* ── Step indicator ── */}
          <div className="flex items-center gap-0 lg:min-w-[28rem]">
            {STEPS.map((step, i) => {
              const Icon = step.icon
              return (
                <div key={step.num} className="flex items-center flex-1">
                  <div className="flex flex-col items-center text-center flex-1 relative">
                    <div className="w-7 h-7 rounded-full border border-accent-primary/50 bg-accent-primary/10 flex items-center justify-center">
                      <span className="text-[10px] font-bold text-accent-primary tabular-nums">
                        {step.num}
                      </span>
                    </div>
                    <div className="mt-1.5 flex items-center gap-1">
                      <Icon className="w-3 h-3 text-text-muted" />
                      <span className="text-[9px] font-semibold uppercase tracking-[0.12em] text-text-muted">
                        {step.label}
                      </span>
                    </div>
                  </div>
                  {i < STEPS.length - 1 && (
                    <div className="h-px w-full bg-gradient-to-r from-accent-primary/40 to-accent-primary/10 mt-[-12px]" />
                  )}
                </div>
              )
            })}
          </div>
        </div>
      </section>

      {/* ── Two-column: Attack Launcher + Custom Event Builder side-by-side ── */}
      <div className="grid gap-4 xl:grid-cols-[1fr_1fr]">
        <AttackLauncher />
        <CustomEventBuilder />
      </div>

      {/* ── Live Processing Pipeline — full-width animated flow ── */}
      <PipelineMotionVisualizer trackedEventId={trackedEventId} className="animate-fade-in" />

      {/* ── Pipeline Transparency X-Ray — full algorithmic visibility ── */}
      <PipelineTransparency className="animate-fade-in" />

      {/* ── Monitor + Inspector grid ── */}
      <div className="grid gap-4 xl:grid-cols-[1.15fr_0.85fr]">
        <div className="space-y-4">
          <ScenarioMonitor />
          <ScenarioHistory />
        </div>
        <div className="space-y-4">
          <ScenarioInspector />
          <SimulationEventTrace />
        </div>
      </div>
    </div>
  )
}

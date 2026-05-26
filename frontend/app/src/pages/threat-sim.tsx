// ============================================================================
// Threat Simulation Page -- Attack launcher + Custom events + Monitor + History
// ============================================================================

import { ClipboardCheck, Radar, Search } from 'lucide-react'
import { useState } from 'react'
import { AttackLauncher } from '@/components/simulation/attack-launcher'
import { CustomEventBuilder } from '@/components/simulation/custom-event-builder'
import { AdaptiveEventLab } from '@/components/simulation/adaptive-event-lab'
import { ScenarioMonitor } from '@/components/simulation/scenario-monitor'
import { ScenarioHistory } from '@/components/simulation/scenario-history'
import { ScenarioInspector } from '@/components/simulation/scenario-inspector'
import { SimulationEventTrace } from '@/components/simulation/simulation-event-trace'
import { PipelineMotionVisualizer } from '@/components/simulation/pipeline-motion-visualizer'
import { PipelineTransparency } from '@/components/simulation/pipeline-transparency'
import { useActivityStore } from '@/stores/use-activity-store'

const STEPS = [
  { num: 1, label: 'PRIME FROM INTEL', icon: Radar },
  { num: 2, label: 'INJECT EVENT CHAIN', icon: Search },
  { num: 3, label: 'APPROVE COUNTER', icon: ClipboardCheck },
] as const

export function ThreatSimPage() {
  const trackedEventId = useActivityStore((s) => s.trackedEventId)
  const [showLegacyTools, setShowLegacyTools] = useState(false)

  return (
    <div className="h-full overflow-y-auto p-4 space-y-4">
      {/* ── Hero header ── */}
      <section className="animate-fade-in rounded-md border border-border-default bg-bg-elevated/95 p-5 backdrop-blur-sm shadow-[inset_0_1px_0_0_rgba(255,255,255,0.03)]">
        <div className="flex items-center gap-2 text-[10px] text-accent-primary uppercase tracking-[0.12em] font-semibold">
          <Radar className="w-3.5 h-3.5" />
          Adaptive Event Lab
        </div>

        <div className="mt-3 flex flex-col gap-4 lg:flex-row lg:items-end lg:justify-between">
          <div>
            <h2 className="text-lg font-semibold text-text-primary leading-tight">
              Turn pre-fraud intelligence into live, analyst-gated response drills
            </h2>
            <p className="mt-1.5 max-w-3xl text-[12px] leading-relaxed text-text-secondary">
              Generate realistic Indian banking event chains from active OSINT/SOCMINT playbooks, watch each backend
              process stage, and approve countermeasures only after internal PayFlow evidence is visible.
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

      <AdaptiveEventLab />

      {/* ── Live Processing Pipeline — full-width animated flow ── */}
      <PipelineMotionVisualizer trackedEventId={trackedEventId} className="animate-fade-in" />

      {/* ── Pipeline Transparency X-Ray — full algorithmic visibility ── */}
      <PipelineTransparency className="animate-fade-in" />

      <section className="rounded-lg border border-border-default bg-bg-surface p-4 shadow-sm">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div>
            <div className="text-[10px] font-bold uppercase tracking-[0.12em] text-text-primary">
              Secondary simulation tools
            </div>
            <p className="mt-1 text-[10px] text-text-secondary">
              Kept available for engineering checks, but collapsed so the judge path stays focused on intel-driven countermeasures.
            </p>
          </div>
          <button
            type="button"
            onClick={() => setShowLegacyTools((value) => !value)}
            className="rounded-md border border-accent-primary/30 bg-bg-surface px-3 py-2 text-[10px] font-bold uppercase tracking-[0.12em] text-accent-primary hover:bg-accent-muted"
          >
            {showLegacyTools ? 'Hide tools' : 'Show tools'}
          </button>
        </div>

        {showLegacyTools && (
          <div className="mt-4 grid gap-4 xl:grid-cols-[1.15fr_0.85fr]">
            <div className="space-y-4">
              <AttackLauncher />
              <ScenarioMonitor />
              <ScenarioHistory />
            </div>
            <div className="space-y-4">
              <CustomEventBuilder />
              <ScenarioInspector />
              <SimulationEventTrace />
            </div>
          </div>
        )}
      </section>
    </div>
  )
}

// ============================================================================
// Simulation Event Trace -- Live feed of injected synthetic events
// ============================================================================

import { useSimulationStore } from '@/stores/use-simulation-store'
import { cn, fmtPaisa, fmtTimestamp, truncId } from '@/lib/utils'
import {
  ListOrdered,
  ArrowRightLeft,
  ShieldAlert,
  Landmark,
  Radio,
} from 'lucide-react'

export function SimulationEventTrace() {
  const recentEvents = useSimulationStore((s) => s.recentEvents)
  const selectedScenarioId = useSimulationStore((s) => s.selectedScenarioId)

  const visible = recentEvents
    .filter((event) => !selectedScenarioId || event.scenarioId === selectedScenarioId)
    .slice(-18)
    .reverse()

  return (
    <div className="animate-fade-in bg-bg-elevated/95 border border-border-default rounded-md p-4 min-h-[20rem] backdrop-blur-sm shadow-[inset_0_1px_0_0_rgba(255,255,255,0.03)]">
      {/* header */}
      <div className="flex items-center justify-between gap-3">
        <div>
          <div className="flex items-center gap-2 text-[10px] font-semibold uppercase tracking-[0.12em] text-text-secondary">
            <ListOrdered className="w-3.5 h-3.5 text-accent-primary" />
            Injected Event Trace
          </div>
          <div className="mt-1 text-[11px] text-text-muted">
            Each row is one synthetic event entering the live pipeline.
          </div>
        </div>
        <div className="flex items-center gap-1.5 text-[10px] font-mono text-text-muted tabular-nums">
          <Radio className="w-3 h-3 text-accent-primary" />
          {visible.length} visible
        </div>
      </div>

      {/* event list */}
      <div className="mt-4 space-y-2">
        {visible.length === 0 ? (
          <div className="flex flex-col items-center justify-center min-h-48 rounded-md border border-dashed border-border-subtle bg-bg-overlay/20">
            <ListOrdered className="w-6 h-6 text-text-muted/40 mb-2" />
            <p className="text-[11px] text-text-muted font-semibold">
              No injected events captured yet
            </p>
            <p className="text-[9px] text-text-muted/60 mt-0.5">
              Events will appear here when a scenario is running
            </p>
          </div>
        ) : (
          visible.map((entry, idx) => (
            <div
              key={entry.id}
              className="animate-fade-in card-hover rounded-md border border-border-subtle bg-bg-overlay/50 px-3 py-2.5"
              style={{ animationDelay: `${idx * 30}ms` }}
            >
              {/* top row: badge + attack label + timestamp + progress */}
              <div className="flex flex-wrap items-center gap-2">
                <TraceBadge label={entry.eventType.toUpperCase()} tone={entry.eventType} />
                <span className="text-[10px] font-semibold text-text-primary">
                  {entry.attackLabel}
                </span>
                <span className="text-[10px] font-mono text-text-muted tabular-nums">
                  {fmtTimestamp(entry.timestamp)}
                </span>
                <span className="ml-auto text-[10px] font-mono font-semibold text-accent-primary tabular-nums">
                  {entry.progressPct.toFixed(0)}%
                </span>
              </div>

              {/* summary */}
              <div className="mt-2 text-[11px] leading-relaxed text-text-secondary">
                {renderSummary(entry)}
              </div>

              {/* metadata row */}
              <div className="mt-2 flex flex-wrap items-center gap-3 text-[9px] font-mono text-text-muted tabular-nums">
                <span>scenario:{truncId(entry.scenarioId, 8)}</span>
                <span className="text-border-subtle">|</span>
                <span>type:{entry.attackType}</span>
                {entry.event.channel && (
                  <>
                    <span className="text-border-subtle">|</span>
                    <span>channel:{entry.event.channel}</span>
                  </>
                )}
                {entry.event.fraud_label && (
                  <>
                    <span className="text-border-subtle">|</span>
                    <span className="text-alert-high">flag:{entry.event.fraud_label}</span>
                  </>
                )}
              </div>
            </div>
          ))
        )}
      </div>
    </div>
  )
}

function renderSummary(entry: ReturnType<typeof useSimulationStore.getState>['recentEvents'][number]) {
  if (entry.eventType === 'transaction') {
    return (
      <>
        <span className="font-semibold text-text-primary">{truncId(entry.event.sender ?? 'unknown', 10)}</span>
        <span> sent </span>
        <span className="font-semibold text-text-primary">{fmtPaisa(entry.event.amount_paisa ?? 0)}</span>
        <span> to </span>
        <span className="font-semibold text-text-primary">{truncId(entry.event.receiver ?? 'unknown', 10)}</span>
        {entry.event.txn_id && (
          <span className="text-text-muted"> via txn {truncId(entry.event.txn_id, 10)}</span>
        )}
      </>
    )
  }

  if (entry.eventType === 'auth') {
    return (
      <>
        <span className="font-semibold text-text-primary">{truncId(entry.event.account ?? 'unknown', 10)}</span>
        <span> received auth event </span>
        <span className="font-semibold text-text-primary">{entry.event.action ?? 'UNKNOWN'}</span>
        <span> from </span>
        <span className="font-semibold text-text-primary">{entry.event.ip ?? 'unknown IP'}</span>
        <span className={cn('ml-1 font-semibold', entry.event.success ? 'text-alert-low' : 'text-alert-critical')}>
          {entry.event.success ? 'SUCCESS' : 'FAIL'}
        </span>
      </>
    )
  }

  if (entry.eventType === 'interbank') {
    return (
      <>
        <span className="font-semibold text-text-primary">{entry.event.sender_ifsc ?? 'unknown'}</span>
        <span> moved </span>
        <span className="font-semibold text-text-primary">{fmtPaisa(entry.event.amount_paisa ?? 0)}</span>
        <span> toward </span>
        <span className="font-semibold text-text-primary">{entry.event.receiver_ifsc ?? 'unknown'}</span>
      </>
    )
  }

  return <span>Unknown simulation payload</span>
}

function TraceBadge({ label, tone }: { label: string; tone: string }) {
  const Icon =
    tone === 'transaction'
      ? ArrowRightLeft
      : tone === 'auth'
        ? ShieldAlert
        : tone === 'interbank'
          ? Landmark
          : Radio

  const classes =
    tone === 'transaction'
      ? 'border-alert-critical/40 bg-alert-critical/10 text-alert-critical'
      : tone === 'auth'
        ? 'border-alert-high/40 bg-alert-high/10 text-alert-high'
        : tone === 'interbank'
          ? 'border-accent-primary/40 bg-accent-primary/10 text-accent-primary'
          : 'border-border-subtle bg-bg-overlay text-text-muted'

  return (
    <span className={cn('inline-flex items-center gap-1 rounded-md border px-2 py-0.5 text-[9px] font-semibold tracking-wider', classes)}>
      <Icon className="w-2.5 h-2.5" />
      {label}
    </span>
  )
}

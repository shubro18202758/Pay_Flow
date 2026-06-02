import {
  BarChart3,
  CheckCircle2,
  FileText,
  GitBranch,
  RadioTower,
  ShieldAlert,
  type LucideIcon,
} from 'lucide-react'
import { useActivityStore, type EventLabReportActivity } from '@/stores/use-activity-store'
import { cn } from '@/lib/utils'

function short(value?: string | null, left = 10): string {
  if (!value) return 'n/a'
  return value.length > left + 4 ? `${value.slice(0, left)}...` : value
}

function compactInr(paisa: number): string {
  const rupees = Number(paisa || 0) / 100
  if (rupees >= 10_000_000) return `INR ${(rupees / 10_000_000).toFixed(1)}Cr`
  if (rupees >= 100_000) return `INR ${(rupees / 100_000).toFixed(1)}L`
  return `INR ${rupees.toLocaleString('en-IN', { maximumFractionDigits: 0 })}`
}

function riskTone(report: EventLabReportActivity): 'critical' | 'high' | 'normal' {
  if (report.riskTier === 'critical' || (report.riskScore ?? 0) >= 0.86) return 'critical'
  if (report.riskTier === 'high' || (report.riskScore ?? 0) >= 0.72) return 'high'
  return 'normal'
}

export function EventEvaluationSyncStrip() {
  const report = useActivityStore((state) => state.eventLabReports[0])
  const activeRunId = useActivityStore((state) => state.activeEventLabRunId)

  if (!report) return null

  const tone = riskTone(report)
  const syncedTargets = report.syncTargets.filter((target) => target.status === 'synced')

  return (
    <section className="shrink-0 border-b border-[#c6d3e3] bg-[linear-gradient(90deg,#f8fbff,#eef6ff_58%,#fff5f4)] px-4 py-2">
      <div className="grid gap-2 xl:grid-cols-[minmax(0,1fr)_minmax(520px,0.9fr)]">
        <div className="min-w-0 rounded-lg border border-[#00579C]/20 bg-white px-3 py-2 shadow-sm">
          <div className="flex flex-wrap items-center justify-between gap-2">
            <div className="min-w-0">
              <div className="flex items-center gap-2 text-[9px] font-black uppercase tracking-[0.14em] text-[#00579C]">
                <RadioTower className="h-3.5 w-3.5 text-[#DA251C]" />
                Event evaluation synced prototype-wide
                {activeRunId === report.runId && (
                  <span className="rounded-full border border-alert-low/25 bg-alert-low/10 px-1.5 py-0.5 text-[7px] text-alert-low">
                    active run
                  </span>
                )}
              </div>
              <div className="mt-1 truncate text-[10px] font-semibold text-[#4b5d76]">
                {report.templateTitle} | {short(report.runId, 14)} | {report.routeLabel || 'route pending'} | {report.eventCount} events, {report.uniqueAccountCount} accounts
              </div>
            </div>
            <div className="flex flex-wrap items-center gap-1.5">
              <span
                className={cn(
                  'rounded-md border px-2 py-1 font-mono text-[10px] font-black uppercase tracking-[0.1em]',
                  tone === 'critical' && 'border-[#DA251C]/25 bg-[#DA251C]/10 text-[#DA251C]',
                  tone === 'high' && 'border-[#f5b400]/35 bg-[#f5b400]/10 text-[#7a5a00]',
                  tone === 'normal' && 'border-[#00579C]/25 bg-[#00579C]/10 text-[#00579C]',
                )}
              >
                {report.riskTier} {report.riskScore != null ? `${Math.round(report.riskScore * 100)}%` : ''}
              </span>
              <span className="rounded-md border border-[#00579C]/20 bg-[#00579C]/5 px-2 py-1 font-mono text-[10px] font-bold text-[#00579C]">
                {compactInr(report.exposurePaisa)}
              </span>
            </div>
          </div>
        </div>

        <div className="grid min-w-0 gap-2 md:grid-cols-[1fr_1fr_1fr]">
          <SyncFact icon={GitBranch} label="Graph" value={`${Number(report.report.route_segments && Array.isArray(report.report.route_segments) ? report.report.route_segments.length : 0)} route legs`} />
          <SyncFact icon={BarChart3} label="Analytics" value={`${Number(report.report.amount_series && Array.isArray(report.report.amount_series) ? report.report.amount_series.length : 0)} amount points`} />
          <SyncFact icon={FileText} label="Report" value={`${syncedTargets.length}/${report.syncTargets.length} targets synced`} />
        </div>
      </div>

      <div className="mt-2 flex flex-wrap gap-1.5">
        {report.syncTargets.map((target) => (
          <span
            key={target.key}
            className={cn(
              'inline-flex items-center gap-1 rounded-full border px-2 py-0.5 text-[8px] font-bold uppercase tracking-[0.1em]',
              target.status === 'synced'
                ? 'border-[#00579C]/20 bg-white text-[#00579C]'
                : 'border-border-subtle bg-bg-elevated text-text-muted',
            )}
          >
            {target.status === 'synced' ? <CheckCircle2 className="h-3 w-3" /> : <ShieldAlert className="h-3 w-3" />}
            {target.label} {target.count}
          </span>
        ))}
      </div>
    </section>
  )
}

function SyncFact({
  icon: Icon,
  label,
  value,
}: {
  icon: LucideIcon
  label: string
  value: string
}) {
  return (
    <div className="min-w-0 rounded-lg border border-[#00579C]/20 bg-white px-3 py-2 shadow-sm">
      <div className="flex items-center gap-2 text-[8px] font-bold uppercase tracking-[0.12em] text-[#617189]">
        <Icon className="h-3.5 w-3.5 text-[#00579C]" />
        {label}
      </div>
      <div className="mt-1 truncate font-mono text-[10px] font-black text-[#14213d]">{value}</div>
    </div>
  )
}

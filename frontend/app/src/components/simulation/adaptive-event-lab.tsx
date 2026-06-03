import { useEffect, useMemo, useRef, useState } from 'react'
import type { ReactNode } from 'react'
import {
  Activity,
  BadgeCheck,
  BarChart3,
  BrainCircuit,
  CheckCircle2,
  ClipboardCheck,
  Clock3,
  FileText,
  Gauge,
  GitBranch,
  Loader2,
  PauseCircle,
  Play,
  Radar,
  RotateCcw,
  ShieldAlert,
  ShieldCheck,
  Sparkles,
  XCircle,
  Zap,
} from 'lucide-react'
import {
  useApproveCountermeasure,
  useCountermeasureProposals,
  useCreateEventLabRun,
  useEventLabExplainability,
  useEventLabRun,
  useEventLabTemplates,
  useLLMStatus,
  usePreviewEventLabRun,
  useRejectCountermeasure,
} from '@/hooks/use-api'
import { useRoleAccess } from '@/hooks/use-rbac'
import {
  useActivityStore,
  type BackendTerminalEntry,
  type BackendTerminalSource,
  type BackendTerminalTone,
  type EventLifecycle,
} from '@/stores/use-activity-store'
import { cn, fmtOptionalMs, fmtOptionalTimestamp } from '@/lib/utils'
import { resolveLLMRuntime, type LLMRuntimeSummary } from '@/lib/llm-runtime'
import { sanitizeOptionalEvidenceText, sanitizePublicTraceText } from '@/lib/evidence-sanitizer'
import type {
  CountermeasureProposal,
  EventLabAnalysisReport,
  EventLabControls,
  EventLabExplainabilityResponse,
  EventLabGeneratedEvent,
  EventLabMode,
  EventLabRunResponse,
  EventLabTemplate,
} from '@/lib/types'

const MODES: EventLabMode[] = ['chain', 'burst', 'single']
const INTENSITY_OPTIONS = [
  { value: 'scale', label: 'scale' },
  { value: 'demo', label: 'control' },
] as const

const CHANNEL_OPTIONS = ['UPI', 'IMPS', 'NEFT', 'RTGS', 'NETBANKING', 'MOBILE', 'POS'] as const
const REGION_OPTIONS = [
  ['mumbai', 'Mumbai'],
  ['delhi', 'Delhi NCR'],
  ['kolkata', 'Kolkata'],
  ['chennai', 'Chennai'],
  ['bengaluru', 'Bengaluru'],
  ['hyderabad', 'Hyderabad'],
  ['lucknow', 'Lucknow'],
  ['jaipur', 'Jaipur'],
  ['guwahati', 'Guwahati'],
  ['ahmedabad', 'Ahmedabad'],
  ['pune', 'Pune'],
  ['patna', 'Patna'],
] as const
const PROFILE_OPTIONS = [
  ['student', 'Student savings'],
  ['salary', 'Salary account'],
  ['merchant', 'Merchant current'],
  ['senior', 'Senior citizen'],
  ['dormant', 'Dormant account'],
  ['shell', 'Shell/current cluster'],
] as const
const RISK_BIAS_OPTIONS = [
  ['balanced', 'Balanced'],
  ['stealth', 'Stealth'],
  ['aggressive', 'Aggressive'],
] as const
const REQUIRED_EVALUATION_STAGES = [
  ['events_injected', 'Ingested'],
  ['ingested', 'Validated'],
  ['ml_scored', 'ML scored'],
  ['graph_investigated', 'Graph checked'],
  ['cb_evaluated', 'Control gate'],
  ['pipeline_dispatched', 'Dispatched'],
  ['qwen_context_loaded', 'Qwen context'],
] as const
const REPORT_READY_STAGES = ['evaluation_complete', 'evidence_ready'] as const
const REPORT_VISUAL_STAGES = ['ingested', 'ml_scored', 'graph_investigated', 'cb_evaluated', 'llm_started', 'verdict'] as const

function hasRunStage(run: EventLabRunResponse | undefined | null, stage: string) {
  return Boolean(run?.stages?.some((item) => item.stage === stage))
}

function isBackendReportReady(run?: EventLabRunResponse | null) {
  if (!run || run.status !== 'evaluated' || !isUsableAnalysisReport(run.analysis_report)) return false
  if (!REPORT_READY_STAGES.every((stage) => hasRunStage(run, stage))) return false

  const coverage = run.analysis_report.stage_coverage ?? {}
  const reportCoversFinalStages = REPORT_READY_STAGES.every((stage) => Number(coverage[stage] ?? 0) > 0)
  const readyTimestamp = Math.max(
    ...REPORT_READY_STAGES.map((stage) => run.stages.find((item) => item.stage === stage)?.timestamp ?? 0),
  )
  const reportTimestamp = Number(run.analysis_report.generated_at ?? 0)
  const reportGeneratedAfterFinalStages = !Number.isFinite(reportTimestamp) || reportTimestamp === 0 || reportTimestamp >= readyTimestamp - 0.25

  return reportCoversFinalStages && reportGeneratedAfterFinalStages
}

function getReportVisualEvidence(run: EventLabRunResponse | undefined | null, events: Map<string, EventLifecycle>) {
  const runIds = run?.event_ids ?? []
  const lifecycles = runIds.map((id) => events.get(id)).filter((item): item is EventLifecycle => Boolean(item))
  const verdictCount = lifecycles.filter((lifecycle) => lifecycle.stages.some((stage) => stage.stage === 'verdict')).length
  const fullStageCount = lifecycles.filter((lifecycle) => {
    const names = new Set(lifecycle.stages.map((stage) => stage.stage))
    return REPORT_VISUAL_STAGES.every((stage) => names.has(stage))
  }).length
  const report = run?.analysis_report
  const backendReportHasEvidence = Boolean(
    report &&
    (report.stage_timeline?.length ?? 0) >= REQUIRED_EVALUATION_STAGES.length &&
    (report.evidence_matrix?.length ?? 0) > 0 &&
    (report.amount_series?.length ?? 0) > 0,
  )
  return {
    generatedCount: runIds.length,
    hydratedCount: lifecycles.length,
    verdictCount,
    fullStageCount,
    backendReportHasEvidence,
    ready:
      runIds.length > 0 &&
      (
        backendReportHasEvidence ||
        (
          lifecycles.length >= runIds.length &&
          verdictCount >= runIds.length &&
          fullStageCount > 0
        )
      ),
  }
}

function isUsableAnalysisReport(report?: EventLabAnalysisReport | null): report is EventLabAnalysisReport {
  const maybe = report as Partial<EventLabAnalysisReport> | undefined
  return Boolean(maybe && typeof maybe.risk_score === 'number' && typeof maybe.risk_tier === 'string' && maybe.risk_tier.length > 0)
}

function runFreshnessScore(run?: EventLabRunResponse | null) {
  if (!run) return -1
  const statusWeight = run.status === 'evaluated' ? 1_000_000 : run.status === 'injected' ? 100_000 : 0
  const stageWeight = (run.stages?.length ?? 0) * 1_000
  const reportTimestamp = Number(run.analysis_report?.generated_at ?? 0)
  const reportWeight = isUsableAnalysisReport(run.analysis_report) ? 10_000 : 0
  return statusWeight + stageWeight + reportWeight + (Number.isFinite(reportTimestamp) ? reportTimestamp : 0)
}

function selectCanonicalRun(
  directRun?: EventLabRunResponse,
  explainabilityRun?: EventLabRunResponse,
  localRun?: EventLabRunResponse,
) {
  return [directRun, explainabilityRun, localRun]
    .filter((candidate): candidate is EventLabRunResponse => Boolean(candidate?.run_id))
    .sort((a, b) => runFreshnessScore(b) - runFreshnessScore(a))[0]
}

function fmtPct(value?: number) {
  if (value == null || Number.isNaN(value)) return 'n/a'
  return `${Math.round(value * 100)}%`
}

function fmtAmount(paisa?: number) {
  if (!paisa) return 'INR 0'
  return `INR ${(paisa / 100).toLocaleString('en-IN', { maximumFractionDigits: 0 })}`
}

function fmtSeconds(seconds?: number) {
  if (seconds == null || !Number.isFinite(seconds)) return 'n/a'
  if (seconds < 60) return `${Math.max(0, Math.round(seconds))}s`
  return `${Math.floor(seconds / 60)}m ${Math.round(seconds % 60)}s`
}

function fmtScore(value?: number) {
  if (value == null || Number.isNaN(value)) return 'n/a'
  return `${Math.round(value * 100)}%`
}

function fmtCount(value: unknown) {
  const n = Number(value)
  return Number.isFinite(n) ? n.toLocaleString('en-IN') : '0'
}

function compactInr(paisa?: number) {
  const rupees = Number(paisa ?? 0) / 100
  if (rupees >= 10_000_000) return `INR ${(rupees / 10_000_000).toFixed(1)}Cr`
  if (rupees >= 100_000) return `INR ${(rupees / 100_000).toFixed(1)}L`
  return fmtAmount(paisa)
}

function regionPoint(lat: unknown, lon: unknown) {
  const latitude = Number(lat)
  const longitude = Number(lon)
  if (!Number.isFinite(latitude) || !Number.isFinite(longitude)) return null
  const x = Math.min(96, Math.max(4, ((longitude - 68) / 30) * 100))
  const y = Math.min(96, Math.max(4, ((36 - latitude) / 28) * 100))
  return { x, y }
}

function short(value?: string, left = 8) {
  if (!value) return 'n/a'
  return value.length > left + 4 ? `${value.slice(0, left)}...` : value
}

function publicEvidenceText(value: string | undefined, fallback = 'Evidence summary unavailable') {
  return sanitizeOptionalEvidenceText(value) ?? fallback
}

function publicEvidenceItems(items: string[]) {
  return items
    .map((item) => sanitizeOptionalEvidenceText(item) ?? '')
    .filter(Boolean)
}

function Panel({
  title,
  icon: Icon,
  badge,
  children,
  className,
}: {
  title: string
  icon: typeof Radar
  badge?: string
  children: ReactNode
  className?: string
}) {
  return (
    <section className={cn('rounded-lg border border-border-default bg-bg-surface shadow-sm', className)}>
      <div className="flex items-center justify-between gap-3 border-b border-border-subtle px-3 py-2.5">
        <div className="flex min-w-0 items-center gap-2">
          <Icon className="h-4 w-4 shrink-0 text-accent-primary" />
          <h3 className="truncate text-[11px] font-bold uppercase tracking-[0.12em] text-text-primary">{title}</h3>
        </div>
        {badge && (
          <span className="shrink-0 rounded-full border border-accent-primary/20 bg-accent-muted px-2 py-0.5 text-[9px] font-bold uppercase tracking-[0.1em] text-accent-primary">
            {badge}
          </span>
        )}
      </div>
      {children}
    </section>
  )
}

function Metric({ label, value, tone = 'blue' }: { label: string; value: string; tone?: 'blue' | 'green' | 'red' | 'amber' }) {
  const toneClass = {
    blue: 'text-accent-primary',
    green: 'text-alert-low',
    red: 'text-alert-critical',
    amber: 'text-alert-medium',
  }[tone]
  return (
    <div className="rounded-md border border-border-subtle bg-bg-elevated/60 p-2">
      <div className="text-[8px] font-bold uppercase tracking-[0.12em] text-text-muted">{label}</div>
      <div className={cn('mt-1 font-mono text-sm font-bold tabular-nums', toneClass)}>{value}</div>
    </div>
  )
}

function TemplateCard({
  template,
  selected,
  onSelect,
}: {
  template: EventLabTemplate
  selected: boolean
  onSelect: () => void
}) {
  const linked = template.linked_playbooks?.[0]
  return (
    <button
      type="button"
      onClick={onSelect}
      className={cn(
        'w-full rounded-md border p-3 text-left transition-all',
        selected
          ? 'border-accent-primary bg-accent-muted shadow-sm'
          : 'border-border-subtle bg-bg-elevated/50 hover:border-accent-primary/50 hover:bg-bg-surface',
      )}
    >
      <div className="flex items-start justify-between gap-3">
        <div className="min-w-0">
          <div className="line-clamp-1 text-[12px] font-bold text-text-primary">{template.title}</div>
          <p className="mt-1 line-clamp-2 text-[10px] leading-relaxed text-text-secondary">{template.description}</p>
        </div>
        <span className={cn(
          'shrink-0 rounded-full border px-2 py-0.5 text-[8px] font-bold uppercase tracking-[0.1em]',
          template.execution_allowed
            ? 'border-alert-low/25 bg-alert-low/10 text-alert-low'
            : 'border-alert-medium/30 bg-alert-medium/10 text-alert-medium',
        )}>
          {template.execution_allowed ? 'executable' : 'advisory'}
        </span>
      </div>
      <div className="mt-2 flex flex-wrap gap-1">
        {template.typologies.slice(0, 4).map((tag) => (
          <span key={tag} className="rounded bg-bg-surface px-1.5 py-0.5 text-[8px] font-semibold uppercase tracking-wide text-text-secondary">
            {tag.replaceAll('_', ' ')}
          </span>
        ))}
      </div>
      <div className="mt-2 flex items-center justify-between gap-2 text-[9px] text-text-muted">
        <span>{template.channels.join(' / ')}</span>
        <span>{linked ? `PBK ${short(linked.playbook_id, 6)}` : 'no linked playbook'}</span>
      </div>
    </button>
  )
}

function EventPreview({ events }: { events: EventLabGeneratedEvent[] }) {
  return (
    <div className="max-h-[360px] space-y-2 overflow-y-auto pr-1">
      {events.map((event) => (
        <div key={event.event_id} className="rounded-md border border-border-subtle bg-bg-elevated/55 p-2">
          <div className="flex items-start justify-between gap-2">
            <div className="min-w-0">
              <div className="flex items-center gap-2">
                <span className="rounded bg-accent-primary px-1.5 py-0.5 font-mono text-[8px] font-bold text-white">
                  {String(event.sequence + 1).padStart(2, '0')}
                </span>
                <span className="truncate text-[11px] font-semibold text-text-primary">{event.narrative}</span>
              </div>
              <div className="mt-1 font-mono text-[9px] text-text-muted">
                {short(event.sender)} {event.receiver ? '->' : ''} {short(event.receiver || event.account)} | {event.channel ?? event.action ?? event.type}
              </div>
            </div>
            <div className="shrink-0 text-right">
              <div className="font-mono text-[10px] font-bold text-text-primary">{fmtAmount(event.amount_paisa)}</div>
              <div className="mt-0.5 text-[8px] uppercase tracking-wide text-text-muted">{event.type}</div>
            </div>
          </div>
          <div className="mt-2 flex flex-wrap gap-1">
            {event.fraud_label && <span className="rounded bg-alert-critical/10 px-1.5 py-0.5 text-[8px] font-semibold text-alert-critical">{event.fraud_label.replaceAll('_', ' ')}</span>}
            {event.counterparty_role && <span className="rounded bg-bg-surface px-1.5 py-0.5 text-[8px] text-text-secondary">{event.counterparty_role}</span>}
            <span className="rounded bg-bg-surface px-1.5 py-0.5 font-mono text-[8px] text-text-muted">{short(event.event_id, 10)}</span>
          </div>
        </div>
      ))}
    </div>
  )
}

function ScenarioControlPanel({
  controls,
  onChange,
}: {
  controls: EventLabControls
  onChange: (next: EventLabControls) => void
}) {
  const setControl = <K extends keyof EventLabControls>(key: K, value: EventLabControls[K]) => {
    onChange({ ...controls, [key]: value })
  }
  const numberValue = (value: unknown) => value == null ? '' : String(value)

  return (
    <div className="rounded-lg border border-[#00579C]/20 bg-[#00579C]/5 p-3">
      <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
        <div className="flex items-center gap-2 text-[10px] font-bold uppercase tracking-[0.12em] text-accent-primary">
          <Gauge className="h-3.5 w-3.5" />
          Fraud event customization
        </div>
        <span className="rounded-full border border-[#DA251C]/20 bg-[#DA251C]/10 px-2 py-0.5 text-[8px] font-bold uppercase tracking-[0.1em] text-[#DA251C]">
          backend-generated, not mock rows
        </span>
      </div>

      <div className="grid gap-2 md:grid-cols-4">
        <label className="space-y-1">
          <span className="block text-[8px] font-bold uppercase tracking-[0.12em] text-text-muted">Events</span>
          <input
            type="number"
            min={1}
            max={30}
            value={numberValue(controls.event_count)}
            onChange={(event) => setControl('event_count', Number(event.target.value) || null)}
            className="h-9 w-full rounded-md border border-border-subtle bg-bg-surface px-2 font-mono text-[11px] text-text-primary outline-none focus:border-accent-primary"
          />
        </label>
        <label className="space-y-1">
          <span className="block text-[8px] font-bold uppercase tracking-[0.12em] text-text-muted">Min INR</span>
          <input
            type="number"
            min={100}
            value={numberValue(controls.min_amount_inr)}
            onChange={(event) => setControl('min_amount_inr', Number(event.target.value) || null)}
            className="h-9 w-full rounded-md border border-border-subtle bg-bg-surface px-2 font-mono text-[11px] text-text-primary outline-none focus:border-accent-primary"
          />
        </label>
        <label className="space-y-1">
          <span className="block text-[8px] font-bold uppercase tracking-[0.12em] text-text-muted">Max INR</span>
          <input
            type="number"
            min={100}
            value={numberValue(controls.max_amount_inr)}
            onChange={(event) => setControl('max_amount_inr', Number(event.target.value) || null)}
            className="h-9 w-full rounded-md border border-border-subtle bg-bg-surface px-2 font-mono text-[11px] text-text-primary outline-none focus:border-accent-primary"
          />
        </label>
        <label className="space-y-1">
          <span className="block text-[8px] font-bold uppercase tracking-[0.12em] text-text-muted">Velocity min</span>
          <input
            type="number"
            min={1}
            max={360}
            value={numberValue(controls.velocity_minutes)}
            onChange={(event) => setControl('velocity_minutes', Number(event.target.value) || null)}
            className="h-9 w-full rounded-md border border-border-subtle bg-bg-surface px-2 font-mono text-[11px] text-text-primary outline-none focus:border-accent-primary"
          />
        </label>
      </div>

      <div className="mt-2 grid gap-2 md:grid-cols-4">
        <label className="space-y-1">
          <span className="block text-[8px] font-bold uppercase tracking-[0.12em] text-text-muted">Primary rail</span>
          <select
            value={controls.primary_channel ?? 'UPI'}
            onChange={(event) => setControl('primary_channel', event.target.value)}
            className="h-9 w-full rounded-md border border-border-subtle bg-bg-surface px-2 text-[10px] font-bold uppercase text-text-primary outline-none focus:border-accent-primary"
          >
            {CHANNEL_OPTIONS.map((channel) => <option key={channel} value={channel}>{channel}</option>)}
          </select>
        </label>
        <label className="space-y-1">
          <span className="block text-[8px] font-bold uppercase tracking-[0.12em] text-text-muted">Secondary rail</span>
          <select
            value={controls.secondary_channel ?? 'IMPS'}
            onChange={(event) => setControl('secondary_channel', event.target.value)}
            className="h-9 w-full rounded-md border border-border-subtle bg-bg-surface px-2 text-[10px] font-bold uppercase text-text-primary outline-none focus:border-accent-primary"
          >
            {CHANNEL_OPTIONS.map((channel) => <option key={channel} value={channel}>{channel}</option>)}
          </select>
        </label>
        <label className="space-y-1">
          <span className="block text-[8px] font-bold uppercase tracking-[0.12em] text-text-muted">Origin city</span>
          <select
            value={controls.origin_region ?? 'kolkata'}
            onChange={(event) => setControl('origin_region', event.target.value)}
            className="h-9 w-full rounded-md border border-border-subtle bg-bg-surface px-2 text-[10px] font-semibold text-text-primary outline-none focus:border-accent-primary"
          >
            {REGION_OPTIONS.map(([value, label]) => <option key={value} value={value}>{label}</option>)}
          </select>
        </label>
        <label className="space-y-1">
          <span className="block text-[8px] font-bold uppercase tracking-[0.12em] text-text-muted">Destination</span>
          <select
            value={controls.destination_region ?? 'delhi'}
            onChange={(event) => setControl('destination_region', event.target.value)}
            className="h-9 w-full rounded-md border border-border-subtle bg-bg-surface px-2 text-[10px] font-semibold text-text-primary outline-none focus:border-accent-primary"
          >
            {REGION_OPTIONS.map(([value, label]) => <option key={value} value={value}>{label}</option>)}
          </select>
        </label>
      </div>

      <div className="mt-2 grid gap-2 md:grid-cols-[140px_minmax(0,1fr)_minmax(0,1fr)]">
        <label className="space-y-1">
          <span className="block text-[8px] font-bold uppercase tracking-[0.12em] text-text-muted">Mule depth</span>
          <input
            type="number"
            min={1}
            max={14}
            value={numberValue(controls.mule_depth)}
            onChange={(event) => setControl('mule_depth', Number(event.target.value) || null)}
            className="h-9 w-full rounded-md border border-border-subtle bg-bg-surface px-2 font-mono text-[11px] text-text-primary outline-none focus:border-accent-primary"
          />
        </label>
        <label className="space-y-1">
          <span className="block text-[8px] font-bold uppercase tracking-[0.12em] text-text-muted">Customer profile</span>
          <select
            value={controls.customer_profile ?? 'student'}
            onChange={(event) => setControl('customer_profile', event.target.value)}
            className="h-9 w-full rounded-md border border-border-subtle bg-bg-surface px-2 text-[10px] font-semibold text-text-primary outline-none focus:border-accent-primary"
          >
            {PROFILE_OPTIONS.map(([value, label]) => <option key={value} value={value}>{label}</option>)}
          </select>
        </label>
        <div className="space-y-1">
          <span className="block text-[8px] font-bold uppercase tracking-[0.12em] text-text-muted">Risk posture</span>
          <div className="grid grid-cols-3 gap-1">
            {RISK_BIAS_OPTIONS.map(([value, label]) => (
              <button
                key={value}
                type="button"
                onClick={() => setControl('risk_bias', value)}
                className={cn(
                  'h-9 rounded-md border px-2 text-[9px] font-bold uppercase tracking-wide',
                  controls.risk_bias === value ? 'border-[#DA251C] bg-[#DA251C] text-white' : 'border-border-subtle bg-bg-surface text-text-secondary',
                )}
              >
                {label}
              </button>
            ))}
          </div>
        </div>
      </div>

      <div className="mt-3 grid gap-2 md:grid-cols-3">
        {([
          ['device_reuse', 'Shared device fingerprint'],
          ['include_auth_signal', 'OTP/auth precursor'],
          ['include_interbank_leg', 'Interbank exit leg'],
        ] as const).map(([key, label]) => (
          <button
            key={key}
            type="button"
            onClick={() => setControl(key, !controls[key])}
            className={cn(
              'flex h-9 items-center justify-between rounded-md border px-3 text-left text-[9px] font-bold uppercase tracking-[0.09em]',
              controls[key]
                ? 'border-accent-primary/35 bg-accent-muted text-accent-primary'
                : 'border-border-subtle bg-bg-surface text-text-secondary',
            )}
          >
            <span>{label}</span>
            <span className={cn('h-2 w-2 rounded-full', controls[key] ? 'bg-accent-primary' : 'bg-text-muted/35')} />
          </button>
        ))}
      </div>
    </div>
  )
}

function chartPoints(values: number[], width = 100, height = 42, pad = 4) {
  const max = Math.max(1, ...values)
  const usableWidth = width - pad * 2
  const usableHeight = height - pad * 2
  return values.map((value, index) => {
    const x = values.length <= 1 ? width / 2 : pad + (index / (values.length - 1)) * usableWidth
    const y = height - pad - (value / max) * usableHeight
    return { x, y, value }
  })
}

function linePath(points: Array<{ x: number; y: number }>) {
  return points.map((point, index) => `${index === 0 ? 'M' : 'L'} ${point.x.toFixed(1)} ${point.y.toFixed(1)}`).join(' ')
}

function ReportChartFrame({
  title,
  badge,
  children,
}: {
  title: string
  badge?: string
  children: ReactNode
}) {
  return (
    <div className="rounded-md border border-border-subtle bg-bg-elevated/45 p-3">
      <div className="mb-2 flex items-center justify-between gap-2">
        <div className="text-[9px] font-bold uppercase tracking-[0.12em] text-text-primary">{title}</div>
        {badge && <span className="truncate font-mono text-[8px] text-text-muted">{badge}</span>}
      </div>
      {children}
    </div>
  )
}

function AmountVelocityPlot({
  amountSeries,
  velocitySeries,
}: {
  amountSeries: Array<Record<string, unknown>>
  velocitySeries: Array<Record<string, unknown>>
}) {
  const amounts = amountSeries.map((item) => Number(item.amount_paisa ?? 0)).filter((value) => Number.isFinite(value) && value > 0)
  const elapsed = velocitySeries.map((item, index) => Number(item.elapsed_minutes ?? index)).filter((value) => Number.isFinite(value))
  const amountPoints = chartPoints(amounts.length ? amounts : [0], 100, 42, 5)
  const elapsedMax = Math.max(1, ...elapsed)
  return (
    <ReportChartFrame title="Amount and velocity plot" badge={`${amounts.length} amount legs`}>
      <svg viewBox="0 0 100 42" className="h-36 w-full rounded-md border border-[#00579C]/15 bg-white">
        {[10, 20, 30].map((y) => <line key={y} x1="4" x2="96" y1={y} y2={y} stroke="#d9e6f3" strokeDasharray="2 2" strokeWidth="0.4" />)}
        <path d={linePath(amountPoints)} fill="none" stroke="#DA251C" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
        <path d={`${linePath(amountPoints)} L ${amountPoints.at(-1)?.x ?? 95} 39 L ${amountPoints[0]?.x ?? 5} 39 Z`} fill="rgba(218,37,28,0.08)" stroke="none" />
        {amountPoints.map((point, index) => (
          <circle key={`${point.x}-${index}`} cx={point.x} cy={point.y} r="1.3" fill={index === amountPoints.length - 1 ? '#DA251C' : '#00579C'} />
        ))}
        {elapsed.slice(0, 10).map((value, index) => {
          const x = 5 + (index / Math.max(1, Math.min(10, elapsed.length) - 1)) * 90
          const h = Math.max(2, (value / elapsedMax) * 16)
          return <rect key={`${value}-${index}`} x={x - 0.9} y={39 - h} width="1.8" height={h} rx="0.6" fill="#00579C" opacity="0.32" />
        })}
        <text x="5" y="7" fontSize="3.4" fill="#52657d" fontFamily="monospace">amount line</text>
        <text x="64" y="7" fontSize="3.4" fill="#52657d" fontFamily="monospace">velocity bars</text>
      </svg>
    </ReportChartFrame>
  )
}

function ChannelDistribution({ rows, total }: { rows: Array<[string, number]>; total: number }) {
  let cursor = 0
  const colors = ['#00579C', '#DA251C', '#2f79b5', '#f5b400', '#17324d', '#6b7280']
  const gradient = rows.length
    ? rows.map(([, count], index) => {
      const start = cursor
      const end = cursor + (count / Math.max(1, total)) * 100
      cursor = end
      return `${colors[index % colors.length]} ${start.toFixed(2)}% ${end.toFixed(2)}%`
    }).join(', ')
    : '#d9e6f3 0% 100%'
  return (
    <ReportChartFrame title="Channel distribution" badge={`${total} observed events`}>
      <div className="grid grid-cols-[88px_minmax(0,1fr)] items-center gap-3">
        <div
          className="h-20 w-20 rounded-full border border-[#00579C]/20 shadow-inner"
          style={{ background: `conic-gradient(${gradient})` }}
        >
          <div className="m-5 h-10 w-10 rounded-full border border-border-subtle bg-bg-surface" />
        </div>
        <div className="space-y-1.5">
          {rows.map(([channel, count], index) => (
            <div key={channel} className="grid grid-cols-[12px_52px_minmax(0,1fr)_32px] items-center gap-2 text-[8px]">
              <span className="h-2.5 w-2.5 rounded-sm" style={{ background: colors[index % colors.length] }} />
              <span className="font-mono font-bold text-text-secondary">{channel}</span>
              <div className="h-1.5 rounded bg-white">
                <div className="h-full rounded" style={{ width: `${Math.max(4, (count / Math.max(1, total)) * 100)}%`, background: colors[index % colors.length] }} />
              </div>
              <span className="text-right font-mono font-bold text-text-primary">{count}</span>
            </div>
          ))}
        </div>
      </div>
    </ReportChartFrame>
  )
}

function RiskFlagDistribution({ rows }: { rows: Array<[string, number]> }) {
  const max = Math.max(1, ...rows.map(([, count]) => count))
  return (
    <ReportChartFrame title="Risk flag distribution" badge={`${rows.length} dominant flags`}>
      <div className="grid gap-1.5">
        {rows.length === 0 ? (
          <div className="rounded border border-dashed border-border-subtle p-2 text-[9px] text-text-muted">No dominant flags were returned for this run.</div>
        ) : rows.map(([flag, count]) => (
          <div key={flag} className="grid grid-cols-[112px_minmax(0,1fr)_28px] items-center gap-2 text-[8px]">
            <span className="truncate font-bold uppercase tracking-wide text-[#DA251C]">{flag.replaceAll('_', ' ')}</span>
            <div className="h-4 overflow-hidden rounded bg-white">
              <div className="h-full rounded bg-[linear-gradient(90deg,#DA251C,#f5b400)]" style={{ width: `${Math.max(6, (count / max) * 100)}%` }} />
            </div>
            <span className="text-right font-mono font-bold text-text-primary">{count}</span>
          </div>
        ))}
      </div>
    </ReportChartFrame>
  )
}

function RiskScoreWaterfall({ rows }: { rows: Array<[string, number]> }) {
  const total = rows.reduce((sum, [, value]) => sum + Math.max(0, Number(value) || 0), 0)
  return (
    <ReportChartFrame title="Risk score contribution" badge={`${Math.round(total * 100)} raw pts`}>
      <div className="space-y-1.5">
        {rows.length === 0 ? (
          <div className="rounded border border-dashed border-border-subtle p-2 text-[9px] text-text-muted">Risk components were not returned for this run.</div>
        ) : rows.map(([label, value]) => {
          const pct = Math.max(0, Math.min(100, value * 100))
          return (
            <div key={label} className="grid grid-cols-[116px_minmax(0,1fr)_36px] items-center gap-2 text-[8px]">
              <span className="truncate font-bold uppercase tracking-wide text-text-secondary">{label.replaceAll('_', ' ')}</span>
              <div className="h-4 overflow-hidden rounded bg-white">
                <div className="h-full rounded bg-[linear-gradient(90deg,#00579C,#DA251C)]" style={{ width: `${Math.max(5, pct)}%` }} />
              </div>
              <span className="text-right font-mono font-bold text-text-primary">{Math.round(pct)}%</span>
            </div>
          )
        })}
      </div>
    </ReportChartFrame>
  )
}

function TimelineBucketChart({ rows }: { rows: Array<Record<string, unknown>> }) {
  const maxAmount = Math.max(1, ...rows.map((row) => Number(row.amount_paisa ?? 0)))
  const maxEvents = Math.max(1, ...rows.map((row) => Number(row.event_count ?? 0)))
  return (
    <ReportChartFrame title="Event velocity buckets" badge={`${rows.length} backend buckets`}>
      <svg viewBox="0 0 100 46" className="h-36 w-full rounded-md border border-[#00579C]/15 bg-white">
        {[12, 24, 36].map((y) => <line key={y} x1="5" x2="96" y1={y} y2={y} stroke="#d9e6f3" strokeDasharray="2 2" strokeWidth="0.4" />)}
        {rows.map((row, index) => {
          const x = 8 + index * (84 / Math.max(1, rows.length - 1))
          const amountHeight = Math.max(1.5, (Number(row.amount_paisa ?? 0) / maxAmount) * 30)
          const eventHeight = Math.max(1.5, (Number(row.event_count ?? 0) / maxEvents) * 20)
          return (
            <g key={`${row.bucket}-${index}`}>
              <rect x={x - 2.5} y={40 - amountHeight} width="5" height={amountHeight} rx="1" fill="#DA251C" opacity="0.82" />
              <rect x={x + 3.1} y={40 - eventHeight} width="2.4" height={eventHeight} rx="0.8" fill="#00579C" opacity="0.65" />
              <text x={x - 2.5} y="44" fontSize="2.8" fill="#52657d" fontFamily="monospace">{String(row.bucket ?? index + 1)}</text>
            </g>
          )
        })}
        <text x="6" y="7" fontSize="3.4" fill="#52657d" fontFamily="monospace">red=amount</text>
        <text x="63" y="7" fontSize="3.4" fill="#52657d" fontFamily="monospace">blue=count</text>
      </svg>
    </ReportChartFrame>
  )
}

function ReportRouteMap({
  geoRows,
  routeLabel,
}: {
  geoRows: Array<Record<string, unknown>>
  routeLabel?: string
}) {
  const points = geoRows
    .map((item) => regionPoint(item.lat, item.lon))
    .filter((point): point is { x: number; y: number } => Boolean(point))
    .map((point) => ({ x: point.x, y: Math.min(64, Math.max(4, point.y * 0.62 + 2)) }))
  const pathD = linePath(points)
  const labels = geoRows
    .filter((item, index) => index === 0 || index === geoRows.length - 1)
    .map((item, index) => ({ item, point: points[index === 0 ? 0 : points.length - 1] }))
    .filter((entry): entry is { item: Record<string, unknown>; point: { x: number; y: number } } => Boolean(entry.point))

  return (
    <ReportChartFrame title="Fund route map" badge={routeLabel || 'selected route'}>
      <svg viewBox="0 0 100 68" className="h-56 w-full rounded-md border border-[#00579C]/20 bg-[linear-gradient(135deg,#eef6ff,#ffffff)]">
        <path d="M30 8 C21 22 17 40 24 52 C33 67 53 65 69 58 C83 51 88 34 78 21 C68 8 48 2 30 8Z" fill="#d8eaf8" stroke="#9fc4e4" strokeWidth="0.8" />
        <path d="M20 48 C34 38 48 34 70 24" fill="none" stroke="#00579C" strokeWidth="0.4" opacity="0.25" strokeDasharray="2 2" />
        {pathD && <path d={pathD} fill="none" stroke="#DA251C" strokeWidth="1.9" strokeLinecap="round" strokeLinejoin="round" strokeDasharray="3 2" />}
        {points.map((point, index) => (
          <g key={`${point.x}-${point.y}-${index}`}>
            <circle cx={point.x} cy={point.y} r={index === points.length - 1 ? 3.5 : 2.2} fill={index === 0 ? '#00579C' : '#DA251C'} opacity="0.9" />
            {index % 3 === 0 && <text x={point.x + 2.8} y={point.y - 1.8} fontSize="3" fill="#17324d" fontFamily="monospace">{index + 1}</text>}
          </g>
        ))}
        {labels.map(({ item, point }, index) => (
          <g key={`${String(item.city)}-${index}`}>
            <rect x={Math.min(82, Math.max(3, point.x + 3))} y={Math.max(4, point.y - 7)} width="15" height="5.5" rx="1.2" fill="#071427" opacity="0.82" />
            <text x={Math.min(83, Math.max(4, point.x + 4))} y={Math.max(8, point.y - 3.2)} fontSize="2.6" fill="#ffffff" fontFamily="monospace">{String(item.city ?? 'node').slice(0, 8)}</text>
          </g>
        ))}
      </svg>
    </ReportChartFrame>
  )
}

function StageTimelineChart({ rows }: { rows: Array<Record<string, unknown>> }) {
  const visible = rows.slice(-12)
  const maxEvents = Math.max(1, ...visible.map((row) => Number(row.event_count ?? 0)))
  return (
    <ReportChartFrame title="Backend stage timeline" badge={`${rows.length} recorded stages`}>
      <div className="space-y-1.5">
        {visible.length === 0 ? (
          <div className="rounded border border-dashed border-border-subtle p-2 text-[9px] text-text-muted">No backend stage timeline was returned.</div>
        ) : visible.map((row, index) => {
          const eventCount = Number(row.event_count ?? 0)
          const width = Math.max(8, (eventCount / maxEvents) * 100)
          return (
            <div key={`${String(row.stage)}-${String(row.timestamp)}-${index}`} className="grid grid-cols-[24px_132px_minmax(0,1fr)_54px] items-center gap-2 text-[8px]">
              <span className="font-mono font-bold text-text-muted">{String(row.sequence ?? index + 1).padStart(2, '0')}</span>
              <span className="truncate font-bold uppercase tracking-wide text-text-secondary">{String(row.stage ?? '').replaceAll('_', ' ')}</span>
              <div className="h-4 overflow-hidden rounded bg-white">
                <div className="h-full rounded bg-[linear-gradient(90deg,#00579C,#2f79b5)]" style={{ width: `${width}%` }} />
              </div>
              <span className="text-right font-mono font-bold text-text-primary">{eventCount} ids</span>
            </div>
          )
        })}
      </div>
    </ReportChartFrame>
  )
}

function ChannelExposureStack({
  countRows,
  amountRows,
}: {
  countRows: Array<[string, number]>
  amountRows: Array<[string, number]>
}) {
  const amountLookup = new Map(amountRows)
  const rows = countRows
    .map(([channel, count]) => ({ channel, count, amount: Number(amountLookup.get(channel) ?? 0) }))
    .sort((a, b) => b.amount - a.amount || b.count - a.count)
  const maxAmount = Math.max(1, ...rows.map((row) => row.amount))
  return (
    <ReportChartFrame title="Channel exposure stack" badge={`${rows.length} rails`}>
      <div className="space-y-1.5">
        {rows.length === 0 ? (
          <div className="rounded border border-dashed border-border-subtle p-2 text-[9px] text-text-muted">No channel exposure rows were returned.</div>
        ) : rows.map((row) => (
          <div key={row.channel} className="grid grid-cols-[58px_minmax(0,1fr)_82px_32px] items-center gap-2 text-[8px]">
            <span className="font-mono font-bold text-[#00579C]">{row.channel}</span>
            <div className="h-5 overflow-hidden rounded bg-white">
              <div className="h-full rounded bg-[linear-gradient(90deg,#00579C,#DA251C)]" style={{ width: `${Math.max(5, (row.amount / maxAmount) * 100)}%` }} />
            </div>
            <span className="text-right font-mono font-bold text-text-primary">{compactInr(row.amount)}</span>
            <span className="text-right font-mono text-text-muted">{row.count}</span>
          </div>
        ))}
      </div>
    </ReportChartFrame>
  )
}

function AccountRoleMix({ rows }: { rows: Array<[string, number]> }) {
  const max = Math.max(1, ...rows.map(([, count]) => count))
  return (
    <ReportChartFrame title="Account role mix" badge={`${rows.length} roles`}>
      <div className="grid gap-1.5">
        {rows.length === 0 ? (
          <div className="rounded border border-dashed border-border-subtle p-2 text-[9px] text-text-muted">No account roles were returned.</div>
        ) : rows.slice(0, 8).map(([role, count]) => (
          <div key={role} className="grid grid-cols-[122px_minmax(0,1fr)_28px] items-center gap-2 text-[8px]">
            <span className="truncate font-bold uppercase tracking-wide text-text-secondary">{role.replaceAll('_', ' ')}</span>
            <div className="h-4 overflow-hidden rounded bg-white">
              <div className="h-full rounded bg-[linear-gradient(90deg,#2f79b5,#00579C)]" style={{ width: `${Math.max(8, (count / max) * 100)}%` }} />
            </div>
            <span className="text-right font-mono font-bold text-text-primary">{count}</span>
          </div>
        ))}
      </div>
    </ReportChartFrame>
  )
}

function RouteSegmentLedger({ rows }: { rows: Array<Record<string, unknown>> }) {
  return (
    <ReportChartFrame title="Route segment ledger" badge={`${rows.length} monetary legs`}>
      <div className="max-h-48 space-y-1.5 overflow-y-auto pr-1">
        {rows.length === 0 ? (
          <div className="rounded border border-dashed border-border-subtle p-2 text-[9px] text-text-muted">No route segments were returned.</div>
        ) : rows.slice(0, 14).map((row, index) => (
          <div key={`${String(row.event_id)}-${index}`} className="rounded border border-border-subtle bg-white px-2 py-1.5">
            <div className="grid grid-cols-[28px_minmax(0,1fr)_84px] items-center gap-2">
              <span className="rounded bg-[#00579C] px-1.5 py-0.5 text-center font-mono text-[8px] font-bold text-white">{String(row.sequence ?? index + 1).padStart(2, '0')}</span>
              <div className="min-w-0">
                <div className="truncate text-[9px] font-bold text-text-primary">
                  {String(row.from_account ?? 'source')} -&gt; {String(row.to_account ?? 'destination')}
                </div>
                <div className="mt-0.5 truncate text-[8px] text-text-muted">
                  {String(row.from_city ?? 'city')} to {String(row.to_city ?? 'city')} | {String(row.channel ?? 'rail')} | {String(row.role ?? 'role')}
                </div>
              </div>
              <span className="text-right font-mono text-[9px] font-bold text-[#DA251C]">{compactInr(Number(row.amount_paisa ?? 0))}</span>
            </div>
          </div>
        ))}
      </div>
    </ReportChartFrame>
  )
}

function EvidenceMatrixHeatmap({ rows }: { rows: Array<Record<string, unknown>> }) {
  const visible = rows.slice(0, 18)
  return (
    <ReportChartFrame title="Evidence matrix heatmap" badge={`${rows.length} evidence rows`}>
      <div className="grid gap-1.5 md:grid-cols-2">
        {visible.length === 0 ? (
          <div className="md:col-span-2 rounded border border-dashed border-border-subtle p-2 text-[9px] text-text-muted">No evidence matrix rows were returned.</div>
        ) : visible.map((row, index) => {
          const weight = Math.max(0, Math.min(1, Number(row.weight ?? 0)))
          return (
            <div key={`${String(row.signal)}-${index}`} className="rounded border border-border-subtle bg-white p-2">
              <div className="flex items-center justify-between gap-2">
                <span className="truncate text-[8px] font-bold uppercase tracking-wide text-text-primary">{String(row.signal ?? 'signal').replaceAll('_', ' ')}</span>
                <span className="font-mono text-[8px] font-bold text-[#00579C]">{Math.round(weight * 100)}%</span>
              </div>
              <div className="mt-1 h-2 overflow-hidden rounded bg-[#eef5fb]">
                <div className="h-full rounded bg-[linear-gradient(90deg,#00579C,#f5b400,#DA251C)]" style={{ width: `${Math.max(4, weight * 100)}%` }} />
              </div>
              <div className="mt-1 truncate text-[7px] font-semibold uppercase tracking-wide text-text-muted">
                {String(row.source ?? 'source')} | count {String(row.count ?? 0)}
              </div>
            </div>
          )
        })}
      </div>
    </ReportChartFrame>
  )
}

function CountermeasureMatrix({ rows }: { rows: Array<Record<string, unknown>> }) {
  return (
    <ReportChartFrame title="Countermeasure matrix" badge={`${rows.length} proposals`}>
      <div className="max-h-44 space-y-1.5 overflow-y-auto pr-1">
        {rows.length === 0 ? (
          <div className="rounded border border-dashed border-border-subtle p-2 text-[9px] text-text-muted">No countermeasure proposals were returned.</div>
        ) : rows.map((row, index) => (
          <div key={`${String(row.proposal_id)}-${index}`} className="grid grid-cols-[minmax(0,1fr)_72px_74px_74px] items-center gap-2 rounded border border-border-subtle bg-white px-2 py-1.5 text-[8px]">
            <div className="min-w-0">
              <div className="truncate font-bold text-text-primary">{String(row.title ?? row.action ?? 'proposal')}</div>
              <div className="mt-0.5 truncate font-mono text-text-muted">{short(String(row.proposal_id ?? ''), 12)} | {short(String(row.primary_target ?? ''), 10)}</div>
            </div>
            <span className="truncate font-mono font-bold uppercase text-[#00579C]">{String(row.status ?? 'pending')}</span>
            <span className={cn('rounded px-1.5 py-0.5 text-center font-bold uppercase', row.execution_allowed ? 'bg-alert-low/10 text-alert-low' : 'bg-[#DA251C]/10 text-[#DA251C]')}>
              {row.execution_allowed ? 'allowed' : 'gated'}
            </span>
            <span className="text-right font-mono text-text-muted">{fmtSeconds(Number(row.ttl_remaining_seconds ?? 0))}</span>
          </div>
        ))}
      </div>
    </ReportChartFrame>
  )
}

function ReportSignalDensityGrid({
  report,
  run,
}: {
  report: EventLabAnalysisReport
  run?: EventLabRunResponse
}) {
  const amountRows = Array.isArray(report.amount_series) ? report.amount_series : []
  const stageRows = Array.isArray(report.stage_timeline) ? report.stage_timeline : []
  const evidenceRows = Array.isArray(report.evidence_matrix) ? report.evidence_matrix : []
  const routeRows = Array.isArray(report.route_segments) ? report.route_segments : []
  const maxLeg = Math.max(0, ...amountRows.map((row) => Number(row.amount_paisa ?? 0)).filter(Number.isFinite))
  const averageLeg = report.transaction_count > 0
    ? Math.round(Number(report.total_exposure_paisa ?? 0) / report.transaction_count)
    : 0
  const coveredStages = REQUIRED_EVALUATION_STAGES.filter(([stage]) => Number(report.stage_coverage?.[stage] ?? 0) > 0).length
  const coveragePct = coveredStages / Math.max(1, REQUIRED_EVALUATION_STAGES.length)
  const knownLatency = Number(run?.latency_metrics?.known_stage_latency_ms ?? 0)
  const routeStats = report.route_stats ?? {}
  const stats = [
    ['Backend coverage', fmtScore(coveragePct), `${coveredStages}/${REQUIRED_EVALUATION_STAGES.length} required stages`],
    ['Evidence rows', fmtCount(evidenceRows.length), 'heuristics + model + backend stages'],
    ['Route segments', fmtCount(routeRows.length), `${fmtCount(routeStats.segment_count)} segment stats`],
    ['Peak leg', compactInr(maxLeg), 'largest transfer observed'],
    ['Avg leg', compactInr(averageLeg), `${fmtCount(report.transaction_count)} monetary legs`],
    ['Known latency', fmtOptionalMs(knownLatency), `${stageRows.length} timeline records`],
  ] as const

  return (
    <div className="grid gap-2 md:grid-cols-3 xl:grid-cols-6">
      {stats.map(([label, value, detail]) => (
        <div key={label} className="rounded-md border border-[#00579C]/18 bg-[linear-gradient(135deg,#ffffff,#f1f7fd)] p-2">
          <div className="text-[7px] font-black uppercase tracking-[0.12em] text-text-muted">{label}</div>
          <div className="mt-1 truncate font-mono text-[12px] font-black text-[#00579C]">{value}</div>
          <div className="mt-0.5 truncate text-[7px] font-semibold text-text-muted">{detail}</div>
        </div>
      ))}
    </div>
  )
}

function StageLatencySlaPlot({ rows }: { rows: Array<Record<string, unknown>> }) {
  const visible = rows.slice(-14)
  const durations = visible.map((row, index) => {
    const explicit = Number(row.duration_ms)
    if (Number.isFinite(explicit) && explicit > 0) return explicit
    const current = Number(row.timestamp ?? 0)
    const previous = Number(visible[index - 1]?.timestamp ?? 0)
    if (index > 0 && Number.isFinite(current) && Number.isFinite(previous) && current > previous) {
      return Math.round((current - previous) * 1000)
    }
    return 0
  })
  const maxDuration = Math.max(1, ...durations)
  const points = chartPoints(durations.map((value) => Math.max(1, value)), 100, 36, 5)
  const pathD = linePath(points)
  const p95 = [...durations].sort((a, b) => a - b)[Math.max(0, Math.ceil(durations.length * 0.95) - 1)] ?? 0

  return (
    <ReportChartFrame title="Stage latency SLA plot" badge={`p95 ${fmtOptionalMs(p95)}`}>
      <svg viewBox="0 0 100 44" className="h-36 w-full rounded-md border border-[#00579C]/15 bg-white">
        {[12, 22, 32].map((y) => <line key={y} x1="5" x2="96" y1={y} y2={y} stroke="#d9e6f3" strokeDasharray="2 2" strokeWidth="0.35" />)}
        {visible.map((row, index) => {
          const x = 8 + index * (84 / Math.max(1, visible.length - 1))
          const height = Math.max(2, (durations[index] / maxDuration) * 27)
          const stage = String(row.stage ?? '').replaceAll('_', ' ')
          const isFinal = /evaluation|evidence|dispatch|verdict/i.test(stage)
          return (
            <g key={`${String(row.stage)}-${index}`}>
              <rect x={x - 2.1} y={36 - height} width="4.2" height={height} rx="1" fill={isFinal ? '#DA251C' : '#00579C'} opacity="0.78" />
              <text x={x - 3.8} y="41" fontSize="2.2" fill="#52657d" fontFamily="monospace">{index + 1}</text>
            </g>
          )
        })}
        {pathD && <path d={pathD} fill="none" stroke="#f5b400" strokeWidth="1.1" strokeLinecap="round" strokeLinejoin="round" />}
        <text x="6" y="7" fontSize="3.2" fill="#52657d" fontFamily="monospace">bar=stage ms, line=latency shape</text>
      </svg>
      <div className="mt-2 grid gap-1.5">
        {visible.slice(-5).map((row, index) => (
          <div key={`${String(row.stage)}-label-${index}`} className="flex items-center justify-between gap-2 rounded border border-border-subtle bg-white px-2 py-1 text-[8px]">
            <span className="truncate font-bold uppercase tracking-wide text-text-secondary">{String(row.stage ?? '').replaceAll('_', ' ')}</span>
            <span className="font-mono font-bold text-[#00579C]">{fmtOptionalMs(durations[visible.length - 5 + index] ?? 0)}</span>
          </div>
        ))}
      </div>
    </ReportChartFrame>
  )
}

function EvidenceSourceDonut({ rows }: { rows: Array<Record<string, unknown>> }) {
  const grouped = rows.reduce<Record<string, number>>((acc, row) => {
    const source = String(row.source ?? 'unknown').replaceAll('_', ' ')
    const weight = Math.max(0.1, Number(row.weight ?? 0))
    acc[source] = (acc[source] ?? 0) + weight
    return acc
  }, {})
  const entries = Object.entries(grouped).sort((a, b) => b[1] - a[1]).slice(0, 6)
  const total = entries.reduce((sum, [, value]) => sum + value, 0) || 1
  const colors = ['#00579C', '#DA251C', '#f5b400', '#0f9f6e', '#2f79b5', '#7a5a00']
  const segments = entries.map(([source, value], index) => {
    const pct = (value / total) * 100
    const previousPct = entries.slice(0, index).reduce((sum, [, itemValue]) => sum + (itemValue / total) * 100, 0)
    return { source, value, pct, offset: 25 - previousPct }
  })

  return (
    <ReportChartFrame title="Evidence source donut" badge={`${rows.length} rows`}>
      <div className="grid grid-cols-[116px_minmax(0,1fr)] items-center gap-3">
        <svg viewBox="0 0 42 42" className="h-28 w-28">
          <circle cx="21" cy="21" r="15.9" fill="none" stroke="#eef5fb" strokeWidth="6" />
          {segments.map((segment, index) => (
            <circle
              key={segment.source}
              cx="21"
              cy="21"
              r="15.9"
              fill="none"
              stroke={colors[index % colors.length]}
              strokeWidth="6"
              strokeDasharray={`${segment.pct} ${100 - segment.pct}`}
              strokeDashoffset={segment.offset}
              strokeLinecap="round"
              transform="rotate(-90 21 21)"
            />
          ))}
          <text x="21" y="19" textAnchor="middle" fontSize="5" fontFamily="monospace" fontWeight="800" fill="#172033">{rows.length}</text>
          <text x="21" y="25" textAnchor="middle" fontSize="3" fontFamily="monospace" fill="#718096">signals</text>
        </svg>
        <div className="space-y-1.5">
          {entries.length === 0 ? (
            <div className="rounded border border-dashed border-border-subtle p-2 text-[9px] text-text-muted">No evidence source rows were returned.</div>
          ) : entries.map(([source, value], index) => (
            <div key={source} className="grid grid-cols-[8px_minmax(0,1fr)_34px] items-center gap-2 text-[8px]">
              <span className="h-2 w-2 rounded-full" style={{ backgroundColor: colors[index % colors.length] }} />
              <span className="truncate font-bold uppercase tracking-wide text-text-secondary">{source}</span>
              <span className="text-right font-mono font-bold text-text-primary">{Math.round((value / total) * 100)}%</span>
            </div>
          ))}
        </div>
      </div>
    </ReportChartFrame>
  )
}

function AccountExposureTreemap({ rows }: { rows: Array<Record<string, unknown>> }) {
  const accountExposure = rows.reduce<Record<string, number>>((acc, row) => {
    const receiver = String(row.to_account ?? row.receiver ?? 'unknown')
    const amount = Number(row.amount_paisa ?? 0)
    if (Number.isFinite(amount) && amount > 0) acc[receiver] = (acc[receiver] ?? 0) + amount
    return acc
  }, {})
  const entries = Object.entries(accountExposure).sort((a, b) => b[1] - a[1]).slice(0, 12)
  const total = entries.reduce((sum, [, amount]) => sum + amount, 0) || 1
  const max = Math.max(1, ...entries.map(([, amount]) => amount))

  return (
    <ReportChartFrame title="Account exposure treemap" badge={`${entries.length} accounts`}>
      {entries.length === 0 ? (
        <div className="rounded border border-dashed border-border-subtle p-2 text-[9px] text-text-muted">No account exposure rows were returned.</div>
      ) : (
        <div className="grid auto-rows-[58px] grid-cols-3 gap-1.5">
          {entries.map(([account, amount], index) => {
            const intensity = amount / max
            return (
              <div
                key={account}
                className={cn(
                  'overflow-hidden rounded-md border p-2',
                  index === 0 ? 'col-span-2 row-span-2' : '',
                )}
                style={{
                  background: `linear-gradient(135deg, rgba(0,87,156,${0.14 + intensity * 0.22}), rgba(218,37,28,${0.10 + intensity * 0.34}))`,
                  borderColor: `rgba(218,37,28,${0.18 + intensity * 0.34})`,
                }}
              >
                <div className="truncate font-mono text-[8px] font-black text-text-primary">{short(account, 12)}</div>
                <div className="mt-1 font-mono text-[10px] font-black text-[#DA251C]">{compactInr(amount)}</div>
                <div className="mt-0.5 text-[7px] font-bold uppercase tracking-wide text-text-muted">{Math.round((amount / total) * 100)}% route exposure</div>
              </div>
            )
          })}
        </div>
      )}
    </ReportChartFrame>
  )
}

function RunAnalysisReport({
  report,
  run,
  llmRuntime,
}: {
  report?: EventLabAnalysisReport
  run?: EventLabRunResponse
  llmRuntime: LLMRuntimeSummary
}) {
  if (!report) {
    return (
      <div className="rounded-lg border border-dashed border-[#00579C]/35 bg-[#00579C]/5 p-6 text-center">
        <BarChart3 className="mx-auto h-6 w-6 text-accent-primary/50" />
        <div className="mt-2 text-[10px] font-bold uppercase tracking-[0.14em] text-text-primary">
          No autonomous report yet
        </div>
        <p className="mx-auto mt-1 max-w-xl text-[9px] leading-relaxed text-text-secondary">
          Preview or launch a fraud event chain. The report is generated from the returned backend events,
          not a static frontend template.
        </p>
      </div>
    )
  }

  const amountSeries = Array.isArray(report.amount_series) ? report.amount_series : []
  const maxAmount = Math.max(1, ...amountSeries.map((item) => Number(item.amount_paisa ?? 0)))
  const channelRows = Object.entries(report.channel_mix ?? {})
  const flagRows = Object.entries(report.risk_flags ?? {})
  const typologyRows = Object.entries(report.typology_mix ?? {}).filter(([label]) => label !== 'unknown')
  const velocityRows = Array.isArray(report.velocity_series) ? report.velocity_series : []
  const strengthRows = Object.entries(report.evidence_strengths ?? {})
  const stageRows = Object.entries(report.stage_coverage ?? {}).filter(([, count]) => count > 0)
  const geoRows = Array.isArray(report.geo_path) ? report.geo_path : []
  const riskComponentRows = Object.entries(report.risk_score_components ?? {}).filter(([, value]) => Number(value) > 0)
  const timelineRows = Array.isArray(report.timeline_buckets) ? report.timeline_buckets : []
  const stageTimelineRows = Array.isArray(report.stage_timeline) ? report.stage_timeline : []
  const routeSegmentRows = Array.isArray(report.route_segments) ? report.route_segments : []
  const evidenceMatrixRows = Array.isArray(report.evidence_matrix) ? report.evidence_matrix : []
  const countermeasureMatrixRows = Array.isArray(report.countermeasure_matrix) ? report.countermeasure_matrix : []
  const channelAmountRows = Object.entries(report.channel_amount_mix ?? {})
  const accountRoleRows = Object.entries(report.account_role_mix ?? {})
  const channelTotal = channelRows.reduce((sum, [, count]) => sum + count, 0)
  const controlRows = [
    ['Run', short(run?.run_id, 12)],
    ['Audit', short(run?.audit_hash, 12)],
    ['Seed', String(report.controls?.seed ?? 'n/a')],
    ['Mode', String(report.controls?.mode ?? 'n/a')],
    ['Route', String(report.route_label || report.controls?.route_label || 'n/a')],
    ['Velocity', `${String(report.controls?.velocity_minutes ?? 'n/a')} min`],
    ['Mule depth', String(report.controls?.mule_depth ?? 'n/a')],
    ['Risk bias', String(report.controls?.risk_bias ?? 'n/a')],
  ]
  const runComplete = Boolean(run?.stages?.some((stage) => ['evaluation_complete', 'analyst_decision', 'ledger_anchored', 'verdict'].includes(stage.stage)))

  return (
    <div className="rounded-lg border border-[#00579C]/30 bg-white shadow-sm">
      <div className="flex flex-wrap items-start justify-between gap-3 border-b border-border-subtle p-3">
        <div className="min-w-0">
          <div className="flex items-center gap-2 text-[10px] font-bold uppercase tracking-[0.14em] text-accent-primary">
            <ClipboardCheck className="h-3.5 w-3.5" />
            Autonomous fraud evaluation report
          </div>
          <p className="mt-1 max-w-4xl text-[10px] leading-relaxed text-text-secondary">
            {sanitizePublicTraceText(report.forensic_summary)}
          </p>
        </div>
        <div className="flex flex-wrap gap-1">
          <span className={cn(
            'rounded-full border px-2 py-0.5 text-[8px] font-bold uppercase tracking-[0.1em]',
            report.risk_tier === 'critical' || report.risk_tier === 'high'
              ? 'border-[#DA251C]/25 bg-[#DA251C]/10 text-[#DA251C]'
              : 'border-[#00579C]/25 bg-[#00579C]/10 text-[#00579C]',
          )}>
            {report.risk_tier} risk
          </span>
          <span className="rounded-full border border-[#f5b400]/30 bg-[#f5b400]/10 px-2 py-0.5 text-[8px] font-bold uppercase tracking-[0.1em] text-[#7a5a00]">
            {runComplete ? 'post-verdict' : 'live evaluation'}
          </span>
        </div>
      </div>

      <div className="grid gap-3 p-3 xl:grid-cols-[minmax(0,1.15fr)_minmax(360px,0.85fr)]">
        <div className="space-y-3">
          <div className="grid gap-2 md:grid-cols-5">
            <Metric label="Verdict" value={report.risk_tier.toUpperCase()} tone={report.risk_score >= 0.72 ? 'red' : 'amber'} />
            <Metric label="Risk score" value={fmtScore(report.risk_score)} tone={report.risk_score >= 0.72 ? 'red' : 'amber'} />
            <Metric label="Exposure" value={compactInr(report.total_exposure_paisa)} tone="red" />
            <Metric label="Accounts" value={fmtCount(report.unique_account_count)} />
            <Metric label="Confidence" value={fmtScore(report.confidence)} tone="green" />
          </div>

          <ReportSignalDensityGrid report={report} run={run} />

          <div className="grid gap-3 lg:grid-cols-[minmax(0,1fr)_300px]">
            <AmountVelocityPlot amountSeries={amountSeries} velocitySeries={velocityRows} />
            <ReportChartFrame title="Run fingerprint" badge={runComplete ? 'finalized' : 'evaluating'}>
              <div className="grid gap-1.5">
                {controlRows.map(([label, value]) => (
                  <div key={label} className="flex items-center justify-between gap-2 rounded border border-border-subtle bg-white px-2 py-1.5 text-[8px]">
                    <span className="font-bold uppercase tracking-[0.1em] text-text-muted">{label}</span>
                    <span className="truncate font-mono font-bold text-text-primary">{value}</span>
                  </div>
                ))}
              </div>
            </ReportChartFrame>
          </div>

          <div className="grid gap-3 lg:grid-cols-2">
            <RiskScoreWaterfall rows={riskComponentRows} />
            <TimelineBucketChart rows={timelineRows} />
          </div>

          <div className="grid gap-3 lg:grid-cols-2">
            <StageLatencySlaPlot rows={stageTimelineRows} />
            <EvidenceSourceDonut rows={evidenceMatrixRows} />
          </div>

          <div className="grid gap-3 lg:grid-cols-2">
            <StageTimelineChart rows={stageTimelineRows} />
            <ChannelExposureStack countRows={channelRows} amountRows={channelAmountRows} />
          </div>

          <div className="grid gap-3 lg:grid-cols-2">
            <div className="rounded-md border border-border-subtle bg-bg-elevated/45 p-3">
              <div className="mb-2 flex items-center justify-between gap-2">
                <div className="flex items-center gap-2 text-[9px] font-bold uppercase tracking-[0.12em] text-text-primary">
                  <BarChart3 className="h-3.5 w-3.5 text-accent-primary" />
                  Amount chain
                </div>
                <span className="font-mono text-[8px] text-text-muted">{amountSeries.length} monetary legs</span>
              </div>
              <div className="space-y-1.5">
                {amountSeries.slice(0, 12).map((item, index) => {
                  const amount = Number(item.amount_paisa ?? 0)
                  const width = Math.max(6, (amount / maxAmount) * 100)
                  return (
                    <div key={`${String(item.event_id)}-${index}`} className="grid grid-cols-[42px_minmax(0,1fr)_76px] items-center gap-2">
                      <span className="font-mono text-[8px] text-text-muted">{String(item.channel ?? item.role ?? 'leg')}</span>
                      <div className="h-5 overflow-hidden rounded bg-white">
                        <div
                          className="h-full rounded bg-[linear-gradient(90deg,#00579C,#DA251C)]"
                          style={{ width: `${width}%` }}
                        />
                      </div>
                      <span className="text-right font-mono text-[8px] font-bold text-text-primary">{compactInr(amount)}</span>
                    </div>
                  )
                })}
              </div>
            </div>

            <ReportRouteMap geoRows={geoRows} routeLabel={report.route_label} />
          </div>

          <div className="grid gap-3 lg:grid-cols-2">
            <RouteSegmentLedger rows={routeSegmentRows} />
            <EvidenceMatrixHeatmap rows={evidenceMatrixRows} />
          </div>

          <AccountExposureTreemap rows={routeSegmentRows} />

          <div className="grid gap-3 lg:grid-cols-3">
            <ChannelDistribution rows={channelRows} total={channelTotal || report.event_count} />
            <RiskFlagDistribution rows={flagRows} />
            <AccountRoleMix rows={accountRoleRows} />
          </div>

          <div className="grid gap-3 lg:grid-cols-[minmax(0,1.2fr)_minmax(300px,0.8fr)]">
            <CountermeasureMatrix rows={countermeasureMatrixRows} />
            <div className="rounded-md border border-border-subtle bg-bg-elevated/45 p-3">
              <div className="mb-2 text-[9px] font-bold uppercase tracking-[0.12em] text-text-primary">Typology and countermeasure state</div>
              <div className="mb-2 flex flex-wrap gap-1">
                {typologyRows.slice(0, 4).map(([label, count]) => (
                  <span key={label} className="rounded border border-[#00579C]/20 bg-[#00579C]/10 px-1.5 py-0.5 text-[8px] font-bold text-[#00579C]">
                    {label.replaceAll('_', ' ')} x{count}
                  </span>
                ))}
              </div>
              <div className="grid grid-cols-3 gap-1">
                <Metric label="Pending" value={String(report.countermeasure_status?.pending ?? 0)} tone="amber" />
                <Metric label="Executed" value={String(report.countermeasure_status?.executed ?? 0)} tone="green" />
                <Metric label="Rejected" value={String(report.countermeasure_status?.rejected ?? 0)} />
              </div>
            </div>
          </div>
        </div>

        <div className="space-y-3">
          <div className="rounded-md border border-[#00579C]/25 bg-[#00579C]/5 p-3">
            <div className="mb-2 flex items-center gap-2 text-[9px] font-bold uppercase tracking-[0.12em] text-accent-primary">
              <BrainCircuit className="h-3.5 w-3.5" />
              Explainability strength
            </div>
            <div className="space-y-2">
              {strengthRows.map(([label, score]) => (
                <div key={label} className="grid grid-cols-[112px_minmax(0,1fr)_38px] items-center gap-2 text-[8px]">
                  <span className="truncate font-semibold uppercase tracking-wide text-text-secondary">{label.replaceAll('_', ' ')}</span>
                  <div className="h-2 rounded bg-white">
                    <div className="h-full rounded bg-[linear-gradient(90deg,#00579C,#2f79b5)]" style={{ width: `${Math.max(3, Math.min(100, score * 100))}%` }} />
                  </div>
                  <span className="text-right font-mono font-bold text-text-primary">{fmtScore(score)}</span>
                </div>
              ))}
            </div>
          </div>

          <div className="rounded-md border border-border-subtle bg-bg-elevated/45 p-3">
            <div className="mb-2 flex items-center gap-2 text-[9px] font-bold uppercase tracking-[0.12em] text-text-primary">
              <GitBranch className="h-3.5 w-3.5 text-accent-primary" />
              Backend stage coverage
            </div>
            <div className="grid gap-1.5 sm:grid-cols-2">
              {stageRows.length === 0 ? (
                <div className="sm:col-span-2 rounded border border-dashed border-border-subtle p-2 text-[9px] text-text-muted">
                  Stage coverage will fill as SSE events arrive.
                </div>
              ) : stageRows.map(([stage, count]) => (
                <div key={stage} className="rounded border border-border-subtle bg-white px-2 py-1.5">
                  <div className="flex items-center justify-between gap-2">
                    <span className="truncate text-[8px] font-bold uppercase tracking-wide text-text-primary">{stage.replaceAll('_', ' ')}</span>
                    <span className="font-mono text-[8px] font-bold text-accent-primary">{count}</span>
                  </div>
                </div>
              ))}
            </div>
          </div>

          <div className="rounded-md border border-[#DA251C]/20 bg-[#DA251C]/5 p-3">
            <div className="mb-2 text-[9px] font-bold uppercase tracking-[0.12em] text-[#DA251C]">
              Analyst-ready next actions
            </div>
            <div className="space-y-1.5">
              {(report.recommended_next_steps ?? []).slice(0, 4).map((step, index) => (
                <div key={`${step}-${index}`} className="rounded border border-white bg-white/70 px-2 py-1.5 text-[9px] leading-relaxed text-text-secondary">
                  <span className="mr-2 font-mono font-bold text-[#DA251C]">{String(index + 1).padStart(2, '0')}</span>
                  {sanitizePublicTraceText(step)}
                </div>
              ))}
            </div>
          </div>

          <div className="rounded-md border border-border-subtle bg-white p-3">
            <div className="text-[9px] font-bold uppercase tracking-[0.12em] text-text-primary">{llmRuntime.model} role in this run</div>
            <p className="mt-1 text-[9px] leading-relaxed text-text-secondary">
              {llmRuntime.model} explains the generated evidence and analyst narrative. The verdict, score, proposals,
              and run report above are derived from PayFlow event data, heuristics, ML/graph stages, and countermeasure state.
            </p>
          </div>
        </div>
      </div>
    </div>
  )
}

function AutonomousReportGate({
  run,
  report,
  backendReady,
  visualEvidence,
  llmRuntime,
  onOpen,
}: {
  run?: EventLabRunResponse
  report?: EventLabAnalysisReport
  backendReady: boolean
  visualEvidence: ReturnType<typeof getReportVisualEvidence>
  llmRuntime: LLMRuntimeSummary
  onOpen: () => void
}) {
  const stageNames = new Set(run?.stages?.map((stage) => stage.stage) ?? [])
  const completedCount = REQUIRED_EVALUATION_STAGES.filter(([stage]) => stageNames.has(stage)).length
  const completed = backendReady
  const progress = run ? Math.round((completedCount / REQUIRED_EVALUATION_STAGES.length) * 100) : 0
  const backendStatus = backendReady ? 'backend complete' : `${completedCount}/${REQUIRED_EVALUATION_STAGES.length} backend stages`
  const visualStatus = visualEvidence.ready
    ? visualEvidence.backendReportHasEvidence
      ? 'backend report evidence ready'
      : `${visualEvidence.fullStageCount} event lifecycle complete`
    : `${visualEvidence.verdictCount}/${Math.max(1, visualEvidence.hydratedCount || visualEvidence.generatedCount)} visual verdicts`

  return (
    <div
      className={cn(
        'rounded-lg border p-3 transition-colors',
        completed && report
          ? 'border-alert-low/30 bg-alert-low/10'
          : 'border-[#00579C]/25 bg-[#00579C]/5',
      )}
    >
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div className="min-w-0">
          <div className="flex items-center gap-2 text-[10px] font-bold uppercase tracking-[0.14em] text-accent-primary">
            <ClipboardCheck className="h-3.5 w-3.5" />
            Autonomous report gate
          </div>
          <p className="mt-1 max-w-3xl text-[10px] leading-relaxed text-text-secondary">
            {completed && report
              ? `${llmRuntime.model} explanation, heuristics, ML score, graph evidence, dispatch, and final evidence readiness are complete. The analyst-ready report is unlocked from the backend run payload.`
              : run
                ? `Report stays locked until backend completion and evidence readiness are confirmed for ${short(run.run_id, 10)}. ${backendStatus}; ${visualStatus}.`
                : 'Launch a fraud event chain to stream the backend evaluation. The report will not appear until backend completion and evidence readiness are confirmed.'}
          </p>
        </div>
        <div className="flex items-center gap-2">
          <div className="w-20 overflow-hidden rounded-full bg-white">
            <div
              className={cn('h-1.5 rounded-full transition-all duration-500', completed ? 'bg-alert-low' : 'bg-accent-primary')}
              style={{ width: `${completed ? 100 : Math.min(94, Math.max(progress, visualEvidence.ready ? 92 : 4))}%` }}
            />
          </div>
          <span className="font-mono text-[9px] font-bold text-text-secondary">{completed ? '100%' : `${progress}%`}</span>
          <button
            type="button"
            onClick={onOpen}
            disabled={!completed || !report}
            className="inline-flex h-8 items-center gap-2 rounded-md border border-accent-primary/35 bg-bg-surface px-3 text-[9px] font-bold uppercase tracking-[0.12em] text-accent-primary hover:bg-accent-muted disabled:cursor-not-allowed disabled:border-border-subtle disabled:text-text-muted disabled:opacity-60"
          >
            <FileText className="h-3.5 w-3.5" />
            Open report
          </button>
        </div>
      </div>

      <div className="mt-3 grid gap-1.5 md:grid-cols-4 xl:grid-cols-7">
        {REQUIRED_EVALUATION_STAGES.map(([stage, label]) => {
          const done = stageNames.has(stage)
          return (
            <div
              key={stage}
              className={cn(
                'rounded-md border px-2 py-1.5 text-[8px] font-bold uppercase tracking-[0.1em]',
                done
                  ? 'border-[#00579C]/30 bg-white text-[#00579C]'
                  : 'border-border-subtle bg-bg-elevated/45 text-text-muted',
              )}
            >
              <span className="flex items-center justify-between gap-2">
                {label}
                {done ? <CheckCircle2 className="h-3 w-3 shrink-0" /> : <Clock3 className="h-3 w-3 shrink-0 opacity-55" />}
              </span>
            </div>
          )
        })}
      </div>
      {run && (
        <div className="mt-2 grid gap-1.5 md:grid-cols-3">
          {[
            ['Backend finalizer', backendReady ? 'evaluation + evidence ready' : 'waiting'],
            ['Hydrated events', `${visualEvidence.hydratedCount}/${visualEvidence.generatedCount || run.event_ids.length}`],
            ['Visible verdicts', `${visualEvidence.verdictCount}/${Math.max(1, visualEvidence.hydratedCount || visualEvidence.generatedCount)}`],
          ].map(([label, value]) => (
            <div key={label} className="rounded-md border border-border-subtle bg-bg-elevated/45 px-2 py-1.5">
              <div className="text-[7px] font-bold uppercase tracking-[0.12em] text-text-muted">{label}</div>
              <div className="mt-0.5 truncate font-mono text-[9px] font-bold text-text-primary">{value}</div>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

function RunReportModal({
  report,
  run,
  llmRuntime,
  onClose,
}: {
  report: EventLabAnalysisReport
  run: EventLabRunResponse
  llmRuntime: LLMRuntimeSummary
  onClose: () => void
}) {
  return (
    <div className="fixed inset-0 z-[120] flex items-center justify-center bg-[#071427]/75 p-4 backdrop-blur-sm">
      <div className="flex max-h-[88vh] w-[min(1180px,calc(100vw-2rem))] flex-col overflow-hidden rounded-xl border border-[#00579C]/40 bg-bg-surface shadow-[0_24px_80px_rgba(7,20,39,0.45)]">
        <div className="flex flex-wrap items-center justify-between gap-3 border-b border-white/10 bg-[linear-gradient(135deg,#071427,#00579C)] px-4 py-3 text-white">
          <div className="min-w-0">
            <div className="flex items-center gap-2 text-[10px] font-bold uppercase tracking-[0.16em] text-white/75">
              <BadgeCheck className="h-4 w-4 text-alert-low" />
              Evaluation complete
            </div>
            <h3 className="mt-1 truncate text-base font-bold tracking-tight">
              Autonomous fraud report for {short(run.run_id, 12)}
            </h3>
          </div>
          <div className="flex items-center gap-2">
            <span className="rounded-full border border-white/15 bg-white/10 px-2 py-1 text-[8px] font-bold uppercase tracking-[0.12em] text-white/80">
              {report.risk_tier} risk
            </span>
            <button
              type="button"
              onClick={onClose}
              className="inline-flex h-8 items-center gap-2 rounded-md border border-white/15 bg-white/10 px-3 text-[9px] font-bold uppercase tracking-[0.12em] text-white hover:bg-white/15"
            >
              <XCircle className="h-3.5 w-3.5" />
              Close
            </button>
          </div>
        </div>
        <div className="overflow-y-auto p-3">
          <RunAnalysisReport report={report} run={run} llmRuntime={llmRuntime} />
        </div>
      </div>
    </div>
  )
}

function RunTimeline({
  run,
  explainability,
  llmRuntime,
  previewQwenExplanation,
}: {
  run?: EventLabRunResponse
  explainability?: EventLabExplainabilityResponse
  llmRuntime: LLMRuntimeSummary
  previewQwenExplanation?: string
}) {
  const groups = explainability?.stage_groups ?? []
  const evidence = explainability?.evidence_panels ?? []
  const runtime = explainability?.runtime
  const backendQwenExplanation = run?.qwen_explanation ?? previewQwenExplanation
  const qwenContextNote = backendQwenExplanation
    ? sanitizePublicTraceText(backendQwenExplanation)
    : `Backend context is pending for this Event Lab selection. ${llmRuntime.model} remains advisory and cannot approve countermeasures. Runtime status: ${llmRuntime.statusLabel}.`
  return (
    <Panel
      title="Backend Visibility And AI Explainability"
      icon={Activity}
      badge={`${runtime?.stage_count ?? run?.stages?.length ?? 0} stages`}
      className="min-h-[260px]"
    >
      <div className="grid gap-3 p-3 xl:grid-cols-[minmax(0,1.25fr)_minmax(340px,0.75fr)]">
        <div className="space-y-3">
          <div className="grid gap-2 md:grid-cols-4">
            <Metric label="Run" value={run ? short(run.run_id, 10) : 'n/a'} />
            <Metric label="Correlation" value={run ? short(run.correlation_id, 10) : 'n/a'} />
            <Metric label="Latest Stage" value={(runtime?.latest_stage ?? 'waiting').replaceAll('_', ' ')} />
            <Metric
              label="Known Latency"
              value={fmtOptionalMs(run?.latency_metrics?.known_stage_latency_ms)}
            />
          </div>

          <div className="rounded-lg border border-[#00579C]/35 bg-[linear-gradient(135deg,#071427_0%,#003f75_58%,#101827_100%)] p-3 text-white shadow-sm">
            <div className="flex flex-wrap items-start justify-between gap-3">
              <div>
                <div className="flex items-center gap-2 text-[9px] font-bold uppercase tracking-[0.14em] text-white/70">
                  <Activity className="h-3.5 w-3.5 text-[#DA251C]" />
                  Backend execution matrix
                </div>
                <p className="mt-2 max-w-3xl text-[10px] leading-relaxed text-white/78">
                  Event Lab stages stream from FastAPI SSE into PayFlow ingestion, ML feature scoring, graph analysis,
                  circuit breaker consensus, bounded {llmRuntime.model}, analyst gate, and audit ledger.
                </p>
              </div>
              <div className="flex flex-wrap gap-1">
                {['FastAPI SSE', 'Feature Engine', 'NetworkX', 'Circuit Breaker', llmRuntime.model, 'Audit Hash'].map((tech) => (
                  <span key={tech} className="rounded border border-white/15 bg-white/10 px-1.5 py-0.5 text-[7px] font-bold uppercase tracking-[0.1em] text-white/75">
                    {tech}
                  </span>
                ))}
              </div>
            </div>
            <div className="mt-3 grid gap-2 md:grid-cols-5">
              {(run?.stages ?? []).slice(-10).length === 0 ? (
                <div className="md:col-span-5 rounded-md border border-white/15 bg-white/10 p-3 text-[10px] text-white/70">
                  Waiting for run stages to arrive from the backend.
                </div>
              ) : (run?.stages ?? []).slice(-10).map((stage, index) => (
                <div key={`${stage.stage}-${stage.timestamp}-${index}`} className="rounded-md border border-white/15 bg-white/10 p-2">
                  <div className="flex items-center justify-between gap-2">
                    <span className="truncate text-[8px] font-bold uppercase tracking-[0.12em] text-white">{stage.stage.replaceAll('_', ' ')}</span>
                    <span className="h-2 w-2 rounded-full bg-[#DA251C] shadow-[0_0_12px_rgba(218,37,28,0.75)]" />
                  </div>
                  <div className="mt-2 flex items-center justify-between gap-2 font-mono text-[8px] text-white/60">
                    <span>{stage.event_ids?.length ?? 0} ids</span>
                    <span>{stage.duration_ms != null ? fmtOptionalMs(stage.duration_ms) : fmtOptionalTimestamp(stage.timestamp)}</span>
                  </div>
                </div>
              ))}
            </div>
          </div>

          {groups.length === 0 ? (
            <div className="flex h-48 items-center justify-center rounded-md border border-dashed border-[#00579C]/35 bg-[#00579C]/10 text-center text-[10px] font-semibold uppercase tracking-[0.12em] text-[#00579C]">
              Launch an intel-linked event run to see every backend stage, proposal, and audit decision
            </div>
          ) : (
            <div className="grid gap-2 lg:grid-cols-2">
              {groups.map((group, groupIndex) => (
                <div
                  key={group.group}
                  className={cn(
                    'rounded-md border p-3',
                    group.completed
                      ? 'border-accent-primary/20 bg-white'
                      : 'border-border-subtle bg-bg-elevated/45 opacity-75',
                  )}
                >
                  <div className="flex items-start justify-between gap-3">
                    <div className="min-w-0">
                      <div className="flex items-center gap-2">
                        <span className={cn(
                          'flex h-5 w-5 items-center justify-center rounded-full text-[9px] font-bold',
                          group.completed ? 'bg-accent-primary text-white' : 'bg-bg-overlay text-text-muted',
                        )}>
                          {groupIndex + 1}
                        </span>
                        <div className="truncate text-[10px] font-bold uppercase tracking-[0.12em] text-text-primary">
                          {group.label}
                        </div>
                      </div>
                      <p className="mt-1 text-[9px] leading-relaxed text-text-secondary">{group.description}</p>
                    </div>
                    <div className="shrink-0 text-right">
                      <div className="font-mono text-[10px] font-bold text-accent-primary">{group.stage_count}</div>
                      <div className="text-[8px] uppercase tracking-wide text-text-muted">stages</div>
                    </div>
                  </div>
                  <div className="mt-3 max-h-36 space-y-1.5 overflow-y-auto pr-1">
                    {group.stages.length === 0 ? (
                      <div className="rounded border border-dashed border-border-subtle px-2 py-2 text-[9px] text-text-muted">
                        Waiting for this stage group.
                      </div>
                    ) : group.stages.map((stage, index) => (
                      <div key={`${stage.stage}-${stage.timestamp}-${index}`} className="rounded border border-border-subtle bg-bg-elevated/55 px-2 py-1.5">
                        <div className="flex items-center justify-between gap-2">
                          <span className="truncate text-[9px] font-bold text-text-primary">{stage.label}</span>
                          <span className="font-mono text-[8px] text-text-muted">
                            {stage.duration_ms != null ? fmtOptionalMs(stage.duration_ms) : fmtOptionalTimestamp(stage.timestamp)}
                          </span>
                        </div>
                        <p className="mt-0.5 line-clamp-2 text-[8px] text-text-secondary">
                          {publicEvidenceText(stage.evidence_summary)}
                        </p>
                      </div>
                    ))}
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>

        <div className="space-y-3">
          <div className="rounded-md border border-accent-primary/20 bg-accent-muted p-3">
            <div className="mb-2 flex items-center gap-2 text-[10px] font-bold uppercase tracking-[0.12em] text-accent-primary">
              <BrainCircuit className="h-3.5 w-3.5" />
              {llmRuntime.model} context guardrail
            </div>
            <p className="text-[10px] leading-relaxed text-text-secondary">
              {qwenContextNote}
            </p>
          </div>

          <div className="grid gap-2 sm:grid-cols-2 xl:grid-cols-1">
            {evidence.map((panel) => (
              <div key={panel.key} className="rounded-md border border-border-subtle bg-white p-3">
                <div className="flex items-start justify-between gap-2">
                  <div>
                    <div className="text-[10px] font-bold uppercase tracking-[0.12em] text-text-primary">{panel.title}</div>
                    <div className="mt-0.5 text-[8px] font-semibold uppercase tracking-[0.1em] text-accent-primary">{panel.authority}</div>
                  </div>
                  <span className="rounded-full border border-border-subtle bg-bg-elevated px-2 py-0.5 text-[8px] font-bold uppercase tracking-[0.1em] text-text-secondary">
                    {panel.status}
                  </span>
                </div>
                <p className="mt-2 line-clamp-3 text-[9px] leading-relaxed text-text-secondary">
                  {publicEvidenceText(panel.summary)}
                </p>
                <div className="mt-2 grid grid-cols-2 gap-1">
                  {Object.entries(panel.metrics).slice(0, 4).map(([label, value]) => (
                    <div key={label} className="rounded bg-bg-elevated/70 px-2 py-1">
                      <div className="text-[7px] font-bold uppercase tracking-wide text-text-muted">{label.replaceAll('_', ' ')}</div>
                      <div className="mt-0.5 truncate font-mono text-[9px] font-bold text-text-primary">{String(value)}</div>
                    </div>
                  ))}
                </div>
                <div className="mt-2 flex flex-wrap gap-1">
                  {publicEvidenceItems(panel.items.filter(Boolean)).slice(0, 3).map((item, index) => (
                    <span key={`${panel.key}-${index}-${item}`} className="max-w-full truncate rounded bg-bg-elevated px-1.5 py-0.5 text-[8px] text-text-secondary">
                      {item}
                    </span>
                  ))}
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </Panel>
  )
}

function ProposalRow({
  proposal,
  onApprove,
  onReject,
  busy,
  canApprove,
  canReject,
  roleLabel,
}: {
  proposal: CountermeasureProposal
  onApprove: (id: string) => void
  onReject: (id: string) => void
  busy: boolean
  canApprove: boolean
  canReject: boolean
  roleLabel: string
}) {
  const [nowSec, setNowSec] = useState(0)
  const executable = proposal.execution_allowed && proposal.status === 'proposed'

  useEffect(() => {
    const updateNow = () => setNowSec(Date.now() / 1000)
    updateNow()
    const interval = window.setInterval(updateNow, 1000)
    return () => window.clearInterval(interval)
  }, [])

  return (
    <div className="rounded-md border border-border-subtle bg-bg-elevated/55 p-3">
      <div className="flex items-start justify-between gap-3">
        <div className="min-w-0">
          <div className="flex items-center gap-2">
            <span className={cn(
              'rounded-full px-2 py-0.5 text-[8px] font-bold uppercase tracking-[0.12em]',
              proposal.status === 'executed' ? 'bg-alert-low/10 text-alert-low' :
                proposal.status === 'rejected' ? 'bg-text-muted/10 text-text-muted' :
                  proposal.status === 'failed' ? 'bg-alert-critical/10 text-alert-critical' :
                    'bg-accent-muted text-accent-primary',
            )}>
              {proposal.status}
            </span>
            <span className="truncate text-[11px] font-bold text-text-primary">{proposal.title}</span>
          </div>
          <p className="mt-1 line-clamp-2 text-[9px] leading-relaxed text-text-secondary">
            {publicEvidenceText(proposal.reason, 'Countermeasure evidence unavailable')}
          </p>
          <div className="mt-2 grid grid-cols-2 gap-1 sm:grid-cols-4">
            <div className="rounded bg-bg-surface px-2 py-1">
              <div className="text-[7px] font-bold uppercase tracking-wide text-text-muted">TTL</div>
              <div className="font-mono text-[9px] font-bold text-text-primary">
                {fmtSeconds(nowSec > 0 ? proposal.expires_at - nowSec : undefined)}
              </div>
            </div>
            <div className="rounded bg-bg-surface px-2 py-1">
              <div className="text-[7px] font-bold uppercase tracking-wide text-text-muted">Triggers</div>
              <div className="font-mono text-[9px] font-bold text-text-primary">{proposal.trigger_event_ids.length}</div>
            </div>
            <div className="rounded bg-bg-surface px-2 py-1">
              <div className="text-[7px] font-bold uppercase tracking-wide text-text-muted">Rollback</div>
              <div className="font-mono text-[9px] font-bold text-text-primary">{proposal.rollback_available ? 'ready' : 'n/a'}</div>
            </div>
            <div className="rounded bg-bg-surface px-2 py-1">
              <div className="text-[7px] font-bold uppercase tracking-wide text-text-muted">Audit</div>
              <div className="truncate font-mono text-[9px] font-bold text-text-primary">{proposal.audit_hash ? short(proposal.audit_hash, 8) : 'pending'}</div>
            </div>
          </div>
          <div className="mt-2 flex flex-wrap gap-1">
            <span className="rounded bg-bg-surface px-1.5 py-0.5 font-mono text-[8px] text-text-muted">{proposal.action}</span>
            {proposal.targets.map((target) => (
              <span key={target} className="rounded bg-bg-surface px-1.5 py-0.5 font-mono text-[8px] text-text-muted">{short(target, 12)}</span>
            ))}
            <span className={cn('rounded px-1.5 py-0.5 text-[8px] font-semibold', proposal.execution_allowed ? 'bg-alert-low/10 text-alert-low' : 'bg-alert-medium/10 text-alert-medium')}>
              {proposal.execution_allowed ? 'execution allowed' : 'advisory only'}
            </span>
          </div>
          {proposal.execution_result && Object.keys(proposal.execution_result).length > 0 && (
            <div className="mt-2 rounded border border-alert-low/20 bg-alert-low/10 px-2 py-1.5 text-[8px] leading-relaxed text-alert-low">
              Result: {String(proposal.execution_result.status ?? 'recorded')} {proposal.execution_result.target ? `on ${short(String(proposal.execution_result.target), 12)}` : ''}
            </div>
          )}
        </div>
        <div className="flex shrink-0 gap-1">
          <button
            type="button"
            onClick={() => onApprove(proposal.proposal_id)}
            disabled={!executable || busy || !canApprove}
            title={
              !canApprove
                ? `${roleLabel} cannot approve executable countermeasures`
                : proposal.execution_allowed ? 'Approve countermeasure' : 'Advisory-only proposal cannot execute'
            }
            className="inline-flex h-8 items-center gap-1 rounded-md border border-alert-low/30 bg-alert-low/10 px-2 text-[9px] font-bold uppercase tracking-[0.1em] text-alert-low disabled:cursor-not-allowed disabled:opacity-40"
          >
            {busy ? <Loader2 className="h-3 w-3 animate-spin" /> : <CheckCircle2 className="h-3 w-3" />}
            Approve
          </button>
          <button
            type="button"
            onClick={() => onReject(proposal.proposal_id)}
            disabled={proposal.status !== 'proposed' || busy || !canReject}
            title={!canReject ? `${roleLabel} cannot reject countermeasure proposals` : 'Reject countermeasure'}
            className="inline-flex h-8 items-center gap-1 rounded-md border border-border-default bg-bg-surface px-2 text-[9px] font-bold uppercase tracking-[0.1em] text-text-secondary disabled:cursor-not-allowed disabled:opacity-40"
          >
            <XCircle className="h-3 w-3" />
            Reject
          </button>
        </div>
      </div>
    </div>
  )
}

function CountermeasureConsole({
  runId,
  explainability,
}: {
  runId: string | null
  explainability?: EventLabExplainabilityResponse
}) {
  const { data } = useCountermeasureProposals(runId)
  const approve = useApproveCountermeasure()
  const reject = useRejectCountermeasure()
  const access = useRoleAccess()
  const proposals = data?.proposals ?? []
  const executed = proposals.filter((p) => p.status === 'executed').length
  const pending = proposals.filter((p) => p.status === 'proposed').length
  const authority = explainability?.authority_matrix ?? []

  return (
    <Panel title="Analyst Countermeasure Console" icon={ClipboardCheck} badge={`${pending} pending`}>
      <div className="space-y-3 p-3">
        <div className="grid grid-cols-3 gap-2">
          <Metric label="Pending" value={String(pending)} tone={pending ? 'amber' : 'green'} />
          <Metric label="Executed" value={String(executed)} tone="green" />
          <Metric label="Rollback" value={explainability?.runtime.rollback_available || proposals.some((p) => p.rollback_available) ? 'ready' : 'n/a'} />
        </div>
        <div className="rounded-md border border-border-subtle bg-bg-elevated/45 px-3 py-2 text-[9px] leading-relaxed text-text-secondary">
          <span className="font-bold text-text-primary">{access.policy.label}</span>
          {' '}scope: {access.policy.escalationScope}
        </div>
        <div className="rounded-md border border-border-subtle bg-white p-2">
          <div className="mb-2 text-[9px] font-bold uppercase tracking-[0.12em] text-text-primary">
            Decision authority chain
          </div>
          <div className="space-y-1.5">
            {authority.length === 0 ? (
              <div className="text-[9px] text-text-muted">
                No run explainability has been returned for the current selection.
              </div>
            ) : authority.map((row) => (
              <div key={row.layer} className="flex items-start justify-between gap-2 rounded bg-bg-elevated/60 px-2 py-1.5">
                <div className="min-w-0">
                  <div className="text-[9px] font-bold text-text-primary">{row.layer}</div>
                  <div className="line-clamp-1 text-[8px] text-text-secondary">{row.role}</div>
                </div>
                <span className={cn(
                  'shrink-0 rounded-full px-1.5 py-0.5 text-[7px] font-bold uppercase tracking-wide',
                  row.can_execute ? 'bg-alert-low/10 text-alert-low' : 'bg-bg-surface text-text-muted',
                )}>
                  {row.authority}
                </span>
              </div>
            ))}
          </div>
        </div>
        <div className="max-h-[430px] space-y-2 overflow-y-auto pr-1">
          {proposals.length === 0 ? (
            <div className="rounded-md border border-dashed border-border-default p-6 text-center text-[10px] font-semibold uppercase tracking-[0.12em] text-text-muted">
              No countermeasure proposals are active for the selected run
            </div>
          ) : proposals.map((proposal) => (
            <ProposalRow
              key={proposal.proposal_id}
              proposal={proposal}
              busy={approve.isPending || reject.isPending}
              canApprove={access.can('countermeasure:decide')}
              canReject={access.can('countermeasure:reject')}
              roleLabel={access.policy.label}
              onApprove={(id) => void approve.mutateAsync(id)}
              onReject={(id) => void reject.mutateAsync(id)}
            />
          ))}
        </div>
      </div>
    </Panel>
  )
}

function terminalTimeLabel(timestamp: number): string {
  if (!Number.isFinite(timestamp) || timestamp <= 0) return '--:--:--'
  const d = new Date(timestamp * 1000)
  return `${d.getHours().toString().padStart(2, '0')}:${d.getMinutes().toString().padStart(2, '0')}:${d.getSeconds().toString().padStart(2, '0')}`
}

function terminalLineClass(tone: BackendTerminalTone): string {
  return {
    info: 'text-[#7dd3fc]',
    success: 'text-[#5eead4]',
    warn: 'text-[#fde68a]',
    danger: 'text-[#fca5a5]',
    model: 'text-[#fda4af]',
    muted: 'text-[#b7c7dd]',
  }[tone]
}

function sourceClass(tone: BackendTerminalTone): string {
  return {
    info: 'border-[#7dd3fc]/25 bg-[#7dd3fc]/10 text-[#7dd3fc]',
    success: 'border-[#5eead4]/25 bg-[#5eead4]/10 text-[#5eead4]',
    warn: 'border-[#fde68a]/25 bg-[#fde68a]/10 text-[#fde68a]',
    danger: 'border-[#fca5a5]/25 bg-[#fca5a5]/10 text-[#fca5a5]',
    model: 'border-[#fda4af]/25 bg-[#fda4af]/10 text-[#fda4af]',
    muted: 'border-white/10 bg-white/5 text-[#b7c7dd]',
  }[tone]
}

function sourceLabel(source: BackendTerminalSource): string {
  return source.replaceAll('_', ' ')
}

function terminalEntryMatches(
  entry: BackendTerminalEntry,
  runId: string | null,
  runEventIds: Set<string>,
  trackedEventId: string | null,
): boolean {
  if (runId && entry.runId === runId) return true
  if (trackedEventId && (entry.txnId === trackedEventId || entry.txnIds?.includes(trackedEventId))) return true
  if (entry.txnId && runEventIds.has(entry.txnId)) return true
  if (entry.txnIds?.some((id) => runEventIds.has(id))) return true
  if (entry.source === 'custom') return true
  if (entry.stage === 'run_launch_requested' || entry.stage === 'run_launch_failed') return true
  return !runId && !trackedEventId
}

function LiveBackendRunTerminal({
  runId,
  run,
  selected,
  previewEvents,
  llmRuntime,
  launchPending,
}: {
  runId: string | null
  run?: EventLabRunResponse
  selected?: EventLabTemplate
  previewEvents: EventLabGeneratedEvent[]
  llmRuntime: LLMRuntimeSummary
  launchPending: boolean
}) {
  const scrollerRef = useRef<HTMLDivElement>(null)
  const [terminalNow, setTerminalNow] = useState(0)
  const activityRun = useActivityStore((state) => runId ? state.eventLabRuns[runId] : undefined)
  const trackedEventId = useActivityStore((state) => state.trackedEventId)
  const terminalEntries = useActivityStore((state) => state.terminalEntries)
  const { data: proposalsData } = useCountermeasureProposals(runId)
  const proposalCount = useMemo(
    () => (proposalsData?.proposals ?? run?.countermeasure_proposals ?? []).length,
    [proposalsData?.proposals, run?.countermeasure_proposals],
  )
  const runEventIds = useMemo(
    () => new Set([...(run?.event_ids ?? []), ...(activityRun?.eventIds ?? [])]),
    [activityRun?.eventIds, run?.event_ids],
  )
  const lines = useMemo(
    () => terminalEntries
      .filter((entry) => terminalEntryMatches(entry, runId, runEventIds, trackedEventId))
      .sort((a, b) => a.seq - b.seq)
      .slice(-160),
    [runEventIds, runId, terminalEntries, trackedEventId],
  )
  const latestLine = lines[lines.length - 1]
  const latestSeq = latestLine?.seq ?? 0

  useEffect(() => {
    const node = scrollerRef.current
    if (!node) return
    node.scrollTop = node.scrollHeight
  }, [latestSeq, lines.length])

  useEffect(() => {
    const tick = () => setTerminalNow(Date.now() / 1000)
    tick()
    const interval = window.setInterval(tick, 1000)
    return () => window.clearInterval(interval)
  }, [])

  const visibleRunId = runId ?? activityRun?.runId ?? null
  const live = Boolean(visibleRunId || trackedEventId || launchPending || lines.length > 0)
  const stageCount = new Set(lines.map((line) => line.stage).filter(Boolean)).size
  const counterCount = lines.filter((line) => line.source === 'counter').length || proposalCount
  const qwenCount = lines.filter((line) => line.source === 'qwen').length
  const latestAge = latestLine && terminalNow > 0 ? Math.max(0, Math.round(terminalNow - latestLine.timestamp)) : null

  return (
    <div className="overflow-hidden rounded-lg border border-[#00579C]/35 bg-[#071427] shadow-[0_12px_30px_rgba(7,20,39,0.18)]">
      <div className="flex flex-wrap items-center justify-between gap-3 border-b border-white/10 bg-[#0b1e38] px-3 py-2">
        <div className="flex min-w-0 items-center gap-2">
          <div className="flex gap-1">
            <span className="h-2 w-2 rounded-full bg-[#DA251C]" />
            <span className="h-2 w-2 rounded-full bg-[#f5b400]" />
            <span className="h-2 w-2 rounded-full bg-[#0f9f6e]" />
          </div>
          <div className="min-w-0">
            <div className="truncate font-mono text-[10px] font-bold uppercase tracking-[0.12em] text-white">
              Live Backend / AI Run Terminal
            </div>
            <div className="truncate font-mono text-[8px] text-white/55">
              {visibleRunId ? `payflow://${short(visibleRunId, 18)}` : trackedEventId ? `event://${short(trackedEventId, 18)}` : 'waiting for live SSE activity'}
            </div>
          </div>
        </div>
        <div className="flex flex-wrap items-center gap-1.5 font-mono text-[8px]">
          <span className={cn(
            'rounded-full border px-2 py-0.5 font-bold uppercase tracking-[0.12em]',
            live ? 'border-[#5eead4]/25 bg-[#5eead4]/10 text-[#5eead4]' : 'border-white/10 bg-white/5 text-white/45',
          )}>
            {live ? 'streaming' : 'standby'}
          </span>
          <span className="rounded-full border border-white/10 bg-white/5 px-2 py-0.5 text-white/65">{lines.length} live rows</span>
          <span className="rounded-full border border-white/10 bg-white/5 px-2 py-0.5 text-white/65">{stageCount} stages</span>
          <span className="rounded-full border border-white/10 bg-white/5 px-2 py-0.5 text-white/65">{counterCount} counters</span>
          <span className="rounded-full border border-white/10 bg-white/5 px-2 py-0.5 text-white/65">{qwenCount} qwen</span>
          <span className="rounded-full border border-white/10 bg-white/5 px-2 py-0.5 text-white/65">
            {latestAge == null ? 'no events yet' : `last ${latestAge}s ago`}
          </span>
          <span className="rounded-full border border-[#fda4af]/25 bg-[#fda4af]/10 px-2 py-0.5 text-[#fda4af]">{llmRuntime.model}</span>
        </div>
      </div>

      <div ref={scrollerRef} className="max-h-[270px] min-h-[220px] overflow-y-auto px-3 py-2 font-mono text-[10px] leading-relaxed custom-scrollbar">
        {lines.length === 0 ? (
          <div className="flex min-h-[190px] items-center justify-center text-center">
            <div className="max-w-xl">
              <div className="mx-auto mb-3 h-2 w-2 animate-pulse rounded-full bg-[#5eead4]" />
              <div className="font-mono text-[11px] font-bold uppercase tracking-[0.16em] text-white/75">
                waiting for live backend events
              </div>
              <div className="mt-2 text-[10px] leading-relaxed text-white/45">
                {launchPending
                  ? `submitting ${selected?.title ?? 'selected template'} to live ingestion`
                  : previewEvents.length > 0
                    ? `${previewEvents.length} preview event(s) are ready; launch or inject to stream pipeline, ML, Qwen, and countermeasure rows here.`
                    : 'Create or inject a fraud event to start the live SSE-backed terminal feed.'}
              </div>
            </div>
          </div>
        ) : lines.map((line, index) => {
          const hot = line.seq === latestSeq || index >= lines.length - 3
          return (
            <div key={line.id} className="grid grid-cols-[58px_92px_minmax(0,1fr)] gap-2 border-b border-white/[0.035] py-1.5 last:border-b-0">
              <span className="text-white/35">{terminalTimeLabel(line.timestamp)}</span>
              <span className={cn('inline-flex w-fit items-center gap-1 rounded border px-1.5 py-0.5 text-[8px] font-bold uppercase tracking-[0.12em]', sourceClass(line.tone))}>
                {hot && <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-current" />}
                {sourceLabel(line.source)}
              </span>
              <span className="min-w-0">
                <span className={cn('break-words font-semibold', terminalLineClass(line.tone))}>{line.title}</span>
                {line.detail && (
                  <>
                    <span className="mx-2 text-white/25">|</span>
                    <span className="break-words text-white/50">{line.detail}</span>
                  </>
                )}
              </span>
            </div>
          )
        })}
      </div>
    </div>
  )
}

export function AdaptiveEventLab() {
  const access = useRoleAccess()
  const { data: templatesData, isLoading } = useEventLabTemplates()
  const { data: llmStatus, isLoading: llmStatusLoading, isError: llmStatusError } = useLLMStatus()
  const preview = usePreviewEventLabRun()
  const launch = useCreateEventLabRun()
  const activeRunFromSse = useActivityStore((s) => s.activeEventLabRunId)
  const lifecycleEvents = useActivityStore((s) => s.events)
  const setActiveRun = useActivityStore((s) => s.setActiveEventLabRunId)
  const setTrackedEventId = useActivityStore((s) => s.setTrackedEventId)
  const onEventLabActivity = useActivityStore((s) => s.onEventLabActivity)
  const appendTerminalEntry = useActivityStore((s) => s.appendTerminalEntry)
  const [selectedTemplateId, setSelectedTemplateId] = useState<string | null>(null)
  const [mode, setMode] = useState<EventLabMode>('chain')
  const [intensity, setIntensity] = useState<'demo' | 'scale'>('scale')
  const [seed, setSeed] = useState(() => Date.now() % 1_000_000)
  const [localRunId, setLocalRunId] = useState<string | null>(null)
  const [localRunResponse, setLocalRunResponse] = useState<EventLabRunResponse | undefined>(undefined)
  const [reportModalRunId, setReportModalRunId] = useState<string | null>(null)
  const autoOpenedReportRunRef = useRef<Set<string>>(new Set())
  const mirroredRunKeyRef = useRef('')
  const [controls, setControls] = useState<EventLabControls>({
    event_count: 21,
    min_amount_inr: 24_000,
    max_amount_inr: 160_000,
    primary_channel: 'UPI',
    secondary_channel: 'IMPS',
    origin_region: 'kolkata',
    destination_region: 'delhi',
    velocity_minutes: 18,
    mule_depth: 5,
    device_reuse: true,
    include_auth_signal: false,
    include_interbank_leg: false,
    customer_profile: 'student',
    risk_bias: 'balanced',
  })

  const templates = useMemo(() => templatesData?.templates ?? [], [templatesData?.templates])
  const selected = useMemo(() => {
    if (!templates.length) return undefined
    return templates.find((item) => item.template_id === selectedTemplateId) ?? templates[0]
  }, [selectedTemplateId, templates])
  const activeRunId = localRunId ?? activeRunFromSse
  const { data: run } = useEventLabRun(activeRunId)
  const { data: explainability } = useEventLabExplainability(activeRunId)
  const currentRun = selectCanonicalRun(run, explainability?.run, localRunResponse)
  const previewEvents = preview.data?.run_preview.events ?? currentRun?.events ?? []
  const policy = preview.data?.run_preview.countermeasure_policy ?? currentRun?.countermeasure_policy
  const trust = policy?.source_trust
  const llmRuntime = resolveLLMRuntime(llmStatus, {
    loading: llmStatusLoading,
    error: llmStatusError,
  })
  const backendReportReady = isBackendReportReady(currentRun)
  const visualReportEvidence = useMemo(
    () => getReportVisualEvidence(currentRun, lifecycleEvents),
    [currentRun, lifecycleEvents],
  )
  const reportReady = backendReportReady
  const completedReport = reportReady && isUsableAnalysisReport(currentRun?.analysis_report)
    ? currentRun.analysis_report
    : undefined
  const completedRunId = completedReport && currentRun?.run_id ? currentRun.run_id : null

  useEffect(() => {
    if (!completedRunId || autoOpenedReportRunRef.current.has(completedRunId)) return undefined
    autoOpenedReportRunRef.current.add(completedRunId)
    const timer = window.setTimeout(() => setReportModalRunId(completedRunId), 650)
    return () => window.clearTimeout(timer)
  }, [completedRunId])

  useEffect(() => {
    if (!currentRun?.run_id) return
    const reportGeneratedAt = isUsableAnalysisReport(currentRun.analysis_report)
      ? Number(currentRun.analysis_report.generated_at ?? 0)
      : 0
    const mirrorKey = [
      currentRun.run_id,
      currentRun.status,
      currentRun.stages?.length ?? 0,
      currentRun.event_ids?.length ?? 0,
      reportGeneratedAt,
    ].join(':')
    if (mirroredRunKeyRef.current === mirrorKey) return
    mirroredRunKeyRef.current = mirrorKey
    setLocalRunResponse(currentRun)
    setActiveRun(currentRun.run_id)
    onEventLabActivity({
      type: currentRun.status === 'evaluated' ? 'run_completed' : 'run_launched',
      run: currentRun,
    })
  }, [currentRun, onEventLabActivity, setActiveRun])

  const launchControls = useMemo<EventLabControls>(() => ({
    ...controls,
    event_count: Number(controls.event_count ?? 0) || undefined,
    min_amount_inr: Number(controls.min_amount_inr ?? 0) || undefined,
    max_amount_inr: Number(controls.max_amount_inr ?? 0) || undefined,
    velocity_minutes: Number(controls.velocity_minutes ?? 0) || undefined,
    mule_depth: Number(controls.mule_depth ?? 0) || undefined,
  }), [controls])

  function applyTemplateDefaults(template: EventLabTemplate) {
    setControls((current) => ({
      ...current,
      primary_channel: template.channels[0] ?? current.primary_channel ?? 'UPI',
      secondary_channel: template.channels[template.channels.length - 1] ?? current.secondary_channel ?? 'IMPS',
      include_auth_signal: template.template_id === 'kyc_apk_phishing' ? true : current.include_auth_signal,
      include_interbank_leg: ['merchant_qr_misuse', 'investment_scam_layering', 'round_trip_shell_loop'].includes(template.template_id)
        ? true
        : current.include_interbank_leg,
      customer_profile:
        template.template_id.includes('merchant') ? 'merchant' :
          template.template_id.includes('dormant') ? 'dormant' :
            template.template_id.includes('kyc') ? 'senior' :
              template.template_id.includes('round_trip') ? 'shell' :
                current.customer_profile ?? 'student',
    }))
  }

  async function handlePreview() {
    if (!selected) return
    await preview.mutateAsync({
      template_id: selected.template_id,
      playbook_id: selected.linked_playbooks?.[0]?.playbook_id ?? null,
      mode,
      intensity,
      seed,
      controls: launchControls,
    })
  }

  async function handleLaunch() {
    if (!selected || !access.can('simulation:write')) return
    setReportModalRunId(null)
    appendTerminalEntry({
      timestamp: Date.now() / 1000,
      source: 'event_lab',
      tone: 'warn',
      title: 'event lab launch request submitted',
      detail: [
        `template=${selected.title}`,
        `mode=${mode}`,
        `intensity=${intensity}`,
        `seed=${seed}`,
        `events=${launchControls.event_count ?? 'auto'}`,
        `rails=${launchControls.primary_channel}/${launchControls.secondary_channel}`,
        `route=${launchControls.origin_region}->${launchControls.destination_region}`,
        `profile=${launchControls.customer_profile}`,
        `analyst_required=true`,
      ].join(' | '),
      stage: 'run_launch_requested',
    })
    try {
      const response = await launch.mutateAsync({
        template_id: selected.template_id,
        playbook_id: selected.linked_playbooks?.[0]?.playbook_id ?? null,
        mode,
        intensity,
        seed,
        analyst_required: true,
        controls: launchControls,
      })
      onEventLabActivity({ type: 'run_launched', run: response })
      setLocalRunResponse(response)
      setLocalRunId(response.run_id)
      setActiveRun(response.run_id)
      const focusEvent = response.events.find((event) => event.type !== 'auth')?.event_id ?? response.event_ids[0]
      if (focusEvent) setTrackedEventId(focusEvent)
    } catch (err) {
      appendTerminalEntry({
        timestamp: Date.now() / 1000,
        source: 'event_lab',
        tone: 'danger',
        title: 'event lab launch failed',
        detail: err instanceof Error ? err.message : String(err),
        stage: 'run_launch_failed',
      })
      throw err
    }
  }

  return (
    <div className="space-y-4">
      <section className="rounded-lg border border-border-default bg-bg-surface p-4 shadow-sm">
        <div className="flex flex-wrap items-start justify-between gap-4">
          <div className="min-w-0">
            <div className="flex items-center gap-2 text-[10px] font-bold uppercase tracking-[0.14em] text-accent-primary">
              <Radar className="h-4 w-4" />
              Adaptive Event Lab
            </div>
            <h2 className="mt-2 text-lg font-bold tracking-tight text-text-primary">
              Generate attacks from active pre-fraud intelligence, then approve countermeasures
            </h2>
            <p className="mt-1 max-w-4xl text-[12px] leading-relaxed text-text-secondary">
              {access.policy.label} scope: {access.policy.escalationScope} Internal PayFlow graph, ML, rules,
              circuit breaker, and ledger remain authoritative before any approved action executes.
            </p>
          </div>
          <div className="grid grid-cols-3 gap-2">
            <Metric label="Templates" value={isLoading ? '...' : String(templates.length)} />
            <Metric label="Intel Trust" value={fmtPct(trust)} tone={trust == null ? undefined : trust >= 0.85 ? 'green' : 'amber'} />
            <Metric label="Authority" value="analyst" />
          </div>
        </div>
      </section>

      <div className="grid gap-4 xl:grid-cols-[360px_minmax(0,1fr)_430px]">
        <Panel title="Intel-Linked Templates" icon={Sparkles} badge={`${templates.length} live`}>
          <div className="max-h-[620px] space-y-2 overflow-y-auto p-3">
            {templates.map((template) => (
              <TemplateCard
                key={template.template_id}
                template={template}
                selected={selected?.template_id === template.template_id}
                onSelect={() => {
                  setSelectedTemplateId(template.template_id)
                  setMode(template.default_mode)
                  applyTemplateDefaults(template)
                }}
              />
            ))}
          </div>
        </Panel>

        <div className="space-y-4">
          <Panel title="Event Chain Creator" icon={Zap} badge={selected?.title ?? 'select template'}>
            <div className="space-y-3 p-3">
              <div className="grid gap-2 md:grid-cols-3">
                <div className="rounded-md border border-border-subtle bg-bg-elevated/55 p-2">
                  <label className="text-[8px] font-bold uppercase tracking-[0.12em] text-text-muted">Mode</label>
                  <div className="mt-2 grid grid-cols-3 gap-1">
                    {MODES.map((item) => (
                      <button
                        key={item}
                        type="button"
                        onClick={() => setMode(item)}
                        className={cn(
                          'rounded-md border px-2 py-1.5 text-[9px] font-bold uppercase tracking-wide',
                          mode === item ? 'border-accent-primary bg-accent-primary text-white' : 'border-border-subtle bg-bg-surface text-text-secondary',
                        )}
                      >
                        {item}
                      </button>
                    ))}
                  </div>
                </div>
                <div className="rounded-md border border-border-subtle bg-bg-elevated/55 p-2">
                  <label className="text-[8px] font-bold uppercase tracking-[0.12em] text-text-muted">Intensity</label>
                  <div className="mt-2 grid grid-cols-2 gap-1">
                    {INTENSITY_OPTIONS.map((item) => (
                      <button
                        key={item.value}
                        type="button"
                        onClick={() => setIntensity(item.value)}
                        className={cn(
                          'rounded-md border px-2 py-1.5 text-[9px] font-bold uppercase tracking-wide',
                          intensity === item.value ? 'border-accent-primary bg-accent-primary text-white' : 'border-border-subtle bg-bg-surface text-text-secondary',
                        )}
                      >
                        {item.label}
                      </button>
                    ))}
                  </div>
                </div>
                <div className="rounded-md border border-border-subtle bg-bg-elevated/55 p-2">
                  <label className="text-[8px] font-bold uppercase tracking-[0.12em] text-text-muted">Seed</label>
                  <input
                    value={seed}
                    onChange={(event) => setSeed(Number(event.target.value) || 0)}
                    className="mt-2 h-8 w-full rounded-md border border-border-subtle bg-bg-surface px-2 font-mono text-[11px] text-text-primary outline-none focus:border-accent-primary"
                    inputMode="numeric"
                  />
                </div>
              </div>

              <ScenarioControlPanel controls={controls} onChange={setControls} />

              <div className="grid gap-2 md:grid-cols-4">
                <Metric label="Policy" value={policy?.authority?.replaceAll('_', ' ') ?? 'approval'} />
                <Metric label="Execution" value={policy?.execution_allowed ? 'allowed' : 'advisory'} tone={policy?.execution_allowed ? 'green' : 'amber'} />
                <Metric label="LLM" value={policy ? (policy.qwen_role ? `bounded ${llmRuntime.statusLabel}` : 'narrative') : 'n/a'} />
                <Metric label="Events" value={String(previewEvents.length)} />
              </div>

              <div className="flex flex-wrap items-center gap-2">
                <button
                  type="button"
                  onClick={() => void handlePreview()}
                  disabled={!selected || preview.isPending}
                  className="inline-flex h-9 items-center gap-2 rounded-md border border-accent-primary/35 bg-bg-surface px-3 text-[10px] font-bold uppercase tracking-[0.12em] text-accent-primary hover:bg-accent-muted disabled:cursor-not-allowed disabled:opacity-50"
                >
                  {preview.isPending ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <FileText className="h-3.5 w-3.5" />}
                  Preview Chain
                </button>
                <button
                  type="button"
                  onClick={() => void handleLaunch()}
                  disabled={!selected || launch.isPending || !access.can('simulation:write')}
                  title={!access.can('simulation:write') ? `${access.policy.label} cannot launch Event Lab runs` : 'Launch Event Lab run'}
                  className="inline-flex h-9 items-center gap-2 rounded-md bg-accent-primary px-4 text-[10px] font-bold uppercase tracking-[0.12em] text-white shadow-sm hover:bg-accent-hover disabled:cursor-not-allowed disabled:opacity-50"
                >
                  {launch.isPending ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Play className="h-3.5 w-3.5" />}
                  Launch Into Pipeline
                </button>
                <div className="ml-auto flex items-center gap-1 text-[9px] font-semibold uppercase tracking-[0.1em] text-text-muted">
                  <ShieldCheck className="h-3.5 w-3.5 text-alert-low" />
                  analyst approval required
                </div>
              </div>

              <LiveBackendRunTerminal
                runId={activeRunId}
                run={currentRun}
                selected={selected}
                previewEvents={previewEvents}
                llmRuntime={llmRuntime}
                launchPending={launch.isPending}
              />

              <AutonomousReportGate
                run={currentRun}
                report={completedReport}
                backendReady={backendReportReady}
                visualEvidence={visualReportEvidence}
                llmRuntime={llmRuntime}
                onOpen={() => {
                  if (currentRun?.run_id && completedReport) setReportModalRunId(currentRun.run_id)
                }}
              />

              <EventPreview events={previewEvents} />
            </div>
          </Panel>
        </div>

        <CountermeasureConsole runId={activeRunId} explainability={explainability} />
      </div>

      <RunTimeline
        run={currentRun}
        explainability={explainability}
        llmRuntime={llmRuntime}
        previewQwenExplanation={preview.data?.run_preview.qwen_explanation}
      />

      <Panel title="Countering Logic Transparency" icon={ShieldAlert}>
        <div className="grid gap-3 p-3 md:grid-cols-4">
          {[
            { icon: Radar, title: 'Intel primes', body: `Active playbooks select scenario seeds, watch terms, and ${llmRuntime.model} context.` },
            { icon: Activity, title: 'Pipeline decides', body: 'Events pass through ingestion, rules, ML, graph, and ledger as normal.' },
            { icon: PauseCircle, title: 'Analyst gates', body: 'Adaptive holds, freezes, routing pauses, device bans, and evidence actions wait for approval.' },
            { icon: RotateCcw, title: 'Audit remains', body: 'Approved or rejected decisions keep TTL, rollback, and audit-hash visibility.' },
          ].map(({ icon: Icon, title, body }) => (
            <div key={title} className="rounded-md border border-border-subtle bg-bg-elevated/50 p-3">
              <div className="flex items-center gap-2 text-[10px] font-bold uppercase tracking-[0.12em] text-text-primary">
                <Icon className="h-4 w-4 text-accent-primary" />
                {title}
              </div>
              <p className="mt-2 text-[10px] leading-relaxed text-text-secondary">{body}</p>
            </div>
          ))}
        </div>
      </Panel>

      {completedReport && currentRun && reportModalRunId === currentRun.run_id && (
        <RunReportModal
          report={completedReport}
          run={currentRun}
          llmRuntime={llmRuntime}
          onClose={() => setReportModalRunId(null)}
        />
      )}

      {(preview.isPending || launch.isPending) && (
        <div className="pointer-events-none fixed bottom-5 left-1/2 z-50 -translate-x-1/2 rounded-md border border-accent-primary/30 bg-bg-surface px-4 py-2 text-[10px] font-bold uppercase tracking-[0.12em] text-accent-primary shadow-lg">
          <span className="inline-flex items-center gap-2">
            <Clock3 className="h-3.5 w-3.5" />
            Event lab processing
          </span>
        </div>
      )}

      {launch.isSuccess && (
        <div className="rounded-md border border-alert-low/25 bg-alert-low/10 p-2 text-[10px] text-alert-low">
          <span className="inline-flex items-center gap-2">
            <BadgeCheck className="h-3.5 w-3.5" />
            Run {short(launch.data.run_id, 10)} injected into the live PayFlow pipeline.
          </span>
        </div>
      )}
    </div>
  )
}

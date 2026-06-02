import { useMemo } from 'react'
import { useActivityStore, type EventLabRunActivity, type EventLifecycle, type PipelineStage, type StageDetail } from '@/stores/use-activity-store'
import { cn, fmtOptionalTimestamp } from '@/lib/utils'
import { useLLMStatus } from '@/hooks/use-api'
import { resolveLLMRuntime, type LLMRuntimeSummary } from '@/lib/llm-runtime'
import {
  Activity,
  AlertTriangle,
  Bot,
  BrainCircuit,
  CheckCircle2,
  Clock3,
  Cpu,
  Database,
  Eye,
  FileText,
  GitBranch,
  MessageSquare,
  Network,
  Radio,
  Scale,
  Shield,
  ShieldCheck,
  TrendingUp,
  type LucideIcon,
} from 'lucide-react'

interface StageConfig {
  key: PipelineStage
  label: string
  owner: string
  icon: LucideIcon
  accent: string
}

const STAGES: StageConfig[] = [
  { key: 'ingested', label: 'Ingestion', owner: 'Schema + event normalizer', icon: Database, accent: '#00579C' },
  { key: 'ml_scored', label: 'ML scoring', owner: 'Feature engine + risk model', icon: BrainCircuit, accent: '#00579C' },
  { key: 'graph_investigated', label: 'Graph scan', owner: 'Fund-flow structure', icon: Network, accent: '#2f79b5' },
  { key: 'cb_evaluated', label: 'Control gate', owner: 'Circuit breaker evidence', icon: ShieldCheck, accent: '#f5b400' },
  { key: 'llm_started', label: 'AI explanation', owner: 'Bounded forensic narrative', icon: Bot, accent: '#DA251C' },
  { key: 'verdict', label: 'Verdict', owner: 'Final evidence state', icon: Scale, accent: '#B51A13' },
]

const STAGE_KEYS = new Set<PipelineStage>(STAGES.map((stage) => stage.key))

function short(value?: string | null, left = 12): string {
  if (!value) return 'n/a'
  return value.length > left + 4 ? `${value.slice(0, left)}...` : value
}

function fmtDuration(ms?: number | null): string {
  if (typeof ms !== 'number' || !Number.isFinite(ms) || ms < 0) return 'n/a'
  return ms < 1000 ? `${Math.round(ms)}ms` : `${(ms / 1000).toFixed(1)}s`
}

function fmtScore(score?: number | null): string {
  if (typeof score !== 'number' || !Number.isFinite(score)) return 'pending'
  return `${Math.round(Math.max(0, Math.min(1, score)) * 100)}%`
}

function fmtInr(paisa?: number | null): string {
  const rupees = Number(paisa ?? 0) / 100
  if (!Number.isFinite(rupees) || rupees <= 0) return 'pending'
  return `INR ${rupees.toLocaleString('en-IN', { maximumFractionDigits: 0 })}`
}

function stageElapsedMs(lifecycle: EventLifecycle | undefined, stageKey: PipelineStage): number | undefined {
  if (!lifecycle) return undefined
  const detail = lifecycle.stages.find((stage) => stage.stage === stageKey)
  if (!detail) return undefined
  if (typeof detail.durationMs === 'number' && Number.isFinite(detail.durationMs) && detail.durationMs >= 0) {
    return detail.durationMs
  }
  const index = lifecycle.stages.findIndex((stage) => stage.stage === stageKey)
  if (index <= 0) return undefined
  const current = Number(lifecycle.stages[index].timestamp)
  const previous = Number(lifecycle.stages[index - 1].timestamp)
  if (!Number.isFinite(current) || !Number.isFinite(previous) || current < previous) return undefined
  return (current - previous) * 1000
}

function reachedStages(lifecycle: EventLifecycle | undefined): Set<PipelineStage> {
  const reached = new Set<PipelineStage>()
  lifecycle?.stages.forEach((stage) => {
    if (STAGE_KEYS.has(stage.stage)) reached.add(stage.stage)
  })
  return reached
}

function activeStageKey(lifecycle: EventLifecycle | undefined, reached: Set<PipelineStage>): PipelineStage | null {
  if (!lifecycle) return null
  let lastIndex = -1
  STAGES.forEach((stage, index) => {
    if (reached.has(stage.key)) lastIndex = index
  })
  return lastIndex < STAGES.length - 1 ? STAGES[lastIndex + 1].key : null
}

function compactMetaValue(value: unknown): string {
  if (value == null || value === '') return ''
  if (Array.isArray(value)) return value.map(compactMetaValue).filter(Boolean).slice(0, 4).join(', ')
  if (typeof value === 'object') {
    const record = value as Record<string, unknown>
    const useful = ['risk_score', 'risk_tier', 'tier', 'route', 'count', 'event_count', 'proposal_count', 'audit_hash']
      .filter((key) => record[key] != null)
      .map((key) => `${key}=${compactMetaValue(record[key])}`)
    return useful.length > 0 ? useful.join(', ') : JSON.stringify(record).slice(0, 90)
  }
  return String(value)
}

function metaChips(meta?: Record<string, unknown>, max = 8): Array<[string, string]> {
  if (!meta) return []
  return Object.entries(meta)
    .map(([key, value]): [string, string] => [key, compactMetaValue(value)])
    .filter(([, value]) => Boolean(value))
    .slice(0, max)
}

function isFallbackLifecycle(lifecycle: EventLifecycle): boolean {
  return (
    lifecycle.confidenceSource === 'deterministic_evidence_fallback' ||
    Boolean(lifecycle.llmParseStatus?.includes('fallback')) ||
    (lifecycle.confidence === 0.5 &&
      (lifecycle.evidenceCited?.length ?? 0) === 0 &&
      Boolean(lifecycle.reasoningSummary?.includes('Unable to reach definitive conclusion')))
  )
}

function eventLabStageLabel(stage: string): string {
  return stage.replace(/_/g, ' ')
}

function StatusPill({ status }: { status: 'complete' | 'active' | 'waiting' }) {
  return (
    <span
      className={cn(
        'inline-flex items-center gap-1 rounded-full border px-2 py-0.5 text-[8px] font-bold uppercase tracking-[0.1em]',
        status === 'complete' && 'border-[#00579C]/25 bg-[#00579C]/10 text-[#00579C]',
        status === 'active' && 'border-[#DA251C]/25 bg-[#DA251C]/10 text-[#DA251C]',
        status === 'waiting' && 'border-border-subtle bg-bg-elevated text-text-muted',
      )}
    >
      {status === 'complete' ? <CheckCircle2 className="h-3 w-3" /> : status === 'active' ? <Activity className="h-3 w-3 animate-pulse" /> : <Clock3 className="h-3 w-3" />}
      {status}
    </span>
  )
}

function KV({ label, value, mono = false, tone = 'default' }: { label: string; value: string; mono?: boolean; tone?: 'default' | 'red' | 'green' | 'blue' }) {
  const toneClass = {
    default: 'text-text-primary',
    red: 'text-[#DA251C]',
    green: 'text-alert-low',
    blue: 'text-[#00579C]',
  }[tone]
  return (
    <div className="rounded-md border border-border-subtle bg-white px-2 py-1.5">
      <div className="text-[7px] font-bold uppercase tracking-[0.12em] text-text-muted">{label}</div>
      <div className={cn('mt-0.5 truncate text-[9px] font-bold', mono && 'font-mono', toneClass)}>{value}</div>
    </div>
  )
}

function ProgressBar({ value, tone = 'blue' }: { value: number; tone?: 'blue' | 'red' | 'amber' | 'green' }) {
  const clamped = Math.max(0, Math.min(100, value))
  const fill = {
    blue: 'bg-[#00579C]',
    red: 'bg-[#DA251C]',
    amber: 'bg-[#f5b400]',
    green: 'bg-alert-low',
  }[tone]
  return (
    <div className="h-2 overflow-hidden rounded-full bg-bg-elevated">
      <div className={cn('h-full rounded-full transition-all duration-700', fill)} style={{ width: `${clamped}%` }} />
    </div>
  )
}

function RunStageLedger({ run, llmRuntime }: { run: EventLabRunActivity; llmRuntime: LLMRuntimeSummary }) {
  const latest = run.stages.at(-1)
  const finalReached = run.stages.some((stage) => stage.stage === 'evaluation_complete' || stage.stage === 'evidence_ready')

  return (
    <section className="rounded-lg border border-[#00579C]/25 bg-white shadow-sm">
      <div className="flex flex-wrap items-start justify-between gap-3 border-b border-border-subtle p-3">
        <div className="min-w-0">
          <div className="flex items-center gap-2 text-[10px] font-bold uppercase tracking-[0.14em] text-[#00579C]">
            <Radio className="h-4 w-4 text-[#DA251C]" />
            Event Lab backend stage ledger
          </div>
          <p className="mt-1 max-w-4xl text-[10px] leading-relaxed text-text-secondary">
            {run.templateTitle} is tracked from live Event Lab run events. The report unlocks only after the run reports
            evaluation completion and evidence readiness.
          </p>
        </div>
        <div className="grid min-w-[300px] grid-cols-3 gap-2">
          <KV label="Run" value={short(run.runId, 10)} mono />
          <KV label="Latest" value={latest ? eventLabStageLabel(latest.stage) : 'waiting'} />
          <KV label="Report" value={finalReached ? 'ready' : 'locked'} tone={finalReached ? 'green' : 'red'} />
        </div>
      </div>

      <div className="grid gap-2 p-3 md:grid-cols-4 xl:grid-cols-6">
        {run.stages.length === 0 ? (
          <div className="md:col-span-4 xl:col-span-6 rounded-md border border-dashed border-[#00579C]/25 bg-[#00579C]/5 p-3 text-[10px] text-text-secondary">
            Waiting for the first backend stage from SSE.
          </div>
        ) : run.stages.slice(-18).map((stage, index) => {
          const isFinal = stage.stage === 'evaluation_complete' || stage.stage === 'evidence_ready'
          return (
            <div key={`${stage.stage}-${stage.timestamp}-${index}`} className={cn(
              'rounded-md border p-2',
              isFinal ? 'border-alert-low/25 bg-alert-low/10' : 'border-border-subtle bg-bg-elevated/50',
            )}>
              <div className="flex items-center justify-between gap-2">
                <span className="truncate text-[8px] font-bold uppercase tracking-[0.12em] text-text-primary">{eventLabStageLabel(stage.stage)}</span>
                {isFinal ? <CheckCircle2 className="h-3.5 w-3.5 shrink-0 text-alert-low" /> : <Activity className="h-3.5 w-3.5 shrink-0 text-[#00579C]" />}
              </div>
              <div className="mt-2 flex items-center justify-between gap-2 font-mono text-[8px] text-text-muted">
                <span>{stage.event_ids?.length ?? 0} ids</span>
                <span>{stage.duration_ms != null ? fmtDuration(stage.duration_ms) : fmtOptionalTimestamp(stage.timestamp)}</span>
              </div>
            </div>
          )
        })}
      </div>

      <div className="grid gap-2 border-t border-border-subtle bg-[#f4f8fc] p-3 md:grid-cols-[minmax(0,1fr)_360px]">
        <div className="grid gap-2 md:grid-cols-4">
          <KV label="Correlation" value={short(run.correlationId, 14)} mono />
          <KV label="Generated events" value={String(run.eventIds.length)} />
          <KV label="Status" value={run.status.replace(/_/g, ' ')} />
          <KV label="Audit hash" value={short(run.auditHash, 14)} mono />
        </div>
        <div className="rounded-md border border-[#00579C]/20 bg-white p-2">
          <div className="flex items-center gap-2 text-[8px] font-bold uppercase tracking-[0.12em] text-[#00579C]">
            <Bot className="h-3.5 w-3.5 text-[#DA251C]" />
            {llmRuntime.model} bounded explanation
          </div>
          <p className="mt-1 line-clamp-3 text-[9px] leading-relaxed text-text-secondary">
            {run.qwenExplanation || `${llmRuntime.model} explanation metadata will appear when the backend reports the AI context stage.`}
          </p>
        </div>
      </div>
    </section>
  )
}

function EvidenceSyncPanel({
  run,
  lifecycle,
  hydratedCount,
  verdictCount,
  stagesReached,
  totalStages,
}: {
  run?: EventLabRunActivity
  lifecycle?: EventLifecycle
  hydratedCount: number
  verdictCount: number
  stagesReached: number
  totalStages: number
}) {
  const progress = totalStages > 0 ? Math.round((stagesReached / totalStages) * 100) : 0
  const totalEvents = run?.eventIds.length ?? (lifecycle ? 1 : 0)
  const verdictPct = totalEvents > 0 ? Math.round((verdictCount / totalEvents) * 100) : 0

  return (
    <section className="rounded-lg border border-[#00579C]/25 bg-white shadow-sm">
      <div className="grid gap-3 p-3 xl:grid-cols-[minmax(0,1fr)_360px]">
        <div className="min-w-0">
          <div className="flex flex-wrap items-center justify-between gap-2">
            <div className="flex items-center gap-2 text-[10px] font-bold uppercase tracking-[0.14em] text-[#00579C]">
              <Eye className="h-4 w-4" />
              Live evidence sync
            </div>
            <div className="flex flex-wrap gap-1.5">
              <StatusPill status={progress >= 100 ? 'complete' : lifecycle ? 'active' : 'waiting'} />
              <span className="rounded-full border border-border-subtle bg-bg-elevated px-2 py-0.5 font-mono text-[8px] font-bold text-text-secondary">
                hydrated {hydratedCount}/{Math.max(totalEvents, hydratedCount)}
              </span>
              <span className="rounded-full border border-[#DA251C]/20 bg-[#DA251C]/10 px-2 py-0.5 font-mono text-[8px] font-bold text-[#DA251C]">
                verdicts {verdictCount}
              </span>
            </div>
          </div>
          <div className="mt-3 grid gap-2 md:grid-cols-4">
            <KV label="Selected event" value={short(lifecycle?.txnId, 16)} mono />
            <KV label="Risk" value={lifecycle?.riskScore != null ? `${fmtScore(lifecycle.riskScore)} ${lifecycle.riskTier ?? ''}` : 'pending'} tone={lifecycle?.riskScore && lifecycle.riskScore > 0.7 ? 'red' : 'blue'} />
            <KV label="Verdict" value={lifecycle?.verdict ?? 'pending'} />
            <KV label="Amount" value={fmtInr(lifecycle?.amountPaisa)} />
          </div>
          <div className="mt-3 grid gap-2 md:grid-cols-2">
            <div>
              <div className="mb-1 flex items-center justify-between text-[8px] font-bold uppercase tracking-[0.12em] text-text-muted">
                <span>Operator stages</span>
                <span>{stagesReached}/{totalStages}</span>
              </div>
              <ProgressBar value={progress} tone={progress >= 100 ? 'green' : 'blue'} />
            </div>
            <div>
              <div className="mb-1 flex items-center justify-between text-[8px] font-bold uppercase tracking-[0.12em] text-text-muted">
                <span>Run verdict hydration</span>
                <span>{verdictPct}%</span>
              </div>
              <ProgressBar value={verdictPct} tone={verdictPct >= 100 ? 'green' : 'red'} />
            </div>
          </div>
        </div>

        <div className="rounded-md border border-border-subtle bg-[#f4f8fc] p-3">
          <div className="text-[8px] font-bold uppercase tracking-[0.12em] text-text-muted">Selected event route</div>
          <div className="mt-2 grid grid-cols-[minmax(0,1fr)_24px_minmax(0,1fr)] items-center gap-2">
            <div className="truncate rounded-md border border-border-subtle bg-white px-2 py-2 font-mono text-[8px] font-bold text-text-primary">
              {lifecycle?.sender || 'sender pending'}
            </div>
            <span className="text-center text-[10px] font-bold text-[#00579C]">to</span>
            <div className="truncate rounded-md border border-border-subtle bg-white px-2 py-2 font-mono text-[8px] font-bold text-text-primary">
              {lifecycle?.receiver || 'receiver pending'}
            </div>
          </div>
          {lifecycle?.topFeatures && lifecycle.topFeatures.length > 0 && (
            <div className="mt-2 flex flex-wrap gap-1">
              {lifecycle.topFeatures.slice(0, 5).map((feature) => (
                <span key={feature} className="rounded border border-[#00579C]/20 bg-[#00579C]/10 px-1.5 py-0.5 text-[7px] font-bold uppercase tracking-wide text-[#00579C]">
                  {feature}
                </span>
              ))}
            </div>
          )}
        </div>
      </div>
    </section>
  )
}

function StageProgressStrip({
  reached,
  active,
}: {
  reached: Set<PipelineStage>
  active: PipelineStage | null
}) {
  return (
    <div className="grid gap-2 md:grid-cols-3 xl:grid-cols-6">
      {STAGES.map((stage, index) => {
        const Icon = stage.icon
        const status: 'complete' | 'active' | 'waiting' = reached.has(stage.key) ? 'complete' : active === stage.key ? 'active' : 'waiting'
        return (
          <div key={stage.key} className={cn(
            'rounded-lg border bg-white p-2 transition-colors',
            status === 'complete' && 'border-[#00579C]/35',
            status === 'active' && 'border-[#DA251C]/35 bg-[#DA251C]/5',
            status === 'waiting' && 'border-border-subtle bg-bg-elevated/40',
          )}>
            <div className="flex items-center justify-between gap-2">
              <div className={cn(
                'flex h-7 w-7 items-center justify-center rounded-md',
                status === 'waiting' ? 'bg-bg-elevated text-text-muted' : 'text-white',
              )} style={status === 'waiting' ? undefined : { background: stage.accent }}>
                <Icon className="h-3.5 w-3.5" />
              </div>
              <span className="font-mono text-[8px] font-bold text-text-muted">{String(index + 1).padStart(2, '0')}</span>
            </div>
            <div className="mt-2 truncate text-[9px] font-bold uppercase tracking-[0.1em] text-text-primary">{stage.label}</div>
            <div className="mt-0.5 truncate text-[8px] text-text-muted">{stage.owner}</div>
          </div>
        )
      })}
    </div>
  )
}

function StageCard({
  config,
  lifecycle,
  reached,
  active,
  llmRuntime,
}: {
  config: StageConfig
  lifecycle?: EventLifecycle
  reached: boolean
  active: boolean
  llmRuntime: LLMRuntimeSummary
}) {
  const Icon = config.icon
  const stageDetail = lifecycle?.stages.find((stage) => stage.stage === config.key)
  const status: 'complete' | 'active' | 'waiting' = reached ? 'complete' : active ? 'active' : 'waiting'
  const chips = metaChips(stageDetail?.meta)

  return (
    <section className={cn(
      'rounded-lg border bg-white shadow-sm',
      status === 'complete' && 'border-[#00579C]/30',
      status === 'active' && 'border-[#DA251C]/35 shadow-[0_10px_28px_rgba(218,37,28,0.12)]',
      status === 'waiting' && 'border-border-subtle opacity-75',
    )}>
      <div className="flex flex-wrap items-start justify-between gap-3 border-b border-border-subtle p-3">
        <div className="flex min-w-0 items-center gap-3">
          <div className={cn('flex h-9 w-9 items-center justify-center rounded-lg text-white', status === 'waiting' && 'bg-bg-elevated text-text-muted')} style={status === 'waiting' ? undefined : { background: config.accent }}>
            <Icon className="h-4 w-4" />
          </div>
          <div className="min-w-0">
            <div className="truncate text-[10px] font-bold uppercase tracking-[0.12em] text-text-primary">{config.label}</div>
            <div className="mt-0.5 flex flex-wrap items-center gap-2 text-[8px] text-text-muted">
              <Cpu className="h-3 w-3" />
              <span>{config.owner}</span>
              {stageDetail && <span className="font-mono">{fmtDuration(stageElapsedMs(lifecycle, config.key))}</span>}
            </div>
          </div>
        </div>
        <StatusPill status={status} />
      </div>

      <div className="space-y-3 p-3">
        <div className="grid gap-2 md:grid-cols-4">
          <KV label="Event" value={short(lifecycle?.txnId, 16)} mono />
          <KV label="Stage time" value={stageDetail ? fmtOptionalTimestamp(stageDetail.timestamp) : 'pending'} />
          <KV label="Duration" value={stageDetail ? fmtDuration(stageElapsedMs(lifecycle, config.key)) : 'pending'} />
          <KV label="Evidence" value={stageDetail ? `${chips.length} metadata fields` : 'waiting'} />
        </div>

        {renderStageEvidence(config.key, lifecycle, stageDetail, llmRuntime)}

        {chips.length > 0 && (
          <div className="flex flex-wrap gap-1">
            {chips.map(([key, value]) => (
              <span key={`${config.key}-${key}`} className="rounded border border-[#00579C]/15 bg-[#00579C]/5 px-1.5 py-0.5 font-mono text-[7px] text-text-secondary">
                {key.replace(/_/g, ' ')}={value.slice(0, 60)}
              </span>
            ))}
          </div>
        )}
      </div>
    </section>
  )
}

function renderStageEvidence(
  stage: PipelineStage,
  lifecycle: EventLifecycle | undefined,
  detail: StageDetail | undefined,
  llmRuntime: LLMRuntimeSummary,
) {
  if (!lifecycle) {
    return (
      <div className="rounded-md border border-dashed border-border-subtle bg-bg-elevated/50 p-3 text-[10px] text-text-secondary">
        Waiting for a selected event lifecycle to hydrate from SSE.
      </div>
    )
  }

  if (stage === 'ingested') {
    return (
      <div className="grid gap-2 md:grid-cols-3">
        <KV label="Sender" value={lifecycle.sender || 'pending'} mono />
        <KV label="Receiver" value={lifecycle.receiver || 'pending'} mono />
        <KV label="Amount" value={fmtInr(lifecycle.amountPaisa)} tone={lifecycle.fraudLabel > 0 ? 'red' : 'blue'} />
      </div>
    )
  }

  if (stage === 'ml_scored') {
    return (
      <div className="rounded-md border border-border-subtle bg-[#f4f8fc] p-3">
        <div className="flex items-center justify-between gap-2 text-[9px] font-bold uppercase tracking-[0.12em] text-text-primary">
          <span>Risk score</span>
          <span className={cn('font-mono', lifecycle.riskScore && lifecycle.riskScore > 0.7 ? 'text-[#DA251C]' : 'text-[#00579C]')}>
            {fmtScore(lifecycle.riskScore)}
          </span>
        </div>
        <div className="mt-2">
          <ProgressBar value={(lifecycle.riskScore ?? 0) * 100} tone={lifecycle.riskScore && lifecycle.riskScore > 0.7 ? 'red' : 'blue'} />
        </div>
        {lifecycle.topFeatures && lifecycle.topFeatures.length > 0 ? (
          <div className="mt-2 flex flex-wrap gap-1">
            {lifecycle.topFeatures.map((feature) => (
              <span key={feature} className="rounded border border-[#00579C]/20 bg-white px-1.5 py-0.5 text-[7px] font-bold uppercase tracking-wide text-[#00579C]">{feature}</span>
            ))}
          </div>
        ) : (
          <p className="mt-2 text-[9px] text-text-muted">Top feature names will appear when the risk scoring payload arrives.</p>
        )}
      </div>
    )
  }

  if (stage === 'graph_investigated') {
    const graphChips = metaChips(detail?.meta, 10)
    return (
      <div className="rounded-md border border-border-subtle bg-[#f4f8fc] p-3">
        <div className="mb-2 flex items-center gap-2 text-[9px] font-bold uppercase tracking-[0.12em] text-text-primary">
          <GitBranch className="h-3.5 w-3.5 text-[#00579C]" />
          Observed graph evidence
        </div>
        {graphChips.length > 0 ? (
          <div className="grid gap-1.5 sm:grid-cols-2 lg:grid-cols-3">
            {graphChips.map(([key, value]) => <KV key={key} label={key.replace(/_/g, ' ')} value={value} />)}
          </div>
        ) : (
          <p className="text-[9px] leading-relaxed text-text-muted">
            Graph metrics are not invented here. This section fills only when backend graph metadata is attached to the selected event stage.
          </p>
        )}
      </div>
    )
  }

  if (stage === 'cb_evaluated') {
    const scores = lifecycle.consensusScores
    const rows: Array<[string, number | null | undefined]> = [
      ['ML', scores?.ml],
      ['Graph', scores?.graph],
      ['GNN', scores?.gnn],
      ['Consensus', scores?.consensus],
    ]
    return (
      <div className="space-y-2 rounded-md border border-border-subtle bg-[#f4f8fc] p-3">
        {rows.map(([label, score]) => (
          <div key={label} className="grid grid-cols-[76px_minmax(0,1fr)_48px] items-center gap-2">
            <span className="text-[8px] font-bold uppercase tracking-[0.12em] text-text-muted">{label}</span>
            <ProgressBar value={(score ?? 0) * 100} tone={score && score > 0.7 ? 'red' : 'blue'} />
            <span className="text-right font-mono text-[8px] font-bold text-text-primary">{fmtScore(score)}</span>
          </div>
        ))}
      </div>
    )
  }

  if (stage === 'llm_started') {
    return (
      <div className="rounded-md border border-[#DA251C]/15 bg-[#DA251C]/5 p-3">
        <div className="grid gap-2 md:grid-cols-4">
          <KV label="Runtime" value={`${llmRuntime.model} (${llmRuntime.statusLabel})`} />
          <KV label="Steps" value={String(lifecycle.thinkingSteps ?? 'pending')} />
          <KV label="Tools" value={String(lifecycle.toolsUsed?.length ?? 0)} />
          <KV label="Duration" value={fmtDuration(lifecycle.totalDurationMs)} />
        </div>
        {lifecycle.reasoningSummary ? (
          <p className="mt-2 rounded-md border border-white bg-white/75 p-2 text-[9px] leading-relaxed text-text-secondary">
            {lifecycle.reasoningSummary}
          </p>
        ) : (
          <p className="mt-2 text-[9px] text-text-muted">Awaiting model-derived explanation payload for the selected event.</p>
        )}
      </div>
    )
  }

  return (
    <div className="rounded-md border border-border-subtle bg-[#f4f8fc] p-3">
      {lifecycle.verdict ? (
        <div className="space-y-2">
          <div className="grid gap-2 md:grid-cols-4">
            <KV label="Verdict" value={lifecycle.verdict} tone={lifecycle.verdict === 'fraudulent' ? 'red' : 'blue'} />
            <KV label="Confidence" value={isFallbackLifecycle(lifecycle) ? 'n/a' : fmtScore(lifecycle.confidence)} />
            <KV label="Typology" value={lifecycle.fraudTypology ?? 'pending'} />
            <KV label="Action" value={lifecycle.recommendedAction ?? 'pending'} />
          </div>
          {lifecycle.evidenceCited && lifecycle.evidenceCited.length > 0 && (
            <div className="grid gap-1.5 md:grid-cols-2">
              {lifecycle.evidenceCited.slice(0, 4).map((item, index) => (
                <div key={`${item}-${index}`} className="rounded-md border border-white bg-white px-2 py-1.5 text-[9px] text-text-secondary">
                  <span className="mr-1 font-mono font-bold text-[#00579C]">{String(index + 1).padStart(2, '0')}</span>
                  {item}
                </div>
              ))}
            </div>
          )}
        </div>
      ) : (
        <div className="flex items-center gap-2 text-[9px] text-text-muted">
          <AlertTriangle className="h-3.5 w-3.5 text-[#f5b400]" />
          Awaiting final classification verdict from the selected event lifecycle.
        </div>
      )}
    </div>
  )
}

interface PipelineTransparencyProps {
  className?: string
}

export function PipelineTransparency({ className }: PipelineTransparencyProps) {
  const events = useActivityStore((state) => state.events)
  const orderedIds = useActivityStore((state) => state.orderedIds)
  const trackedEventId = useActivityStore((state) => state.trackedEventId)
  const eventLabRuns = useActivityStore((state) => state.eventLabRuns)
  const activeEventLabRunId = useActivityStore((state) => state.activeEventLabRunId)
  const { data: llmStatus, isLoading: llmStatusLoading, isError: llmStatusError } = useLLMStatus()

  const activeRun = activeEventLabRunId ? eventLabRuns[activeEventLabRunId] : undefined
  const activeRunEventIds = activeRun?.eventIds ?? []
  const trackedLifecycleId = trackedEventId && events.has(trackedEventId) ? trackedEventId : null
  const runTrackedLifecycleId = trackedLifecycleId && activeRunEventIds.includes(trackedLifecycleId) ? trackedLifecycleId : null
  const runLifecycleId = activeRunEventIds.find((id) => events.has(id)) ?? null
  const activeId = activeRun
    ? runTrackedLifecycleId ?? runLifecycleId
    : trackedLifecycleId ?? orderedIds.find((id) => events.has(id)) ?? null
  const lifecycle = activeId ? events.get(activeId) : undefined

  const llmRuntime = useMemo(
    () => resolveLLMRuntime(llmStatus, {
      lifecycleModel: lifecycle?.modelUsed,
      loading: llmStatusLoading,
      error: llmStatusError,
    }),
    [llmStatus, lifecycle?.modelUsed, llmStatusLoading, llmStatusError],
  )
  const reached = useMemo(() => reachedStages(lifecycle), [lifecycle])
  const active = useMemo(() => activeStageKey(lifecycle, reached), [lifecycle, reached])
  const stagesReached = reached.size
  const hydratedRunEventCount = activeRunEventIds.filter((id) => events.has(id)).length
  const verdictRunEventCount = activeRunEventIds.filter((id) => events.get(id)?.stages.some((stage) => stage.stage === 'verdict')).length

  return (
    <div className={cn('space-y-3 rounded-lg border border-[#00579C]/25 bg-[#eef5fb] p-3 shadow-sm', className)}>
      <div className="flex flex-wrap items-start justify-between gap-3 rounded-lg border border-[#00579C]/20 bg-white p-3">
        <div className="min-w-0">
          <div className="flex items-center gap-2 text-[10px] font-bold uppercase tracking-[0.14em] text-[#00579C]">
            <Activity className="h-4 w-4 text-[#DA251C]" />
            Pipeline transparency - live evidence board
          </div>
          <p className="mt-1 max-w-4xl text-[10px] leading-relaxed text-text-secondary">
            This view is rendered from SSE event lifecycle state and Event Lab run stages. Empty fields stay pending until the backend emits matching evidence.
          </p>
        </div>
        <div className="grid min-w-[330px] grid-cols-3 gap-2">
          <KV label="Tracked run" value={activeRun ? short(activeRun.runId, 10) : 'none'} mono />
          <KV label="Operator stage" value={`${stagesReached}/${STAGES.length}`} />
          <KV label="LLM status" value={llmRuntime.statusLabel} tone={llmRuntime.running || llmRuntime.reachable ? 'green' : 'red'} />
        </div>
      </div>

      {activeRun && <RunStageLedger run={activeRun} llmRuntime={llmRuntime} />}

      {(activeRun || lifecycle) && (
        <EvidenceSyncPanel
          run={activeRun}
          lifecycle={lifecycle}
          hydratedCount={hydratedRunEventCount || (lifecycle ? 1 : 0)}
          verdictCount={verdictRunEventCount || (lifecycle?.stages.some((stage) => stage.stage === 'verdict') ? 1 : 0)}
          stagesReached={stagesReached}
          totalStages={STAGES.length}
        />
      )}

      <StageProgressStrip reached={reached} active={active} />

      {!lifecycle && !activeRun ? (
        <div className="flex flex-col items-center justify-center rounded-lg border border-dashed border-[#00579C]/25 bg-white py-10 text-center">
          <FileText className="mb-3 h-8 w-8 text-[#00579C]/35" />
          <p className="text-[11px] font-bold uppercase tracking-[0.12em] text-text-primary">No active run selected</p>
          <p className="mt-1 max-w-sm text-[9px] leading-relaxed text-text-secondary">
            Launch an Adaptive Event Lab run or inject a custom event to see backend stages, model outputs, and final verdict evidence here.
          </p>
        </div>
      ) : !lifecycle ? (
        <div className="rounded-lg border border-[#00579C]/25 bg-white p-4 text-[10px] leading-relaxed text-text-secondary">
          The Event Lab run is streaming. Waiting for at least one generated event lifecycle to hydrate through ingestion,
          scoring, graph, control, explanation, and verdict stages.
        </div>
      ) : (
        <div className="grid gap-3 xl:grid-cols-2">
          {STAGES.map((stage) => (
            <StageCard
              key={stage.key}
              config={stage}
              lifecycle={lifecycle}
              reached={reached.has(stage.key)}
              active={active === stage.key}
              llmRuntime={llmRuntime}
            />
          ))}
        </div>
      )}

      <div className="grid gap-3 rounded-lg border border-[#00579C]/20 bg-white p-3 lg:grid-cols-3">
        <div className="rounded-md border border-border-subtle bg-[#f4f8fc] p-3">
          <div className="flex items-center gap-2 text-[9px] font-bold uppercase tracking-[0.12em] text-text-primary">
            <TrendingUp className="h-3.5 w-3.5 text-[#00579C]" />
            Live stage coverage
          </div>
          <p className="mt-1 text-[9px] leading-relaxed text-text-secondary">
            {activeRun
              ? `${activeRun.stages.length} backend run stages recorded; ${hydratedRunEventCount} generated events are hydrated in the frontend lifecycle store.`
              : `${orderedIds.length} event lifecycles are currently retained in the frontend activity buffer.`}
          </p>
        </div>
        <div className="rounded-md border border-border-subtle bg-[#f4f8fc] p-3">
          <div className="flex items-center gap-2 text-[9px] font-bold uppercase tracking-[0.12em] text-text-primary">
            <Shield className="h-3.5 w-3.5 text-[#DA251C]" />
            Decision boundary
          </div>
          <p className="mt-1 text-[9px] leading-relaxed text-text-secondary">
            AI explanation remains non-authoritative. The panel separates observed model narrative from ML score, graph evidence,
            control gate, analyst status, and final verdict.
          </p>
        </div>
        <div className="rounded-md border border-border-subtle bg-[#f4f8fc] p-3">
          <div className="flex items-center gap-2 text-[9px] font-bold uppercase tracking-[0.12em] text-text-primary">
            <MessageSquare className="h-3.5 w-3.5 text-[#00579C]" />
            Current model
          </div>
          <p className="mt-1 text-[9px] leading-relaxed text-text-secondary">
            {llmRuntime.model} is shown from the live LLM status endpoint or selected event lifecycle metadata. If no payload arrives,
            the field stays pending instead of displaying invented output.
          </p>
        </div>
      </div>
    </div>
  )
}

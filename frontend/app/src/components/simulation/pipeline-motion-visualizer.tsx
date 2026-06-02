import { useMemo, useState } from 'react'
import { useActivityStore, type EventLifecycle, type PipelineStage } from '@/stores/use-activity-store'
import { cn, fmtOptionalTimestamp } from '@/lib/utils'
import {
  Activity,
  ArrowRight,
  Bot,
  BrainCircuit,
  CheckCircle2,
  ChevronDown,
  ChevronUp,
  Clock3,
  Database,
  Network,
  Radio,
  Scale,
  ShieldCheck,
  type LucideIcon,
} from 'lucide-react'

interface StageConfig {
  key: PipelineStage
  label: string
  owner: string
  icon: LucideIcon
  color: string
}

const PIPELINE_STAGES: StageConfig[] = [
  { key: 'ingested', label: 'Event ingestion', owner: 'Backend schema gate', icon: Database, color: '#00579C' },
  { key: 'ml_scored', label: 'ML feature engine', owner: 'Risk scoring', icon: BrainCircuit, color: '#00579C' },
  { key: 'graph_investigated', label: 'Graph analysis', owner: 'Fund-flow structure', icon: Network, color: '#2f79b5' },
  { key: 'cb_evaluated', label: 'Control gate', owner: 'Circuit breaker', icon: ShieldCheck, color: '#f5b400' },
  { key: 'llm_started', label: 'AI explanation', owner: 'Bounded Qwen context', icon: Bot, color: '#DA251C' },
  { key: 'verdict', label: 'Final verdict', owner: 'Evidence state', icon: Scale, color: '#B51A13' },
]

const PIPELINE_STAGE_KEYS = new Set<PipelineStage>(PIPELINE_STAGES.map((stage) => stage.key))

type StageStatus = 'idle' | 'active' | 'complete'

function short(value?: string | null, left = 12): string {
  if (!value) return 'n/a'
  return value.length > left + 4 ? `${value.slice(0, left)}...` : value
}

function fmtDuration(ms?: number | null): string {
  if (typeof ms !== 'number' || !Number.isFinite(ms) || ms < 0) return 'n/a'
  return ms < 1000 ? `${Math.round(ms)}ms` : `${(ms / 1000).toFixed(1)}s`
}

function fmtInr(paisa?: number | null): string {
  const rupees = Number(paisa ?? 0) / 100
  if (!Number.isFinite(rupees) || rupees <= 0) return 'pending'
  return `INR ${rupees.toLocaleString('en-IN', { maximumFractionDigits: 0 })}`
}

function hasTimestamp(value: unknown): value is number {
  return typeof value === 'number' && Number.isFinite(value) && value > 0
}

function countRecognizedStages(lifecycle: EventLifecycle | undefined): number {
  if (!lifecycle) return 0
  const reached = new Set<PipelineStage>()
  lifecycle.stages.forEach((detail) => {
    if (PIPELINE_STAGE_KEYS.has(detail.stage)) reached.add(detail.stage)
  })
  return reached.size
}

function getStageStatuses(lifecycle: EventLifecycle | undefined): Map<PipelineStage, StageStatus> {
  const statuses = new Map<PipelineStage, StageStatus>()
  const completed = new Set(lifecycle?.stages.map((stage) => stage.stage) ?? [])
  let lastCompleteIndex = -1

  PIPELINE_STAGES.forEach((stage, index) => {
    if (completed.has(stage.key)) lastCompleteIndex = index
  })

  PIPELINE_STAGES.forEach((stage, index) => {
    if (completed.has(stage.key)) {
      statuses.set(stage.key, 'complete')
    } else if (lifecycle && index === lastCompleteIndex + 1) {
      statuses.set(stage.key, 'active')
    } else {
      statuses.set(stage.key, 'idle')
    }
  })

  return statuses
}

function getStageDuration(lifecycle: EventLifecycle | undefined, stageKey: PipelineStage): string {
  if (!lifecycle) return 'pending'
  const detail = lifecycle.stages.find((stage) => stage.stage === stageKey)
  if (!detail) return 'pending'
  if (typeof detail.durationMs === 'number' && Number.isFinite(detail.durationMs) && detail.durationMs >= 0) {
    return fmtDuration(detail.durationMs)
  }
  const index = lifecycle.stages.findIndex((stage) => stage.stage === stageKey)
  if (index <= 0) return '0ms'
  const current = lifecycle.stages[index].timestamp
  const previous = lifecycle.stages[index - 1].timestamp
  if (!hasTimestamp(current) || !hasTimestamp(previous) || current < previous) return 'pending'
  return fmtDuration((current - previous) * 1000)
}

function StageNode({
  config,
  status,
  duration,
  selected,
  onClick,
}: {
  config: StageConfig
  status: StageStatus
  duration: string
  selected: boolean
  onClick: () => void
}) {
  const Icon = config.icon
  return (
    <button
      type="button"
      onClick={onClick}
      className={cn(
        'group min-w-[160px] flex-1 rounded-lg border bg-white p-3 text-left transition-all',
        status === 'complete' && 'border-[#00579C]/35 shadow-sm',
        status === 'active' && 'border-[#DA251C]/35 bg-[#DA251C]/5 shadow-[0_10px_28px_rgba(218,37,28,0.12)]',
        status === 'idle' && 'border-border-subtle bg-bg-elevated/45 opacity-75',
        selected && 'ring-2 ring-[#00579C]/25',
      )}
    >
      <div className="flex items-start justify-between gap-2">
        <div className={cn('flex h-9 w-9 items-center justify-center rounded-lg', status === 'idle' ? 'bg-bg-elevated text-text-muted' : 'text-white')} style={status === 'idle' ? undefined : { background: config.color }}>
          <Icon className="h-4 w-4" />
        </div>
        {status === 'complete' ? (
          <CheckCircle2 className="h-4 w-4 text-alert-low" />
        ) : status === 'active' ? (
          <Activity className="h-4 w-4 animate-pulse text-[#DA251C]" />
        ) : (
          <Clock3 className="h-4 w-4 text-text-muted" />
        )}
      </div>
      <div className="mt-3 text-[9px] font-bold uppercase tracking-[0.1em] text-text-primary">{config.label}</div>
      <div className="mt-0.5 text-[8px] text-text-muted">{config.owner}</div>
      <div className="mt-2 inline-flex rounded-full border border-border-subtle bg-white px-2 py-0.5 font-mono text-[8px] font-bold text-text-secondary">
        {status === 'active' ? 'processing' : status === 'complete' ? duration : 'pending'}
      </div>
    </button>
  )
}

function Connector({ active }: { active: boolean }) {
  return (
    <div className="hidden w-8 items-center justify-center xl:flex">
      <div className={cn('h-px flex-1', active ? 'bg-[#00579C]' : 'bg-border-subtle')} />
      <ArrowRight className={cn('h-3.5 w-3.5', active ? 'text-[#00579C]' : 'text-text-muted/50')} />
    </div>
  )
}

function PipelineGlobalStats({
  events,
  orderedIds,
  filterIds,
}: {
  events: Map<string, EventLifecycle>
  orderedIds: string[]
  filterIds?: string[]
}) {
  const stats = useMemo(() => {
    let processing = 0
    let completed = 0
    let fraudulent = 0
    const ids = filterIds?.length ? filterIds : orderedIds.slice(0, 50)

    for (const id of ids) {
      const lifecycle = events.get(id)
      if (!lifecycle) continue
      const stages = new Set(lifecycle.stages.map((stage) => stage.stage))
      if (stages.has('verdict')) {
        completed += 1
        if (lifecycle.verdict === 'fraudulent') fraudulent += 1
      } else if (stages.size > 0) {
        processing += 1
      }
    }

    return { processing, completed, fraudulent, total: ids.length }
  }, [events, filterIds, orderedIds])

  return (
    <div className="grid gap-2 md:grid-cols-4">
      {[
        ['In pipeline', stats.processing, 'blue'],
        ['Completed', stats.completed, 'green'],
        ['Flagged fraud', stats.fraudulent, 'red'],
        ['Tracked ids', stats.total, 'blue'],
      ].map(([label, value, tone]) => (
        <div key={label} className="rounded-md border border-border-subtle bg-white p-2">
          <div className="text-[7px] font-bold uppercase tracking-[0.12em] text-text-muted">{label}</div>
          <div className={cn(
            'mt-1 font-mono text-sm font-bold tabular-nums',
            tone === 'green' && 'text-alert-low',
            tone === 'red' && 'text-[#DA251C]',
            tone === 'blue' && 'text-[#00579C]',
          )}>
            {value}
          </div>
        </div>
      ))}
    </div>
  )
}

function EventSelector({
  events,
  orderedIds,
  selectedId,
  filterIds,
  onSelect,
}: {
  events: Map<string, EventLifecycle>
  orderedIds: string[]
  selectedId: string | null
  filterIds?: string[]
  onSelect: (id: string | null) => void
}) {
  const recentEvents = useMemo(() => {
    const ids = filterIds?.length ? filterIds : orderedIds.slice(0, 12)
    return ids
      .map((id) => {
        const lifecycle = events.get(id)
        if (!lifecycle) return null
        return {
          id,
          lifecycle,
          stagesCompleted: countRecognizedStages(lifecycle),
          totalStages: PIPELINE_STAGES.length,
        }
      })
      .filter((event): event is { id: string; lifecycle: EventLifecycle; stagesCompleted: number; totalStages: number } => Boolean(event))
  }, [events, filterIds, orderedIds])

  if (recentEvents.length === 0) {
    return (
      <div className="rounded-md border border-dashed border-[#00579C]/25 bg-[#00579C]/5 p-3 text-center text-[9px] text-text-secondary">
        No matching run events are hydrated in the pipeline yet.
      </div>
    )
  }

  return (
    <div className="max-h-48 space-y-1 overflow-y-auto pr-1">
      {recentEvents.map((event) => {
        const isSelected = event.id === selectedId
        const hasVerdict = event.lifecycle.stages.some((stage) => stage.stage === 'verdict')
        const progressPct = Math.round((event.stagesCompleted / event.totalStages) * 100)

        return (
          <button
            key={event.id}
            type="button"
            onClick={() => onSelect(isSelected ? null : event.id)}
            className={cn(
              'grid w-full grid-cols-[28px_minmax(0,1fr)_48px] items-center gap-2 rounded-md border px-2 py-1.5 text-left transition-colors',
              isSelected ? 'border-[#00579C]/35 bg-[#00579C]/10' : 'border-border-subtle bg-white hover:bg-[#f4f8fc]',
            )}
          >
            <div className={cn('flex h-6 w-6 items-center justify-center rounded-full border', hasVerdict ? 'border-alert-low/30 bg-alert-low/10 text-alert-low' : 'border-[#00579C]/25 bg-[#00579C]/10 text-[#00579C]')}>
              {hasVerdict ? <CheckCircle2 className="h-3.5 w-3.5" /> : <Activity className="h-3.5 w-3.5 animate-pulse" />}
            </div>
            <div className="min-w-0">
              <div className="flex min-w-0 items-center gap-2">
                <code className="truncate font-mono text-[8px] font-bold text-text-primary">{short(event.id, 14)}</code>
                {event.lifecycle.fraudLabel > 0 && (
                  <span className="rounded bg-[#DA251C]/10 px-1 py-0.5 text-[6px] font-bold uppercase text-[#DA251C]">fraud</span>
                )}
              </div>
              <div className="mt-0.5 truncate text-[7px] text-text-muted">
                {event.lifecycle.sender || 'sender'} to {event.lifecycle.receiver || 'receiver'} - {fmtInr(event.lifecycle.amountPaisa)}
              </div>
            </div>
            <span className="text-right font-mono text-[8px] font-bold text-text-secondary">{progressPct}%</span>
          </button>
        )
      })}
    </div>
  )
}

function EventEvidencePanel({
  lifecycle,
  selectedStage,
}: {
  lifecycle?: EventLifecycle
  selectedStage: PipelineStage | null
}) {
  if (!lifecycle) {
    return (
      <div className="rounded-lg border border-dashed border-[#00579C]/25 bg-white p-4 text-[10px] text-text-secondary">
        Waiting for a generated event lifecycle to hydrate through SSE.
      </div>
    )
  }

  const stageDetail = selectedStage ? lifecycle.stages.find((stage) => stage.stage === selectedStage) : lifecycle.stages.at(-1)
  const stageLabel = selectedStage ? PIPELINE_STAGES.find((stage) => stage.key === selectedStage)?.label : 'Latest observed stage'

  return (
    <div className="rounded-lg border border-[#00579C]/25 bg-white p-3">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div className="flex items-center gap-2 text-[9px] font-bold uppercase tracking-[0.12em] text-[#00579C]">
          <Radio className="h-3.5 w-3.5 text-[#DA251C]" />
          {stageLabel}
        </div>
        <span className="rounded-full border border-border-subtle bg-bg-elevated px-2 py-0.5 font-mono text-[8px] text-text-secondary">
          {stageDetail ? fmtOptionalTimestamp(stageDetail.timestamp) : 'pending'}
        </span>
      </div>
      <div className="mt-3 grid gap-2 md:grid-cols-4">
        <Info label="Event" value={short(lifecycle.txnId, 14)} mono />
        <Info label="Amount" value={fmtInr(lifecycle.amountPaisa)} />
        <Info label="Risk" value={lifecycle.riskScore != null ? `${Math.round(lifecycle.riskScore * 100)}% ${lifecycle.riskTier ?? ''}` : 'pending'} tone={lifecycle.riskScore && lifecycle.riskScore > 0.7 ? 'red' : 'blue'} />
        <Info label="Verdict" value={lifecycle.verdict ?? 'pending'} tone={lifecycle.verdict === 'fraudulent' ? 'red' : 'blue'} />
      </div>
      <div className="mt-2 grid grid-cols-[minmax(0,1fr)_28px_minmax(0,1fr)] items-center gap-2">
        <div className="truncate rounded-md border border-border-subtle bg-[#f4f8fc] px-2 py-2 font-mono text-[8px] font-bold text-text-primary">
          {lifecycle.sender || 'sender pending'}
        </div>
        <span className="text-center text-[10px] font-bold text-[#00579C]">to</span>
        <div className="truncate rounded-md border border-border-subtle bg-[#f4f8fc] px-2 py-2 font-mono text-[8px] font-bold text-text-primary">
          {lifecycle.receiver || 'receiver pending'}
        </div>
      </div>
      {stageDetail?.meta && Object.keys(stageDetail.meta).length > 0 && (
        <div className="mt-2 flex flex-wrap gap-1">
          {Object.entries(stageDetail.meta).slice(0, 8).map(([key, value]) => (
            <span key={key} className="rounded border border-[#00579C]/15 bg-[#00579C]/5 px-1.5 py-0.5 font-mono text-[7px] text-text-secondary">
              {key.replace(/_/g, ' ')}={String(value).slice(0, 44)}
            </span>
          ))}
        </div>
      )}
      {lifecycle.topFeatures && lifecycle.topFeatures.length > 0 && (
        <div className="mt-2 flex flex-wrap gap-1">
          {lifecycle.topFeatures.slice(0, 5).map((feature) => (
            <span key={feature} className="rounded border border-[#00579C]/20 bg-[#00579C]/10 px-1.5 py-0.5 text-[7px] font-bold uppercase text-[#00579C]">
              {feature}
            </span>
          ))}
        </div>
      )}
    </div>
  )
}

function Info({ label, value, mono = false, tone = 'blue' }: { label: string; value: string; mono?: boolean; tone?: 'blue' | 'red' | 'green' }) {
  return (
    <div className="rounded-md border border-border-subtle bg-[#f4f8fc] px-2 py-1.5">
      <div className="text-[7px] font-bold uppercase tracking-[0.12em] text-text-muted">{label}</div>
      <div className={cn('mt-0.5 truncate text-[9px] font-bold', mono && 'font-mono', tone === 'red' ? 'text-[#DA251C]' : tone === 'green' ? 'text-alert-low' : 'text-[#00579C]')}>
        {value}
      </div>
    </div>
  )
}

interface PipelineMotionVisualizerProps {
  trackedEventId?: string | null
  compact?: boolean
  className?: string
}

export function PipelineMotionVisualizer({ trackedEventId, compact, className }: PipelineMotionVisualizerProps) {
  const events = useActivityStore((state) => state.events)
  const orderedIds = useActivityStore((state) => state.orderedIds)
  const eventLabRuns = useActivityStore((state) => state.eventLabRuns)
  const activeEventLabRunId = useActivityStore((state) => state.activeEventLabRunId)
  const [selectedId, setSelectedId] = useState<string | null>(null)
  const [selectedStage, setSelectedStage] = useState<PipelineStage | null>(null)
  const [expanded, setExpanded] = useState(!compact)

  const trackedLifecycleId = trackedEventId && events.has(trackedEventId) ? trackedEventId : null
  const activeRun = activeEventLabRunId ? eventLabRuns[activeEventLabRunId] : undefined
  const activeRunEventIds = useMemo(() => activeRun?.eventIds ?? [], [activeRun?.eventIds])

  const autoSelectedId = useMemo(() => {
    if (activeRun) {
      if (trackedLifecycleId && activeRunEventIds.includes(trackedLifecycleId)) return trackedLifecycleId
      if (selectedId && activeRunEventIds.includes(selectedId) && events.has(selectedId)) return selectedId
      return activeRunEventIds.find((id) => events.has(id)) ?? null
    }
    if (trackedLifecycleId) return trackedLifecycleId
    if (selectedId && events.has(selectedId)) return selectedId
    return orderedIds.find((id) => events.has(id)) ?? null
  }, [activeRun, activeRunEventIds, trackedLifecycleId, selectedId, events, orderedIds])

  const activeId = autoSelectedId
  const lifecycle = activeId ? events.get(activeId) : undefined
  const stageStatuses = useMemo(() => getStageStatuses(lifecycle), [lifecycle])
  const currentStages = countRecognizedStages(lifecycle)
  const completionPct = lifecycle ? Math.round((currentStages / PIPELINE_STAGES.length) * 100) : 0

  return (
    <section className={cn('rounded-lg border border-[#00579C]/25 bg-[#eef5fb] p-3 shadow-sm', className)}>
      <div className="flex flex-wrap items-start justify-between gap-3 rounded-lg border border-[#00579C]/20 bg-white p-3">
        <div className="min-w-0">
          <div className="flex items-center gap-2 text-[10px] font-bold uppercase tracking-[0.14em] text-[#00579C]">
            <Activity className="h-4 w-4 text-[#DA251C]" />
            Live processing pipeline
          </div>
          <p className="mt-1 max-w-4xl text-[10px] leading-relaxed text-text-secondary">
            {lifecycle
              ? `Tracking ${short(activeId, 16)} through ${currentStages}/${PIPELINE_STAGES.length} observed operator stages.`
              : activeRun
                ? `Tracking run ${short(activeRun.runId, 14)} while generated event lifecycles hydrate from SSE.`
                : 'Launch an Event Lab chain or inject a custom fraud event to watch live backend processing.'}
          </p>
        </div>
        <div className="flex items-center gap-3">
          <div className="min-w-[120px]">
            <div className="mb-1 flex items-center justify-between text-[8px] font-bold uppercase tracking-[0.12em] text-text-muted">
              <span>Complete</span>
              <span>{completionPct}%</span>
            </div>
            <div className="h-2 overflow-hidden rounded-full bg-bg-elevated">
              <div className={cn('h-full rounded-full transition-all duration-700', completionPct >= 100 ? 'bg-alert-low' : 'bg-[#00579C]')} style={{ width: `${completionPct}%` }} />
            </div>
          </div>
          {!compact && (
            <button
              type="button"
              onClick={() => setExpanded((value) => !value)}
              className="inline-flex h-8 w-8 items-center justify-center rounded-md border border-[#00579C]/20 bg-white text-[#00579C] hover:bg-[#00579C]/10"
              aria-label={expanded ? 'Collapse pipeline details' : 'Expand pipeline details'}
            >
              {expanded ? <ChevronUp className="h-4 w-4" /> : <ChevronDown className="h-4 w-4" />}
            </button>
          )}
        </div>
      </div>

      <div className="mt-3 space-y-3">
        <PipelineGlobalStats events={events} orderedIds={orderedIds} filterIds={activeRun ? activeRunEventIds : undefined} />

        {activeRun && !lifecycle && (
          <div className="rounded-md border border-[#00579C]/25 bg-white px-3 py-2 text-[9px] leading-relaxed text-text-secondary">
            Active Event Lab run <span className="font-mono font-bold text-[#00579C]">{activeRun.runId}</span> has {activeRun.eventIds.length} generated IDs.
            The pipeline animates when matching lifecycle stages arrive from backend SSE.
          </div>
        )}

        <div className="flex flex-col gap-2 xl:flex-row xl:items-stretch">
          {PIPELINE_STAGES.map((stage, index) => {
            const rawStatus = stageStatuses.get(stage.key) ?? 'idle'
            const visualStatus = completionPct >= 100 ? 'complete' : rawStatus
            return (
              <div key={stage.key} className="flex flex-1 items-center gap-2">
                <StageNode
                  config={stage}
                  status={visualStatus}
                  duration={getStageDuration(lifecycle, stage.key)}
                  selected={selectedStage === stage.key}
                  onClick={() => setSelectedStage(selectedStage === stage.key ? null : stage.key)}
                />
                {index < PIPELINE_STAGES.length - 1 && (
                  <Connector active={visualStatus === 'complete'} />
                )}
              </div>
            )
          })}
        </div>

        <EventEvidencePanel lifecycle={lifecycle} selectedStage={selectedStage} />

        {expanded && !compact && (
          <div className="rounded-lg border border-[#00579C]/20 bg-white p-3">
            <div className="mb-2 flex items-center justify-between gap-2">
              <div className="flex items-center gap-2 text-[9px] font-bold uppercase tracking-[0.12em] text-text-primary">
                <Radio className="h-3.5 w-3.5 text-[#00579C]" />
                Recent pipeline events
              </div>
              {activeId && (
                <button
                  type="button"
                  onClick={() => { setSelectedId(null); setSelectedStage(null) }}
                  className="text-[8px] font-bold uppercase tracking-[0.12em] text-[#00579C] hover:text-[#DA251C]"
                >
                  Clear selection
                </button>
              )}
            </div>
            <EventSelector
              events={events}
              orderedIds={orderedIds}
              selectedId={activeId ?? null}
              filterIds={activeRun ? activeRunEventIds : undefined}
              onSelect={(id) => { setSelectedId(id); setSelectedStage(null) }}
            />
          </div>
        )}
      </div>
    </section>
  )
}

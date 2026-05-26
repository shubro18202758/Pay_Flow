import { useMemo, useState } from 'react'
import type { ReactNode } from 'react'
import {
  Activity,
  BadgeCheck,
  BrainCircuit,
  CheckCircle2,
  ClipboardCheck,
  Clock3,
  FileText,
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
  usePreviewEventLabRun,
  useRejectCountermeasure,
} from '@/hooks/use-api'
import { useActivityStore } from '@/stores/use-activity-store'
import { cn } from '@/lib/utils'
import type {
  CountermeasureProposal,
  EventLabExplainabilityResponse,
  EventLabGeneratedEvent,
  EventLabMode,
  EventLabRunResponse,
  EventLabTemplate,
} from '@/lib/types'

const MODES: EventLabMode[] = ['chain', 'burst', 'single']

function fmtPct(value?: number) {
  if (value == null || Number.isNaN(value)) return '0%'
  return `${Math.round(value * 100)}%`
}

function fmtAmount(paisa?: number) {
  if (!paisa) return 'INR 0'
  return `INR ${(paisa / 100).toLocaleString('en-IN', { maximumFractionDigits: 0 })}`
}

function fmtSeconds(seconds?: number) {
  if (seconds == null) return '0s'
  if (seconds < 60) return `${Math.max(0, Math.round(seconds))}s`
  return `${Math.floor(seconds / 60)}m ${Math.round(seconds % 60)}s`
}

function short(value?: string, left = 8) {
  if (!value) return 'n/a'
  return value.length > left + 4 ? `${value.slice(0, left)}...` : value
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

function RunTimeline({
  run,
  explainability,
}: {
  run?: EventLabRunResponse
  explainability?: EventLabExplainabilityResponse
}) {
  const groups = explainability?.stage_groups ?? []
  const evidence = explainability?.evidence_panels ?? []
  const runtime = explainability?.runtime
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
            <Metric label="Known Latency" value={`${run?.latency_metrics?.known_stage_latency_ms ?? 0} ms`} />
          </div>

          {groups.length === 0 ? (
            <div className="flex h-48 items-center justify-center rounded-md border border-dashed border-border-default bg-bg-elevated/50 text-center text-[10px] font-semibold uppercase tracking-[0.12em] text-text-muted">
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
                            {stage.duration_ms != null ? `${stage.duration_ms} ms` : new Date(stage.timestamp * 1000).toLocaleTimeString()}
                          </span>
                        </div>
                        <p className="mt-0.5 line-clamp-2 text-[8px] text-text-secondary">{stage.evidence_summary}</p>
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
              Qwen 3.5 4B bounded copilot
            </div>
            <p className="text-[10px] leading-relaxed text-text-secondary">
              {run?.qwen_explanation ?? 'Qwen explanation appears after preview or launch. It remains a narrative copilot and cannot approve countermeasures.'}
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
                <p className="mt-2 line-clamp-3 text-[9px] leading-relaxed text-text-secondary">{panel.summary}</p>
                <div className="mt-2 grid grid-cols-2 gap-1">
                  {Object.entries(panel.metrics).slice(0, 4).map(([label, value]) => (
                    <div key={label} className="rounded bg-bg-elevated/70 px-2 py-1">
                      <div className="text-[7px] font-bold uppercase tracking-wide text-text-muted">{label.replaceAll('_', ' ')}</div>
                      <div className="mt-0.5 truncate font-mono text-[9px] font-bold text-text-primary">{String(value)}</div>
                    </div>
                  ))}
                </div>
                <div className="mt-2 flex flex-wrap gap-1">
                  {panel.items.filter(Boolean).slice(0, 3).map((item) => (
                    <span key={item} className="max-w-full truncate rounded bg-bg-elevated px-1.5 py-0.5 text-[8px] text-text-secondary">
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
}: {
  proposal: CountermeasureProposal
  onApprove: (id: string) => void
  onReject: (id: string) => void
  busy: boolean
}) {
  const executable = proposal.execution_allowed && proposal.status === 'proposed'
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
          <p className="mt-1 line-clamp-2 text-[9px] leading-relaxed text-text-secondary">{proposal.reason}</p>
          <div className="mt-2 grid grid-cols-2 gap-1 sm:grid-cols-4">
            <div className="rounded bg-bg-surface px-2 py-1">
              <div className="text-[7px] font-bold uppercase tracking-wide text-text-muted">TTL</div>
              <div className="font-mono text-[9px] font-bold text-text-primary">{fmtSeconds(proposal.expires_at - Date.now() / 1000)}</div>
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
            disabled={!executable || busy}
            title={proposal.execution_allowed ? 'Approve countermeasure' : 'Advisory-only proposal cannot execute'}
            className="inline-flex h-8 items-center gap-1 rounded-md border border-alert-low/30 bg-alert-low/10 px-2 text-[9px] font-bold uppercase tracking-[0.1em] text-alert-low disabled:cursor-not-allowed disabled:opacity-40"
          >
            {busy ? <Loader2 className="h-3 w-3 animate-spin" /> : <CheckCircle2 className="h-3 w-3" />}
            Approve
          </button>
          <button
            type="button"
            onClick={() => onReject(proposal.proposal_id)}
            disabled={proposal.status !== 'proposed' || busy}
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
        <div className="rounded-md border border-border-subtle bg-white p-2">
          <div className="mb-2 text-[9px] font-bold uppercase tracking-[0.12em] text-text-primary">
            Decision authority chain
          </div>
          <div className="space-y-1.5">
            {authority.length === 0 ? (
              <div className="text-[9px] text-text-muted">Run explainability appears after launch.</div>
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
              Countermeasure proposals appear after launch
            </div>
          ) : proposals.map((proposal) => (
            <ProposalRow
              key={proposal.proposal_id}
              proposal={proposal}
              busy={approve.isPending || reject.isPending}
              onApprove={(id) => void approve.mutateAsync(id)}
              onReject={(id) => void reject.mutateAsync(id)}
            />
          ))}
        </div>
      </div>
    </Panel>
  )
}

export function AdaptiveEventLab() {
  const { data: templatesData, isLoading } = useEventLabTemplates()
  const preview = usePreviewEventLabRun()
  const launch = useCreateEventLabRun()
  const activeRunFromSse = useActivityStore((s) => s.activeEventLabRunId)
  const setActiveRun = useActivityStore((s) => s.setActiveEventLabRunId)
  const setTrackedEventId = useActivityStore((s) => s.setTrackedEventId)
  const [selectedTemplateId, setSelectedTemplateId] = useState<string | null>(null)
  const [mode, setMode] = useState<EventLabMode>('chain')
  const [intensity, setIntensity] = useState<'demo' | 'scale'>('demo')
  const [seed, setSeed] = useState(2026)
  const [localRunId, setLocalRunId] = useState<string | null>(null)

  const templates = useMemo(() => templatesData?.templates ?? [], [templatesData?.templates])
  const selected = useMemo(() => {
    if (!templates.length) return undefined
    return templates.find((item) => item.template_id === selectedTemplateId) ?? templates[0]
  }, [selectedTemplateId, templates])
  const activeRunId = localRunId ?? activeRunFromSse
  const { data: run } = useEventLabRun(activeRunId)
  const { data: explainability } = useEventLabExplainability(activeRunId)
  const previewEvents = preview.data?.run_preview.events ?? run?.events ?? []
  const policy = preview.data?.run_preview.countermeasure_policy ?? run?.countermeasure_policy
  const trust = policy?.source_trust ?? 0

  async function handlePreview() {
    if (!selected) return
    await preview.mutateAsync({
      template_id: selected.template_id,
      playbook_id: selected.linked_playbooks?.[0]?.playbook_id ?? null,
      mode,
      intensity,
      seed,
    })
  }

  async function handleLaunch() {
    if (!selected) return
    const response = await launch.mutateAsync({
      template_id: selected.template_id,
      playbook_id: selected.linked_playbooks?.[0]?.playbook_id ?? null,
      mode,
      intensity,
      seed,
      analyst_required: true,
    })
    setLocalRunId(response.run_id)
    setActiveRun(response.run_id)
    if (response.event_ids[0]) setTrackedEventId(response.event_ids[0])
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
              External OSINT/SOCMINT playbooks shape the event chain and Qwen explanation. Internal PayFlow graph, ML, rules,
              circuit breaker, and ledger remain authoritative before any analyst-approved action executes.
            </p>
          </div>
          <div className="grid grid-cols-3 gap-2">
            <Metric label="Templates" value={isLoading ? '...' : String(templates.length)} />
            <Metric label="Intel Trust" value={fmtPct(trust)} tone={trust >= 0.85 ? 'green' : 'amber'} />
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
                    {(['demo', 'scale'] as const).map((item) => (
                      <button
                        key={item}
                        type="button"
                        onClick={() => setIntensity(item)}
                        className={cn(
                          'rounded-md border px-2 py-1.5 text-[9px] font-bold uppercase tracking-wide',
                          intensity === item ? 'border-accent-primary bg-accent-primary text-white' : 'border-border-subtle bg-bg-surface text-text-secondary',
                        )}
                      >
                        {item}
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

              <div className="grid gap-2 md:grid-cols-4">
                <Metric label="Policy" value={policy?.authority?.replaceAll('_', ' ') ?? 'approval'} />
                <Metric label="Execution" value={policy?.execution_allowed ? 'allowed' : 'advisory'} tone={policy?.execution_allowed ? 'green' : 'amber'} />
                <Metric label="Qwen" value={policy?.qwen_role ? 'bounded' : 'ready'} />
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
                  disabled={!selected || launch.isPending}
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

              <EventPreview events={previewEvents} />
            </div>
          </Panel>
        </div>

        <CountermeasureConsole runId={activeRunId} explainability={explainability} />
      </div>

      <RunTimeline run={explainability?.run ?? run} explainability={explainability} />

      <Panel title="Countering Logic Transparency" icon={ShieldAlert}>
        <div className="grid gap-3 p-3 md:grid-cols-4">
          {[
            { icon: Radar, title: 'Intel primes', body: 'Active playbooks select scenario seeds, watch terms, and Qwen context.' },
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

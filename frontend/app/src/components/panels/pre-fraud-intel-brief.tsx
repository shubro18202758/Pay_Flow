// ============================================================================
// Pre-Fraud Intel Brief -- shared OSINT/SOCMINT bridge for core fraud surfaces
// ============================================================================

import {
  Activity,
  BrainCircuit,
  ExternalLink,
  Globe2,
  Radar,
  RefreshCw,
  ShieldCheck,
  Sparkles,
} from 'lucide-react'
import type { ComponentType } from 'react'
import {
  useIntelPlaybooks,
  useIntelSignals,
  useIntelSources,
  useIntelTrends,
  useIntelTuningStatus,
  useRefreshIntel,
  useSimulateIntelSignal,
} from '@/hooks/use-api'
import { useUIStore } from '@/stores/use-ui-store'
import { cn } from '@/lib/utils'
import type { PreFraudEvidenceContext } from '@/lib/types'

type Variant = 'overview' | 'sidebar' | 'drawer' | 'case' | 'evidence'

function pct(value: number | undefined) {
  return `${Math.round((value ?? 0) * 100)}%`
}

function shortHash(value: string | undefined) {
  if (!value) return 'n/a'
  return value.length > 10 ? `${value.slice(0, 10)}...` : value
}

function titleCase(value: string) {
  return value.replace(/_/g, ' ').replace(/\b\w/g, (ch) => ch.toUpperCase())
}

function getEvidenceContext(context?: PreFraudEvidenceContext | null) {
  return {
    trends: context?.top_trends ?? [],
    playbooks: context?.active_playbooks ?? [],
    status: context?.tuning_status ?? null,
  }
}

export function PreFraudIntelBrief({
  variant = 'overview',
  context,
  className,
}: {
  variant?: Variant
  context?: PreFraudEvidenceContext | null
  className?: string
}) {
  const setActiveTab = useUIStore((s) => s.setActiveTab)
  const { data: sources } = useIntelSources()
  const { data: signals } = useIntelSignals()
  const { data: trends } = useIntelTrends()
  const { data: playbooks } = useIntelPlaybooks()
  const { data: tuning } = useIntelTuningStatus()
  const refresh = useRefreshIntel()
  const simulate = useSimulateIntelSignal()

  const evidence = getEvidenceContext(context)
  const topTrend = evidence.trends[0] ?? trends?.trends?.[0]
  const activePlaybooks = evidence.playbooks.length
    ? evidence.playbooks
    : (playbooks?.playbooks ?? []).filter((p) => p.promotion_status === 'applied')
  const status = evidence.status ?? playbooks?.tuning_status ?? tuning ?? null
  const sourceCount = context?.source_count ?? sources?.sources.length ?? 0
  const signalCount = context?.signal_count ?? signals?.count ?? 0
  const activeCount = context?.active_playbooks.length ?? status?.active_playbooks ?? activePlaybooks.length
  const compact = variant === 'sidebar' || variant === 'evidence'
  const dense = variant === 'overview'
  const busy = refresh.isPending || simulate.isPending

  async function primeIntel() {
    await refresh.mutateAsync(undefined)
    await simulate.mutateAsync('digital_arrest_mule')
  }

  if (variant === 'overview') {
    return (
      <section className={cn('ubi-page-band border-b bg-bg-surface/90 px-4 py-2.5', className)}>
        <div className="flex flex-wrap items-center gap-3">
          <div className="flex min-w-0 items-center gap-2">
            <div className="flex h-8 w-8 items-center justify-center rounded-md border border-accent-primary/25 bg-accent-primary/10">
              <Radar className="h-4 w-4 text-accent-primary" />
            </div>
            <div className="min-w-0">
              <div className="flex items-center gap-2">
                <span className="text-[10px] font-extrabold uppercase tracking-[0.14em] text-text-primary">
                  Union Bank Pre-Fraud Intelligence Layer
                </span>
                <span className="rounded border border-alert-low/25 bg-alert-low/10 px-2 py-0.5 text-[8px] font-bold uppercase tracking-[0.12em] text-alert-low">
                  {activeCount} active playbooks
                </span>
              </div>
              <p className="truncate text-[9px] text-text-muted">
                {topTrend?.title ?? 'Public-source fraud signals fused before graph detection and FIU evidence generation'}
              </p>
            </div>
          </div>
          <div className="flex min-w-0 flex-1 items-center gap-2 overflow-x-auto">
            <MiniMetric label="sources" value={String(sourceCount)} />
            <MiniMetric label="signals" value={String(signalCount)} />
            <MiniMetric label="trust" value={pct(topTrend?.trust_score)} />
            <MiniMetric label="india fit" value={pct(topTrend?.india_relevance_score)} />
            <MiniMetric label="bounded tuning" value={status?.rollback_available ? 'audited' : 'shadow'} />
            <MiniMetric label="qwen" value={status?.qwen_model ?? 'qwen3.5:4b'} />
          </div>
          <div className="flex items-center gap-2">
            <button
              onClick={() => void primeIntel()}
              disabled={busy}
              className="inline-flex items-center gap-1.5 rounded-md border border-accent-primary/40 bg-bg-surface px-2.5 py-1.5 text-[9px] font-bold uppercase tracking-[0.12em] text-accent-primary transition-colors hover:bg-accent-primary hover:text-white disabled:opacity-50"
            >
              <Sparkles className="h-3.5 w-3.5" />
              Prime Intel
            </button>
            <button
              onClick={() => setActiveTab('pre-fraud-intel')}
              className="inline-flex items-center gap-1.5 rounded-md border border-border-default px-2.5 py-1.5 text-[9px] font-bold uppercase tracking-[0.12em] text-text-secondary transition-colors hover:border-accent-primary hover:text-accent-primary"
            >
              <ExternalLink className="h-3.5 w-3.5" />
              Radar
            </button>
          </div>
        </div>
      </section>
    )
  }

  return (
    <section
      className={cn(
        'rounded-lg border border-accent-primary/20 bg-bg-surface shadow-sm',
        compact ? 'p-2.5' : 'p-3',
        className,
      )}
    >
      <div className="mb-2 flex items-center justify-between gap-2">
        <div className="flex min-w-0 items-center gap-2">
          <Radar className="h-4 w-4 shrink-0 text-accent-primary" />
          <div className="min-w-0">
            <div className="truncate text-[10px] font-bold uppercase tracking-[0.14em] text-text-primary">
              Pre-Fraud Intel Bridge
            </div>
            {!compact && (
              <div className="truncate text-[9px] text-text-muted">
                Preventive context before graph detection, not a verdict override
              </div>
            )}
          </div>
        </div>
        <span className="shrink-0 rounded-md border border-alert-low/25 bg-alert-low/10 px-2 py-1 text-[8px] font-bold uppercase tracking-[0.12em] text-alert-low">
          {activeCount} active playbooks
        </span>
      </div>

      <div className={cn('grid gap-2', dense ? 'grid-cols-5' : compact ? 'grid-cols-2' : 'grid-cols-4')}>
        <MiniMetric label="sources" value={String(sourceCount)} />
        <MiniMetric label="signals" value={String(signalCount)} />
        <MiniMetric label="top trust" value={pct(topTrend?.trust_score)} />
        <MiniMetric label="india fit" value={pct(topTrend?.india_relevance_score)} />
      </div>

      <div className="mt-2 rounded-md border border-border-subtle bg-bg-elevated/45 p-2">
        <div className="mb-1 flex items-center gap-1.5 text-[9px] font-semibold uppercase tracking-[0.12em] text-text-secondary">
          <Globe2 className="h-3 w-3 text-accent-primary" />
          Emerging External Pattern
        </div>
        <p className="line-clamp-2 text-[9px] leading-relaxed text-text-muted">
          {topTrend?.title ?? 'No live external cluster loaded yet'}
        </p>
        <div className="mt-1.5 flex flex-wrap gap-1">
          {(topTrend?.typologies ?? []).slice(0, 3).map((typology) => (
            <span key={typology} className="rounded bg-bg-overlay px-1.5 py-0.5 text-[8px] font-semibold text-text-muted">
              {titleCase(typology)}
            </span>
          ))}
        </div>
      </div>

      <div className="mt-2 space-y-1.5">
        {activePlaybooks.slice(0, compact ? 2 : 3).map((playbook) => (
          <div key={playbook.playbook_id} className="rounded-md border border-border-subtle bg-bg-elevated/45 p-2">
            <div className="mb-1 flex items-center justify-between gap-2">
              <span className="truncate text-[9px] font-semibold text-text-primary" title={playbook.title}>
                {playbook.title}
              </span>
              <span className="font-mono text-[8px] text-text-muted">{shortHash(playbook.audit_hash)}</span>
            </div>
            <div className="flex flex-wrap gap-1">
              {playbook.watchlist_terms.slice(0, compact ? 2 : 4).map((term) => (
                <span key={term} className="rounded bg-bg-overlay px-1.5 py-0.5 text-[8px] font-semibold text-text-muted">
                  {term}
                </span>
              ))}
            </div>
          </div>
        ))}
      </div>

      {!compact && (
        <div className="mt-2 grid grid-cols-2 gap-2">
          <Guardrail icon={BrainCircuit} label="Qwen context tuned" />
          <Guardrail icon={ShieldCheck} label="Evidence remains authoritative" />
          <Guardrail icon={Activity} label="Bounded tuning" value={`${status?.bounded_queue.depth ?? 0}/${status?.bounded_queue.max_depth ?? 64}`} />
          <Guardrail icon={RefreshCw} label="Rollback ready" value={status?.rollback_available ? 'yes' : 'no'} />
        </div>
      )}
    </section>
  )
}

function MiniMetric({ label, value }: { label: string; value: string }) {
  return (
    <div className="min-w-0 rounded-md border border-border-subtle bg-bg-elevated/55 p-1.5">
      <div className="truncate text-[7px] font-bold uppercase tracking-[0.12em] text-text-muted">{label}</div>
      <div className="truncate font-mono text-[10px] font-semibold text-text-primary" title={value}>
        {value}
      </div>
    </div>
  )
}

function Guardrail({
  icon: Icon,
  label,
  value,
}: {
  icon: ComponentType<{ className?: string }>
  label: string
  value?: string
}) {
  return (
    <div className="flex min-w-0 items-center gap-1.5 rounded-md border border-border-subtle bg-bg-elevated/45 p-2">
      <Icon className="h-3.5 w-3.5 shrink-0 text-accent-primary" />
      <span className="truncate text-[8px] font-semibold uppercase tracking-[0.1em] text-text-muted">{label}</span>
      {value && <span className="ml-auto font-mono text-[8px] text-text-secondary">{value}</span>}
    </div>
  )
}

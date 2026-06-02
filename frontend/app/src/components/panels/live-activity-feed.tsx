// ============================================================================
// Live Activity Feed -- Real-time event pipeline transparency
// ============================================================================

import { useEffect, useMemo, useState } from 'react'
import { useActivityStore } from '@/stores/use-activity-store'
import { useUIStore } from '@/stores/use-ui-store'
import { PipelineStageBar } from '@/components/panels/pipeline-stage-bar'
import { SeverityBadge } from '@/components/shared/severity-badge'
import { verdictToSeverity } from '@/lib/severity'
import { fmtPaisa, truncId, cn } from '@/lib/utils'
import { FRAUD_PATTERN_LABELS } from '@/lib/types'
import { riskTierBadgeClass, UBI_CONSENSUS_SERIES } from '@/lib/union-bank-theme'
import { Activity, ArrowRight, Clock, Shield, Filter } from 'lucide-react'
import type { EventLifecycle } from '@/stores/use-activity-store'

type RiskFilter = 'all' | 'critical' | 'high' | 'medium' | 'low'
type VerdictFilter = 'all' | 'fraudulent' | 'suspicious' | 'legitimate'

function formatConsensusScore(score: number | null | undefined): string {
  return typeof score === 'number' && Number.isFinite(score) && score >= 0
    ? `${(score * 100).toFixed(0)}%`
    : 'N/A'
}

function isFallbackLifecycle(lifecycle: EventLifecycle): boolean {
  return (
    lifecycle.confidenceSource === 'deterministic_evidence_fallback' ||
    lifecycle.llmParseStatus?.includes('fallback') ||
    (lifecycle.confidence === 0.5 &&
      (lifecycle.evidenceCited?.length ?? 0) === 0 &&
      Boolean(lifecycle.reasoningSummary?.includes('Unable to reach definitive conclusion')))
  )
}

export function LiveActivityFeed() {
  const events = useActivityStore((s) => s.events)
  const orderedIds = useActivityStore((s) => s.orderedIds)
  const setSelectedEvent = useUIStore((s) => s.setSelectedEvent)
  const [riskFilter, setRiskFilter] = useState<RiskFilter>('all')
  const [verdictFilter, setVerdictFilter] = useState<VerdictFilter>('all')
  const [showFilters, setShowFilters] = useState(false)
  const [nowSec, setNowSec] = useState(() => Math.floor(Date.now() / 1000))

  useEffect(() => {
    const timer = window.setInterval(() => {
      setNowSec(Math.floor(Date.now() / 1000))
    }, 1000)
    return () => window.clearInterval(timer)
  }, [])

  const filteredIds = useMemo(() => {
    return orderedIds
      .filter((id) => {
        const lc = events.get(id)
        if (!lc) return false
        if (riskFilter !== 'all' && lc.riskTier !== riskFilter) return false
        if (verdictFilter !== 'all') {
          const v = lc.verdict?.toLowerCase()
          if (!v || !v.includes(verdictFilter)) return false
        }
        return true
      })
      .sort((a, b) => {
        const aEvent = events.get(a)
        const bEvent = events.get(b)
        const aHitl = aEvent?.analystStatus ? 1 : 0
        const bHitl = bEvent?.analystStatus ? 1 : 0
        if (aHitl !== bHitl) return bHitl - aHitl
        const aSeen = aEvent?.analystUpdatedAt ?? aEvent?.firstSeen ?? 0
        const bSeen = bEvent?.analystUpdatedAt ?? bEvent?.firstSeen ?? 0
        return bSeen - aSeen
      })
      .slice(0, 28)
  }, [orderedIds, events, riskFilter, verdictFilter])

  const RISK_OPTIONS: RiskFilter[] = ['all', 'critical', 'high', 'medium', 'low']
  const VERDICT_OPTIONS: VerdictFilter[] = ['all', 'fraudulent', 'suspicious', 'legitimate']

  return (
    <div className="flex flex-col h-full min-h-0">
      <div className="px-3 py-2 flex items-center gap-1.5 shrink-0">
        <Activity className="w-3 h-3 text-accent-primary" />
        <span className="text-[10px] font-semibold uppercase tracking-[0.12em] text-text-secondary">
          Live Activity
        </span>
        <button
          onClick={() => setShowFilters((p) => !p)}
          className={cn(
            'ml-1 p-0.5 rounded transition-colors',
            showFilters ? 'text-accent-primary bg-accent-primary/10' : 'text-text-muted hover:text-text-secondary',
          )}
          title="Toggle filters"
        >
          <Filter className="w-3 h-3" />
        </button>
        <span className="ml-auto text-[9px] font-mono tabular-nums text-text-muted">
          {filteredIds.length}/{orderedIds.length}
        </span>
      </div>

      {showFilters && (
        <div className="px-3 pb-2 space-y-1.5 border-b border-border-subtle shrink-0 animate-fade-in">
          <div className="flex items-center gap-1 flex-wrap">
            <span className="text-[8px] text-text-muted uppercase tracking-wider w-8 shrink-0">Risk</span>
            {RISK_OPTIONS.map((r) => (
              <button
                key={r}
                onClick={() => setRiskFilter(r)}
                className={cn(
                  'px-2 py-0.5 rounded text-[8px] font-semibold uppercase tracking-wider transition-all',
                  riskFilter === r
                    ? 'bg-accent-primary/20 text-accent-primary border border-accent-primary/30'
                    : 'text-text-muted hover:text-text-secondary border border-transparent',
                )}
              >
                {r}
              </button>
            ))}
          </div>
          <div className="flex items-center gap-1 flex-wrap">
            <span className="text-[8px] text-text-muted uppercase tracking-wider w-8 shrink-0">Verd</span>
            {VERDICT_OPTIONS.map((v) => (
              <button
                key={v}
                onClick={() => setVerdictFilter(v)}
                className={cn(
                  'px-2 py-0.5 rounded text-[8px] font-semibold uppercase tracking-wider transition-all',
                  verdictFilter === v
                    ? 'bg-accent-primary/20 text-accent-primary border border-accent-primary/30'
                    : 'text-text-muted hover:text-text-secondary border border-transparent',
                )}
              >
                {v}
              </button>
            ))}
          </div>
        </div>
      )}

      <div className="flex-1 overflow-y-auto min-h-0 px-2 pb-2 space-y-1.5">
        {filteredIds.length === 0 ? (
          <div className="flex flex-col items-center justify-center h-full gap-2 text-center px-4">
            <Shield className="w-6 h-6 text-text-muted/30" />
            <p className="text-[10px] text-text-muted">
              {orderedIds.length === 0
                ? 'No events tracked yet. Launch an event drill or inject a transaction to see pipeline activity.'
                : 'No events match the current filters.'}
            </p>
          </div>
        ) : (
          filteredIds.map((id) => {
            const lifecycle = events.get(id)
            if (!lifecycle) return null
            return (
              <EventCard
                key={id}
                lifecycle={lifecycle}
                nowSec={nowSec}
                onClick={() => setSelectedEvent(id)}
              />
            )
          })
        )}
      </div>
    </div>
  )
}

function EventCard({
  lifecycle,
  nowSec,
  onClick,
}: {
  lifecycle: EventLifecycle
  nowSec: number
  onClick: () => void
}) {
  const hasFirstSeen = Number.isFinite(lifecycle.firstSeen) && lifecycle.firstSeen > 0
  const elapsed = hasFirstSeen ? Math.max(0, Math.round(nowSec - lifecycle.firstSeen)) : null
  const hasVerdict = !!lifecycle.verdict
  const isFraud = (lifecycle.fraudLabel ?? 0) > 0
  const stageKeys = lifecycle.stages.map((s) => s.stage)

  return (
    <div
      onClick={onClick}
      className="rounded-md border border-border-subtle bg-bg-elevated/70 p-2 card-hover cursor-pointer transition-all duration-200 animate-fade-in"
    >
      {/* Header: sender → receiver | amount */}
      <div className="flex items-center gap-1 mb-1.5">
        <span className="text-[9px] font-mono text-text-primary truncate max-w-[70px]">
          {truncId(lifecycle.sender || lifecycle.txnId, 8)}
        </span>
        <ArrowRight className="w-2.5 h-2.5 text-text-muted shrink-0" />
        <span className="text-[9px] font-mono text-text-primary truncate max-w-[70px]">
          {lifecycle.receiver ? truncId(lifecycle.receiver, 8) : 'receiver pending'}
        </span>
        {lifecycle.amountPaisa > 0 && (
          <span className="ml-auto text-[9px] font-mono tabular-nums text-text-secondary">
            {fmtPaisa(lifecycle.amountPaisa)}
          </span>
        )}
      </div>

      {/* Pipeline stage bar */}
      <div className="mb-1.5">
        <PipelineStageBar completedStages={stageKeys} />
      </div>

      {/* Bottom row: risk/verdict + timing */}
      <div className="flex items-center gap-1.5 flex-wrap">
        {lifecycle.riskScore != null && (
          <span className={cn('rounded border px-1 py-0.5 font-mono text-[8px]', riskTierBadgeClass(lifecycle.riskTier))}>
            ML {(lifecycle.riskScore * 100).toFixed(0)}%
          </span>
        )}

        {hasVerdict && (
          <SeverityBadge severity={verdictToSeverity(lifecycle.verdict!)} />
        )}

        {lifecycle.analystStatus && (
          <span className={cn(
            'text-[8px] font-mono px-1 py-0.5 rounded border uppercase',
            lifecycle.analystStatus === 'approved'
              ? 'bg-alert-low/10 text-alert-low border-alert-low/30'
              : lifecycle.analystStatus === 'rejected'
                ? 'bg-text-muted/10 text-text-muted border-border-subtle'
                : lifecycle.analystStatus === 'escalated'
                  ? 'bg-alert-escalated/10 text-alert-escalated border-alert-escalated/30'
                  : 'bg-alert-medium/10 text-alert-medium border-alert-medium/30',
          )}>
            HITL {lifecycle.analystStatus.replace(/_/g, ' ')}
          </span>
        )}

        {isFraud && lifecycle.fraudLabel > 0 && (
          <span className="text-[8px] font-mono text-text-muted truncate max-w-[80px]">
            {FRAUD_PATTERN_LABELS[lifecycle.fraudLabel] ?? `Type ${lifecycle.fraudLabel}`}
          </span>
        )}

        {lifecycle.confidence != null && (
          <span className="text-[8px] font-mono tabular-nums text-text-muted">
            {isFallbackLifecycle(lifecycle) ? 'n/a' : `${(lifecycle.confidence * 100).toFixed(0)}%`}
          </span>
        )}

        <span className="ml-auto flex items-center gap-0.5 text-[8px] font-mono tabular-nums text-text-muted">
          <Clock className="w-2.5 h-2.5" />
          {elapsed == null ? 'n/a' : elapsed < 60 ? `${elapsed}s` : `${Math.floor(elapsed / 60)}m`}
        </span>
      </div>

      {/* Consensus breakdown (when CB evaluated) */}
      {lifecycle.consensusScores && (
        <div className="mt-1.5 pt-1.5 border-t border-border-subtle">
          <div className="flex gap-2 text-[8px] font-mono tabular-nums">
            {UBI_CONSENSUS_SERIES.map((item) => (
              <span key={item.key} className={lifecycle.consensusScores?.[item.key] == null ? 'text-text-muted' : item.text}>
                {item.label} {formatConsensusScore(lifecycle.consensusScores?.[item.key])}
              </span>
            ))}
          </div>
        </div>
      )}

      {/* Verdict reasoning summary */}
      {lifecycle.reasoningSummary && (
        <div className="mt-1 text-[8px] leading-relaxed text-text-muted line-clamp-2">
          {lifecycle.reasoningSummary}
        </div>
      )}

      {lifecycle.analystReason && (
        <div className="mt-1 text-[8px] leading-relaxed text-text-muted line-clamp-2">
          Analyst note: {lifecycle.analystReason}
        </div>
      )}
    </div>
  )
}

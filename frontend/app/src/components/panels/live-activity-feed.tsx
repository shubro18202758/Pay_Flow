// ============================================================================
// Live Activity Feed -- Real-time event pipeline transparency
// ============================================================================

import { useState, useMemo } from 'react'
import { useActivityStore } from '@/stores/use-activity-store'
import { useUIStore } from '@/stores/use-ui-store'
import { PipelineStageBar } from '@/components/panels/pipeline-stage-bar'
import { SeverityBadge, verdictToSeverity } from '@/components/shared/severity-badge'
import { fmtPaisa, truncId, cn } from '@/lib/utils'
import { FRAUD_PATTERN_LABELS } from '@/lib/types'
import { Activity, ArrowRight, Clock, Shield, Filter } from 'lucide-react'
import type { EventLifecycle } from '@/stores/use-activity-store'

type RiskFilter = 'all' | 'critical' | 'high' | 'medium' | 'low'
type VerdictFilter = 'all' | 'fraudulent' | 'suspicious' | 'legitimate'

export function LiveActivityFeed() {
  const events = useActivityStore((s) => s.events)
  const orderedIds = useActivityStore((s) => s.orderedIds)
  const setSelectedEvent = useUIStore((s) => s.setSelectedEvent)
  const [riskFilter, setRiskFilter] = useState<RiskFilter>('all')
  const [verdictFilter, setVerdictFilter] = useState<VerdictFilter>('all')
  const [showFilters, setShowFilters] = useState(false)

  const filteredIds = useMemo(() => {
    return orderedIds.filter((id) => {
      const lc = events.get(id)
      if (!lc) return false
      if (riskFilter !== 'all' && lc.riskTier !== riskFilter) return false
      if (verdictFilter !== 'all') {
        const v = lc.verdict?.toLowerCase()
        if (!v || !v.includes(verdictFilter)) return false
      }
      return true
    }).slice(0, 50)
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
                ? 'No events tracked yet. Launch a simulation to see pipeline activity.'
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
  onClick,
}: {
  lifecycle: EventLifecycle
  onClick: () => void
}) {
  const elapsed = Math.round(Date.now() / 1000 - lifecycle.firstSeen)
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
          {truncId(lifecycle.receiver || '???', 8)}
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
          <span className={`text-[8px] font-mono px-1 py-0.5 rounded border ${
            lifecycle.riskTier === 'critical' ? 'bg-red-500/15 text-red-400 border-red-500/30'
            : lifecycle.riskTier === 'high' ? 'bg-orange-500/15 text-orange-400 border-orange-500/30'
            : 'bg-yellow-500/15 text-yellow-400 border-yellow-500/30'
          }`}>
            ML {(lifecycle.riskScore * 100).toFixed(0)}%
          </span>
        )}

        {hasVerdict && (
          <SeverityBadge severity={verdictToSeverity(lifecycle.verdict!)} />
        )}

        {isFraud && lifecycle.fraudLabel > 0 && (
          <span className="text-[8px] font-mono text-text-muted truncate max-w-[80px]">
            {FRAUD_PATTERN_LABELS[lifecycle.fraudLabel] ?? `Type ${lifecycle.fraudLabel}`}
          </span>
        )}

        {lifecycle.confidence != null && (
          <span className="text-[8px] font-mono tabular-nums text-text-muted">
            {(lifecycle.confidence * 100).toFixed(0)}%
          </span>
        )}

        <span className="ml-auto flex items-center gap-0.5 text-[8px] font-mono tabular-nums text-text-muted">
          <Clock className="w-2.5 h-2.5" />
          {elapsed < 60 ? `${elapsed}s` : `${Math.floor(elapsed / 60)}m`}
        </span>
      </div>

      {/* Consensus breakdown (when CB evaluated) */}
      {lifecycle.consensusScores && (
        <div className="mt-1.5 pt-1.5 border-t border-border-subtle">
          <div className="flex gap-2 text-[8px] font-mono tabular-nums">
            <span className="text-blue-400">ML {(lifecycle.consensusScores.ml * 100).toFixed(0)}%</span>
            <span className="text-purple-400">GNN {(lifecycle.consensusScores.gnn * 100).toFixed(0)}%</span>
            <span className="text-green-400">Graph {(lifecycle.consensusScores.graph * 100).toFixed(0)}%</span>
          </div>
        </div>
      )}

      {/* Verdict reasoning summary */}
      {lifecycle.reasoningSummary && (
        <div className="mt-1 text-[8px] leading-relaxed text-text-muted line-clamp-2">
          {lifecycle.reasoningSummary}
        </div>
      )}
    </div>
  )
}

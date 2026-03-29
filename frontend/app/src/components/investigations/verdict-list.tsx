// ============================================================================
// Verdict List -- Expandable agent verdicts with full evidence + cross-nav
// ============================================================================

import { useState } from 'react'
import { Scale, Inbox, ChevronDown, ChevronUp, Eye, Network, Clock, Wrench, Brain, FileText } from 'lucide-react'
import { useDashboardStore } from '@/stores/use-dashboard-store'
import { useUIStore } from '@/stores/use-ui-store'
import { SeverityBadge, verdictToSeverity } from '@/components/shared/severity-badge'
import { truncId, fmtTimestamp } from '@/lib/utils'
import { cn } from '@/lib/utils'
import type { SSEAgentVerdict } from '@/lib/types'

const FILTERS = ['all', 'fraudulent', 'suspicious', 'legitimate', 'escalated'] as const
type Filter = (typeof FILTERS)[number]

function VerdictCard({ entry }: { entry: { id: string; timestamp: number; data: SSEAgentVerdict } }) {
  const [expanded, setExpanded] = useState(false)
  const setActiveTab = useUIStore((s) => s.setActiveTab)
  const setSelectedNode = useUIStore((s) => s.setSelectedNode)
  const v = entry.data

  const severity = verdictToSeverity(v.verdict)

  const handleViewInGraph = () => {
    if (v.node_id) {
      setSelectedNode(v.node_id)
      setActiveTab('overview')
    }
  }

  return (
    <div className="card-hover bg-bg-elevated rounded-lg border border-border-subtle animate-fade-in">
      {/* Collapsed view */}
      <button
        onClick={() => setExpanded(!expanded)}
        className="w-full text-left p-3"
      >
        <div className="flex items-center gap-2 mb-1.5">
          <SeverityBadge severity={severity} label={v.verdict.toUpperCase()} />
          <span className="text-[10px] font-mono font-bold text-text-primary tracking-wide">
            {truncId(v.txn_id, 12)}
          </span>
          {v.node_id && (
            <span className="text-[8px] font-mono text-text-muted">
              {truncId(v.node_id, 10)}
            </span>
          )}
          <span className="text-[9px] font-mono text-text-muted ml-auto tabular-nums">
            {(v.confidence * 100).toFixed(1)}%
          </span>
          {expanded ? <ChevronUp className="w-3 h-3 text-text-muted" /> : <ChevronDown className="w-3 h-3 text-text-muted" />}
        </div>

        {v.fraud_typology && (
          <div className="text-[9px] text-text-secondary mb-1 font-mono">
            <span className="text-text-muted/60 uppercase tracking-wider mr-1">Typology:</span>
            {v.fraud_typology}
          </div>
        )}

        {!expanded && v.reasoning_summary && (
          <div className="text-[9px] text-text-muted leading-relaxed line-clamp-2">
            {v.reasoning_summary}
          </div>
        )}

        <div className="flex items-center gap-3 mt-2 pt-1.5 border-t border-border-subtle/30 text-[8px] text-text-muted/60 font-mono tabular-nums">
          <span>{v.recommended_action}</span>
          <span>{v.thinking_steps} steps</span>
          <span>{v.total_duration_ms?.toFixed(0)}ms</span>
          {(v as unknown as { nlu_findings_count?: number }).nlu_findings_count ? (
            <span className="text-amber-400">NLU:{(v as unknown as { nlu_findings_count: number }).nlu_findings_count}</span>
          ) : null}
          <span className="ml-auto">{fmtTimestamp(entry.timestamp)}</span>
        </div>
      </button>

      {/* Expanded detail */}
      {expanded && (
        <div className="px-3 pb-3 pt-0 border-t border-border-subtle space-y-3 animate-fade-in">
          {/* Full reasoning */}
          {v.reasoning_summary && (
            <div>
              <div className="flex items-center gap-1.5 mb-1">
                <Brain className="w-3 h-3 text-accent-primary" />
                <span className="text-[8px] font-semibold uppercase tracking-wider text-text-muted">Reasoning</span>
              </div>
              <p className="text-[9px] text-text-secondary leading-relaxed bg-bg-overlay/40 rounded p-2 border border-border-subtle">
                {v.reasoning_summary}
              </p>
            </div>
          )}

          {/* Evidence cited */}
          {v.evidence_cited && v.evidence_cited.length > 0 && (
            <div>
              <div className="flex items-center gap-1.5 mb-1">
                <FileText className="w-3 h-3 text-accent-primary" />
                <span className="text-[8px] font-semibold uppercase tracking-wider text-text-muted">
                  Evidence ({v.evidence_cited.length})
                </span>
              </div>
              <div className="flex flex-wrap gap-1">
                {v.evidence_cited.map((e, i) => (
                  <span key={i} className="text-[8px] font-mono px-1.5 py-0.5 rounded bg-bg-overlay border border-border-subtle text-text-secondary">
                    {e}
                  </span>
                ))}
              </div>
            </div>
          )}

          {/* Tools used */}
          {v.tools_used && v.tools_used.length > 0 && (
            <div>
              <div className="flex items-center gap-1.5 mb-1">
                <Wrench className="w-3 h-3 text-accent-primary" />
                <span className="text-[8px] font-semibold uppercase tracking-wider text-text-muted">
                  Tools ({v.tools_used.length})
                </span>
              </div>
              <div className="flex flex-wrap gap-1">
                {v.tools_used.map((t, i) => (
                  <span key={i} className="text-[8px] font-mono px-1.5 py-0.5 rounded bg-accent-primary/10 border border-accent-primary/20 text-accent-primary">
                    {t}
                  </span>
                ))}
              </div>
            </div>
          )}

          {/* Duration breakdown */}
          <div className="flex items-center gap-3 text-[8px] font-mono text-text-muted">
            <Clock className="w-3 h-3" />
            <span>Total: {v.total_duration_ms?.toFixed(0)}ms</span>
            <span>Steps: {v.thinking_steps}</span>
          </div>

          {/* Cross-navigation buttons */}
          <div className="flex items-center gap-2 pt-2 border-t border-border-subtle">
            {v.node_id && (
              <button
                onClick={handleViewInGraph}
                className="flex items-center gap-1 px-2 py-1 rounded text-[8px] font-semibold uppercase tracking-wider text-accent-primary bg-accent-primary/10 border border-accent-primary/20 hover:bg-accent-primary/20 transition-colors"
              >
                <Network className="w-3 h-3" />
                View in Graph
              </button>
            )}
            <button
              onClick={() => setExpanded(false)}
              className="flex items-center gap-1 px-2 py-1 rounded text-[8px] font-semibold uppercase tracking-wider text-text-muted hover:text-text-secondary transition-colors"
            >
              <Eye className="w-3 h-3" />
              Collapse
            </button>
          </div>
        </div>
      )}
    </div>
  )
}

export function VerdictList() {
  const agentLog = useDashboardStore((s) => s.agentLog)
  const [filter, setFilter] = useState<Filter>('all')

  const verdicts = agentLog
    .filter((e) => e.type === 'verdict')
    .reverse()

  const filtered = filter === 'all'
    ? verdicts
    : verdicts.filter((e) => {
        const v = (e.data as SSEAgentVerdict).verdict.toLowerCase()
        return v === filter || (filter === 'escalated' && v.includes('escalat'))
      })

  return (
    <div className="flex flex-col h-full">
      {/* Header */}
      <div className="flex items-center gap-2.5 px-4 py-2.5 border-b border-border-subtle shrink-0 bg-bg-surface">
        <Scale className="w-3.5 h-3.5 text-accent-primary" />
        <span className="text-[10px] font-semibold uppercase tracking-[0.12em] text-text-secondary">
          Verdicts
        </span>

        {/* Filter pills */}
        <div className="flex items-center gap-1 ml-2">
          {FILTERS.map((f) => (
            <button
              key={f}
              onClick={() => setFilter(f)}
              className={cn(
                'px-2.5 py-0.5 rounded-md text-[9px] font-semibold uppercase tracking-wider transition-all',
                filter === f
                  ? 'bg-accent-primary/20 text-accent-primary border border-accent-primary/30 shadow-sm shadow-accent-primary/10'
                  : 'text-text-muted hover:text-text-secondary hover:bg-white/[0.04] border border-transparent',
              )}
            >
              {f}
            </button>
          ))}
        </div>

        <span className="ml-auto text-[9px] font-mono text-text-muted tabular-nums">
          {filtered.length} results
        </span>
      </div>

      {/* List */}
      <div className="flex-1 overflow-y-auto">
        {filtered.length === 0 ? (
          <div className="flex items-center justify-center h-full animate-fade-in">
            <div className="text-center space-y-3">
              <Inbox className="w-8 h-8 text-text-muted/30 mx-auto" />
              <p className="text-text-muted text-[10px] uppercase tracking-[0.12em]">
                No verdicts to display
              </p>
            </div>
          </div>
        ) : (
          <div className="p-3 space-y-2">
            {filtered.map((entry) => (
              <VerdictCard
                key={entry.id}
                entry={entry as { id: string; timestamp: number; data: SSEAgentVerdict }}
              />
            ))}
          </div>
        )}
      </div>
    </div>
  )
}

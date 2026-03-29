// ============================================================================
// Event Detail Drawer -- Expanded event inspection with pipeline timeline
// ============================================================================

import { useActivityStore } from '@/stores/use-activity-store'
import { useUIStore } from '@/stores/use-ui-store'
import { useDashboardStore } from '@/stores/use-dashboard-store'
import { PipelineStageBar } from '@/components/panels/pipeline-stage-bar'
import { SeverityBadge, verdictToSeverity } from '@/components/shared/severity-badge'
import { fmtPaisa, fmtTimestamp, truncId, cn } from '@/lib/utils'
import { FRAUD_PATTERN_LABELS } from '@/lib/types'
import { X, ArrowRight, Clock, Fingerprint, BarChart3, Brain, Wrench, Gavel, Network } from 'lucide-react'
import type { EventLifecycle, StageDetail } from '@/stores/use-activity-store'
import type { SSEAgentThinking, SSEAgentToolCall, SSEAgentVerdict } from '@/lib/types'

const STAGE_LABELS: Record<string, string> = {
  ingested: 'Ingested',
  ml_scored: 'ML Scored',
  graph_investigated: 'Graph Investigated',
  cb_evaluated: 'Circuit Breaker',
  llm_started: 'LLM Analysis',
  verdict: 'Verdict Emitted',
}

export function EventDetailDrawer() {
  const selectedEventId = useUIStore((s) => s.selectedEventId)
  const setSelectedEvent = useUIStore((s) => s.setSelectedEvent)
  const events = useActivityStore((s) => s.events)

  if (!selectedEventId) {
    return (
      <div className="flex items-center justify-center h-full text-text-muted text-[10px]">
        Select an event from the activity feed to inspect
      </div>
    )
  }

  const lifecycle = events.get(selectedEventId)
  if (!lifecycle) {
    return (
      <div className="flex items-center justify-center h-full text-text-muted text-[10px]">
        Event {truncId(selectedEventId, 12)} not found in activity store
      </div>
    )
  }

  return (
    <div className="flex h-full overflow-hidden">
      {/* Left pane: Transaction details + Pipeline timeline */}
      <div className="flex-1 border-r border-border-subtle overflow-y-auto p-3 space-y-3">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Fingerprint className="w-3.5 h-3.5 text-accent-primary" />
            <span className="text-[10px] font-semibold uppercase tracking-[0.12em] text-text-secondary">
              Event Inspector
            </span>
          </div>
          <button
            onClick={() => setSelectedEvent(null)}
            className="w-5 h-5 flex items-center justify-center rounded hover:bg-bg-overlay text-text-muted hover:text-text-primary transition-colors"
          >
            <X className="w-3 h-3" />
          </button>
        </div>

        {/* Transaction header */}
        <div className="rounded-md border border-border-subtle bg-bg-surface p-2.5">
          <div className="flex items-center gap-1.5 mb-2">
            <span className="text-[10px] font-mono font-semibold text-text-primary">
              {truncId(lifecycle.txnId, 16)}
            </span>
            {lifecycle.fraudLabel > 0 && (
              <span className="text-[8px] font-mono px-1.5 py-0.5 rounded bg-red-500/15 text-red-400 border border-red-500/30">
                {FRAUD_PATTERN_LABELS[lifecycle.fraudLabel] ?? `Type ${lifecycle.fraudLabel}`}
              </span>
            )}
          </div>
          <div className="grid grid-cols-2 gap-y-1.5 text-[9px]">
            <div>
              <span className="text-text-muted">Sender</span>
              <div className="font-mono text-text-secondary">{truncId(lifecycle.sender || 'N/A', 14)}</div>
            </div>
            <div>
              <span className="text-text-muted">Receiver</span>
              <div className="font-mono text-text-secondary">{truncId(lifecycle.receiver || 'N/A', 14)}</div>
            </div>
            <div>
              <span className="text-text-muted">Amount</span>
              <div className="font-mono text-text-secondary tabular-nums">
                {lifecycle.amountPaisa > 0 ? fmtPaisa(lifecycle.amountPaisa) : 'N/A'}
              </div>
            </div>
            <div>
              <span className="text-text-muted">First Seen</span>
              <div className="font-mono text-text-secondary tabular-nums">
                {fmtTimestamp(lifecycle.firstSeen)}
              </div>
            </div>
          </div>
        </div>

        {/* Pipeline progress */}
        <div>
          <div className="text-[9px] font-semibold uppercase tracking-[0.12em] text-text-muted mb-2">
            Pipeline Progress
          </div>
          <PipelineStageBar completedStages={lifecycle.stages.map((s) => s.stage)} stageEntries={lifecycle.stages} expanded />
        </div>

        {/* Pipeline timeline (waterfall) */}
        <div>
          <div className="text-[9px] font-semibold uppercase tracking-[0.12em] text-text-muted mb-2">
            Stage Timeline
          </div>
          <div className="space-y-1">
            {lifecycle.stages.map((entry, i) => (
              <StageTimelineEntry
                key={entry.stage}
                entry={entry}
                prevTimestamp={i > 0 ? lifecycle.stages[i - 1].timestamp : lifecycle.firstSeen}
                isLast={i === lifecycle.stages.length - 1}
              />
            ))}
          </div>
        </div>
      </div>

      {/* Right pane: Consensus breakdown + Verdict detail */}
      <div className="flex-1 overflow-y-auto p-3 space-y-3">
        {/* Consensus breakdown */}
        {lifecycle.consensusScores && (
          <div>
            <div className="flex items-center gap-1.5 mb-2">
              <BarChart3 className="w-3 h-3 text-accent-primary" />
              <span className="text-[9px] font-semibold uppercase tracking-[0.12em] text-text-muted">
                Consensus Breakdown
              </span>
            </div>
            <ConsensusBar scores={lifecycle.consensusScores} />
          </div>
        )}

        {/* Risk score */}
        {lifecycle.riskScore != null && (
          <div className="rounded-md border border-border-subtle bg-bg-surface p-2.5">
            <div className="text-[9px] text-text-muted mb-1">ML Risk Score</div>
            <div className="flex items-center gap-2">
              <div className="flex-1 h-1.5 bg-bg-overlay rounded-full overflow-hidden">
                <div
                  className={`h-full rounded-full transition-all duration-500 ${
                    lifecycle.riskScore > 0.8 ? 'bg-red-500' : lifecycle.riskScore > 0.5 ? 'bg-orange-500' : 'bg-green-500'
                  }`}
                  style={{ width: `${lifecycle.riskScore * 100}%` }}
                />
              </div>
              <span className="text-[10px] font-mono tabular-nums text-text-primary">
                {(lifecycle.riskScore * 100).toFixed(1)}%
              </span>
            </div>
            {lifecycle.riskTier && (
              <span className="text-[8px] font-mono uppercase text-text-muted mt-1 block">
                Tier: {lifecycle.riskTier}
              </span>
            )}
          </div>
        )}

        {/* ML Top Features */}
        {lifecycle.topFeatures && lifecycle.topFeatures.length > 0 && (
          <TopFeaturesChart features={lifecycle.topFeatures} />
        )}

        {/* Investigation Timeline */}
        <InvestigationTimeline txnId={lifecycle.txnId} />

        {/* Verdict detail */}
        {lifecycle.verdict && (
          <div className="rounded-md border border-border-subtle bg-bg-surface p-2.5 space-y-2">
            <div className="flex items-center gap-2">
              <SeverityBadge severity={verdictToSeverity(lifecycle.verdict)} />
              {lifecycle.confidence != null && (
                <span className="text-[9px] font-mono tabular-nums text-text-secondary">
                  {(lifecycle.confidence * 100).toFixed(0)}% confidence
                </span>
              )}
            </div>

            {lifecycle.fraudTypology && (
              <div className="text-[9px]">
                <span className="text-text-muted">Typology: </span>
                <span className="font-mono text-text-secondary">{lifecycle.fraudTypology}</span>
              </div>
            )}

            {lifecycle.recommendedAction && (
              <div className="text-[9px]">
                <span className="text-text-muted">Action: </span>
                <span className="font-mono text-text-secondary">{lifecycle.recommendedAction}</span>
              </div>
            )}

            {lifecycle.reasoningSummary && (
              <div className="text-[9px] leading-relaxed text-text-secondary border-t border-border-subtle pt-2">
                {lifecycle.reasoningSummary}
              </div>
            )}

            {lifecycle.sender && (
              <button
                onClick={() => {
                  useUIStore.getState().setSelectedNode(lifecycle.sender!)
                  useUIStore.getState().setActiveTab('overview')
                }}
                className="flex items-center gap-1 px-2 py-1 rounded text-[8px] font-semibold uppercase tracking-wider text-accent-primary bg-accent-primary/10 border border-accent-primary/20 hover:bg-accent-primary/20 transition-colors mt-1"
              >
                <Network className="w-3 h-3" />
                Open in Graph
              </button>
            )}
          </div>
        )}

        {/* Empty state for right pane */}
        {!lifecycle.verdict && !lifecycle.consensusScores && lifecycle.riskScore == null && (
          <div className="flex flex-col items-center justify-center h-full gap-2 text-center px-4">
            <Clock className="w-6 h-6 text-text-muted/30" />
            <p className="text-[10px] text-text-muted">
              Awaiting further pipeline analysis for this event.
            </p>
          </div>
        )}
      </div>
    </div>
  )
}

function StageTimelineEntry({
  entry,
  prevTimestamp,
  isLast,
}: {
  entry: StageDetail
  prevTimestamp: number
  isLast: boolean
}) {
  const delta = Math.max(0, entry.timestamp - prevTimestamp)
  const deltaStr = delta < 1 ? `${(delta * 1000).toFixed(0)}ms` : `${delta.toFixed(1)}s`

  return (
    <div className="flex items-start gap-2">
      {/* Timeline dot + line */}
      <div className="flex flex-col items-center shrink-0">
        <div className="w-2 h-2 rounded-full bg-accent-primary mt-0.5" />
        {!isLast && <div className="w-px h-4 bg-border-subtle" />}
      </div>

      {/* Content */}
      <div className="flex items-center gap-2 min-w-0 pb-1">
        <span className="text-[9px] font-mono font-semibold text-text-primary">
          {STAGE_LABELS[entry.stage] ?? entry.stage}
        </span>
        <span className="text-[8px] font-mono tabular-nums text-text-muted">
          +{deltaStr}
        </span>
        <span className="text-[8px] font-mono tabular-nums text-text-muted ml-auto">
          {fmtTimestamp(entry.timestamp)}
        </span>
      </div>
    </div>
  )
}

function ConsensusBar({ scores }: { scores: { ml: number; gnn: number; graph: number } }) {
  const total = scores.ml + scores.gnn + scores.graph
  const mlPct = total > 0 ? (scores.ml / total) * 100 : 35
  const gnnPct = total > 0 ? (scores.gnn / total) * 100 : 35
  const graphPct = total > 0 ? (scores.graph / total) * 100 : 30

  return (
    <div className="space-y-1.5">
      {/* Stacked bar */}
      <div className="h-3 rounded-full overflow-hidden flex bg-bg-overlay">
        <div className="bg-blue-500 transition-all duration-500" style={{ width: `${mlPct}%` }} />
        <div className="bg-purple-500 transition-all duration-500" style={{ width: `${gnnPct}%` }} />
        <div className="bg-green-500 transition-all duration-500" style={{ width: `${graphPct}%` }} />
      </div>
      {/* Labels */}
      <div className="flex justify-between text-[8px] font-mono tabular-nums">
        <span className="text-blue-400">ML {(scores.ml * 100).toFixed(0)}%</span>
        <span className="text-purple-400">GNN {(scores.gnn * 100).toFixed(0)}%</span>
        <span className="text-green-400">Graph {(scores.graph * 100).toFixed(0)}%</span>
      </div>
    </div>
  )
}

function TopFeaturesChart({ features }: { features: string[] }) {
  return (
    <div className="rounded-md border border-border-subtle bg-bg-surface p-2.5">
      <div className="flex items-center gap-1.5 mb-2">
        <BarChart3 className="w-3 h-3 text-accent-primary" />
        <span className="text-[9px] font-semibold uppercase tracking-[0.12em] text-text-muted">
          Top ML Features ({features.length})
        </span>
      </div>
      <div className="space-y-1">
        {features.slice(0, 8).map((feat, i) => (
          <div key={i} className="flex items-center gap-2">
            <span className="text-[8px] font-mono text-text-muted w-3 shrink-0 tabular-nums text-right">
              {i + 1}
            </span>
            <div className="flex-1 h-1.5 bg-bg-overlay rounded-full overflow-hidden">
              <div
                className="h-full bg-accent-primary/60 rounded-full"
                style={{ width: `${100 - i * 12}%` }}
              />
            </div>
            <span className="text-[8px] font-mono text-text-secondary truncate max-w-[120px]">
              {feat}
            </span>
          </div>
        ))}
      </div>
    </div>
  )
}

function InvestigationTimeline({ txnId }: { txnId: string }) {
  const agentLog = useDashboardStore((s) => s.agentLog)

  const entries = agentLog.filter((e) => {
    const d = e.data as { txn_id?: string }
    return d.txn_id === txnId
  })

  if (entries.length === 0) return null

  return (
    <div className="rounded-md border border-border-subtle bg-bg-surface p-2.5">
      <div className="flex items-center gap-1.5 mb-2">
        <Brain className="w-3 h-3 text-accent-primary" />
        <span className="text-[9px] font-semibold uppercase tracking-[0.12em] text-text-muted">
          Investigation Timeline ({entries.length} events)
        </span>
      </div>
      <div className="space-y-0.5 max-h-[200px] overflow-y-auto">
        {entries.map((entry) => {
          if (entry.type === 'thinking') {
            const d = entry.data as SSEAgentThinking
            return (
              <div key={entry.id} className="flex items-start gap-1.5 py-0.5">
                <Brain className="w-2.5 h-2.5 text-cyan-400 shrink-0 mt-0.5" />
                <span className="text-[8px] text-text-secondary leading-relaxed line-clamp-2">{d.content}</span>
              </div>
            )
          }
          if (entry.type === 'tool_call') {
            const d = entry.data as SSEAgentToolCall
            return (
              <div key={entry.id} className="flex items-center gap-1.5 py-0.5">
                <Wrench className="w-2.5 h-2.5 text-violet-400 shrink-0" />
                <span className="text-[8px] font-mono text-text-secondary">{d.tool_name}</span>
                <span className={cn('text-[8px] font-bold', d.success ? 'text-emerald-400' : 'text-red-400')}>
                  {d.success ? 'OK' : 'FAIL'}
                </span>
                <span className="text-[8px] text-text-muted/50 font-mono">{d.duration_ms}ms</span>
              </div>
            )
          }
          const d = entry.data as SSEAgentVerdict
          return (
            <div key={entry.id} className="flex items-center gap-1.5 py-0.5 mt-1 pt-1 border-t border-border-subtle/50">
              <Gavel className="w-2.5 h-2.5 text-accent-primary shrink-0" />
              <span className={cn(
                'text-[8px] font-bold',
                d.verdict === 'FRAUDULENT' ? 'text-red-400' : d.verdict === 'SUSPICIOUS' ? 'text-amber-400' : 'text-emerald-400',
              )}>
                {d.verdict}
              </span>
              <span className="text-[8px] text-text-muted">conf:{(d.confidence * 100).toFixed(0)}%</span>
              <span className="text-[8px] text-text-muted font-mono">{d.total_duration_ms}ms</span>
            </div>
          )
        })}
      </div>
    </div>
  )
}

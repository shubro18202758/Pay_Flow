// ============================================================================
// Forensic Evidence Chain — Visual investigation timeline showing reasoning flow
// ============================================================================

import { useDashboardStore } from '@/stores/use-dashboard-store'
import { SeverityBadge, verdictToSeverity } from '@/components/shared/severity-badge'
import { cn, fmtTimestamp, truncId, fmtDuration } from '@/lib/utils'
import { Search, Brain, Wrench, Scale, ChevronDown, Link2 } from 'lucide-react'
import { useState, useMemo } from 'react'
import type { AgentLogEntry, SSEAgentThinking, SSEAgentToolCall, SSEAgentVerdict } from '@/lib/types'

interface InvestigationChain {
  txnId: string
  entries: AgentLogEntry[]
  verdict: AgentLogEntry | undefined
  startTime: number
  endTime: number
}

const STEP_CONFIG = {
  thinking: { icon: Brain, color: 'text-blue-400', dotBg: 'bg-blue-500', label: 'Reasoning' },
  tool_call: { icon: Wrench, color: 'text-amber-400', dotBg: 'bg-amber-500', label: 'Tool Call' },
  verdict:   { icon: Scale, color: 'text-emerald-400', dotBg: 'bg-emerald-500', label: 'Verdict' },
} as const

export function ForensicEvidenceChain() {
  const agentLog = useDashboardStore((s) => s.agentLog)
  const [expandedTxn, setExpandedTxn] = useState<string | null>(null)

  // Group entries by txn_id into investigation chains
  const chains = useMemo(() => {
    const map = new Map<string, AgentLogEntry[]>()
    for (const entry of agentLog) {
      const arr = map.get(entry.txn_id) ?? []
      arr.push(entry)
      map.set(entry.txn_id, arr)
    }

    const result: InvestigationChain[] = []
    for (const [txnId, entries] of map) {
      entries.sort((a, b) => a.timestamp - b.timestamp)
      result.push({
        txnId,
        entries,
        verdict: entries.find((e) => e.type === 'verdict'),
        startTime: entries[0].timestamp,
        endTime: entries[entries.length - 1].timestamp,
      })
    }
    return result.sort((a, b) => b.endTime - a.endTime).slice(0, 10) // latest 10
  }, [agentLog])

  return (
    <div className="rounded-lg border border-border-subtle bg-bg-card p-3">
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-1.5 text-[10px] font-semibold uppercase tracking-[0.12em] text-text-secondary">
          <Search className="w-3.5 h-3.5 text-accent-primary/70" />
          <span>Forensic Evidence Chains</span>
        </div>
        <span className="text-[8px] font-mono text-text-muted">
          {chains.length} investigations
        </span>
      </div>

      {chains.length === 0 ? (
        <div className="flex items-center justify-center h-16 text-[10px] text-text-muted">
          No investigations recorded yet. Waiting for LLM agent…
        </div>
      ) : (
        <div className="space-y-1.5 max-h-80 overflow-y-auto scrollbar-thin">
          {chains.map((chain) => {
            const isExpanded = expandedTxn === chain.txnId
            const verdictData = chain.verdict?.data as SSEAgentVerdict | undefined
            const stepCounts = {
              thinking: chain.entries.filter((e) => e.type === 'thinking').length,
              tool_call: chain.entries.filter((e) => e.type === 'tool_call').length,
            }
            const duration = chain.endTime - chain.startTime

            return (
              <div
                key={chain.txnId}
                className={cn(
                  'rounded-md border transition-all duration-200',
                  isExpanded ? 'border-accent-primary/30 bg-bg-elevated' : 'border-border-subtle/50 bg-bg-elevated/50 hover:bg-bg-elevated/80',
                )}
              >
                {/* Chain Header */}
                <button
                  className="w-full flex items-center gap-2 p-2 text-left"
                  onClick={() => setExpandedTxn(isExpanded ? null : chain.txnId)}
                >
                  <Link2 className="w-3 h-3 text-text-muted shrink-0" />
                  <span className="text-[10px] font-mono text-text-primary">
                    {truncId(chain.txnId, 12)}
                  </span>
                  <div className="flex items-center gap-1 ml-1">
                    <span className="text-[8px] text-blue-400/70">{stepCounts.thinking}T</span>
                    <span className="text-[8px] text-amber-400/70">{stepCounts.tool_call}C</span>
                  </div>
                  {verdictData && (
                    <SeverityBadge severity={verdictToSeverity(verdictData.verdict)} />
                  )}
                  <span className="ml-auto text-[8px] font-mono tabular-nums text-text-muted">
                    {fmtDuration(duration)}
                  </span>
                  <ChevronDown className={cn(
                    'w-3 h-3 text-text-muted transition-transform duration-200',
                    isExpanded && 'rotate-180',
                  )} />
                </button>

                {/* Expanded Timeline */}
                {isExpanded && (
                  <div className="px-2 pb-2.5 pt-0.5">
                    <div className="relative ml-3 border-l border-border-subtle/50 pl-3 space-y-2">
                      {chain.entries.map((entry, i) => {
                        const cfg = STEP_CONFIG[entry.type]
                        const StepIcon = cfg.icon
                        return (
                          <div key={entry.id} className="relative">
                            {/* Timeline dot */}
                            <div className={cn(
                              'absolute -left-[17px] top-1 w-2 h-2 rounded-full ring-2 ring-bg-card',
                              cfg.dotBg,
                            )} />
                            <div className="text-[9px]">
                              <div className="flex items-center gap-1.5 mb-0.5">
                                <StepIcon className={cn('w-2.5 h-2.5', cfg.color)} />
                                <span className={cn('font-medium', cfg.color)}>{cfg.label}</span>
                                <span className="ml-auto font-mono tabular-nums text-text-muted text-[8px]">
                                  {fmtTimestamp(entry.timestamp)}
                                </span>
                              </div>
                              <div className="text-text-secondary leading-relaxed">
                                {renderStepContent(entry)}
                              </div>
                            </div>
                          </div>
                        )
                      })}
                    </div>
                  </div>
                )}
              </div>
            )
          })}
        </div>
      )}
    </div>
  )
}

function renderStepContent(entry: AgentLogEntry): string {
  switch (entry.type) {
    case 'thinking': {
      const d = entry.data as SSEAgentThinking
      const text = d.content ?? ''
      return text.length > 120 ? text.slice(0, 120) + '…' : text
    }
    case 'tool_call': {
      const d = entry.data as SSEAgentToolCall
      return `${d.tool_name ?? 'unknown'}(${d.output_summary ? truncId(d.output_summary, 60) : '…'})`
    }
    case 'verdict': {
      const d = entry.data as SSEAgentVerdict
      return `${d.verdict} | ${d.reasoning_summary ?? 'No summary'}`
    }
  }
}

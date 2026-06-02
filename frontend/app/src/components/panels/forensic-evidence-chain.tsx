// ============================================================================
// Forensic Evidence Chain — Visual investigation timeline showing evidence-rationale flow
// ============================================================================

import { useDashboardStore } from '@/stores/use-dashboard-store'
import { SeverityBadge } from '@/components/shared/severity-badge'
import { verdictToSeverity } from '@/lib/severity'
import { cn, fmtOptionalMs, fmtOptionalTimestamp, truncId, fmtDuration } from '@/lib/utils'
import { Search, Brain, Wrench, Scale, ChevronDown, Link2, Database, Loader2 } from 'lucide-react'
import { useState, useMemo } from 'react'
import { useInvestigation } from '@/hooks/use-api'
import type { AgentLogEntry, SSEAgentThinking, SSEAgentToolCall, SSEAgentVerdict } from '@/lib/types'

interface InvestigationChain {
  txnId: string
  entries: AgentLogEntry[]
  verdict: AgentLogEntry | undefined
  startTime: number | null
  endTime: number | null
}

const STEP_CONFIG = {
  thinking: { icon: Brain, color: 'text-accent-primary', dotBg: 'bg-accent-primary', label: 'Evidence' },
  tool_call: { icon: Wrench, color: 'text-[#DA251C]', dotBg: 'bg-[#DA251C]', label: 'Tool Call' },
  verdict:   { icon: Scale, color: 'text-[#00579C]', dotBg: 'bg-[#00579C]', label: 'Verdict' },
} as const

export function ForensicEvidenceChain() {
  const agentLog = useDashboardStore((s) => s.agentLog)
  const [expandedTxn, setExpandedTxn] = useState<string | null>(null)
  const { data: investigation, isLoading: investigationLoading, isError: investigationError } = useInvestigation(expandedTxn)

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
      const validTimestamps = entries
        .map((entry) => entry.timestamp)
        .filter((timestamp) => Number.isFinite(timestamp) && timestamp > 0)
      result.push({
        txnId,
        entries,
        verdict: entries.find((e) => e.type === 'verdict'),
        startTime: validTimestamps[0] ?? null,
        endTime: validTimestamps[validTimestamps.length - 1] ?? null,
      })
    }
    return result.sort((a, b) => (b.endTime ?? 0) - (a.endTime ?? 0)).slice(0, 10) // latest 10
  }, [agentLog])

  return (
    <div className="rounded-lg border border-border-subtle bg-bg-deep p-3">
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
          No investigations recorded yet. Waiting for bounded investigator evidence...
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
            const duration =
              chain.startTime != null && chain.endTime != null && chain.endTime >= chain.startTime
                ? chain.endTime - chain.startTime
                : null

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
                    <span className="text-[8px] text-accent-primary/70">{stepCounts.thinking}E</span>
                    <span className="text-[8px] text-[#DA251C]/70">{stepCounts.tool_call}C</span>
                  </div>
                  {verdictData && (
                    <SeverityBadge severity={verdictToSeverity(verdictData.verdict)} />
                  )}
                  <span className="ml-auto text-[8px] font-mono tabular-nums text-text-muted">
                    {duration != null ? fmtDuration(duration) : 'n/a'}
                  </span>
                  <ChevronDown className={cn(
                    'w-3 h-3 text-text-muted transition-transform duration-200',
                    isExpanded && 'rotate-180',
                  )} />
                </button>

                {/* Expanded Timeline */}
                {isExpanded && (
                  (() => {
                    const persistedEvidence =
                      investigation?.txn_id === chain.txnId
                        ? investigation.evidence_collected
                        : undefined
                    const liveEvidence = evidenceFromLogEntries(chain.entries)
                    const hasPersistedEvidence = Object.keys(persistedEvidence ?? {}).length > 0
                    return (
                      <div className="px-2 pb-2.5 pt-0.5">
                        <div className="relative ml-3 border-l border-border-subtle/50 pl-3 space-y-2">
                          {chain.entries.map((entry) => {
                            const cfg = STEP_CONFIG[entry.type]
                            const StepIcon = cfg.icon
                            return (
                              <div key={entry.id} className="relative">
                                {/* Timeline dot */}
                                <div className={cn(
                                  'absolute -left-[17px] top-1 w-2 h-2 rounded-full ring-2 ring-bg-deep',
                                  cfg.dotBg,
                                )} />
                                <div className="text-[9px]">
                                  <div className="flex items-center gap-1.5 mb-0.5">
                                    <StepIcon className={cn('w-2.5 h-2.5', cfg.color)} />
                                    <span className={cn('font-medium', cfg.color)}>{cfg.label}</span>
                                    <span className="ml-auto font-mono tabular-nums text-text-muted text-[8px]">
                                      {fmtOptionalTimestamp(entry.timestamp)}
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
                        <BackendEvidence
                          loading={investigationLoading && Object.keys(liveEvidence).length === 0}
                          error={investigationError && Object.keys(liveEvidence).length === 0}
                          evidence={hasPersistedEvidence ? persistedEvidence : liveEvidence}
                          evidenceSource={hasPersistedEvidence ? 'persisted investigation record' : 'live SSE tool summary'}
                          model={verdictData?.model_used ?? investigation?.verdict?.model_used}
                          parseStatus={verdictData?.llm_parse_status ?? investigation?.verdict?.llm_parse_status}
                          confidenceSource={verdictData?.confidence_source ?? investigation?.verdict?.confidence_source}
                        />
                      </div>
                    )
                  })()
                )}
              </div>
            )
          })}
        </div>
      )}
    </div>
  )
}

function BackendEvidence({
  loading,
  error,
  evidence,
  evidenceSource,
  model,
  parseStatus,
  confidenceSource,
}: {
  loading: boolean
  error: boolean
  evidence: Record<string, unknown> | undefined
  evidenceSource: string
  model?: string | null
  parseStatus?: string
  confidenceSource?: string
}) {
  const entries = Object.entries(evidence ?? {})
  return (
    <div className="mt-3 rounded-md border border-border-subtle/60 bg-bg-surface/70 p-2.5">
      <div className="mb-2 flex items-center gap-1.5 text-[9px] font-semibold uppercase tracking-[0.12em] text-text-muted">
        <Database className="h-3 w-3 text-accent-primary" />
        Backend Evidence
        <span className="rounded bg-bg-deep px-1.5 py-0.5 font-mono text-[7px] normal-case tracking-normal text-text-muted">
          {evidenceSource}
        </span>
        {model && <span className="ml-auto font-mono normal-case tracking-normal text-text-muted">{model}</span>}
      </div>

      {loading ? (
        <div className="flex items-center gap-2 text-[9px] text-text-muted">
          <Loader2 className="h-3 w-3 animate-spin" />
          Loading investigation record
        </div>
      ) : error ? (
        <div className="text-[9px] leading-relaxed text-text-muted">
          Investigation record is not available from the backend for this transaction.
        </div>
      ) : entries.length === 0 ? (
        <div className="text-[9px] leading-relaxed text-text-muted">
          No backend tool evidence was recorded for this transaction.
        </div>
      ) : (
        <div className="space-y-1.5">
          <div className="grid grid-cols-2 gap-1.5">
            {parseStatus && <EvidenceBadge label="Parse" value={parseStatus} />}
            {confidenceSource && <EvidenceBadge label="Confidence" value={confidenceSource} />}
          </div>
          {entries.slice(0, 5).map(([toolName, raw]) => (
            <ToolEvidenceRow key={toolName} toolName={toolName} raw={raw} />
          ))}
        </div>
      )}
    </div>
  )
}

function EvidenceBadge({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded border border-border-subtle bg-bg-deep/55 px-2 py-1">
      <div className="text-[7px] uppercase tracking-[0.12em] text-text-muted">{label}</div>
      <div className="truncate font-mono text-[8px] text-text-secondary" title={value}>{value}</div>
    </div>
  )
}

function ToolEvidenceRow({ toolName, raw }: { toolName: string; raw: unknown }) {
  const payload = raw && typeof raw === 'object' ? raw as Record<string, unknown> : {}
  const success = payload.success
  const data = payload.data && typeof payload.data === 'object'
    ? payload.data as Record<string, unknown>
    : {}
  const duration = typeof data._execution_ms === 'number'
    ? fmtOptionalMs(data._execution_ms, 1)
    : ''
  const summary = summarizeToolData(toolName, data)

  return (
    <div className="rounded border border-border-subtle/50 bg-bg-deep/45 px-2 py-1.5">
      <div className="flex items-center gap-2">
        <span className="truncate font-mono text-[9px] font-semibold text-text-primary">{toolName}</span>
        {typeof success === 'boolean' && (
          <span className={cn('rounded px-1 py-0.5 text-[7px] font-bold uppercase', success ? 'bg-alert-low/10 text-alert-low' : 'bg-alert-critical/10 text-alert-critical')}>
            {success ? 'ok' : 'fail'}
          </span>
        )}
        {duration && <span className="ml-auto font-mono text-[8px] text-text-muted">{duration}</span>}
      </div>
      {summary && (
        <div className="mt-1 text-[8px] leading-relaxed text-text-muted">
          {summary}
        </div>
      )}
    </div>
  )
}

function evidenceFromLogEntries(entries: AgentLogEntry[]): Record<string, unknown> {
  const evidence: Record<string, unknown> = {}
  for (const entry of entries) {
    if (entry.type !== 'tool_call') continue
    const tool = entry.data as SSEAgentToolCall
    const name = tool.tool_name ?? 'unknown_tool'
    let parsed: Record<string, unknown> | null = null
    if (tool.output_summary) {
      try {
        const value = JSON.parse(tool.output_summary)
        if (value && typeof value === 'object') parsed = value as Record<string, unknown>
      } catch {
        parsed = null
      }
    }
    evidence[name] = parsed ?? {
      tool_name: name,
      success: tool.success,
      data: {
        summary: tool.output_summary,
        _execution_ms: tool.duration_ms,
      },
    }
  }
  return evidence
}

function summarizeToolData(toolName: string, data: Record<string, unknown>): string {
  if (typeof data.summary === 'string' && data.summary.trim()) {
    return data.summary
  }
  if (toolName === 'query_graph_database') {
    const connections = data.connections && typeof data.connections === 'object'
      ? data.connections as Record<string, unknown>
      : {}
    const patterns = data.patterns && typeof data.patterns === 'object'
      ? data.patterns as Record<string, unknown>
      : {}
    return [
      `senders ${String(connections.distinct_senders ?? 'n/a')}`,
      `receivers ${String(connections.distinct_receivers ?? 'n/a')}`,
      `cycles ${String(patterns.cycles_found ?? 'n/a')}`,
      `mule ${String(patterns.mule_network_detected ?? 'n/a')}`,
    ].join(' | ')
  }
  if (toolName === 'get_ml_feature_analysis') {
    const topFeatures = data.top_features && typeof data.top_features === 'object'
      ? Object.entries(data.top_features as Record<string, unknown>).slice(0, 3)
      : []
    return topFeatures.map(([name, value]) => `${name}: ${String(value)}`).join(' | ')
  }
  if (toolName === 'read_audit_logs') {
    return `ledger entries ${String(data.entries_count ?? 0)}`
  }
  if (toolName === 'check_node_freeze_status') {
    return `node ${String(data.node_id ?? 'n/a')} frozen ${String(data.is_frozen ?? 'n/a')}`
  }
  return Object.keys(data).slice(0, 4).join(' | ')
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

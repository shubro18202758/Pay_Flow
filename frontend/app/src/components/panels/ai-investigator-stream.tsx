// ============================================================================
// AI Investigator Stream -- Live terminal-style Chain-of-Thought renderer
// Grouped by investigation (txn_id) with collapsible groups
// ============================================================================

import { useRef, useEffect, useState, useMemo } from 'react'
import { Terminal, Brain, Wrench, Gavel, ChevronDown, ChevronRight } from 'lucide-react'
import { useDashboardStore } from '@/stores/use-dashboard-store'
import { cn, fmtTimestamp, truncId } from '@/lib/utils'
import type { AgentLogEntry, SSEAgentThinking, SSEAgentToolCall, SSEAgentVerdict } from '@/lib/types'

const MAX_VISIBLE = 200

type ViewMode = 'grouped' | 'flat'

function getTxnId(entry: AgentLogEntry): string {
  const d = entry.data as { txn_id?: string }
  return d.txn_id ?? 'unknown'
}

interface InvestigationGroup {
  txnId: string
  entries: AgentLogEntry[]
  hasVerdict: boolean
  verdictResult?: string
}

export function AIInvestigatorStream() {
  const agentLog = useDashboardStore((s) => s.agentLog)
  const scrollRef = useRef<HTMLDivElement>(null)
  const [autoScroll, setAutoScroll] = useState(true)
  const [viewMode, setViewMode] = useState<ViewMode>('grouped')
  const [collapsedGroups, setCollapsedGroups] = useState<Set<string>>(new Set())

  const visible = useMemo(
    () => (agentLog.length > MAX_VISIBLE ? agentLog.slice(-MAX_VISIBLE) : agentLog),
    [agentLog],
  )

  const groups = useMemo(() => {
    const map = new Map<string, InvestigationGroup>()
    for (const entry of visible) {
      const txnId = getTxnId(entry)
      let group = map.get(txnId)
      if (!group) {
        group = { txnId, entries: [], hasVerdict: false }
        map.set(txnId, group)
      }
      group.entries.push(entry)
      if (entry.type === 'verdict') {
        group.hasVerdict = true
        group.verdictResult = (entry.data as SSEAgentVerdict).verdict
      }
    }
    // Return in order of last activity (most recent last)
    return [...map.values()].sort((a, b) => {
      const aLast = a.entries[a.entries.length - 1].timestamp
      const bLast = b.entries[b.entries.length - 1].timestamp
      return aLast - bLast
    })
  }, [visible])

  const toggleGroup = (txnId: string) => {
    setCollapsedGroups((prev) => {
      const next = new Set(prev)
      if (next.has(txnId)) next.delete(txnId)
      else next.add(txnId)
      return next
    })
  }

  // Auto-scroll to bottom on new entries
  useEffect(() => {
    if (autoScroll && scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight
    }
  }, [visible, autoScroll])

  // Detect manual scroll-up to pause auto-scroll
  function handleScroll() {
    const el = scrollRef.current
    if (!el) return
    const atBottom = el.scrollHeight - el.scrollTop - el.clientHeight < 40
    setAutoScroll(atBottom)
  }

  return (
    <div className="flex flex-col h-full bg-bg-deep rounded-lg border border-border-subtle overflow-hidden">
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-2.5 border-b border-border-subtle shrink-0 bg-bg-surface">
        <div className="flex items-center gap-2.5">
          <Terminal className="w-3.5 h-3.5 text-emerald-400" />
          <span className="relative flex h-2 w-2">
            <span className={cn(
              'absolute inline-flex h-full w-full rounded-full opacity-75',
              agentLog.length > 0 ? 'animate-ping bg-emerald-400' : 'bg-gray-500',
            )} />
            <span className={cn(
              'relative inline-flex rounded-full h-2 w-2',
              agentLog.length > 0 ? 'bg-emerald-400' : 'bg-gray-500',
            )} />
          </span>
          <span className="text-[10px] font-semibold uppercase tracking-[0.12em] text-text-secondary">
            AI Investigator Stream
          </span>
        </div>
        <div className="flex items-center gap-2">
          <div className="flex items-center gap-0.5 bg-bg-overlay rounded p-0.5">
            {(['grouped', 'flat'] as ViewMode[]).map((m) => (
              <button
                key={m}
                onClick={() => setViewMode(m)}
                className={cn(
                  'px-2 py-0.5 rounded text-[8px] font-semibold uppercase tracking-wider transition-all',
                  viewMode === m
                    ? 'bg-accent-primary/20 text-accent-primary'
                    : 'text-text-muted hover:text-text-secondary',
                )}
              >
                {m}
              </button>
            ))}
          </div>
          <span className="text-[9px] text-text-muted font-mono tabular-nums">
            {groups.length} inv · {agentLog.length} entries
          </span>
        </div>
      </div>

      {/* Terminal body */}
      <div
        ref={scrollRef}
        onScroll={handleScroll}
        className="flex-1 overflow-y-auto p-3 font-mono text-[11px] leading-relaxed space-y-0.5"
      >
        {visible.length === 0 && (
          <div className="flex items-center justify-center h-full animate-fade-in">
            <div className="text-center space-y-3">
              <Terminal className="w-8 h-8 text-text-muted/30 mx-auto" />
              <p className="text-text-muted text-[10px] uppercase tracking-[0.12em]">
                Awaiting agent activity...
              </p>
              <p className="text-text-muted/50 text-[9px] max-w-[240px] mx-auto leading-relaxed">
                Launch a threat simulation to see the AI's Chain-of-Thought reasoning
              </p>
            </div>
          </div>
        )}

        {viewMode === 'grouped' ? (
          groups.map((group) => {
            const isCollapsed = collapsedGroups.has(group.txnId)
            const verdictColor = group.verdictResult === 'FRAUDULENT'
              ? 'border-red-500/40 bg-red-500/[0.03]'
              : group.verdictResult === 'SUSPICIOUS'
                ? 'border-amber-500/40 bg-amber-500/[0.03]'
                : group.hasVerdict
                  ? 'border-emerald-500/40 bg-emerald-500/[0.03]'
                  : 'border-border-subtle bg-white/[0.01]'
            return (
              <div key={group.txnId} className={cn('rounded-md border mb-2 overflow-hidden transition-all', verdictColor)}>
                <button
                  onClick={() => toggleGroup(group.txnId)}
                  className="w-full flex items-center gap-2 px-2.5 py-1.5 hover:bg-white/[0.03] transition-colors text-left"
                >
                  {isCollapsed ? <ChevronRight className="w-3 h-3 text-text-muted shrink-0" /> : <ChevronDown className="w-3 h-3 text-text-muted shrink-0" />}
                  <span className="text-[10px] text-amber-200/70 font-semibold">txn:{truncId(group.txnId, 10)}</span>
                  <span className="text-[9px] text-text-muted">{group.entries.length} events</span>
                  {group.hasVerdict && (
                    <span className={cn(
                      'text-[8px] font-bold px-1.5 py-0.5 rounded',
                      group.verdictResult === 'FRAUDULENT' ? 'text-red-400 bg-red-500/10'
                        : group.verdictResult === 'SUSPICIOUS' ? 'text-amber-400 bg-amber-500/10'
                          : 'text-emerald-400 bg-emerald-500/10',
                    )}>
                      {group.verdictResult}
                    </span>
                  )}
                  {!group.hasVerdict && (
                    <span className="text-[8px] text-cyan-400 animate-pulse">in progress</span>
                  )}
                  <span className="ml-auto text-[8px] text-text-muted/50">
                    {fmtTimestamp(group.entries[group.entries.length - 1].timestamp)}
                  </span>
                </button>
                {!isCollapsed && (
                  <div className="px-1 pb-1 space-y-0.5">
                    {group.entries.map((entry) => (
                      <LogLine key={entry.id} entry={entry} />
                    ))}
                  </div>
                )}
              </div>
            )
          })
        ) : (
          visible.map((entry) => (
            <LogLine key={entry.id} entry={entry} />
          ))
        )}

        {/* Blinking cursor */}
        {agentLog.length > 0 && (
          <span className="inline-block w-2 h-4 rounded-sm bg-emerald-400 animate-pulse ml-1" />
        )}
      </div>

      {/* Scroll indicator */}
      {!autoScroll && (
        <button
          onClick={() => {
            setAutoScroll(true)
            scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: 'smooth' })
          }}
          className="absolute bottom-3 right-4 flex items-center gap-1.5 text-[9px] bg-accent-primary/90 text-white px-2.5 py-1 rounded-md hover:bg-accent-primary transition-colors shadow-lg shadow-accent-primary/20"
        >
          <ChevronDown className="w-3 h-3" />
          Resume auto-scroll
        </button>
      )}
    </div>
  )
}

// -- Individual log entry renderer --

function LogLine({ entry }: { entry: AgentLogEntry }) {
  const ts = fmtTimestamp(entry.timestamp)

  if (entry.type === 'thinking') {
    const d = entry.data as SSEAgentThinking
    return (
      <div className="flex gap-2 py-1 px-2 hover:bg-white/[0.03] rounded-md transition-colors group">
        <span className="text-text-muted/50 shrink-0 text-[10px]">{ts}</span>
        <span className="flex items-center gap-1 text-cyan-400 shrink-0 font-semibold">
          <Brain className="w-3 h-3" />
          <span className="text-[10px]">THINK</span>
        </span>
        <span className="text-amber-200/70 shrink-0 text-[10px]">txn:{truncId(d.txn_id, 8)}</span>
        <span className="text-text-secondary/90">{d.content}</span>
      </div>
    )
  }

  if (entry.type === 'tool_call') {
    const d = entry.data as SSEAgentToolCall
    return (
      <div className="flex gap-2 py-1 px-2 hover:bg-white/[0.03] rounded-md transition-colors bg-violet-500/[0.03] group">
        <span className="text-text-muted/50 shrink-0 text-[10px]">{ts}</span>
        <span className="flex items-center gap-1 text-violet-400 shrink-0 font-semibold">
          <Wrench className="w-3 h-3" />
          <span className="text-[10px]">TOOL</span>
        </span>
        <span className="text-amber-200/70 shrink-0 text-[10px]">txn:{truncId(d.txn_id, 8)}</span>
        <span className="text-text-secondary/90">
          {d.tool_name}
          <span className={cn(
            'font-bold ml-1',
            d.success ? 'text-emerald-400' : 'text-red-400',
          )}>
            {d.success ? 'OK' : 'FAIL'}
          </span>
          <span className="text-text-muted/40 ml-1">({d.duration_ms}ms)</span>
        </span>
      </div>
    )
  }

  // verdict
  const d = entry.data as SSEAgentVerdict
  const verdictColor =
    d.verdict === 'FRAUDULENT'
      ? 'text-red-400'
      : d.verdict === 'SUSPICIOUS'
        ? 'text-amber-400'
        : 'text-emerald-400'

  const glowColor =
    d.verdict === 'FRAUDULENT'
      ? 'shadow-red-500/10'
      : d.verdict === 'SUSPICIOUS'
        ? 'shadow-amber-500/10'
        : 'shadow-emerald-500/10'

  const borderColor =
    d.verdict === 'FRAUDULENT'
      ? 'border-red-500/30'
      : d.verdict === 'SUSPICIOUS'
        ? 'border-amber-500/30'
        : 'border-emerald-500/30'

  return (
    <div className={cn(
      'py-2 px-3 my-1.5 rounded-md border-l-2 transition-all',
      'bg-white/[0.03] shadow-lg',
      borderColor,
      glowColor,
    )}>
      <div className="flex gap-2 items-center">
        <span className="text-text-muted/50 shrink-0 text-[10px]">{ts}</span>
        <span className="flex items-center gap-1 text-accent-primary font-bold shrink-0">
          <Gavel className="w-3.5 h-3.5" />
          <span className="text-[10px]">VERDICT</span>
        </span>
        <span className="text-amber-200/70 shrink-0 text-[10px]">txn:{truncId(d.txn_id, 8)}</span>
        <span className={cn('font-bold', verdictColor)}>{d.verdict}</span>
        <span className="text-text-muted font-mono text-[10px]">conf:{(d.confidence * 100).toFixed(0)}%</span>
      </div>
      {d.reasoning_summary && (
        <div className="ml-[5rem] mt-1 text-text-muted/80 text-[10px] italic leading-relaxed">
          {d.reasoning_summary}
        </div>
      )}
      <div className="ml-[5rem] mt-1 flex gap-3 text-[9px] text-text-muted/50">
        <span>action: {d.recommended_action}</span>
        <span>steps: {d.thinking_steps}</span>
        <span>tools: {d.tools_used?.join(', ')}</span>
        <span className="font-mono">{d.total_duration_ms}ms</span>
      </div>
    </div>
  )
}

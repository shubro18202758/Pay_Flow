// ============================================================================
// Agent Chain-of-Thought -- Thinking steps, tool calls, verdicts (bottom drawer)
// Enhanced: expandable entries, tool args preview, confidence bar, evidence tags
// ============================================================================

import { useRef, useEffect, useState, useCallback } from 'react'
import { useDashboardStore } from '@/stores/use-dashboard-store'
import { SeverityBadge, verdictToSeverity } from '@/components/shared/severity-badge'
import { cn, truncId } from '@/lib/utils'
import type { SSEAgentThinking, SSEAgentToolCall, SSEAgentVerdict } from '@/lib/types'
import {
  Brain,
  Wrench,
  Gavel,
  BrainCircuit,
  ChevronDown,
  ChevronRight,
  CheckCircle2,
  XCircle,
  Clock,
  Layers,
  AlertTriangle,
  Shield,
} from 'lucide-react'

/* ── Expandable wrapper ── */
function Expandable({
  children,
  preview,
  defaultOpen = false,
}: {
  children: React.ReactNode
  preview: React.ReactNode
  defaultOpen?: boolean
}) {
  const [open, setOpen] = useState(defaultOpen)
  return (
    <div>
      <div
        className="flex items-start gap-1 cursor-pointer select-none"
        onClick={() => setOpen(!open)}
      >
        {open ? (
          <ChevronDown className="w-2.5 h-2.5 text-text-muted shrink-0 mt-0.5" />
        ) : (
          <ChevronRight className="w-2.5 h-2.5 text-text-muted shrink-0 mt-0.5" />
        )}
        <div className="flex-1 min-w-0">{preview}</div>
      </div>
      {open && <div className="ml-3.5 mt-1">{children}</div>}
    </div>
  )
}

/* ── Confidence mini-bar ── */
function ConfidenceBar({ value }: { value: number }) {
  const pct = Math.round(value * 100)
  const color =
    pct >= 85 ? 'bg-alert-critical' :
    pct >= 60 ? 'bg-alert-high' :
    pct >= 40 ? 'bg-alert-medium' :
    'bg-alert-low'
  return (
    <div className="flex items-center gap-1.5">
      <div className="w-12 h-1 rounded-full bg-bg-elevated overflow-hidden">
        <div className={cn('h-full rounded-full transition-all duration-500', color)} style={{ width: `${pct}%` }} />
      </div>
      <span className="text-[9px] font-mono tabular-nums text-text-muted">{pct}%</span>
    </div>
  )
}

/* ── Latency badge ── */
function LatencyBadge({ ms }: { ms: number }) {
  const color = ms < 100 ? 'text-green-400' : ms < 500 ? 'text-blue-400' : ms < 2000 ? 'text-amber-400' : 'text-red-400'
  return <span className={cn('text-[9px] font-mono tabular-nums', color)}>{ms.toFixed(0)}ms</span>
}

export function AgentCoT() {
  const agentLog = useDashboardStore((s) => s.agentLog)
  const scrollRef = useRef<HTMLDivElement>(null)
  const [autoScroll, setAutoScroll] = useState(true)

  const thinkingCount = agentLog.filter((e) => e.type === 'thinking').length
  const toolCount = agentLog.filter((e) => e.type === 'tool_call').length
  const verdictCount = agentLog.filter((e) => e.type === 'verdict').length

  useEffect(() => {
    const el = scrollRef.current
    if (el && autoScroll) {
      el.scrollTop = el.scrollHeight
    }
  }, [agentLog.length, autoScroll])

  const handleScroll = useCallback(() => {
    const el = scrollRef.current
    if (!el) return
    const nearBottom = el.scrollHeight - el.scrollTop - el.clientHeight < 50
    setAutoScroll(nearBottom)
  }, [])

  return (
    <div className="flex flex-col h-full">
      {/* ── Header ── */}
      <div className="flex items-center justify-between px-3 py-1.5 border-b border-border-subtle shrink-0 bg-bg-surface/50">
        <div className="flex items-center gap-2">
          <BrainCircuit className="w-3.5 h-3.5 text-accent-primary" />
          <span className="text-[10px] font-semibold uppercase tracking-[0.12em] text-text-secondary">
            Qwen 3.5 Agent CoT
          </span>
        </div>
        <div className="flex items-center gap-2.5">
          {agentLog.length > 0 && (
            <div className="flex items-center gap-2 text-[8px] font-mono text-text-muted">
              <span className="flex items-center gap-0.5">
                <Brain className="w-2.5 h-2.5 text-accent-primary" />
                {thinkingCount}
              </span>
              <span className="flex items-center gap-0.5">
                <Wrench className="w-2.5 h-2.5 text-alert-escalated" />
                {toolCount}
              </span>
              <span className="flex items-center gap-0.5">
                <Gavel className="w-2.5 h-2.5 text-text-secondary" />
                {verdictCount}
              </span>
            </div>
          )}
          <span className="text-[9px] font-mono text-text-muted tabular-nums">
            {agentLog.length}
          </span>
        </div>
      </div>

      {/* ── Log entries ── */}
      <div ref={scrollRef} onScroll={handleScroll} className="flex-1 overflow-y-auto p-2 space-y-1">
        {agentLog.length === 0 ? (
          <div className="flex flex-col items-center justify-center h-full gap-3 animate-fade-in">
            <div className="relative">
              <BrainCircuit className="w-8 h-8 text-text-muted/20" />
              <div className="absolute inset-0 animate-radar-ping opacity-20">
                <BrainCircuit className="w-8 h-8 text-accent-primary/40" />
              </div>
            </div>
            <div className="text-center">
              <p className="text-text-muted text-[10px] uppercase tracking-[0.12em] font-semibold">
                Awaiting agent activity
              </p>
              <p className="text-text-muted/50 text-[9px] mt-1">
                LangGraph think → act → observe → verdict loop
              </p>
            </div>
          </div>
        ) : (
          agentLog.map((entry) => {
            /* ── Thinking step ── */
            if (entry.type === 'thinking') {
              const d = entry.data as SSEAgentThinking
              return (
                <Expandable
                  key={entry.id}
                  preview={
                    <div className="flex items-start gap-2 p-1.5 rounded-md text-[10px] font-mono leading-snug bg-bg-elevated/60 border-l-2 border-accent-primary hover:bg-bg-elevated transition-colors">
                      <Brain className="w-3 h-3 text-accent-primary shrink-0 mt-0.5" />
                      <div className="flex-1 min-w-0">
                        <div className="flex items-center gap-2">
                          <span className="text-accent-primary text-[9px] font-bold">THINK</span>
                          <span className="text-text-muted text-[8px]">
                            {truncId(d.txn_id, 8)}
                          </span>
                          <span className="text-text-muted/60 text-[8px] tabular-nums">
                            step {d.iteration}
                          </span>
                        </div>
                        <p className="text-text-primary text-[9px] mt-0.5 truncate">{d.content}</p>
                      </div>
                    </div>
                  }
                >
                  <div className="border-l border-border-subtle pl-2 py-1">
                    <p className="text-text-secondary text-[9px] font-mono leading-relaxed whitespace-pre-wrap">
                      {d.content}
                    </p>
                  </div>
                </Expandable>
              )
            }

            /* ── Tool call ── */
            if (entry.type === 'tool_call') {
              const d = entry.data as SSEAgentToolCall
              return (
                <Expandable
                  key={entry.id}
                  preview={
                    <div className="flex items-start gap-2 p-1.5 rounded-md text-[10px] font-mono leading-snug bg-bg-elevated/60 border-l-2 border-alert-escalated hover:bg-bg-elevated transition-colors">
                      <Wrench className="w-3 h-3 text-alert-escalated shrink-0 mt-0.5" />
                      <div className="flex items-center gap-2 flex-wrap flex-1 min-w-0">
                        <span className="text-text-primary font-semibold">{d.tool_name}</span>
                        {d.success ? (
                          <CheckCircle2 className="w-3 h-3 text-alert-low shrink-0" />
                        ) : (
                          <XCircle className="w-3 h-3 text-alert-critical shrink-0" />
                        )}
                        <LatencyBadge ms={d.duration_ms} />
                      </div>
                    </div>
                  }
                >
                  <div className="border-l border-border-subtle pl-2 py-1 space-y-1">
                    <div className="flex items-center gap-2 text-[9px]">
                      <Clock className="w-2.5 h-2.5 text-text-muted" />
                      <span className="text-text-muted">Duration: {d.duration_ms.toFixed(1)}ms</span>
                      <span className={d.success ? 'text-alert-low' : 'text-alert-critical'}>{d.success ? 'SUCCESS' : 'FAILED'}</span>
                    </div>
                    {d.tool_args && (
                      <pre className="text-[8px] font-mono text-text-muted/80 bg-bg-deep/50 rounded p-1.5 overflow-x-auto max-h-20">
                        {typeof d.tool_args === 'string' ? d.tool_args : JSON.stringify(d.tool_args, null, 2)}
                      </pre>
                    )}
                    {d.result_summary && (
                      <p className="text-[9px] text-text-secondary">{d.result_summary}</p>
                    )}
                  </div>
                </Expandable>
              )
            }

            /* ── Verdict ── */
            const d = entry.data as SSEAgentVerdict
            const severity = verdictToSeverity(d.verdict)
            return (
              <Expandable
                key={entry.id}
                defaultOpen={severity === 'critical' || severity === 'high'}
                preview={
                  <div
                    className={cn(
                      'p-2 rounded-md text-[10px] font-mono leading-snug bg-bg-elevated border-l-3',
                      severity === 'critical' && 'border-alert-critical',
                      severity === 'high' && 'border-alert-high',
                      severity === 'low' && 'border-alert-low',
                      severity === 'escalated' && 'border-alert-escalated',
                      severity === 'medium' && 'border-alert-medium',
                    )}
                  >
                    <div className="flex items-center gap-2">
                      <Gavel className="w-3 h-3 text-text-secondary shrink-0" />
                      <SeverityBadge severity={severity} label={d.verdict.toUpperCase()} />
                      <span className="text-text-primary font-bold">{truncId(d.txn_id, 10)}</span>
                      <div className="ml-auto">
                        <ConfidenceBar value={d.confidence} />
                      </div>
                    </div>
                  </div>
                }
              >
                <div className="border-l border-border-subtle pl-2 py-1.5 space-y-1.5">
                  {d.reasoning_summary && (
                    <div className="text-[9px] text-text-secondary leading-relaxed">
                      <span className="text-text-muted font-semibold uppercase text-[8px]">Reasoning: </span>
                      {d.reasoning_summary}
                    </div>
                  )}
                  {d.evidence && d.evidence.length > 0 && (
                    <div className="flex items-start gap-1.5 text-[9px]">
                      <AlertTriangle className="w-2.5 h-2.5 text-amber-400 shrink-0 mt-0.5" />
                      <div className="flex flex-wrap gap-1">
                        {d.evidence.map((ev: string, i: number) => (
                          <span key={i} className="px-1.5 py-0.5 rounded bg-bg-deep text-text-muted text-[8px] font-mono border border-border-subtle">
                            {ev}
                          </span>
                        ))}
                      </div>
                    </div>
                  )}
                  {d.recommended_action && (
                    <div className="flex items-start gap-1.5 text-[9px]">
                      <Shield className="w-2.5 h-2.5 text-accent-primary shrink-0 mt-0.5" />
                      <span className="text-text-secondary">{d.recommended_action}</span>
                    </div>
                  )}
                  <div className="flex items-center gap-3 text-[8px] text-text-muted tabular-nums">
                    <span><Layers className="w-2 h-2 inline mr-0.5" />{d.thinking_steps} steps</span>
                    <span><Wrench className="w-2 h-2 inline mr-0.5" />{d.tools_used?.length ?? 0} tools</span>
                    <span><Clock className="w-2 h-2 inline mr-0.5" />{d.total_duration_ms?.toFixed(0)}ms</span>
                  </div>
                </div>
              </Expandable>
            )
          })
        )}
      </div>

      {/* ── Auto-scroll indicator ── */}
      {!autoScroll && agentLog.length > 0 && (
        <button
          onClick={() => {
            setAutoScroll(true)
            const el = scrollRef.current
            if (el) el.scrollTop = el.scrollHeight
          }}
          className="absolute bottom-2 right-4 px-2 py-1 text-[9px] font-mono bg-accent-primary/20 text-accent-primary rounded-md border border-accent-primary/30 hover:bg-accent-primary/30 transition-colors"
        >
          ↓ Jump to latest
        </button>
      )}
    </div>
  )
}

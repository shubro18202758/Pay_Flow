// ============================================================================
// Trigger Feed -- Live verdict feed with typology clustering + simulation fallback
// ============================================================================

import { useDashboardStore } from '@/stores/use-dashboard-store'
import { useSimulationStore } from '@/stores/use-simulation-store'
import { useFraudTypology } from '@/hooks/use-api'
import { SeverityBadge, verdictToSeverity } from '@/components/shared/severity-badge'
import { fmtPaisa, fmtTimestamp, truncId, cn } from '@/lib/utils'
import { Bell, Inbox, Filter, Tag } from 'lucide-react'
import { useState, useMemo } from 'react'
import type { SSEAgentVerdict, AgentLogEntry } from '@/lib/types'

const TYPOLOGY_COLORS: Record<string, string> = {
  layering: 'bg-red-500/20 text-red-400 border-red-500/30',
  'round-tripping': 'bg-orange-500/20 text-orange-400 border-orange-500/30',
  structuring: 'bg-amber-500/20 text-amber-400 border-amber-500/30',
  'dormant activation': 'bg-purple-500/20 text-purple-400 border-purple-500/30',
  'profile mismatch': 'bg-blue-500/20 text-blue-400 border-blue-500/30',
  'upi mule network': 'bg-rose-500/20 text-rose-400 border-rose-500/30',
  'circular laundering': 'bg-pink-500/20 text-pink-400 border-pink-500/30',
  'velocity phishing': 'bg-cyan-500/20 text-cyan-400 border-cyan-500/30',
}

function getTypologyColor(typology: string): string {
  const key = typology.toLowerCase().replace(/_/g, ' ')
  return TYPOLOGY_COLORS[key] ?? 'bg-zinc-500/20 text-zinc-400 border-zinc-500/30'
}

export function AlertFeed() {
  const agentLog = useDashboardStore((s) => s.agentLog)
  const recentEvents = useSimulationStore((s) => s.recentEvents)
  const { data: typologyData } = useFraudTypology()
  const [filterTypology, setFilterTypology] = useState<string | null>(null)

  const allVerdicts = useMemo(() =>
    agentLog.filter((e) => e.type === 'verdict').reverse().slice(0, 30),
    [agentLog],
  )

  const verdicts = useMemo(() => {
    if (!filterTypology) return allVerdicts.slice(0, 24)
    return allVerdicts
      .filter((e) => {
        const v = e.data as SSEAgentVerdict
        return v.fraud_typology?.toLowerCase().replace(/_/g, ' ') === filterTypology.toLowerCase()
      })
      .slice(0, 24)
  }, [allVerdicts, filterTypology])

  const traceFallback = recentEvents.slice(-12).reverse()
  const hasVerdicts = allVerdicts.length > 0

  // Collect active typologies for filter chips
  const activeTypologies = useMemo(() => {
    const counts = new Map<string, number>()
    for (const entry of allVerdicts) {
      const v = entry.data as SSEAgentVerdict
      if (v.fraud_typology) {
        const key = v.fraud_typology
        counts.set(key, (counts.get(key) ?? 0) + 1)
      }
    }
    return Array.from(counts.entries()).sort((a, b) => b[1] - a[1])
  }, [allVerdicts])

  return (
    <div className="flex flex-col h-full">
      {/* Header */}
      <div className="flex items-center justify-between px-3 py-2.5 border-b border-border-subtle shrink-0">
        <div className="flex items-center gap-1.5">
          <Bell className="w-3.5 h-3.5 text-accent-primary" />
          <span className="text-[10px] font-semibold uppercase tracking-[0.12em] text-text-secondary">
            {hasVerdicts ? 'Trigger Feed' : 'Injected Triggers'}
          </span>
        </div>
        <span className="text-[10px] font-mono tabular-nums text-text-muted">
          {hasVerdicts ? `${verdicts.length} verdicts` : `${traceFallback.length} events`}
        </span>
      </div>

      {/* Typology Filter Chips */}
      {hasVerdicts && activeTypologies.length > 0 && (
        <div className="flex items-center gap-1 px-2 py-1.5 border-b border-border-subtle/50 overflow-x-auto shrink-0">
          <Filter className="w-3 h-3 text-text-muted shrink-0" />
          <button
            className={cn(
              'shrink-0 rounded-full px-2 py-0.5 text-[8px] font-medium border transition-all',
              !filterTypology ? 'bg-accent-primary/20 text-accent-primary border-accent-primary/30' : 'bg-bg-deep/30 text-text-muted border-border-subtle/30 hover:bg-bg-deep/50',
            )}
            onClick={() => setFilterTypology(null)}
          >
            All
          </button>
          {activeTypologies.map(([typ, count]) => (
            <button
              key={typ}
              className={cn(
                'shrink-0 rounded-full px-2 py-0.5 text-[8px] font-medium border transition-all',
                filterTypology === typ ? getTypologyColor(typ) : 'bg-bg-deep/30 text-text-muted border-border-subtle/30 hover:bg-bg-deep/50',
              )}
              onClick={() => setFilterTypology(filterTypology === typ ? null : typ)}
            >
              {typ.replace(/_/g, ' ')} ({count})
            </button>
          ))}
        </div>
      )}

      {/* Typology Distribution Bar (from backend analytics) */}
      {typologyData && typologyData.total > 0 && hasVerdicts && (
        <div className="px-3 py-1.5 border-b border-border-subtle/30">
          <div className="flex h-1.5 rounded-full overflow-hidden bg-bg-deep/50">
            {Object.entries(typologyData.typology).map(([name, count], i) => {
              const pct = (count / typologyData.total) * 100
              if (pct < 1) return null
              return (
                <div
                  key={name}
                  className="h-full transition-all duration-500"
                  style={{
                    width: `${pct}%`,
                    backgroundColor: `hsl(${(i * 40 + 0) % 360}, 60%, 55%)`,
                  }}
                  title={`${name}: ${count} (${pct.toFixed(1)}%)`}
                />
              )
            })}
          </div>
        </div>
      )}

      {/* Content */}
      <div className="flex-1 overflow-y-auto">
        {!hasVerdicts && traceFallback.length === 0 ? (
          /* Empty state */
          <div className="flex flex-col items-center justify-center h-full gap-2.5 animate-fade-in">
            <Inbox className="w-6 h-6 text-text-muted/50" />
            <p className="text-text-muted text-[10px] uppercase tracking-[0.12em]">
              No triggers yet
            </p>
            <p className="text-text-muted/60 text-[9px] px-6 text-center leading-relaxed">
              Launch a simulation to see live events stream in
            </p>
          </div>
        ) : hasVerdicts ? (
          /* Verdict cards */
          <div className="p-2 space-y-1.5">
            {verdicts.map((entry, idx) => {
              const v = entry.data as SSEAgentVerdict
              const severity = verdictToSeverity(v.verdict)
              return (
                <div
                  key={entry.id}
                  className="rounded-md bg-bg-elevated/60 border border-border-subtle/50 p-2.5 card-hover transition-all duration-200 animate-fade-in"
                  style={{ animationDelay: `${idx * 30}ms` }}
                >
                  <div className="flex items-center gap-2">
                    <SeverityBadge severity={severity} />
                    <span className="flex-1 truncate text-[10px] font-mono text-text-primary">
                      {truncId(v.txn_id)}
                    </span>
                    <span className="text-[9px] font-mono tabular-nums text-text-muted shrink-0">
                      {(v.confidence * 100).toFixed(0)}%
                    </span>
                  </div>
                  {v.fraud_typology && (
                    <div className="mt-1.5 flex items-center gap-1.5">
                      <Tag className="w-2.5 h-2.5 shrink-0 text-text-muted" />
                      <span className={cn(
                        'inline-block rounded-full px-2 py-0.5 text-[8px] font-semibold border',
                        getTypologyColor(v.fraud_typology),
                      )}>
                        {v.fraud_typology.replace(/_/g, ' ')}
                      </span>
                    </div>
                  )}
                  {v.reasoning_summary && (
                    <div className="mt-1 line-clamp-3 text-[9px] leading-relaxed text-text-secondary">
                      {v.reasoning_summary}
                    </div>
                  )}
                  <div className="mt-2 flex flex-wrap items-center gap-2 text-[8px] font-mono text-text-muted">
                    <span className="px-1.5 py-0.5 rounded bg-bg-deep/50 border border-border-subtle/30">
                      {v.recommended_action}
                    </span>
                    <span className="tabular-nums">{v.thinking_steps} steps</span>
                    <span className="tabular-nums">{v.tools_used?.length ?? 0} tools</span>
                    <span className="ml-auto tabular-nums">{fmtTimestamp(entry.timestamp)}</span>
                  </div>
                </div>
              )
            })}
          </div>
        ) : (
          /* Trace fallback cards */
          <div className="p-2 space-y-1.5">
            {traceFallback.map((entry, idx) => (
              <div
                key={entry.id}
                className="rounded-md bg-bg-elevated/60 border border-border-subtle/50 p-2.5 card-hover transition-all duration-200 animate-fade-in"
                style={{ animationDelay: `${idx * 30}ms` }}
              >
                <div className="flex items-center gap-2 text-[10px]">
                  <span className="rounded border border-accent-primary/30 bg-accent-primary/10 px-1.5 py-0.5 text-[8px] font-semibold tracking-[0.12em] text-accent-primary">
                    {entry.eventType.toUpperCase()}
                  </span>
                  <span className="truncate text-text-primary">{entry.attackLabel}</span>
                  <span className="ml-auto font-mono tabular-nums text-text-muted">
                    {fmtTimestamp(entry.timestamp)}
                  </span>
                </div>
                <div className="mt-1.5 text-[9px] leading-relaxed text-text-secondary font-mono">
                  {entry.eventType === 'transaction' && (
                    <>
                      {truncId(entry.event.sender ?? 'unknown', 10)} {'->'} {truncId(entry.event.receiver ?? 'unknown', 10)} | {fmtPaisa(entry.event.amount_paisa ?? 0)}
                    </>
                  )}
                  {entry.eventType === 'auth' && (
                    <>
                      {truncId(entry.event.account ?? 'unknown', 10)} | {entry.event.action ?? 'AUTH'} | {entry.event.success ? 'SUCCESS' : 'FAIL'}
                    </>
                  )}
                  {entry.eventType === 'interbank' && (
                    <>
                      {entry.event.sender_ifsc ?? 'unknown'} {'->'} {entry.event.receiver_ifsc ?? 'unknown'} | {fmtPaisa(entry.event.amount_paisa ?? 0)}
                    </>
                  )}
                </div>
                <div className="mt-1.5 text-[8px] font-mono tabular-nums text-text-muted">
                  progress {entry.progressPct.toFixed(0)}%
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  )
}

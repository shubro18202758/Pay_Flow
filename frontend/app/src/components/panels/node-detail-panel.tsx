// ============================================================================
// Node Detail Panel -- Rich inspector for selected graph node
// Enhanced: risk gauge, velocity sparkline, flow direction bar, threat level
// ============================================================================

import { useMemo } from 'react'
import { useDashboardStore } from '@/stores/use-dashboard-store'
import { useUIStore } from '@/stores/use-ui-store'
import { MetricCard } from '@/components/shared/metric-card'
import { SeverityBadge, verdictToSeverity } from '@/components/shared/severity-badge'
import { FRAUD_PATTERN_LABELS } from '@/lib/types'
import type { CytoEdge, SSEAgentVerdict } from '@/lib/types'
import { cn, fmtPaisa, fmtNum, truncId, fmtTimestamp } from '@/lib/utils'
import {
  X,
  BarChart3,
  Fingerprint,
  AlertTriangle,
  BrainCircuit,
  ArrowRightLeft,
  Building2,
  Network,
  Users,
  TrendingUp,
  ShieldAlert,
  ArrowDownLeft,
  ArrowUpRight,
  Clock,
  Zap,
} from 'lucide-react'

// -- Status badge colors --
const STATUS_STYLES: Record<string, string> = {
  frozen: 'bg-alert-critical/15 text-alert-critical border-alert-critical/30',
  paused: 'bg-alert-high/15 text-alert-high border-alert-high/30',
  suspicious: 'bg-alert-medium/15 text-alert-medium border-alert-medium/30',
  normal: 'bg-bg-elevated text-text-secondary border-border-subtle',
}

/** Compact risk level gauge with SVG arc */
function RiskGauge({ score, label }: { score: number; label: string }) {
  const pct = Math.min(100, Math.max(0, score * 100))
  const radius = 28
  const circumference = Math.PI * radius // semi-circle
  const offset = circumference * (1 - pct / 100)
  const color =
    pct >= 80 ? '#ef4444' : pct >= 60 ? '#f59e0b' : pct >= 30 ? '#3b82f6' : '#22c55e'

  return (
    <div className="flex flex-col items-center gap-1">
      <svg width="64" height="36" viewBox="0 0 64 36">
        <path
          d="M 4 34 A 28 28 0 0 1 60 34"
          fill="none"
          stroke="oklch(0.25 0.01 260)"
          strokeWidth="4"
          strokeLinecap="round"
        />
        <path
          d="M 4 34 A 28 28 0 0 1 60 34"
          fill="none"
          stroke={color}
          strokeWidth="4"
          strokeLinecap="round"
          strokeDasharray={circumference}
          strokeDashoffset={offset}
          className="transition-all duration-700"
        />
        <text x="32" y="30" textAnchor="middle" fill={color} fontSize="12" fontFamily="monospace" fontWeight="bold">
          {pct.toFixed(0)}
        </text>
      </svg>
      <span className="text-[8px] uppercase tracking-[0.12em] text-text-muted font-semibold">{label}</span>
    </div>
  )
}

/** Compact in/out flow direction bar */
function FlowBar({ inCount, outCount }: { inCount: number; outCount: number }) {
  const total = inCount + outCount
  if (total === 0) return null
  const inPct = (inCount / total) * 100
  return (
    <div className="space-y-1">
      <div className="flex items-center justify-between text-[8px] font-mono text-text-muted">
        <span className="flex items-center gap-0.5">
          <ArrowDownLeft className="w-2.5 h-2.5 text-accent-positive" />IN {inCount}
        </span>
        <span className="flex items-center gap-0.5">
          OUT {outCount}<ArrowUpRight className="w-2.5 h-2.5 text-alert-high" />
        </span>
      </div>
      <div className="flex h-1.5 rounded-full overflow-hidden bg-bg-deep">
        <div className="bg-accent-positive/80 transition-all duration-500" style={{ width: `${inPct}%` }} />
        <div className="bg-alert-high/60 flex-1" />
      </div>
    </div>
  )
}

/** Threat level from fraud counts */
function computeThreatLevel(fraudCounts: Record<number, number>, totalTxns: number): { level: string; score: number; color: string } {
  const fraudTxns = Object.values(fraudCounts).reduce((a, b) => a + b, 0)
  const patternTypes = Object.keys(fraudCounts).length
  const ratio = totalTxns > 0 ? fraudTxns / totalTxns : 0
  const score = Math.min(1, ratio * 2 + patternTypes * 0.15)
  if (score >= 0.7) return { level: 'CRITICAL', score, color: 'text-alert-critical' }
  if (score >= 0.45) return { level: 'HIGH', score, color: 'text-alert-high' }
  if (score >= 0.2) return { level: 'MEDIUM', score, color: 'text-alert-medium' }
  return { level: 'LOW', score, color: 'text-alert-low' }
}

export function NodeDetailPanel() {
  const selectedNodeId = useUIStore((s) => s.selectedNodeId)
  const setSelectedNode = useUIStore((s) => s.setSelectedNode)
  const graphNodes = useDashboardStore((s) => s.graphNodes)
  const graphEdges = useDashboardStore((s) => s.graphEdges)
  const agentLog = useDashboardStore((s) => s.agentLog)

  // Find the selected node
  const node = useMemo(
    () => graphNodes.find((n) => n.data.id === selectedNodeId),
    [graphNodes, selectedNodeId],
  )

  // Connected edges (inbound + outbound)
  const connectedEdges = useMemo(() => {
    if (!selectedNodeId) return []
    return graphEdges.filter(
      (e) => e.data.source === selectedNodeId || e.data.target === selectedNodeId,
    )
  }, [graphEdges, selectedNodeId])

  // Quick metrics
  const metrics = useMemo(() => {
    const inbound = connectedEdges.filter((e) => e.data.target === selectedNodeId)
    const outbound = connectedEdges.filter((e) => e.data.source === selectedNodeId)
    const totalVolume = connectedEdges.reduce((sum, e) => sum + e.data.amount_paisa, 0)
    return {
      totalTxns: connectedEdges.length,
      totalVolume,
      inboundCount: inbound.length,
      outboundCount: outbound.length,
    }
  }, [connectedEdges, selectedNodeId])

  // Fraud indicator counts
  const fraudCounts = useMemo(() => {
    const counts: Record<number, number> = {}
    for (const e of connectedEdges) {
      const label = e.data.fraud_label
      if (label > 0) {
        counts[label] = (counts[label] ?? 0) + 1
      }
    }
    return counts
  }, [connectedEdges])

  // Connected accounts (counterparties)
  const counterparties = useMemo(() => {
    if (!selectedNodeId) return []
    const map = new Map<string, { txnCount: number; volume: number; hasFraud: boolean }>()
    for (const e of connectedEdges) {
      const cp = e.data.source === selectedNodeId ? e.data.target : e.data.source
      const existing = map.get(cp) ?? { txnCount: 0, volume: 0, hasFraud: false }
      existing.txnCount++
      existing.volume += e.data.amount_paisa
      if (e.data.fraud_label > 0) existing.hasFraud = true
      map.set(cp, existing)
    }
    return [...map.entries()]
      .sort((a, b) => b[1].txnCount - a[1].txnCount)
      .slice(0, 10)
  }, [connectedEdges, selectedNodeId])

  // Unique device fingerprints
  const fingerprints = useMemo(() => {
    const set = new Set<string>()
    for (const e of connectedEdges) {
      if (e.data.device_fingerprint) set.add(e.data.device_fingerprint)
    }
    return [...set]
  }, [connectedEdges])

  // Agent verdicts for this node
  const verdicts = useMemo(() => {
    if (!selectedNodeId) return []
    return agentLog.filter(
      (entry) =>
        entry.type === 'verdict' &&
        (entry.data as SSEAgentVerdict).node_id === selectedNodeId,
    )
  }, [agentLog, selectedNodeId])

  // Recent transactions (20 most recent)
  const recentTxns = useMemo(() => {
    return [...connectedEdges]
      .sort((a, b) => b.data.timestamp - a.data.timestamp)
      .slice(0, 20)
  }, [connectedEdges])

  if (!selectedNodeId || !node) return null

  const status = node.data.status
  const accountType = node.data.account_type ?? 'UNKNOWN'
  const communityId = node.data.community_id ?? -1
  const totalVolumePaisa = node.data.total_volume_paisa ?? metrics.totalVolume
  const firstSeen = node.data.first_seen
  const lastSeen = node.data.last_seen
  const threat = computeThreatLevel(fraudCounts, metrics.totalTxns)

  return (
    <div className="w-80 h-full bg-bg-surface border-l border-border-default overflow-y-auto shrink-0 animate-slide-in-right relative">
      {/* Subtle left edge glow */}
      <div className="absolute top-0 left-0 bottom-0 w-px bg-gradient-to-b from-accent-primary/0 via-accent-primary/30 to-accent-primary/0" />

      {/* Header */}
      <div className="sticky top-0 bg-bg-surface/95 backdrop-blur-sm border-b border-border-subtle p-3 flex items-center justify-between z-10">
        <div className="flex items-center gap-2 min-w-0">
          <div className={cn(
            'w-2.5 h-2.5 rounded-full shrink-0 ring-2 ring-offset-1 ring-offset-bg-surface',
            status === 'frozen' && 'bg-alert-critical ring-alert-critical/30',
            status === 'paused' && 'bg-alert-high ring-alert-high/30',
            status === 'suspicious' && 'bg-alert-medium ring-alert-medium/30',
            status === 'normal' && 'bg-accent-primary ring-accent-primary/30',
          )} />
          <span className="text-xs font-mono text-text-primary truncate">
            {truncId(selectedNodeId, 16)}
          </span>
        </div>
        <button
          onClick={() => setSelectedNode(null)}
          className="text-text-muted hover:text-text-primary hover:bg-bg-elevated rounded-md p-1 transition-all duration-200"
          title="Close panel"
        >
          <X className="w-3.5 h-3.5" />
        </button>
      </div>

      <div className="p-3 space-y-5">
        {/* Account Header with Status + Type + Community */}
        <div className="animate-fade-in">
          <div className="flex items-center gap-2 flex-wrap">
            <span className={cn(
              'inline-flex items-center px-2.5 py-0.5 text-[9px] font-bold uppercase tracking-[0.12em] rounded-md border',
              STATUS_STYLES[status] ?? STATUS_STYLES.normal,
            )}>
              {status}
            </span>
            <span className="inline-flex items-center gap-1 px-2 py-0.5 text-[9px] font-bold uppercase tracking-wider rounded-md bg-accent-primary/10 text-accent-primary border border-accent-primary/20">
              <Building2 className="w-2.5 h-2.5" />
              {accountType}
            </span>
            {communityId >= 0 && (
              <span className="inline-flex items-center gap-1 px-2 py-0.5 text-[9px] font-semibold uppercase tracking-wider rounded-md bg-bg-elevated text-text-secondary border border-border-subtle">
                <Network className="w-2.5 h-2.5" />
                Community {communityId}
              </span>
            )}
          </div>
          {(firstSeen || lastSeen) && (
            <div className="mt-2 flex gap-4 text-[8px] font-mono text-text-muted tabular-nums">
              {firstSeen && <span>First: {fmtTimestamp(firstSeen)}</span>}
              {lastSeen && <span>Last: {fmtTimestamp(lastSeen)}</span>}
            </div>
          )}
        </div>

        {/* Quick Metrics + Risk Gauge */}
        <div className="animate-fade-in" style={{ animationDelay: '50ms' }}>
          <SectionHeader icon={BarChart3} label="Metrics" />
          <div className="flex items-start gap-3">
            <div className="flex-1 grid grid-cols-2 gap-2">
              <MetricCard label="Total Txns" value={fmtNum(metrics.totalTxns)} />
              <MetricCard label="Volume" value={fmtPaisa(totalVolumePaisa)} />
            </div>
            <RiskGauge score={threat.score} label={threat.level} />
          </div>
          <div className="mt-2">
            <FlowBar inCount={metrics.inboundCount} outCount={metrics.outboundCount} />
          </div>
          {/* Threat summary */}
          {Object.keys(fraudCounts).length > 0 && (
            <div className={cn(
              'mt-2 flex items-center gap-2 px-2 py-1.5 rounded-md border text-[9px] font-mono',
              threat.level === 'CRITICAL' && 'bg-alert-critical/10 border-alert-critical/30',
              threat.level === 'HIGH' && 'bg-alert-high/10 border-alert-high/30',
              threat.level === 'MEDIUM' && 'bg-alert-medium/10 border-alert-medium/30',
              threat.level === 'LOW' && 'bg-alert-low/10 border-alert-low/30',
            )}>
              <ShieldAlert className={cn('w-3 h-3 shrink-0', threat.color)} />
              <span className={cn('font-bold', threat.color)}>{threat.level}</span>
              <span className="text-text-muted">
                {Object.values(fraudCounts).reduce((a, b) => a + b, 0)} fraud txns across {Object.keys(fraudCounts).length} pattern{Object.keys(fraudCounts).length > 1 ? 's' : ''}
              </span>
            </div>
          )}
        </div>

        {/* Fraud Indicators */}
        {Object.keys(fraudCounts).length > 0 && (
          <div className="animate-fade-in" style={{ animationDelay: '100ms' }}>
            <SectionHeader icon={AlertTriangle} label="Fraud Classification" />
            <div className="space-y-1.5">
              {Object.entries(fraudCounts).map(([label, count]) => (
                <div
                  key={label}
                  className="flex items-center justify-between bg-alert-critical/5 border border-alert-critical/20 rounded-md px-2.5 py-1.5 card-hover transition-all duration-200"
                >
                  <span className="text-[10px] font-medium text-alert-critical">
                    {FRAUD_PATTERN_LABELS[Number(label)] ?? `Type ${label}`}
                  </span>
                  <span className="text-[10px] font-mono font-bold tabular-nums text-alert-critical">
                    {count}
                  </span>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Connected Accounts (clickable) */}
        {counterparties.length > 0 && (
          <div className="animate-fade-in" style={{ animationDelay: '125ms' }}>
            <SectionHeader icon={Users} label={`Connected Accounts (${counterparties.length})`} />
            <div className="space-y-1">
              {counterparties.map(([cpId, info]) => (
                <button
                  key={cpId}
                  onClick={() => setSelectedNode(cpId)}
                  className="w-full flex items-center gap-2 bg-bg-elevated/70 border border-border-subtle rounded-md px-2.5 py-1.5 card-hover transition-all duration-200 text-left"
                >
                  <span className="text-[10px] font-mono text-text-secondary truncate flex-1">
                    {truncId(cpId, 12)}
                  </span>
                  <span className="text-[9px] font-mono tabular-nums text-text-muted">{info.txnCount} txns</span>
                  <span className="text-[9px] font-mono tabular-nums text-text-muted">{fmtPaisa(info.volume)}</span>
                  {info.hasFraud && <span className="w-1.5 h-1.5 rounded-full bg-alert-critical shrink-0" />}
                </button>
              ))}
            </div>
          </div>
        )}

        {/* Agent Verdicts */}
        {verdicts.length > 0 && (
          <div className="animate-fade-in" style={{ animationDelay: '150ms' }}>
            <SectionHeader icon={BrainCircuit} label={`Agent Verdicts (${verdicts.length})`} />
            <div className="space-y-2">
              {verdicts.slice(-5).map((entry) => {
                const d = entry.data as SSEAgentVerdict
                return (
                  <div
                    key={entry.id}
                    className="bg-bg-elevated/70 border border-border-subtle rounded-md p-2.5 card-hover transition-all duration-200"
                  >
                    <div className="flex items-center justify-between mb-1.5">
                      <SeverityBadge severity={verdictToSeverity(d.verdict)} />
                      <span className="text-[9px] font-mono tabular-nums text-text-muted">
                        {(d.confidence * 100).toFixed(0)}%
                      </span>
                    </div>
                    {d.fraud_typology && (
                      <div className="text-[9px] text-accent-primary font-mono mb-1">{d.fraud_typology}</div>
                    )}
                    {d.reasoning_summary && (
                      <p className="text-[10px] text-text-secondary leading-relaxed">
                        {d.reasoning_summary}
                      </p>
                    )}
                    {d.evidence_cited && d.evidence_cited.length > 0 && (
                      <div className="flex flex-wrap gap-1 mt-1.5">
                        {d.evidence_cited.slice(0, 5).map((e, i) => (
                          <span key={i} className="text-[7px] font-mono px-1 py-0.5 rounded bg-bg-overlay border border-border-subtle text-text-muted">
                            {e}
                          </span>
                        ))}
                      </div>
                    )}
                    <div className="flex gap-3 mt-1.5 text-[8px] font-mono text-text-muted/60">
                      <span>{d.recommended_action}</span>
                      <span>{d.thinking_steps} steps</span>
                      <span>{d.total_duration_ms?.toFixed(0)}ms</span>
                    </div>
                  </div>
                )
              })}
            </div>
          </div>
        )}

        {/* Device Fingerprints */}
        {fingerprints.length > 0 && (
          <div className="animate-fade-in" style={{ animationDelay: '200ms' }}>
            <SectionHeader icon={Fingerprint} label={`Device Fingerprints (${fingerprints.length})`} />
            <div className="space-y-1">
              {fingerprints.map((fp) => (
                <div
                  key={fp}
                  className="text-[10px] font-mono text-text-secondary bg-bg-elevated/70 rounded-md px-2.5 py-1.5 border border-border-subtle card-hover transition-all duration-200"
                >
                  {fp}
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Recent Transactions */}
        {recentTxns.length > 0 && (
          <div className="animate-fade-in" style={{ animationDelay: '250ms' }}>
            <SectionHeader icon={ArrowRightLeft} label={`Recent Transactions (${recentTxns.length})`} />
            <div className="space-y-1">
              {recentTxns.map((e: CytoEdge) => {
                const isOutbound = e.data.source === selectedNodeId
                const counterparty = isOutbound ? e.data.target : e.data.source
                return (
                  <div
                    key={e.data.id}
                    className="flex items-center gap-1.5 bg-bg-elevated/70 border border-border-subtle rounded-md px-2.5 py-1.5 card-hover transition-all duration-200"
                  >
                    <span className={cn(
                      'text-[9px] font-bold w-3 shrink-0',
                      isOutbound ? 'text-alert-high' : 'text-accent-positive',
                    )}>
                      {isOutbound ? '\u2192' : '\u2190'}
                    </span>
                    <span className="text-[10px] font-mono text-text-secondary truncate flex-1">
                      {truncId(counterparty, 10)}
                    </span>
                    <span className="text-[10px] font-mono tabular-nums text-text-primary shrink-0">
                      {fmtPaisa(e.data.amount_paisa)}
                    </span>
                    <span className="text-[9px] tabular-nums text-text-muted shrink-0">
                      {fmtTimestamp(e.data.timestamp)}
                    </span>
                    {e.data.fraud_label > 0 && (
                      <span className="w-1.5 h-1.5 rounded-full bg-alert-critical ring-2 ring-alert-critical/20 shrink-0" title={e.data.fraud_label_name ?? FRAUD_PATTERN_LABELS[e.data.fraud_label]} />
                    )}
                  </div>
                )
              })}
            </div>
          </div>
        )}
      </div>
    </div>
  )
}

/** Reusable section header with icon */
function SectionHeader({ icon: Icon, label }: { icon: typeof X; label: string }) {
  return (
    <h4 className="flex items-center gap-1.5 text-[10px] text-text-muted uppercase tracking-[0.12em] font-medium mb-2.5">
      <Icon className="w-3.5 h-3.5 text-accent-primary/60" />
      {label}
    </h4>
  )
}

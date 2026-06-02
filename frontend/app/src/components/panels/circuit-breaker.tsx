// ============================================================================
// Circuit Breaker Panel -- Freeze orders + breaker metrics + consensus detail
// ============================================================================

import { useDashboardStore } from '@/stores/use-dashboard-store'
import { useCircuitBreakerStatus } from '@/hooks/use-api'
import { GaugeBar } from '@/components/shared/gauge-bar'
import { fmtNum, fmtOptionalTimestamp, truncId } from '@/lib/utils'
import { useEffect, useState } from 'react'
import { Zap, ShieldOff, Ban, PauseCircle, AlertTriangle, ShieldCheck, Clock, Loader2 } from 'lucide-react'
import { cn } from '@/lib/utils'
import type { LucideIcon } from 'lucide-react'
import type { FreezeOrder } from '@/lib/types'

export function CircuitBreakerPanel() {
  const frozenCount = useDashboardStore((s) => s.frozenCount)
  const pendingAlerts = useDashboardStore((s) => s.pendingAlerts)
  const bannedDevices = useDashboardStore((s) => s.bannedDevices)
  const routingPausedNodes = useDashboardStore((s) => s.routingPausedNodes)
  const freezeOrders = useDashboardStore((s) => s.freezeOrders)
  const setCircuitBreaker = useDashboardStore((s) => s.setCircuitBreaker)

  const { data: cbData, isError } = useCircuitBreakerStatus()
  const hasBreakerSnapshot = !!cbData

  useEffect(() => {
    if (cbData) {
      setCircuitBreaker(
        cbData.freeze_orders,
        cbData.snapshot as Record<string, unknown>,
        cbData.agent_breaker as Record<string, unknown>,
      )
    }
  }, [cbData, setCircuitBreaker])

  return (
    <div className="flex flex-col h-full">
      <div className="flex items-center justify-between px-3 py-1.5 border-b border-border-subtle shrink-0 bg-bg-surface/50">
        <div className="flex items-center gap-2">
          <Zap className="w-3.5 h-3.5 text-alert-high" />
          <span className="text-[10px] font-semibold uppercase tracking-[0.12em] text-text-secondary">
            Circuit Breaker
          </span>
        </div>
        <span className={cn(
          'text-[9px] font-bold uppercase tracking-wider',
          hasBreakerSnapshot && frozenCount > 0
            ? 'text-alert-critical animate-pulse'
            : hasBreakerSnapshot
              ? 'text-alert-low'
              : isError
                ? 'text-alert-high'
                : 'text-text-muted',
        )}>
          {hasBreakerSnapshot && frozenCount > 0
            ? 'Active'
            : hasBreakerSnapshot
              ? 'Synced'
              : isError
                ? 'Unavailable'
                : 'Syncing'}
        </span>
      </div>
      <div className="flex-1 overflow-y-auto p-2.5 space-y-3">
        {/* Metrics */}
        <div className="grid grid-cols-2 gap-1.5">
          <MetricRow icon={ShieldOff} label="Frozen Nodes" value={hasBreakerSnapshot ? frozenCount : null} danger={hasBreakerSnapshot && frozenCount > 0} />
          <MetricRow icon={AlertTriangle} label="Pending Alerts" value={hasBreakerSnapshot ? pendingAlerts : null} warning={hasBreakerSnapshot && pendingAlerts > 5} />
          <MetricRow icon={Ban} label="Devices Banned" value={hasBreakerSnapshot ? bannedDevices : null} />
          <MetricRow icon={PauseCircle} label="Routing Paused" value={hasBreakerSnapshot ? routingPausedNodes : null} warning={hasBreakerSnapshot && routingPausedNodes > 0} />
        </div>

        {/* Freeze pressure gauge */}
        <GaugeBar
          label="Freeze Pressure"
          value={hasBreakerSnapshot ? Math.min(frozenCount * 10, 100) : null}
          color={frozenCount > 5 ? 'critical' : frozenCount > 2 ? 'high' : 'accent'}
        />

        {/* Freeze orders with consensus breakdown */}
        {freezeOrders.length > 0 && (
          <div className="mt-1 space-y-2">
            <div className="text-[9px] text-text-muted uppercase tracking-[0.12em] font-semibold">
              Active Freeze Orders ({freezeOrders.length})
            </div>
            {freezeOrders.map((order) => (
              <FreezeOrderCard key={order.node_id} order={order} />
            ))}
          </div>
        )}

        {!hasBreakerSnapshot && !isError && (
          <div className="flex flex-col items-center justify-center py-4 gap-2 animate-fade-in">
            <Loader2 className="w-5 h-5 animate-spin text-text-muted/60" />
            <p className="text-text-muted text-[10px] uppercase tracking-[0.12em]">
              Syncing breaker snapshot
            </p>
          </div>
        )}

        {isError && (
          <div className="flex flex-col items-center justify-center py-4 gap-2 animate-fade-in">
            <AlertTriangle className="w-5 h-5 text-alert-high/70" />
            <p className="text-text-muted text-[10px] uppercase tracking-[0.12em]">
              Breaker snapshot unavailable
            </p>
          </div>
        )}

        {hasBreakerSnapshot && freezeOrders.length === 0 && frozenCount === 0 && (
          <div className="flex flex-col items-center justify-center py-4 gap-2 animate-fade-in">
            <ShieldCheck className="w-5 h-5 text-alert-low/40" />
            <p className="text-text-muted text-[10px] uppercase tracking-[0.12em]">
              No active freeze orders
            </p>
          </div>
        )}
      </div>
    </div>
  )
}

function FreezeOrderCard({ order }: { order: FreezeOrder }) {
  const [nowSec, setNowSec] = useState(0)
  const hasSubs = [order.ml_risk_score, order.gnn_risk_score, order.graph_evidence_score]
    .some((value) => value != null && Number.isFinite(value) && value >= 0)
  const hasFreezeTime = typeof order.freeze_timestamp === 'number' && Number.isFinite(order.freeze_timestamp) && order.freeze_timestamp > 0
  const shouldTick = Boolean(order.ttl_seconds && order.ttl_seconds > 0 && hasFreezeTime)
  const ttlRemaining = shouldTick && nowSec > 0
    ? Math.max(0, order.ttl_seconds! - (nowSec - order.freeze_timestamp!))
    : null

  useEffect(() => {
    if (!shouldTick) return
    const updateNow = () => setNowSec(Date.now() / 1000)
    updateNow()
    const interval = window.setInterval(updateNow, 1000)
    return () => window.clearInterval(interval)
  }, [shouldTick])

  return (
    <div className="rounded-md border border-alert-critical/20 bg-alert-critical/[0.03] p-2.5 space-y-2 card-hover transition-all duration-200">
      {/* Header */}
      <div className="flex items-center gap-2">
        <span className="text-[10px] font-mono font-semibold text-alert-critical">
          {truncId(order.node_id, 12)}
        </span>
        <span className="text-[9px] font-mono tabular-nums text-text-primary">
          score: {order.consensus_score?.toFixed(3) ?? '\u2014'}
        </span>
        {order.freeze_timestamp != null && (
          <span className="ml-auto flex items-center gap-0.5 text-[8px] font-mono text-text-muted">
            <Clock className="w-2.5 h-2.5" />
            {fmtOptionalTimestamp(order.freeze_timestamp)}
          </span>
        )}
      </div>

      {/* Consensus Sub-scores */}
      {hasSubs && (
        <div className="space-y-1">
          <div className="text-[8px] font-semibold uppercase tracking-wider text-text-muted">Consensus Breakdown</div>
          <div className="flex gap-1.5">
            <ScoreBar label="ML" value={order.ml_risk_score} color="bg-[#00579C]" />
            <ScoreBar label="GNN" value={order.gnn_risk_score} color="bg-[#00579C]" />
            <ScoreBar label="Graph" value={order.graph_evidence_score} color="bg-[#00579C]" />
          </div>
        </div>
      )}

      {/* Reason */}
      <div className="text-[9px] text-text-secondary">{order.reason}</div>

      {/* TTL countdown */}
      {ttlRemaining != null && order.ttl_seconds && order.ttl_seconds > 0 && (
        <div className="flex items-center gap-2">
          <div className="flex-1 h-1 bg-bg-overlay rounded-full overflow-hidden">
            <div
              className="h-full bg-alert-high/60 rounded-full transition-all"
              style={{ width: `${(ttlRemaining / order.ttl_seconds) * 100}%` }}
            />
          </div>
          <span className="text-[8px] font-mono tabular-nums text-text-muted">
            TTL: {Math.floor(ttlRemaining)}s
          </span>
        </div>
      )}
    </div>
  )
}

function ScoreBar({ label, value, color }: { label: string; value?: number; color: string }) {
  const available = value != null && Number.isFinite(value) && value >= 0
  const pct = available ? Math.min(100, Math.max(0, value * 100)) : 0
  return (
    <div className="flex-1">
      <div className="flex items-center justify-between text-[7px] font-mono mb-0.5">
        <span className="text-text-muted">{label}</span>
        <span className="tabular-nums text-text-secondary">{available ? `${pct.toFixed(0)}%` : 'N/A'}</span>
      </div>
      <div className="h-1.5 bg-bg-overlay rounded-full overflow-hidden">
        <div className={cn('h-full rounded-full transition-all duration-500', color)} style={{ width: `${pct}%` }} />
      </div>
    </div>
  )
}

function MetricRow({
  icon: Icon,
  label,
  value,
  danger = false,
  warning = false,
}: {
  icon: LucideIcon
  label: string
  value: number | null
  danger?: boolean
  warning?: boolean
}) {
  return (
    <div className="flex items-center gap-2 p-2 rounded-md bg-bg-elevated/60 border border-border-subtle/50">
      <Icon className={cn(
        'w-3.5 h-3.5 shrink-0',
        danger ? 'text-alert-critical' : warning ? 'text-alert-medium' : 'text-text-muted',
      )} />
      <div className="flex flex-col min-w-0">
        <span className="text-[9px] text-text-muted uppercase tracking-wider truncate">{label}</span>
        <span
          className={cn(
            'text-sm font-mono font-bold tabular-nums',
            danger ? 'text-alert-critical' : warning ? 'text-alert-medium' : 'text-text-primary',
          )}
        >
          {value == null ? 'n/a' : fmtNum(value)}
        </span>
      </div>
    </div>
  )
}

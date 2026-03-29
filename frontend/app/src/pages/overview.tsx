// ============================================================================
// Overview Page -- Stat strip + Sigma Graph + Node Detail Panel + Right Sidebar
// Enhanced: richer KPIs, live uptime, accent pulse, throughput display
// ============================================================================

import { useState, useEffect } from 'react'
import { useDashboardStore } from '@/stores/use-dashboard-store'
import { useSnapshot } from '@/hooks/use-api'
import SigmaGraph from '@/components/panels/sigma-graph'
import { NodeDetailPanel } from '@/components/panels/node-detail-panel'
import { RightSidebar } from '@/components/layout/right-sidebar'
import { useUIStore } from '@/stores/use-ui-store'
import { fmtNum } from '@/lib/utils'
import { cn } from '@/lib/utils'
import {
  ArrowRightLeft,
  Brain,
  Bell,
  CircleDot,
  GitBranch,
  AlertTriangle,
  Gauge,
  Thermometer,
  Activity,
  Shield,
  TrendingUp,
  Eye,
} from 'lucide-react'
import type { LucideIcon } from 'lucide-react'

/* ── Stat definition ── */
interface StatDef {
  label: string
  value: string
  accent: string
  iconAccent: string
  Icon: LucideIcon
  pulse?: boolean
}

/* ── Live throughput counter ── */
function useThroughput() {
  const [tps, setTps] = useState(0)
  const eventsIngested = useDashboardStore((s) => s.orchestrator?.events_ingested ?? 0)
  const [prevCount, setPrevCount] = useState(eventsIngested)
  const [prevTime, setPrevTime] = useState(Date.now())

  useEffect(() => {
    const now = Date.now()
    const elapsed = (now - prevTime) / 1000
    if (elapsed > 0.5) {
      const delta = eventsIngested - prevCount
      setTps(Math.round(delta / elapsed))
      setPrevCount(eventsIngested)
      setPrevTime(now)
    }
  }, [eventsIngested, prevCount, prevTime])

  return tps
}

/* ── Compact horizontal KPI strip shown above the graph ── */
function StatStrip() {
  const sseOrch = useDashboardStore((s) => s.orchestrator)
  const sseHw   = useDashboardStore((s) => s.hardware)
  const graphMetrics = useDashboardStore((s) => s.graphMetrics)
  const { data: snap } = useSnapshot()
  const tps = useThroughput()

  const orch  = sseOrch ?? (snap as any)?.orchestrator ?? null
  const hw    = sseHw   ?? (snap as any)?.hardware     ?? null
  const graphSz = useDashboardStore.getState().graphSize ?? (snap as any)?.graph?.graph ?? null
  const cb    = (snap as any)?.circuit_breaker          ?? null

  const gpuUtil = hw?.gpu_utilization_pct ?? 0

  const stats: StatDef[] = [
    {
      label: 'Transactions',
      value: fmtNum(orch?.events_ingested ?? 0),
      accent: 'text-cyan-400',
      iconAccent: 'text-cyan-500/70',
      Icon: ArrowRightLeft,
    },
    {
      label: 'Throughput',
      value: `${tps}/s`,
      accent: tps > 0 ? 'text-green-400' : 'text-text-muted',
      iconAccent: tps > 0 ? 'text-green-500/70' : 'text-text-muted/50',
      Icon: Activity,
      pulse: tps > 0,
    },
    {
      label: 'ML Inferences',
      value: fmtNum(orch?.ml_inferences ?? 0),
      accent: 'text-violet-400',
      iconAccent: 'text-violet-500/70',
      Icon: Brain,
    },
    {
      label: 'Alerts',
      value: fmtNum(orch?.alerts_routed ?? 0),
      accent: 'text-amber-400',
      iconAccent: 'text-amber-500/70',
      Icon: Bell,
    },
    {
      label: 'Nodes',
      value: fmtNum(graphSz?.nodes ?? 0),
      accent: 'text-blue-400',
      iconAccent: 'text-blue-500/70',
      Icon: CircleDot,
    },
    {
      label: 'Edges',
      value: fmtNum(graphSz?.edges ?? 0),
      accent: 'text-blue-300',
      iconAccent: 'text-blue-400/70',
      Icon: GitBranch,
    },
    {
      label: 'Mule Nets',
      value: fmtNum(graphMetrics?.mule_detections ?? 0),
      accent: (graphMetrics?.mule_detections ?? 0) > 0 ? 'text-rose-400' : 'text-text-muted',
      iconAccent: (graphMetrics?.mule_detections ?? 0) > 0 ? 'text-rose-500/70' : 'text-text-muted/50',
      Icon: AlertTriangle,
      pulse: (graphMetrics?.mule_detections ?? 0) > 0,
    },
    {
      label: 'Frozen',
      value: fmtNum(cb?.pending_alerts ?? 0),
      accent: (cb?.pending_alerts ?? 0) > 0 ? 'text-red-400' : 'text-text-muted',
      iconAccent: (cb?.pending_alerts ?? 0) > 0 ? 'text-red-500/70' : 'text-text-muted/50',
      Icon: Shield,
    },
    {
      label: 'Fraud Ratio',
      value: (() => {
        const edges = useDashboardStore.getState().graphEdges
        if (!edges || edges.length === 0) return '0%'
        const fraud = edges.filter((e: any) => (e.data.fraud_label ?? 0) > 0).length
        return `${((fraud / edges.length) * 100).toFixed(1)}%`
      })(),
      accent: 'text-rose-400',
      iconAccent: 'text-rose-500/70',
      Icon: TrendingUp,
    },
    {
      label: 'Suspicious',
      value: (() => {
        const nodes = useDashboardStore.getState().graphNodes
        return fmtNum(nodes.filter((n: any) => n.data.status === 'suspicious').length)
      })(),
      accent: 'text-yellow-400',
      iconAccent: 'text-yellow-500/70',
      Icon: Eye,
    },
    {
      label: 'GPU',
      value: `${gpuUtil}%`,
      accent: gpuUtil > 85 ? 'text-red-400' : gpuUtil > 60 ? 'text-amber-400' : 'text-green-400',
      iconAccent: gpuUtil > 85 ? 'text-red-500/70' : gpuUtil > 60 ? 'text-amber-500/70' : 'text-green-500/70',
      Icon: Gauge,
    },
    {
      label: 'VRAM',
      value: `${((hw?.gpu_vram_used_mb ?? 0) / 1024).toFixed(1)}G`,
      accent: 'text-green-300',
      iconAccent: 'text-green-400/70',
      Icon: Thermometer,
    },
  ]

  return (
    <div className="flex items-stretch border-b border-border-subtle bg-bg-surface/80 backdrop-blur-sm shrink-0 overflow-x-auto animate-fade-in">
      {stats.map(({ label, value, accent, iconAccent, Icon, pulse }) => (
        <div
          key={label}
          className="group flex items-center gap-2 px-3 py-1.5 border-r border-border-subtle last:border-r-0 shrink-0 card-hover cursor-default"
        >
          <div className={cn(
            'flex items-center justify-center w-6 h-6 rounded-md bg-bg-elevated/60 border border-border-subtle group-hover:border-border-default transition-colors',
            pulse && 'animate-data-pulse',
          )}>
            <Icon className={cn('w-3 h-3', iconAccent)} strokeWidth={1.75} />
          </div>
          <div className="flex flex-col items-start">
            <span className={cn('text-[13px] font-mono font-semibold tabular-nums leading-tight', accent)}>
              {value}
            </span>
            <span className="text-[8px] uppercase tracking-[0.12em] text-text-muted">
              {label}
            </span>
          </div>
        </div>
      ))}
    </div>
  )
}

export function OverviewPage() {
  const selectedNodeId = useUIStore((s) => s.selectedNodeId)

  return (
    <div className="flex flex-col h-full">
      <StatStrip />
      <div className="flex flex-1 min-h-0">
        <div className="flex-1 min-w-0 h-full bg-bg-deep">
          <SigmaGraph />
        </div>
        {selectedNodeId && <NodeDetailPanel />}
        <RightSidebar />
      </div>
    </div>
  )
}

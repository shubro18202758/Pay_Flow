// ============================================================================
// Overview Page -- Stat strip + Sigma Graph + Node Detail Panel + Right Sidebar
// Enhanced: richer KPIs, live uptime, accent pulse, throughput display
// ============================================================================

import { useEffect, useMemo, useRef, useState } from 'react'
import { useDashboardStore } from '@/stores/use-dashboard-store'
import {
  useCreateEvidencePackage,
  useLaunchPS3Scenario,
  usePS3Readiness,
  useRefreshIntel,
  useSnapshot,
  useSimulateIntelSignal,
} from '@/hooks/use-api'
import SigmaGraph from '@/components/panels/sigma-graph'
import { PreFraudIntelBrief } from '@/components/panels/pre-fraud-intel-brief'
import { NodeDetailPanel } from '@/components/panels/node-detail-panel'
import { RightSidebar } from '@/components/layout/right-sidebar'
import { useRoleAccess } from '@/hooks/use-rbac'
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
  FileText,
  Play,
  Radar,
  Sparkles,
} from 'lucide-react'
import type { LucideIcon } from 'lucide-react'
import type { PS3ScenarioId } from '@/lib/types'

/* ── Stat definition ── */
interface StatDef {
  label: string
  value: string
  accent: string
  iconAccent: string
  Icon: LucideIcon
  pulse?: boolean
  available?: boolean
}

/* ── Live throughput counter ── */
function useThroughput() {
  const [tps, setTps] = useState(0)
  const eventsIngested = useDashboardStore((s) => s.orchestrator?.events_ingested ?? 0)
  const previous = useRef({ count: eventsIngested, time: 0 })

  useEffect(() => {
    const now = Date.now()
    if (previous.current.time === 0) {
      previous.current = { count: eventsIngested, time: now }
      return
    }
    const elapsed = (now - previous.current.time) / 1000
    if (elapsed > 0.5) {
      const delta = eventsIngested - previous.current.count
      setTps(Math.round(delta / elapsed))
      previous.current = { count: eventsIngested, time: now }
    }
  }, [eventsIngested])

  return tps
}

/* ── Compact horizontal KPI strip shown above the graph ── */
function StatStrip() {
  const sseOrch = useDashboardStore((s) => s.orchestrator)
  const sseHw   = useDashboardStore((s) => s.hardware)
  const graphMetrics = useDashboardStore((s) => s.graphMetrics)
  const graphSize = useDashboardStore((s) => s.graphSize)
  const graphSummary = useDashboardStore((s) => s.graphSummary)
  const sseFrozenCount = useDashboardStore((s) => s.frozenCount)
  const { data: snap } = useSnapshot()
  const tps = useThroughput()

  const orch = sseOrch ?? snap?.orchestrator ?? null
  const hw = sseHw ?? snap?.hardware ?? null
  const graphSz = graphSize ?? snap?.graph?.graph ?? null
  const frozenFromSnapshot = snap?.circuit_breaker?.frozen_count
  const hasCircuitSnapshot = Boolean(snap?.circuit_breaker)
  const frozenCount = typeof frozenFromSnapshot === 'number' ? frozenFromSnapshot : sseFrozenCount
  const hasGraphSummary = Boolean(graphSz) || graphSummary.nodeCount > 0 || graphSummary.edgeCount > 0

  const gpuUtil = hw?.gpu_utilization_pct
  const formatMaybeNum = (value: number | null | undefined) =>
    typeof value === 'number' && Number.isFinite(value) ? fmtNum(value) : 'n/a'
  const unavailableAccent = 'text-text-muted'
  const unavailableIcon = 'text-text-muted/45'

  const stats: StatDef[] = [
    {
      label: 'Transactions',
      value: formatMaybeNum(orch?.events_ingested),
      accent: orch ? 'text-accent-primary' : unavailableAccent,
      iconAccent: orch ? 'text-accent-primary/70' : unavailableIcon,
      Icon: ArrowRightLeft,
      available: Boolean(orch),
    },
    {
      label: 'Throughput',
      value: orch ? `${tps}/s` : 'n/a',
      accent: orch && tps > 0 ? 'text-[#00579C]' : unavailableAccent,
      iconAccent: orch && tps > 0 ? 'text-[#00579C]/70' : unavailableIcon,
      Icon: Activity,
      pulse: Boolean(orch && tps > 0),
      available: Boolean(orch),
    },
    {
      label: 'ML Inferences',
      value: formatMaybeNum(orch?.ml_inferences),
      accent: orch ? 'text-[#00579C]' : unavailableAccent,
      iconAccent: orch ? 'text-[#00579C]/70' : unavailableIcon,
      Icon: Brain,
      available: Boolean(orch),
    },
    {
      label: 'Alerts',
      value: formatMaybeNum(orch?.alerts_routed),
      accent: orch ? 'text-[#DA251C]' : unavailableAccent,
      iconAccent: orch ? 'text-[#DA251C]/70' : unavailableIcon,
      Icon: Bell,
      available: Boolean(orch),
    },
    {
      label: 'Nodes',
      value: formatMaybeNum(graphSz?.nodes),
      accent: graphSz ? 'text-[#00579C]' : unavailableAccent,
      iconAccent: graphSz ? 'text-[#00579C]/70' : unavailableIcon,
      Icon: CircleDot,
      available: Boolean(graphSz),
    },
    {
      label: 'Edges',
      value: formatMaybeNum(graphSz?.edges),
      accent: graphSz ? 'text-[#00579C]' : unavailableAccent,
      iconAccent: graphSz ? 'text-[#00579C]/70' : unavailableIcon,
      Icon: GitBranch,
      available: Boolean(graphSz),
    },
    {
      label: 'Mule Nets',
      value: formatMaybeNum(graphMetrics?.mule_detections),
      accent: graphMetrics && graphMetrics.mule_detections > 0 ? 'text-[#DA251C]' : unavailableAccent,
      iconAccent: graphMetrics && graphMetrics.mule_detections > 0 ? 'text-[#DA251C]/70' : unavailableIcon,
      Icon: AlertTriangle,
      pulse: Boolean(graphMetrics && graphMetrics.mule_detections > 0),
      available: Boolean(graphMetrics),
    },
    {
      label: 'Frozen',
      value: hasCircuitSnapshot ? fmtNum(frozenCount) : 'n/a',
      accent: hasCircuitSnapshot && frozenCount > 0 ? 'text-[#DA251C]' : unavailableAccent,
      iconAccent: hasCircuitSnapshot && frozenCount > 0 ? 'text-[#DA251C]/70' : unavailableIcon,
      Icon: Shield,
      available: hasCircuitSnapshot,
    },
    {
      label: 'Fraud Ratio',
      value: hasGraphSummary
        ? graphSummary.edgeCount === 0
          ? '0%'
          : `${((graphSummary.fraudEdges / graphSummary.edgeCount) * 100).toFixed(1)}%`
        : 'n/a',
      accent: hasGraphSummary ? 'text-[#DA251C]' : unavailableAccent,
      iconAccent: hasGraphSummary ? 'text-[#DA251C]/70' : unavailableIcon,
      Icon: TrendingUp,
      available: hasGraphSummary,
    },
    {
      label: 'Suspicious',
      value: hasGraphSummary ? fmtNum(graphSummary.suspiciousNodes) : 'n/a',
      accent: hasGraphSummary ? 'text-[#DA251C]' : unavailableAccent,
      iconAccent: hasGraphSummary ? 'text-[#DA251C]/70' : unavailableIcon,
      Icon: Eye,
      available: hasGraphSummary,
    },
    {
      label: 'GPU',
      value: typeof gpuUtil === 'number' ? `${gpuUtil}%` : 'n/a',
      accent: typeof gpuUtil === 'number'
        ? gpuUtil > 85 ? 'text-[#DA251C]' : gpuUtil > 60 ? 'text-[#DA251C]' : 'text-[#00579C]'
        : unavailableAccent,
      iconAccent: typeof gpuUtil === 'number'
        ? gpuUtil > 85 ? 'text-[#DA251C]/70' : gpuUtil > 60 ? 'text-[#DA251C]/70' : 'text-[#00579C]/70'
        : unavailableIcon,
      Icon: Gauge,
      available: Boolean(hw),
    },
    {
      label: 'VRAM',
      value: hw ? `${(hw.gpu_vram_used_mb / 1024).toFixed(1)}G` : 'n/a',
      accent: hw ? 'text-[#00579C]' : unavailableAccent,
      iconAccent: hw ? 'text-[#00579C]/70' : unavailableIcon,
      Icon: Thermometer,
      available: Boolean(hw),
    },
  ]

  return (
    <div className="flex items-stretch border-b border-border-subtle bg-bg-surface/80 backdrop-blur-sm shrink-0 overflow-x-auto animate-fade-in">
      {stats.map(({ label, value, accent, iconAccent, Icon, pulse, available }) => (
        <div
          key={label}
          title={available === false ? `${label} awaiting backend telemetry` : undefined}
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

function CommandRail() {
  const access = useRoleAccess()
  const setActiveTab = useUIStore((s) => s.setActiveTab)
  const setActiveCaseId = useUIStore((s) => s.setActiveCaseId)
  const activeCaseId = useUIStore((s) => s.activeCaseId)
  const setLatestEvidencePackage = useUIStore((s) => s.setLatestEvidencePackage)
  const refreshIntel = useRefreshIntel()
  const simulateIntel = useSimulateIntelSignal()
  const launchPS3 = useLaunchPS3Scenario()
  const evidence = useCreateEvidencePackage()
  const { data: readiness } = usePS3Readiness()
  const [status, setStatus] = useState('Continuous stream online')

  const readyCount = useMemo(
    () => readiness?.requirements.filter((item) => item.status === 'ready').length ?? 0,
    [readiness],
  )
  const readinessTotal = readiness?.requirements.length ?? null

  async function primeIntel() {
    if (!access.can('intel:write')) {
      setStatus(`${access.policy.label} cannot refresh preventive intelligence`)
      return
    }
    setStatus('Refreshing external intelligence')
    await refreshIntel.mutateAsync(undefined)
    await simulateIntel.mutateAsync('digital_arrest_mule')
    setStatus('Adaptive playbooks refreshed')
  }

  async function launchCase(scenario: PS3ScenarioId = 'rapid_layering', navigate = true): Promise<string> {
    if (!access.can('case:launch')) {
      setStatus(`${access.policy.label} cannot launch fund-flow cases`)
      return activeCaseId ?? ''
    }
    setStatus('Launching fund-flow trace case')
    const response = await launchPS3.mutateAsync({ scenario, intensity: 'scale', seed: Date.now() % 100000 })
    setActiveCaseId(response.primary_case_id)
    setStatus(`Case ${response.primary_case_id} opened`)
    if (navigate) setActiveTab('investigations')
    return response.primary_case_id
  }

  async function packageEvidence() {
    if (!access.can('evidence:package')) {
      setStatus(`${access.policy.label} cannot generate FIU evidence packages`)
      return
    }
    const targetCaseId = activeCaseId ?? await launchCase('round_tripping', false)
    if (!targetCaseId) return
    setStatus('Generating FIU evidence package')
    const pkg = await evidence.mutateAsync(targetCaseId)
    setLatestEvidencePackage(pkg)
    setStatus(`Evidence package ready for ${targetCaseId}`)
    setActiveTab('investigations')
  }

  const busy = refreshIntel.isPending || simulateIntel.isPending || launchPS3.isPending || evidence.isPending

  return (
    <section className="shrink-0 border-b border-border-default bg-bg-surface px-4 py-2 shadow-sm">
      <div className="flex flex-wrap items-center gap-2">
        <div className="mr-2 min-w-48">
          <div className="text-[10px] font-extrabold uppercase tracking-[0.14em] text-text-primary">
            Fund-Flow Operations Command Rail
          </div>
          <div className="font-mono text-[9px] text-text-muted">
            {access.policy.label} | Preventive intel {'>'} fund trace {'>'} case workbench {'>'} FIU package | {readyCount}/{readinessTotal ?? 'n/a'} ready | {status}
          </div>
        </div>
        <CommandButton
          icon={Sparkles}
          label="Prime Intel"
          busy={busy}
          disabled={!access.can('intel:write')}
          disabledReason={`${access.policy.label} can view intelligence but cannot refresh or simulate signals`}
          onClick={() => void primeIntel()}
        />
        <CommandButton
          icon={Play}
          label="Launch Case"
          busy={busy}
          disabled={!access.can('case:launch')}
          disabledReason={`${access.policy.label} cannot launch fund-flow case drills`}
          onClick={() => void launchCase('rapid_layering')}
        />
        <CommandButton
          icon={FileText}
          label="FIU Package"
          busy={busy}
          disabled={!access.can('evidence:package')}
          disabledReason={`${access.policy.label} cannot generate FIU evidence packages`}
          onClick={() => void packageEvidence()}
        />
        <CommandButton
          icon={Radar}
          label="Intel Radar"
          onClick={() => setActiveTab('pre-fraud-intel')}
        />
      </div>
    </section>
  )
}

function CommandButton({
  icon: Icon,
  label,
  busy,
  disabled,
  disabledReason,
  onClick,
}: {
  icon: LucideIcon
  label: string
  busy?: boolean
  disabled?: boolean
  disabledReason?: string
  onClick: () => void
}) {
  const isDisabled = Boolean(busy || disabled)
  return (
    <button
      onClick={onClick}
      disabled={isDisabled}
      title={disabled && disabledReason ? disabledReason : label}
      className="inline-flex h-8 items-center gap-2 rounded-md border border-border-default bg-bg-surface px-3 text-[9px] font-bold uppercase tracking-[0.12em] text-text-secondary shadow-sm transition-colors hover:border-accent-primary hover:bg-accent-primary/10 hover:text-accent-primary disabled:cursor-not-allowed disabled:opacity-60"
    >
      <Icon className="h-3.5 w-3.5" />
      {label}
    </button>
  )
}

export function OverviewPage() {
  const selectedNodeId = useUIStore((s) => s.selectedNodeId)

  return (
    <div className="flex flex-col h-full">
      <StatStrip />
      <PreFraudIntelBrief variant="overview" />
      <CommandRail />
      <div className="flex flex-1 min-h-0 flex-col overflow-y-auto lg:flex-row lg:overflow-hidden">
        <div className="min-h-[520px] flex-1 bg-bg-deep lg:h-full lg:min-h-0 lg:min-w-0">
          <SigmaGraph />
        </div>
        {selectedNodeId && (
          <div className="hidden lg:block">
            <NodeDetailPanel />
          </div>
        )}
        <div className="hidden lg:flex">
          <RightSidebar />
        </div>
      </div>
    </div>
  )
}

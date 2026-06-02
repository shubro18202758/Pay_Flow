// ============================================================================
// Quick Stats -- Key metrics at a glance (SSE-live with snapshot fallback)
// ============================================================================

import { useDashboardStore } from '@/stores/use-dashboard-store'
import { useCountermeasureProposals, useEscalations, useSnapshot } from '@/hooks/use-api'
import { MetricCard } from '@/components/shared/metric-card'
import { fmtNum } from '@/lib/utils'
import { Gauge, Activity, Cpu, Shield, UserCheck, ClipboardCheck } from 'lucide-react'
import type { SystemSnapshot } from '@/lib/types'

function metric(value: number | null | undefined, digits = 0, suffix = ''): string {
  if (value == null || !Number.isFinite(value)) return 'n/a'
  return `${digits > 0 ? value.toFixed(digits) : fmtNum(value)}${suffix}`
}

export function QuickStats() {
  const sseOrchestrator = useDashboardStore((s) => s.orchestrator)
  const sseHardware     = useDashboardStore((s) => s.hardware)
  const frozenCount     = useDashboardStore((s) => s.frozenCount)
  const threatSim       = useDashboardStore((s) => s.threatSimulation)

  const { data: snap } = useSnapshot()
  const { data: escalations } = useEscalations()
  const { data: countermeasures } = useCountermeasureProposals()

  // Use SSE data when live; fall back to latest REST snapshot
  const snapshot = snap as Partial<SystemSnapshot> | undefined
  const orch = sseOrchestrator ?? snapshot?.orchestrator ?? null
  const hw   = sseHardware    ?? snapshot?.hardware      ?? null
  const circuitBreaker = snapshot?.circuit_breaker as { frozen_count?: number } | undefined
  const snapshotThreatSim = snapshot?.threat_simulation as { active_attacks?: number } | undefined
  const displayFrozenCount = circuitBreaker?.frozen_count ?? frozenCount
  const hasFrozenEvidence = circuitBreaker?.frozen_count != null || frozenCount > 0
  const activeAttacks = threatSim?.active_attacks ?? snapshotThreatSim?.active_attacks
  const pendingEscalations = (escalations ?? []).filter((item) => item.status === 'pending_review').length
  const proposals = countermeasures?.proposals
  const proposedCountermeasures = (proposals ?? []).filter((item) => item.status === 'proposed').length

  return (
    <div className="p-2.5">
      <div className="text-[10px] font-semibold uppercase tracking-[0.12em] text-text-secondary mb-2.5 px-1">
        Quick Stats
      </div>
      <div className="grid grid-cols-2 gap-2">
        <MetricCard
          label="Events/sec"
          value={orch?.events_per_sec != null ? orch.events_per_sec.toFixed(1) : 'n/a'}
          sub={orch ? `${fmtNum(orch.events_ingested)} total` : 'orchestrator syncing'}
          icon={Gauge}
          accent="text-accent-primary"
        />
        <MetricCard
          label="ML Inferences"
          value={metric(orch?.ml_inferences)}
          sub={orch ? `${fmtNum(orch.alerts_routed)} alerts` : 'orchestrator syncing'}
          icon={Activity}
          accent="text-alert-low"
        />
        <MetricCard
          label="GPU Util"
          value={hw ? `${hw.gpu_utilization_pct.toFixed(0)}%` : 'n/a'}
          sub={hw ? `${hw.gpu_vram_used_mb.toFixed(0)} / ${hw.gpu_vram_total_mb.toFixed(0)} MB` : 'hardware telemetry syncing'}
          icon={Cpu}
          accent="text-alert-medium"
        />
        <MetricCard
          label="Frozen Nodes"
          value={hasFrozenEvidence ? displayFrozenCount : 'n/a'}
          sub={activeAttacks != null ? `${activeAttacks} active attacks` : 'threat engine syncing'}
          icon={Shield}
          accent="text-alert-critical"
        />
        <MetricCard
          label="Analyst Queue"
          value={escalations ? pendingEscalations : 'n/a'}
          sub={escalations ? `${escalations.length} total cases` : 'analyst queue syncing'}
          icon={UserCheck}
          accent={pendingEscalations > 0 ? 'text-alert-escalated' : 'text-alert-low'}
        />
        <MetricCard
          label="Countermeasures"
          value={proposals ? proposedCountermeasures : 'n/a'}
          sub={proposals ? `${proposals.length} proposals` : 'countermeasure sync pending'}
          icon={ClipboardCheck}
          accent={proposedCountermeasures > 0 ? 'text-alert-medium' : 'text-alert-low'}
        />
      </div>
    </div>
  )
}

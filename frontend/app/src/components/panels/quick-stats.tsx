// ============================================================================
// Quick Stats -- Key metrics at a glance (SSE-live with snapshot fallback)
// ============================================================================

import { useDashboardStore } from '@/stores/use-dashboard-store'
import { useSnapshot } from '@/hooks/use-api'
import { MetricCard } from '@/components/shared/metric-card'
import { fmtNum } from '@/lib/utils'
import { Gauge, Activity, Cpu, Shield } from 'lucide-react'

export function QuickStats() {
  const sseOrchestrator = useDashboardStore((s) => s.orchestrator)
  const sseHardware     = useDashboardStore((s) => s.hardware)
  const frozenCount     = useDashboardStore((s) => s.frozenCount)
  const threatSim       = useDashboardStore((s) => s.threatSimulation)

  const { data: snap } = useSnapshot()

  // Use SSE data when live; fall back to latest REST snapshot
  const orch = sseOrchestrator ?? (snap as any)?.orchestrator ?? null
  const hw   = sseHardware    ?? (snap as any)?.hardware      ?? null

  return (
    <div className="p-2.5">
      <div className="text-[10px] font-semibold uppercase tracking-[0.12em] text-text-secondary mb-2.5 px-1">
        Quick Stats
      </div>
      <div className="grid grid-cols-2 gap-2">
        <MetricCard
          label="Events/sec"
          value={orch?.events_per_sec != null ? orch.events_per_sec.toFixed(1) : '\u2014'}
          sub={`${fmtNum(orch?.events_ingested ?? 0)} total`}
          icon={Gauge}
          accent="accent-primary"
        />
        <MetricCard
          label="ML Inferences"
          value={fmtNum(orch?.ml_inferences ?? 0)}
          sub={`${fmtNum(orch?.alerts_routed ?? 0)} alerts`}
          icon={Activity}
          accent="accent-positive"
        />
        <MetricCard
          label="GPU Util"
          value={`${(hw?.gpu_utilization_pct ?? 0).toFixed(0)}%`}
          sub={`${hw?.gpu_vram_used_mb?.toFixed(0) ?? '\u2014'} / ${hw?.gpu_vram_total_mb?.toFixed(0) ?? '\u2014'} MB`}
          icon={Cpu}
          accent="alert-medium"
        />
        <MetricCard
          label="Frozen Nodes"
          value={frozenCount}
          sub={`${threatSim?.active_attacks ?? 0} active attacks`}
          icon={Shield}
          accent="alert-critical"
        />
      </div>
    </div>
  )
}

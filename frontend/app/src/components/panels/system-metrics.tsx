// ============================================================================
// System Metrics -- Hardware + pipeline + graph + agent telemetry
// ============================================================================

import { useDashboardStore } from '@/stores/use-dashboard-store'
import { MetricCard } from '@/components/shared/metric-card'
import { GaugeBar } from '@/components/shared/gauge-bar'
import { fmtNum } from '@/lib/utils'
import { useDriftStatus, useConsortiumStatus } from '@/hooks/use-api'
import {
  Activity,
  Cpu,
  Gauge,
  BrainCircuit,
  Network,
  ArrowRightLeft,
  Brain,
  Bell,
  CircleDot,
  GitBranch,
  AlertTriangle,
  RotateCcw,
  Zap,
  TrendingDown,
  Globe,
  Shield,
  ShieldCheck,
} from 'lucide-react'
import type { LucideIcon } from 'lucide-react'

function metric(value: number | null | undefined, digits = 0, suffix = ''): string {
  if (value == null || !Number.isFinite(value)) return 'n/a'
  return `${digits > 0 ? value.toFixed(digits) : fmtNum(value)}${suffix}`
}

export function SystemMetrics() {
  const orchestrator = useDashboardStore((s) => s.orchestrator)
  const hardware = useDashboardStore((s) => s.hardware)
  const graphMetrics = useDashboardStore((s) => s.graphMetrics)
  const graphSize = useDashboardStore((s) => s.graphSize)
  const agentMetrics = useDashboardStore((s) => s.agentMetrics)

  return (
    <div className="space-y-5 p-4">
      {/* Pipeline Section */}
      <Section title="Pipeline" icon={Activity}>
        <div className="grid grid-cols-2 gap-2">
          <MetricCard
            label="Events Ingested"
            value={metric(orchestrator?.events_ingested)}
            icon={ArrowRightLeft}
            accent="text-[#00579C]"
          />
          <MetricCard
            label="Throughput"
            value={orchestrator ? `${orchestrator.events_per_sec.toFixed(1)}/s` : 'n/a'}
            icon={Gauge}
            accent="text-accent-primary"
          />
          <MetricCard
            label="ML Inferences"
            value={metric(orchestrator?.ml_inferences)}
            icon={Brain}
            accent="text-[#00579C]"
          />
          <MetricCard
            label="Alerts Routed"
            value={metric(orchestrator?.alerts_routed)}
            icon={Bell}
            accent="text-[#DA251C]"
          />
        </div>
      </Section>

      {/* Hardware Section */}
      <Section title="Hardware" icon={Cpu}>
        <div className="space-y-2.5">
          <GaugeBar
            label="GPU VRAM"
            value={hardware ? hardware.gpu_vram_used_mb / Math.max(hardware.gpu_vram_total_mb, 1) * 100 : null}
            color={
              hardware && hardware.gpu_vram_used_mb / Math.max(hardware.gpu_vram_total_mb, 1) > 0.9
                ? 'critical'
                : 'accent'
            }
          />
          <GaugeBar
            label="GPU Utilization"
            value={hardware?.gpu_utilization_pct}
            color={
              hardware && hardware.gpu_utilization_pct > 90 ? 'critical' : 'accent'
            }
          />
          <GaugeBar
            label="CPU Utilization"
            value={hardware?.cpu_utilization_pct}
            color={
              hardware && hardware.cpu_utilization_pct > 90 ? 'critical' : 'accent'
            }
          />
          <div className="grid grid-cols-2 gap-2">
            <MetricCard
              label="LLM Tokens/sec"
              value={hardware ? hardware.llm_tps.toFixed(1) : 'n/a'}
              icon={Zap}
              accent="text-[#00579C]"
            />
            <MetricCard
              label="Load Shedding"
              value={hardware ? (hardware.load_shed_active ? 'ACTIVE' : 'OFF') : 'n/a'}
              accent={!hardware ? 'text-text-muted' : hardware.load_shed_active ? 'text-alert-critical' : 'text-alert-low'}
            />
          </div>
        </div>
      </Section>

      {/* Agent Section */}
      <Section title="Agent" icon={BrainCircuit}>
        <div className="grid grid-cols-2 gap-2">
          <MetricCard
            label="Completed"
            value={metric(agentMetrics?.completed)}
            icon={BrainCircuit}
            accent="text-accent-primary"
          />
          <MetricCard
            label="Fraudulent"
            value={metric(agentMetrics?.verdicts.fraudulent)}
            icon={AlertTriangle}
            accent="text-alert-critical"
          />
          <MetricCard
            label="Suspicious"
            value={metric(agentMetrics?.verdicts.suspicious)}
            accent="text-alert-high"
          />
          <MetricCard
            label="Breaker Triggered"
            value={metric(agentMetrics?.agent_breaker_triggered)}
            accent="text-alert-medium"
          />
        </div>
      </Section>

      {/* Graph Section */}
      <Section title="Transaction Graph" icon={Network}>
        <div className="grid grid-cols-2 gap-2">
          <MetricCard
            label="Nodes"
            value={metric(graphSize?.nodes)}
            icon={CircleDot}
            accent="text-[#00579C]"
          />
          <MetricCard
            label="Edges"
            value={metric(graphSize?.edges)}
            icon={GitBranch}
            accent="text-[#00579C]"
          />
          <MetricCard
            label="Mule Detections"
            value={metric(graphMetrics?.mule_detections)}
            icon={AlertTriangle}
            accent="text-[#DA251C]"
          />
          <MetricCard
            label="Cycle Detections"
            value={metric(graphMetrics?.cycle_detections)}
            icon={RotateCcw}
            accent="text-[#DA251C]"
          />
        </div>
      </Section>

      {/* Model Health Section */}
      <ModelHealthSection />

      {/* Consortium Network Section */}
      <ConsortiumSection />
    </div>
  )
}

function ModelHealthSection() {
  const { data: drift } = useDriftStatus()
  const psi = drift?.psi
  const driftedFeatureCount = drift?.feature_drift
    ? drift.feature_drift.filter((f) => f.has_drift).length
    : null

  const severityColor: Record<string, string> = {
    none: 'text-[#00579C]',
    low: 'text-alert-low',
    moderate: 'text-alert-medium',
    high: 'text-alert-high',
    critical: 'text-alert-critical',
  }

  return (
    <Section title="Model Health" icon={TrendingDown}>
      <div className="grid grid-cols-2 gap-2">
        <MetricCard
          label="Drift Severity"
          value={drift?.severity?.toUpperCase() ?? 'n/a'}
          icon={Shield}
          accent={drift ? (severityColor[drift.severity ?? 'none'] ?? 'text-text-secondary') : 'text-text-muted'}
        />
        <MetricCard
          label="PSI Score"
          value={drift?.psi?.toFixed(4) ?? 'n/a'}
          icon={TrendingDown}
          accent={
            !drift
              ? 'text-text-muted'
              : psi != null && psi > 0.2
              ? 'text-alert-critical'
              : psi != null && psi > 0.1
                ? 'text-alert-high'
                : 'text-[#00579C]'
          }
        />
        <MetricCard
          label="Features Drifted"
          value={driftedFeatureCount == null ? 'n/a' : String(driftedFeatureCount)}
          accent="text-[#DA251C]"
        />
        <MetricCard
          label="Sample Size"
          value={drift ? metric(drift.current_size) : 'n/a'}
          accent="text-text-secondary"
        />
      </div>
      {drift?.recommendation && (
        <p className="mt-2 text-[10px] text-text-muted leading-relaxed px-1">
          {drift.recommendation}
        </p>
      )}
    </Section>
  )
}

function ConsortiumSection() {
  const { data: consortium } = useConsortiumStatus()
  const memberCount = consortium?.member_count ?? consortium?.member_banks
  const verifiedCount = consortium?.verified_proofs ?? consortium?.verified_alerts

  return (
    <Section title="Consortium Network" icon={Globe}>
      <div className="grid grid-cols-2 gap-2">
        <MetricCard
          label="Member Banks"
          value={consortium ? metric(memberCount) : 'n/a'}
          icon={Globe}
          accent="text-[#00579C]"
        />
        <MetricCard
          label="Active Alerts"
          value={consortium ? metric(consortium.active_alerts) : 'n/a'}
          icon={Bell}
          accent="text-[#DA251C]"
        />
        <MetricCard
          label="ZKP Verified"
          value={consortium ? metric(verifiedCount) : 'n/a'}
          icon={ShieldCheck}
          accent="text-[#00579C]"
        />
        <MetricCard
          label="Rejected Proofs"
          value={consortium ? metric(consortium.rejected_proofs) : 'n/a'}
          icon={AlertTriangle}
          accent={
            !consortium
              ? 'text-text-muted'
              : consortium.rejected_proofs > 0
              ? 'text-alert-high'
              : 'text-text-secondary'
          }
        />
      </div>
    </Section>
  )
}

function Section({ title, icon: Icon, children }: { title: string; icon: LucideIcon; children: React.ReactNode }) {
  return (
    <div className="animate-fade-in">
      <div className="flex items-center gap-2 mb-2.5 pb-1.5 border-b border-border-subtle">
        <Icon className="w-3.5 h-3.5 text-accent-primary" />
        <span className="text-[9px] text-text-secondary uppercase tracking-[0.15em] font-semibold">
          {title}
        </span>
      </div>
      {children}
    </div>
  )
}

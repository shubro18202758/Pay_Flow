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

export function SystemMetrics() {
  const orchestrator = useDashboardStore((s) => s.orchestrator)
  const hardware = useDashboardStore((s) => s.hardware)
  const graphMetrics = useDashboardStore((s) => s.graphMetrics)
  const graphSize = useDashboardStore((s) => s.graphSize)
  const agentMetrics = useDashboardStore((s) => s.agentMetrics)

  return (
    <div className="p-4 space-y-5 overflow-y-auto h-full">
      {/* Pipeline Section */}
      <Section title="Pipeline" icon={Activity}>
        <div className="grid grid-cols-2 gap-2">
          <MetricCard
            label="Events Ingested"
            value={fmtNum(orchestrator?.events_ingested ?? 0)}
            icon={ArrowRightLeft}
            accent="text-cyan-400"
          />
          <MetricCard
            label="Throughput"
            value={`${(orchestrator?.events_per_sec ?? 0).toFixed(1)}/s`}
            icon={Gauge}
            accent="text-accent-primary"
          />
          <MetricCard
            label="ML Inferences"
            value={fmtNum(orchestrator?.ml_inferences ?? 0)}
            icon={Brain}
            accent="text-violet-400"
          />
          <MetricCard
            label="Alerts Routed"
            value={fmtNum(orchestrator?.alerts_routed ?? 0)}
            icon={Bell}
            accent="text-amber-400"
          />
        </div>
      </Section>

      {/* Hardware Section */}
      <Section title="Hardware" icon={Cpu}>
        <div className="space-y-2.5">
          <GaugeBar
            label="GPU VRAM"
            value={(hardware?.gpu_vram_used_mb ?? 0) / Math.max(hardware?.gpu_vram_total_mb ?? 1, 1) * 100}
            color={
              (hardware?.gpu_vram_used_mb ?? 0) / Math.max(hardware?.gpu_vram_total_mb ?? 1, 1) > 0.9
                ? 'critical'
                : 'accent'
            }
          />
          <GaugeBar
            label="GPU Utilization"
            value={hardware?.gpu_utilization_pct ?? 0}
            color={
              (hardware?.gpu_utilization_pct ?? 0) > 90 ? 'critical' : 'accent'
            }
          />
          <GaugeBar
            label="CPU Utilization"
            value={hardware?.cpu_utilization_pct ?? 0}
            color={
              (hardware?.cpu_utilization_pct ?? 0) > 90 ? 'critical' : 'accent'
            }
          />
          <div className="grid grid-cols-2 gap-2">
            <MetricCard
              label="LLM Tokens/sec"
              value={(hardware?.llm_tps ?? 0).toFixed(1)}
              icon={Zap}
              accent="text-emerald-400"
            />
            <MetricCard
              label="Load Shedding"
              value={hardware?.load_shed_active ? 'ACTIVE' : 'OFF'}
              accent={hardware?.load_shed_active ? 'text-alert-critical' : 'text-alert-low'}
            />
          </div>
        </div>
      </Section>

      {/* Agent Section */}
      <Section title="Agent" icon={BrainCircuit}>
        <div className="grid grid-cols-2 gap-2">
          <MetricCard
            label="Completed"
            value={fmtNum(agentMetrics?.completed ?? 0)}
            icon={BrainCircuit}
            accent="text-accent-primary"
          />
          <MetricCard
            label="Fraudulent"
            value={fmtNum(agentMetrics?.verdicts.fraudulent ?? 0)}
            icon={AlertTriangle}
            accent="text-alert-critical"
          />
          <MetricCard
            label="Suspicious"
            value={fmtNum(agentMetrics?.verdicts.suspicious ?? 0)}
            accent="text-alert-high"
          />
          <MetricCard
            label="Breaker Triggered"
            value={fmtNum(agentMetrics?.agent_breaker_triggered ?? 0)}
            accent="text-alert-medium"
          />
        </div>
      </Section>

      {/* Graph Section */}
      <Section title="Transaction Graph" icon={Network}>
        <div className="grid grid-cols-2 gap-2">
          <MetricCard
            label="Nodes"
            value={fmtNum(graphSize?.nodes ?? 0)}
            icon={CircleDot}
            accent="text-blue-400"
          />
          <MetricCard
            label="Edges"
            value={fmtNum(graphSize?.edges ?? 0)}
            icon={GitBranch}
            accent="text-blue-300"
          />
          <MetricCard
            label="Mule Detections"
            value={fmtNum(graphMetrics?.mule_detections ?? 0)}
            icon={AlertTriangle}
            accent="text-red-400"
          />
          <MetricCard
            label="Cycle Detections"
            value={fmtNum(graphMetrics?.cycle_detections ?? 0)}
            icon={RotateCcw}
            accent="text-orange-400"
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

  const severityColor: Record<string, string> = {
    none: 'text-emerald-400',
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
          value={drift?.severity?.toUpperCase() ?? '—'}
          icon={Shield}
          accent={severityColor[drift?.severity ?? 'none'] ?? 'text-text-secondary'}
        />
        <MetricCard
          label="PSI Score"
          value={drift?.psi?.toFixed(4) ?? '—'}
          icon={TrendingDown}
          accent={
            (drift?.psi ?? 0) > 0.2
              ? 'text-alert-critical'
              : (drift?.psi ?? 0) > 0.1
                ? 'text-alert-high'
                : 'text-emerald-400'
          }
        />
        <MetricCard
          label="Features Drifted"
          value={String(drift?.feature_drift?.filter((f) => f.has_drift).length ?? 0)}
          accent="text-amber-400"
        />
        <MetricCard
          label="Sample Size"
          value={fmtNum(drift?.current_size ?? 0)}
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

  return (
    <Section title="Consortium Network" icon={Globe}>
      <div className="grid grid-cols-2 gap-2">
        <MetricCard
          label="Member Banks"
          value={String(consortium?.member_count ?? 0)}
          icon={Globe}
          accent="text-cyan-400"
        />
        <MetricCard
          label="Active Alerts"
          value={String(consortium?.active_alerts ?? 0)}
          icon={Bell}
          accent="text-amber-400"
        />
        <MetricCard
          label="ZKP Verified"
          value={String(consortium?.verified_proofs ?? 0)}
          icon={ShieldCheck}
          accent="text-emerald-400"
        />
        <MetricCard
          label="Rejected Proofs"
          value={String(consortium?.rejected_proofs ?? 0)}
          icon={AlertTriangle}
          accent={
            (consortium?.rejected_proofs ?? 0) > 0
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

// ============================================================================
// System Page -- Full system health overview with header
// ============================================================================

import { SystemMetrics } from '@/components/panels/system-metrics'
import { Cpu, Activity } from 'lucide-react'

export function SystemPage() {
  return (
    <div className="h-full flex flex-col">
      {/* Page header */}
      <div className="flex items-center gap-4 px-5 py-4 border-b border-border-default bg-bg-surface/30 shrink-0 animate-fade-in">
        <div className="flex items-center justify-center w-9 h-9 rounded-lg bg-accent-primary/10 border border-accent-primary/20">
          <Cpu className="w-5 h-5 text-accent-primary" />
        </div>
        <div>
          <h1 className="text-sm font-semibold text-text-primary flex items-center gap-2">
            System Health
            <Activity className="w-3.5 h-3.5 text-text-muted" />
          </h1>
          <p className="text-[10px] text-text-secondary mt-0.5">
            Hardware telemetry, pipeline throughput, agent performance, and graph analytics
          </p>
        </div>
      </div>

      {/* Content */}
      <div className="flex-1 min-h-0 overflow-hidden">
        <SystemMetrics />
      </div>
    </div>
  )
}

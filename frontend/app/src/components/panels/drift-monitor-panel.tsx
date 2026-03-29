// ============================================================================
// Drift Monitor Panel -- Model drift detection with PSI/KS/JS visualizations
// ============================================================================

import { TrendingDown, AlertTriangle, CheckCircle, Loader2, RefreshCw } from 'lucide-react'
import { cn } from '@/lib/utils'
import { useDriftStatus } from '@/hooks/use-api'
import type { DriftSeverity } from '@/lib/types'

const SEVERITY_CONFIG: Record<DriftSeverity, { color: string; bg: string; label: string; border: string }> = {
  NONE:     { color: 'text-green-400',  bg: 'bg-green-500/15',  border: 'border-green-500/30', label: 'No Drift' },
  LOW:      { color: 'text-cyan-400',   bg: 'bg-cyan-500/15',   border: 'border-cyan-500/30', label: 'Low Drift' },
  MODERATE: { color: 'text-amber-400',  bg: 'bg-amber-500/15',  border: 'border-amber-500/30', label: 'Moderate Drift' },
  HIGH:     { color: 'text-orange-400', bg: 'bg-orange-500/15', border: 'border-orange-500/30', label: 'High Drift' },
  CRITICAL: { color: 'text-red-400',    bg: 'bg-red-500/15',    border: 'border-red-500/30', label: 'Critical Drift' },
}

function MetricBar({ label, value, max, color }: { label: string; value: number; max: number; color: string }) {
  const pct = Math.min((value / max) * 100, 100)
  return (
    <div className="space-y-1">
      <div className="flex items-center justify-between">
        <span className="text-[9px] text-text-muted uppercase tracking-wider">{label}</span>
        <span className="text-[10px] font-mono text-text-secondary">{value.toFixed(4)}</span>
      </div>
      <div className="h-2 bg-bg-surface rounded-full overflow-hidden">
        <div
          className={cn('h-full rounded-full transition-all duration-500', color)}
          style={{ width: `${pct}%` }}
        />
      </div>
    </div>
  )
}

export function DriftMonitorPanel() {
  const { data, isLoading, refetch } = useDriftStatus()

  if (isLoading) {
    return (
      <div className="flex flex-col h-full bg-bg-deep rounded-lg border border-border-subtle">
        <div className="flex items-center gap-2 px-4 py-3 border-b border-border-subtle bg-bg-surface/50">
          <TrendingDown className="w-4 h-4 text-amber-400" />
          <span className="text-xs font-semibold text-text-primary tracking-wide">Model Drift</span>
        </div>
        <div className="flex-1 flex items-center justify-center">
          <Loader2 className="w-5 h-5 animate-spin text-amber-400" />
        </div>
      </div>
    )
  }

  // Handle no data / no reference state
  const hasError = !data || 'error' in data || 'status' in data
  const severity: DriftSeverity = hasError ? 'NONE' : (data.severity as DriftSeverity)
  const config = SEVERITY_CONFIG[severity]

  return (
    <div className="flex flex-col h-full bg-bg-deep rounded-lg border border-border-subtle overflow-hidden">
      {/* Header */}
      <div className="flex items-center gap-2 px-4 py-3 border-b border-border-subtle bg-bg-surface/50">
        <TrendingDown className="w-4 h-4 text-amber-400" />
        <span className="text-xs font-semibold text-text-primary tracking-wide">Model Drift Monitor</span>
        <button
          onClick={() => void refetch()}
          className="ml-auto p-1 rounded hover:bg-bg-elevated/60 transition-colors"
          title="Refresh"
        >
          <RefreshCw className="w-3 h-3 text-text-muted" />
        </button>
      </div>

      {/* Content */}
      <div className="flex-1 min-h-0 overflow-y-auto custom-scrollbar p-4 space-y-4">
        {hasError ? (
          <div className="flex flex-col items-center justify-center h-full text-text-muted text-[10px] gap-2 animate-fade-in">
            <CheckCircle className="w-8 h-8 opacity-30 text-green-400" />
            <p>No reference distribution set yet.</p>
            <p className="text-[9px]">Drift monitoring begins after model training.</p>
          </div>
        ) : (
          <div className="space-y-4 animate-fade-in">
            {/* Severity badge */}
            <div className={cn('flex items-center gap-2 px-3 py-2 rounded-md border', config.bg, config.border)}>
              {severity === 'NONE' ? (
                <CheckCircle className={cn('w-4 h-4', config.color)} />
              ) : (
                <AlertTriangle className={cn('w-4 h-4', config.color)} />
              )}
              <span className={cn('text-[11px] font-semibold', config.color)}>{config.label}</span>
              <span className="text-[9px] text-text-muted ml-auto font-mono">
                {data.reference_size} ref / {data.current_size} cur
              </span>
            </div>

            {/* Metrics */}
            <div className="space-y-3">
              <MetricBar
                label="Population Stability Index (PSI)"
                value={data.psi}
                max={0.5}
                color={data.psi >= 0.25 ? 'bg-red-500' : data.psi >= 0.1 ? 'bg-amber-500' : 'bg-green-500'}
              />
              <MetricBar
                label="Kolmogorov-Smirnov Statistic"
                value={data.ks_statistic}
                max={1.0}
                color={data.ks_statistic >= 0.3 ? 'bg-red-500' : data.ks_statistic >= 0.15 ? 'bg-amber-500' : 'bg-green-500'}
              />
              <MetricBar
                label="Jensen-Shannon Divergence"
                value={data.js_divergence}
                max={0.5}
                color={data.js_divergence >= 0.2 ? 'bg-red-500' : data.js_divergence >= 0.1 ? 'bg-amber-500' : 'bg-green-500'}
              />
              <div className="space-y-1">
                <span className="text-[9px] text-text-muted uppercase tracking-wider">KS p-value</span>
                <span className={cn(
                  'block text-[11px] font-mono',
                  data.ks_p_value < 0.05 ? 'text-red-400' : 'text-green-400',
                )}>
                  {data.ks_p_value.toFixed(6)}
                  {data.ks_p_value < 0.05 && (
                    <span className="text-[9px] text-red-400/70 ml-2">Statistically significant</span>
                  )}
                </span>
              </div>
            </div>

            {/* Recommendation */}
            {data.recommendation && (
              <div className="bg-bg-surface/60 rounded-md border border-border-subtle p-3">
                <p className="text-[9px] text-text-muted uppercase tracking-wider mb-1">Recommendation</p>
                <p className="text-[10px] text-text-secondary leading-relaxed">{data.recommendation}</p>
              </div>
            )}

            {/* Feature-level drift */}
            {data.feature_drift && data.feature_drift.length > 0 && (
              <div>
                <p className="text-[9px] text-text-muted uppercase tracking-wider mb-2">
                  Feature-Level Drift ({data.feature_drift.length})
                </p>
                <div className="space-y-1">
                  {data.feature_drift.slice(0, 15).map((fd) => {
                    const fCfg = SEVERITY_CONFIG[fd.severity]
                    return (
                      <div key={fd.feature} className="flex items-center gap-2 text-[9px]">
                        <span className={cn('w-1.5 h-1.5 rounded-full', fCfg.bg.replace('/15', '/60'))} />
                        <span className="text-text-secondary font-mono flex-1 truncate">{fd.feature}</span>
                        <span className={cn('font-mono', fCfg.color)}>{fd.psi.toFixed(4)}</span>
                      </div>
                    )
                  })}
                </div>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  )
}

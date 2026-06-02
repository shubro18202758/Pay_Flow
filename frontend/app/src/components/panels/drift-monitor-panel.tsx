// ============================================================================
// Drift Monitor Panel -- Model drift detection with PSI/KS/JS visualizations
// ============================================================================

import { TrendingDown, AlertTriangle, CheckCircle, Loader2, RefreshCw } from 'lucide-react'
import { cn } from '@/lib/utils'
import { useDriftStatus } from '@/hooks/use-api'
import type { DriftSeverity } from '@/lib/types'

const SEVERITY_CONFIG: Record<DriftSeverity, { color: string; bg: string; label: string; border: string }> = {
  NONE:     { color: 'text-[#00579C]',  bg: 'bg-[#00579C]/15',  border: 'border-[#00579C]/30', label: 'No Drift' },
  LOW:      { color: 'text-[#00579C]',   bg: 'bg-[#00579C]/15',   border: 'border-[#00579C]/30', label: 'Low Drift' },
  MODERATE: { color: 'text-[#DA251C]',  bg: 'bg-[#DA251C]/15',  border: 'border-[#DA251C]/30', label: 'Moderate Drift' },
  HIGH:     { color: 'text-[#DA251C]', bg: 'bg-[#DA251C]/15', border: 'border-[#DA251C]/30', label: 'High Drift' },
  CRITICAL: { color: 'text-[#DA251C]',    bg: 'bg-[#DA251C]/15',    border: 'border-[#DA251C]/30', label: 'Critical Drift' },
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
          <TrendingDown className="w-4 h-4 text-[#DA251C]" />
          <span className="text-xs font-semibold text-text-primary tracking-wide">Model Drift</span>
        </div>
        <div className="flex-1 flex items-center justify-center">
          <Loader2 className="w-5 h-5 animate-spin text-[#DA251C]" />
        </div>
      </div>
    )
  }

  const apiError = data && 'error' in data
    ? String((data as { error?: string }).error ?? 'Drift detector unavailable')
    : null
  const status = data?.status
  const isNoReference = !apiError && status === 'no_reference'
  const isWarming = !apiError && status === 'warming'
  const isUnavailable = !data || !!apiError || isNoReference
  const severity: DriftSeverity = isUnavailable || isWarming ? 'NONE' : (data.severity as DriftSeverity)
  const config = SEVERITY_CONFIG[severity]
  const requiredCurrentSize = data?.required_current_size ?? 250
  const warmingPct = data
    ? Math.min((data.current_size / Math.max(requiredCurrentSize, 1)) * 100, 100)
    : 0

  return (
    <div className="flex flex-col h-full bg-bg-deep rounded-lg border border-border-subtle overflow-hidden">
      {/* Header */}
      <div className="flex items-center gap-2 px-4 py-3 border-b border-border-subtle bg-bg-surface/50">
        <TrendingDown className="w-4 h-4 text-[#DA251C]" />
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
        {!data || apiError ? (
          <div className="flex flex-col items-center justify-center h-full text-text-muted text-[10px] gap-2 animate-fade-in text-center">
            <AlertTriangle className="w-8 h-8 opacity-40 text-[#DA251C]" />
            <p>{apiError ?? 'Drift detector response unavailable.'}</p>
          </div>
        ) : isNoReference ? (
          <div className="flex flex-col items-center justify-center h-full text-text-muted text-[10px] gap-2 animate-fade-in">
            <CheckCircle className="w-8 h-8 opacity-30 text-[#00579C]" />
            <p>No training reference distribution yet.</p>
            <p className="text-[9px] text-center">{data.message ?? data.recommendation}</p>
          </div>
        ) : isWarming ? (
          <div className="space-y-4 animate-fade-in">
            <div className="rounded-md border border-[#00579C]/25 bg-[#00579C]/10 p-3">
              <div className="flex items-center gap-2">
                <CheckCircle className="w-4 h-4 text-[#00579C]" />
                <span className="text-[11px] font-semibold text-[#00579C]">
                  Reference Ready
                </span>
                <span className="ml-auto text-[9px] text-text-muted font-mono">
                  {data.reference_size} ref
                </span>
              </div>
              <p className="mt-2 text-[10px] text-text-secondary leading-relaxed">
                {data.message ?? 'Live prediction window is still warming.'}
              </p>
            </div>

            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <span className="text-[9px] text-text-muted uppercase tracking-wider">Live prediction window</span>
                <span className="text-[10px] font-mono text-text-secondary">
                  {data.current_size} / {requiredCurrentSize}
                </span>
              </div>
              <div className="h-2 bg-bg-surface rounded-full overflow-hidden">
                <div
                  className="h-full rounded-full bg-[#00579C] transition-all duration-500"
                  style={{ width: `${warmingPct}%` }}
                />
              </div>
              <p className="text-[9px] text-text-muted leading-relaxed">
                Drift statistics will appear after enough live model predictions are recorded.
              </p>
            </div>
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
                color={data.psi >= 0.25 ? 'bg-[#DA251C]' : data.psi >= 0.1 ? 'bg-[#DA251C]' : 'bg-[#00579C]'}
              />
              <MetricBar
                label="Kolmogorov-Smirnov Statistic"
                value={data.ks_statistic}
                max={1.0}
                color={data.ks_statistic >= 0.3 ? 'bg-[#DA251C]' : data.ks_statistic >= 0.15 ? 'bg-[#DA251C]' : 'bg-[#00579C]'}
              />
              <MetricBar
                label="Jensen-Shannon Divergence"
                value={data.js_divergence}
                max={0.5}
                color={data.js_divergence >= 0.2 ? 'bg-[#DA251C]' : data.js_divergence >= 0.1 ? 'bg-[#DA251C]' : 'bg-[#00579C]'}
              />
              <div className="space-y-1">
                <span className="text-[9px] text-text-muted uppercase tracking-wider">KS p-value</span>
                <span className={cn(
                  'block text-[11px] font-mono',
                  data.ks_p_value < 0.05 ? 'text-[#DA251C]' : 'text-[#00579C]',
                )}>
                  {data.ks_p_value.toFixed(6)}
                  {data.ks_p_value < 0.05 && (
                    <span className="text-[9px] text-[#DA251C]/70 ml-2">Statistically significant</span>
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

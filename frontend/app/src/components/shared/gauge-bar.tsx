// ============================================================================
// Gauge Bar -- Horizontal fill bar with gradient and glow effects
// ============================================================================

import { cn } from '@/lib/utils'

interface Props {
  value: number // 0-100
  max?: number
  label?: string
  color?: 'accent' | 'critical' | 'high' | 'medium' | 'low'
  className?: string
}

const colorMap = {
  accent: 'bg-accent-primary',
  critical: 'bg-alert-critical',
  high: 'bg-alert-high',
  medium: 'bg-alert-medium',
  low: 'bg-alert-low',
}

const glowMap = {
  accent: 'shadow-[0_0_8px_oklch(0.55_0.14_250_/_0.3)]',
  critical: 'shadow-[0_0_8px_oklch(0.55_0.25_25_/_0.3)]',
  high: 'shadow-[0_0_8px_oklch(0.60_0.18_50_/_0.3)]',
  medium: 'shadow-[0_0_8px_oklch(0.65_0.18_85_/_0.3)]',
  low: 'shadow-[0_0_8px_oklch(0.60_0.15_145_/_0.3)]',
}

export function GaugeBar({ value, max = 100, label, color = 'accent', className }: Props) {
  const pct = Math.min(100, Math.max(0, (value / max) * 100))

  return (
    <div className={cn('space-y-1.5', className)}>
      {label && (
        <div className="flex justify-between text-[9px]">
          <span className="text-text-secondary font-medium uppercase tracking-wider">{label}</span>
          <span className="font-mono font-semibold text-text-primary tabular-nums">
            {value.toFixed(1)}%
          </span>
        </div>
      )}
      <div className="h-1.5 bg-bg-deep rounded-full overflow-hidden border border-border-subtle/50">
        <div
          className={cn(
            'h-full rounded-full transition-[width] duration-500 ease-out',
            colorMap[color],
            pct > 60 && glowMap[color],
          )}
          style={{ width: `${pct}%` }}
        />
      </div>
    </div>
  )
}

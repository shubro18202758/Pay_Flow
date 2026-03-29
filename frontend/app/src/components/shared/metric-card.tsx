// ============================================================================
// Metric Card -- Institutional-grade KPI card with subtle glow effects
// ============================================================================

import { cn } from '@/lib/utils'
import type { LucideIcon } from 'lucide-react'

interface Props {
  label: string
  value: string | number
  sub?: string
  icon?: LucideIcon
  accent?: string
  className?: string
}

export function MetricCard({ label, value, sub, icon: Icon, accent, className }: Props) {
  return (
    <div
      className={cn(
        'bg-bg-elevated/80 rounded-lg p-3 border border-border-subtle card-hover',
        className,
      )}
    >
      <div className="flex items-center justify-between mb-1">
        <div className="text-[9px] text-text-muted uppercase tracking-[0.12em] font-semibold">
          {label}
        </div>
        {Icon && <Icon className={cn('w-3.5 h-3.5', accent ?? 'text-text-muted')} />}
      </div>
      <div className={cn(
        'text-lg font-bold font-mono leading-tight tabular-nums',
        accent ?? 'text-text-primary',
      )}>
        {value}
      </div>
      {sub && (
        <div className="text-[9px] text-text-secondary mt-1 font-medium">{sub}</div>
      )}
    </div>
  )
}

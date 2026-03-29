// ============================================================================
// Severity Badge -- CRITICAL / SUSPICIOUS / LEGITIMATE / ESCALATED
// ============================================================================

import { cn } from '@/lib/utils'
import {
  ShieldAlert,
  AlertTriangle,
  Eye,
  ShieldCheck,
  UserCheck,
} from 'lucide-react'
import type { LucideIcon } from 'lucide-react'

type Severity = 'critical' | 'high' | 'medium' | 'low' | 'escalated'

interface Props {
  severity: Severity
  label?: string
  className?: string
}

const severityConfig: Record<Severity, { style: string; icon: LucideIcon; defaultLabel: string }> = {
  critical: {
    style: 'bg-alert-critical/15 text-alert-critical border-alert-critical/30',
    icon: ShieldAlert,
    defaultLabel: 'FRAUD',
  },
  high: {
    style: 'bg-alert-high/15 text-alert-high border-alert-high/30',
    icon: AlertTriangle,
    defaultLabel: 'SUSPICIOUS',
  },
  medium: {
    style: 'bg-alert-medium/15 text-alert-medium border-alert-medium/30',
    icon: Eye,
    defaultLabel: 'REVIEW',
  },
  low: {
    style: 'bg-alert-low/15 text-alert-low border-alert-low/30',
    icon: ShieldCheck,
    defaultLabel: 'LEGITIMATE',
  },
  escalated: {
    style: 'bg-alert-escalated/15 text-alert-escalated border-alert-escalated/30',
    icon: UserCheck,
    defaultLabel: 'ESCALATED',
  },
}

export function SeverityBadge({ severity, label, className }: Props) {
  const config = severityConfig[severity]
  const Icon = config.icon

  return (
    <span
      className={cn(
        'inline-flex items-center gap-1 px-2 py-0.5 text-[8px] font-bold uppercase tracking-[0.12em] rounded-md border',
        config.style,
        className,
      )}
    >
      <Icon className="w-2.5 h-2.5" />
      {label ?? config.defaultLabel}
    </span>
  )
}

export function verdictToSeverity(verdict: string): Severity {
  const v = verdict.toLowerCase()
  if (v === 'fraudulent') return 'critical'
  if (v === 'suspicious') return 'high'
  if (v.includes('escalat')) return 'escalated'
  if (v === 'legitimate') return 'low'
  return 'medium'
}

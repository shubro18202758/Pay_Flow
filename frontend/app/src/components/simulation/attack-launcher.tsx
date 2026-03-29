// ============================================================================
// Attack Launcher -- API-driven cards with dynamic parameter configuration
// ============================================================================

import { useState } from 'react'
import { useAttackTypes, useLaunchAttack } from '@/hooks/use-api'
import { cn } from '@/lib/utils'
import {
  Network,
  RefreshCcw,
  Zap,
  AlertTriangle,
  Crosshair,
  ChevronDown,
  ChevronUp,
} from 'lucide-react'
import type { LucideIcon } from 'lucide-react'
import type { AttackTypeDetail } from '@/lib/types'

const ATTACK_ICONS: Record<string, { icon: LucideIcon; iconColor: string }> = {
  upi_mule_network: { icon: Network, iconColor: 'text-alert-critical' },
  circular_laundering: { icon: RefreshCcw, iconColor: 'text-alert-high' },
  velocity_phishing: { icon: Zap, iconColor: 'text-alert-medium' },
}

function formatParamValue(value: number, label: string): string {
  if (label.includes('Amount') || label.includes('INR')) {
    const inr = value
    if (inr >= 10_000_000) return `₹${(inr / 10_000_000).toFixed(1)}Cr`
    if (inr >= 100_000) return `₹${(inr / 100_000).toFixed(1)}L`
    if (inr >= 1000) return `₹${(inr / 1000).toFixed(0)}K`
    return `₹${inr}`
  }
  return String(value)
}

function AttackCard({
  type,
  detail,
  index,
}: {
  type: string
  detail: AttackTypeDetail
  index: number
}) {
  const launch = useLaunchAttack()
  const [isLaunching, setIsLaunching] = useState(false)
  const [expanded, setExpanded] = useState(false)
  const [params, setParams] = useState<Record<string, number>>(() => {
    const defaults: Record<string, number> = {}
    for (const [key, schema] of Object.entries(detail.params)) {
      defaults[key] = schema.default
    }
    return defaults
  })

  const iconInfo = ATTACK_ICONS[type] ?? { icon: AlertTriangle, iconColor: 'text-text-muted' }
  const Icon = iconInfo.icon

  async function handleLaunch() {
    setIsLaunching(true)
    try {
      await launch.mutateAsync({ attack_type: type, params })
    } finally {
      setIsLaunching(false)
    }
  }

  return (
    <div
      className="card-hover animate-fade-in bg-bg-elevated/95 border border-border-default rounded-md p-4 flex flex-col backdrop-blur-sm shadow-[inset_0_1px_0_0_rgba(255,255,255,0.03)]"
      style={{ animationDelay: `${index * 60}ms` }}
    >
      <div className="flex items-center gap-2.5 mb-2">
        <div className="w-8 h-8 rounded-lg border border-border-subtle bg-bg-overlay/80 flex items-center justify-center">
          <Icon className={cn('w-4 h-4', iconInfo.iconColor)} />
        </div>
        <h3 className="text-[11px] font-bold text-text-primary uppercase tracking-[0.12em] flex-1">
          {detail.label}
        </h3>
      </div>

      <p className="text-[10px] text-text-secondary leading-relaxed mb-3">
        {detail.description}
      </p>

      {/* Phases */}
      <div className="mb-3 space-y-1">
        {detail.phases.map((phase, i) => (
          <div key={i} className="flex items-start gap-1.5">
            <span className="text-[8px] font-mono text-accent-primary mt-0.5 shrink-0">{i + 1}.</span>
            <span className="text-[9px] text-text-muted leading-snug">{phase}</span>
          </div>
        ))}
      </div>

      {/* Parameter toggles */}
      <button
        onClick={() => setExpanded(!expanded)}
        className="flex items-center gap-1 text-[9px] text-accent-primary font-semibold uppercase tracking-wider mb-2 hover:text-accent-primary/80 transition-colors"
      >
        {expanded ? <ChevronUp className="w-3 h-3" /> : <ChevronDown className="w-3 h-3" />}
        Configure Parameters ({Object.keys(detail.params).length})
      </button>

      {expanded && (
        <div className="mb-3 space-y-3 bg-bg-overlay/40 rounded-md p-3 border border-border-subtle">
          {Object.entries(detail.params).map(([key, schema]) => (
            <div key={key}>
              <div className="flex justify-between items-center mb-1">
                <label className="text-[9px] font-semibold text-text-primary uppercase tracking-wider">
                  {schema.label}
                </label>
                <span className="text-[10px] font-mono font-bold text-accent-primary tabular-nums">
                  {formatParamValue(params[key], schema.label)}
                </span>
              </div>
              <input
                type="range"
                min={schema.min}
                max={schema.max}
                step={schema.step}
                value={params[key]}
                onChange={(e) => setParams((p) => ({ ...p, [key]: Number(e.target.value) }))}
                className="w-full h-1.5 bg-bg-deep rounded-full appearance-none cursor-pointer accent-accent-primary [&::-webkit-slider-thumb]:w-3 [&::-webkit-slider-thumb]:h-3 [&::-webkit-slider-thumb]:rounded-full [&::-webkit-slider-thumb]:bg-accent-primary [&::-webkit-slider-thumb]:appearance-none"
              />
              <p className="text-[8px] text-text-muted mt-0.5">{schema.description}</p>
            </div>
          ))}
        </div>
      )}

      {/* Launch */}
      <div className="mt-auto">
        <button
          onClick={() => void handleLaunch()}
          disabled={isLaunching || launch.isPending}
          className={cn(
            'group relative w-full py-2 rounded-md text-[10px] font-bold uppercase tracking-wider transition-all duration-200',
            'border border-accent-primary/70 text-accent-primary',
            'hover:bg-accent-primary hover:text-bg-deep hover:shadow-[0_0_16px_oklch(0.55_0.14_250_/_0.35)]',
            'disabled:opacity-40 disabled:cursor-not-allowed disabled:hover:shadow-none disabled:hover:bg-transparent disabled:hover:text-accent-primary',
          )}
        >
          <span className="flex items-center justify-center gap-1.5">
            <Crosshair className={cn(
              'w-3 h-3 transition-transform duration-200',
              !isLaunching && 'group-hover:rotate-90',
            )} />
            {isLaunching ? 'Launching...' : 'Launch Attack'}
          </span>
        </button>
      </div>
    </div>
  )
}

export function AttackLauncher() {
  const { data } = useAttackTypes()
  const attacks = data?.attacks ?? {}

  return (
    <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
      {Object.entries(attacks).map(([type, detail], idx) => (
        <AttackCard key={type} type={type} detail={detail} index={idx} />
      ))}

      {Object.keys(attacks).length === 0 && (
        <div className="col-span-3 flex flex-col items-center justify-center py-12 animate-fade-in">
          <AlertTriangle className="w-5 h-5 text-text-muted mb-2" />
          <p className="text-text-muted text-[10px] uppercase tracking-wider font-semibold">
            Connect to backend to load attack types
          </p>
        </div>
      )}
    </div>
  )
}

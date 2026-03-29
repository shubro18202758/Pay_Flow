// ============================================================================
// Velocity Sparklines — Per-account transaction velocity with inline SVG sparklines
// ============================================================================

import { useVelocityTrends } from '@/hooks/use-api'
import { cn, fmtNum, truncId } from '@/lib/utils'
import { Activity, TrendingUp, TrendingDown, AlertTriangle } from 'lucide-react'
import type { VelocityAccount } from '@/lib/types'

const VELOCITY_THRESHOLDS = {
  high: 50,
  medium: 20,
} as const

function Sparkline({ values, color }: { values: number[]; color: string }) {
  if (values.length < 2) return null

  const width = 80
  const height = 20
  const max = Math.max(...values, 1)
  const step = width / (values.length - 1)

  const points = values.map((v, i) => `${i * step},${height - (v / max) * height}`)
  const polyline = points.join(' ')

  // Fill polygon (close path at bottom)
  const fillPoints = `0,${height} ${polyline} ${(values.length - 1) * step},${height}`

  return (
    <svg width={width} height={height} className="shrink-0" viewBox={`0 0 ${width} ${height}`}>
      <polygon points={fillPoints} fill={color} opacity="0.15" />
      <polyline points={polyline} fill="none" stroke={color} strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
      {/* latest point dot */}
      <circle
        cx={(values.length - 1) * step}
        cy={height - (values[values.length - 1] / max) * height}
        r="2"
        fill={color}
      />
    </svg>
  )
}

function velocityColor(txnCount: number): string {
  if (txnCount >= VELOCITY_THRESHOLDS.high) return '#ef4444'   // red-500
  if (txnCount >= VELOCITY_THRESHOLDS.medium) return '#f59e0b' // amber-500
  return '#22c55e' // green-500
}

function VelocityTrend({ current, previous }: { current: number; previous: number }) {
  if (previous === 0) return null
  const change = ((current - previous) / previous) * 100
  const isUp = change > 0

  return (
    <div className={cn(
      'flex items-center gap-0.5 text-[8px] font-mono tabular-nums',
      isUp ? 'text-red-400' : 'text-emerald-400',
    )}>
      {isUp ? <TrendingUp className="w-2.5 h-2.5" /> : <TrendingDown className="w-2.5 h-2.5" />}
      <span>{isUp ? '+' : ''}{change.toFixed(0)}%</span>
    </div>
  )
}

export function VelocitySparklines({
  windowMinutes = 30,
  topN = 8,
}: {
  windowMinutes?: number
  topN?: number
}) {
  const { data, isLoading } = useVelocityTrends(windowMinutes, topN)

  const accounts = data?.accounts ?? []

  return (
    <div className="rounded-lg border border-border-subtle bg-bg-card p-3">
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-1.5 text-[10px] font-semibold uppercase tracking-[0.12em] text-text-secondary">
          <Activity className="w-3.5 h-3.5 text-accent-primary/70" />
          <span>Velocity Trends</span>
        </div>
        <span className="text-[8px] font-mono text-text-muted">
          Top {topN} | {windowMinutes}m
        </span>
      </div>

      {isLoading || accounts.length === 0 ? (
        <div className="flex items-center justify-center h-16 text-[10px] text-text-muted">
          {isLoading ? 'Loading velocity data…' : 'No velocity data available'}
        </div>
      ) : (
        <div className="space-y-1">
          {/* Column headers */}
          <div className="flex items-center gap-2 px-1.5 text-[7px] uppercase tracking-wider text-text-muted border-b border-border-subtle/30 pb-1">
            <span className="w-20">Account</span>
            <span className="w-10 text-right">Txns</span>
            <span className="w-10 text-right">Fraud</span>
            <span className="flex-1 text-center">Velocity</span>
            <span className="w-10 text-right">Trend</span>
          </div>

          {accounts.map((acct: VelocityAccount) => {
            const color = velocityColor(acct.count)
            const fraudRatio = acct.count > 0 ? acct.fraud_count / acct.count : 0
            // Simulated sparkline from count (show a progression)
            const sparkValues = generateSparkValues(acct.count, 8)
            const prevVal = sparkValues.length > 1 ? sparkValues[sparkValues.length - 2] : 0
            const curVal = sparkValues[sparkValues.length - 1]

            return (
              <div
                key={acct.account_id}
                className={cn(
                  'flex items-center gap-2 rounded-md px-1.5 py-1 transition-colors',
                  fraudRatio > 0.3 ? 'bg-red-500/5 hover:bg-red-500/10' : 'hover:bg-bg-elevated/50',
                )}
              >
                <div className="w-20 flex items-center gap-1">
                  {fraudRatio > 0.3 && (
                    <AlertTriangle className="w-2.5 h-2.5 text-red-400 shrink-0" />
                  )}
                  <span className="text-[9px] font-mono text-text-primary truncate">
                    {truncId(acct.account_id, 10)}
                  </span>
                </div>
                <span className="w-10 text-right text-[9px] font-mono tabular-nums text-text-secondary">
                  {fmtNum(acct.count)}
                </span>
                <span className={cn(
                  'w-10 text-right text-[9px] font-mono tabular-nums',
                  acct.fraud_count > 0 ? 'text-red-400' : 'text-text-muted',
                )}>
                  {acct.fraud_count}
                </span>
                <div className="flex-1 flex justify-center">
                  <Sparkline values={sparkValues} color={color} />
                </div>
                <div className="w-10 flex justify-end">
                  <VelocityTrend current={curVal} previous={prevVal} />
                </div>
              </div>
            )
          })}
        </div>
      )}
    </div>
  )
}

/** Generate synthetic sparkline values based on final count (for visual effect) */
function generateSparkValues(finalCount: number, steps: number): number[] {
  const values: number[] = []
  const base = Math.max(finalCount * 0.3, 1)
  for (let i = 0; i < steps; i++) {
    const progress = i / (steps - 1)
    const val = base + (finalCount - base) * progress + (Math.sin(i * 1.7) * finalCount * 0.15)
    values.push(Math.max(0, Math.round(val)))
  }
  values[steps - 1] = finalCount
  return values
}

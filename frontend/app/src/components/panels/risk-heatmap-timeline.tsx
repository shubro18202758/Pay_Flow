// ============================================================================
// Risk Heatmap Timeline — Temporal risk concentration grid with fraud overlays
// ============================================================================

import { useTemporalHeatmap } from '@/hooks/use-api'
import { cn, fmtNum } from '@/lib/utils'
import { Clock, Flame, TrendingUp } from 'lucide-react'
import type { TemporalBucket } from '@/lib/types'

const RISK_GRADIENT = [
  'bg-emerald-500/20',
  'bg-emerald-500/40',
  'bg-lime-500/40',
  'bg-yellow-500/50',
  'bg-amber-500/50',
  'bg-orange-500/60',
  'bg-red-500/50',
  'bg-red-500/70',
  'bg-red-500/90',
  'bg-rose-600',
]

function riskClass(ratio: number): string {
  const idx = Math.min(Math.floor(ratio * 10), 9)
  return RISK_GRADIENT[idx]
}

function formatBucketTime(ts: number): string {
  const d = new Date(ts * 1000)
  return d.toLocaleTimeString('en-IN', { hour: '2-digit', minute: '2-digit', hour12: false })
}

export function RiskHeatmapTimeline({
  bucketSeconds = 120,
  lookbackMinutes = 30,
}: {
  bucketSeconds?: number
  lookbackMinutes?: number
}) {
  const { data, isLoading } = useTemporalHeatmap(bucketSeconds, lookbackMinutes)

  const buckets = data?.buckets ?? []
  const maxTxn = Math.max(...buckets.map((b: TemporalBucket) => b.txn_count), 1)

  return (
    <div className="rounded-lg border border-border-subtle bg-bg-card p-3">
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-1.5 text-[10px] font-semibold uppercase tracking-[0.12em] text-text-secondary">
          <Clock className="w-3.5 h-3.5 text-accent-primary/70" />
          <span>Risk Heatmap Timeline</span>
        </div>
        <div className="flex items-center gap-3 text-[8px] text-text-muted">
          <span>{lookbackMinutes}m window</span>
          <span>{bucketSeconds}s buckets</span>
        </div>
      </div>

      {isLoading || buckets.length === 0 ? (
        <div className="flex items-center justify-center h-20 text-[10px] text-text-muted">
          {isLoading ? 'Loading temporal data…' : 'No data in current window'}
        </div>
      ) : (
        <>
          {/* Heatmap Grid */}
          <div className="flex gap-0.5 items-end mb-1.5">
            {buckets.map((bucket: TemporalBucket, i: number) => {
              const fraudRatio = bucket.txn_count > 0
                ? bucket.fraud_count / bucket.txn_count
                : 0
              const heightPct = (bucket.txn_count / maxTxn) * 100
              return (
                <div key={i} className="flex-1 flex flex-col items-stretch gap-0.5" title={
                  `${formatBucketTime(bucket.bucket_start)}\n${bucket.txn_count} txns | ${bucket.fraud_count} fraud (${(fraudRatio * 100).toFixed(1)}%)\n₹${(bucket.total_paisa / 100).toLocaleString('en-IN')}`
                }>
                  {/* Volume bar */}
                  <div className="relative h-10 flex items-end">
                    <div
                      className={cn(
                        'w-full rounded-t-sm transition-all duration-300',
                        riskClass(fraudRatio),
                      )}
                      style={{ height: `${Math.max(heightPct, 5)}%` }}
                    />
                  </div>
                  {/* Fraud indicator dot */}
                  {bucket.fraud_count > 0 && (
                    <div className="flex justify-center">
                      <div className={cn(
                        'w-1.5 h-1.5 rounded-full',
                        fraudRatio > 0.3 ? 'bg-red-500 animate-pulse' : fraudRatio > 0.1 ? 'bg-orange-400' : 'bg-amber-400/60',
                      )} />
                    </div>
                  )}
                </div>
              )
            })}
          </div>

          {/* Time axis */}
          <div className="flex justify-between text-[7px] font-mono text-text-muted mt-0.5">
            {buckets.length > 0 && (
              <>
                <span>{formatBucketTime(buckets[0].bucket_start)}</span>
                {buckets.length > 4 && (
                  <span>{formatBucketTime(buckets[Math.floor(buckets.length / 2)].bucket_start)}</span>
                )}
                <span>{formatBucketTime(buckets[buckets.length - 1].bucket_start)}</span>
              </>
            )}
          </div>

          {/* Summary row */}
          <div className="flex items-center gap-4 mt-2.5 pt-2 border-t border-border-subtle/50">
            <SummaryChip
              icon={TrendingUp}
              label="Total Txns"
              value={fmtNum(buckets.reduce((s: number, b: TemporalBucket) => s + b.txn_count, 0))}
            />
            <SummaryChip
              icon={Flame}
              label="Fraud Hits"
              value={fmtNum(buckets.reduce((s: number, b: TemporalBucket) => s + b.fraud_count, 0))}
              highlight
            />
            <div className="ml-auto flex items-center gap-1 text-[8px] text-text-muted">
              <span className="w-2 h-2 rounded-sm bg-emerald-500/40" /> Low
              <span className="w-2 h-2 rounded-sm bg-amber-500/50 ml-1" /> Med
              <span className="w-2 h-2 rounded-sm bg-red-500/70 ml-1" /> High
            </div>
          </div>
        </>
      )}
    </div>
  )
}

function SummaryChip({
  icon: Icon,
  label,
  value,
  highlight,
}: {
  icon: typeof Clock
  label: string
  value: string
  highlight?: boolean
}) {
  return (
    <div className="flex items-center gap-1.5">
      <Icon className={cn('w-3 h-3', highlight ? 'text-red-400' : 'text-text-muted')} />
      <div>
        <div className="text-[7px] uppercase tracking-wider text-text-muted">{label}</div>
        <div className={cn('text-[11px] font-mono font-semibold tabular-nums', highlight ? 'text-red-400' : 'text-text-primary')}>
          {value}
        </div>
      </div>
    </div>
  )
}

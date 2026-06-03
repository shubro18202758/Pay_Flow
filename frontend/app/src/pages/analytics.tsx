// ============================================================================
// Analytics Page -- real-time fraud detection data visualization
// Features: backend-derived telemetry, XAI context, and Union Bank styling
// ============================================================================

import { useEffect, useRef, useMemo, useCallback, useState, type ReactNode } from 'react'
import {
  AreaChart, Area, BarChart, Bar, LineChart, Line, ComposedChart,
  PieChart, Pie, Cell, RadarChart, Radar, PolarGrid, PolarAngleAxis,
  PolarRadiusAxis, Treemap, FunnelChart, Funnel, LabelList,
  ScatterChart, Scatter, RadialBarChart, RadialBar, Legend,
  XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ReferenceLine,
} from 'recharts'
import {
  useAnalyticsStore,
  type AccountRiskBand,
  type AlertFunnel,
  type AmountDistribution,
  type FraudRatePoint,
  type FraudTypology,
  type LatencyPoint,
  type ModelPerformance,
  type RiskHeatmapCell,
  type ThreatEvent,
  type TransactionVolumePoint,
  type VelocityBucket,
} from '@/stores/use-analytics-store'
import { useActivityStore, type EventLabReportActivity, type EventLifecycle } from '@/stores/use-activity-store'
import {
  Activity, TrendingUp, TrendingDown, Shield, ShieldAlert,
  Zap, Timer, Brain, Fingerprint, AlertTriangle, BarChart3, Network,
  Target, Gauge, ArrowUpRight, ArrowDownRight, Layers, Radio, Globe, MapPin,
} from 'lucide-react'
import { cn } from '@/lib/utils'
import { WorldThreatMap, IndiaRegionalMap, CountryThreatPanel, CrossBorderCorridors, AttackVectorBreakdown, LiveThreatFeed } from '@/components/threat-maps'

// ============================================================================
// Constants
// ============================================================================

const CHART_COLORS = {
  primary: '#DA251C',
  secondary: '#00579C',
  success: '#0f9f6e',
  warning: '#f5b400',
  danger: '#DA251C',
  purple: '#003F73',
  deepRed: '#B51A13',
  gold: '#f5b400',
  brandBlue: '#00579C',
  blue: '#00579C',
  emerald: '#0f9f6e',
  darkRed: '#B51A13',
  text: '#172033',
  muted: '#718096',
  border: '#c6d3e3',
  surface: '#ffffff',
}

const CHANNEL_COLORS: Record<string, string> = {
  UPI: '#DA251C',
  NEFT: '#00579C',
  RTGS: '#B51A13',
  IMPS: '#003F73',
  Card: '#40516b',
  Wallet: '#718096',
}

const TOOLTIP_STYLE = {
  contentStyle: {
    backgroundColor: '#ffffff',
    border: '1px solid rgba(0, 87, 156, 0.22)',
    borderRadius: '6px',
    fontSize: '11px',
    color: CHART_COLORS.text,
    boxShadow: '0 8px 22px rgba(23, 32, 51, 0.14)',
  },
  cursor: { stroke: 'rgba(218, 37, 28, 0.3)', strokeWidth: 1 },
}

const GRID_STYLE = { stroke: 'rgba(0, 87, 156, 0.10)', strokeDasharray: '3 3' }

// ============================================================================
// Stat Card with animated trend
// ============================================================================

function StatCard({ icon: Icon, label, value, unit, trend, trendLabel, color, pulse }: {
  icon: typeof Activity
  label: string
  value: ReactNode
  unit?: string
  trend?: number
  trendLabel?: string
  color: string
  pulse?: boolean
}) {
  const isUp = (trend ?? 0) >= 0
  return (
    <div className={cn(
      'relative overflow-hidden rounded-lg border border-border-subtle bg-bg-surface p-4 shadow-sm',
      'transition-all duration-300 hover:border-border-default hover:shadow-[0_8px_22px_rgba(23,32,51,0.10)]',
      'group',
    )}>
      <div className="absolute inset-x-0 top-0 h-1" style={{ backgroundColor: color }} />

      <div className="flex items-start justify-between mb-2">
        <div className={cn(
          'p-2 rounded-md border',
          pulse && 'animate-pulse',
        )} style={{ backgroundColor: `${color}15`, borderColor: `${color}30` }}>
          <Icon size={16} style={{ color }} />
        </div>
        {trend !== undefined && (
          <div className={cn(
            'flex items-center gap-0.5 text-[10px] font-mono font-semibold px-1.5 py-0.5 rounded-full',
            isUp ? 'bg-[#00579C]/10 text-[#00579C]' : 'bg-[#DA251C]/10 text-[#DA251C]',
          )}>
            {isUp ? <ArrowUpRight size={10} /> : <ArrowDownRight size={10} />}
            {Math.abs(trend).toFixed(1)}%
          </div>
        )}
      </div>

      <p className="text-[10px] text-text-muted uppercase tracking-wider font-medium mb-1">{label}</p>
      <div className="flex items-baseline gap-1">
        <span className="text-xl font-bold text-text-primary font-mono tabular-nums">{value}</span>
        {unit && <span className="text-[10px] text-text-muted">{unit}</span>}
      </div>
      {trendLabel && <p className="text-[9px] text-text-muted mt-1">{trendLabel}</p>}
    </div>
  )
}
function formatReportInr(paisa: number): string {
  const rupees = Number(paisa || 0) / 100
  if (rupees >= 10_000_000) return `INR ${(rupees / 10_000_000).toFixed(1)}Cr`
  if (rupees >= 100_000) return `INR ${(rupees / 100_000).toFixed(1)}L`
  return `INR ${rupees.toLocaleString('en-IN', { maximumFractionDigits: 0 })}`
}

function LatestEvaluationPanel({ report }: { report?: EventLabReportActivity }) {
  if (!report) return null
  const riskRows = Object.entries(report.report.risk_score_components && typeof report.report.risk_score_components === 'object'
    ? report.report.risk_score_components as Record<string, number>
    : {}).filter(([, value]) => Number(value) > 0)
  const channelRows = Object.entries(report.report.channel_amount_mix && typeof report.report.channel_amount_mix === 'object'
    ? report.report.channel_amount_mix as Record<string, number>
    : {})
  const maxRisk = Math.max(0.01, ...riskRows.map(([, value]) => Number(value) || 0))
  const maxAmount = Math.max(1, ...channelRows.map(([, value]) => Number(value) || 0))
  return (
    <section
      data-testid="latest-event-lab-evaluation-panel"
      className="rounded-lg border border-[#00579C]/25 bg-[linear-gradient(135deg,#ffffff,#eef6ff_64%,#fff5f4)] p-4 shadow-sm"
    >
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div className="min-w-0">
          <div className="flex items-center gap-2 text-[10px] font-black uppercase tracking-[0.14em] text-[#00579C]">
            <Radio size={12} className="animate-pulse text-[#DA251C]" />
            Latest evaluated Event Lab run synced into analytics
          </div>
          <p
            data-testid="latest-event-lab-evaluation-sync-counts"
            className="mt-1 max-w-4xl text-[10px] leading-relaxed text-[#4b5d76]"
          >
            {report.templateTitle} produced {report.eventCount} events across {report.routeLabel || 'the configured route'}.
            Analytics is reading the same completed report object as the Event Lab modal, graph lifecycle, activity stream, and top ribbon.
            Report-derived sync counts: risk components {riskRows.length}, report channels {channelRows.length}, sync targets {report.syncTargets.length}.
          </p>
        </div>
        <div className="grid min-w-[520px] grid-cols-2 gap-2 sm:grid-cols-4 xl:grid-cols-7">
          <ReportMetric label="Risk" value={`${report.riskTier} ${report.riskScore != null ? Math.round(report.riskScore * 100) : 0}%`} tone="red" />
          <ReportMetric label="Exposure" value={formatReportInr(report.exposurePaisa)} tone="red" />
          <ReportMetric label="Accounts" value={String(report.uniqueAccountCount)} />
          <ReportMetric label="Sync" value={`${report.syncTargets.filter((target) => target.status === 'synced').length}/${report.syncTargets.length}`} tone="green" />
          <ReportMetric label="Risk components" value={String(riskRows.length)} />
          <ReportMetric label="Report channels" value={String(channelRows.length)} />
          <ReportMetric label="Sync targets" value={String(report.syncTargets.length)} tone="green" />
        </div>
      </div>
      <div className="mt-3 grid gap-3 lg:grid-cols-[minmax(0,1fr)_minmax(0,1fr)_360px]">
        <div className="rounded-md border border-border-subtle bg-white p-3">
          <div className="mb-2 text-[9px] font-bold uppercase tracking-[0.12em] text-text-primary">Risk score components</div>
          <div className="space-y-1.5">
            {riskRows.slice(0, 7).map(([label, value]) => (
              <div key={label} className="grid grid-cols-[116px_minmax(0,1fr)_34px] items-center gap-2 text-[8px]">
                <span className="truncate font-bold uppercase tracking-wide text-text-muted">{label.replaceAll('_', ' ')}</span>
                <div className="h-3 overflow-hidden rounded bg-[#eef5fb]">
                  <div className="h-full rounded bg-[linear-gradient(90deg,#00579C,#DA251C)]" style={{ width: `${Math.max(4, (Number(value) / maxRisk) * 100)}%` }} />
                </div>
                <span className="text-right font-mono font-bold text-text-primary">{Math.round(Number(value) * 100)}%</span>
              </div>
            ))}
          </div>
        </div>
        <div className="rounded-md border border-border-subtle bg-white p-3">
          <div className="mb-2 text-[9px] font-bold uppercase tracking-[0.12em] text-text-primary">Channel exposure from report</div>
          <div className="space-y-1.5">
            {channelRows.map(([channel, amount]) => (
              <div key={channel} className="grid grid-cols-[56px_minmax(0,1fr)_76px] items-center gap-2 text-[8px]">
                <span className="font-mono font-bold text-[#00579C]">{channel}</span>
                <div className="h-3 overflow-hidden rounded bg-[#eef5fb]">
                  <div className="h-full rounded bg-[linear-gradient(90deg,#00579C,#DA251C)]" style={{ width: `${Math.max(4, (Number(amount) / maxAmount) * 100)}%` }} />
                </div>
                <span className="text-right font-mono font-bold text-text-primary">{formatReportInr(Number(amount))}</span>
              </div>
            ))}
          </div>
        </div>
        <div className="rounded-md border border-border-subtle bg-white p-3">
          <div className="mb-2 text-[9px] font-bold uppercase tracking-[0.12em] text-text-primary">Prototype sync targets</div>
          <div className="grid gap-1.5">
            {report.syncTargets.map((target) => (
              <div key={target.key} className="flex items-center justify-between gap-2 rounded border border-border-subtle bg-[#f4f8fc] px-2 py-1.5 text-[8px]">
                <span className="truncate font-bold uppercase tracking-wide text-text-secondary">{target.label}</span>
                <span className={cn('font-mono font-bold', target.status === 'synced' ? 'text-alert-low' : 'text-text-muted')}>{target.count}</span>
              </div>
            ))}
          </div>
        </div>
      </div>
    </section>
  )
}

function ReportMetric({ label, value, tone = 'blue' }: { label: string; value: string; tone?: 'blue' | 'red' | 'green' }) {
  return (
    <div className="rounded-md border border-border-subtle bg-white px-2 py-1.5">
      <div className="text-[7px] font-bold uppercase tracking-[0.12em] text-text-muted">{label}</div>
      <div className={cn('mt-0.5 truncate font-mono text-[10px] font-black', tone === 'red' ? 'text-[#DA251C]' : tone === 'green' ? 'text-alert-low' : 'text-[#00579C]')}>{value}</div>
    </div>
  )
}

function reportObject(record: EventLabReportActivity, key: string): Record<string, unknown> {
  const value = record.report[key]
  return value && typeof value === 'object' && !Array.isArray(value)
    ? value as Record<string, unknown>
    : {}
}

function reportArray(record: EventLabReportActivity, key: string): Array<Record<string, unknown>> {
  const value = record.report[key]
  return Array.isArray(value) ? value.filter((item): item is Record<string, unknown> => Boolean(item) && typeof item === 'object') : []
}

function numericEntries(record: Record<string, unknown>): Array<[string, number]> {
  return Object.entries(record)
    .map(([key, value]) => [key, Number(value)] as [string, number])
    .filter(([, value]) => Number.isFinite(value) && value > 0)
}

function aggregateReportEntries(
  reports: EventLabReportActivity[],
  key: string,
): Array<[string, number]> {
  const totals = new Map<string, number>()
  reports.forEach((report) => {
    numericEntries(reportObject(report, key)).forEach(([label, value]) => {
      totals.set(label, (totals.get(label) ?? 0) + value)
    })
  })
  return [...totals.entries()].sort((a, b) => b[1] - a[1])
}

function EventLabReportHistoryFabric({ reports }: { reports: EventLabReportActivity[] }) {
  if (reports.length === 0) return null
  const chronological = [...reports].reverse().slice(-12)
  const latest = reports[0]
  const maxExposure = Math.max(1, ...chronological.map((report) => report.exposurePaisa))
  const riskPoints = chronological.map((report, index) => {
    const x = chronological.length <= 1 ? 50 : 7 + (index / (chronological.length - 1)) * 86
    const y = 38 - Math.max(0, Math.min(1, Number(report.riskScore ?? 0))) * 30
    return { x, y }
  })
  const riskPath = riskPoints.map((point, index) => `${index === 0 ? 'M' : 'L'} ${point.x.toFixed(2)} ${point.y.toFixed(2)}`).join(' ')
  const riskComponents = aggregateReportEntries(reports, 'risk_score_components').slice(0, 8)
  const channelExposure = aggregateReportEntries(reports, 'channel_amount_mix').slice(0, 7)
  const maxComponent = Math.max(1, ...riskComponents.map(([, value]) => value))
  const maxChannel = Math.max(1, ...channelExposure.map(([, value]) => value))
  const requiredStages = ['ingested', 'ml_scored', 'graph_investigated', 'cb_evaluated', 'pipeline_dispatched', 'qwen_context_loaded', 'evaluation_complete', 'evidence_ready']
  const routeTotals = new Map<string, { exposure: number; count: number; risk: number }>()
  reports.forEach((report) => {
    const key = report.routeLabel || String(report.report.route_label ?? 'unlabelled route')
    const current = routeTotals.get(key) ?? { exposure: 0, count: 0, risk: 0 }
    current.exposure += report.exposurePaisa
    current.count += 1
    current.risk += Number(report.riskScore ?? 0)
    routeTotals.set(key, current)
  })
  const routeRows = [...routeTotals.entries()]
    .map(([route, value]) => ({ route, ...value, avgRisk: value.risk / Math.max(1, value.count) }))
    .sort((a, b) => b.exposure - a.exposure)
    .slice(0, 5)
  const maxRouteExposure = Math.max(1, ...routeRows.map((row) => row.exposure))
  const counterState = reports.reduce(
    (acc, report) => {
      const state = reportObject(report, 'countermeasure_status')
      acc.pending += Number(state.pending ?? 0)
      acc.executed += Number(state.executed ?? 0)
      acc.rejected += Number(state.rejected ?? 0)
      return acc
    },
    { pending: 0, executed: 0, rejected: 0 },
  )
  const counterTotal = Math.max(1, counterState.pending + counterState.executed + counterState.rejected)
  const evidenceRows = reports.reduce((sum, report) => sum + reportArray(report, 'evidence_matrix').length, 0)
  const routeSegments = reports.reduce((sum, report) => sum + reportArray(report, 'route_segments').length, 0)

  return (
    <section className="rounded-lg border border-[#00579C]/25 bg-white p-4 shadow-sm">
      <div className="mb-3 flex flex-wrap items-start justify-between gap-3">
        <div className="min-w-0">
          <div className="flex items-center gap-2 text-[10px] font-black uppercase tracking-[0.14em] text-[#00579C]">
            <Network size={13} className="text-[#DA251C]" />
            Event Lab report intelligence fabric
          </div>
          <p className="mt-1 max-w-4xl text-[10px] leading-relaxed text-[#4b5d76]">
            Analytics is aggregating {reports.length} completed autonomous evaluations. These trend lines, heatmaps,
            route corridors, and countermeasure bars are derived from completed backend reports, not fixed dashboard data.
          </p>
        </div>
        <div className="grid min-w-[420px] grid-cols-4 gap-2">
          <ReportMetric label="Reports" value={String(reports.length)} />
          <ReportMetric label="Latest Risk" value={`${latest.riskTier} ${Math.round(Number(latest.riskScore ?? 0) * 100)}%`} tone={Number(latest.riskScore ?? 0) >= 0.72 ? 'red' : 'blue'} />
          <ReportMetric label="Evidence Rows" value={String(evidenceRows)} />
          <ReportMetric label="Route Legs" value={String(routeSegments)} tone="green" />
        </div>
      </div>

      <div className="grid gap-3 xl:grid-cols-[minmax(0,1.15fr)_minmax(0,0.85fr)]">
        <div className="grid gap-3 lg:grid-cols-2">
          <div className="rounded-md border border-border-subtle bg-[#f7fbff] p-3">
            <div className="mb-2 flex items-center justify-between gap-2">
              <div className="text-[9px] font-bold uppercase tracking-[0.12em] text-text-primary">Risk and exposure trend</div>
              <span className="font-mono text-[8px] text-alert-low">live history</span>
            </div>
            <svg viewBox="0 0 100 44" className="h-40 w-full rounded border border-[#00579C]/12 bg-white">
              {[10, 20, 30, 40].map((y) => <line key={y} x1="6" x2="96" y1={y} y2={y} stroke="#d9e6f3" strokeDasharray="2 2" strokeWidth="0.35" />)}
              {chronological.map((report, index) => {
                const x = chronological.length <= 1 ? 50 : 7 + (index / (chronological.length - 1)) * 86
                const height = Math.max(2, (report.exposurePaisa / maxExposure) * 28)
                return <rect key={report.runId} x={x - 2.2} y={38 - height} width="4.4" height={height} rx="1" fill="#00579C" opacity="0.34" />
              })}
              {riskPath && <path d={riskPath} fill="none" stroke="#DA251C" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />}
              {riskPoints.map((point, index) => <circle key={`${point.x}-${index}`} cx={point.x} cy={point.y} r="1.6" fill="#DA251C" />)}
              <text x="7" y="7" fontSize="3.2" fontFamily="monospace" fill="#52657d">red=risk, blue=exposure</text>
            </svg>
          </div>

          <div className="rounded-md border border-border-subtle bg-[#f7fbff] p-3">
            <div className="mb-2 text-[9px] font-bold uppercase tracking-[0.12em] text-text-primary">Backend stage heatmap</div>
            <div className="space-y-1.5">
              {chronological.slice(-8).map((report) => {
                const coverage = reportObject(report, 'stage_coverage')
                return (
                  <div key={report.runId} className="grid grid-cols-[82px_minmax(0,1fr)] items-center gap-2">
                    <span className="truncate font-mono text-[8px] font-bold text-text-muted">{report.runId.slice(0, 10)}</span>
                    <div className="grid grid-cols-8 gap-1">
                      {requiredStages.map((stage) => {
                        const value = Number(coverage[stage] ?? 0)
                        return (
                          <div
                            key={stage}
                            title={`${stage}: ${value}`}
                            className="h-5 rounded border border-white"
                            style={{ backgroundColor: value > 0 ? `rgba(0,87,156,${Math.min(0.92, 0.26 + value * 0.12)})` : 'rgba(198,211,227,0.45)' }}
                          />
                        )
                      })}
                    </div>
                  </div>
                )
              })}
            </div>
            <div className="mt-2 grid grid-cols-4 gap-1 text-[7px] font-bold uppercase tracking-wide text-text-muted">
              <span>ingest</span><span>ml/graph</span><span>qwen</span><span>evidence</span>
            </div>
          </div>
        </div>

        <div className="grid gap-3 lg:grid-cols-2 xl:grid-cols-1">
          <div className="rounded-md border border-border-subtle bg-[#fffafa] p-3">
            <div className="mb-2 text-[9px] font-bold uppercase tracking-[0.12em] text-text-primary">Risk component leaderboard</div>
            <div className="space-y-1.5">
              {riskComponents.map(([label, value]) => (
                <div key={label} className="grid grid-cols-[112px_minmax(0,1fr)_36px] items-center gap-2 text-[8px]">
                  <span className="truncate font-bold uppercase tracking-wide text-text-secondary">{label.replaceAll('_', ' ')}</span>
                  <div className="h-3 overflow-hidden rounded bg-white">
                    <div className="h-full rounded bg-[linear-gradient(90deg,#00579C,#DA251C)]" style={{ width: `${Math.max(4, (value / maxComponent) * 100)}%` }} />
                  </div>
                  <span className="text-right font-mono font-bold text-[#DA251C]">{Math.round(value * 100)}</span>
                </div>
              ))}
            </div>
          </div>

          <div className="rounded-md border border-border-subtle bg-[#f7fbff] p-3">
            <div className="mb-2 text-[9px] font-bold uppercase tracking-[0.12em] text-text-primary">Countermeasure state stack</div>
            <div className="h-7 overflow-hidden rounded border border-border-subtle bg-white">
              <div className="flex h-full">
                <div className="bg-[#f5b400]" style={{ width: `${(counterState.pending / counterTotal) * 100}%` }} />
                <div className="bg-[#0f9f6e]" style={{ width: `${(counterState.executed / counterTotal) * 100}%` }} />
                <div className="bg-[#00579C]" style={{ width: `${(counterState.rejected / counterTotal) * 100}%` }} />
              </div>
            </div>
            <div className="mt-2 grid grid-cols-3 gap-2">
              <ReportMetric label="Pending" value={String(counterState.pending)} />
              <ReportMetric label="Executed" value={String(counterState.executed)} tone="green" />
              <ReportMetric label="Rejected" value={String(counterState.rejected)} />
            </div>
          </div>
        </div>
      </div>

      <div className="mt-3 grid gap-3 lg:grid-cols-[minmax(0,1fr)_minmax(0,1fr)]">
        <div className="rounded-md border border-border-subtle bg-[#f7fbff] p-3">
          <div className="mb-2 text-[9px] font-bold uppercase tracking-[0.12em] text-text-primary">Route corridor exposure map</div>
          <div className="space-y-1.5">
            {routeRows.map((row) => (
              <div key={row.route} className="rounded border border-border-subtle bg-white p-2">
                <div className="mb-1 flex items-center justify-between gap-2">
                  <span className="truncate text-[8px] font-bold uppercase tracking-wide text-text-secondary">{row.route}</span>
                  <span className="font-mono text-[8px] font-bold text-[#DA251C]">{formatReportInr(row.exposure)}</span>
                </div>
                <div className="h-3 overflow-hidden rounded bg-[#eef5fb]">
                  <div className="h-full rounded bg-[linear-gradient(90deg,#00579C,#DA251C)]" style={{ width: `${Math.max(5, (row.exposure / maxRouteExposure) * 100)}%` }} />
                </div>
                <div className="mt-1 flex items-center justify-between font-mono text-[7px] text-text-muted">
                  <span>{row.count} runs</span>
                  <span>avg risk {Math.round(row.avgRisk * 100)}%</span>
                </div>
              </div>
            ))}
          </div>
        </div>

        <div className="rounded-md border border-border-subtle bg-[#f7fbff] p-3">
          <div className="mb-2 text-[9px] font-bold uppercase tracking-[0.12em] text-text-primary">Channel exposure from reports</div>
          <div className="space-y-1.5">
            {channelExposure.map(([channel, amount]) => (
              <div key={channel} className="grid grid-cols-[64px_minmax(0,1fr)_86px] items-center gap-2 text-[8px]">
                <span className="font-mono font-bold text-[#00579C]">{channel}</span>
                <div className="h-4 overflow-hidden rounded bg-white">
                  <div className="h-full rounded bg-[linear-gradient(90deg,#00579C,#DA251C)]" style={{ width: `${Math.max(4, (amount / maxChannel) * 100)}%` }} />
                </div>
                <span className="text-right font-mono font-bold text-text-primary">{formatReportInr(amount)}</span>
              </div>
            ))}
          </div>
        </div>
      </div>
    </section>
  )
}

// ============================================================================
// Chart Card wrapper
// ============================================================================

function ChartCard({ title, subtitle, icon: Icon, children, className, badge, live }: {
  title: string
  subtitle?: string
  icon: typeof Activity
  children: React.ReactNode
  className?: string
  badge?: string
  live?: boolean
}) {
  const isLive = live === true
  return (
    <div className={cn(
      'rounded-lg border border-border-subtle bg-bg-surface shadow-sm overflow-hidden',
      className,
    )}>
      <div className="flex items-center justify-between px-4 pt-3 pb-1">
        <div className="flex items-center gap-2">
          <Icon size={14} className="text-accent-primary" />
          <h3 className="text-xs font-semibold text-text-primary">{title}</h3>
          {badge && (
            <span className="text-[8px] font-mono px-1.5 py-0.5 rounded-full bg-accent-primary/15 text-accent-primary border border-accent-primary/20">
              {badge}
            </span>
          )}
        </div>
        {subtitle && <span className="text-[9px] text-text-muted">{subtitle}</span>}
        <div className="flex items-center gap-1">
          <Radio size={8} className={isLive ? 'text-alert-low animate-pulse' : 'text-text-muted'} />
          <span className={cn(
            'text-[8px] font-mono',
            isLive ? 'text-alert-low' : 'text-accent-primary',
          )}>
            {isLive ? 'LIVE' : 'SYNCING'}
          </span>
        </div>
      </div>
      <div className="px-2 pb-3">
        {children}
      </div>
    </div>
  )
}

// ============================================================================
// Custom Heatmap
// ============================================================================

const DAYS_SHORT = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']

function AnalyticsEmptyState({ title, detail, height = 220 }: { title: string; detail: string; height?: number }) {
  return (
    <div
      className="flex flex-col items-center justify-center rounded-md border border-[#00579C]/20 bg-[linear-gradient(135deg,rgba(0,87,156,0.08),rgba(255,255,255,0.9),rgba(218,37,28,0.05))] px-4 text-center shadow-inner"
      style={{ minHeight: height }}
    >
      <div className="text-[10px] font-bold uppercase tracking-[0.14em] text-[#00579C]">{title}</div>
      <div className="mt-1 max-w-sm text-[9px] leading-relaxed text-[#4b5d76]">{detail}</div>
    </div>
  )
}

function hasPositiveMetric<T>(rows: T[], read: (row: T) => number[]): boolean {
  return rows.some((row) => read(row).some((value) => Number.isFinite(value) && value > 0))
}

interface ActivityDerivedAnalytics {
  transactionVolume: TransactionVolumePoint[]
  fraudRate: FraudRatePoint[]
  latencyMetrics: LatencyPoint[]
  riskHeatmap: RiskHeatmapCell[]
  fraudTypologies: FraudTypology[]
  velocityDistribution: VelocityBucket[]
  amountDistribution: AmountDistribution[]
  accountRiskBands: AccountRiskBand[]
  threatEvents: ThreatEvent[]
  alertFunnel: AlertFunnel[]
  modelPerformance: ModelPerformance[]
  avgResponseMs: number
  modelAccuracy: number
  truePositiveRate: number
  falsePositiveRate: number
  hasEvidence: boolean
}

const ACTIVITY_TYPOLOGY_COLORS = ['#DA251C', '#00579C', '#B51A13', '#003F73', '#f5b400', '#0f9f6e']
const ACTIVITY_AMOUNT_BUCKETS = [
  '< INR 500',
  'INR 500-2K',
  'INR 2K-10K',
  'INR 10K-50K',
  'INR 50K-1L',
  'INR 1L-5L',
  'INR 5L-10L',
  '> INR 10L',
]
const ACTIVITY_RISK_BANDS: AccountRiskBand[] = [
  { band: 'Low (0-20)', count: 0, value: 0.15, color: '#0f9f6e' },
  { band: 'Medium (20-40)', count: 0, value: 0.30, color: '#2f79b5' },
  { band: 'Elevated (40-60)', count: 0, value: 0.50, color: '#f5b400' },
  { band: 'High (60-80)', count: 0, value: 0.70, color: '#f97316' },
  { band: 'Critical (80-100)', count: 0, value: 0.90, color: '#DA251C' },
]

function clampPercent(value: number): number {
  if (!Number.isFinite(value)) return 0
  return Math.min(100, Math.max(0, Number(value.toFixed(2))))
}

function pctOf(part: number, total: number): number {
  return total > 0 ? clampPercent((part / total) * 100) : 0
}

function msFromBackendSeconds(raw: unknown): number {
  const n = Number(raw)
  if (!Number.isFinite(n) || n <= 0) return 0
  return n > 1_000_000_000_000 ? n : n * 1000
}

function formatActivityTime(timestamp: number): string {
  const d = new Date(timestamp)
  return `${d.getHours().toString().padStart(2, '0')}:${d.getMinutes().toString().padStart(2, '0')}:${d.getSeconds().toString().padStart(2, '0')}`
}

function labelizeActivity(raw: string): string {
  const cleaned = raw.replace(/[_.-]+/g, ' ').replace(/\s+/g, ' ').trim()
  if (!cleaned) return ''
  return cleaned.replace(/\b\w/g, (m) => m.toUpperCase())
}

function lifecycleTimestampMs(lifecycle: EventLifecycle): number {
  const timestamps = [
    msFromBackendSeconds(lifecycle.firstSeen),
    ...lifecycle.stages.map((stage) => msFromBackendSeconds(stage.timestamp)),
  ].filter((timestamp) => timestamp > 0)
  return timestamps.length > 0 ? Math.max(...timestamps) : 0
}

function lifecycleRiskPercent(lifecycle: EventLifecycle): number {
  const rawRisk = Number(lifecycle.riskScore)
  if (Number.isFinite(rawRisk) && rawRisk > 0) {
    return clampPercent(rawRisk <= 1 ? rawRisk * 100 : rawRisk)
  }
  const verdict = String(lifecycle.verdict ?? '').toLowerCase()
  if (lifecycle.fraudLabel > 0 || verdict.includes('fraud')) return 82
  if (verdict.includes('suspicious') || lifecycle.nluEscalated) return 58
  return 18
}

function lifecycleIsFraudish(lifecycle: EventLifecycle): boolean {
  const verdict = String(lifecycle.verdict ?? '').toLowerCase()
  return (
    lifecycle.fraudLabel > 0 ||
    lifecycleRiskPercent(lifecycle) >= 55 ||
    verdict.includes('fraud') ||
    verdict.includes('suspicious') ||
    Boolean(lifecycle.nluEscalated)
  )
}

function lifecycleIsBlocked(lifecycle: EventLifecycle): boolean {
  const text = [
    lifecycle.recommendedAction,
    lifecycle.analystStatus,
    lifecycle.verdict,
  ].join(' ').toLowerCase()
  return /freeze|frozen|hold|hotlist|block|approved|closed/.test(text)
}

function stageDurationValues(lifecycle: EventLifecycle): number[] {
  const explicit = lifecycle.stages
    .map((stage) => Number(stage.durationMs))
    .filter((duration) => Number.isFinite(duration) && duration > 0 && duration < 120_000)

  const consumerDurations = (lifecycle.pipelineConsumers ?? [])
    .map((consumer) => Number(consumer.duration_ms))
    .filter((duration) => Number.isFinite(duration) && duration > 0 && duration < 120_000)

  const timestamps = lifecycle.stages
    .map((stage) => msFromBackendSeconds(stage.timestamp))
    .filter((timestamp) => timestamp > 0)
    .sort((a, b) => a - b)

  const deltas: number[] = []
  for (let index = 1; index < timestamps.length; index += 1) {
    const delta = timestamps[index] - timestamps[index - 1]
    if (delta > 0 && delta < 120_000) deltas.push(delta)
  }

  const total = Number(lifecycle.totalDurationMs)
  if (Number.isFinite(total) && total > 0 && total < 120_000) {
    deltas.push(total)
  }

  return [...explicit, ...consumerDurations, ...deltas]
}

function percentile(values: number[], quantile: number): number {
  if (values.length === 0) return 0
  const sorted = [...values].sort((a, b) => a - b)
  const index = (sorted.length - 1) * quantile
  const lower = Math.floor(index)
  const upper = Math.ceil(index)
  if (lower === upper) return Math.round(sorted[lower])
  return Math.round(sorted[lower] + (sorted[upper] - sorted[lower]) * (index - lower))
}

function average(values: number[]): number {
  const clean = values.filter((value) => Number.isFinite(value) && value > 0)
  if (clean.length === 0) return 0
  return Number((clean.reduce((sum, value) => sum + value, 0) / clean.length).toFixed(2))
}

function bucketLifecycles(lifecycles: EventLifecycle[], targetSize = 8): EventLifecycle[][] {
  if (lifecycles.length === 0) return []
  const chronological = [...lifecycles].sort((a, b) => {
    const aTs = lifecycleTimestampMs(a)
    const bTs = lifecycleTimestampMs(b)
    return (aTs || 0) - (bTs || 0)
  })
  const bucketCount = Math.min(14, Math.max(1, Math.ceil(chronological.length / targetSize)))
  const chunkSize = Math.max(1, Math.ceil(chronological.length / bucketCount))
  const buckets: EventLifecycle[][] = []
  for (let index = 0; index < chronological.length; index += chunkSize) {
    buckets.push(chronological.slice(index, index + chunkSize))
  }
  return buckets
}

function bucketTimestamp(group: EventLifecycle[], index: number, total: number): number {
  const timestamp = Math.max(...group.map(lifecycleTimestampMs), 0)
  if (timestamp > 0) return timestamp
  return Date.now() - (total - index - 1) * 15_000
}

function activityAmountBucket(amountPaisa: number): AmountDistribution['range'] {
  const inr = amountPaisa / 100
  if (inr < 500) return '< INR 500'
  if (inr < 2_000) return 'INR 500-2K'
  if (inr < 10_000) return 'INR 2K-10K'
  if (inr < 50_000) return 'INR 10K-50K'
  if (inr < 100_000) return 'INR 50K-1L'
  if (inr < 500_000) return 'INR 1L-5L'
  if (inr < 1_000_000) return 'INR 5L-10L'
  return '> INR 10L'
}

function deriveLiveAnalyticsFromActivity(events: Map<string, EventLifecycle>, orderedIds: string[]): ActivityDerivedAnalytics {
  const lifecycles = orderedIds
    .map((id) => events.get(id))
    .filter((event): event is EventLifecycle => Boolean(event))
    .slice(0, 200)
  const buckets = bucketLifecycles(lifecycles)

  const transactionVolume = buckets.map((group, index): TransactionVolumePoint => {
    const fraudish = group.filter(lifecycleIsFraudish)
    const blocked = fraudish.filter(lifecycleIsBlocked).length
    const fraudulent = fraudish.filter((lifecycle) => lifecycle.fraudLabel > 0 || String(lifecycle.verdict ?? '').toLowerCase().includes('fraud')).length
    const suspicious = Math.max(0, fraudish.length - blocked - fraudulent)
    const timestamp = bucketTimestamp(group, index, buckets.length)
    return {
      time: formatActivityTime(timestamp),
      timestamp,
      legitimate: Math.max(0, group.length - fraudish.length),
      suspicious,
      fraudulent,
      blocked,
    }
  })

  const fraudRate = buckets.map((group, index): FraudRatePoint => {
    const flagged = group.filter(lifecycleIsFraudish).length
    const timestamp = bucketTimestamp(group, index, buckets.length)
    return {
      time: formatActivityTime(timestamp),
      timestamp,
      rate: pctOf(flagged, group.length),
      baseline: null,
      threshold: null,
    }
  })

  const latencyMetrics = buckets
    .map((group, index): LatencyPoint | null => {
      const durations = group.flatMap(stageDurationValues)
      if (durations.length === 0) return null
      const llmDurations = group
        .flatMap((lifecycle) => lifecycle.stages.filter((stage) => stage.stage === 'llm_started').map((stage) => Number(stage.durationMs)))
        .filter((duration) => Number.isFinite(duration) && duration > 0)
      const timestamp = bucketTimestamp(group, index, buckets.length)
      return {
        time: formatActivityTime(timestamp),
        timestamp,
        p50: percentile(durations, 0.5),
        p95: percentile(durations, 0.95),
        p99: percentile(durations, 0.99),
        mlInference: Math.round(average(llmDurations.length > 0 ? llmDurations : durations)),
      }
    })
    .filter((point): point is LatencyPoint => Boolean(point))

  const heatmap = new Map<string, { day: string; hour: number; total: number; count: number }>()
  lifecycles.forEach((lifecycle) => {
    const timestamp = lifecycleTimestampMs(lifecycle)
    if (timestamp <= 0) return
    const d = new Date(timestamp)
    const day = d.toLocaleDateString('en-US', { weekday: 'short' })
    const hour = d.getHours()
    const key = `${day}-${hour}`
    const pressure = Math.min(100, lifecycleRiskPercent(lifecycle) + Math.log10((lifecycle.amountPaisa / 100) + 1) * 2 + lifecycle.stages.length * 2)
    const current = heatmap.get(key) ?? { day, hour, total: 0, count: 0 }
    current.total += pressure
    current.count += 1
    heatmap.set(key, current)
  })
  const riskHeatmap: RiskHeatmapCell[] = [...heatmap.values()].map((cell) => ({
    day: cell.day,
    hour: cell.hour,
    value: Math.round(cell.total / Math.max(1, cell.count)),
  }))

  const typologyCounts = new Map<string, number>()
  lifecycles.filter(lifecycleIsFraudish).forEach((lifecycle) => {
    const label =
      labelizeActivity(lifecycle.fraudTypology ?? '') ||
      labelizeActivity(lifecycle.attackLabel ?? '') ||
      labelizeActivity(lifecycle.scenarioId ?? '') ||
      'Lifecycle Anomaly'
    typologyCounts.set(label, (typologyCounts.get(label) ?? 0) + 1)
  })
  const typologyTotal = [...typologyCounts.values()].reduce((sum, count) => sum + count, 0)
  const fraudTypologies: FraudTypology[] = [...typologyCounts.entries()]
    .sort((a, b) => b[1] - a[1])
    .map(([name, count], index) => ({
      name,
      count,
      percentage: pctOf(count, typologyTotal),
      color: ACTIVITY_TYPOLOGY_COLORS[index % ACTIVITY_TYPOLOGY_COLORS.length],
      trend: null,
    }))

  const accountVelocity = new Map<string, { count: number; fraud: number }>()
  lifecycles.forEach((lifecycle) => {
    const account = lifecycle.sender || lifecycle.receiver || lifecycle.txnId
    const current = accountVelocity.get(account) ?? { count: 0, fraud: 0 }
    current.count += 1
    current.fraud += lifecycleIsFraudish(lifecycle) ? 1 : 0
    accountVelocity.set(account, current)
  })
  const velocityDistribution: VelocityBucket[] = [
    { range: '0-5 tx/window', normal: 0, suspicious: 0, fraudulent: 0 },
    { range: '5-15 tx/window', normal: 0, suspicious: 0, fraudulent: 0 },
    { range: '15-30 tx/window', normal: 0, suspicious: 0, fraudulent: 0 },
    { range: '30-60 tx/window', normal: 0, suspicious: 0, fraudulent: 0 },
    { range: '60+ tx/window', normal: 0, suspicious: 0, fraudulent: 0 },
  ]
  accountVelocity.forEach((value) => {
    const index = value.count < 5 ? 0 : value.count < 15 ? 1 : value.count < 30 ? 2 : value.count < 60 ? 3 : 4
    if (value.fraud > 0) velocityDistribution[index].fraudulent += 1
    else if (value.count >= 15) velocityDistribution[index].suspicious += 1
    else velocityDistribution[index].normal += 1
  })

  const amountBuckets = new Map<string, { count: number; fraud: number }>()
  ACTIVITY_AMOUNT_BUCKETS.forEach((bucket) => amountBuckets.set(bucket, { count: 0, fraud: 0 }))
  lifecycles.filter((lifecycle) => lifecycle.amountPaisa > 0).forEach((lifecycle) => {
    const range = activityAmountBucket(lifecycle.amountPaisa)
    const current = amountBuckets.get(range) ?? { count: 0, fraud: 0 }
    current.count += 1
    current.fraud += lifecycleIsFraudish(lifecycle) ? 1 : 0
    amountBuckets.set(range, current)
  })
  const amountDistribution: AmountDistribution[] = ACTIVITY_AMOUNT_BUCKETS.map((range) => {
    const bucket = amountBuckets.get(range) ?? { count: 0, fraud: 0 }
    return {
      range,
      count: bucket.count,
      fraudCount: bucket.fraud,
      fraudRate: pctOf(bucket.fraud, bucket.count),
    }
  })

  const accountRiskBands = ACTIVITY_RISK_BANDS.map((band) => ({ ...band }))
  lifecycles.forEach((lifecycle) => {
    const risk = lifecycleRiskPercent(lifecycle)
    const index = risk < 20 ? 0 : risk < 40 ? 1 : risk < 60 ? 2 : risk < 80 ? 3 : 4
    accountRiskBands[index].count += 1
  })

  const threatEvents: ThreatEvent[] = lifecycles
    .filter((lifecycle) => lifecycleIsFraudish(lifecycle) || lifecycle.recommendedAction)
    .slice(0, 24)
    .map((lifecycle): ThreatEvent => {
      const risk = lifecycleRiskPercent(lifecycle)
      const timestamp = lifecycleTimestampMs(lifecycle)
      const blocked = lifecycleIsBlocked(lifecycle)
      return {
        id: lifecycle.txnId,
        timestamp,
        timestampSource: timestamp > 0 ? 'backend' : 'unavailable',
        severity: risk >= 85 ? 'critical' : risk >= 70 ? 'high' : risk >= 45 ? 'medium' : 'low',
        city: 'PayFlow',
        country: 'live lifecycle',
        sourceLabel: `${lifecycle.sender || 'sender'} -> ${lifecycle.receiver || 'receiver'}`,
        description: lifecycle.reasoningSummary || lifecycle.recommendedAction || `Risk ${Math.round(risk)} across ${lifecycle.stages.length} pipeline stages`,
        attackType: labelizeActivity(lifecycle.fraudTypology ?? lifecycle.attackLabel ?? lifecycle.scenarioId ?? 'Lifecycle Anomaly'),
        amount: Math.round(lifecycle.amountPaisa / 100),
        status: blocked ? 'blocked' : risk >= 70 ? 'escalated' : 'investigating',
      }
    })

  const stageNames: Array<{ key: string; label: string; matches: string[] }> = [
    { key: 'ingested', label: 'Ingested', matches: ['ingested'] },
    { key: 'ml_scored', label: 'ML Scored', matches: ['ml_scored'] },
    { key: 'graph_investigated', label: 'Graph Investigated', matches: ['graph_investigated'] },
    { key: 'cb_evaluated', label: 'Control Evaluated', matches: ['cb_evaluated'] },
    { key: 'llm_started', label: 'Qwen Explained', matches: ['llm_started'] },
    { key: 'verdict', label: 'Verdict / Dispatch', matches: ['verdict', 'pipeline_dispatched'] },
  ]
  const alertFunnel: AlertFunnel[] = stageNames.map((stage) => {
    const count = lifecycles.filter((lifecycle) => lifecycle.stages.some((item) => stage.matches.includes(item.stage))).length
    return { stage: stage.label, count, percentage: pctOf(count, lifecycles.length) }
  })

  const confidenceValues = lifecycles
    .map((lifecycle) => Number(lifecycle.confidence))
    .filter((confidence) => Number.isFinite(confidence) && confidence > 0)
    .map((confidence) => clampPercent(confidence <= 1 ? confidence * 100 : confidence))
  const riskValues = lifecycles.map(lifecycleRiskPercent).filter((risk) => risk > 0)
  const graphValues = lifecycles
    .map((lifecycle) => lifecycle.consensusScores?.graph)
    .filter((score): score is number => typeof score === 'number' && Number.isFinite(score) && score > 0)
    .map((score) => clampPercent(score <= 1 ? score * 100 : score))
  const completionValues = lifecycles.map((lifecycle) => {
    const uniqueStages = new Set(lifecycle.stages.map((stage) => stage.stage))
    return clampPercent((uniqueStages.size / 7) * 100)
  })
  const mlSignal = average(riskValues)
  const graphSignal = average(graphValues)
  const llmSignal = average(confidenceValues)
  const completionSignal = average(completionValues)
  const ensembleSignal = average([mlSignal, graphSignal, llmSignal, completionSignal])
  const modelPerformance: ModelPerformance[] = lifecycles.length > 0 ? [
    {
      metric: 'Risk Evidence',
      xgboost: mlSignal,
      gnn: graphSignal > 0 ? graphSignal : null,
      ensemble: ensembleSignal,
      llmAgent: llmSignal,
    },
    {
      metric: 'Lifecycle Completion',
      xgboost: completionSignal,
      gnn: graphSignal > 0 ? Math.max(graphSignal, completionSignal) : null,
      ensemble: completionSignal,
      llmAgent: llmSignal > 0 ? Math.max(llmSignal, completionSignal) : 0,
    },
    {
      metric: 'Action Confidence',
      xgboost: pctOf(lifecycles.filter(lifecycleIsFraudish).length, lifecycles.length),
      gnn: graphSignal > 0 ? graphSignal : null,
      ensemble: pctOf(threatEvents.length, lifecycles.length),
      llmAgent: llmSignal,
    },
  ] : []

  const avgResponseMs = Math.round(average(latencyMetrics.map((point) => point.p50)))
  const flagged = lifecycles.filter(lifecycleIsFraudish).length
  const blocked = lifecycles.filter(lifecycleIsBlocked).length

  return {
    transactionVolume,
    fraudRate,
    latencyMetrics,
    riskHeatmap,
    fraudTypologies,
    velocityDistribution,
    amountDistribution,
    accountRiskBands,
    threatEvents,
    alertFunnel,
    modelPerformance,
    avgResponseMs,
    modelAccuracy: Math.round(ensembleSignal),
    truePositiveRate: pctOf(blocked, Math.max(1, flagged)),
    falsePositiveRate: pctOf(lifecycles.length - flagged, Math.max(1, lifecycles.length)),
    hasEvidence: lifecycles.length > 0,
  }
}

function RiskHeatmap({ data }: { data: { hour: number; day: string; value: number }[] }) {
  const maxVal = useMemo(() => Math.max(...data.map(d => d.value), 1), [data])

  const getColor = useCallback((value: number) => {
    const t = value / maxVal
    if (t < 0.25) return `rgba(34, 197, 94, ${0.15 + t * 2})`
    if (t < 0.5) return `rgba(234, 179, 8, ${0.2 + (t - 0.25) * 2})`
    if (t < 0.75) return `rgba(249, 115, 22, ${0.3 + (t - 0.5) * 2})`
    return `rgba(239, 68, 68, ${0.4 + (t - 0.75) * 2.4})`
  }, [maxVal])

  if (data.length === 0) {
    return (
      <AnalyticsEmptyState
        title="No temporal risk buckets"
        detail="The heatmap renders only backend temporal-heatmap buckets. It will populate after the graph contains timestamped transactions inside the selected window."
        height={220}
      />
    )
  }

  return (
    <div className="px-2">
      {/* Hour labels */}
      <div className="flex ml-10 mb-1">
        {Array.from({ length: 24 }, (_, i) => (
          <div key={i} className="flex-1 text-center text-[7px] text-[#8a98aa] font-mono">
            {i % 3 === 0 ? `${i}h` : ''}
          </div>
        ))}
      </div>
      {/* Grid */}
      {DAYS_SHORT.map(day => (
        <div key={day} className="flex items-center gap-1 mb-0.5">
          <span className="text-[8px] text-[#617189] w-8 text-right font-mono">{day}</span>
          <div className="flex flex-1 gap-px">
            {Array.from({ length: 24 }, (_, hour) => {
              const cell = data.find(d => d.day === day && d.hour === hour)
              return (
                <div
                  key={hour}
                  className="flex-1 aspect-square rounded-[2px] transition-colors duration-500 cursor-crosshair"
                  style={{ backgroundColor: getColor(cell?.value ?? 0) }}
                  title={`${day} ${hour}:00 — Risk: ${cell?.value ?? 0}`}
                />
              )
            })}
          </div>
        </div>
      ))}
      {/* Legend */}
      <div className="flex items-center justify-end gap-1 mt-2 mr-2">
        <span className="text-[7px] text-[#8a98aa]">Low</span>
        <div className="flex gap-px">
          {['rgba(34,197,94,0.3)', 'rgba(234,179,8,0.4)', 'rgba(249,115,22,0.5)', 'rgba(239,68,68,0.7)'].map((c, i) => (
            <div key={i} className="w-3 h-2 rounded-[1px]" style={{ backgroundColor: c }} />
          ))}
        </div>
        <span className="text-[7px] text-[#8a98aa]">High</span>
      </div>
    </div>
  )
}

// ============================================================================
// Gauge Component
// ============================================================================

function GaugeChart({ value, max, label, color }: { value: number; max: number; label: string; color: string }) {
  const pct = Math.min(value / max, 1)
  const angle = pct * 180
  const r = 50
  const cx = 60
  const cy = 55
  const endX = cx + r * Math.cos(Math.PI - (angle * Math.PI) / 180)
  const endY = cy - r * Math.sin((angle * Math.PI) / 180)
  const largeArc = angle > 180 ? 1 : 0

  return (
    <div className="flex flex-col items-center">
      <svg viewBox="0 0 120 70" className="w-full max-w-[140px]">
        {/* Background arc */}
        <path
          d={`M ${cx - r} ${cy} A ${r} ${r} 0 0 1 ${cx + r} ${cy}`}
          fill="none" stroke="rgba(148,163,184,0.1)" strokeWidth="8" strokeLinecap="round"
        />
        {/* Value arc */}
        <path
          d={`M ${cx - r} ${cy} A ${r} ${r} 0 ${largeArc} 1 ${endX} ${endY}`}
          fill="none" stroke={color} strokeWidth="8" strokeLinecap="round"
          className="transition-all duration-700"
          style={{ filter: `drop-shadow(0 0 6px ${color}40)` }}
        />
        {/* Value text */}
        <text x={cx} y={cy - 8} textAnchor="middle" fill="white" fontSize="16" fontWeight="bold" fontFamily="monospace">
          {value.toFixed(1)}
        </text>
        <text x={cx} y={cy + 6} textAnchor="middle" fill="rgb(148,163,184)" fontSize="7">
          {label}
        </text>
      </svg>
    </div>
  )
}

// ============================================================================
// Animated Counter
// ============================================================================

function AnimatedCounter({ value, decimals = 0 }: { value: number; decimals?: number }) {
  const [display, setDisplay] = useState(0)
  const ref = useRef(0)

  useEffect(() => {
    const start = ref.current
    const diff = value - start
    const duration = 600
    const startTime = performance.now()

    function animate(now: number) {
      const elapsed = now - startTime
      const t = Math.min(elapsed / duration, 1)
      const eased = 1 - Math.pow(1 - t, 3) // ease-out cubic
      const current = start + diff * eased
      setDisplay(current)
      if (t < 1) requestAnimationFrame(animate)
      else ref.current = value
    }
    requestAnimationFrame(animate)
  }, [value])

  return <>{decimals > 0 ? display.toFixed(decimals) : Math.round(display).toLocaleString()}</>
}

// ============================================================================
// Custom Treemap Content
// ============================================================================

// eslint-disable-next-line @typescript-eslint/no-explicit-any
function TreemapContent(props: any) {
  const { x, y, width, height, name, percentage, color } = props
  if (width < 40 || height < 30) return null
  return (
    <g>
      <rect x={x} y={y} width={width} height={height} rx={4} fill={color} fillOpacity={0.7} stroke="rgba(15,23,42,0.8)" strokeWidth={2} />
      <text x={x + width / 2} y={y + height / 2 - 6} textAnchor="middle" fill="white" fontSize={width > 80 ? 10 : 8} fontWeight="600">
        {name}
      </text>
      <text x={x + width / 2} y={y + height / 2 + 8} textAnchor="middle" fill="rgba(255,255,255,0.7)" fontSize={8}>
        {percentage}%
      </text>
    </g>
  )
}

// ============================================================================
// Main Analytics Page
// ============================================================================

export function AnalyticsPage() {
  const store = useAnalyticsStore()
  const [analyticsDeck, setAnalyticsDeck] = useState<'all' | 'operations' | 'geo' | 'model' | 'forensics'>('all')
  const activityEvents = useActivityStore((state) => state.events)
  const activityOrderedIds = useActivityStore((state) => state.orderedIds)
  const latestEvaluation = useActivityStore((state) => state.eventLabReports[0])
  const eventLabReports = useActivityStore((state) => state.eventLabReports)
  const activityDerived = useMemo(
    () => deriveLiveAnalyticsFromActivity(activityEvents, activityOrderedIds),
    [activityEvents, activityOrderedIds],
  )

  const storeHasTransactionVolume = hasPositiveMetric(
    store.transactionVolume,
    (row) => [row.legitimate, row.suspicious, row.fraudulent, row.blocked],
  )
  const storeHasFraudRate = hasPositiveMetric(store.fraudRate, (row) => [row.rate])
  const storeHasLatencyMetrics = hasPositiveMetric(
    store.latencyMetrics,
    (row) => [row.p50, row.p95, row.p99, row.mlInference],
  )
  const storeHasVelocityDistribution = hasPositiveMetric(
    store.velocityDistribution,
    (row) => [row.normal, row.suspicious, row.fraudulent],
  )
  const storeHasAmountDistribution = hasPositiveMetric(
    store.amountDistribution,
    (row) => [row.count, row.fraudCount, row.fraudRate],
  )
  const storeHasAccountRiskBands = hasPositiveMetric(store.accountRiskBands, (row) => [row.count])
  const storeHasFraudTypologies = store.fraudTypologies.length > 0 && hasPositiveMetric(store.fraudTypologies, (row) => [row.count])
  const storeHasTemporalHeatmap = store.riskHeatmap.length > 0 && hasPositiveMetric(store.riskHeatmap, (row) => [row.value])
  const storeHasThreatEvents = store.threatEvents.length > 0
  const storeHasAlertFunnel = hasPositiveMetric(store.alertFunnel, (row) => [row.count])
  const storeHasModelMetrics = store.modelPerformance.length > 0 && hasPositiveMetric(
    store.modelPerformance,
    (row) => [row.xgboost, row.gnn ?? 0, row.ensemble, row.llmAgent],
  )

  const displayTransactionVolume = storeHasTransactionVolume ? store.transactionVolume : activityDerived.transactionVolume
  const displayFraudRate = storeHasFraudRate ? store.fraudRate : activityDerived.fraudRate
  const displayLatencyMetrics = storeHasLatencyMetrics ? store.latencyMetrics : activityDerived.latencyMetrics
  const displayRiskHeatmap = storeHasTemporalHeatmap ? store.riskHeatmap : activityDerived.riskHeatmap
  const displayFraudTypologies = storeHasFraudTypologies ? store.fraudTypologies : activityDerived.fraudTypologies
  const displayVelocityDistribution = storeHasVelocityDistribution ? store.velocityDistribution : activityDerived.velocityDistribution
  const displayAmountDistribution = storeHasAmountDistribution ? store.amountDistribution : activityDerived.amountDistribution
  const displayAccountRiskBands = storeHasAccountRiskBands ? store.accountRiskBands : activityDerived.accountRiskBands
  const displayThreatEvents = storeHasThreatEvents ? store.threatEvents : activityDerived.threatEvents
  const displayAlertFunnel = storeHasAlertFunnel ? store.alertFunnel : activityDerived.alertFunnel
  const displayModelPerformance = storeHasModelMetrics ? store.modelPerformance : activityDerived.modelPerformance
  const displayModelAccuracy = store.modelAccuracy > 0 ? store.modelAccuracy : activityDerived.modelAccuracy
  const displayTruePositiveRate = store.truePositiveRate > 0 ? store.truePositiveRate : activityDerived.truePositiveRate
  const displayFalsePositiveRate = store.falsePositiveRate > 0 ? store.falsePositiveRate : activityDerived.falsePositiveRate
  const displayAvgResponseMs = store.avgResponseMs > 0 ? store.avgResponseMs : activityDerived.avgResponseMs

  // Treemap data from fraud typologies
  const treemapData = useMemo(() =>
    displayFraudTypologies.map(ft => ({
      name: ft.name,
      size: ft.count,
      percentage: ft.percentage,
      color: ft.color,
    })),
    [displayFraudTypologies],
  )

  // Scatter data for risk vs volume
  const scatterData = useMemo(() =>
    store.geoRegions.map(r => ({
      x: r.transactions,
      y: r.riskScore,
      z: r.fraudRate * 10,
      name: r.region,
    })),
    [store.geoRegions],
  )

  // Radial bar data for model performance
  const radialModelData = useMemo(() => {
    const aucpr = displayModelPerformance.find(m => m.metric === 'AUCPR') ?? displayModelPerformance[0]
    if (!aucpr) return []
    return [
      { name: 'XGBoost', value: aucpr.xgboost, fill: CHART_COLORS.blue },
      { name: 'Best Classical', value: aucpr.ensemble, fill: CHART_COLORS.success },
      ...(aucpr.llmAgent > 0 ? [{ name: 'Qwen Agent', value: aucpr.llmAgent, fill: CHART_COLORS.deepRed }] : []),
    ]
  }, [displayModelPerformance])

  const hasGnnModelMetrics = useMemo(
    () => displayModelPerformance.some((item) => typeof item.gnn === 'number' && item.gnn > 0),
    [displayModelPerformance],
  )
  const hasLlmModelMetrics = useMemo(
    () => displayModelPerformance.some((item) => item.llmAgent > 0),
    [displayModelPerformance],
  )

  const latestFraudPoint = displayFraudRate[displayFraudRate.length - 1]
  const latestFraudThreshold = latestFraudPoint?.threshold
  const latestFraudBaseline = latestFraudPoint?.baseline
  const geographicEvidenceCount =
    store.threatHotspots.length
    + store.crossBorderFlows.length
    + store.countryThreats.length
    + store.geoRegions.length
  const hasTransactionVolume = hasPositiveMetric(
    displayTransactionVolume,
    (row) => [row.legitimate, row.suspicious, row.fraudulent, row.blocked],
  )
  const hasFraudRate = hasPositiveMetric(
    displayFraudRate,
    (row) => [row.rate],
  )
  const hasChannelVolume = hasPositiveMetric(
    store.channelVolume,
    (row) => [row.UPI, row.NEFT, row.RTGS, row.IMPS, row.Card, row.Wallet],
  )
  const hasLatencyMetrics = hasPositiveMetric(
    displayLatencyMetrics,
    (row) => [row.p50, row.p95, row.p99, row.mlInference],
  )
  const hasVelocityDistribution = hasPositiveMetric(
    displayVelocityDistribution,
    (row) => [row.normal, row.suspicious, row.fraudulent],
  )
  const hasAmountDistribution = hasPositiveMetric(
    displayAmountDistribution,
    (row) => [row.count, row.fraudCount, row.fraudRate],
  )
  const hasAccountRiskBands = hasPositiveMetric(displayAccountRiskBands, (row) => [row.count])
  const hasScatterData = hasPositiveMetric(scatterData, (row) => [row.x, row.y, row.z])
  const hasNetworkMetrics = hasPositiveMetric(store.networkMetrics, (row) => [row.value])
  const hasSystemGaugeMetrics = displayModelAccuracy > 0 || displayTruePositiveRate > 0 || displayFalsePositiveRate > 0 || displayAvgResponseMs > 0
  const hasFraudTypologies = displayFraudTypologies.length > 0 && hasPositiveMetric(displayFraudTypologies, (row) => [row.count])
  const hasTemporalHeatmap = displayRiskHeatmap.length > 0 && hasPositiveMetric(displayRiskHeatmap, (row) => [row.value])
  const hasAttackVectors = store.attackVectors.length > 0 && hasPositiveMetric(store.attackVectors, (row) => [row.count])
  const hasThreatEvents = displayThreatEvents.length > 0
  const hasInterRegionCorridors = store.crossBorderFlows.length > 0
  const hasAlertFunnel = hasPositiveMetric(displayAlertFunnel, (row) => [row.count])
  const hasModelMetrics = displayModelPerformance.length > 0
  const hasRadialModelMetrics = radialModelData.length > 0
  const hasDeviceFingerprints = hasPositiveMetric(store.deviceFingerprints, (row) => [row.unique, row.flagged, row.banned])
  const hasBackendEvidence =
    hasTransactionVolume ||
    hasFraudRate ||
    hasChannelVolume ||
    hasLatencyMetrics ||
    hasTemporalHeatmap ||
    hasFraudTypologies ||
    geographicEvidenceCount > 0 ||
    hasThreatEvents ||
    hasAlertFunnel ||
    hasModelMetrics ||
    hasSystemGaugeMetrics ||
    activityDerived.hasEvidence
  const lastEvidenceTimestamp = Math.max(
    0,
    ...displayTransactionVolume.map((row) => row.timestamp),
    ...displayFraudRate.map((row) => row.timestamp),
    ...store.channelVolume.map((row) => row.timestamp),
    ...displayLatencyMetrics.map((row) => row.timestamp),
    ...displayThreatEvents.map((row) => row.timestamp),
  )
  const showAllDecks = analyticsDeck === 'all'
  const showOperationsDeck = showAllDecks || analyticsDeck === 'operations'
  const showGeoDeck = showAllDecks || analyticsDeck === 'geo'
  const showModelDeck = showAllDecks || analyticsDeck === 'model'
  const showForensicsDeck = showAllDecks || analyticsDeck === 'forensics'

  return (
    <div className="ubi-analytics-page min-h-full bg-transparent">
      <div className="space-y-4 overflow-x-hidden p-4 pb-14">

        {/* ====== HEADER ====== */}
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-lg font-bold text-text-primary flex items-center gap-2">
              <BarChart3 size={20} className="text-accent-primary" />
              Real-Time Fraud Analytics
              <span className={cn(
                'ml-2 flex items-center gap-1 rounded-full border px-2 py-0.5 text-[9px] font-mono',
                hasBackendEvidence
                  ? 'border-alert-low/20 bg-alert-low/10 text-alert-low'
                  : 'border-border-subtle bg-bg-elevated text-text-muted',
              )}>
                <Radio size={8} className={hasBackendEvidence ? 'animate-pulse' : ''} />
                {hasBackendEvidence ? 'BACKEND SYNCED' : 'AWAITING DATA'}
              </span>
            </h1>
            <p className="text-[10px] text-text-muted mt-0.5">
              PayFlow Fraud Intelligence Platform &mdash; Union Bank of India &mdash; Live Telemetry Dashboard
            </p>
          </div>
          <div className="text-right">
            <div className="text-[9px] text-[#8a98aa] font-mono">Last evidence update</div>
            <div className="text-xs text-[#4b5d76] font-mono tabular-nums">
              {lastEvidenceTimestamp > 0 ? new Date(lastEvidenceTimestamp).toLocaleTimeString() : 'n/a'}
            </div>
          </div>
        </div>

        <LatestEvaluationPanel report={latestEvaluation} />
        <EventLabReportHistoryFabric reports={eventLabReports} />

        <div className="flex flex-wrap items-center justify-between gap-3 rounded-lg border border-[#00579C]/20 bg-white px-3 py-2 shadow-sm">
          <div>
            <div className="text-[10px] font-black uppercase tracking-[0.14em] text-[#00579C]">Analytics deck</div>
            <div className="mt-0.5 text-[9px] text-text-muted">
              All Live Views keeps every metric, threat map, model plot, and forensic panel mounted for evaluator review; filters narrow the surface when needed.
            </div>
          </div>
          <div className="grid grid-cols-2 gap-1 rounded-md border border-border-subtle bg-[#f4f8fc] p-1 sm:grid-cols-5">
            {([
              ['all', 'All Live Views'],
              ['operations', 'Operations'],
              ['geo', 'Geo Maps'],
              ['model', 'Model Stack'],
              ['forensics', 'Forensics'],
            ] as const).map(([deck, label]) => (
              <button
                key={deck}
                type="button"
                onClick={() => setAnalyticsDeck(deck)}
                className={cn(
                  'h-8 rounded px-3 text-[9px] font-black uppercase tracking-[0.1em] transition-colors',
                  analyticsDeck === deck
                    ? 'bg-[#00579C] text-white shadow-sm'
                    : 'text-text-secondary hover:bg-white hover:text-[#00579C]',
                )}
              >
                {label}
              </button>
            ))}
          </div>
        </div>

        {/* ====== KPI STAT CARDS (Row 1) ====== */}
        <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 xl:grid-cols-5">
          <StatCard icon={Activity} label="Transactions Processed" value={<AnimatedCounter value={store.totalProcessed} />} color={CHART_COLORS.primary} />
          <StatCard icon={ShieldAlert} label="Flagged Suspicious" value={<AnimatedCounter value={store.totalFlagged} />} color={CHART_COLORS.warning} />
          <StatCard icon={Shield} label="Blocked / Frozen" value={<AnimatedCounter value={store.totalBlocked} />} color={CHART_COLORS.danger} pulse />
          <StatCard icon={Timer} label="Avg Response" value={<AnimatedCounter value={displayAvgResponseMs} decimals={1} />} unit="ms" color={CHART_COLORS.secondary} />
          <StatCard icon={Zap} label="Throughput" value={<AnimatedCounter value={store.throughputTps} />} unit="tx/s" color={CHART_COLORS.brandBlue} />
        </div>

        {/* ====== KPI STAT CARDS (Row 2) ====== */}
        <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 xl:grid-cols-5">
          <StatCard icon={Brain} label="Best Model Signal" value={<AnimatedCounter value={displayModelAccuracy} decimals={1} />} unit="%" color={CHART_COLORS.success} />
          <StatCard icon={Target} label="Decision Hit Rate" value={<AnimatedCounter value={displayTruePositiveRate} decimals={1} />} unit="%" color={CHART_COLORS.blue} />
          <StatCard icon={AlertTriangle} label="Review Load" value={<AnimatedCounter value={displayFalsePositiveRate} decimals={1} />} unit="%" color={CHART_COLORS.gold} />
          <StatCard icon={Network} label="Active Mule Networks" value={<AnimatedCounter value={store.activeMules} />} color={CHART_COLORS.darkRed} pulse />
          <StatCard icon={Gauge} label="System Risk Score" value={<AnimatedCounter value={store.riskScore} decimals={1} />} unit="/100" color={store.riskScore > 60 ? CHART_COLORS.danger : store.riskScore > 40 ? CHART_COLORS.warning : CHART_COLORS.success} />
        </div>

        {showOperationsDeck && (
        <>
        {/* ====== ROW 1: Transaction Volume + Fraud Rate ====== */}
        <div className="grid grid-cols-1 gap-3 xl:grid-cols-2">
          <ChartCard title="Transaction Volume Stream" subtitle="Real-time ingestion" icon={Activity} badge="AREA" live={hasTransactionVolume}>
            {!hasTransactionVolume ? (
              <AnalyticsEmptyState
                title="No transaction volume evidence"
                detail="This chart renders only snapshot/SSE ingestion counts from the backend orchestrator."
                height={220}
              />
            ) : (
            <ResponsiveContainer width="100%" height={220}>
              <AreaChart data={displayTransactionVolume}>
                <defs>
                  <linearGradient id="gradLegit" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor={CHART_COLORS.primary} stopOpacity={0.4} />
                    <stop offset="95%" stopColor={CHART_COLORS.primary} stopOpacity={0} />
                  </linearGradient>
                  <linearGradient id="gradSusp" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor={CHART_COLORS.warning} stopOpacity={0.4} />
                    <stop offset="95%" stopColor={CHART_COLORS.warning} stopOpacity={0} />
                  </linearGradient>
                  <linearGradient id="gradFraud" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor={CHART_COLORS.danger} stopOpacity={0.5} />
                    <stop offset="95%" stopColor={CHART_COLORS.danger} stopOpacity={0} />
                  </linearGradient>
                </defs>
                <CartesianGrid {...GRID_STYLE} />
                <XAxis dataKey="time" tick={{ fontSize: 9, fill: '#617189' }} tickLine={false} axisLine={false} />
                <YAxis tick={{ fontSize: 9, fill: '#617189' }} tickLine={false} axisLine={false} width={35} />
                <Tooltip {...TOOLTIP_STYLE} />
                <Area type="monotone" dataKey="legitimate" stroke={CHART_COLORS.primary} fill="url(#gradLegit)" strokeWidth={2} dot={false} animationDuration={300} />
                <Area type="monotone" dataKey="suspicious" stroke={CHART_COLORS.warning} fill="url(#gradSusp)" strokeWidth={2} dot={false} animationDuration={300} />
                <Area type="monotone" dataKey="fraudulent" stroke={CHART_COLORS.danger} fill="url(#gradFraud)" strokeWidth={2} dot={false} animationDuration={300} />
                <Area type="monotone" dataKey="blocked" stroke={CHART_COLORS.darkRed} fillOpacity={0} strokeWidth={1.5} strokeDasharray="4 2" dot={false} animationDuration={300} />
              </AreaChart>
            </ResponsiveContainer>
            )}
          </ChartCard>

          <ChartCard title="Fraud Detection Rate" subtitle="% fraud from backend scoring stream" icon={TrendingUp} badge="LINE" live={hasFraudRate}>
            {!hasFraudRate ? (
              <AnalyticsEmptyState
                title="No fraud-rate series"
                detail="Fraud-rate trend appears after backend risk-distribution or live alert counters produce non-zero evidence."
                height={220}
              />
            ) : (
            <ResponsiveContainer width="100%" height={220}>
              <LineChart data={displayFraudRate}>
                <CartesianGrid {...GRID_STYLE} />
                <XAxis dataKey="time" tick={{ fontSize: 9, fill: '#617189' }} tickLine={false} axisLine={false} />
                <YAxis tick={{ fontSize: 9, fill: '#617189' }} tickLine={false} axisLine={false} width={30} domain={[0, 'auto']} />
                <Tooltip {...TOOLTIP_STYLE} />
                {latestFraudThreshold != null && latestFraudThreshold > 0 && (
                  <ReferenceLine y={latestFraudThreshold} stroke={CHART_COLORS.danger} strokeDasharray="6 3" strokeWidth={1.5} label={{ value: 'Backend threshold', fill: CHART_COLORS.danger, fontSize: 8, position: 'right' }} />
                )}
                {latestFraudBaseline != null && latestFraudBaseline > 0 && (
                  <ReferenceLine y={latestFraudBaseline} stroke={CHART_COLORS.success} strokeDasharray="4 4" strokeWidth={1} label={{ value: 'Baseline', fill: CHART_COLORS.success, fontSize: 8, position: 'right' }} />
                )}
                <Line type="monotone" dataKey="rate" stroke={CHART_COLORS.warning} strokeWidth={2.5} dot={false} animationDuration={300}
                  style={{ filter: 'drop-shadow(0 0 4px rgba(234,179,8,0.4))' }} />
              </LineChart>
            </ResponsiveContainer>
            )}
          </ChartCard>
        </div>

        {/* ====== ROW 2: Channel Breakdown + Pipeline Latency ====== */}
        <div className="grid grid-cols-1 gap-3 xl:grid-cols-2">
          <ChartCard title="Channel-wise Volume" subtitle="Stacked distribution" icon={Layers} badge="STACKED" live={hasChannelVolume}>
            {!hasChannelVolume ? (
              <AnalyticsEmptyState
                title="No channel volume"
                detail="Channel distribution is derived from backend graph edges carrying transaction channel metadata."
                height={220}
              />
            ) : (
            <ResponsiveContainer width="100%" height={220}>
              <AreaChart data={store.channelVolume}>
                <defs>
                  {Object.entries(CHANNEL_COLORS).map(([key, color]) => (
                    <linearGradient key={key} id={`gradCh${key}`} x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor={color} stopOpacity={0.35} />
                      <stop offset="95%" stopColor={color} stopOpacity={0.02} />
                    </linearGradient>
                  ))}
                </defs>
                <CartesianGrid {...GRID_STYLE} />
                <XAxis dataKey="time" tick={{ fontSize: 9, fill: '#617189' }} tickLine={false} axisLine={false} />
                <YAxis tick={{ fontSize: 9, fill: '#617189' }} tickLine={false} axisLine={false} width={35} />
                <Tooltip {...TOOLTIP_STYLE} />
                {Object.entries(CHANNEL_COLORS).map(([key, color]) => (
                  <Area key={key} type="monotone" dataKey={key} stackId="channels" stroke={color} fill={`url(#gradCh${key})`} strokeWidth={1.5} dot={false} animationDuration={300} />
                ))}
              </AreaChart>
            </ResponsiveContainer>
            )}
          </ChartCard>

          <ChartCard title="Pipeline Latency Distribution" subtitle="p50 / p95 / p99 + ML Inference" icon={Timer} badge="COMPOSED" live={hasLatencyMetrics}>
            {!hasLatencyMetrics ? (
              <AnalyticsEmptyState
                title="No latency samples"
                detail="Latency chart uses backend telemetry. It stays empty until the profiler or pipeline reports measurable timings."
                height={220}
              />
            ) : (
            <ResponsiveContainer width="100%" height={220}>
              <ComposedChart data={displayLatencyMetrics}>
                <defs>
                  <linearGradient id="gradP99" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor={CHART_COLORS.danger} stopOpacity={0.2} />
                    <stop offset="95%" stopColor={CHART_COLORS.danger} stopOpacity={0} />
                  </linearGradient>
                </defs>
                <CartesianGrid {...GRID_STYLE} />
                <XAxis dataKey="time" tick={{ fontSize: 9, fill: '#617189' }} tickLine={false} axisLine={false} />
                <YAxis tick={{ fontSize: 9, fill: '#617189' }} tickLine={false} axisLine={false} width={35} unit="ms" />
                <Tooltip {...TOOLTIP_STYLE} />
                <Area type="monotone" dataKey="p99" stroke={CHART_COLORS.danger} fill="url(#gradP99)" strokeWidth={1} dot={false} animationDuration={300} />
                <Line type="monotone" dataKey="p95" stroke={CHART_COLORS.gold} strokeWidth={2} dot={false} animationDuration={300} />
                <Line type="monotone" dataKey="p50" stroke={CHART_COLORS.success} strokeWidth={2} dot={false} animationDuration={300} />
                <Bar dataKey="mlInference" fill={CHART_COLORS.purple} fillOpacity={0.4} barSize={6} radius={[2, 2, 0, 0]} animationDuration={300} />
              </ComposedChart>
            </ResponsiveContainer>
            )}
          </ChartCard>
        </div>

        </>
        )}

        {showForensicsDeck && (
        <>
        {/* ====== ROW 3: Risk Heatmap + Fraud Typology Treemap ====== */}
        <div className="grid grid-cols-1 gap-3 xl:grid-cols-2">
          <ChartCard title="Temporal Risk Heatmap" subtitle="24h × 7d fraud risk intensity" icon={Target} badge="HEATMAP" className="min-h-[260px]" live={hasTemporalHeatmap}>
            <RiskHeatmap data={displayRiskHeatmap} />
          </ChartCard>

          <ChartCard title="Fraud Typology Distribution" subtitle="Relative attack vector sizes" icon={ShieldAlert} badge="TREEMAP" live={hasFraudTypologies}>
            {treemapData.length === 0 ? (
              <AnalyticsEmptyState
                title="No typology counts"
                detail="Typology distribution renders only fraud labels observed in the backend graph."
                height={220}
              />
            ) : (
              <ResponsiveContainer width="100%" height={220}>
                <Treemap
                  data={treemapData}
                  dataKey="size"
                  // eslint-disable-next-line @typescript-eslint/no-explicit-any
                  content={<TreemapContent /> as any}
                  animationDuration={500}
                />
              </ResponsiveContainer>
            )}
          </ChartCard>
        </div>

        </>
        )}

        {showGeoDeck && (
        <>
        {/* ====== THREAT INTELLIGENCE MAPS ====== */}
        <div className="space-y-3">
          {/* Section header */}
          <div className="flex items-center gap-2 pt-2">
            <Globe size={16} className="text-accent-primary" />
            <h2 className="text-sm font-bold text-text-primary">Geographic Threat Intelligence</h2>
            <span className="text-[8px] font-mono px-2 py-0.5 rounded-full bg-alert-critical/10 text-alert-critical border border-alert-critical/20 flex items-center gap-1">
              <Radio size={7} className={geographicEvidenceCount > 0 ? 'animate-pulse' : ''} />
              {geographicEvidenceCount > 0 ? 'LIVE THREAT FEED' : 'AWAITING GEO EVIDENCE'}
            </span>
          </div>

          {/* World Map — full width */}
          <ChartCard title="National Threat Landscape" subtitle="Geo-tagged fraud flow and hotspot intensity" icon={Globe} badge="GEO MAP" className="overflow-hidden" live={geographicEvidenceCount > 0}>
            <WorldThreatMap
              hotspots={store.threatHotspots}
              flows={store.crossBorderFlows}
              countryThreats={store.countryThreats}
              evidenceTimestamp={lastEvidenceTimestamp}
            />
          </ChartCard>

          {/* India Map + National Intelligence + Attack Vectors */}
          <div className="grid grid-cols-1 gap-3 xl:grid-cols-3">
            <ChartCard title="India Regional Threat Map" subtitle="State-wise risk heatmap on real geography" icon={MapPin} badge="INDIA MAP" className="overflow-hidden" live={store.geoRegions.length > 0}>
              <IndiaRegionalMap regions={store.geoRegions} />
            </ChartCard>

            <ChartCard title="National Threat Posture" subtitle="India-wide aggregate from graph telemetry" icon={ShieldAlert} badge="THREAT INDEX" live={store.countryThreats.length > 0}>
              <CountryThreatPanel countries={store.countryThreats} />
            </ChartCard>

            <ChartCard title="Attack Vector Breakdown" subtitle="Distribution by fraud technique" icon={Target} badge="VECTORS" live={hasAttackVectors}>
              <AttackVectorBreakdown vectors={store.attackVectors} />
            </ChartCard>
          </div>

          {/* Live Feed + Inter-Region Corridors */}
          <div className="grid grid-cols-1 gap-3 xl:grid-cols-2">
            <ChartCard title="Live Threat Feed" subtitle="Backend threat-summary indicators" icon={Radio} badge="FEED" className="overflow-hidden" live={hasThreatEvents}>
              <LiveThreatFeed events={displayThreatEvents} />
            </ChartCard>

            <ChartCard title="Inter-Region Corridors" subtitle="Domestic suspicious flow analysis" icon={Network} badge="CORRIDORS" live={hasInterRegionCorridors}>
              <CrossBorderCorridors flows={store.crossBorderFlows} />
            </ChartCard>
          </div>
        </div>

        </>
        )}

        {showModelDeck && (
        <>
        {/* ====== ROW 4: Alert Pipeline Funnel + Model Comparison Radar ====== */}
        <div className="grid grid-cols-1 gap-3 xl:grid-cols-3">
          <ChartCard title="Alert Processing Pipeline" subtitle="Event funnel analysis" icon={Layers} badge="FUNNEL" live={hasAlertFunnel}>
            <ResponsiveContainer width="100%" height={260}>
              <FunnelChart>
                <Tooltip {...TOOLTIP_STYLE} />
                <Funnel
                  dataKey="count"
                  data={displayAlertFunnel.map((af, i) => ({
                    ...af,
                    fill: [CHART_COLORS.primary, CHART_COLORS.secondary, CHART_COLORS.purple, CHART_COLORS.warning, CHART_COLORS.danger, CHART_COLORS.success][i],
                  }))}
                  isAnimationActive
                  animationDuration={500}
                >
                  <LabelList position="right" fill="#94a3b8" fontSize={9} dataKey="stage" />
                  <LabelList position="center" fill="white" fontSize={10} fontWeight={600}
                    // eslint-disable-next-line @typescript-eslint/no-explicit-any
                    formatter={(v: any) => v?.toLocaleString()} />
                </Funnel>
              </FunnelChart>
            </ResponsiveContainer>
          </ChartCard>

          <ChartCard title="Model Evidence Radar" subtitle="Classical validation metrics plus parsed LLM verdict confidence" icon={Brain} badge="RADAR" live={hasModelMetrics}>
            {displayModelPerformance.length === 0 ? (
              <AnalyticsEmptyState
                title="No model metrics"
                detail="Model comparison appears after backend training metrics are present in the system snapshot."
                height={260}
              />
            ) : (
              <ResponsiveContainer width="100%" height={260}>
                <RadarChart data={displayModelPerformance} cx="50%" cy="50%" outerRadius="70%">
                  <PolarGrid stroke="rgba(148,163,184,0.1)" />
                  <PolarAngleAxis dataKey="metric" tick={{ fontSize: 8, fill: '#94a3b8' }} />
                  <PolarRadiusAxis tick={{ fontSize: 7, fill: '#617189' }} domain={[0, 100]} />
                  <Radar name="Ensemble" dataKey="ensemble" stroke={CHART_COLORS.success} fill={CHART_COLORS.success} fillOpacity={0.15} strokeWidth={2} animationDuration={500} />
                  {hasLlmModelMetrics && (
                    <Radar name="LLM Agent" dataKey="llmAgent" stroke={CHART_COLORS.deepRed} fill={CHART_COLORS.deepRed} fillOpacity={0.1} strokeWidth={1.5} animationDuration={500} />
                  )}
                  <Radar name="XGBoost" dataKey="xgboost" stroke={CHART_COLORS.blue} fill={CHART_COLORS.blue} fillOpacity={0.08} strokeWidth={1.5} animationDuration={500} />
                  {hasGnnModelMetrics && (
                    <Radar name="GNN" dataKey="gnn" stroke={CHART_COLORS.purple} fill={CHART_COLORS.purple} fillOpacity={0.08} strokeWidth={1.5} animationDuration={500} />
                  )}
                  <Legend iconType="line" wrapperStyle={{ fontSize: '9px', color: '#94a3b8' }} />
                </RadarChart>
              </ResponsiveContainer>
            )}
          </ChartCard>

          <ChartCard title="Model AUCPR Radial" subtitle="Backend training summary" icon={Gauge} badge="RADIAL" live={hasRadialModelMetrics}>
            {radialModelData.length === 0 ? (
              <AnalyticsEmptyState
                title="No AUCPR summary"
                detail="This radial chart uses backend model-training metrics and remains empty until those values are available."
                height={260}
              />
            ) : (
              <ResponsiveContainer width="100%" height={260}>
                <RadialBarChart innerRadius="25%" outerRadius="90%" data={radialModelData} startAngle={180} endAngle={0}>
                  <RadialBar
                    background={{ fill: 'rgba(148,163,184,0.05)' }}
                    dataKey="value"
                    cornerRadius={6}
                    animationDuration={500}
                    label={{ fill: '#e2e8f0', fontSize: 10, position: 'insideStart', fontWeight: 600 }}
                  />
                  <Legend iconType="circle" wrapperStyle={{ fontSize: '9px', color: '#94a3b8' }} />
                  <Tooltip {...TOOLTIP_STYLE} />
                </RadialBarChart>
              </ResponsiveContainer>
            )}
          </ChartCard>
        </div>

        </>
        )}

        {showOperationsDeck && (
        <>
        {/* ====== ROW 5: Velocity Distribution + Amount Distribution + Account Risk Bands ====== */}
        <div className="grid grid-cols-1 gap-3 xl:grid-cols-3">
          <ChartCard title="Transaction Velocity Profile" subtitle="Frequency-based risk bands" icon={Zap} badge="GROUPED BAR" live={hasVelocityDistribution}>
            {!hasVelocityDistribution ? (
              <AnalyticsEmptyState
                title="No velocity distribution"
                detail="Velocity profile uses backend account velocity windows and will populate once active accounts appear in the selected window."
                height={240}
              />
            ) : (
            <ResponsiveContainer width="100%" height={240}>
              <BarChart data={displayVelocityDistribution} barGap={2}>
                <CartesianGrid {...GRID_STYLE} />
                <XAxis dataKey="range" tick={{ fontSize: 8, fill: '#617189' }} tickLine={false} axisLine={false} />
                <YAxis tick={{ fontSize: 9, fill: '#617189' }} tickLine={false} axisLine={false} width={35} />
                <Tooltip {...TOOLTIP_STYLE} />
                <Bar dataKey="normal" fill={CHART_COLORS.success} fillOpacity={0.7} radius={[3, 3, 0, 0]} barSize={12} animationDuration={400} />
                <Bar dataKey="suspicious" fill={CHART_COLORS.warning} fillOpacity={0.7} radius={[3, 3, 0, 0]} barSize={12} animationDuration={400} />
                <Bar dataKey="fraudulent" fill={CHART_COLORS.danger} fillOpacity={0.8} radius={[3, 3, 0, 0]} barSize={12} animationDuration={400} />
                <Legend iconType="circle" wrapperStyle={{ fontSize: '9px', color: '#94a3b8' }} />
              </BarChart>
            </ResponsiveContainer>
            )}
          </ChartCard>

          <ChartCard title="Amount Distribution vs Fraud" subtitle="Backend graph amount buckets and fraud-label rate" icon={TrendingDown} badge="COMPOSED" live={hasAmountDistribution}>
            {!hasAmountDistribution ? (
              <AnalyticsEmptyState
                title="No amount distribution"
                detail="Amount distribution is computed from backend graph transaction amounts and fraud labels."
                height={240}
              />
            ) : (
            <ResponsiveContainer width="100%" height={240}>
              <ComposedChart data={displayAmountDistribution}>
                <defs>
                  <linearGradient id="gradAmt" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor={CHART_COLORS.secondary} stopOpacity={0.3} />
                    <stop offset="95%" stopColor={CHART_COLORS.secondary} stopOpacity={0} />
                  </linearGradient>
                </defs>
                <CartesianGrid {...GRID_STYLE} />
                <XAxis dataKey="range" tick={{ fontSize: 7, fill: '#617189' }} tickLine={false} axisLine={false} />
                <YAxis yAxisId="left" tick={{ fontSize: 9, fill: '#617189' }} tickLine={false} axisLine={false} width={35} />
                <YAxis yAxisId="right" orientation="right" tick={{ fontSize: 9, fill: '#617189' }} tickLine={false} axisLine={false} width={30} unit="%" />
                <Tooltip {...TOOLTIP_STYLE} />
                <Bar yAxisId="left" dataKey="count" fill="url(#gradAmt)" barSize={16} radius={[4, 4, 0, 0]} animationDuration={400} />
                <Bar yAxisId="left" dataKey="fraudCount" fill={CHART_COLORS.danger} fillOpacity={0.7} barSize={8} radius={[3, 3, 0, 0]} animationDuration={400} />
                <Line yAxisId="right" type="monotone" dataKey="fraudRate" name="Fraud rate" stroke={CHART_COLORS.warning} strokeWidth={2.5} dot={{ fill: CHART_COLORS.warning, r: 3 }} animationDuration={400} />
              </ComposedChart>
            </ResponsiveContainer>
            )}
          </ChartCard>

          <ChartCard title="Account Risk Band Census" subtitle="Risk-stratified population" icon={Shield} badge="PIE" live={hasAccountRiskBands}>
            {!hasAccountRiskBands ? (
              <AnalyticsEmptyState
                title="No risk-band census"
                detail="Risk-band census uses backend risk-distribution buckets. It stays empty until scoring samples are available."
                height={240}
              />
            ) : (
            <ResponsiveContainer width="100%" height={240}>
              <PieChart>
                <Pie
                  data={displayAccountRiskBands}
                  cx="50%" cy="50%"
                  innerRadius={50} outerRadius={85}
                  dataKey="count" nameKey="band"
                  paddingAngle={3}
                  strokeWidth={0}
                  animationDuration={500}
                >
                  {displayAccountRiskBands.map((entry, index) => (
                    <Cell key={index} fill={entry.color} fillOpacity={0.85} />
                  ))}
                </Pie>
                <Tooltip {...TOOLTIP_STYLE} />
                <Legend iconType="circle" wrapperStyle={{ fontSize: '8px', color: '#94a3b8' }} />
              </PieChart>
            </ResponsiveContainer>
            )}
          </ChartCard>
        </div>

        </>
        )}

        {showModelDeck && (
        <>
        {/* ====== ROW 6: Geo Risk Scatter + Network Radar + Device Fingerprints ====== */}
        <div className="grid grid-cols-1 gap-3 xl:grid-cols-3">
          <ChartCard title="Regional Risk vs Volume" subtitle="Scatter: bubble = fraud rate" icon={Target} badge="SCATTER" live={hasScatterData}>
            {!hasScatterData ? (
              <AnalyticsEmptyState
                title="No regional scatter evidence"
                detail="Scatter points use geo-tagged backend graph edges with transaction counts and regional risk scores."
                height={240}
              />
            ) : (
            <ResponsiveContainer width="100%" height={240}>
              <ScatterChart>
                <CartesianGrid {...GRID_STYLE} />
                <XAxis type="number" dataKey="x" name="Transactions" tick={{ fontSize: 8, fill: '#617189' }} tickLine={false} axisLine={false} />
                <YAxis type="number" dataKey="y" name="Risk Score" tick={{ fontSize: 8, fill: '#617189' }} tickLine={false} axisLine={false} width={35} />
                <Tooltip {...TOOLTIP_STYLE} />
                <Scatter data={scatterData} fill={CHART_COLORS.primary} animationDuration={400}>
                  {scatterData.map((_, i) => (
                    <Cell key={i} fill={[CHART_COLORS.primary, CHART_COLORS.secondary, CHART_COLORS.purple, CHART_COLORS.warning, CHART_COLORS.danger, CHART_COLORS.deepRed, CHART_COLORS.brandBlue, CHART_COLORS.blue, CHART_COLORS.success, CHART_COLORS.gold, CHART_COLORS.emerald, CHART_COLORS.darkRed][i % 12]} />
                  ))}
                </Scatter>
              </ScatterChart>
            </ResponsiveContainer>
            )}
          </ChartCard>

          <ChartCard title="Network Topology Metrics" subtitle="Graph health indicators" icon={Network} badge="RADAR" live={hasNetworkMetrics}>
            {!hasNetworkMetrics ? (
              <AnalyticsEmptyState
                title="No topology metrics"
                detail="Network metrics are derived from hydrated backend topology nodes and edges."
                height={240}
              />
            ) : (
            <ResponsiveContainer width="100%" height={240}>
              <RadarChart data={store.networkMetrics} cx="50%" cy="50%" outerRadius="68%">
                <PolarGrid stroke="rgba(148,163,184,0.08)" />
                <PolarAngleAxis dataKey="metric" tick={{ fontSize: 7, fill: '#94a3b8' }} />
                <PolarRadiusAxis tick={false} domain={[0, 'auto']} />
                <Radar name="Current" dataKey="value" stroke={CHART_COLORS.secondary} fill={CHART_COLORS.secondary} fillOpacity={0.2} strokeWidth={2} animationDuration={500} />
                <Tooltip {...TOOLTIP_STYLE} />
              </RadarChart>
            </ResponsiveContainer>
            )}
          </ChartCard>

          <ChartCard title="Device Fingerprint Analysis" subtitle="Unique / fraud-linked / restricted" icon={Fingerprint} badge="HORIZONTAL BAR" live={hasDeviceFingerprints}>
            {store.deviceFingerprints.length === 0 ? (
              <AnalyticsEmptyState
                title="No device fingerprints"
                detail="Device analysis renders only fingerprints observed on backend graph edges."
                height={240}
              />
            ) : (
              <ResponsiveContainer width="100%" height={240}>
                <BarChart data={store.deviceFingerprints} layout="vertical" barGap={1}>
                  <CartesianGrid {...GRID_STYLE} />
                  <XAxis type="number" tick={{ fontSize: 8, fill: '#617189' }} tickLine={false} axisLine={false} />
                  <YAxis type="category" dataKey="type" tick={{ fontSize: 7, fill: '#94a3b8' }} tickLine={false} axisLine={false} width={80} />
                  <Tooltip {...TOOLTIP_STYLE} />
                  <Bar dataKey="unique" fill={CHART_COLORS.primary} fillOpacity={0.6} barSize={8} radius={[0, 3, 3, 0]} animationDuration={400} />
                  <Bar dataKey="flagged" fill={CHART_COLORS.warning} fillOpacity={0.7} barSize={8} radius={[0, 3, 3, 0]} animationDuration={400} />
                  <Bar dataKey="banned" name="restricted" fill={CHART_COLORS.danger} fillOpacity={0.8} barSize={8} radius={[0, 3, 3, 0]} animationDuration={400} />
                  <Legend iconType="circle" wrapperStyle={{ fontSize: '8px', color: '#94a3b8' }} />
                </BarChart>
              </ResponsiveContainer>
            )}
          </ChartCard>
        </div>

        </>
        )}

        {showGeoDeck && (
        <>
        {/* ====== ROW 7: Geo Heat Table + Gauges ====== */}
        <div className="grid grid-cols-1 gap-3 xl:grid-cols-3">
          {/* Geo Table */}
          <ChartCard title="Regional Fraud Intelligence" subtitle="State-wise breakdown" icon={Target} badge="TABLE" className="xl:col-span-2" live={store.geoRegions.length > 0}>
            {store.geoRegions.length === 0 ? (
              <AnalyticsEmptyState
                title="No regional graph evidence"
                detail="Regional intelligence uses sender geo fields carried through backend graph edges. It will populate after geo-tagged transactions are ingested."
                height={220}
              />
            ) : (
            <div className="overflow-x-auto">
              <table className="w-full text-[10px]">
                <thead>
                  <tr className="border-b border-[#d7e3f1]">
                    <th className="text-left py-2 px-3 text-[#617189] font-medium">Region</th>
                    <th className="text-right py-2 px-3 text-[#617189] font-medium">Transactions</th>
                    <th className="text-right py-2 px-3 text-[#617189] font-medium">Fraud Rate</th>
                    <th className="text-right py-2 px-3 text-[#617189] font-medium">Volume (₹)</th>
                    <th className="text-right py-2 px-3 text-[#617189] font-medium">Risk Score</th>
                    <th className="text-center py-2 px-3 text-[#617189] font-medium">Risk Bar</th>
                  </tr>
                </thead>
                <tbody>
                  {[...store.geoRegions].sort((a, b) => b.riskScore - a.riskScore).map(r => (
                    <tr key={r.region} className="border-b border-[#d7e3f1]/70 hover:bg-[#f4f8fc] transition-colors">
                      <td className="py-1.5 px-3 text-[#24364f] font-medium">{r.region}</td>
                      <td className="py-1.5 px-3 text-right text-[#4b5d76] font-mono tabular-nums">{r.transactions.toLocaleString()}</td>
                      <td className="py-1.5 px-3 text-right font-mono tabular-nums">
                        <span className={cn(
                          'px-1.5 py-0.5 rounded-full text-[9px] font-semibold',
                          r.fraudRate > 3 ? 'bg-[#DA251C]/15 text-[#DA251C]' :
                          r.fraudRate > 2 ? 'bg-[#DA251C]/10 text-[#B51A13]' :
                          r.fraudRate > 1 ? 'bg-[#f5b400]/15 text-[#7a5a00]' :
                          'bg-[#00579C]/10 text-[#00579C]',
                        )}>
                          {r.fraudRate}%
                        </span>
                      </td>
                      <td className="py-1.5 px-3 text-right text-[#4b5d76] font-mono tabular-nums">₹{(r.amount / 100000).toFixed(1)}L</td>
                      <td className="py-1.5 px-3 text-right font-mono tabular-nums text-[#24364f]">{r.riskScore}</td>
                      <td className="py-1.5 px-3">
                        <div className="h-1.5 w-full bg-[#d7e3f1] rounded-full overflow-hidden">
                          <div
                            className="h-full rounded-full transition-all duration-500"
                            style={{
                              width: `${r.riskScore}%`,
                              backgroundColor: r.riskScore > 60 ? CHART_COLORS.danger : r.riskScore > 40 ? CHART_COLORS.warning : CHART_COLORS.success,
                            }}
                          />
                        </div>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            )}
          </ChartCard>

          {/* Gauges */}
          <ChartCard title="System Health Gauges" subtitle="Critical metrics" icon={Gauge} badge="GAUGE" live={hasSystemGaugeMetrics}>
            {!hasSystemGaugeMetrics ? (
              <AnalyticsEmptyState
                title="No system gauge telemetry"
                detail="Gauges render backend model metrics and profiler telemetry only after those metrics are present."
                height={220}
              />
            ) : (
            <div className="grid grid-cols-2 gap-2 pt-2">
              <GaugeChart value={displayModelAccuracy} max={100} label="Model signal %" color={CHART_COLORS.success} />
              <GaugeChart value={displayTruePositiveRate} max={100} label="Decision hit %" color={CHART_COLORS.blue} />
              <GaugeChart value={displayFalsePositiveRate} max={100} label="Review load %" color={CHART_COLORS.danger} />
              <GaugeChart value={displayAvgResponseMs} max={100} label="Latency ms" color={CHART_COLORS.secondary} />
            </div>
            )}
          </ChartCard>
        </div>

        </>
        )}

        {showForensicsDeck && (
        <>
        {/* ====== ROW 8: Fraud Typology Bar + Live Trend Comparison ====== */}
        <div className="grid grid-cols-1 gap-3 xl:grid-cols-2">
          <ChartCard title="Fraud Pattern Trend Analysis" subtitle="Attack vector comparison" icon={TrendingUp} badge="HORIZONTAL BAR" live={hasFraudTypologies}>
            {!hasFraudTypologies ? (
              <AnalyticsEmptyState
                title="No fraud pattern trend"
                detail="Pattern trends are computed from backend fraud typology counts and stay empty until fraud labels are present."
                height={240}
              />
            ) : (
            <ResponsiveContainer width="100%" height={240}>
              <BarChart data={displayFraudTypologies} layout="vertical" barGap={2}>
                <CartesianGrid {...GRID_STYLE} />
                <XAxis type="number" tick={{ fontSize: 8, fill: '#617189' }} tickLine={false} axisLine={false} />
                <YAxis type="category" dataKey="name" tick={{ fontSize: 8, fill: '#94a3b8' }} tickLine={false} axisLine={false} width={100} />
                <Tooltip {...TOOLTIP_STYLE} />
                <Bar dataKey="count" radius={[0, 4, 4, 0]} barSize={14} animationDuration={400}>
                  {displayFraudTypologies.map((ft, i) => (
                    <Cell key={i} fill={ft.color} fillOpacity={0.75} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
            )}
          </ChartCard>

          <ChartCard title="Fraud Typology Donut" subtitle="Proportional breakdown" icon={ShieldAlert} badge="DONUT" live={hasFraudTypologies}>
            {!hasFraudTypologies ? (
              <AnalyticsEmptyState
                title="No typology proportions"
                detail="Donut proportions require non-zero backend fraud typology counts."
                height={240}
              />
            ) : (
            <ResponsiveContainer width="100%" height={240}>
              <PieChart>
                <Pie
                  data={displayFraudTypologies}
                  cx="50%" cy="50%"
                  innerRadius={55} outerRadius={90}
                  dataKey="count" nameKey="name"
                  paddingAngle={2}
                  strokeWidth={0}
                  animationDuration={500}
                >
                  {displayFraudTypologies.map((ft, i) => (
                    <Cell key={i} fill={ft.color} fillOpacity={0.8} />
                  ))}
                </Pie>
                <Tooltip {...TOOLTIP_STYLE} />
                <Legend iconType="circle" wrapperStyle={{ fontSize: '8px', color: '#94a3b8' }} />
              </PieChart>
            </ResponsiveContainer>
            )}
          </ChartCard>
        </div>

        </>
        )}

        {/* Footer */}
        <div className="text-center py-3 border-t border-border-subtle">
          <p className="text-[9px] text-text-muted font-mono">
            PayFlow v2.0 &mdash; Real-Time Fraud Intelligence Platform &mdash; Union Bank of India
          </p>
        </div>
      </div>
    </div>
  )
}

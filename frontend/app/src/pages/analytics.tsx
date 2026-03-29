// ============================================================================
// Analytics Page -- Cutting-edge real-time fraud detection data visualization
// Features: 15+ chart types, live streaming data, dark SOC theme
// ============================================================================

import { useEffect, useRef, useMemo, useCallback, useState } from 'react'
import {
  AreaChart, Area, BarChart, Bar, LineChart, Line, ComposedChart,
  PieChart, Pie, Cell, RadarChart, Radar, PolarGrid, PolarAngleAxis,
  PolarRadiusAxis, Treemap, FunnelChart, Funnel, LabelList,
  ScatterChart, Scatter, RadialBarChart, RadialBar, Legend,
  XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ReferenceLine,
} from 'recharts'
import { useAnalyticsStore } from '@/stores/use-analytics-store'
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
  primary: '#6366f1',
  secondary: '#06b6d4',
  success: '#22c55e',
  warning: '#eab308',
  danger: '#ef4444',
  purple: '#a855f7',
  pink: '#ec4899',
  orange: '#f97316',
  teal: '#14b8a6',
  blue: '#3b82f6',
  emerald: '#10b981',
  rose: '#f43f5e',
}

const CHANNEL_COLORS: Record<string, string> = {
  UPI: '#6366f1',
  NEFT: '#06b6d4',
  RTGS: '#22c55e',
  IMPS: '#eab308',
  Card: '#a855f7',
  Wallet: '#ec4899',
}

const TOOLTIP_STYLE = {
  contentStyle: {
    backgroundColor: 'rgba(15, 23, 42, 0.95)',
    border: '1px solid rgba(99, 102, 241, 0.3)',
    borderRadius: '8px',
    backdropFilter: 'blur(12px)',
    fontSize: '11px',
    color: '#e2e8f0',
    boxShadow: '0 8px 32px rgba(0,0,0,0.4)',
  },
  cursor: { stroke: 'rgba(99, 102, 241, 0.3)', strokeWidth: 1 },
}

const GRID_STYLE = { stroke: 'rgba(148, 163, 184, 0.06)', strokeDasharray: '3 3' }

// ============================================================================
// Stat Card with animated trend
// ============================================================================

function StatCard({ icon: Icon, label, value, unit, trend, trendLabel, color, pulse }: {
  icon: typeof Activity
  label: string
  value: string | number
  unit?: string
  trend?: number
  trendLabel?: string
  color: string
  pulse?: boolean
}) {
  const isUp = (trend ?? 0) >= 0
  return (
    <div className={cn(
      'relative overflow-hidden rounded-xl border border-slate-800/80 bg-gradient-to-br from-slate-900/90 to-slate-950/90 p-4',
      'backdrop-blur-sm transition-all duration-300 hover:border-slate-700/60 hover:shadow-lg hover:shadow-black/20',
      'group',
    )}>
      {/* Glow accent */}
      <div className="absolute top-0 right-0 w-20 h-20 rounded-full opacity-10 blur-2xl -translate-y-1/2 translate-x-1/2"
           style={{ backgroundColor: color }} />

      <div className="flex items-start justify-between mb-2">
        <div className={cn(
          'p-2 rounded-lg bg-gradient-to-br',
          pulse && 'animate-pulse',
        )} style={{ backgroundColor: `${color}15`, borderColor: `${color}30` }}>
          <Icon size={16} style={{ color }} />
        </div>
        {trend !== undefined && (
          <div className={cn(
            'flex items-center gap-0.5 text-[10px] font-mono font-semibold px-1.5 py-0.5 rounded-full',
            isUp ? 'bg-emerald-500/10 text-emerald-400' : 'bg-rose-500/10 text-rose-400',
          )}>
            {isUp ? <ArrowUpRight size={10} /> : <ArrowDownRight size={10} />}
            {Math.abs(trend).toFixed(1)}%
          </div>
        )}
      </div>

      <p className="text-[10px] text-slate-500 uppercase tracking-wider font-medium mb-1">{label}</p>
      <div className="flex items-baseline gap-1">
        <span className="text-xl font-bold text-white font-mono tabular-nums">{value}</span>
        {unit && <span className="text-[10px] text-slate-500">{unit}</span>}
      </div>
      {trendLabel && <p className="text-[9px] text-slate-600 mt-1">{trendLabel}</p>}
    </div>
  )
}

// ============================================================================
// Chart Card wrapper
// ============================================================================

function ChartCard({ title, subtitle, icon: Icon, children, className, badge }: {
  title: string
  subtitle?: string
  icon: typeof Activity
  children: React.ReactNode
  className?: string
  badge?: string
}) {
  return (
    <div className={cn(
      'rounded-xl border border-slate-800/80 bg-gradient-to-br from-slate-900/80 to-slate-950/90',
      'backdrop-blur-sm overflow-hidden',
      className,
    )}>
      <div className="flex items-center justify-between px-4 pt-3 pb-1">
        <div className="flex items-center gap-2">
          <Icon size={14} className="text-slate-500" />
          <h3 className="text-xs font-semibold text-slate-300">{title}</h3>
          {badge && (
            <span className="text-[8px] font-mono px-1.5 py-0.5 rounded-full bg-indigo-500/15 text-indigo-400 border border-indigo-500/20">
              {badge}
            </span>
          )}
        </div>
        {subtitle && <span className="text-[9px] text-slate-600">{subtitle}</span>}
        <div className="flex items-center gap-1">
          <Radio size={8} className="text-emerald-500 animate-pulse" />
          <span className="text-[8px] text-emerald-500/80 font-mono">LIVE</span>
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

function RiskHeatmap({ data }: { data: { hour: number; day: string; value: number }[] }) {
  const maxVal = useMemo(() => Math.max(...data.map(d => d.value), 1), [data])

  const getColor = useCallback((value: number) => {
    const t = value / maxVal
    if (t < 0.25) return `rgba(34, 197, 94, ${0.15 + t * 2})`
    if (t < 0.5) return `rgba(234, 179, 8, ${0.2 + (t - 0.25) * 2})`
    if (t < 0.75) return `rgba(249, 115, 22, ${0.3 + (t - 0.5) * 2})`
    return `rgba(239, 68, 68, ${0.4 + (t - 0.75) * 2.4})`
  }, [maxVal])

  return (
    <div className="px-2">
      {/* Hour labels */}
      <div className="flex ml-10 mb-1">
        {Array.from({ length: 24 }, (_, i) => (
          <div key={i} className="flex-1 text-center text-[7px] text-slate-600 font-mono">
            {i % 3 === 0 ? `${i}h` : ''}
          </div>
        ))}
      </div>
      {/* Grid */}
      {DAYS_SHORT.map(day => (
        <div key={day} className="flex items-center gap-1 mb-0.5">
          <span className="text-[8px] text-slate-500 w-8 text-right font-mono">{day}</span>
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
        <span className="text-[7px] text-slate-600">Low</span>
        <div className="flex gap-px">
          {['rgba(34,197,94,0.3)', 'rgba(234,179,8,0.4)', 'rgba(249,115,22,0.5)', 'rgba(239,68,68,0.7)'].map((c, i) => (
            <div key={i} className="w-3 h-2 rounded-[1px]" style={{ backgroundColor: c }} />
          ))}
        </div>
        <span className="text-[7px] text-slate-600">High</span>
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
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null)

  // Start live data streaming
  useEffect(() => {
    // Initial burst
    for (let i = 0; i < 15; i++) store.tick()

    intervalRef.current = setInterval(() => store.tick(), 1500)
    return () => { if (intervalRef.current) clearInterval(intervalRef.current) }
  }, []) // eslint-disable-line react-hooks/exhaustive-deps

  // Treemap data from fraud typologies
  const treemapData = useMemo(() =>
    store.fraudTypologies.map(ft => ({
      name: ft.name,
      size: ft.count,
      percentage: ft.percentage,
      color: ft.color,
    })),
    [store.fraudTypologies],
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
    const f1 = store.modelPerformance.find(m => m.metric === 'F1 Score')
    if (!f1) return []
    return [
      { name: 'XGBoost', value: f1.xgboost, fill: CHART_COLORS.blue },
      { name: 'GNN', value: f1.gnn, fill: CHART_COLORS.purple },
      { name: 'Ensemble', value: f1.ensemble, fill: CHART_COLORS.success },
      { name: 'LLM Agent', value: f1.llmAgent, fill: CHART_COLORS.pink },
    ]
  }, [store.modelPerformance])

  return (
    <div className="h-full overflow-y-auto custom-scrollbar bg-gradient-to-b from-slate-950 via-slate-950 to-slate-900">
      <div className="p-4 space-y-4">

        {/* ====== HEADER ====== */}
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-lg font-bold text-white flex items-center gap-2">
              <BarChart3 size={20} className="text-indigo-400" />
              Real-Time Fraud Analytics
              <span className="ml-2 flex items-center gap-1 text-[9px] font-mono px-2 py-0.5 rounded-full bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
                <Radio size={8} className="animate-pulse" /> STREAMING
              </span>
            </h1>
            <p className="text-[10px] text-slate-500 mt-0.5">
              PayFlow Fraud Intelligence Platform &mdash; Union Bank of India &mdash; Live Telemetry Dashboard
            </p>
          </div>
          <div className="text-right">
            <div className="text-[9px] text-slate-600 font-mono">Last update</div>
            <div className="text-xs text-slate-400 font-mono tabular-nums">{new Date().toLocaleTimeString()}</div>
          </div>
        </div>

        {/* ====== KPI STAT CARDS (Row 1) ====== */}
        <div className="grid grid-cols-5 gap-3">
          <StatCard icon={Activity} label="Transactions Processed" value={<AnimatedCounter value={store.totalProcessed} />} color={CHART_COLORS.primary} trend={2.4} trendLabel="vs last hour" />
          <StatCard icon={ShieldAlert} label="Flagged Suspicious" value={<AnimatedCounter value={store.totalFlagged} />} color={CHART_COLORS.warning} trend={-1.8} trendLabel="Alert rate declining" />
          <StatCard icon={Shield} label="Blocked / Frozen" value={<AnimatedCounter value={store.totalBlocked} />} color={CHART_COLORS.danger} pulse trend={5.2} trendLabel="Auto-enforced" />
          <StatCard icon={Timer} label="Avg Response" value={<AnimatedCounter value={store.avgResponseMs} decimals={1} />} unit="ms" color={CHART_COLORS.secondary} trend={-3.1} trendLabel="Sub-50ms target" />
          <StatCard icon={Zap} label="Throughput" value={<AnimatedCounter value={store.throughputTps} />} unit="tx/s" color={CHART_COLORS.teal} trend={1.6} trendLabel="Elastic scaling" />
        </div>

        {/* ====== KPI STAT CARDS (Row 2) ====== */}
        <div className="grid grid-cols-5 gap-3">
          <StatCard icon={Brain} label="Model Accuracy" value={<AnimatedCounter value={store.modelAccuracy} decimals={1} />} unit="%" color={CHART_COLORS.success} trend={0.3} trendLabel="Ensemble weighted" />
          <StatCard icon={Target} label="True Positive Rate" value={<AnimatedCounter value={store.truePositiveRate} decimals={1} />} unit="%" color={CHART_COLORS.blue} trend={1.2} trendLabel="Recall improving" />
          <StatCard icon={AlertTriangle} label="False Positive Rate" value={<AnimatedCounter value={store.falsePositiveRate} decimals={2} />} unit="%" color={CHART_COLORS.orange} trend={-2.5} trendLabel="Reducing noise" />
          <StatCard icon={Network} label="Active Mule Networks" value={<AnimatedCounter value={store.activeMules} />} color={CHART_COLORS.rose} pulse trend={8.3} trendLabel="Graph detection" />
          <StatCard icon={Gauge} label="System Risk Score" value={<AnimatedCounter value={store.riskScore} decimals={1} />} unit="/100" color={store.riskScore > 60 ? CHART_COLORS.danger : store.riskScore > 40 ? CHART_COLORS.warning : CHART_COLORS.success} trend={-0.7} trendLabel="Composite index" />
        </div>

        {/* ====== ROW 1: Transaction Volume + Fraud Rate ====== */}
        <div className="grid grid-cols-2 gap-3">
          <ChartCard title="Transaction Volume Stream" subtitle="Real-time ingestion" icon={Activity} badge="AREA">
            <ResponsiveContainer width="100%" height={220}>
              <AreaChart data={store.transactionVolume}>
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
                <XAxis dataKey="time" tick={{ fontSize: 9, fill: '#64748b' }} tickLine={false} axisLine={false} />
                <YAxis tick={{ fontSize: 9, fill: '#64748b' }} tickLine={false} axisLine={false} width={35} />
                <Tooltip {...TOOLTIP_STYLE} />
                <Area type="monotone" dataKey="legitimate" stroke={CHART_COLORS.primary} fill="url(#gradLegit)" strokeWidth={2} dot={false} animationDuration={300} />
                <Area type="monotone" dataKey="suspicious" stroke={CHART_COLORS.warning} fill="url(#gradSusp)" strokeWidth={2} dot={false} animationDuration={300} />
                <Area type="monotone" dataKey="fraudulent" stroke={CHART_COLORS.danger} fill="url(#gradFraud)" strokeWidth={2} dot={false} animationDuration={300} />
                <Area type="monotone" dataKey="blocked" stroke={CHART_COLORS.rose} fillOpacity={0} strokeWidth={1.5} strokeDasharray="4 2" dot={false} animationDuration={300} />
              </AreaChart>
            </ResponsiveContainer>
          </ChartCard>

          <ChartCard title="Fraud Detection Rate" subtitle="% fraudulent vs threshold" icon={TrendingUp} badge="LINE">
            <ResponsiveContainer width="100%" height={220}>
              <LineChart data={store.fraudRate}>
                <CartesianGrid {...GRID_STYLE} />
                <XAxis dataKey="time" tick={{ fontSize: 9, fill: '#64748b' }} tickLine={false} axisLine={false} />
                <YAxis tick={{ fontSize: 9, fill: '#64748b' }} tickLine={false} axisLine={false} width={30} domain={[0, 'auto']} />
                <Tooltip {...TOOLTIP_STYLE} />
                <ReferenceLine y={3.5} stroke={CHART_COLORS.danger} strokeDasharray="6 3" strokeWidth={1.5} label={{ value: 'ALERT', fill: CHART_COLORS.danger, fontSize: 8, position: 'right' }} />
                <ReferenceLine y={1.2} stroke={CHART_COLORS.success} strokeDasharray="4 4" strokeWidth={1} label={{ value: 'Baseline', fill: CHART_COLORS.success, fontSize: 8, position: 'right' }} />
                <Line type="monotone" dataKey="rate" stroke={CHART_COLORS.warning} strokeWidth={2.5} dot={false} animationDuration={300}
                  style={{ filter: 'drop-shadow(0 0 4px rgba(234,179,8,0.4))' }} />
              </LineChart>
            </ResponsiveContainer>
          </ChartCard>
        </div>

        {/* ====== ROW 2: Channel Breakdown + Pipeline Latency ====== */}
        <div className="grid grid-cols-2 gap-3">
          <ChartCard title="Channel-wise Volume" subtitle="Stacked distribution" icon={Layers} badge="STACKED">
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
                <XAxis dataKey="time" tick={{ fontSize: 9, fill: '#64748b' }} tickLine={false} axisLine={false} />
                <YAxis tick={{ fontSize: 9, fill: '#64748b' }} tickLine={false} axisLine={false} width={35} />
                <Tooltip {...TOOLTIP_STYLE} />
                {Object.entries(CHANNEL_COLORS).map(([key, color]) => (
                  <Area key={key} type="monotone" dataKey={key} stackId="channels" stroke={color} fill={`url(#gradCh${key})`} strokeWidth={1.5} dot={false} animationDuration={300} />
                ))}
              </AreaChart>
            </ResponsiveContainer>
          </ChartCard>

          <ChartCard title="Pipeline Latency Distribution" subtitle="p50 / p95 / p99 + ML Inference" icon={Timer} badge="COMPOSED">
            <ResponsiveContainer width="100%" height={220}>
              <ComposedChart data={store.latencyMetrics}>
                <defs>
                  <linearGradient id="gradP99" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor={CHART_COLORS.danger} stopOpacity={0.2} />
                    <stop offset="95%" stopColor={CHART_COLORS.danger} stopOpacity={0} />
                  </linearGradient>
                </defs>
                <CartesianGrid {...GRID_STYLE} />
                <XAxis dataKey="time" tick={{ fontSize: 9, fill: '#64748b' }} tickLine={false} axisLine={false} />
                <YAxis tick={{ fontSize: 9, fill: '#64748b' }} tickLine={false} axisLine={false} width={35} unit="ms" />
                <Tooltip {...TOOLTIP_STYLE} />
                <Area type="monotone" dataKey="p99" stroke={CHART_COLORS.danger} fill="url(#gradP99)" strokeWidth={1} dot={false} animationDuration={300} />
                <Line type="monotone" dataKey="p95" stroke={CHART_COLORS.orange} strokeWidth={2} dot={false} animationDuration={300} />
                <Line type="monotone" dataKey="p50" stroke={CHART_COLORS.success} strokeWidth={2} dot={false} animationDuration={300} />
                <Bar dataKey="mlInference" fill={CHART_COLORS.purple} fillOpacity={0.4} barSize={6} radius={[2, 2, 0, 0]} animationDuration={300} />
              </ComposedChart>
            </ResponsiveContainer>
          </ChartCard>
        </div>

        {/* ====== ROW 3: Risk Heatmap + Fraud Typology Treemap ====== */}
        <div className="grid grid-cols-2 gap-3">
          <ChartCard title="Temporal Risk Heatmap" subtitle="24h × 7d fraud risk intensity" icon={Target} badge="HEATMAP" className="min-h-[260px]">
            <RiskHeatmap data={store.riskHeatmap} />
          </ChartCard>

          <ChartCard title="Fraud Typology Distribution" subtitle="Relative attack vector sizes" icon={ShieldAlert} badge="TREEMAP">
            <ResponsiveContainer width="100%" height={220}>
              <Treemap
                data={treemapData}
                dataKey="size"
                // eslint-disable-next-line @typescript-eslint/no-explicit-any
                content={<TreemapContent /> as any}
                animationDuration={500}
              />
            </ResponsiveContainer>
          </ChartCard>
        </div>

        {/* ====== THREAT INTELLIGENCE MAPS ====== */}
        <div className="space-y-3">
          {/* Section header */}
          <div className="flex items-center gap-2 pt-2">
            <Globe size={16} className="text-indigo-400" />
            <h2 className="text-sm font-bold text-white">Geographic Threat Intelligence</h2>
            <span className="text-[8px] font-mono px-2 py-0.5 rounded-full bg-rose-500/10 text-rose-400 border border-rose-500/20 flex items-center gap-1">
              <Radio size={7} className="animate-pulse" /> LIVE THREAT FEED
            </span>
          </div>

          {/* World Map — full width */}
          <ChartCard title="Global Threat Landscape" subtitle="Cross-border fraud flow & hotspot intensity" icon={Globe} badge="GEO MAP" className="overflow-hidden">
            <WorldThreatMap
              hotspots={store.threatHotspots}
              flows={store.crossBorderFlows}
              countryThreats={store.countryThreats}
            />
          </ChartCard>

          {/* India Map + Country Intelligence + Attack Vectors */}
          <div className="grid grid-cols-3 gap-3">
            <ChartCard title="India Regional Threat Map" subtitle="State-wise risk heatmap on real geography" icon={MapPin} badge="INDIA MAP" className="col-span-1 overflow-hidden">
              <IndiaRegionalMap regions={store.geoRegions} />
            </ChartCard>

            <ChartCard title="Country Threat Intelligence" subtitle="Top 10 threat-origin nations" icon={ShieldAlert} badge="THREAT INDEX">
              <CountryThreatPanel countries={store.countryThreats} />
            </ChartCard>

            <ChartCard title="Attack Vector Breakdown" subtitle="Distribution by fraud technique" icon={Target} badge="VECTORS">
              <AttackVectorBreakdown vectors={store.attackVectors} />
            </ChartCard>
          </div>

          {/* Live Feed + Cross-Border Corridors */}
          <div className="grid grid-cols-2 gap-3">
            <ChartCard title="Live Threat Feed" subtitle="Real-time event stream" icon={Radio} badge="LIVE" className="overflow-hidden">
              <LiveThreatFeed events={store.threatEvents} />
            </ChartCard>

            <ChartCard title="Cross-Border Corridors" subtitle="International suspicious flow analysis" icon={Network} badge="CORRIDORS">
              <CrossBorderCorridors flows={store.crossBorderFlows} />
            </ChartCard>
          </div>
        </div>

        {/* ====== ROW 4: Alert Pipeline Funnel + Model Comparison Radar ====== */}
        <div className="grid grid-cols-3 gap-3">
          <ChartCard title="Alert Processing Pipeline" subtitle="Event funnel analysis" icon={Layers} badge="FUNNEL">
            <ResponsiveContainer width="100%" height={260}>
              <FunnelChart>
                <Tooltip {...TOOLTIP_STYLE} />
                <Funnel
                  dataKey="count"
                  data={store.alertFunnel.map((af, i) => ({
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

          <ChartCard title="ML Model Performance Radar" subtitle="Multi-model comparison" icon={Brain} badge="RADAR">
            <ResponsiveContainer width="100%" height={260}>
              <RadarChart data={store.modelPerformance} cx="50%" cy="50%" outerRadius="70%">
                <PolarGrid stroke="rgba(148,163,184,0.1)" />
                <PolarAngleAxis dataKey="metric" tick={{ fontSize: 8, fill: '#94a3b8' }} />
                <PolarRadiusAxis tick={{ fontSize: 7, fill: '#64748b' }} domain={[70, 100]} />
                <Radar name="Ensemble" dataKey="ensemble" stroke={CHART_COLORS.success} fill={CHART_COLORS.success} fillOpacity={0.15} strokeWidth={2} animationDuration={500} />
                <Radar name="LLM Agent" dataKey="llmAgent" stroke={CHART_COLORS.pink} fill={CHART_COLORS.pink} fillOpacity={0.1} strokeWidth={1.5} animationDuration={500} />
                <Radar name="XGBoost" dataKey="xgboost" stroke={CHART_COLORS.blue} fill={CHART_COLORS.blue} fillOpacity={0.08} strokeWidth={1.5} animationDuration={500} />
                <Radar name="GNN" dataKey="gnn" stroke={CHART_COLORS.purple} fill={CHART_COLORS.purple} fillOpacity={0.08} strokeWidth={1.5} animationDuration={500} />
                <Legend iconType="line" wrapperStyle={{ fontSize: '9px', color: '#94a3b8' }} />
              </RadarChart>
            </ResponsiveContainer>
          </ChartCard>

          <ChartCard title="Model F1 Score Radial" subtitle="Head-to-head comparison" icon={Gauge} badge="RADIAL">
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
          </ChartCard>
        </div>

        {/* ====== ROW 5: Velocity Distribution + Amount Distribution + Account Risk Bands ====== */}
        <div className="grid grid-cols-3 gap-3">
          <ChartCard title="Transaction Velocity Profile" subtitle="Frequency-based risk bands" icon={Zap} badge="GROUPED BAR">
            <ResponsiveContainer width="100%" height={240}>
              <BarChart data={store.velocityDistribution} barGap={2}>
                <CartesianGrid {...GRID_STYLE} />
                <XAxis dataKey="range" tick={{ fontSize: 8, fill: '#64748b' }} tickLine={false} axisLine={false} />
                <YAxis tick={{ fontSize: 9, fill: '#64748b' }} tickLine={false} axisLine={false} width={35} />
                <Tooltip {...TOOLTIP_STYLE} />
                <Bar dataKey="normal" fill={CHART_COLORS.success} fillOpacity={0.7} radius={[3, 3, 0, 0]} barSize={12} animationDuration={400} />
                <Bar dataKey="suspicious" fill={CHART_COLORS.warning} fillOpacity={0.7} radius={[3, 3, 0, 0]} barSize={12} animationDuration={400} />
                <Bar dataKey="fraudulent" fill={CHART_COLORS.danger} fillOpacity={0.8} radius={[3, 3, 0, 0]} barSize={12} animationDuration={400} />
                <Legend iconType="circle" wrapperStyle={{ fontSize: '9px', color: '#94a3b8' }} />
              </BarChart>
            </ResponsiveContainer>
          </ChartCard>

          <ChartCard title="Amount Distribution vs Fraud" subtitle="Transaction size risk correlation" icon={TrendingDown} badge="COMPOSED">
            <ResponsiveContainer width="100%" height={240}>
              <ComposedChart data={store.amountDistribution}>
                <defs>
                  <linearGradient id="gradAmt" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor={CHART_COLORS.secondary} stopOpacity={0.3} />
                    <stop offset="95%" stopColor={CHART_COLORS.secondary} stopOpacity={0} />
                  </linearGradient>
                </defs>
                <CartesianGrid {...GRID_STYLE} />
                <XAxis dataKey="range" tick={{ fontSize: 7, fill: '#64748b' }} tickLine={false} axisLine={false} />
                <YAxis yAxisId="left" tick={{ fontSize: 9, fill: '#64748b' }} tickLine={false} axisLine={false} width={35} />
                <YAxis yAxisId="right" orientation="right" tick={{ fontSize: 9, fill: '#64748b' }} tickLine={false} axisLine={false} width={30} unit="%" />
                <Tooltip {...TOOLTIP_STYLE} />
                <Bar yAxisId="left" dataKey="count" fill="url(#gradAmt)" barSize={16} radius={[4, 4, 0, 0]} animationDuration={400} />
                <Bar yAxisId="left" dataKey="fraudCount" fill={CHART_COLORS.danger} fillOpacity={0.7} barSize={8} radius={[3, 3, 0, 0]} animationDuration={400} />
                <Line yAxisId="right" type="monotone" dataKey="avgRisk" stroke={CHART_COLORS.warning} strokeWidth={2.5} dot={{ fill: CHART_COLORS.warning, r: 3 }} animationDuration={400} />
              </ComposedChart>
            </ResponsiveContainer>
          </ChartCard>

          <ChartCard title="Account Risk Band Census" subtitle="Risk-stratified population" icon={Shield} badge="PIE">
            <ResponsiveContainer width="100%" height={240}>
              <PieChart>
                <Pie
                  data={store.accountRiskBands}
                  cx="50%" cy="50%"
                  innerRadius={50} outerRadius={85}
                  dataKey="count" nameKey="band"
                  paddingAngle={3}
                  strokeWidth={0}
                  animationDuration={500}
                >
                  {store.accountRiskBands.map((entry, index) => (
                    <Cell key={index} fill={entry.color} fillOpacity={0.85} />
                  ))}
                </Pie>
                <Tooltip {...TOOLTIP_STYLE} />
                <Legend iconType="circle" wrapperStyle={{ fontSize: '8px', color: '#94a3b8' }} />
              </PieChart>
            </ResponsiveContainer>
          </ChartCard>
        </div>

        {/* ====== ROW 6: Geo Risk Scatter + Network Radar + Device Fingerprints ====== */}
        <div className="grid grid-cols-3 gap-3">
          <ChartCard title="Regional Risk vs Volume" subtitle="Scatter: bubble = fraud rate" icon={Target} badge="SCATTER">
            <ResponsiveContainer width="100%" height={240}>
              <ScatterChart>
                <CartesianGrid {...GRID_STYLE} />
                <XAxis type="number" dataKey="x" name="Transactions" tick={{ fontSize: 8, fill: '#64748b' }} tickLine={false} axisLine={false} />
                <YAxis type="number" dataKey="y" name="Risk Score" tick={{ fontSize: 8, fill: '#64748b' }} tickLine={false} axisLine={false} width={35} />
                <Tooltip {...TOOLTIP_STYLE} />
                <Scatter data={scatterData} fill={CHART_COLORS.primary} animationDuration={400}>
                  {scatterData.map((_, i) => (
                    <Cell key={i} fill={[CHART_COLORS.primary, CHART_COLORS.secondary, CHART_COLORS.purple, CHART_COLORS.warning, CHART_COLORS.danger, CHART_COLORS.pink, CHART_COLORS.teal, CHART_COLORS.blue, CHART_COLORS.success, CHART_COLORS.orange, CHART_COLORS.emerald, CHART_COLORS.rose][i % 12]} />
                  ))}
                </Scatter>
              </ScatterChart>
            </ResponsiveContainer>
          </ChartCard>

          <ChartCard title="Network Topology Metrics" subtitle="Graph health indicators" icon={Network} badge="RADAR">
            <ResponsiveContainer width="100%" height={240}>
              <RadarChart data={store.networkMetrics} cx="50%" cy="50%" outerRadius="68%">
                <PolarGrid stroke="rgba(148,163,184,0.08)" />
                <PolarAngleAxis dataKey="metric" tick={{ fontSize: 7, fill: '#94a3b8' }} />
                <PolarRadiusAxis tick={false} domain={[0, 'auto']} />
                <Radar name="Current" dataKey="value" stroke={CHART_COLORS.secondary} fill={CHART_COLORS.secondary} fillOpacity={0.2} strokeWidth={2} animationDuration={500} />
                <Tooltip {...TOOLTIP_STYLE} />
              </RadarChart>
            </ResponsiveContainer>
          </ChartCard>

          <ChartCard title="Device Fingerprint Analysis" subtitle="Unique / Flagged / Banned" icon={Fingerprint} badge="HORIZONTAL BAR">
            <ResponsiveContainer width="100%" height={240}>
              <BarChart data={store.deviceFingerprints} layout="vertical" barGap={1}>
                <CartesianGrid {...GRID_STYLE} />
                <XAxis type="number" tick={{ fontSize: 8, fill: '#64748b' }} tickLine={false} axisLine={false} />
                <YAxis type="category" dataKey="type" tick={{ fontSize: 7, fill: '#94a3b8' }} tickLine={false} axisLine={false} width={80} />
                <Tooltip {...TOOLTIP_STYLE} />
                <Bar dataKey="unique" fill={CHART_COLORS.primary} fillOpacity={0.6} barSize={8} radius={[0, 3, 3, 0]} animationDuration={400} />
                <Bar dataKey="flagged" fill={CHART_COLORS.warning} fillOpacity={0.7} barSize={8} radius={[0, 3, 3, 0]} animationDuration={400} />
                <Bar dataKey="banned" fill={CHART_COLORS.danger} fillOpacity={0.8} barSize={8} radius={[0, 3, 3, 0]} animationDuration={400} />
                <Legend iconType="circle" wrapperStyle={{ fontSize: '8px', color: '#94a3b8' }} />
              </BarChart>
            </ResponsiveContainer>
          </ChartCard>
        </div>

        {/* ====== ROW 7: Geo Heat Table + Gauges ====== */}
        <div className="grid grid-cols-3 gap-3">
          {/* Geo Table */}
          <ChartCard title="Regional Fraud Intelligence" subtitle="State-wise breakdown" icon={Target} badge="TABLE" className="col-span-2">
            <div className="overflow-x-auto">
              <table className="w-full text-[10px]">
                <thead>
                  <tr className="border-b border-slate-800">
                    <th className="text-left py-2 px-3 text-slate-500 font-medium">Region</th>
                    <th className="text-right py-2 px-3 text-slate-500 font-medium">Transactions</th>
                    <th className="text-right py-2 px-3 text-slate-500 font-medium">Fraud Rate</th>
                    <th className="text-right py-2 px-3 text-slate-500 font-medium">Volume (₹)</th>
                    <th className="text-right py-2 px-3 text-slate-500 font-medium">Risk Score</th>
                    <th className="text-center py-2 px-3 text-slate-500 font-medium">Risk Bar</th>
                  </tr>
                </thead>
                <tbody>
                  {store.geoRegions.sort((a, b) => b.riskScore - a.riskScore).map(r => (
                    <tr key={r.region} className="border-b border-slate-800/40 hover:bg-slate-800/20 transition-colors">
                      <td className="py-1.5 px-3 text-slate-300 font-medium">{r.region}</td>
                      <td className="py-1.5 px-3 text-right text-slate-400 font-mono tabular-nums">{r.transactions.toLocaleString()}</td>
                      <td className="py-1.5 px-3 text-right font-mono tabular-nums">
                        <span className={cn(
                          'px-1.5 py-0.5 rounded-full text-[9px] font-semibold',
                          r.fraudRate > 3 ? 'bg-red-500/15 text-red-400' :
                          r.fraudRate > 2 ? 'bg-orange-500/15 text-orange-400' :
                          r.fraudRate > 1 ? 'bg-yellow-500/15 text-yellow-400' :
                          'bg-green-500/15 text-green-400',
                        )}>
                          {r.fraudRate}%
                        </span>
                      </td>
                      <td className="py-1.5 px-3 text-right text-slate-400 font-mono tabular-nums">₹{(r.amount / 100000).toFixed(1)}L</td>
                      <td className="py-1.5 px-3 text-right font-mono tabular-nums text-slate-300">{r.riskScore}</td>
                      <td className="py-1.5 px-3">
                        <div className="h-1.5 w-full bg-slate-800 rounded-full overflow-hidden">
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
          </ChartCard>

          {/* Gauges */}
          <ChartCard title="System Health Gauges" subtitle="Critical metrics" icon={Gauge} badge="GAUGE">
            <div className="grid grid-cols-2 gap-2 pt-2">
              <GaugeChart value={store.modelAccuracy} max={100} label="Accuracy %" color={CHART_COLORS.success} />
              <GaugeChart value={store.truePositiveRate} max={100} label="TPR %" color={CHART_COLORS.blue} />
              <GaugeChart value={store.falsePositiveRate} max={10} label="FPR %" color={CHART_COLORS.danger} />
              <GaugeChart value={store.avgResponseMs} max={100} label="Latency ms" color={CHART_COLORS.secondary} />
            </div>
          </ChartCard>
        </div>

        {/* ====== ROW 8: Fraud Typology Bar + Live Trend Comparison ====== */}
        <div className="grid grid-cols-2 gap-3">
          <ChartCard title="Fraud Pattern Trend Analysis" subtitle="Attack vector comparison" icon={TrendingUp} badge="HORIZONTAL BAR">
            <ResponsiveContainer width="100%" height={240}>
              <BarChart data={store.fraudTypologies} layout="vertical" barGap={2}>
                <CartesianGrid {...GRID_STYLE} />
                <XAxis type="number" tick={{ fontSize: 8, fill: '#64748b' }} tickLine={false} axisLine={false} />
                <YAxis type="category" dataKey="name" tick={{ fontSize: 8, fill: '#94a3b8' }} tickLine={false} axisLine={false} width={100} />
                <Tooltip {...TOOLTIP_STYLE} />
                <Bar dataKey="count" radius={[0, 4, 4, 0]} barSize={14} animationDuration={400}>
                  {store.fraudTypologies.map((ft, i) => (
                    <Cell key={i} fill={ft.color} fillOpacity={0.75} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </ChartCard>

          <ChartCard title="Fraud Typology Donut" subtitle="Proportional breakdown" icon={ShieldAlert} badge="DONUT">
            <ResponsiveContainer width="100%" height={240}>
              <PieChart>
                <Pie
                  data={store.fraudTypologies}
                  cx="50%" cy="50%"
                  innerRadius={55} outerRadius={90}
                  dataKey="count" nameKey="name"
                  paddingAngle={2}
                  strokeWidth={0}
                  animationDuration={500}
                >
                  {store.fraudTypologies.map((ft, i) => (
                    <Cell key={i} fill={ft.color} fillOpacity={0.8} />
                  ))}
                </Pie>
                <Tooltip {...TOOLTIP_STYLE} />
                <Legend iconType="circle" wrapperStyle={{ fontSize: '8px', color: '#94a3b8' }} />
              </PieChart>
            </ResponsiveContainer>
          </ChartCard>
        </div>

        {/* Footer */}
        <div className="text-center py-3 border-t border-slate-800/50">
          <p className="text-[9px] text-slate-600 font-mono">
            PayFlow v2.0 &mdash; Real-Time Fraud Intelligence Platform &mdash; Union Bank of India &mdash; Idea 2.0 Hackathon
          </p>
        </div>
      </div>
    </div>
  )
}

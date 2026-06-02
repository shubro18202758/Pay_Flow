// ============================================================================
// Analytics Store -- backend-derived fraud analytics only
// ============================================================================

import { create } from 'zustand'
import type {
  CytoEdge,
  CytoNode,
  FraudTypologyResponse,
  HardwareSnapshot,
  OrchestratorMetrics,
  RiskDistributionResponse,
  SSEAgentData,
  SSEAgentVerdict,
  SSERiskScoreAlert,
  SystemSnapshot,
  TemporalHeatmapResponse,
  ThreatSummaryResponse,
  VelocityTrendsResponse,
} from '@/lib/types'

export interface TimeSeriesPoint {
  time: string
  timestamp: number
}

export interface TransactionVolumePoint extends TimeSeriesPoint {
  legitimate: number
  suspicious: number
  fraudulent: number
  blocked: number
}

export interface FraudRatePoint extends TimeSeriesPoint {
  rate: number
  baseline: number | null
  threshold: number | null
}

export interface ChannelVolumePoint extends TimeSeriesPoint {
  UPI: number
  NEFT: number
  RTGS: number
  IMPS: number
  Card: number
  Wallet: number
}

export interface LatencyPoint extends TimeSeriesPoint {
  p50: number
  p95: number
  p99: number
  mlInference: number
}

export interface RiskHeatmapCell {
  hour: number
  day: string
  value: number
}

export interface GeoRegion {
  region: string
  transactions: number
  fraudRate: number
  amount: number
  riskScore: number
  lat: number
  lng: number
}

export interface ThreatHotspot {
  id: string
  city: string
  country: string
  lat: number
  lng: number
  threatLevel: number
  activeAlerts: number
  attackType: string
  intensity: number
}

export interface CrossBorderFlow {
  from: { lat: number; lng: number; label: string }
  to: { lat: number; lng: number; label: string }
  amount: number
  riskScore: number
  channel: string
  txCount: number
  lastDetected: number
  trend?: number | null
}

export interface CountryThreat {
  iso: string
  name: string
  threatIndex: number
  incidents: number
  blocked: number
  primaryAttack: string
  trend?: number | null
  blockRate: number
}

export interface ThreatEvent {
  id: string
  timestamp: number
  timestampSource: 'backend' | 'unavailable'
  severity: 'low' | 'medium' | 'high' | 'critical'
  city: string
  country: string
  sourceLabel: string
  description: string
  attackType: string
  amount: number
  status: 'detected' | 'investigating' | 'blocked' | 'escalated'
}

export interface AttackVectorStat {
  attackType: string
  count: number
  percentage: number
  trend?: number | null
  color: string
}

export interface FraudTypology {
  name: string
  count: number
  percentage: number
  color: string
  trend?: number | null
}

export interface ModelPerformance {
  metric: string
  xgboost: number
  gnn: number | null
  ensemble: number
  llmAgent: number
}

export interface VelocityBucket {
  range: string
  normal: number
  suspicious: number
  fraudulent: number
}

export interface AccountRiskBand {
  band: string
  count: number
  value: number
  color: string
}

export interface NetworkMetric {
  metric: string
  value: number
  fullMark: number
}

export interface AlertFunnel {
  stage: string
  count: number
  percentage: number
}

export interface DeviceFingerprint {
  type: string
  unique: number
  flagged: number
  banned: number
}

export interface AmountDistribution {
  range: string
  count: number
  fraudCount: number
  fraudRate: number
}

interface AnalyticsState {
  transactionVolume: TransactionVolumePoint[]
  fraudRate: FraudRatePoint[]
  channelVolume: ChannelVolumePoint[]
  latencyMetrics: LatencyPoint[]
  riskHeatmap: RiskHeatmapCell[]
  geoRegions: GeoRegion[]
  fraudTypologies: FraudTypology[]
  modelPerformance: ModelPerformance[]
  velocityDistribution: VelocityBucket[]
  accountRiskBands: AccountRiskBand[]
  networkMetrics: NetworkMetric[]
  alertFunnel: AlertFunnel[]
  deviceFingerprints: DeviceFingerprint[]
  amountDistribution: AmountDistribution[]
  threatHotspots: ThreatHotspot[]
  crossBorderFlows: CrossBorderFlow[]
  countryThreats: CountryThreat[]
  threatEvents: ThreatEvent[]
  attackVectors: AttackVectorStat[]
  totalProcessed: number
  totalFlagged: number
  totalBlocked: number
  avgResponseMs: number
  modelAccuracy: number
  falsePositiveRate: number
  truePositiveRate: number
  activeMules: number
  riskScore: number
  throughputTps: number
  seenVerdictIds: string[]
  ingestSystemSnapshot: (snapshot: SystemSnapshot | Record<string, unknown>) => void
  ingestGraphTopology: (nodes: CytoNode[], edges: CytoEdge[]) => void
  ingestGraphBatch: (nodes: CytoNode[], edges: CytoEdge[]) => void
  ingestRiskScore: (data: SSERiskScoreAlert, serverTimestamp?: number) => void
  ingestAgentEvent: (data: SSEAgentData) => void
  hydrateAnalytics: (data: {
    riskDistribution?: RiskDistributionResponse
    fraudTypology?: FraudTypologyResponse
    velocityTrends?: VelocityTrendsResponse
    temporalHeatmap?: TemporalHeatmapResponse
    threatSummary?: ThreatSummaryResponse
  }) => void
  tick: () => void
}

const MAX_SERIES = 60
const TYPOLOGY_COLORS = ['#DA251C', '#00579C', '#B51A13', '#003F73', '#f97316', '#f5b400', '#0f9f6e', '#718096']
type ChannelKey = keyof Omit<ChannelVolumePoint, 'time' | 'timestamp'>

const CHANNEL_LABELS: Record<string, ChannelKey> = {
  '1': 'UPI',
  '2': 'NEFT',
  '3': 'RTGS',
  '4': 'IMPS',
  '5': 'Card',
  '6': 'Wallet',
  UPI: 'UPI',
  NEFT: 'NEFT',
  RTGS: 'RTGS',
  IMPS: 'IMPS',
  CARD: 'Card',
  Card: 'Card',
  WALLET: 'Wallet',
  Wallet: 'Wallet',
}

const EMPTY_CHANNELS: Record<ChannelKey, number> = {
  UPI: 0,
  NEFT: 0,
  RTGS: 0,
  IMPS: 0,
  Card: 0,
  Wallet: 0,
}

const INDIA_REGION_CENTERS = [
  { region: 'Visakhapatnam / Andhra Pradesh', city: 'Visakhapatnam', lat: 17.6868, lng: 83.2185 },
  { region: 'Bengaluru / Karnataka', city: 'Bengaluru', lat: 12.9716, lng: 77.5946 },
  { region: 'Chennai / Tamil Nadu', city: 'Chennai', lat: 13.0827, lng: 80.2707 },
  { region: 'Mumbai / Maharashtra', city: 'Mumbai', lat: 18.9388, lng: 72.8354 },
  { region: 'Delhi NCR', city: 'Delhi', lat: 28.6139, lng: 77.2090 },
  { region: 'Kolkata / West Bengal', city: 'Kolkata', lat: 22.5726, lng: 88.3639 },
  { region: 'Ahmedabad / Gujarat', city: 'Ahmedabad', lat: 23.0225, lng: 72.5714 },
  { region: 'Jaipur / Rajasthan', city: 'Jaipur', lat: 26.9124, lng: 75.7873 },
  { region: 'Lucknow / Uttar Pradesh', city: 'Lucknow', lat: 26.8467, lng: 80.9462 },
  { region: 'Patna / Bihar', city: 'Patna', lat: 25.6093, lng: 85.1376 },
  { region: 'Kochi / Kerala', city: 'Kochi', lat: 9.9312, lng: 76.2673 },
] as const

type IndiaRegionCenter = typeof INDIA_REGION_CENTERS[number]

interface RegionRollup {
  center: IndiaRegionCenter
  transactions: number
  fraudCount: number
  amountInr: number
  attackCounts: Map<string, number>
}

interface CorridorRollup {
  from: IndiaRegionCenter
  to: IndiaRegionCenter
  channel: string
  txCount: number
  fraudCount: number
  amountInr: number
  lastDetected: number
}

function timeLabel(ts?: number | null): string {
  if (typeof ts !== 'number' || !Number.isFinite(ts) || ts <= 0) return 'n/a'
  const d = new Date(ts)
  return `${d.getHours().toString().padStart(2, '0')}:${d.getMinutes().toString().padStart(2, '0')}:${d.getSeconds().toString().padStart(2, '0')}`
}

function msTimestampOrZero(raw: unknown): number {
  const n = Number(raw)
  return Number.isFinite(n) && n > 0 ? n * 1000 : 0
}

function dayLabel(tsSeconds: number): string {
  if (!Number.isFinite(tsSeconds) || tsSeconds <= 0) return 'n/a'
  return new Date(tsSeconds * 1000).toLocaleDateString('en-US', { weekday: 'short' })
}

function clampMetric(value: number, min = 0, max = 100): number {
  if (!Number.isFinite(value)) return min
  return Math.min(max, Math.max(min, value))
}

function pct(part: number, total: number): number {
  return total > 0 ? clampMetric(Number(((part / total) * 100).toFixed(2))) : 0
}

function scoreToPercent(value: unknown): number {
  const n = Number(value ?? 0)
  return Number.isFinite(n) ? clampMetric(Number((n * 100).toFixed(1))) : 0
}

function evidenceBackedLlmConfidence(data: SSEAgentData): number | null {
  if (data.type !== 'verdict') return null
  const confidence = Number(data.confidence)
  const confidenceSource = String(data.confidence_source ?? '')
  const parseStatus = String(data.llm_parse_status ?? '')
  const sourceIsQwenJson =
    confidenceSource === 'qwen_json' ||
    confidenceSource === 'qwen_json_evidence'
  const statusIsParseable =
    parseStatus === '' ||
    parseStatus === 'parsed' ||
    parseStatus === 'json_extracted' ||
    parseStatus === 'parsed_from_forced_json'
  if (
    !Number.isFinite(confidence) ||
    confidence < 0 ||
    confidence > 1 ||
    !sourceIsQwenJson ||
    !statusIsParseable
  ) {
    return null
  }
  return Number((confidence * 100).toFixed(1))
}

function labelForTypology(raw: string): string {
  return raw.replace(/_/g, ' ').replace(/\b\w/g, (m) => m.toUpperCase())
}

function channelLabel(edge: CytoEdge): string {
  const raw = String(edge.data.channel ?? '').toUpperCase()
  return CHANNEL_LABELS[raw] ?? CHANNEL_LABELS[String(edge.data.channel ?? '')] ?? 'Other'
}

function hasGeo(lat: unknown, lng: unknown): lat is number {
  const nLat = Number(lat)
  const nLng = Number(lng)
  return Number.isFinite(nLat) && Number.isFinite(nLng) && nLat >= 6 && nLat <= 37 && nLng >= 68 && nLng <= 98
}

function nearestRegion(lat: number, lng: number): IndiaRegionCenter {
  let best: IndiaRegionCenter = INDIA_REGION_CENTERS[0]
  let bestDistance = Number.POSITIVE_INFINITY
  for (const region of INDIA_REGION_CENTERS) {
    const dLat = lat - region.lat
    const dLng = lng - region.lng
    const distance = dLat * dLat + dLng * dLng
    if (distance < bestDistance) {
      best = region
      bestDistance = distance
    }
  }
  return best
}

function riskFromCounts(transactions: number, fraudCount: number, amountInr: number): number {
  const fraudRate = pct(fraudCount, transactions)
  const volumePressure = Math.log10(amountInr + 1) * 3
  const velocityPressure = Math.log10(transactions + 1) * 8
  return Math.round(Math.min(100, fraudRate * 0.72 + volumePressure + velocityPressure))
}

function dominantAttack(counts: Map<string, number>): string {
  let label = 'No fraud label'
  let best = 0
  counts.forEach((count, key) => {
    if (count > best) {
      best = count
      label = key
    }
  })
  return label
}

function riskBand(score: number): AccountRiskBand['band'] {
  if (score < 0.2) return 'Low (0-20)'
  if (score < 0.4) return 'Medium (20-40)'
  if (score < 0.6) return 'Elevated (40-60)'
  if (score < 0.8) return 'High (60-80)'
  return 'Critical (80-100)'
}

function bucketAmount(amountPaisa: number): AmountDistribution['range'] {
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

function deriveAmountDistribution(edges: CytoEdge[]): AmountDistribution[] {
  const order = ['< INR 500', 'INR 500-2K', 'INR 2K-10K', 'INR 10K-50K', 'INR 50K-1L', 'INR 1L-5L', 'INR 5L-10L', '> INR 10L']
  const buckets = new Map<string, { count: number; fraudCount: number }>()
  for (const range of order) buckets.set(range, { count: 0, fraudCount: 0 })
  for (const edge of edges) {
    const range = bucketAmount(edge.data.amount_paisa ?? 0)
    const current = buckets.get(range) ?? { count: 0, fraudCount: 0 }
    const isFraud = (edge.data.fraud_label ?? 0) > 0
    current.count += 1
    current.fraudCount += isFraud ? 1 : 0
    buckets.set(range, current)
  }
  return order.map((range) => {
    const b = buckets.get(range)!
    return {
      range,
      count: b.count,
      fraudCount: b.fraudCount,
      fraudRate: pct(b.fraudCount, b.count),
    }
  })
}

function deriveChannelVolume(edges: CytoEdge[]): ChannelVolumePoint[] {
  if (!edges.length) return []
  const counts = { ...EMPTY_CHANNELS }
  for (const edge of edges) {
    const label = channelLabel(edge)
    if (label in counts) counts[label as ChannelKey] += 1
  }
  const latestSeconds = Math.max(...edges.map((edge) => Number(edge.data.timestamp ?? 0)))
  const latestTs = Number.isFinite(latestSeconds) && latestSeconds > 0 ? latestSeconds * 1000 : 0
  return [{ time: timeLabel(latestTs), timestamp: latestTs, ...counts }]
}

function collectRegionalRollups(edges: CytoEdge[]): Map<string, RegionRollup> {
  const rollups = new Map<string, RegionRollup>()
  for (const edge of edges) {
    const lat = Number(edge.data.sender_geo_lat)
    const lng = Number(edge.data.sender_geo_lon)
    if (!hasGeo(lat, lng)) continue
    const center = nearestRegion(lat, lng)
    const key = center.region
    const current = rollups.get(key) ?? {
      center,
      transactions: 0,
      fraudCount: 0,
      amountInr: 0,
      attackCounts: new Map<string, number>(),
    }
    const fraud = (edge.data.fraud_label ?? 0) > 0
    const attack = labelForTypology(edge.data.fraud_label_name || 'NONE')
    current.transactions += 1
    current.fraudCount += fraud ? 1 : 0
    current.amountInr += (edge.data.amount_paisa ?? 0) / 100
    if (fraud) current.attackCounts.set(attack, (current.attackCounts.get(attack) ?? 0) + 1)
    rollups.set(key, current)
  }
  return rollups
}

function deriveGeoRegions(edges: CytoEdge[]): GeoRegion[] {
  return [...collectRegionalRollups(edges).values()]
    .map((item) => ({
      region: item.center.region,
      transactions: item.transactions,
      fraudRate: pct(item.fraudCount, item.transactions),
      amount: Math.round(item.amountInr),
      riskScore: riskFromCounts(item.transactions, item.fraudCount, item.amountInr),
      lat: item.center.lat,
      lng: item.center.lng,
    }))
    .sort((a, b) => b.riskScore - a.riskScore)
}

function deriveThreatHotspots(edges: CytoEdge[]): ThreatHotspot[] {
  return [...collectRegionalRollups(edges).values()]
    .filter((item) => item.fraudCount > 0)
    .map((item) => {
      const threatLevel = riskFromCounts(item.transactions, item.fraudCount, item.amountInr)
      return {
        id: item.center.region,
        city: item.center.city,
        country: 'India',
        lat: item.center.lat,
        lng: item.center.lng,
        threatLevel,
        activeAlerts: item.fraudCount,
        attackType: dominantAttack(item.attackCounts),
        intensity: Number((threatLevel / 100).toFixed(2)),
      }
    })
    .sort((a, b) => b.threatLevel - a.threatLevel)
    .slice(0, 12)
}

function deriveInterRegionFlows(edges: CytoEdge[]): CrossBorderFlow[] {
  const corridors = new Map<string, CorridorRollup>()
  for (const edge of edges) {
    const sLat = Number(edge.data.sender_geo_lat)
    const sLng = Number(edge.data.sender_geo_lon)
    const rLat = Number(edge.data.receiver_geo_lat)
    const rLng = Number(edge.data.receiver_geo_lon)
    if (!hasGeo(sLat, sLng) || !hasGeo(rLat, rLng)) continue
    const from = nearestRegion(sLat, sLng)
    const to = nearestRegion(rLat, rLng)
    if (from.region === to.region) continue
    const channel = channelLabel(edge)
    const key = `${from.region}->${to.region}:${channel}`
    const current = corridors.get(key) ?? {
      from,
      to,
      channel,
      txCount: 0,
      fraudCount: 0,
      amountInr: 0,
      lastDetected: 0,
    }
    current.txCount += 1
    current.fraudCount += (edge.data.fraud_label ?? 0) > 0 ? 1 : 0
    current.amountInr += (edge.data.amount_paisa ?? 0) / 100
    current.lastDetected = Math.max(current.lastDetected, Number(edge.data.timestamp ?? 0) * 1000)
    corridors.set(key, current)
  }
  return [...corridors.values()]
    .map((item) => ({
      from: { lat: item.from.lat, lng: item.from.lng, label: item.from.city },
      to: { lat: item.to.lat, lng: item.to.lng, label: item.to.city },
      amount: Math.round(item.amountInr),
      riskScore: riskFromCounts(item.txCount, item.fraudCount, item.amountInr),
      channel: item.channel,
      txCount: item.txCount,
      lastDetected: item.lastDetected,
      trend: null,
    }))
    .sort((a, b) => b.riskScore - a.riskScore)
    .slice(0, 15)
}

function deriveNationalThreat(edges: CytoEdge[]): CountryThreat[] {
  if (!edges.length) return []
  const attackCounts = new Map<string, number>()
  let incidents = 0
  let fraudCount = 0
  let amountInr = 0
  for (const edge of edges) {
    const fraud = (edge.data.fraud_label ?? 0) > 0
    incidents += 1
    fraudCount += fraud ? 1 : 0
    amountInr += (edge.data.amount_paisa ?? 0) / 100
    if (fraud) {
      const attack = labelForTypology(edge.data.fraud_label_name || 'UNKNOWN')
      attackCounts.set(attack, (attackCounts.get(attack) ?? 0) + 1)
    }
  }
  return [{
    iso: 'IND',
    name: 'India',
    threatIndex: riskFromCounts(incidents, fraudCount, amountInr),
    incidents,
    blocked: fraudCount,
    primaryAttack: dominantAttack(attackCounts),
    trend: null,
    blockRate: pct(fraudCount, incidents),
  }]
}

function deriveDeviceFingerprints(edges: CytoEdge[]): DeviceFingerprint[] {
  const devices = new Map<string, { tx: number; fraud: number; amountInr: number; accounts: Set<string> }>()
  for (const edge of edges) {
    const fp = edge.data.device_fingerprint
    if (!fp) continue
    const current = devices.get(fp) ?? { tx: 0, fraud: 0, amountInr: 0, accounts: new Set<string>() }
    current.tx += 1
    current.fraud += (edge.data.fraud_label ?? 0) > 0 ? 1 : 0
    current.amountInr += (edge.data.amount_paisa ?? 0) / 100
    current.accounts.add(edge.data.source)
    current.accounts.add(edge.data.target)
    devices.set(fp, current)
  }
  if (devices.size === 0) return []
  const rows = [...devices.values()]
  const fraudLinked = rows.filter((item) => item.fraud > 0)
  const multiAccount = rows.filter((item) => item.accounts.size >= 3)
  const highValue = rows.filter((item) => item.amountInr >= 500_000)
  return [
    { type: 'Unique devices', unique: devices.size, flagged: fraudLinked.length, banned: 0 },
    { type: 'Multi-account devices', unique: multiAccount.length, flagged: multiAccount.filter((item) => item.fraud > 0).length, banned: 0 },
    { type: 'High-value devices', unique: highValue.length, flagged: highValue.filter((item) => item.fraud > 0).length, banned: 0 },
  ]
}

function deriveVelocityBuckets(accounts: VelocityTrendsResponse['accounts'] = []): VelocityBucket[] {
  const result: VelocityBucket[] = [
    { range: '0-5 tx/window', normal: 0, suspicious: 0, fraudulent: 0 },
    { range: '5-15 tx/window', normal: 0, suspicious: 0, fraudulent: 0 },
    { range: '15-30 tx/window', normal: 0, suspicious: 0, fraudulent: 0 },
    { range: '30-60 tx/window', normal: 0, suspicious: 0, fraudulent: 0 },
    { range: '60+ tx/window', normal: 0, suspicious: 0, fraudulent: 0 },
  ]
  for (const acct of accounts) {
    const idx = acct.count < 5 ? 0 : acct.count < 15 ? 1 : acct.count < 30 ? 2 : acct.count < 60 ? 3 : 4
    if (acct.fraud_count > 0) result[idx].fraudulent += 1
    else if (acct.count >= 15) result[idx].suspicious += 1
    else result[idx].normal += 1
  }
  return result
}

function deriveFraudTypologies(data?: FraudTypologyResponse): FraudTypology[] {
  const total = Number(data?.total ?? 0)
  const typology = data?.typology ?? {}
  if (!Number.isFinite(total) || total <= 0 || Object.keys(typology).length === 0) return []
  return Object.entries(typology)
    .sort((a, b) => b[1] - a[1])
    .map(([name, count], index) => ({
      name: labelForTypology(name),
      count,
      percentage: pct(count, total),
      color: TYPOLOGY_COLORS[index % TYPOLOGY_COLORS.length],
      trend: null,
    }))
}

function deriveAttackVectors(data?: FraudTypologyResponse): AttackVectorStat[] {
  const typologies = deriveFraudTypologies(data)
  return typologies.map((item) => ({
    attackType: item.name,
    count: item.count,
    percentage: item.percentage,
    trend: null,
    color: item.color,
  }))
}

function deriveRiskBands(data?: RiskDistributionResponse): AccountRiskBand[] {
  const labels: AccountRiskBand[] = [
    { band: 'Low (0-20)', count: 0, value: 0.15, color: '#0f9f6e' },
    { band: 'Medium (20-40)', count: 0, value: 0.30, color: '#2f79b5' },
    { band: 'Elevated (40-60)', count: 0, value: 0.50, color: '#f5b400' },
    { band: 'High (60-80)', count: 0, value: 0.70, color: '#f97316' },
    { band: 'Critical (80-100)', count: 0, value: 0.90, color: '#DA251C' },
  ]
  if (!data?.buckets?.length) return labels
  data.buckets.forEach((count, idx) => {
    const bandIdx = Math.min(Math.floor(idx / 2), labels.length - 1)
    labels[bandIdx] = { ...labels[bandIdx], count: labels[bandIdx].count + count }
  })
  return labels
}

function deriveRiskHeatmap(data?: TemporalHeatmapResponse): RiskHeatmapCell[] {
  if (!data?.buckets?.length) return []
  return data.buckets.filter((bucket) => Number.isFinite(bucket.bucket_start) && bucket.bucket_start > 0).map((bucket) => {
    const fraudRate = pct(bucket.fraud_count, bucket.txn_count)
    return {
      day: dayLabel(bucket.bucket_start),
      hour: new Date(bucket.bucket_start * 1000).getHours(),
      value: Math.round(Math.min(100, fraudRate + Math.log10(bucket.txn_count + 1) * 10)),
    }
  })
}

function deriveThreatEvents(data?: ThreatSummaryResponse): ThreatEvent[] {
  if (!data?.indicators?.length) return []
  const hasBackendTimestamp = typeof data.generated_at === 'number' && Number.isFinite(data.generated_at)
  const generatedAtMs = hasBackendTimestamp ? data.generated_at! * 1000 : 0
  const timestampSource = hasBackendTimestamp ? 'backend' : 'unavailable'
  return data.indicators.map((indicator, index) => ({
    id: `${indicator.signal}-${index}`,
    timestamp: generatedAtMs,
    timestampSource,
    severity: indicator.severity,
    city: 'PayFlow',
    country: 'backend telemetry',
    sourceLabel: 'Backend threat-summary',
    description: indicator.detail,
    attackType: indicator.signal.replace(/_/g, ' '),
    amount: 0,
    status:
      indicator.signal === 'HIGH_FREEZE_RATE' || indicator.signal === 'ACTIVE_FREEZES'
        ? 'blocked'
        : indicator.signal === 'ACTIVE_ATTACK' ||
            indicator.signal === 'MULTI_VECTOR_ATTACK' ||
            indicator.signal === 'HIGH_FRAUD_DENSITY' ||
            indicator.signal === 'ELEVATED_FRAUD'
          ? 'investigating'
          : 'detected',
  }))
}

function deriveNetworkMetrics(nodes: CytoNode[], edges: CytoEdge[]): NetworkMetric[] {
  const fraudEdges = edges.filter((edge) => (edge.data.fraud_label ?? 0) > 0).length
  const frozenNodes = nodes.filter((node) => node.data.status === 'frozen').length
  const suspiciousNodes = nodes.filter((node) => node.data.status === 'suspicious').length
  const avgDegree = nodes.length > 0 ? (edges.length * 2) / nodes.length : 0
  return [
    { metric: 'Avg Degree', value: Number((avgDegree * 10).toFixed(1)), fullMark: 120 },
    { metric: 'Fraud Edge Ratio', value: pct(fraudEdges, edges.length), fullMark: 100 },
    { metric: 'Frozen Nodes', value: frozenNodes, fullMark: Math.max(10, nodes.length) },
    { metric: 'Suspicious Nodes', value: suspiciousNodes, fullMark: Math.max(10, nodes.length) },
    { metric: 'Graph Density', value: nodes.length > 1 ? Number(((edges.length / (nodes.length * (nodes.length - 1))) * 100).toFixed(2)) : 0, fullMark: 100 },
  ]
}

function addOrReplaceSeries<T extends TimeSeriesPoint>(series: T[], point: T): T[] {
  const previous = series[series.length - 1]
  if (previous && point.timestamp - previous.timestamp < 900) {
    return [...series.slice(0, -1), point]
  }
  return [...series, point].slice(-MAX_SERIES)
}

function baseState() {
  return {
    transactionVolume: [] as TransactionVolumePoint[],
    fraudRate: [] as FraudRatePoint[],
    channelVolume: [] as ChannelVolumePoint[],
    latencyMetrics: [] as LatencyPoint[],
    riskHeatmap: [] as RiskHeatmapCell[],
    geoRegions: [] as GeoRegion[],
    fraudTypologies: [] as FraudTypology[],
    modelPerformance: [] as ModelPerformance[],
    velocityDistribution: [] as VelocityBucket[],
    accountRiskBands: deriveRiskBands(),
    networkMetrics: [] as NetworkMetric[],
    alertFunnel: [] as AlertFunnel[],
    deviceFingerprints: [] as DeviceFingerprint[],
    amountDistribution: [] as AmountDistribution[],
    threatHotspots: [] as ThreatHotspot[],
    crossBorderFlows: [] as CrossBorderFlow[],
    countryThreats: [] as CountryThreat[],
    threatEvents: [] as ThreatEvent[],
    attackVectors: [] as AttackVectorStat[],
    totalProcessed: 0,
    totalFlagged: 0,
    totalBlocked: 0,
    avgResponseMs: 0,
    modelAccuracy: 0,
    falsePositiveRate: 0,
    truePositiveRate: 0,
    activeMules: 0,
    riskScore: 0,
    throughputTps: 0,
    seenVerdictIds: [] as string[],
  }
}

function verdictIngestionKey(data: SSEAgentVerdict): string {
  return [
    data.txn_id,
    data.verdict,
    data.recommended_action,
    data.model_used ?? '',
  ].join('|')
}

export const useAnalyticsStore = create<AnalyticsState>((set, get) => ({
  ...baseState(),

  ingestSystemSnapshot: (snapshot) =>
    set((state) => {
      const typed = snapshot as SystemSnapshot
      const raw = snapshot as Record<string, Record<string, unknown> | undefined>
      const orchestrator = typed.orchestrator as OrchestratorMetrics | undefined
      const hardware = typed.hardware as HardwareSnapshot | undefined
      const circuitBreaker = typed.circuit_breaker as Record<string, unknown> | undefined
      const agentMetrics = typed.agent?.metrics
      const xgboost = raw.xgboost
      const randomForest = raw.random_forest
      const logistic = raw.logistic_regression
      const now = Date.now()

      const processed = orchestrator?.events_ingested ?? state.totalProcessed
      const flagged = orchestrator?.alerts_routed ?? state.totalFlagged
      const blocked = Number(circuitBreaker?.frozen_count ?? state.totalBlocked)
      const fraudPct = pct(flagged, processed)

      const transactionPoint: TransactionVolumePoint = {
        time: timeLabel(now),
        timestamp: now,
        legitimate: Math.max(0, processed - flagged),
        suspicious: Math.max(0, flagged - blocked),
        fraudulent: agentMetrics?.verdicts?.fraudulent ?? 0,
        blocked,
      }
      const fraudRatePoint: FraudRatePoint = {
        time: transactionPoint.time,
        timestamp: now,
        rate: fraudPct,
        baseline: null,
        threshold: null,
      }
      const latencyPoint: LatencyPoint = {
        time: transactionPoint.time,
        timestamp: now,
        p50: 0,
        p95: 0,
        p99: 0,
        mlInference: hardware?.llm_tps ? Number((1000 / Math.max(hardware.llm_tps, 1)).toFixed(1)) : 0,
      }
      const xgbAucpr = scoreToPercent(xgboost?.best_aucpr)
      const rfAucpr = scoreToPercent(randomForest?.best_aucpr)
      const lrAucpr = scoreToPercent(logistic?.aucpr)
      const lrAucRoc = scoreToPercent(logistic?.auc_roc)
      const bestModelAucpr = Math.max(xgbAucpr, rfAucpr, lrAucpr, state.modelAccuracy)
      const existingLlmConfidence = state.modelPerformance.find((m) => m.metric === 'LLM Confidence')
      const modelPerformance: ModelPerformance[] = [
        {
          metric: 'AUCPR',
          xgboost: xgbAucpr,
          gnn: null,
          ensemble: bestModelAucpr,
          llmAgent: 0,
        },
      ]
      if (lrAucRoc > 0) {
        modelPerformance.push({
          metric: 'ROC-AUC',
          xgboost: 0,
          gnn: null,
          ensemble: lrAucRoc,
          llmAgent: 0,
        })
      }
      if (existingLlmConfidence) {
        modelPerformance.push(existingLlmConfidence)
      }

      return {
        transactionVolume: addOrReplaceSeries(state.transactionVolume, transactionPoint),
        fraudRate: addOrReplaceSeries(state.fraudRate, fraudRatePoint),
        latencyMetrics: addOrReplaceSeries(state.latencyMetrics, latencyPoint),
        totalProcessed: processed,
        totalFlagged: flagged,
        totalBlocked: blocked,
        avgResponseMs: latencyPoint.mlInference,
        modelAccuracy: bestModelAucpr,
        falsePositiveRate: lrAucpr,
        truePositiveRate: xgbAucpr,
        modelPerformance: modelPerformance.some((item) => item.ensemble > 0 || item.xgboost > 0)
          ? modelPerformance
          : state.modelPerformance,
        activeMules: typed.graph?.metrics?.mule_detections ?? state.activeMules,
        riskScore: clampMetric(Number((fraudPct * 3).toFixed(1))),
        throughputTps: Math.round(orchestrator?.events_per_sec ?? 0),
        alertFunnel: [
          { stage: 'Events Ingested', count: processed, percentage: 100 },
          { stage: 'ML Flagged', count: flagged, percentage: fraudPct },
          { stage: 'Frozen', count: blocked, percentage: pct(blocked, processed) },
          { stage: 'Agent Verdicts', count: agentMetrics?.completed ?? 0, percentage: pct(agentMetrics?.completed ?? 0, processed) },
        ],
      }
    }),

  ingestGraphTopology: (nodes, edges) =>
    set({
      channelVolume: deriveChannelVolume(edges),
      networkMetrics: deriveNetworkMetrics(nodes, edges),
      amountDistribution: deriveAmountDistribution(edges),
      geoRegions: deriveGeoRegions(edges),
      threatHotspots: deriveThreatHotspots(edges),
      crossBorderFlows: deriveInterRegionFlows(edges),
      countryThreats: deriveNationalThreat(edges),
      deviceFingerprints: deriveDeviceFingerprints(edges),
    }),

  ingestGraphBatch: (nodes, edges) => {
    if (!nodes.length && !edges.length) return
    const current = get()
    const graphEdges = edges.length ? edges : []
    set({
      channelVolume: graphEdges.length ? deriveChannelVolume(graphEdges) : current.channelVolume,
      networkMetrics: deriveNetworkMetrics(nodes, graphEdges),
      amountDistribution: graphEdges.length ? deriveAmountDistribution(graphEdges) : current.amountDistribution,
      geoRegions: graphEdges.length ? deriveGeoRegions(graphEdges) : current.geoRegions,
      threatHotspots: graphEdges.length ? deriveThreatHotspots(graphEdges) : current.threatHotspots,
      crossBorderFlows: graphEdges.length ? deriveInterRegionFlows(graphEdges) : current.crossBorderFlows,
      countryThreats: graphEdges.length ? deriveNationalThreat(graphEdges) : current.countryThreats,
      deviceFingerprints: graphEdges.length ? deriveDeviceFingerprints(graphEdges) : current.deviceFingerprints,
    })
  },

  ingestRiskScore: (data, serverTimestamp) =>
    set((state) => {
      const observedAt = msTimestampOrZero(serverTimestamp)
      const band = riskBand(data.risk_score)
      const bands = state.accountRiskBands.map((item) =>
        item.band === band ? { ...item, count: item.count + 1 } : item,
      )
      return {
        totalFlagged: data.tier === 'low' ? state.totalFlagged : state.totalFlagged + 1,
        riskScore: clampMetric(Number((data.risk_score * 100).toFixed(1))),
        accountRiskBands: bands,
        fraudRate: addOrReplaceSeries(state.fraudRate, {
          time: timeLabel(observedAt),
          timestamp: observedAt,
          rate: pct(state.totalFlagged + (data.tier === 'low' ? 0 : 1), Math.max(state.totalProcessed, 1)),
          baseline: null,
          threshold: null,
        }),
      }
    }),

  ingestAgentEvent: (data) => {
    if (data.type !== 'verdict') return
    set((state) => {
      const verdictKey = verdictIngestionKey(data)
      if (state.seenVerdictIds.includes(verdictKey)) return {}
      const fraudulent = data.verdict?.toLowerCase().includes('fraud') ? 1 : 0
      const confidence = evidenceBackedLlmConfidence(data)
      const withoutLlmConfidence = confidence == null
        ? state.modelPerformance
        : state.modelPerformance.filter((item) => item.metric !== 'LLM Confidence')
      return {
        seenVerdictIds: [verdictKey, ...state.seenVerdictIds].slice(0, 500),
        totalBlocked: data.recommended_action?.toLowerCase().includes('freeze') ? state.totalBlocked + 1 : state.totalBlocked,
        modelPerformance: confidence == null
          ? withoutLlmConfidence
          : [
              ...withoutLlmConfidence,
              {
                metric: 'LLM Confidence',
                xgboost: 0,
                gnn: null,
                ensemble: 0,
                llmAgent: confidence,
              },
            ],
        totalFlagged: state.totalFlagged + fraudulent,
      }
    })
  },

  hydrateAnalytics: ({ riskDistribution, fraudTypology, velocityTrends, temporalHeatmap, threatSummary }) =>
    set((state) => ({
      fraudTypologies: fraudTypology ? deriveFraudTypologies(fraudTypology) : state.fraudTypologies,
      attackVectors: fraudTypology ? deriveAttackVectors(fraudTypology) : state.attackVectors,
      accountRiskBands: riskDistribution ? deriveRiskBands(riskDistribution) : state.accountRiskBands,
      riskHeatmap: temporalHeatmap ? deriveRiskHeatmap(temporalHeatmap) : state.riskHeatmap,
      velocityDistribution: velocityTrends ? deriveVelocityBuckets(velocityTrends.accounts) : state.velocityDistribution,
      threatEvents: threatSummary ? deriveThreatEvents(threatSummary) : state.threatEvents,
      totalProcessed: riskDistribution?.total ?? state.totalProcessed,
      riskScore: riskDistribution ? clampMetric(Number(((riskDistribution.mean ?? 0) * 100).toFixed(1))) : state.riskScore,
      activeMules: velocityTrends?.accounts?.filter((acct) => acct.fraud_count > 0).length ?? state.activeMules,
      totalBlocked: threatSummary?.frozen_count ?? state.totalBlocked,
    })),

  tick: () => {
    const state = get()
    if (state.totalProcessed === 0 && state.totalFlagged === 0) return
    set((current) => ({
      transactionVolume: current.transactionVolume,
      fraudRate: current.fraudRate,
    }))
  },
}))

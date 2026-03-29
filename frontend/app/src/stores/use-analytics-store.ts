// ============================================================================
// Analytics Store -- Real-time live metrics for fraud detection analytics
// Generates streaming synthetic financial data tied to PayFlow's domain
// ============================================================================

import { create } from 'zustand'

// --- Time-series data points ---

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
  baseline: number
  threshold: number
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
  threatLevel: number   // 0-100
  activeAlerts: number
  attackType: string
  intensity: number     // pulse size multiplier
}

export interface CrossBorderFlow {
  from: { lat: number; lng: number; label: string }
  to:   { lat: number; lng: number; label: string }
  amount: number
  riskScore: number
  channel: string
  txCount: number
  lastDetected: number
  trend: number
}

export interface CountryThreat {
  iso: string
  name: string
  threatIndex: number
  incidents: number
  blocked: number
  primaryAttack: string
  trend: number
  blockRate: number
}

export interface ThreatEvent {
  id: string
  timestamp: number
  severity: 'low' | 'medium' | 'high' | 'critical'
  city: string
  country: string
  description: string
  attackType: string
  amount: number
  status: 'detected' | 'investigating' | 'blocked' | 'escalated'
}

export interface AttackVectorStat {
  attackType: string
  count: number
  percentage: number
  avgRisk: number
  trend: number
  color: string
}

export interface FraudTypology {
  name: string
  count: number
  percentage: number
  color: string
  trend: number
}

export interface ModelPerformance {
  metric: string
  xgboost: number
  gnn: number
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
  avgRisk: number
}

// --- Store ---

interface AnalyticsState {
  // Time-series
  transactionVolume: TransactionVolumePoint[]
  fraudRate: FraudRatePoint[]
  channelVolume: ChannelVolumePoint[]
  latencyMetrics: LatencyPoint[]

  // Aggregated
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

  // Counters
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

  // Actions
  tick: () => void
}

// --- Generators ---

const DAYS = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'] as const
const REGIONS: { name: string; lat: number; lng: number }[] = [
  { name: 'Maharashtra', lat: 19.076, lng: 72.877 },
  { name: 'Delhi NCR', lat: 28.614, lng: 77.209 },
  { name: 'Karnataka', lat: 12.972, lng: 77.595 },
  { name: 'Tamil Nadu', lat: 13.083, lng: 80.271 },
  { name: 'West Bengal', lat: 22.572, lng: 88.364 },
  { name: 'Gujarat', lat: 23.023, lng: 72.572 },
  { name: 'Rajasthan', lat: 26.912, lng: 75.787 },
  { name: 'Uttar Pradesh', lat: 26.847, lng: 80.947 },
  { name: 'Telangana', lat: 17.384, lng: 78.457 },
  { name: 'Kerala', lat: 9.931, lng: 76.267 },
  { name: 'Madhya Pradesh', lat: 23.259, lng: 77.413 },
  { name: 'Punjab', lat: 30.734, lng: 76.779 },
]

const FRAUD_TYPES = [
  { name: 'Account Takeover', color: '#ef4444' },
  { name: 'Money Mule', color: '#f97316' },
  { name: 'Synthetic Identity', color: '#eab308' },
  { name: 'Card Cloning', color: '#a855f7' },
  { name: 'SIM Swap', color: '#ec4899' },
  { name: 'Phishing/Vishing', color: '#06b6d4' },
  { name: 'Loan Fraud', color: '#22c55e' },
  { name: 'Insider Collusion', color: '#6366f1' },
]

const MAX_SERIES = 60

function randBetween(min: number, max: number): number {
  return min + Math.random() * (max - min)
}

function timeLabel(): string {
  const d = new Date()
  return `${d.getHours().toString().padStart(2, '0')}:${d.getMinutes().toString().padStart(2, '0')}:${d.getSeconds().toString().padStart(2, '0')}`
}

function generateHeatmap(): RiskHeatmapCell[] {
  const cells: RiskHeatmapCell[] = []
  for (const day of DAYS) {
    for (let hour = 0; hour < 24; hour++) {
      // Higher fraud risk at night (0-6) and around noon peaks
      const nightBoost = hour < 6 || hour > 22 ? 30 : 0
      const peakBoost = (hour >= 11 && hour <= 14) || (hour >= 18 && hour <= 21) ? 20 : 0
      const weekendBoost = (day === 'Sat' || day === 'Sun') ? 15 : 0
      cells.push({
        hour,
        day,
        value: Math.round(randBetween(5, 40) + nightBoost + peakBoost + weekendBoost),
      })
    }
  }
  return cells
}

function generateGeoRegions(): GeoRegion[] {
  return REGIONS.map(r => ({
    region: r.name,
    lat: r.lat,
    lng: r.lng,
    transactions: Math.round(randBetween(12000, 180000)),
    fraudRate: parseFloat(randBetween(0.3, 4.8).toFixed(2)),
    amount: Math.round(randBetween(50, 800)) * 100000,
    riskScore: parseFloat(randBetween(15, 85).toFixed(1)),
  }))
}

const THREAT_HOTSPOTS: { city: string; country: string; lat: number; lng: number }[] = [
  { city: 'Mumbai', country: 'India', lat: 19.076, lng: 72.877 },
  { city: 'Delhi', country: 'India', lat: 28.614, lng: 77.209 },
  { city: 'Bengaluru', country: 'India', lat: 12.972, lng: 77.595 },
  { city: 'Lagos', country: 'Nigeria', lat: 6.524, lng: 3.379 },
  { city: 'Moscow', country: 'Russia', lat: 55.756, lng: 37.618 },
  { city: 'Beijing', country: 'China', lat: 39.904, lng: 116.407 },
  { city: 'Bucharest', country: 'Romania', lat: 44.432, lng: 26.104 },
  { city: 'São Paulo', country: 'Brazil', lat: -23.550, lng: -46.633 },
  { city: 'Jakarta', country: 'Indonesia', lat: -6.175, lng: 106.846 },
  { city: 'Karachi', country: 'Pakistan', lat: 24.860, lng: 67.001 },
  { city: 'Dubai', country: 'UAE', lat: 25.205, lng: 55.271 },
  { city: 'London', country: 'UK', lat: 51.507, lng: -0.128 },
  { city: 'Hong Kong', country: 'China', lat: 22.319, lng: 114.170 },
  { city: 'Singapore', country: 'Singapore', lat: 1.352, lng: 103.820 },
]

const ATTACK_TYPES = [
  'Account Takeover', 'Phishing Relay', 'SIM Swap', 'Card Cloning',
  'Money Mule Ring', 'Synthetic ID', 'Credential Stuffing', 'BEC Fraud',
]

function generateThreatHotspots(): ThreatHotspot[] {
  return THREAT_HOTSPOTS.map((h, i) => ({
    id: `hotspot-${i}`,
    ...h,
    threatLevel: parseFloat(randBetween(10, 95).toFixed(0)),
    activeAlerts: Math.round(randBetween(0, 45)),
    attackType: ATTACK_TYPES[Math.floor(Math.random() * ATTACK_TYPES.length)],
    intensity: parseFloat(randBetween(0.4, 2.5).toFixed(1)),
  }))
}

function generateCrossBorderFlows(): CrossBorderFlow[] {
  const seeds = [
    { from: { lat: 6.524, lng: 3.379, label: 'Lagos' }, to: { lat: 19.076, lng: 72.877, label: 'Mumbai' }, channel: 'SWIFT' },
    { from: { lat: 55.756, lng: 37.618, label: 'Moscow' }, to: { lat: 28.614, lng: 77.209, label: 'Delhi' }, channel: 'RTGS' },
    { from: { lat: 39.904, lng: 116.407, label: 'Beijing' }, to: { lat: 12.972, lng: 77.595, label: 'Bengaluru' }, channel: 'NEFT' },
    { from: { lat: 24.860, lng: 67.001, label: 'Karachi' }, to: { lat: 28.614, lng: 77.209, label: 'Delhi' }, channel: 'Hawala' },
    { from: { lat: 25.205, lng: 55.271, label: 'Dubai' }, to: { lat: 19.076, lng: 72.877, label: 'Mumbai' }, channel: 'SWIFT' },
    { from: { lat: -23.550, lng: -46.633, label: 'São Paulo' }, to: { lat: 22.572, lng: 88.364, label: 'Kolkata' }, channel: 'Wire' },
    { from: { lat: 51.507, lng: -0.128, label: 'London' }, to: { lat: 13.083, lng: 80.271, label: 'Chennai' }, channel: 'SWIFT' },
    { from: { lat: 1.352, lng: 103.820, label: 'Singapore' }, to: { lat: 17.384, lng: 78.457, label: 'Hyderabad' }, channel: 'NEFT' },
  ]
  return seeds.map(f => ({
    ...f,
    amount: Math.round(randBetween(5, 500)) * 10000,
    riskScore: parseFloat(randBetween(25, 92).toFixed(1)),
    txCount: Math.round(randBetween(5, 120)),
    lastDetected: Date.now() - Math.round(randBetween(30000, 1800000)),
    trend: parseFloat(randBetween(-20, 35).toFixed(1)),
  }))
}

const COUNTRY_THREATS: { iso: string; name: string }[] = [
  { iso: 'IND', name: 'India' }, { iso: 'NGA', name: 'Nigeria' }, { iso: 'RUS', name: 'Russia' },
  { iso: 'CHN', name: 'China' }, { iso: 'ROU', name: 'Romania' }, { iso: 'BRA', name: 'Brazil' },
  { iso: 'IDN', name: 'Indonesia' }, { iso: 'PAK', name: 'Pakistan' }, { iso: 'ARE', name: 'UAE' },
  { iso: 'GBR', name: 'United Kingdom' }, { iso: 'USA', name: 'United States' }, { iso: 'UKR', name: 'Ukraine' },
  { iso: 'VNM', name: 'Vietnam' }, { iso: 'PHL', name: 'Philippines' }, { iso: 'MYS', name: 'Malaysia' },
  { iso: 'THA', name: 'Thailand' }, { iso: 'KOR', name: 'South Korea' }, { iso: 'JPN', name: 'Japan' },
  { iso: 'DEU', name: 'Germany' }, { iso: 'FRA', name: 'France' },
]

function generateCountryThreats(): CountryThreat[] {
  return COUNTRY_THREATS.map(c => {
    const incidents = Math.round(randBetween(10, 1200))
    const blocked = Math.round(randBetween(5, incidents * 0.8))
    return {
      ...c,
      threatIndex: parseFloat(randBetween(5, 92).toFixed(0)),
      incidents,
      blocked,
      primaryAttack: ATTACK_TYPES[Math.floor(Math.random() * ATTACK_TYPES.length)],
      trend: parseFloat(randBetween(-15, 30).toFixed(1)),
      blockRate: parseFloat(((blocked / Math.max(1, incidents)) * 100).toFixed(1)),
    }
  })
}

const THREAT_DESCRIPTIONS = [
  'Credential stuffing attack detected from proxy network',
  'Unusual account takeover pattern in batch transfers',
  'SIM swap fraud ring activity — multiple IMSI changes',
  'Synthetic identity cluster identified via GNN',
  'Cross-border money mule chain activated',
  'Card cloning operation at compromised POS terminal',
  'Phishing campaign targeting high-value accounts',
  'BEC fraud attempt via spoofed sender domain',
  'Velocity anomaly — 47 tx in 2 min from single device',
  'Dormant account reactivated with large RTGS transfer',
  'Multi-hop transaction layering through shell accounts',
  'Device fingerprint mismatch — potential account hijack',
] as const

const SEVERITIES: ThreatEvent['severity'][] = ['low', 'medium', 'high', 'critical']
const STATUSES: ThreatEvent['status'][] = ['detected', 'investigating', 'blocked', 'escalated']

function generateThreatEvents(): ThreatEvent[] {
  const events: ThreatEvent[] = []
  const now = Date.now()
  for (let i = 0; i < 20; i++) {
    const spot = THREAT_HOTSPOTS[Math.floor(Math.random() * THREAT_HOTSPOTS.length)]
    events.push({
      id: `evt-${i}-${now}`,
      timestamp: now - i * Math.round(randBetween(15000, 120000)),
      severity: SEVERITIES[Math.floor(Math.random() * SEVERITIES.length)],
      city: spot.city,
      country: spot.country,
      description: THREAT_DESCRIPTIONS[Math.floor(Math.random() * THREAT_DESCRIPTIONS.length)],
      attackType: ATTACK_TYPES[Math.floor(Math.random() * ATTACK_TYPES.length)],
      amount: Math.round(randBetween(5000, 5000000)),
      status: STATUSES[Math.floor(Math.random() * STATUSES.length)],
    })
  }
  return events
}

const ATTACK_COLORS: Record<string, string> = {
  'Account Takeover': '#ef4444',
  'Phishing Relay': '#06b6d4',
  'SIM Swap': '#ec4899',
  'Card Cloning': '#a855f7',
  'Money Mule Ring': '#f97316',
  'Synthetic ID': '#eab308',
  'Credential Stuffing': '#6366f1',
  'BEC Fraud': '#14b8a6',
}

function generateAttackVectors(): AttackVectorStat[] {
  const types = ATTACK_TYPES.map(t => ({
    attackType: t,
    count: Math.round(randBetween(20, 500)),
    avgRisk: parseFloat(randBetween(30, 85).toFixed(1)),
    trend: parseFloat(randBetween(-20, 35).toFixed(1)),
    color: ATTACK_COLORS[t] || '#6366f1',
  }))
  const total = types.reduce((s, t) => s + t.count, 0)
  return types.map(t => ({
    ...t,
    percentage: parseFloat(((t.count / total) * 100).toFixed(1)),
  })).sort((a, b) => b.count - a.count)
}

function generateFraudTypologies(): FraudTypology[] {
  const counts = FRAUD_TYPES.map(ft => ({
    ...ft,
    count: Math.round(randBetween(20, 500)),
    trend: parseFloat(randBetween(-15, 25).toFixed(1)),
  }))
  const total = counts.reduce((s, c) => s + c.count, 0)
  return counts.map(c => ({
    ...c,
    percentage: parseFloat(((c.count / total) * 100).toFixed(1)),
  }))
}

function generateModelPerformance(): ModelPerformance[] {
  return [
    { metric: 'Precision', xgboost: randBetween(88, 95), gnn: randBetween(85, 93), ensemble: randBetween(91, 97), llmAgent: randBetween(89, 96) },
    { metric: 'Recall', xgboost: randBetween(82, 92), gnn: randBetween(86, 94), ensemble: randBetween(89, 96), llmAgent: randBetween(87, 95) },
    { metric: 'F1 Score', xgboost: randBetween(85, 93), gnn: randBetween(85, 93), ensemble: randBetween(90, 96), llmAgent: randBetween(88, 95) },
    { metric: 'AUC-ROC', xgboost: randBetween(90, 97), gnn: randBetween(89, 96), ensemble: randBetween(93, 99), llmAgent: randBetween(91, 97) },
    { metric: 'Accuracy', xgboost: randBetween(92, 98), gnn: randBetween(91, 97), ensemble: randBetween(94, 99), llmAgent: randBetween(93, 98) },
  ].map(m => ({
    ...m,
    xgboost: parseFloat(m.xgboost.toFixed(1)),
    gnn: parseFloat(m.gnn.toFixed(1)),
    ensemble: parseFloat(m.ensemble.toFixed(1)),
    llmAgent: parseFloat(m.llmAgent.toFixed(1)),
  }))
}

function generateVelocity(): VelocityBucket[] {
  return [
    { range: '0-5 tx/min', normal: Math.round(randBetween(400, 800)), suspicious: Math.round(randBetween(5, 20)), fraudulent: Math.round(randBetween(1, 5)) },
    { range: '5-15 tx/min', normal: Math.round(randBetween(200, 500)), suspicious: Math.round(randBetween(15, 50)), fraudulent: Math.round(randBetween(5, 20)) },
    { range: '15-30 tx/min', normal: Math.round(randBetween(50, 150)), suspicious: Math.round(randBetween(30, 80)), fraudulent: Math.round(randBetween(10, 40)) },
    { range: '30-60 tx/min', normal: Math.round(randBetween(10, 40)), suspicious: Math.round(randBetween(40, 100)), fraudulent: Math.round(randBetween(20, 60)) },
    { range: '60+ tx/min', normal: Math.round(randBetween(2, 10)), suspicious: Math.round(randBetween(50, 120)), fraudulent: Math.round(randBetween(30, 80)) },
  ]
}

function generateAccountRiskBands(): AccountRiskBand[] {
  return [
    { band: 'Low (0-20)', count: Math.round(randBetween(8000, 15000)), value: 0.15, color: '#22c55e' },
    { band: 'Medium (20-40)', count: Math.round(randBetween(3000, 6000)), value: 0.30, color: '#84cc16' },
    { band: 'Elevated (40-60)', count: Math.round(randBetween(1000, 3000)), value: 0.50, color: '#eab308' },
    { band: 'High (60-80)', count: Math.round(randBetween(200, 800)), value: 0.70, color: '#f97316' },
    { band: 'Critical (80-100)', count: Math.round(randBetween(30, 200)), value: 0.90, color: '#ef4444' },
  ]
}

function generateNetworkMetrics(): NetworkMetric[] {
  return [
    { metric: 'Clustering Coeff', value: parseFloat(randBetween(0.3, 0.8).toFixed(2)) * 100, fullMark: 100 },
    { metric: 'Avg Degree', value: parseFloat(randBetween(3, 12).toFixed(1)) * 10, fullMark: 120 },
    { metric: 'Density', value: parseFloat(randBetween(0.01, 0.15).toFixed(3)) * 1000, fullMark: 150 },
    { metric: 'Modularity', value: parseFloat(randBetween(0.4, 0.9).toFixed(2)) * 100, fullMark: 100 },
    { metric: 'Betweenness', value: parseFloat(randBetween(0.1, 0.6).toFixed(2)) * 100, fullMark: 100 },
    { metric: 'PageRank Entropy', value: parseFloat(randBetween(2, 5).toFixed(1)) * 20, fullMark: 100 },
  ]
}

function generateAlertFunnel(): AlertFunnel[] {
  const ingested = Math.round(randBetween(50000, 120000))
  const flagged = Math.round(ingested * randBetween(0.03, 0.08))
  const mlFiltered = Math.round(flagged * randBetween(0.4, 0.7))
  const agentReviewed = Math.round(mlFiltered * randBetween(0.5, 0.8))
  const confirmed = Math.round(agentReviewed * randBetween(0.3, 0.6))
  const actionTaken = Math.round(confirmed * randBetween(0.7, 0.95))
  return [
    { stage: 'Events Ingested', count: ingested, percentage: 100 },
    { stage: 'ML Flagged', count: flagged, percentage: parseFloat(((flagged / ingested) * 100).toFixed(1)) },
    { stage: 'GNN Filtered', count: mlFiltered, percentage: parseFloat(((mlFiltered / ingested) * 100).toFixed(1)) },
    { stage: 'Agent Reviewed', count: agentReviewed, percentage: parseFloat(((agentReviewed / ingested) * 100).toFixed(1)) },
    { stage: 'Confirmed Fraud', count: confirmed, percentage: parseFloat(((confirmed / ingested) * 100).toFixed(1)) },
    { stage: 'Action Taken', count: actionTaken, percentage: parseFloat(((actionTaken / ingested) * 100).toFixed(1)) },
  ]
}

function generateDeviceFingerprints(): DeviceFingerprint[] {
  return [
    { type: 'Mobile (Android)', unique: Math.round(randBetween(8000, 25000)), flagged: Math.round(randBetween(100, 500)), banned: Math.round(randBetween(10, 60)) },
    { type: 'Mobile (iOS)', unique: Math.round(randBetween(5000, 15000)), flagged: Math.round(randBetween(40, 200)), banned: Math.round(randBetween(5, 30)) },
    { type: 'Desktop Browser', unique: Math.round(randBetween(3000, 10000)), flagged: Math.round(randBetween(80, 350)), banned: Math.round(randBetween(15, 50)) },
    { type: 'API / M2M', unique: Math.round(randBetween(500, 2000)), flagged: Math.round(randBetween(20, 100)), banned: Math.round(randBetween(5, 25)) },
    { type: 'POS Terminal', unique: Math.round(randBetween(1000, 5000)), flagged: Math.round(randBetween(30, 150)), banned: Math.round(randBetween(3, 20)) },
    { type: 'ATM', unique: Math.round(randBetween(500, 3000)), flagged: Math.round(randBetween(10, 80)), banned: Math.round(randBetween(2, 15)) },
  ]
}

function generateAmountDistribution(): AmountDistribution[] {
  return [
    { range: '< ₹500', count: Math.round(randBetween(5000, 15000)), fraudCount: Math.round(randBetween(10, 40)), avgRisk: parseFloat(randBetween(5, 15).toFixed(1)) },
    { range: '₹500-2K', count: Math.round(randBetween(8000, 20000)), fraudCount: Math.round(randBetween(30, 80)), avgRisk: parseFloat(randBetween(8, 20).toFixed(1)) },
    { range: '₹2K-10K', count: Math.round(randBetween(6000, 15000)), fraudCount: Math.round(randBetween(50, 150)), avgRisk: parseFloat(randBetween(12, 30).toFixed(1)) },
    { range: '₹10K-50K', count: Math.round(randBetween(3000, 8000)), fraudCount: Math.round(randBetween(80, 250)), avgRisk: parseFloat(randBetween(20, 45).toFixed(1)) },
    { range: '₹50K-1L', count: Math.round(randBetween(1000, 4000)), fraudCount: Math.round(randBetween(60, 200)), avgRisk: parseFloat(randBetween(30, 55).toFixed(1)) },
    { range: '₹1L-5L', count: Math.round(randBetween(500, 2000)), fraudCount: Math.round(randBetween(40, 150)), avgRisk: parseFloat(randBetween(40, 65).toFixed(1)) },
    { range: '₹5L-10L', count: Math.round(randBetween(100, 600)), fraudCount: Math.round(randBetween(20, 80)), avgRisk: parseFloat(randBetween(50, 75).toFixed(1)) },
    { range: '> ₹10L', count: Math.round(randBetween(20, 150)), fraudCount: Math.round(randBetween(10, 50)), avgRisk: parseFloat(randBetween(60, 90).toFixed(1)) },
  ]
}

export const useAnalyticsStore = create<AnalyticsState>((set) => ({
  transactionVolume: [],
  fraudRate: [],
  channelVolume: [],
  latencyMetrics: [],
  riskHeatmap: generateHeatmap(),
  geoRegions: generateGeoRegions(),
  fraudTypologies: generateFraudTypologies(),
  modelPerformance: generateModelPerformance(),
  velocityDistribution: generateVelocity(),
  accountRiskBands: generateAccountRiskBands(),
  networkMetrics: generateNetworkMetrics(),
  alertFunnel: generateAlertFunnel(),
  deviceFingerprints: generateDeviceFingerprints(),
  amountDistribution: generateAmountDistribution(),
  threatHotspots: generateThreatHotspots(),
  crossBorderFlows: generateCrossBorderFlows(),
  countryThreats: generateCountryThreats(),
  threatEvents: generateThreatEvents(),
  attackVectors: generateAttackVectors(),

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

  tick: () =>
    set((state) => {
      const t = timeLabel()
      const ts = Date.now()

      const legit = Math.round(randBetween(800, 2200))
      const susp = Math.round(randBetween(30, 120))
      const fraud = Math.round(randBetween(5, 40))
      const blocked = Math.round(randBetween(3, 25))

      const newTxn: TransactionVolumePoint = { time: t, timestamp: ts, legitimate: legit, suspicious: susp, fraudulent: fraud, blocked }
      const newFR: FraudRatePoint = { time: t, timestamp: ts, rate: parseFloat((fraud / (legit + susp + fraud) * 100).toFixed(2)), baseline: 1.2, threshold: 3.5 }
      const newCh: ChannelVolumePoint = {
        time: t, timestamp: ts,
        UPI: Math.round(randBetween(400, 1200)),
        NEFT: Math.round(randBetween(100, 400)),
        RTGS: Math.round(randBetween(50, 200)),
        IMPS: Math.round(randBetween(200, 600)),
        Card: Math.round(randBetween(150, 500)),
        Wallet: Math.round(randBetween(80, 300)),
      }
      const newLat: LatencyPoint = {
        time: t, timestamp: ts,
        p50: parseFloat(randBetween(8, 25).toFixed(1)),
        p95: parseFloat(randBetween(40, 120).toFixed(1)),
        p99: parseFloat(randBetween(100, 350).toFixed(1)),
        mlInference: parseFloat(randBetween(15, 60).toFixed(1)),
      }

      const totalP = state.totalProcessed + legit + susp + fraud
      const totalF = state.totalFlagged + susp + fraud
      const totalB = state.totalBlocked + blocked

      return {
        transactionVolume: [...state.transactionVolume, newTxn].slice(-MAX_SERIES),
        fraudRate: [...state.fraudRate, newFR].slice(-MAX_SERIES),
        channelVolume: [...state.channelVolume, newCh].slice(-MAX_SERIES),
        latencyMetrics: [...state.latencyMetrics, newLat].slice(-MAX_SERIES),

        totalProcessed: totalP,
        totalFlagged: totalF,
        totalBlocked: totalB,
        avgResponseMs: parseFloat(randBetween(12, 45).toFixed(1)),
        modelAccuracy: parseFloat(randBetween(94, 99.2).toFixed(1)),
        falsePositiveRate: parseFloat(randBetween(0.8, 3.2).toFixed(2)),
        truePositiveRate: parseFloat(randBetween(92, 98.5).toFixed(1)),
        activeMules: Math.round(randBetween(3, 25)),
        riskScore: parseFloat(randBetween(20, 75).toFixed(1)),
        throughputTps: Math.round(randBetween(800, 2500)),

        // Slowly refresh aggregated every tick (small perturbations)
        geoRegions: state.geoRegions.map(r => ({
          ...r,
          transactions: r.transactions + Math.round(randBetween(50, 500)),
          fraudRate: parseFloat(Math.max(0.1, r.fraudRate + randBetween(-0.2, 0.2)).toFixed(2)),
        })),
        fraudTypologies: generateFraudTypologies(),
        modelPerformance: generateModelPerformance(),
        velocityDistribution: generateVelocity(),
        alertFunnel: generateAlertFunnel(),
        networkMetrics: generateNetworkMetrics(),
        threatHotspots: state.threatHotspots.map(h => ({
          ...h,
          threatLevel: parseFloat(Math.max(0, Math.min(100, h.threatLevel + randBetween(-8, 8))).toFixed(0)),
          activeAlerts: Math.max(0, h.activeAlerts + Math.round(randBetween(-3, 5))),
          intensity: parseFloat(Math.max(0.3, Math.min(3, h.intensity + randBetween(-0.3, 0.3))).toFixed(1)),
        })),
        crossBorderFlows: state.crossBorderFlows.map(f => ({
          ...f,
          amount: Math.max(10000, f.amount + Math.round(randBetween(-50000, 80000))),
          riskScore: parseFloat(Math.max(10, Math.min(100, f.riskScore + randBetween(-5, 5))).toFixed(1)),
          txCount: Math.max(1, f.txCount + Math.round(randBetween(-2, 5))),
          lastDetected: Math.random() > 0.7 ? Date.now() : f.lastDetected,
          trend: parseFloat(Math.max(-30, Math.min(50, f.trend + randBetween(-3, 3))).toFixed(1)),
        })),
        countryThreats: state.countryThreats.map(c => {
          const newInc = Math.max(0, c.incidents + Math.round(randBetween(-10, 25)))
          const newBlk = Math.max(0, c.blocked + Math.round(randBetween(-5, 15)))
          return {
            ...c,
            threatIndex: parseFloat(Math.max(0, Math.min(100, c.threatIndex + randBetween(-4, 4))).toFixed(0)),
            incidents: newInc,
            blocked: newBlk,
            blockRate: parseFloat(((newBlk / Math.max(1, newInc)) * 100).toFixed(1)),
            trend: parseFloat(Math.max(-30, Math.min(50, c.trend + randBetween(-2, 2))).toFixed(1)),
          }
        }),
        threatEvents: [
          {
            id: `evt-${Date.now()}`,
            timestamp: Date.now(),
            severity: SEVERITIES[Math.floor(Math.random() * SEVERITIES.length)],
            city: THREAT_HOTSPOTS[Math.floor(Math.random() * THREAT_HOTSPOTS.length)].city,
            country: THREAT_HOTSPOTS[Math.floor(Math.random() * THREAT_HOTSPOTS.length)].country,
            description: THREAT_DESCRIPTIONS[Math.floor(Math.random() * THREAT_DESCRIPTIONS.length)],
            attackType: ATTACK_TYPES[Math.floor(Math.random() * ATTACK_TYPES.length)],
            amount: Math.round(randBetween(5000, 5000000)),
            status: STATUSES[Math.floor(Math.random() * STATUSES.length)],
          },
          ...state.threatEvents.slice(0, 24),
        ],
        attackVectors: state.attackVectors.map(v => ({
          ...v,
          count: Math.max(1, v.count + Math.round(randBetween(-5, 15))),
          avgRisk: parseFloat(Math.max(10, Math.min(95, v.avgRisk + randBetween(-3, 3))).toFixed(1)),
          trend: parseFloat(Math.max(-30, Math.min(50, v.trend + randBetween(-3, 3))).toFixed(1)),
        })),
      }
    }),
}))

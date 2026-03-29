// ============================================================================
// Threat Intelligence Visualizations
// World map, India regional map, attack vectors, live feeds, corridor analysis
// ============================================================================

import { useMemo, useState, useCallback } from 'react'
import {
  ComposableMap,
  Geographies,
  Geography,
  Marker,
  Line,
  ZoomableGroup,
} from 'react-simple-maps'
import type {
  ThreatHotspot,
  CrossBorderFlow,
  GeoRegion,
  CountryThreat,
  ThreatEvent,
  AttackVectorStat,
} from '../stores/use-analytics-store'

const WORLD_GEO_URL = 'https://cdn.jsdelivr.net/npm/world-atlas@2/countries-110m.json'

// ISO 3166-1 numeric → alpha-3 mapping for world-atlas@2 geo.id
const ISO_NUM_TO_A3: Record<string, string> = {
  '004':'AFG','008':'ALB','012':'DZA','024':'AGO','032':'ARG','036':'AUS',
  '040':'AUT','050':'BGD','056':'BEL','064':'BTN','068':'BOL','076':'BRA',
  '100':'BGR','104':'MMR','116':'KHM','120':'CMR','124':'CAN','144':'LKA',
  '152':'CHL','156':'CHN','170':'COL','180':'COD','188':'CRI','191':'HRV',
  '192':'CUB','196':'CYP','203':'CZE','208':'DNK','218':'ECU','222':'SLV',
  '231':'ETH','233':'EST','246':'FIN','250':'FRA','268':'GEO','276':'DEU',
  '288':'GHA','300':'GRC','320':'GTM','332':'HTI','340':'HND','348':'HUN',
  '352':'ISL','356':'IND','360':'IDN','364':'IRN','368':'IRQ','372':'IRL',
  '376':'ISR','380':'ITA','384':'CIV','392':'JPN','398':'KAZ','400':'JOR',
  '404':'KEN','408':'PRK','410':'KOR','414':'KWT','418':'LAO','422':'LBN',
  '428':'LVA','434':'LBY','440':'LTU','442':'LUX','450':'MDG','454':'MWI',
  '458':'MYS','466':'MLI','478':'MRT','484':'MEX','496':'MNG','504':'MAR',
  '508':'MOZ','512':'OMN','516':'NAM','524':'NPL','528':'NLD','554':'NZL',
  '558':'NIC','562':'NER','566':'NGA','578':'NOR','586':'PAK','591':'PAN',
  '598':'PNG','600':'PRY','604':'PER','608':'PHL','616':'POL','620':'PRT',
  '634':'QAT','642':'ROU','643':'RUS','646':'RWA','682':'SAU','686':'SEN',
  '694':'SLE','702':'SGP','703':'SVK','704':'VNM','705':'SVN','706':'SOM',
  '710':'ZAF','716':'ZWE','724':'ESP','728':'SSD','729':'SDN','740':'SUR',
  '748':'SWZ','752':'SWE','756':'CHE','760':'SYR','762':'TJK','764':'THA',
  '768':'TGO','780':'TTO','784':'ARE','788':'TUN','792':'TUR','795':'TKM',
  '800':'UGA','804':'UKR','818':'EGY','826':'GBR','834':'TZA','840':'USA',
  '854':'BFA','858':'URY','860':'UZB','862':'VEN','887':'YEM','894':'ZMB',
}

// Country centroids for label placement (lat, lng)
const COUNTRY_CENTERS: Record<string, [number, number]> = {
  IND: [78.9, 22.0], NGA: [8.0, 9.5], RUS: [100, 60], CHN: [104, 35],
  ROU: [25, 46], BRA: [-51, -10], IDN: [118, -3], PAK: [69, 30],
  ARE: [54, 24], GBR: [-3, 54], USA: [-98, 39], UKR: [32, 49],
  VNM: [108, 14], PHL: [122, 13], MYS: [110, 4], THA: [101, 15],
  KOR: [128, 36], JPN: [138, 36], DEU: [10, 51], FRA: [2, 47],
}

// --- Color scales ---

function threatColor(level: number, opacity = 1): string {
  if (level < 20) return `rgba(34, 197, 94, ${opacity})`
  if (level < 40) return `rgba(132, 204, 22, ${opacity})`
  if (level < 55) return `rgba(234, 179, 8, ${opacity})`
  if (level < 70) return `rgba(249, 115, 22, ${opacity})`
  if (level < 85) return `rgba(239, 68, 68, ${opacity})`
  return `rgba(220, 38, 38, ${opacity})`
}

function threatLabel(level: number): string {
  if (level < 20) return 'LOW'
  if (level < 40) return 'GUARDED'
  if (level < 55) return 'ELEVATED'
  if (level < 70) return 'HIGH'
  if (level < 85) return 'SEVERE'
  return 'CRITICAL'
}

function riskColor(score: number): string {
  if (score < 25) return '#22c55e'
  if (score < 40) return '#84cc16'
  if (score < 55) return '#eab308'
  if (score < 70) return '#f97316'
  return '#ef4444'
}

function severityColor(sev: string): string {
  switch (sev) {
    case 'critical': return '#dc2626'
    case 'high': return '#ef4444'
    case 'medium': return '#f97316'
    case 'low': return '#eab308'
    default: return '#64748b'
  }
}

function timeAgo(ts: number): string {
  const s = Math.floor((Date.now() - ts) / 1000)
  if (s < 60) return `${s}s ago`
  const m = Math.floor(s / 60)
  if (m < 60) return `${m}m ago`
  return `${Math.floor(m / 60)}h ago`
}

function formatCurrency(val: number): string {
  if (val >= 10000000) return `₹${(val / 10000000).toFixed(1)}Cr`
  if (val >= 100000) return `₹${(val / 100000).toFixed(1)}L`
  if (val >= 1000) return `₹${(val / 1000).toFixed(1)}K`
  return `₹${val}`
}


// ============================================================================
// 1. WORLD THREAT MAP — comprehensive labelled geographic intelligence
// ============================================================================

interface WorldThreatMapProps {
  hotspots: ThreatHotspot[]
  flows: CrossBorderFlow[]
  countryThreats: CountryThreat[]
}

export function WorldThreatMap({ hotspots, flows, countryThreats }: WorldThreatMapProps) {
  const [tooltip, setTooltip] = useState<{ text: string; x: number; y: number } | null>(null)

  const threatLookup = useMemo(() => {
    const m = new Map<string, CountryThreat>()
    for (const ct of countryThreats) m.set(ct.iso, ct)
    return m
  }, [countryThreats])

  const getCountryFill = useCallback((geoId: string) => {
    const iso3 = ISO_NUM_TO_A3[geoId]
    if (!iso3) return 'rgba(30, 41, 59, 0.6)'
    const ct = threatLookup.get(iso3)
    if (!ct) return 'rgba(30, 41, 59, 0.6)'
    return threatColor(ct.threatIndex, 0.5)
  }, [threatLookup])

  // Computed aggregates for enhanced HUD
  const totalAlerts = useMemo(() => hotspots.reduce((s, h) => s + h.activeAlerts, 0), [hotspots])
  const criticalCount = useMemo(() => hotspots.filter(h => h.threatLevel > 75).length, [hotspots])
  const activeFlows = useMemo(() => flows.filter(f => f.riskScore > 50).length, [flows])
  const totalVolumeAtRisk = useMemo(() => flows.reduce((s, f) => s + f.amount, 0), [flows])
  const avgBlockRate = useMemo(() => {
    const rates = countryThreats.map(c => c.blockRate)
    return rates.length ? (rates.reduce((s, r) => s + r, 0) / rates.length).toFixed(1) : '0'
  }, [countryThreats])
  const totalIncidents = useMemo(() => countryThreats.reduce((s, c) => s + c.incidents, 0), [countryThreats])
  const peakThreat = useMemo(() => {
    const top = [...countryThreats].sort((a, b) => b.threatIndex - a.threatIndex)[0]
    return top ? { name: top.name, level: top.threatIndex } : { name: '-', level: 0 }
  }, [countryThreats])

  // Labelled high-threat countries (top 8 by threat index)
  const labelledCountries = useMemo(() =>
    [...countryThreats]
      .sort((a, b) => b.threatIndex - a.threatIndex)
      .slice(0, 8)
      .filter(c => COUNTRY_CENTERS[c.iso]),
  [countryThreats])

  return (
    <div className="relative w-full" style={{ aspectRatio: '2 / 1', minHeight: 360 }}>
      {/* Enhanced HUD — two rows of metrics */}
      <div className="absolute top-2 left-2 z-10 space-y-1.5">
        <div className="flex gap-1.5">
          {[
            { label: 'MONITORED COUNTRIES', value: countryThreats.length, color: 'text-slate-200' },
            { label: 'ACTIVE HOTSPOTS', value: hotspots.length, color: 'text-cyan-400' },
            { label: 'TOTAL INCIDENTS', value: totalIncidents.toLocaleString(), color: 'text-amber-400' },
            { label: 'CRITICAL ZONES', value: criticalCount, color: 'text-red-400' },
          ].map(s => (
            <div key={s.label} className="px-2.5 py-1.5 rounded-md bg-slate-900/90 border border-slate-700/40 backdrop-blur-sm">
              <div className="text-[7px] text-slate-500 font-mono tracking-wider">{s.label}</div>
              <div className={`text-sm font-bold font-mono ${s.color}`}>{s.value}</div>
            </div>
          ))}
        </div>
        <div className="flex gap-1.5">
          {[
            { label: 'VOLUME AT RISK', value: formatCurrency(totalVolumeAtRisk), color: 'text-rose-400' },
            { label: 'AVG BLOCK RATE', value: `${avgBlockRate}%`, color: parseFloat(avgBlockRate) > 50 ? 'text-green-400' : 'text-amber-400' },
            { label: 'ACTIVE FLOWS', value: `${activeFlows} / ${flows.length}`, color: 'text-indigo-400' },
            { label: 'PEAK THREAT', value: `${peakThreat.name} (${peakThreat.level})`, color: 'text-red-300' },
          ].map(s => (
            <div key={s.label} className="px-2.5 py-1.5 rounded-md bg-slate-900/90 border border-slate-700/40 backdrop-blur-sm">
              <div className="text-[7px] text-slate-500 font-mono tracking-wider">{s.label}</div>
              <div className={`text-[12px] font-bold font-mono ${s.color}`}>{s.value}</div>
            </div>
          ))}
        </div>
      </div>

      {/* Timestamp */}
      <div className="absolute top-2 right-2 z-10 flex items-center gap-1.5 px-2.5 py-1.5 rounded-md bg-slate-900/90 border border-slate-700/40 backdrop-blur-sm">
        <div className="w-1.5 h-1.5 rounded-full bg-green-500" />
        <span className="text-[8px] text-green-400 font-mono font-semibold">LIVE</span>
        <span className="text-[8px] text-slate-500 font-mono">
          {new Date().toLocaleTimeString('en-IN', { hour: '2-digit', minute: '2-digit', second: '2-digit', hour12: false })} IST
        </span>
      </div>

      <ComposableMap
        projection="geoMercator"
        projectionConfig={{ scale: 130, center: [40, 15] }}
        className="w-full h-full"
        style={{ backgroundColor: 'transparent' }}
      >
        <ZoomableGroup>
          {/* Country fills */}
          <Geographies geography={WORLD_GEO_URL}>
            {({ geographies }) =>
              geographies.map((geo) => (
                <Geography
                  key={geo.rsmKey}
                  geography={geo}
                  fill={getCountryFill(geo.id)}
                  stroke="rgba(100, 116, 139, 0.25)"
                  strokeWidth={0.4}
                  style={{
                    default: { outline: 'none' },
                    hover: { fill: 'rgba(99, 102, 241, 0.45)', outline: 'none', cursor: 'pointer' },
                    pressed: { outline: 'none' },
                  }}
                  onMouseEnter={(e) => {
                    const iso3 = ISO_NUM_TO_A3[geo.id]
                    const ct = iso3 ? threatLookup.get(iso3) : undefined
                    if (ct) {
                      const rect = (e.target as SVGElement).closest('svg')?.getBoundingClientRect()
                      setTooltip({
                        text: `${ct.name} — Threat: ${ct.threatIndex}/100 (${threatLabel(ct.threatIndex)}) | ${ct.incidents.toLocaleString()} incidents | ${ct.blocked.toLocaleString()} blocked (${ct.blockRate}%) | ${ct.primaryAttack} | Trend: ${ct.trend > 0 ? '↑' : '↓'}${Math.abs(ct.trend)}%`,
                        x: rect ? e.clientX - rect.left : 200,
                        y: rect ? e.clientY - rect.top : 20,
                      })
                    }
                  }}
                  onMouseLeave={() => setTooltip(null)}
                />
              ))
            }
          </Geographies>

          {/* Country name labels for high-threat nations */}
          {labelledCountries.map(c => {
            const center = COUNTRY_CENTERS[c.iso]
            if (!center) return null
            return (
              <Marker key={`label-${c.iso}`} coordinates={center}>
                <text
                  textAnchor="middle"
                  y={-2}
                  style={{
                    fontSize: 6.5,
                    fill: threatColor(c.threatIndex, 0.85),
                    fontFamily: 'monospace',
                    fontWeight: 700,
                    letterSpacing: '0.5px',
                    textShadow: '0 0 4px rgba(0,0,0,0.8)',
                    pointerEvents: 'none',
                  }}
                >
                  {c.name.toUpperCase()}
                </text>
                <text
                  textAnchor="middle"
                  y={6}
                  style={{
                    fontSize: 5,
                    fill: 'rgba(148, 163, 184, 0.7)',
                    fontFamily: 'monospace',
                    fontWeight: 500,
                    pointerEvents: 'none',
                  }}
                >
                  TI:{c.threatIndex} | {c.incidents} events
                </text>
              </Marker>
            )
          })}

          {/* Cross-border flow lines — static gradient, no animation */}
          {flows.map((flow, i) => (
            <Line
              key={`flow-${i}`}
              from={[flow.from.lng, flow.from.lat]}
              to={[flow.to.lng, flow.to.lat]}
              stroke={threatColor(flow.riskScore, 0.4)}
              strokeWidth={Math.max(0.8, flow.riskScore / 30)}
              strokeLinecap="round"
              strokeDasharray={flow.riskScore > 70 ? '0' : '6 3'}
              style={{ pointerEvents: 'none' }}
            />
          ))}

          {/* Flow corridor labels at midpoint */}
          {flows.filter(f => f.riskScore > 50).map((flow, i) => {
            const midLng = (flow.from.lng + flow.to.lng) / 2
            const midLat = (flow.from.lat + flow.to.lat) / 2 - 1.5
            return (
              <Marker key={`flabel-${i}`} coordinates={[midLng, midLat]}>
                <text
                  textAnchor="middle"
                  style={{
                    fontSize: 4.5,
                    fill: 'rgba(226,232,240,0.55)',
                    fontFamily: 'monospace',
                    fontWeight: 500,
                    pointerEvents: 'none',
                  }}
                >
                  {flow.channel} · {formatCurrency(flow.amount)}
                </text>
              </Marker>
            )
          })}

          {/* Threat hotspot markers — solid markers, NO animate-ping */}
          {hotspots.map((spot) => {
            const mainRadius = Math.max(3, 2 + spot.threatLevel / 25)
            return (
              <Marker key={spot.id} coordinates={[spot.lng, spot.lat]}>
                {/* Glow ring — static, no animation */}
                <circle
                  r={mainRadius + 3}
                  fill="none"
                  stroke={threatColor(spot.threatLevel, 0.25)}
                  strokeWidth={1}
                />
                {/* Threat area fill */}
                <circle r={mainRadius + 1} fill={threatColor(spot.threatLevel, 0.12)} />
                {/* Core dot */}
                <circle
                  r={mainRadius}
                  fill={threatColor(spot.threatLevel, 0.85)}
                  stroke="rgba(255,255,255,0.3)"
                  strokeWidth={0.5}
                  className="cursor-pointer"
                  onMouseEnter={(e) => {
                    const rect = (e.target as SVGElement).closest('svg')?.getBoundingClientRect()
                    setTooltip({
                      text: `${spot.city}, ${spot.country} | Threat: ${spot.threatLevel}/100 (${threatLabel(spot.threatLevel)}) | ${spot.activeAlerts} active alerts | ${spot.attackType} | Intensity: ${spot.intensity.toFixed(1)}`,
                      x: rect ? e.clientX - rect.left : 200,
                      y: rect ? e.clientY - rect.top : 20,
                    })
                  }}
                  onMouseLeave={() => setTooltip(null)}
                />
                {/* ALWAYS show city label */}
                <text
                  textAnchor="start"
                  x={mainRadius + 3}
                  y={1}
                  style={{
                    fontSize: 7,
                    fill: 'rgba(226,232,240,0.9)',
                    fontFamily: 'monospace',
                    fontWeight: 600,
                    textShadow: '0 0 6px rgba(0,0,0,0.9), 0 0 2px rgba(0,0,0,0.9)',
                    pointerEvents: 'none',
                  }}
                >
                  {spot.city}
                </text>
                {/* Alert count badge next to city name */}
                {spot.activeAlerts > 0 && (
                  <text
                    textAnchor="start"
                    x={mainRadius + 3}
                    y={9}
                    style={{
                      fontSize: 5,
                      fill: spot.threatLevel > 70 ? 'rgba(239,68,68,0.8)' : 'rgba(148,163,184,0.6)',
                      fontFamily: 'monospace',
                      fontWeight: 500,
                      pointerEvents: 'none',
                    }}
                  >
                    {spot.activeAlerts} alerts · {spot.attackType}
                  </text>
                )}
              </Marker>
            )
          })}

          {/* Flow origin/destination markers with labels */}
          {flows.map((flow, i) => (
            <Marker key={`fend-${i}`} coordinates={[flow.to.lng, flow.to.lat]}>
              <circle r={2.5} fill={threatColor(flow.riskScore, 0.8)} stroke="rgba(255,255,255,0.15)" strokeWidth={0.5} />
            </Marker>
          ))}
        </ZoomableGroup>
      </ComposableMap>

      {/* Following tooltip */}
      {tooltip && (
        <div
          className="absolute px-3 py-2 rounded-lg bg-slate-900/95 border border-slate-700/60 backdrop-blur-sm text-[10px] text-slate-300 font-mono max-w-[380px] pointer-events-none z-20 shadow-xl"
          style={{ left: Math.min(tooltip.x + 12, 600), top: Math.max(tooltip.y - 10, 4) }}
        >
          {tooltip.text}
        </div>
      )}

      {/* Enhanced legend */}
      <div className="absolute bottom-2 left-2 z-10 flex flex-col gap-1.5">
        {/* Threat level scale */}
        <div className="flex items-center gap-1 px-2.5 py-1.5 rounded-md bg-slate-900/90 border border-slate-700/40 backdrop-blur-sm">
          <span className="text-[7px] text-slate-500 font-mono mr-1">THREAT LEVEL</span>
          {[
            { level: 10, label: 'LOW' },
            { level: 35, label: 'GUARD' },
            { level: 55, label: 'ELEV' },
            { level: 72, label: 'HIGH' },
            { level: 90, label: 'CRIT' },
          ].map(l => (
            <div key={l.level} className="flex flex-col items-center">
              <div className="w-5 h-2.5 rounded-[2px]" style={{ backgroundColor: threatColor(l.level, 0.8) }} />
              <span className="text-[5px] text-slate-600 font-mono mt-0.5">{l.label}</span>
            </div>
          ))}
        </div>
        {/* Symbol key */}
        <div className="flex items-center gap-3 px-2.5 py-1 rounded-md bg-slate-900/90 border border-slate-700/40 backdrop-blur-sm">
          <div className="flex items-center gap-1">
            <div className="w-2 h-2 rounded-full bg-red-500" />
            <span className="text-[6px] text-slate-500 font-mono">Hotspot</span>
          </div>
          <div className="flex items-center gap-1">
            <div className="w-4 h-[1.5px] bg-amber-500 rounded" />
            <span className="text-[6px] text-slate-500 font-mono">Flow (solid=high risk)</span>
          </div>
          <div className="flex items-center gap-1">
            <div className="w-4 h-[1.5px] bg-cyan-500 rounded" style={{ borderBottom: '1.5px dashed #06b6d4', height: 0 }} />
            <span className="text-[6px] text-slate-500 font-mono">Flow (dashed=moderate)</span>
          </div>
        </div>
      </div>

      {/* Summary bar at bottom-right */}
      <div className="absolute bottom-2 right-2 z-10 px-2.5 py-1.5 rounded-md bg-slate-900/90 border border-slate-700/40 backdrop-blur-sm">
        <div className="text-[6px] text-slate-500 font-mono tracking-wider mb-0.5">GLOBAL THREAT ASSESSMENT</div>
        <div className="flex items-center gap-1">
          <div className="w-3 h-3 rounded-sm" style={{ backgroundColor: threatColor(peakThreat.level, 0.8) }} />
          <span className="text-[10px] font-mono font-bold" style={{ color: threatColor(peakThreat.level) }}>
            {threatLabel(peakThreat.level)}
          </span>
          <span className="text-[8px] text-slate-500 font-mono ml-1">
            {totalAlerts.toLocaleString()} total alerts
          </span>
        </div>
      </div>
    </div>
  )
}


// ============================================================================
// 2. INDIA REGIONAL MAP — proper geography with clear labelling
// ============================================================================

interface IndiaRegionalMapProps {
  regions: GeoRegion[]
}

export function IndiaRegionalMap({ regions }: IndiaRegionalMapProps) {
  const [hovered, setHovered] = useState<GeoRegion | null>(null)

  const sorted = useMemo(() =>
    [...regions].sort((a, b) => a.riskScore - b.riskScore), [regions])

  const avgRisk = useMemo(() =>
    (regions.reduce((s, r) => s + r.riskScore, 0) / regions.length).toFixed(1), [regions])

  const highRiskCount = useMemo(() => regions.filter(r => r.riskScore > 60).length, [regions])

  const totalVolume = useMemo(() => regions.reduce((s, r) => s + r.amount, 0), [regions])

  const topState = useMemo(() => {
    const s = [...regions].sort((a, b) => b.riskScore - a.riskScore)[0]
    return s ? s.region : '-'
  }, [regions])

  return (
    <div className="relative w-full" style={{ minHeight: 370 }}>
      {/* Enhanced summary badges */}
      <div className="absolute top-1 left-1 z-10 flex flex-wrap gap-1.5">
        <div className="px-2 py-1 rounded-md bg-slate-900/90 border border-slate-700/40 backdrop-blur-sm">
          <div className="text-[6px] text-slate-500 font-mono">STATES MONITORED</div>
          <div className="text-[12px] font-bold text-white font-mono">{regions.length}</div>
        </div>
        <div className="px-2 py-1 rounded-md bg-slate-900/90 border border-slate-700/40 backdrop-blur-sm">
          <div className="text-[6px] text-slate-500 font-mono">AVG RISK INDEX</div>
          <div className="text-[12px] font-bold font-mono" style={{ color: riskColor(parseFloat(avgRisk)) }}>{avgRisk}/100</div>
        </div>
        <div className="px-2 py-1 rounded-md bg-slate-900/90 border border-red-900/30 backdrop-blur-sm">
          <div className="text-[6px] text-slate-500 font-mono">HIGH RISK STATES</div>
          <div className="text-[12px] font-bold text-red-400 font-mono">{highRiskCount}</div>
        </div>
        <div className="px-2 py-1 rounded-md bg-slate-900/90 border border-slate-700/40 backdrop-blur-sm">
          <div className="text-[6px] text-slate-500 font-mono">TOTAL VOLUME</div>
          <div className="text-[12px] font-bold text-indigo-400 font-mono">{formatCurrency(totalVolume)}</div>
        </div>
      </div>

      <ComposableMap
        projection="geoMercator"
        projectionConfig={{ scale: 850, center: [82, 23] }}
        className="w-full h-full"
        style={{ backgroundColor: 'transparent' }}
        width={400}
        height={370}
      >
        {/* Geography — India highlighted */}
        <Geographies geography={WORLD_GEO_URL}>
          {({ geographies }) =>
            geographies.map((geo) => {
              const isIndia = geo.id === '356'
              return (
                <Geography
                  key={geo.rsmKey}
                  geography={geo}
                  fill={isIndia ? 'rgba(99, 102, 241, 0.1)' : 'rgba(30, 41, 59, 0.3)'}
                  stroke={isIndia ? 'rgba(99, 102, 241, 0.45)' : 'rgba(100, 116, 139, 0.1)'}
                  strokeWidth={isIndia ? 1 : 0.3}
                  style={{
                    default: { outline: 'none' },
                    hover: { outline: 'none' },
                    pressed: { outline: 'none' },
                  }}
                />
              )
            })
          }
        </Geographies>

        {/* Inter-state network connections */}
        {sorted.map((r, i) =>
          sorted.slice(i + 1, i + 2).map((r2, j) => (
            <Line
              key={`c-${i}-${j}`}
              from={[r.lng, r.lat]}
              to={[r2.lng, r2.lat]}
              stroke="rgba(99, 102, 241, 0.07)"
              strokeWidth={0.5}
              strokeDasharray="4 4"
            />
          ))
        )}

        {/* State markers — NO animate-ping, solid markers with labels */}
        {sorted.map((r) => {
          const color = riskColor(r.riskScore)
          const rad = 3 + (r.riskScore / 100) * 7
          const isH = hovered?.region === r.region

          return (
            <Marker key={r.region} coordinates={[r.lng, r.lat]}>
              {/* Static glow ring for high-risk — NO animation */}
              {r.riskScore > 60 && (
                <circle
                  r={rad + 3}
                  fill="none"
                  stroke={color}
                  strokeWidth={1}
                  opacity={0.25}
                />
              )}
              {/* Halo */}
              <circle r={rad} fill={color} opacity={isH ? 0.3 : 0.12} />
              {/* Core */}
              <circle
                r={Math.max(2.5, rad * 0.5)}
                fill={color}
                opacity={isH ? 1 : 0.85}
                stroke="rgba(255,255,255,0.25)"
                strokeWidth={isH ? 1.5 : 0.5}
                className="cursor-pointer"
                onMouseEnter={() => setHovered(r)}
                onMouseLeave={() => setHovered(null)}
              />
              {/* Always show state name label */}
              <text
                textAnchor="start"
                x={Math.max(2.5, rad * 0.5) + 3}
                y={1}
                style={{
                  fontSize: isH ? 7 : 5.5,
                  fill: isH ? '#fff' : 'rgba(226,232,240,0.85)',
                  fontFamily: 'monospace',
                  fontWeight: isH ? 700 : 500,
                  textShadow: '0 0 4px rgba(0,0,0,0.9)',
                  pointerEvents: 'none',
                }}
              >
                {r.region}
              </text>
              {/* Risk score label under name */}
              <text
                textAnchor="start"
                x={Math.max(2.5, rad * 0.5) + 3}
                y={8}
                style={{
                  fontSize: 4.5,
                  fill: color,
                  fontFamily: 'monospace',
                  fontWeight: 600,
                  pointerEvents: 'none',
                }}
              >
                Risk: {r.riskScore}
              </text>
            </Marker>
          )
        })}
      </ComposableMap>

      {/* Hover detail panel */}
      {hovered && (
        <div className="absolute bottom-2 right-2 px-3 py-2.5 rounded-lg bg-slate-900/95 border border-slate-700/60 backdrop-blur-sm z-10 shadow-xl min-w-[170px]">
          <div className="text-[11px] font-bold text-white mb-1.5">{hovered.region}</div>
          <div className="space-y-1 text-[9px]">
            <div className="flex justify-between gap-4">
              <span className="text-slate-500">Transactions</span>
              <span className="text-slate-300 font-mono">{hovered.transactions.toLocaleString()}</span>
            </div>
            <div className="flex justify-between gap-4">
              <span className="text-slate-500">Fraud Rate</span>
              <span className="font-mono" style={{ color: riskColor(hovered.fraudRate * 15) }}>{hovered.fraudRate}%</span>
            </div>
            <div className="flex justify-between gap-4">
              <span className="text-slate-500">Volume</span>
              <span className="text-slate-300 font-mono">{formatCurrency(hovered.amount)}</span>
            </div>
            <div className="flex justify-between gap-4">
              <span className="text-slate-500">Risk Score</span>
              <span className="font-mono font-bold" style={{ color: riskColor(hovered.riskScore) }}>{hovered.riskScore}/100</span>
            </div>
          </div>
          <div className="mt-1.5 h-1.5 w-full bg-slate-800 rounded-full overflow-hidden">
            <div
              className="h-full rounded-full transition-all duration-500"
              style={{ width: `${hovered.riskScore}%`, backgroundColor: riskColor(hovered.riskScore) }}
            />
          </div>
        </div>
      )}

      {/* Highest risk state badge */}
      <div className="absolute bottom-2 left-1 z-10 px-2 py-1 rounded-md bg-slate-900/90 border border-red-900/30 backdrop-blur-sm">
        <div className="text-[6px] text-slate-500 font-mono">HIGHEST RISK</div>
        <div className="text-[9px] font-bold text-red-400 font-mono">{topState}</div>
      </div>
    </div>
  )
}


// ============================================================================
// 3. COUNTRY THREAT PANEL — enriched intelligence view with rank numbers
// ============================================================================

interface CountryThreatPanelProps {
  countries: CountryThreat[]
}

export function CountryThreatPanel({ countries }: CountryThreatPanelProps) {
  const sorted = useMemo(() =>
    [...countries].sort((a, b) => b.threatIndex - a.threatIndex).slice(0, 10), [countries])
  const maxThreat = Math.max(...sorted.map(c => c.threatIndex), 1)
  const totalIncidents = sorted.reduce((s, c) => s + c.incidents, 0)
  const totalBlocked = sorted.reduce((s, c) => s + c.blocked, 0)

  return (
    <div className="space-y-0.5">
      {/* Summary stats */}
      <div className="flex items-center gap-3 mb-2 px-1">
        <div>
          <span className="text-[7px] text-slate-500 font-mono">TOTAL INCIDENTS: </span>
          <span className="text-[9px] text-amber-400 font-mono font-bold">{totalIncidents.toLocaleString()}</span>
        </div>
        <div>
          <span className="text-[7px] text-slate-500 font-mono">TOTAL BLOCKED: </span>
          <span className="text-[9px] text-green-400 font-mono font-bold">{totalBlocked.toLocaleString()}</span>
        </div>
      </div>

      {/* Column headers */}
      <div className="flex items-center gap-1 text-[7px] text-slate-600 font-mono px-1 pb-1 border-b border-slate-800/40">
        <span className="w-[14px]">#</span>
        <span className="w-[58px]">COUNTRY</span>
        <span className="flex-1">THREAT INDEX</span>
        <span className="w-[38px] text-right">EVENTS</span>
        <span className="w-[36px] text-right">BLOCK%</span>
        <span className="w-[55px] text-right">ATTACK TYPE</span>
        <span className="w-[14px] text-right">Δ</span>
      </div>

      {sorted.map((c, idx) => (
        <div key={c.iso} className="flex items-center gap-1 group hover:bg-slate-800/20 rounded px-1 py-[3px] transition-colors">
          {/* Rank */}
          <span className="w-[14px] text-[7px] text-slate-600 font-mono">{idx + 1}</span>

          {/* Country name */}
          <span className="w-[58px] text-[8px] text-slate-300 font-mono font-medium truncate">{c.name}</span>

          {/* Threat bar + score */}
          <div className="flex-1 flex items-center gap-1">
            <div className="flex-1 h-3 bg-slate-800/40 rounded overflow-hidden">
              <div
                className="h-full rounded transition-all duration-700"
                style={{
                  width: `${(c.threatIndex / maxThreat) * 100}%`,
                  backgroundColor: threatColor(c.threatIndex, 0.7),
                }}
              />
            </div>
            <span
              className="text-[9px] font-mono font-bold w-[20px] text-right"
              style={{ color: threatColor(c.threatIndex, 0.9) }}
            >
              {c.threatIndex}
            </span>
          </div>

          {/* Incidents */}
          <span className="w-[38px] text-[8px] text-slate-400 font-mono text-right tabular-nums">
            {c.incidents.toLocaleString()}
          </span>

          {/* Block rate */}
          <span
            className="w-[36px] text-[8px] font-mono text-right tabular-nums font-semibold"
            style={{ color: c.blockRate > 60 ? '#22c55e' : c.blockRate > 30 ? '#eab308' : '#ef4444' }}
          >
            {c.blockRate}%
          </span>

          {/* Primary attack type */}
          <span className="w-[55px] text-[7px] text-slate-500 font-mono text-right truncate">
            {c.primaryAttack}
          </span>

          {/* Trend arrow with value */}
          <span className={`w-[14px] text-[8px] font-mono text-right ${c.trend > 0 ? 'text-red-400' : 'text-green-400'}`}>
            {c.trend > 0 ? '↑' : '↓'}
          </span>
        </div>
      ))}
    </div>
  )
}


// ============================================================================
// 4. CROSS-BORDER CORRIDORS — enhanced intelligence table
// ============================================================================

interface CrossBorderCorridorsProps {
  flows: CrossBorderFlow[]
}

export function CrossBorderCorridors({ flows }: CrossBorderCorridorsProps) {
  const sorted = useMemo(() =>
    [...flows].sort((a, b) => b.riskScore - a.riskScore), [flows])

  const totalVolume = flows.reduce((s, f) => s + f.amount, 0)
  const highRiskCount = flows.filter(f => f.riskScore > 70).length

  return (
    <div className="space-y-2">
      {/* Quick summary */}
      <div className="flex items-center gap-4 px-2">
        <div>
          <span className="text-[7px] text-slate-500 font-mono">CORRIDORS: </span>
          <span className="text-[9px] text-slate-300 font-mono font-bold">{flows.length}</span>
        </div>
        <div>
          <span className="text-[7px] text-slate-500 font-mono">TOTAL VOLUME: </span>
          <span className="text-[9px] text-indigo-400 font-mono font-bold">{formatCurrency(totalVolume)}</span>
        </div>
        <div>
          <span className="text-[7px] text-slate-500 font-mono">HIGH RISK: </span>
          <span className="text-[9px] text-red-400 font-mono font-bold">{highRiskCount}</span>
        </div>
      </div>

      <div className="overflow-x-auto">
        <table className="w-full text-[9px]">
          <thead>
            <tr className="border-b border-slate-800">
              <th className="py-1.5 px-2 text-left text-slate-500 font-medium font-mono text-[8px]">Corridor</th>
              <th className="py-1.5 px-2 text-right text-slate-500 font-medium font-mono text-[8px]">Channel</th>
              <th className="py-1.5 px-2 text-right text-slate-500 font-medium font-mono text-[8px]">Tx Count</th>
              <th className="py-1.5 px-2 text-right text-slate-500 font-medium font-mono text-[8px]">Volume</th>
              <th className="py-1.5 px-2 text-right text-slate-500 font-medium font-mono text-[8px]">Risk</th>
              <th className="py-1.5 px-2 text-right text-slate-500 font-medium font-mono text-[8px]">Trend</th>
              <th className="py-1.5 px-2 text-right text-slate-500 font-medium font-mono text-[8px]">Last Seen</th>
              <th className="py-1.5 px-2 text-center text-slate-500 font-medium font-mono text-[8px]">Status</th>
            </tr>
          </thead>
          <tbody>
            {sorted.map((f, i) => (
              <tr key={i} className="border-b border-slate-800/30 hover:bg-slate-800/20 transition-colors">
                <td className="py-1.5 px-2 text-left">
                  <span className="text-slate-300 font-medium">{f.from.label}</span>
                  <span className="text-slate-600 mx-1">→</span>
                  <span className="text-slate-300 font-medium">{f.to.label}</span>
                </td>
                <td className="py-1.5 px-2 text-right">
                  <span className="text-[8px] text-slate-400 font-mono px-1 py-0.5 rounded bg-slate-800/40">{f.channel}</span>
                </td>
                <td className="py-1.5 px-2 text-right text-slate-300 font-mono tabular-nums">{f.txCount}</td>
                <td className="py-1.5 px-2 text-right text-slate-400 font-mono tabular-nums">{formatCurrency(f.amount)}</td>
                <td className="py-1.5 px-2 text-right">
                  <span
                    className="px-1.5 py-0.5 rounded-full text-[8px] font-bold"
                    style={{ backgroundColor: `${riskColor(f.riskScore)}15`, color: riskColor(f.riskScore) }}
                  >
                    {f.riskScore}
                  </span>
                </td>
                <td className={`py-1.5 px-2 text-right text-[8px] font-mono ${f.trend > 0 ? 'text-red-400' : 'text-green-400'}`}>
                  {f.trend > 0 ? '↑' : '↓'}{Math.abs(f.trend).toFixed(0)}%
                </td>
                <td className="py-1.5 px-2 text-right text-[8px] text-slate-500 font-mono">{timeAgo(f.lastDetected)}</td>
                <td className="py-1.5 px-2 text-center">
                  {f.riskScore > 70 ? (
                    <span className="inline-flex items-center gap-0.5 text-[7px] text-red-400 font-mono font-bold">
                      <span className="w-1.5 h-1.5 rounded-full bg-red-500" /> ALERT
                    </span>
                  ) : f.riskScore > 45 ? (
                    <span className="inline-flex items-center gap-0.5 text-[7px] text-yellow-400 font-mono">
                      <span className="w-1.5 h-1.5 rounded-full bg-yellow-500" /> WATCH
                    </span>
                  ) : (
                    <span className="inline-flex items-center gap-0.5 text-[7px] text-green-400 font-mono">
                      <span className="w-1.5 h-1.5 rounded-full bg-green-500" /> CLEAR
                    </span>
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}


// ============================================================================
// 5. ATTACK VECTOR BREAKDOWN — enhanced with totals and ranking
// ============================================================================

interface AttackVectorBreakdownProps {
  vectors: AttackVectorStat[]
}

export function AttackVectorBreakdown({ vectors }: AttackVectorBreakdownProps) {
  const maxCount = Math.max(...vectors.map(v => v.count), 1)
  const totalCount = vectors.reduce((s, v) => s + v.count, 0)

  return (
    <div className="space-y-1.5 px-1">
      {/* Summary header */}
      <div className="flex items-center justify-between mb-1 pb-1 border-b border-slate-800/30">
        <span className="text-[8px] text-slate-500 font-mono">TOTAL EVENTS: <span className="text-slate-300 font-bold">{totalCount.toLocaleString()}</span></span>
        <span className="text-[8px] text-slate-500 font-mono">{vectors.length} TYPES</span>
      </div>

      {vectors.map((v, idx) => (
        <div key={v.attackType} className="group">
          <div className="flex items-center justify-between mb-0.5">
            <div className="flex items-center gap-1.5">
              <span className="text-[7px] text-slate-600 font-mono w-[10px]">{idx + 1}</span>
              <div className="w-2 h-2 rounded-sm" style={{ backgroundColor: v.color }} />
              <span className="text-[9px] text-slate-300 font-mono font-medium">{v.attackType}</span>
            </div>
            <div className="flex items-center gap-2">
              <span className="text-[8px] text-slate-400 font-mono font-bold">{v.percentage}%</span>
              <span className={`text-[8px] font-mono ${v.trend > 0 ? 'text-red-400' : 'text-green-400'}`}>
                {v.trend > 0 ? '↑' : '↓'}{Math.abs(v.trend).toFixed(0)}%
              </span>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <div className="flex-1 h-2.5 bg-slate-800/40 rounded overflow-hidden">
              <div
                className="h-full rounded transition-all duration-500"
                style={{ width: `${(v.count / maxCount) * 100}%`, backgroundColor: v.color, opacity: 0.75 }}
              />
            </div>
            <span className="text-[8px] text-slate-300 font-mono w-[32px] text-right tabular-nums font-bold">{v.count}</span>
          </div>
          <div className="flex items-center gap-1.5 mt-0.5">
            <span className="text-[7px] text-slate-600 font-mono">AVG RISK</span>
            <div className="w-16 h-1.5 bg-slate-800/40 rounded-full overflow-hidden">
              <div className="h-full rounded-full" style={{ width: `${v.avgRisk}%`, backgroundColor: riskColor(v.avgRisk) }} />
            </div>
            <span className="text-[7px] font-mono font-semibold" style={{ color: riskColor(v.avgRisk) }}>{v.avgRisk}</span>
          </div>
        </div>
      ))}
    </div>
  )
}


// ============================================================================
// 6. LIVE THREAT FEED — real-time event stream, no absurd animations
// ============================================================================

interface LiveThreatFeedProps {
  events: ThreatEvent[]
}

export function LiveThreatFeed({ events }: LiveThreatFeedProps) {
  const criticalCount = events.filter(e => e.severity === 'critical').length
  const highCount = events.filter(e => e.severity === 'high').length
  const blockedCount = events.filter(e => e.status === 'blocked').length

  return (
    <div className="space-y-1.5">
      {/* Feed summary bar */}
      <div className="flex items-center gap-3 px-2 pb-1.5 border-b border-slate-800/30">
        <span className="text-[7px] text-slate-500 font-mono">EVENTS: <span className="text-slate-300 font-bold">{events.length}</span></span>
        <span className="text-[7px] text-red-400 font-mono">CRITICAL: {criticalCount}</span>
        <span className="text-[7px] text-orange-400 font-mono">HIGH: {highCount}</span>
        <span className="text-[7px] text-green-400 font-mono">BLOCKED: {blockedCount}</span>
      </div>

      <div className="space-y-1 max-h-[260px] overflow-y-auto pr-1" style={{ scrollbarWidth: 'thin', scrollbarColor: '#1e293b transparent' }}>
        {events.slice(0, 15).map((evt) => (
          <div
            key={evt.id}
            className="flex items-start gap-2 px-2.5 py-1.5 rounded-lg hover:bg-slate-800/20 transition-colors border-l-2"
            style={{ borderLeftColor: severityColor(evt.severity) }}
          >
            {/* Severity indicator — static, no animation */}
            <div className="mt-1 flex-shrink-0">
              <div
                className="w-2 h-2 rounded-full"
                style={{ backgroundColor: severityColor(evt.severity) }}
              />
            </div>

            {/* Content */}
            <div className="flex-1 min-w-0">
              <div className="flex items-center gap-1.5">
                <span
                  className="text-[7px] font-mono font-bold px-1.5 py-0.5 rounded uppercase"
                  style={{ backgroundColor: `${severityColor(evt.severity)}15`, color: severityColor(evt.severity) }}
                >
                  {evt.severity}
                </span>
                <span className="text-[9px] text-slate-300 font-mono font-medium">{evt.city}, {evt.country}</span>
                <span className="text-[8px] text-slate-600 font-mono ml-auto flex-shrink-0">{timeAgo(evt.timestamp)}</span>
              </div>
              <p className="text-[9px] text-slate-400 mt-0.5 leading-relaxed truncate">{evt.description}</p>
              <div className="flex items-center gap-2 mt-0.5">
                <span className="text-[8px] text-slate-500 font-mono">{evt.attackType}</span>
                {evt.amount > 0 && (
                  <span className="text-[8px] text-slate-500 font-mono">{formatCurrency(evt.amount)}</span>
                )}
                <span className={`text-[7px] font-mono px-1.5 py-0.5 rounded font-semibold ${
                  evt.status === 'blocked' ? 'text-green-400 bg-green-900/20' :
                  evt.status === 'escalated' ? 'text-red-400 bg-red-900/20' :
                  evt.status === 'investigating' ? 'text-amber-400 bg-amber-900/20' :
                  'text-cyan-400 bg-cyan-900/20'
                }`}>
                  {evt.status.toUpperCase()}
                </span>
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}

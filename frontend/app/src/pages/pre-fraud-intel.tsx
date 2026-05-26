// ============================================================================
// Pre-Fraud Intel Page -- preventive OSINT/SOCMINT cockpit for Union Bank PS3
// ============================================================================

import { useEffect, useMemo, useState } from 'react'
import {
  Area,
  AreaChart,
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Pie,
  PieChart,
  ResponsiveContainer,
  Tooltip,
  Line,
  LineChart,
  XAxis,
  YAxis,
} from 'recharts'
import {
  Activity,
  ArrowRight,
  BadgeCheck,
  BrainCircuit,
  Crosshair,
  ExternalLink,
  Film,
  Globe2,
  Image as ImageIcon,
  Landmark,
  Layers,
  MapPin,
  Newspaper,
  Play,
  Radar,
  Radio,
  RefreshCw,
  Route,
  ShieldCheck,
  Video,
  Zap,
} from 'lucide-react'
import {
  useIntelCockpit,
  useIntelMedia,
  useIntelPlaybooks,
  useIntelSignals,
  useIntelSources,
  useIntelTrends,
  useIntelTuningStatus,
  useLaunchPS3Scenario,
  useRefreshIntel,
  useSimulateIntelSignal,
} from '@/hooks/use-api'
import { useUIStore } from '@/stores/use-ui-store'
import { cn } from '@/lib/utils'
import type {
  AdaptivePlaybook,
  ExternalThreatSignal,
  FraudTrendCluster,
  IntelCockpitResponse,
  IntelGeoHotspot,
  IntelMediaPreview,
  IntelSourceConfig,
} from '@/lib/types'

const DEMO_SCENARIOS = [
  { id: 'digital_arrest_mule', label: 'Digital Arrest Mule Burst' },
  { id: 'kyc_phishing', label: 'KYC APK Phishing' },
  { id: 'loan_app_mule', label: 'Loan App Collections' },
  { id: 'investment_scam', label: 'Investment Scam Chain' },
]

const CHART_COLORS = ['#22d3ee', '#34d399', '#f59e0b', '#fb7185', '#a78bfa', '#60a5fa']

function pct(value: number | undefined) {
  return `${Math.round((value ?? 0) * 100)}%`
}

function timeLabel(value: number | null | undefined) {
  if (!value) return 'not polled'
  return new Date(value * 1000).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
}

function ageLabel(seconds: number | null | undefined) {
  if (seconds == null) return 'not yet'
  if (seconds < 60) return `${Math.max(0, Math.round(seconds))}s ago`
  if (seconds < 3600) return `${Math.round(seconds / 60)}m ago`
  return `${Math.round(seconds / 3600)}h ago`
}

function xmlEscape(value: string) {
  return value.replace(/[<>&"']/g, (ch) => ({ '<': '&lt;', '>': '&gt;', '&': '&amp;', '"': '&quot;', "'": '&apos;' })[ch] ?? ch)
}

function thumbnailPalette(key: string) {
  if (key.includes('kyc')) return ['#0f766e', '#22d3ee', '#f59e0b']
  if (key.includes('loan')) return ['#7c2d12', '#fb923c', '#34d399']
  if (key.includes('investment')) return ['#14532d', '#84cc16', '#38bdf8']
  if (key.includes('dormant')) return ['#312e81', '#a78bfa', '#f43f5e']
  if (key.includes('layering')) return ['#1e1b4b', '#a78bfa', '#22d3ee']
  return ['#082f49', '#22d3ee', '#fb7185']
}

function stableNumber(value: string) {
  return value.split('').reduce((sum, ch) => sum + ch.charCodeAt(0), 0)
}

function mediaUrl(preview: IntelMediaPreview) {
  const status = mediaStatus(preview)
  if (status === 'real_video_embed') return preview.thumbnail_url || preview.publisher_logo_url || ''
  return preview.thumbnail_url || preview.image_url || preview.media_url || ''
}

function mediaStatus(preview: IntelMediaPreview) {
  if (preview.preview_status) return preview.preview_status
  if (preview.image_status === 'real_image' || preview.image_status === 'real_video_embed') return preview.image_status
  if (preview.media_origin === 'publisher_logo') return 'publisher_logo_only'
  if (preview.media_origin === 'generated_poster') return 'generated_fallback'
  if (['gdelt_social_image', 'open_graph_image', 'live_image', 'bing_news_image'].includes(String(preview.media_origin))) return 'real_image'
  return 'source_card'
}

function isRealImage(preview: IntelMediaPreview) {
  return mediaStatus(preview) === 'real_image' && Boolean(mediaUrl(preview))
}

function isRealVideo(preview: IntelMediaPreview) {
  return mediaStatus(preview) === 'real_video_embed' && Boolean(preview.video_embed_url) && preview.embed_allowed !== false
}

function isVideoSource(preview: IntelMediaPreview) {
  return isRealVideo(preview) || preview.media_type === 'video' || Boolean(preview.video_page_url)
}

function isSourceCard(preview: IntelMediaPreview) {
  return ['source_card', 'publisher_logo_only'].includes(mediaStatus(preview))
}

function isFallbackMedia(preview: IntelMediaPreview) {
  return mediaStatus(preview) === 'generated_fallback' || mediaStatus(preview) === 'broken'
}

function mediaRank(preview: IntelMediaPreview) {
  if (isRealVideo(preview)) return 0
  if (isRealImage(preview)) return 1
  if (isVideoSource(preview)) return 2
  if (isSourceCard(preview)) return 3
  return 3
}

function mediaOriginLabel(preview: IntelMediaPreview) {
  const status = mediaStatus(preview)
  if (status === 'real_video_embed') return 'verified video'
  if (status === 'real_image') return 'real image'
  if (status === 'source_card') return 'source card'
  if (status === 'publisher_logo_only') return 'publisher logo only'
  if (status === 'generated_fallback') return 'fallback'
  const origin = preview.media_origin || (mediaUrl(preview) ? 'live_image' : 'generated_poster')
  if (origin === 'gdelt_social_image') return 'live news image'
  if (origin === 'open_graph_image') return 'source page image'
  if (origin === 'live_image') return 'live source image'
  if (origin === 'publisher_logo') return 'publisher signal'
  return 'generated fallback'
}

function mediaSvgDataUri(preview: IntelMediaPreview) {
  const [bg, accent, hot] = thumbnailPalette(preview.thumbnail_key)
  const seed = stableNumber(`${preview.media_id}-${preview.title}-${preview.source_kind}`)
  const waveA = 208 + (seed % 44)
  const waveB = 140 + (seed % 58)
  const orbX = 420 + (seed % 96)
  const orbY = 82 + (seed % 72)
  const barWidth = 260 + (seed % 220)
  const words = preview.title.split(/\s+/)
  const lines: string[] = []
  for (const word of words) {
    const current = lines[lines.length - 1] ?? ''
    if (!current) lines.push(word)
    else if (`${current} ${word}`.length <= 34) lines[lines.length - 1] = `${current} ${word}`
    else if (lines.length < 2) lines.push(word)
  }
  if (lines.length === 2 && words.join(' ').length > lines.join(' ').length) {
    lines[1] = `${lines[1].slice(0, 31)}...`
  }
  const kind = xmlEscape(preview.source_kind.toUpperCase())
  const svg = `
    <svg xmlns="http://www.w3.org/2000/svg" width="640" height="360" viewBox="0 0 640 360">
      <defs>
        <linearGradient id="g" x1="0" x2="1" y1="0" y2="1">
          <stop offset="0" stop-color="${bg}"/>
          <stop offset="0.58" stop-color="#020617"/>
          <stop offset="1" stop-color="#0f172a"/>
        </linearGradient>
        <pattern id="grid" width="28" height="28" patternUnits="userSpaceOnUse">
          <path d="M 28 0 L 0 0 0 28" fill="none" stroke="${accent}" stroke-opacity=".12" stroke-width="1"/>
        </pattern>
      </defs>
      <rect width="640" height="360" fill="url(#g)"/>
      <rect width="640" height="360" fill="url(#grid)"/>
      <circle cx="${orbX}" cy="${orbY}" r="${74 + (seed % 28)}" fill="${accent}" opacity=".13"/>
      <circle cx="${orbX + 42}" cy="${orbY + 38}" r="${34 + (seed % 22)}" fill="${hot}" opacity=".18"/>
      <path d="M70 ${waveA} C160 ${waveB},210 ${waveA + 54},312 ${waveB + 30} S472 ${waveB - 24},570 ${waveA - 8}" stroke="${accent}" stroke-width="7" fill="none" stroke-opacity=".8"/>
      <path d="M70 ${waveA + 36} C158 ${waveB + 42},232 ${waveA + 82},328 ${waveB + 70} S460 ${waveB + 48},570 ${waveA + 46}" stroke="${hot}" stroke-width="4" fill="none" stroke-opacity=".65"/>
      <g opacity=".85">
        <circle cx="92" cy="248" r="10" fill="${accent}"/>
        <circle cx="206" cy="264" r="14" fill="${hot}"/>
        <circle cx="326" cy="211" r="12" fill="${accent}"/>
        <circle cx="470" cy="169" r="16" fill="${hot}"/>
        <circle cx="568" cy="207" r="11" fill="${accent}"/>
      </g>
      <rect x="32" y="30" width="182" height="28" rx="6" fill="#020617" opacity=".72" stroke="${accent}" stroke-opacity=".5"/>
      <text x="44" y="49" fill="${accent}" font-family="Arial, sans-serif" font-size="13" font-weight="700">${kind}</text>
      <text x="34" y="96" fill="#f8fafc" font-family="Arial, sans-serif" font-size="25" font-weight="800">${xmlEscape(lines[0] ?? preview.title.slice(0, 34))}</text>
      <text x="34" y="126" fill="#f8fafc" font-family="Arial, sans-serif" font-size="25" font-weight="800">${xmlEscape(lines[1] ?? '')}</text>
      <text x="36" y="158" fill="#94a3b8" font-family="Arial, sans-serif" font-size="14">Public-source preview | India banking fraud intelligence</text>
      <rect x="36" y="295" width="568" height="8" rx="4" fill="#0f172a" opacity=".88"/>
      <rect x="36" y="295" width="${barWidth}" height="8" rx="4" fill="${accent}"/>
      <text x="36" y="316" fill="#cbd5e1" font-family="Arial, sans-serif" font-size="11" font-weight="700">GENERATED FALLBACK | NO VERIFIED MEDIA THUMBNAIL</text>
    </svg>`
  return `data:image/svg+xml;charset=utf-8,${encodeURIComponent(svg)}`
}

function ToneBadge({ label, tone = 'slate' }: { label: string; tone?: 'slate' | 'emerald' | 'amber' | 'rose' | 'cyan' }) {
  const classes = {
    emerald: 'border-emerald-400/30 bg-emerald-500/10 text-emerald-300',
    amber: 'border-amber-400/30 bg-amber-500/10 text-amber-300',
    rose: 'border-rose-400/30 bg-rose-500/10 text-rose-300',
    cyan: 'border-cyan-400/30 bg-cyan-500/10 text-cyan-200',
    slate: 'border-border-default bg-bg-overlay text-text-muted',
  }[tone]
  return (
    <span className={cn('inline-flex items-center rounded-md border px-2 py-1 text-[9px] font-bold uppercase tracking-[0.12em]', classes)}>
      {label}
    </span>
  )
}

function Metric({ label, value, accent = 'text-text-primary' }: { label: string; value: string; accent?: string }) {
  return (
    <div className="min-w-0 rounded-md border border-border-subtle bg-bg-elevated/55 p-2">
      <div className="mb-1 truncate text-[8px] font-semibold uppercase tracking-[0.12em] text-text-muted">{label}</div>
      <div className={cn('truncate font-mono text-[12px] font-semibold', accent)} title={value}>{value}</div>
    </div>
  )
}

function Panel({
  title,
  icon: Icon,
  badge,
  children,
  className,
}: {
  title: string
  icon: typeof Radar
  badge?: string
  children: React.ReactNode
  className?: string
}) {
  return (
    <section className={cn('min-w-0 rounded-lg border border-border-subtle bg-bg-surface shadow-sm', className)}>
      <div className="flex items-center justify-between gap-3 border-b border-border-subtle px-3 py-2.5">
        <div className="flex min-w-0 items-center gap-2 text-[10px] font-semibold uppercase tracking-[0.14em] text-text-secondary">
          <Icon className="h-4 w-4 text-accent-primary" />
          <span className="truncate">{title}</span>
        </div>
        {badge && <ToneBadge label={badge} tone="cyan" />}
      </div>
      {children}
    </section>
  )
}

function MediaPreviewCard({ preview, prominent = false }: { preview: IntelMediaPreview; prominent?: boolean }) {
  const generatedPoster = mediaSvgDataUri(preview)
  const status = mediaStatus(preview)
  const sourceCard = isSourceCard(preview)
  const fallbackOnly = isFallbackMedia(preview)
  const realImage = isRealImage(preview)
  const realVideo = isRealVideo(preview)
  const poster = realImage ? mediaUrl(preview) : (fallbackOnly ? generatedPoster : '')
  const statusTone: 'amber' | 'emerald' | 'rose' = realImage || realVideo ? 'emerald' : fallbackOnly ? 'rose' : 'amber'
  return (
    <article className={cn('group relative overflow-hidden rounded-lg border border-border-subtle bg-bg-elevated/60', prominent ? 'min-h-[270px]' : 'min-h-[184px]')}>
      {realVideo ? (
        <iframe
          src={preview.video_embed_url ?? undefined}
          title={preview.title}
          loading="lazy"
          referrerPolicy="no-referrer-when-downgrade"
          allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
          className="absolute inset-0 h-full w-full border-0 bg-[#071427]"
        />
      ) : realImage ? (
        <img
          src={poster}
          alt={preview.caption}
          onError={(event) => { event.currentTarget.src = preview.publisher_logo_url || generatedPoster }}
          className="absolute inset-0 h-full w-full object-cover opacity-90"
        />
      ) : fallbackOnly ? (
        <img
          src={generatedPoster}
          alt={preview.caption}
          className="absolute inset-0 h-full w-full object-cover opacity-90"
        />
      ) : (
        <div className="absolute inset-0 bg-[linear-gradient(135deg,#ffffff_0%,#eef6ff_55%,#d8ecff_100%)]">
          <div className="absolute inset-x-0 top-0 h-1.5 bg-[linear-gradient(90deg,#ed1b24,#0057a8)]" />
          <div className="absolute left-5 top-6 flex items-center gap-3">
            <div className="flex h-16 w-16 items-center justify-center rounded-xl border border-[#0057a8]/20 bg-white shadow-sm">
              {preview.publisher_logo_url ? (
                <img src={preview.publisher_logo_url} alt="" className="h-10 w-10 object-contain" />
              ) : (
                <Newspaper className="h-8 w-8 text-accent-primary" />
              )}
            </div>
            <div className="min-w-0">
              <div className="text-[10px] font-bold uppercase tracking-[0.12em] text-accent-primary">{preview.source_kind}</div>
              <div className="mt-1 max-w-[360px] truncate text-sm font-bold text-text-primary">{preview.publisher}</div>
              <div className="mt-1 max-w-[360px] truncate font-mono text-[10px] text-text-muted">{preview.source_domain || preview.source_url}</div>
            </div>
          </div>
          <div className="absolute inset-x-5 bottom-5 rounded-lg border border-border-subtle bg-white/85 p-3 shadow-sm">
            <div className="mb-1 text-[9px] font-bold uppercase tracking-[0.12em] text-amber-600">Source card | no verified article media</div>
            <div className="line-clamp-2 text-[13px] font-semibold leading-snug text-text-primary">{preview.title}</div>
            <a
              href={preview.video_page_url || preview.source_url}
              target="_blank"
              rel="noreferrer"
              className="mt-2 inline-flex items-center gap-1 rounded bg-[#0057a8] px-2 py-1 text-[8px] font-bold uppercase tracking-[0.08em] text-white"
            >
              Open source
              <ExternalLink className="h-3 w-3" />
            </a>
          </div>
        </div>
      )}
      {!sourceCard && <div className="absolute inset-0 bg-gradient-to-t from-black/75 via-black/25 to-transparent" />}
      <div className="absolute left-3 top-3 flex items-center gap-2">
        <ToneBadge label={mediaOriginLabel(preview)} tone={statusTone} />
        {preview.source_tier && <ToneBadge label={preview.source_tier.replace('_', ' ')} />}
      </div>
      {realVideo && (
        <div className="absolute right-3 top-3 flex items-center gap-1 rounded-md border border-white/15 bg-black/45 px-2 py-1 text-[9px] font-bold text-white">
          <Play className="h-3 w-3" />
          {preview.video_provider ?? 'video'}
        </div>
      )}
      <div className={cn('absolute inset-x-0 bottom-0 p-3', sourceCard && 'hidden')}>
        <div className="mb-1 flex items-center gap-2 text-[9px] uppercase tracking-[0.12em] text-white/80">
          {realVideo ? <Film className="h-3.5 w-3.5 text-rose-300" /> : <ImageIcon className="h-3.5 w-3.5 text-cyan-300" />}
          <span className="truncate">{preview.source_domain || preview.publisher}</span>
        </div>
        <h3 className={cn('line-clamp-2 font-semibold leading-tight text-white', prominent ? 'text-base' : 'text-[12px]')} title={preview.title}>
          {preview.title}
        </h3>
        <p className="mt-1 line-clamp-2 text-[10px] leading-relaxed text-white/80">{preview.caption}</p>
        <div className="mt-2 flex flex-wrap gap-1.5">
          {preview.image_status && (
            <span className="rounded bg-white/85 px-2 py-0.5 text-[8px] font-semibold uppercase tracking-[0.08em] text-[#0057a8]">
              {status.replace(/_/g, ' ')}
            </span>
          )}
          {(preview.typologies ?? []).slice(0, 3).map((typology) => (
            <span key={typology} className="rounded bg-bg-overlay/85 px-2 py-0.5 text-[8px] font-semibold uppercase tracking-[0.08em] text-text-secondary">
              {typology.replace(/_/g, ' ')}
            </span>
          ))}
        </div>
        {(preview.video_page_url || preview.source_url) && (
          <a
            href={preview.video_page_url || preview.source_url}
            target="_blank"
            rel="noreferrer"
            className="mt-2 inline-flex items-center gap-1 rounded bg-white/85 px-2 py-1 text-[8px] font-bold uppercase tracking-[0.08em] text-[#0057a8]"
          >
            Open source
            <ExternalLink className="h-3 w-3" />
          </a>
        )}
      </div>
    </article>
  )
}

type MediaFilter = 'all' | 'images' | 'videos' | 'source_cards' | 'fallbacks'

function MediaCommandDeck({ previews }: { previews: IntelMediaPreview[] }) {
  const [activeId, setActiveId] = useState<string | null>(null)
  const [filter, setFilter] = useState<MediaFilter>('all')
  const sorted = useMemo(() => [...previews].sort((a, b) => mediaRank(a) - mediaRank(b)), [previews])
  const filtered = useMemo(() => sorted.filter((preview) => {
    if (filter === 'images') return isRealImage(preview)
    if (filter === 'videos') return isVideoSource(preview)
    if (filter === 'source_cards') return isSourceCard(preview)
    if (filter === 'fallbacks') return isFallbackMedia(preview)
    return true
  }), [filter, sorted])
  const selected = filtered.find((preview) => preview.media_id === activeId) ?? filtered[0]
  const filters: Array<{ id: MediaFilter; label: string; count: number }> = [
    { id: 'all', label: 'All', count: sorted.length },
    { id: 'images', label: 'Images', count: sorted.filter(isRealImage).length },
    { id: 'videos', label: 'Videos', count: sorted.filter(isVideoSource).length },
    { id: 'source_cards', label: 'Source Cards', count: sorted.filter(isSourceCard).length },
    { id: 'fallbacks', label: 'Fallbacks', count: sorted.filter(isFallbackMedia).length },
  ]
  if (!sorted.length) return null
  return (
    <div className="space-y-3">
      <div className="flex flex-wrap gap-1.5">
        {filters.map((item) => (
          <button
            key={item.id}
            onClick={() => {
              setFilter(item.id)
              setActiveId(null)
            }}
            className={cn(
              'rounded-md border px-2.5 py-1 text-[8px] font-bold uppercase tracking-[0.12em]',
              filter === item.id
                ? 'border-accent-primary bg-accent-primary text-white'
                : 'border-border-subtle bg-bg-elevated text-text-muted hover:border-accent-primary hover:text-accent-primary',
            )}
          >
            {item.label} <span className="font-mono">{item.count}</span>
          </button>
        ))}
      </div>
      {filters.find((item) => item.id === 'videos')?.count === 0 && (
        <div className="rounded-md border border-amber-400/35 bg-amber-500/10 px-3 py-2 text-[9px] font-semibold text-amber-700">
          No verified public video source found in the current pulse. Embeds appear only when public provider metadata is available.
        </div>
      )}
      {!selected ? (
        <div className="flex min-h-[320px] items-center justify-center rounded-lg border border-dashed border-border-default bg-bg-elevated/45 p-6 text-center">
          <div>
            <Video className="mx-auto mb-3 h-8 w-8 text-text-muted" />
            <div className="text-sm font-bold text-text-primary">No verified public media in this filter</div>
            <div className="mt-1 max-w-md text-[10px] leading-relaxed text-text-muted">
              PayFlow only shows videos or images when source pages expose real public metadata. Switch to Source Cards to inspect publisher evidence.
            </div>
          </div>
        </div>
      ) : (
      <div className="grid min-h-[320px] grid-cols-1 gap-3 xl:grid-cols-[minmax(0,1.15fr)_minmax(300px,0.85fr)]">
        <MediaPreviewCard preview={selected} prominent />
        <div className="min-h-0 max-h-[420px] space-y-2 overflow-auto pr-1">
          {filtered.slice(0, 14).map((preview) => {
            const thumb = isFallbackMedia(preview) ? mediaSvgDataUri(preview) : mediaUrl(preview)
            return (
              <button
                key={preview.media_id}
                onClick={() => setActiveId(preview.media_id)}
                className={cn(
                  'flex w-full items-center gap-3 rounded-lg border p-2 text-left transition-colors',
                  preview.media_id === selected.media_id
                    ? 'border-accent-primary/60 bg-accent-primary/10'
                    : 'border-border-subtle bg-bg-elevated/45 hover:border-border-default',
                )}
              >
                {thumb ? (
                  <img
                    src={thumb}
                    alt=""
                    onError={(event) => { event.currentTarget.src = mediaSvgDataUri(preview) }}
                    className={cn(
                      'h-16 w-24 shrink-0 rounded-md border border-border-subtle bg-white object-cover',
                      isSourceCard(preview) && 'object-contain p-3',
                    )}
                  />
                ) : (
                  <div className="flex h-16 w-24 shrink-0 items-center justify-center rounded-md border border-border-subtle bg-white">
                    <Newspaper className="h-6 w-6 text-accent-primary" />
                  </div>
                )}
                <div className="min-w-0">
                  <div className="mb-1 flex items-center gap-1.5 text-[8px] uppercase tracking-[0.12em] text-text-muted">
                    {isRealVideo(preview) ? <Video className="h-3 w-3 text-rose-300" /> : <ImageIcon className="h-3 w-3 text-cyan-300" />}
                    {mediaOriginLabel(preview)} | {preview.source_kind}
                  </div>
                  <div className="line-clamp-2 text-[10px] font-semibold text-text-primary">{preview.title}</div>
                  <div className="mt-1 truncate font-mono text-[8px] text-text-muted">
                    {pct(preview.trust_score)} trust | {mediaStatus(preview).replace(/_/g, ' ')}{preview.source_domain ? ` | ${preview.source_domain}` : ''}
                  </div>
                </div>
              </button>
            )
          })}
        </div>
      </div>
      )}
    </div>
  )
}

function LiveVelocityChart({ cockpit }: { cockpit?: IntelCockpitResponse }) {
  const data = cockpit?.signal_timeline ?? []
  return (
    <Panel title="Live Signal Velocity" icon={Activity} badge={`${cockpit?.metrics.live_mentions ?? 0} mentions`}>
      <div className="h-[220px] p-2">
        <ResponsiveContainer width="100%" height="100%">
          <AreaChart data={data}>
            <defs>
              <linearGradient id="officialFill" x1="0" x2="0" y1="0" y2="1">
                <stop offset="0%" stopColor="#22d3ee" stopOpacity={0.7} />
                <stop offset="100%" stopColor="#22d3ee" stopOpacity={0.02} />
              </linearGradient>
              <linearGradient id="socialFill" x1="0" x2="0" y1="0" y2="1">
                <stop offset="0%" stopColor="#fb7185" stopOpacity={0.5} />
                <stop offset="100%" stopColor="#fb7185" stopOpacity={0.02} />
              </linearGradient>
            </defs>
            <CartesianGrid stroke="rgba(148,163,184,0.08)" vertical={false} />
            <XAxis dataKey="time" tick={{ fontSize: 9, fill: '#64748b' }} tickLine={false} axisLine={false} />
            <YAxis tick={{ fontSize: 9, fill: '#64748b' }} tickLine={false} axisLine={false} width={28} />
            <Tooltip contentStyle={{ background: '#020617', border: '1px solid #1e293b', borderRadius: 8, color: '#e2e8f0', fontSize: 11 }} />
            <Area type="monotone" dataKey="official" stackId="1" stroke="#22d3ee" fill="url(#officialFill)" strokeWidth={2} />
            <Area type="monotone" dataKey="news" stackId="1" stroke="#34d399" fill="#34d39922" strokeWidth={2} />
            <Area type="monotone" dataKey="social" stackId="1" stroke="#fb7185" fill="url(#socialFill)" strokeWidth={2} />
          </AreaChart>
        </ResponsiveContainer>
      </div>
    </Panel>
  )
}

function ChannelExposureChart({ cockpit }: { cockpit?: IntelCockpitResponse }) {
  const data = cockpit?.channel_exposure ?? []
  return (
    <Panel title="Channel Exposure" icon={Zap}>
      <div className="h-[220px] p-2">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={data} layout="vertical" margin={{ left: 8, right: 12, top: 4, bottom: 4 }}>
            <CartesianGrid stroke="rgba(148,163,184,0.08)" horizontal={false} />
            <XAxis type="number" domain={[0, 1]} hide />
            <YAxis type="category" dataKey="channel" width={72} tick={{ fontSize: 9, fill: '#94a3b8' }} axisLine={false} tickLine={false} />
            <Tooltip formatter={(value) => pct(Number(value))} contentStyle={{ background: '#020617', border: '1px solid #1e293b', borderRadius: 8, color: '#e2e8f0', fontSize: 11 }} />
            <Bar dataKey="exposure" radius={[0, 6, 6, 0]}>
              {data.map((_, index) => <Cell key={index} fill={CHART_COLORS[index % CHART_COLORS.length]} />)}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
    </Panel>
  )
}

function SourceVelocityPanel({ cockpit }: { cockpit?: IntelCockpitResponse }) {
  const data = cockpit?.source_velocity_series ?? []
  return (
    <Panel title="Source Velocity By Tier" icon={Activity} badge={`${data.length} buckets`}>
      <div className="h-[220px] p-2">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={data}>
            <CartesianGrid stroke="rgba(0,87,168,0.10)" vertical={false} />
            <XAxis dataKey="time" tick={{ fontSize: 9, fill: '#64748b' }} tickLine={false} axisLine={false} minTickGap={18} />
            <YAxis tick={{ fontSize: 9, fill: '#64748b' }} tickLine={false} axisLine={false} width={28} />
            <Tooltip contentStyle={{ background: '#ffffff', border: '1px solid #c6d3e3', borderRadius: 8, color: '#172033', fontSize: 11 }} />
            <Line type="monotone" dataKey="official" stroke="#0057a8" strokeWidth={2.2} dot={false} />
            <Line type="monotone" dataKey="news" stroke="#10b981" strokeWidth={2} dot={false} />
            <Line type="monotone" dataKey="social" stroke="#f59e0b" strokeWidth={2} dot={false} />
            <Line type="monotone" dataKey="open_web" stroke="#d71920" strokeWidth={2} dot={false} />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </Panel>
  )
}

function TypologyVelocityPanel({ cockpit }: { cockpit?: IntelCockpitResponse }) {
  const data = cockpit?.typology_velocity_series ?? []
  return (
    <Panel title="Emerging Typology Velocity" icon={Radar} badge={`${data.length} typologies`}>
      <div className="space-y-2 p-3">
        {data.map((row) => (
          <div key={row.typology} className="grid grid-cols-[minmax(0,1fr)_72px_54px] items-center gap-2 rounded-md border border-border-subtle bg-bg-elevated/60 p-2">
            <div className="min-w-0">
              <div className="truncate text-[10px] font-bold text-text-primary">{row.label}</div>
              <div className="font-mono text-[8px] text-text-muted">{row.mentions} public mentions | {row.signals} signals</div>
            </div>
            <div className="h-2 overflow-hidden rounded-full bg-white">
              <div className="h-full rounded-full bg-alert-critical" style={{ width: `${Math.round(row.velocity * 100)}%` }} />
            </div>
            <div className="text-right font-mono text-[10px] font-bold text-accent-primary">{pct(row.velocity)}</div>
          </div>
        ))}
      </div>
    </Panel>
  )
}

function ChannelTypologyHeatmapPanel({ cockpit }: { cockpit?: IntelCockpitResponse }) {
  const rows = cockpit?.channel_typology_heatmap ?? []
  const channels = ['UPI', 'IMPS', 'DIGITAL_BANKING', 'NEFT', 'RTGS', 'CARDS', 'BRANCH']
  return (
    <Panel title="Channel X Typology Heatmap" icon={Layers} badge={`${rows.length} rows`}>
      <div className="overflow-auto p-3">
        <div className="min-w-[620px] space-y-2">
          <div className="grid grid-cols-[170px_repeat(7,52px)_54px] gap-1 text-[8px] font-bold uppercase tracking-[0.1em] text-text-muted">
            <span>Typology</span>
            {channels.map((channel) => <span key={channel} className="text-center">{channel.replace('DIGITAL_BANKING', 'DIG')}</span>)}
            <span className="text-right">Trust</span>
          </div>
          {rows.slice(0, 9).map((row) => {
            const max = Math.max(...channels.map((channel) => Number(row[channel] ?? 0)), 1)
            return (
              <div key={String(row.typology)} className="grid grid-cols-[170px_repeat(7,52px)_54px] items-center gap-1 rounded-md border border-border-subtle bg-bg-elevated/55 p-1.5">
                <span className="truncate text-[9px] font-semibold text-text-primary" title={String(row.label)}>{String(row.label)}</span>
                {channels.map((channel, index) => {
                  const value = Number(row[channel] ?? 0)
                  return (
                    <div
                      key={channel}
                      className="flex h-7 items-center justify-center rounded border border-border-subtle font-mono text-[9px] font-bold text-text-primary"
                      style={{ backgroundColor: `${CHART_COLORS[index % CHART_COLORS.length]}${Math.round(35 + (value / max) * 140).toString(16).padStart(2, '0')}` }}
                    >
                      {value}
                    </div>
                  )
                })}
                <span className="text-right font-mono text-[9px] text-alert-low">{pct(Number(row.trust ?? 0))}</span>
              </div>
            )
          })}
        </div>
      </div>
    </Panel>
  )
}

function MediaEvidenceMatrixPanel({ cockpit }: { cockpit?: IntelCockpitResponse }) {
  const rows = cockpit?.media_evidence_matrix ?? []
  const total = cockpit?.metrics.media_items ?? rows.reduce((sum, row) => sum + row.items, 0)
  const live = cockpit?.metrics.live_media_items ?? 0
  const videos = cockpit?.metrics.real_videos ?? 0
  const sourceCards = (cockpit?.metrics.source_cards ?? 0) + (cockpit?.metrics.publisher_logo_only ?? 0)
  const fallback = cockpit?.metrics.generated_fallbacks ?? Math.max(0, total - live - sourceCards)
  const health = cockpit?.metrics.media_health ?? (total > 0 ? live / total : 0)
  const official = rows.reduce((sum, row) => sum + row.official, 0)
  const news = rows.reduce((sum, row) => sum + row.news, 0)
  const statusRows = [
    { label: 'Verified source media', value: live, ratio: total > 0 ? live / total : 0 },
    { label: 'Verified video embeds', value: videos, ratio: total > 0 ? videos / total : 0 },
    { label: 'Source cards', value: sourceCards, ratio: total > 0 ? sourceCards / total : 0 },
    { label: 'Official-source previews', value: official, ratio: total > 0 ? official / total : 0 },
    { label: 'News-source previews', value: news, ratio: total > 0 ? news / total : 0 },
    { label: 'Generated fallback posters', value: fallback, ratio: total > 0 ? fallback / total : 0 },
  ]
  return (
    <Panel title="Media Evidence Health" icon={ImageIcon} badge={`${live} real`}>
      <div className="space-y-2 p-3">
        <div className="grid grid-cols-3 gap-2">
          <Metric label="Health" value={pct(health)} accent="text-emerald-300" />
          <Metric label="Real" value={`${live}/${total}`} accent="text-accent-primary" />
          <Metric label="Fallback" value={String(fallback)} accent={fallback > 0 ? 'text-amber-300' : 'text-emerald-300'} />
        </div>
        {rows.map((row) => (
          <div key={row.origin} className="rounded-md border border-border-subtle bg-bg-elevated/55 p-2">
            <div className="mb-1 flex items-center justify-between gap-2">
              <span className="truncate text-[10px] font-bold text-text-primary">{row.label}</span>
              <span className="font-mono text-[9px] font-bold text-accent-primary">{row.items} items</span>
            </div>
            <div className="grid grid-cols-4 gap-1 text-center font-mono text-[8px] text-text-muted">
              <span className="rounded bg-white py-1">official {row.official}</span>
              <span className="rounded bg-white py-1">news {row.news}</span>
              <span className="rounded bg-white py-1">social {row.social}</span>
              <span className="rounded bg-white py-1">open {row.open_web}</span>
            </div>
          </div>
        ))}
        <div className="space-y-1.5 rounded-md border border-border-subtle bg-bg-elevated/45 p-2">
          {statusRows.map((row) => (
            <div key={row.label} className="grid grid-cols-[minmax(0,1fr)_44px] items-center gap-2">
              <div className="min-w-0">
                <div className="flex items-center justify-between gap-2 text-[8px] font-semibold uppercase tracking-[0.08em] text-text-muted">
                  <span className="truncate">{row.label}</span>
                  <span className="font-mono text-text-primary">{row.value}</span>
                </div>
                <div className="mt-1 h-1.5 overflow-hidden rounded-full bg-white">
                  <div className="h-full rounded-full bg-accent-primary" style={{ width: `${Math.max(4, Math.round(row.ratio * 100))}%` }} />
                </div>
              </div>
              <span className="text-right font-mono text-[9px] text-text-muted">{pct(row.ratio)}</span>
            </div>
          ))}
        </div>
      </div>
    </Panel>
  )
}

function SourceMixPanel({ cockpit }: { cockpit?: IntelCockpitResponse }) {
  const data = cockpit?.source_mix ?? []
  return (
    <Panel title="Source Trust Mix" icon={Landmark}>
      <div className="grid grid-cols-[150px_minmax(0,1fr)] gap-2 p-3">
        <div className="h-[160px]">
          <ResponsiveContainer width="100%" height="100%">
            <PieChart>
              <Pie data={data} dataKey="signals" innerRadius={42} outerRadius={68} paddingAngle={3}>
                {data.map((_, index) => <Cell key={index} fill={CHART_COLORS[index % CHART_COLORS.length]} />)}
              </Pie>
            </PieChart>
          </ResponsiveContainer>
        </div>
        <div className="space-y-2">
          {data.map((row, index) => (
            <div key={row.tier} className="flex items-center justify-between gap-2 rounded-md border border-border-subtle bg-bg-elevated/45 px-2 py-1.5">
              <div className="min-w-0">
                <div className="truncate text-[10px] font-semibold text-text-primary">{row.label}</div>
                <div className="font-mono text-[8px] text-text-muted">{row.sources} sources | {row.signals} signals</div>
              </div>
              <span className="h-2 w-2 shrink-0 rounded-full" style={{ background: CHART_COLORS[index % CHART_COLORS.length] }} />
            </div>
          ))}
        </div>
      </div>
    </Panel>
  )
}

function PublicPulsePanel({ cockpit }: { cockpit?: IntelCockpitResponse }) {
  const rows = cockpit?.social_pulse ?? []
  return (
    <Panel title="Public Pulse Sources" icon={Globe2} badge={`pulse ${cockpit?.live_state?.pulse_seq ?? 0}`}>
      <div className="space-y-2 p-3">
        {rows.map((row) => (
          <div key={row.label} className="rounded-md border border-border-subtle bg-bg-elevated/50 p-2">
            <div className="mb-1 flex items-center justify-between gap-2">
              <span className="truncate text-[10px] font-semibold text-text-primary">{row.label}</span>
              <span className="font-mono text-[9px] font-bold text-accent-primary">{row.mentions}</span>
            </div>
            <div className="h-1.5 overflow-hidden rounded-full bg-bg-overlay">
              <div
                className="h-full rounded-full bg-accent-primary transition-all duration-500"
                style={{ width: `${Math.round(row.velocity * 100)}%` }}
              />
            </div>
            <div className="mt-1 flex justify-between text-[8px] text-text-muted">
              <span className="truncate">{row.note}</span>
              <span className="font-mono">{pct(row.trust)} trust</span>
            </div>
          </div>
        ))}
      </div>
    </Panel>
  )
}

function IndiaIntelMap({ hotspots, links = [], freshnessSec }: {
  hotspots: IntelGeoHotspot[]
  links?: Array<{ source: string; target: string; weight: number; channel: string }>
  freshnessSec?: number
}) {
  const [mode, setMode] = useState<'risk' | 'velocity' | 'trust'>('risk')
  const [channel, setChannel] = useState('ALL')
  const project = (lat: number, lng: number) => ({
    x: Math.max(70, Math.min(505, ((lng - 67.2) / 25.2) * 500 + 42)),
    y: Math.max(48, Math.min(370, 386 - ((lat - 7.2) / 28.2) * 326)),
  })
  const channels = ['ALL', ...Array.from(new Set(hotspots.flatMap((item) => item.channels ?? []))).slice(0, 7)]
  const visible = hotspots.filter((item) => channel === 'ALL' || item.channels?.includes(channel))
  const byLabel = new Map(visible.map((item) => [item.label, item]))
  const valueOf = (item: IntelGeoHotspot) => (
    mode === 'velocity' ? item.velocity ?? item.risk ?? 0 :
      mode === 'trust' ? item.trust ?? item.risk ?? 0 :
        item.risk ?? item.weight ?? 0
  )
  const modeLabel = mode === 'risk' ? 'Risk' : mode === 'velocity' ? 'Velocity' : 'Trust'
  return (
    <Panel title="India Fraud Signal Map" icon={MapPin} badge={`${visible.length} live hotspots`} className="self-start">
      <div className="border-b border-border-subtle px-3 py-2">
        <div className="flex flex-wrap items-center justify-between gap-2">
          <div className="flex flex-wrap gap-1.5">
            {(['risk', 'velocity', 'trust'] as const).map((item) => (
              <button
                key={item}
                onClick={() => setMode(item)}
                className={cn(
                  'rounded-md border px-2 py-1 text-[8px] font-bold uppercase tracking-[0.12em]',
                  mode === item ? 'border-accent-primary bg-accent-primary text-white' : 'border-border-subtle bg-bg-elevated text-text-muted',
                )}
              >
                {item}
              </button>
            ))}
          </div>
          <select
            value={channel}
            onChange={(event) => setChannel(event.target.value)}
            className="h-7 rounded-md border border-border-subtle bg-bg-elevated px-2 text-[9px] font-semibold text-text-secondary"
          >
            {channels.map((item) => <option key={item} value={item}>{item}</option>)}
          </select>
        </div>
      </div>
      <div className="relative overflow-hidden p-3">
        <div className="pointer-events-none absolute left-4 top-4 z-10 rounded-md border border-border-subtle bg-bg-surface/90 px-2 py-1 font-mono text-[8px] text-text-muted">
          source freshness {ageLabel(freshnessSec)}
        </div>
        <svg viewBox="0 0 600 420" className="h-[500px] w-full rounded-lg border border-border-subtle bg-[#eef6ff]">
          <defs>
            <pattern id="india-grid" width="24" height="24" patternUnits="userSpaceOnUse">
              <path d="M24 0H0V24" fill="none" stroke="#0057a8" strokeOpacity=".07" />
            </pattern>
            <linearGradient id="india-fill" x1="0" x2="1" y1="0" y2="1">
              <stop offset="0" stopColor="#dff4ff" />
              <stop offset="1" stopColor="#ffffff" />
            </linearGradient>
          </defs>
          <rect width="600" height="420" fill="url(#india-grid)" />
          <path
            d="M250 32 L306 52 L351 95 L379 151 L430 187 L461 238 L434 307 L378 334 L328 373 L283 394 L250 346 L223 294 L182 256 L151 203 L164 134 Z"
            fill="url(#india-fill)"
            stroke="#0057a8"
            strokeWidth="2.5"
          />
          <path
            d="M266 45 L309 65 L342 100 L369 157 L414 191 L439 238 L407 289 L358 314 L314 349 L283 382"
            fill="none"
            stroke="#0057a8"
            strokeOpacity=".12"
            strokeWidth="10"
          />
          {links.map((link, index) => {
            const source = byLabel.get(link.source)
            const target = byLabel.get(link.target)
            if (!source || !target) return null
            const a = project(source.lat, source.lng)
            const b = project(target.lat, target.lng)
            const midX = (a.x + b.x) / 2
            const midY = Math.min(a.y, b.y) - 24 - index * 2
            return (
              <path
                key={`${link.source}-${link.target}`}
                d={`M ${a.x} ${a.y} Q ${midX} ${midY} ${b.x} ${b.y}`}
                fill="none"
                stroke={link.channel === 'IMPS' ? '#d71920' : '#0057a8'}
                strokeOpacity={0.18 + link.weight * 0.34}
                strokeWidth={1.5 + link.weight * 3}
              />
            )
          })}
          {visible.map((hotspot, index) => {
            const p = project(hotspot.lat, hotspot.lng)
            const metric = valueOf(hotspot)
            const radius = 5 + metric * 19
            const color = mode === 'trust' ? '#10b981' : mode === 'velocity' ? '#f59e0b' : CHART_COLORS[index % CHART_COLORS.length]
            return (
              <g key={hotspot.label}>
                <circle cx={p.x} cy={p.y} r={radius + 16} fill={color} opacity=".08" />
                <circle cx={p.x} cy={p.y} r={radius + 7} fill="none" stroke={color} strokeOpacity=".22" strokeWidth="2" />
                <circle cx={p.x} cy={p.y} r={radius} fill={color} opacity=".28" stroke={color} strokeWidth="1.5" />
                <circle cx={p.x} cy={p.y} r={Math.max(4, radius * 0.42)} fill={color} />
                <text x={p.x + radius + 5} y={p.y - 2} fill="#172033" fontSize="10" fontFamily="monospace" fontWeight="700">{hotspot.label}</text>
                <text x={p.x + radius + 5} y={p.y + 11} fill={color} fontSize="9" fontFamily="monospace">{modeLabel} {pct(metric)}</text>
              </g>
            )
          })}
        </svg>
        <div className="mt-3 grid grid-cols-2 gap-2 md:grid-cols-4">
          {visible.slice(0, 8).map((hotspot) => (
            <div key={hotspot.label} className="rounded-md border border-border-subtle bg-bg-elevated/85 p-2">
              <div className="truncate text-[9px] font-bold text-text-primary">{hotspot.label}</div>
              <div className="mt-1 font-mono text-[8px] text-text-muted">
                {hotspot.signals ?? 0} signals | {hotspot.primary_channel ?? 'UPI'} | {pct(valueOf(hotspot))}
              </div>
            </div>
          ))}
        </div>
      </div>
    </Panel>
  )
}

function TypologyHeatmap({ cockpit }: { cockpit?: IntelCockpitResponse }) {
  const rows = cockpit?.typology_matrix.slice(0, 7) ?? []
  return (
    <Panel title="Typology Evidence Matrix" icon={Layers}>
      <div className="space-y-2 p-3">
        {rows.map((row) => {
          const max = Math.max(row.official, row.news, row.social, row.open_web, 1)
          return (
            <div key={row.typology} className="grid grid-cols-[minmax(0,1.1fr)_repeat(4,42px)_48px] items-center gap-2 rounded-md border border-border-subtle bg-bg-elevated/45 p-2">
              <div className="truncate text-[9px] font-semibold text-text-primary" title={row.label}>{row.label}</div>
              {[row.official, row.news, row.social, row.open_web].map((value, index) => (
                <div key={index} className="h-6 rounded border border-border-subtle" style={{ backgroundColor: `${CHART_COLORS[index]}${Math.round(35 + (value / max) * 120).toString(16)}` }}>
                  <div className="flex h-full items-center justify-center font-mono text-[9px] font-bold text-text-primary">{value}</div>
                </div>
              ))}
              <div className="text-right font-mono text-[9px] text-emerald-300">{pct(row.trust)}</div>
            </div>
          )
        })}
      </div>
    </Panel>
  )
}

function SignalRadarList({ signals }: { signals: ExternalThreatSignal[] }) {
  return (
    <Panel title="Signal Evidence Stream" icon={Radio} badge={`${signals.length} signals`} className="h-full min-h-0">
      <div className="max-h-[520px] space-y-2 overflow-auto p-3">
        {signals.map((signal) => (
          <article key={signal.signal_id} className="grid grid-cols-[86px_minmax(0,1fr)] gap-3 rounded-md border border-border-subtle bg-bg-elevated/55 p-2">
            {mediaUrl(signal.media_preview) && !isFallbackMedia(signal.media_preview) ? (
              <img
                src={mediaUrl(signal.media_preview)}
                alt=""
                onError={(event) => { event.currentTarget.src = mediaSvgDataUri(signal.media_preview) }}
                className={cn('h-20 w-[86px] rounded-md border border-border-subtle bg-white object-cover', isSourceCard(signal.media_preview) && 'object-contain p-3')}
              />
            ) : (
              <div className="flex h-20 w-[86px] items-center justify-center rounded-md border border-border-subtle bg-white">
                <Newspaper className="h-6 w-6 text-accent-primary" />
              </div>
            )}
            <div className="min-w-0">
              <div className="mb-1 flex items-start justify-between gap-2">
                <h3 className="line-clamp-2 text-[11px] font-semibold text-text-primary">{signal.title}</h3>
                <span className="font-mono text-[10px] font-bold text-emerald-300">{pct(signal.trust_score)}</span>
              </div>
              <p className="line-clamp-2 text-[9px] leading-relaxed text-text-muted">{signal.normalized_text}</p>
              <div className="mt-2 flex flex-wrap gap-1.5">
                {signal.typologies.slice(0, 3).map((typology) => (
                  <span key={typology} className="rounded bg-bg-overlay px-2 py-0.5 text-[8px] font-semibold uppercase tracking-[0.08em] text-text-secondary">
                    {typology.replace(/_/g, ' ')}
                  </span>
                ))}
              </div>
            </div>
          </article>
        ))}
      </div>
    </Panel>
  )
}

function SourceLadder({ sources }: { sources: IntelSourceConfig[] }) {
  const grouped = ['tier_0', 'tier_1', 'tier_2', 'tier_3'].map((tier) => ({
    tier,
    sources: sources.filter((source) => source.tier === tier),
  }))
  return (
    <Panel title="Trusted Source Ladder" icon={ShieldCheck} badge={`${sources.length} sources`}>
      <div className="max-h-[185px] space-y-2 overflow-auto p-3 pr-2">
        {grouped.map((group) => (
          <div key={group.tier} className="rounded-md border border-border-subtle bg-bg-elevated/45 p-2">
            <div className="mb-2 flex items-center justify-between">
              <span className="text-[9px] font-bold uppercase tracking-[0.12em] text-text-primary">{group.tier.replace('_', ' ')}</span>
              <span className="font-mono text-[9px] text-text-muted">{group.sources.length}</span>
            </div>
            <div className="space-y-1.5">
              {group.sources.slice(0, 3).map((source) => (
                <a key={source.source_id} href={source.url.startsWith('http') ? source.url : undefined} target="_blank" rel="noreferrer" className="block min-w-0 rounded bg-bg-overlay px-2 py-1.5 hover:bg-bg-surface">
                  <div className="flex items-center justify-between gap-2">
                    <span className="truncate text-[9px] font-semibold text-text-secondary" title={source.name}>{source.name}</span>
                    {source.url.startsWith('http') && <ExternalLink className="h-3 w-3 shrink-0 text-text-muted" />}
                  </div>
                  <div className="mt-0.5 flex items-center justify-between gap-2 text-[8px] text-text-muted">
                    <span className="truncate">{source.category}</span>
                    <span className="font-mono">{timeLabel(source.last_polled_at)}</span>
                  </div>
                </a>
              ))}
            </div>
          </div>
        ))}
      </div>
    </Panel>
  )
}

function TrendCards({ trends, playbooks }: { trends: FraudTrendCluster[]; playbooks: AdaptivePlaybook[] }) {
  return (
    <Panel title="Trend Clusters And Playbooks" icon={BrainCircuit} badge={`${playbooks.length} playbooks`}>
      <div className="max-h-[250px] space-y-2 overflow-auto p-3 pr-2">
        {trends.slice(0, 4).map((trend) => {
          const playbook = playbooks.find((item) => item.trend_id === trend.trend_id)
          return (
            <article key={trend.trend_id} className="rounded-md border border-border-subtle bg-bg-elevated/55 p-3">
              <div className="mb-2 flex items-start justify-between gap-3">
                <div className="min-w-0">
                  <h3 className="line-clamp-2 text-[11px] font-semibold text-text-primary">{trend.title}</h3>
                  <div className="mt-1 font-mono text-[8px] text-text-muted">{trend.evidence_count} evidence items | {trend.source_tiers.join(', ')}</div>
                </div>
                <ToneBadge label={playbook?.promotion_status ?? 'cluster'} tone={playbook?.promotion_status === 'applied' ? 'emerald' : 'amber'} />
              </div>
              <div className="grid grid-cols-3 gap-2">
                <Metric label="Velocity" value={pct(trend.velocity_score)} accent="text-cyan-300" />
                <Metric label="India Fit" value={pct(trend.india_relevance_score)} accent="text-emerald-300" />
                <Metric label="Trust" value={pct(trend.trust_score)} accent="text-amber-300" />
              </div>
              <div className="mt-2 flex flex-wrap gap-1.5">
                {trend.affected_channels.slice(0, 4).map((channel) => (
                  <span key={channel} className="rounded bg-bg-overlay px-2 py-0.5 text-[8px] font-semibold uppercase tracking-[0.08em] text-text-secondary">{channel}</span>
                ))}
              </div>
            </article>
          )
        })}
      </div>
    </Panel>
  )
}

function FusionGraph({ cockpit }: { cockpit?: IntelCockpitResponse }) {
  const nodes = cockpit?.fusion_graph.nodes ?? []
  const links = cockpit?.fusion_graph.links ?? []
  const kinds = ['source', 'signal', 'trend'] as const
  const visibleNodes = kinds.flatMap((kind) => {
    const limit = kind === 'source' ? 7 : kind === 'signal' ? 9 : 6
    return nodes.filter((node) => node.kind === kind).slice(0, limit)
  })
  const columns: Record<string, number> = { source: 96, signal: 378, trend: 646 }
  const labels: Record<string, string> = { source: 'trusted sources', signal: 'external signals', trend: 'adaptive trends' }
  const positioned = visibleNodes.map((node, index) => {
    const tierItems = visibleNodes.filter((item) => item.kind === node.kind)
    const localIndex = tierItems.findIndex((item) => item.id === node.id)
    const spread = Math.min(42, 236 / Math.max(1, tierItems.length - 1))
    const yStart = tierItems.length > 6 ? 58 : 82
    return {
      ...node,
      x: columns[node.kind] ?? 378,
      y: yStart + localIndex * spread + (index % 2) * 3,
    }
  })
  const byId = new Map(positioned.map((node) => [node.id, node]))
  return (
    <Panel title="Source-To-Trend Fusion Graph" icon={Route} className="h-full min-h-0">
      <div className="h-[520px] p-3">
        <svg viewBox="0 0 760 360" className="h-full w-full rounded-md border border-border-subtle bg-bg-elevated/45">
          <defs>
            <pattern id="fusion-grid" width="28" height="28" patternUnits="userSpaceOnUse">
              <path d="M28 0H0V28" fill="none" stroke="#0057a8" strokeOpacity=".06" />
            </pattern>
          </defs>
          <rect width="760" height="360" fill="url(#fusion-grid)" />
          {kinds.map((kind) => (
            <g key={kind}>
              <line x1={columns[kind]} y1="42" x2={columns[kind]} y2="320" stroke="#0057a8" strokeOpacity=".14" strokeDasharray="4 8" />
              <text x={columns[kind]} y="28" textAnchor="middle" fill="#64748b" fontSize="9" fontFamily="monospace" fontWeight="700">
                {labels[kind].toUpperCase()}
              </text>
            </g>
          ))}
          {links.slice(0, 34).map((link, index) => {
            const source = byId.get(link.source)
            const target = byId.get(link.target)
            if (!source || !target) return null
            const mid = (source.x + target.x) / 2
            return (
              <path
                key={index}
                d={`M ${source.x} ${source.y} C ${mid} ${source.y}, ${mid} ${target.y}, ${target.x} ${target.y}`}
                fill="none"
                stroke={target.kind === 'trend' ? '#0057a8' : '#475569'}
                strokeWidth={1.1 + link.weight * 2.4}
                strokeOpacity={0.24 + Math.min(0.44, link.weight * 0.34)}
                strokeLinecap="round"
              />
            )
          })}
          {positioned.map((node, index) => (
            <g key={node.id}>
              <circle cx={node.x} cy={node.y} r={node.kind === 'trend' ? 10 : 7} fill={CHART_COLORS[index % CHART_COLORS.length]} opacity=".94" />
              <circle cx={node.x} cy={node.y} r={node.kind === 'trend' ? 18 : 13} fill={CHART_COLORS[index % CHART_COLORS.length]} opacity=".12" />
              <text
                x={node.kind === 'trend' ? node.x - 14 : node.x + 14}
                y={node.y + 3}
                textAnchor={node.kind === 'trend' ? 'end' : 'start'}
                fill="#172033"
                fontSize="9"
                fontFamily="monospace"
                fontWeight="700"
              >
                {node.label.slice(0, node.kind === 'signal' ? 32 : 26)}
              </text>
            </g>
          ))}
        </svg>
      </div>
    </Panel>
  )
}

function DemoConsole({
  demoScenario,
  setDemoScenario,
  busy,
  onRefresh,
  onDemo,
  onOpenOverview,
  lastCaseId,
}: {
  demoScenario: string
  setDemoScenario: (value: string) => void
  busy: boolean
  onRefresh: () => void
  onDemo: () => void
  onOpenOverview: () => void
  lastCaseId: string | null
}) {
  return (
    <Panel title="Judge Demo Console" icon={Crosshair}>
      <div className="space-y-3 p-3">
        <div className="grid grid-cols-2 gap-2">
          {DEMO_SCENARIOS.map((scenario) => (
            <button
              key={scenario.id}
              onClick={() => setDemoScenario(scenario.id)}
              className={cn(
                'rounded-md border p-2 text-left text-[10px] font-semibold transition-colors',
                demoScenario === scenario.id
                  ? 'border-accent-primary/60 bg-accent-primary/10 text-text-primary'
                  : 'border-border-subtle bg-bg-elevated/45 text-text-muted hover:border-border-default',
              )}
            >
              {scenario.label}
            </button>
          ))}
        </div>
        <div className="grid grid-cols-3 gap-2">
          <button onClick={onRefresh} disabled={busy} className="inline-flex items-center justify-center gap-1.5 rounded-md border border-border-default px-3 py-2 text-[9px] font-bold uppercase tracking-[0.12em] text-text-secondary transition-colors hover:border-accent-primary hover:text-accent-primary disabled:cursor-not-allowed disabled:opacity-50">
            <RefreshCw className="h-3.5 w-3.5" />
            Refresh
          </button>
          <button onClick={onDemo} disabled={busy} className="inline-flex items-center justify-center gap-1.5 rounded-md border border-accent-primary/60 px-3 py-2 text-[9px] font-bold uppercase tracking-[0.12em] text-accent-primary transition-colors hover:bg-accent-primary hover:text-bg-deep disabled:cursor-not-allowed disabled:opacity-50">
            <Play className="h-3.5 w-3.5" />
            Demo
          </button>
          <button onClick={onOpenOverview} className="inline-flex items-center justify-center gap-1.5 rounded-md border border-emerald-400/40 px-3 py-2 text-[9px] font-bold uppercase tracking-[0.12em] text-emerald-300 transition-colors hover:bg-emerald-400 hover:text-bg-deep">
            <ArrowRight className="h-3.5 w-3.5" />
            Graph
          </button>
        </div>
        {lastCaseId && (
          <div className="rounded-md border border-emerald-400/25 bg-emerald-500/10 p-2 text-[9px] text-emerald-300">
            Preventive signal active before PS3 case {lastCaseId}
          </div>
        )}
      </div>
    </Panel>
  )
}

export function PreFraudIntelPage() {
  const [demoScenario, setDemoScenario] = useState('digital_arrest_mule')
  const [lastCaseId, setLastCaseId] = useState<string | null>(null)
  const [autoRefreshRequested, setAutoRefreshRequested] = useState(false)
  const setActiveTab = useUIStore((s) => s.setActiveTab)
  const setActiveCaseId = useUIStore((s) => s.setActiveCaseId)
  const { data: sourcesData } = useIntelSources()
  const { data: signalsData } = useIntelSignals()
  const { data: trendsData } = useIntelTrends()
  const { data: playbooksData } = useIntelPlaybooks()
  const { data: tuningData } = useIntelTuningStatus()
  const { data: cockpit } = useIntelCockpit()
  const { data: mediaData } = useIntelMedia()
  const refresh = useRefreshIntel()
  const simulate = useSimulateIntelSignal()
  const launchPS3 = useLaunchPS3Scenario()

  const sources = sourcesData?.sources ?? []
  const signals = signalsData?.signals ?? []
  const trends = cockpit?.top_trends ?? trendsData?.trends ?? []
  const playbooks = cockpit?.active_playbooks ?? playbooksData?.playbooks ?? []
  const previews = mediaData?.media_previews ?? cockpit?.media_previews ?? signals.map((signal) => signal.media_preview).filter(Boolean)
  const busy = refresh.isPending || simulate.isPending || launchPS3.isPending

  useEffect(() => {
    const hasArticleMedia = previews.some((preview) => (
      isRealImage(preview) || isRealVideo(preview)
    ))
    const isStale = (cockpit?.metrics.freshness_sec ?? 0) > 900
    if (!autoRefreshRequested && !refresh.isPending && previews.length > 0 && (!hasArticleMedia || isStale)) {
      const id = window.setTimeout(() => {
        setAutoRefreshRequested(true)
        void refresh.mutateAsync(undefined)
      }, 800)
      return () => window.clearTimeout(id)
    }
    return undefined
  }, [autoRefreshRequested, cockpit?.metrics.freshness_sec, previews, refresh])

  const heroMetrics = useMemo(() => [
    { label: 'External Signals', value: String(cockpit?.metrics.signal_count ?? signals.length), accent: 'text-cyan-300' },
    { label: 'Active Sources', value: String(cockpit?.metrics.active_sources ?? sources.length), accent: 'text-emerald-300' },
    { label: 'Velocity Index', value: pct(cockpit?.metrics.velocity_index), accent: 'text-accent-primary' },
    { label: 'Trust Index', value: pct(cockpit?.metrics.trust_index), accent: 'text-amber-300' },
    { label: 'Corroborated', value: pct(cockpit?.metrics.corroboration_rate), accent: 'text-emerald-300' },
    { label: 'Map Coverage', value: pct(cockpit?.metrics.map_coverage), accent: 'text-accent-primary' },
    { label: 'Real Media', value: `${cockpit?.metrics.live_media_items ?? mediaData?.summary.live_media ?? 0}/${cockpit?.metrics.media_items ?? previews.length}`, accent: 'text-rose-300' },
    { label: 'Videos', value: String(cockpit?.metrics.real_videos ?? mediaData?.summary.real_videos ?? 0), accent: 'text-accent-primary' },
    { label: 'Media Health', value: pct(cockpit?.metrics.media_health ?? mediaData?.summary.health), accent: 'text-emerald-300' },
    { label: 'Active Playbooks', value: String(tuningData?.active_playbooks ?? cockpit?.metrics.active_playbooks ?? 0), accent: 'text-violet-300' },
  ], [cockpit, mediaData?.summary.health, mediaData?.summary.live_media, previews.length, signals.length, sources.length, tuningData?.active_playbooks])

  async function runPreventiveDemo() {
    await simulate.mutateAsync(demoScenario)
    const launched = await launchPS3.mutateAsync({ scenario: 'rapid_layering', intensity: 'demo', seed: 2026 })
    setActiveCaseId(launched.primary_case_id)
    setLastCaseId(launched.primary_case_id)
  }

  return (
    <div className="flex h-full flex-col bg-transparent">
      <div className="ubi-page-band shrink-0 border-b px-5 py-4">
        <div className="flex flex-wrap items-center justify-between gap-4">
          <div className="flex min-w-0 items-center gap-3">
            <div className="flex h-10 w-10 items-center justify-center rounded-lg border border-accent-primary/25 bg-bg-surface shadow-sm">
              <Radar className="h-5 w-5 text-accent-primary" />
            </div>
            <div className="min-w-0">
              <div className="flex items-center gap-2">
                <h1 className="truncate text-base font-bold tracking-wide text-text-primary">Union Bank Pre-Fraud Intelligence Desk</h1>
                <ToneBadge label="entry layer" tone="cyan" />
              </div>
              <p className="mt-0.5 truncate text-[10px] text-text-muted">
                OSINT/SOCMINT fusion before fund-flow detection | source media, India signal maps, playbook tuning, Qwen {tuningData?.qwen_model ?? 'qwen3.5:4b-q4_K_M'}
              </p>
            </div>
          </div>
          <div className="flex flex-wrap items-center gap-2">
            <ToneBadge label={`${cockpit?.live_state?.public_mode?.replace(/_/g, ' ') ?? 'live public pulse'}`} tone="emerald" />
            <ToneBadge label={`fresh ${ageLabel(cockpit?.metrics.freshness_sec)}`} tone="cyan" />
            <ToneBadge label="rollback ready" tone={tuningData?.rollback_available ? 'emerald' : 'slate'} />
            <button onClick={() => setActiveTab('overview')} className="inline-flex h-8 items-center gap-2 rounded-md border border-border-default px-3 text-[9px] font-bold uppercase tracking-[0.12em] text-text-secondary hover:border-accent-primary hover:text-accent-primary">
              Open Fund-Flow Overview
              <ArrowRight className="h-3.5 w-3.5" />
            </button>
          </div>
        </div>
        <div className="mt-3 grid grid-cols-2 gap-2 md:grid-cols-5 2xl:grid-cols-10">
          {heroMetrics.map((metric) => <Metric key={metric.label} {...metric} />)}
        </div>
      </div>

      <div className="min-h-0 flex-1 overflow-auto p-4">
        <div className="grid auto-rows-min grid-cols-12 items-start gap-3">
          <div className="col-span-12 2xl:col-span-8">
            <Panel title="Source Media Intelligence Board" icon={Newspaper} badge={`${previews.length} previews`}>
              <div className="p-3">
                <MediaCommandDeck previews={previews} />
              </div>
            </Panel>
          </div>

          <div className="col-span-12 grid gap-3 xl:grid-cols-2 2xl:col-span-4 2xl:grid-cols-1">
            <DemoConsole
              demoScenario={demoScenario}
              setDemoScenario={setDemoScenario}
              busy={busy}
              onRefresh={() => void refresh.mutateAsync(undefined)}
              onDemo={() => void runPreventiveDemo()}
              onOpenOverview={() => setActiveTab('overview')}
              lastCaseId={lastCaseId}
            />
            <TrendCards trends={trends} playbooks={playbooks} />
          </div>

          <div className="col-span-12 md:col-span-6 2xl:col-span-3">
            <LiveVelocityChart cockpit={cockpit} />
          </div>
          <div className="col-span-12 md:col-span-6 2xl:col-span-3">
            <ChannelExposureChart cockpit={cockpit} />
          </div>
          <div className="col-span-12 md:col-span-6 2xl:col-span-3">
            <SourceVelocityPanel cockpit={cockpit} />
          </div>
          <div className="col-span-12 md:col-span-6 2xl:col-span-3">
            <TypologyVelocityPanel cockpit={cockpit} />
          </div>
          <div className="col-span-12 2xl:col-span-7">
            <IndiaIntelMap
              hotspots={cockpit?.geo_hotspots ?? []}
              links={cockpit?.geo_links ?? []}
              freshnessSec={cockpit?.metrics.freshness_sec}
            />
          </div>
          <div className="col-span-12 grid items-start gap-3 md:grid-cols-2 2xl:col-span-5">
            <SourceMixPanel cockpit={cockpit} />
            <PublicPulsePanel cockpit={cockpit} />
            <MediaEvidenceMatrixPanel cockpit={cockpit} />
            <TypologyHeatmap cockpit={cockpit} />
          </div>

          <div className="col-span-12 grid items-start gap-3 lg:grid-cols-3">
            <SourceLadder sources={sources} />
            <Panel title="Runtime Guardrails" icon={ShieldCheck}>
              <div className="space-y-2 p-3">
                <div className="grid grid-cols-2 gap-2">
                  <Metric label="Queue" value={`${tuningData?.bounded_queue.depth ?? 0}/${tuningData?.bounded_queue.max_depth ?? 64}`} />
                  <Metric label="Shadow" value={String(tuningData?.shadow_changes ?? 0)} />
                  <Metric label="Advisory" value={String(tuningData?.advisory_changes ?? 0)} />
                  <Metric label="Rollback" value={tuningData?.rollback_available ? 'available' : 'none'} />
                </div>
                <div className="rounded-md border border-border-subtle bg-bg-elevated/45 p-2 text-[9px] leading-relaxed text-text-muted">
                  External signals tune watchlists, scenario seeds, and Qwen context only. Graph, ML, rules, ledger, and circuit-breaker evidence remain authoritative.
                </div>
              </div>
            </Panel>
            <Panel title="India Sovereignty Fit" icon={Globe2}>
              <div className="space-y-2 p-3 text-[9px] leading-relaxed text-text-muted">
                <div className="rounded-md border border-border-subtle bg-bg-elevated/45 p-2">Official Indian sources outrank public news and social chatter.</div>
                <div className="rounded-md border border-border-subtle bg-bg-elevated/45 p-2">No customer PII is exported; external feeds only shape preventive context.</div>
                <div className="rounded-md border border-border-subtle bg-bg-elevated/45 p-2">UPI, KYC, mule-account, Hindi/Hinglish, and diaspora signals receive higher relevance.</div>
                <div className="rounded-md border border-border-subtle bg-bg-elevated/45 p-2">Countermeasures stay analyst-gated; OSINT can prime playbooks, not force action.</div>
              </div>
            </Panel>
          </div>

          <div className="col-span-12">
            <ChannelTypologyHeatmapPanel cockpit={cockpit} />
          </div>
          <div className="col-span-12 xl:col-span-6 2xl:col-span-5">
            <SignalRadarList signals={signals} />
          </div>
          <div className="col-span-12 xl:col-span-6 2xl:col-span-7">
            <FusionGraph cockpit={cockpit} />
          </div>
        </div>
      </div>

      {busy && (
        <div className="pointer-events-none fixed bottom-5 left-1/2 z-50 -translate-x-1/2 rounded-md border border-accent-primary/30 bg-bg-overlay px-4 py-2 text-[10px] font-semibold uppercase tracking-[0.12em] text-accent-primary shadow-lg">
          <span className="inline-flex items-center gap-2">
            <BadgeCheck className="h-3.5 w-3.5" />
            Intelligence cycle running
          </span>
        </div>
      )}
    </div>
  )
}

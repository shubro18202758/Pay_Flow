// ============================================================================
// Pipeline Motion Visualizer — Animated real-time processing pipeline flow
// Shows events flowing through: Ingestion → ML → Graph → Circuit Breaker → AI → Verdict
// Driven entirely by real SSE events from backend — NOT hardcoded animations
// ============================================================================

import { useMemo, useState, useEffect, useRef, useCallback } from 'react'
import { useActivityStore, type EventLifecycle, type PipelineStage, type StageDetail } from '@/stores/use-activity-store'
import { cn } from '@/lib/utils'
import {
  ArrowRight,
  Database,
  BrainCircuit,
  Network,
  ShieldCheck,
  Bot,
  Scale,
  Sparkles,
  ChevronDown,
  ChevronUp,
  Radio,
  Zap,
  Clock,
  Activity,
  X,
} from 'lucide-react'

// ── Stage Constants ──────────────────────────────────────────────────────────

interface StageConfig {
  key: PipelineStage
  label: string
  shortLabel: string
  icon: typeof Database
  description: string
  color: string       // oklch color for active state
  glowColor: string   // glow shadow color
}

const PIPELINE_STAGES: StageConfig[] = [
  {
    key: 'ingested',
    label: 'Event Ingestion',
    shortLabel: 'IN',
    icon: Database,
    description: 'Schema validation + CRC32 checksum verification',
    color: 'oklch(0.65 0.18 200)',
    glowColor: 'rgba(56, 189, 248, 0.4)',
  },
  {
    key: 'ml_scored',
    label: 'ML Feature Engine',
    shortLabel: 'ML',
    icon: BrainCircuit,
    description: '36-dimensional feature extraction + XGBoost fraud scoring',
    color: 'oklch(0.65 0.18 280)',
    glowColor: 'rgba(168, 85, 247, 0.4)',
  },
  {
    key: 'graph_investigated',
    label: 'Graph Analysis',
    shortLabel: 'GR',
    icon: Network,
    description: 'NetworkX pattern scan — mule detection, cycle detection, centrality',
    color: 'oklch(0.65 0.18 150)',
    glowColor: 'rgba(34, 197, 94, 0.4)',
  },
  {
    key: 'cb_evaluated',
    label: 'Circuit Breaker',
    shortLabel: 'CB',
    icon: ShieldCheck,
    description: 'Multi-model consensus — ML + GNN + structural evidence scoring',
    color: 'oklch(0.65 0.18 50)',
    glowColor: 'rgba(251, 191, 36, 0.4)',
  },
  {
    key: 'llm_started',
    label: 'Qwen AI Agent',
    shortLabel: 'AI',
    icon: Bot,
    description: 'LangGraph forensic investigation with Qwen 3.5 (Ollama)',
    color: 'oklch(0.65 0.18 25)',
    glowColor: 'rgba(251, 113, 133, 0.4)',
  },
  {
    key: 'verdict',
    label: 'Final Verdict',
    shortLabel: 'VD',
    icon: Scale,
    description: 'Classification verdict + HITL escalation decision + blockchain anchor',
    color: 'oklch(0.55 0.18 250)',
    glowColor: 'rgba(96, 165, 250, 0.4)',
  },
]

// ── Helper: compute stage status ──────────────────────────────────────────────

type StageStatus = 'idle' | 'active' | 'complete' | 'error'

function getStageStatuses(lifecycle: EventLifecycle | undefined): Map<PipelineStage, StageStatus> {
  const map = new Map<PipelineStage, StageStatus>()
  const completedStages = new Set(lifecycle?.stages.map((s) => s.stage) ?? [])

  let lastCompleteIdx = -1
  for (let i = PIPELINE_STAGES.length - 1; i >= 0; i--) {
    if (completedStages.has(PIPELINE_STAGES[i].key)) {
      lastCompleteIdx = i
      break
    }
  }

  for (let i = 0; i < PIPELINE_STAGES.length; i++) {
    const stage = PIPELINE_STAGES[i]
    if (completedStages.has(stage.key)) {
      map.set(stage.key, 'complete')
    } else if (i === lastCompleteIdx + 1 && lifecycle) {
      map.set(stage.key, 'active')
    } else {
      map.set(stage.key, 'idle')
    }
  }
  return map
}

function getStageDuration(lifecycle: EventLifecycle | undefined, stageKey: PipelineStage): string | null {
  if (!lifecycle) return null
  const detail = lifecycle.stages.find((s) => s.stage === stageKey)
  if (!detail?.durationMs) {
    // Calculate from stage timestamps
    const idx = lifecycle.stages.findIndex((s) => s.stage === stageKey)
    if (idx > 0) {
      const delta = lifecycle.stages[idx].timestamp - lifecycle.stages[idx - 1].timestamp
      if (delta < 1) return `${(delta * 1000).toFixed(0)}ms`
      return `${delta.toFixed(1)}s`
    }
    return null
  }
  return detail.durationMs < 1000 ? `${detail.durationMs.toFixed(0)}ms` : `${(detail.durationMs / 1000).toFixed(1)}s`
}

// ── Flowing Particles (CSS-driven animated dots along connectors) ─────────

function FlowingParticles({ active }: { active: boolean }) {
  if (!active) return null
  return (
    <div className="absolute inset-0 overflow-hidden pointer-events-none">
      {[0, 1, 2].map((i) => (
        <div
          key={i}
          className="absolute top-1/2 -translate-y-1/2 w-1.5 h-1.5 rounded-full bg-accent-primary"
          style={{
            animation: `pipeline-flow 1.5s ease-in-out infinite`,
            animationDelay: `${i * 0.5}s`,
            filter: 'blur(0.5px)',
            boxShadow: '0 0 6px 2px oklch(0.55 0.14 250 / 0.5)',
          }}
        />
      ))}
    </div>
  )
}

// ── Stage Node (individual pipeline stage card) ────────────────────────────

function StageNode({
  config,
  status,
  duration,
  stageDetail,
  isLast,
  onClick,
  selected,
}: {
  config: StageConfig
  status: StageStatus
  duration: string | null
  stageDetail?: StageDetail
  isLast: boolean
  onClick?: () => void
  selected?: boolean
}) {
  const Icon = config.icon

  return (
    <div className="flex items-center flex-1 min-w-0">
      {/* Stage card */}
      <button
        onClick={onClick}
        className={cn(
          'relative flex flex-col items-center gap-1.5 px-3 py-3 rounded-lg border transition-all duration-500 w-full min-w-0 cursor-pointer group',
          status === 'complete' && 'bg-gradient-to-b border-green-500/30',
          status === 'active' && 'border-accent-primary/60',
          status === 'idle' && 'bg-bg-overlay/60 border-border-subtle/70',
          status === 'error' && 'bg-red-500/5 border-red-500/30',
          selected && 'ring-1 ring-accent-primary/50',
        )}
        style={
          status === 'complete'
            ? { background: `linear-gradient(to bottom, ${config.color}15, transparent)`, boxShadow: `0 0 12px ${config.glowColor}` }
            : status === 'active'
              ? { boxShadow: `0 0 20px ${config.glowColor}, 0 0 40px ${config.glowColor}` }
              : undefined
        }
      >
        {/* Active processing ring */}
        {status === 'active' && (
          <>
            <div
              className="absolute inset-0 rounded-lg opacity-60"
              style={{
                background: `conic-gradient(from 0deg, transparent, ${config.color}, transparent)`,
                animation: 'pipeline-stage-spin 2s linear infinite',
                maskImage: 'radial-gradient(transparent 60%, black 62%, black 100%)',
                WebkitMaskImage: 'radial-gradient(transparent 60%, black 62%, black 100%)',
              }}
            />
            <div className="absolute inset-[1px] rounded-lg bg-bg-elevated" />
          </>
        )}

        {/* Icon */}
        <div
          className={cn(
            'relative z-10 w-8 h-8 rounded-lg flex items-center justify-center transition-all duration-500',
            status === 'complete' && 'text-white',
            status === 'active' && 'text-white',
            status === 'idle' && 'bg-bg-deep/80 text-text-muted/70',
            status === 'error' && 'bg-red-500/20 text-red-400',
          )}
          style={
            status === 'complete'
              ? { background: config.color }
              : status === 'active'
                ? { background: `${config.color}BB`, animation: 'pipeline-icon-pulse 1.5s ease-in-out infinite' }
                : undefined
          }
        >
          <Icon className="w-4 h-4" />
        </div>

        {/* Label */}
        <span
          className={cn(
            'relative z-10 text-[9px] font-bold uppercase tracking-[0.1em] text-center leading-tight transition-colors duration-300',
            status === 'complete' && 'text-text-primary',
            status === 'active' && 'text-accent-primary',
            status === 'idle' && 'text-text-muted/70',
          )}
        >
          {config.label}
        </span>

        {/* Duration / status badge */}
        <div className="relative z-10 h-4 flex items-center">
          {status === 'complete' && duration && (
            <span
              className="text-[8px] font-mono tabular-nums font-bold px-1.5 py-0.5 rounded-full"
              style={{ color: config.color, background: `${config.color}15` }}
            >
              {duration}
            </span>
          )}
          {status === 'active' && (
            <div className="flex items-center gap-1">
              <div
                className="w-1.5 h-1.5 rounded-full"
                style={{ background: config.color, animation: 'pipeline-dot-pulse 1s ease-in-out infinite' }}
              />
              <span className="text-[7px] uppercase tracking-wider text-accent-primary/80 font-semibold">
                Processing
              </span>
            </div>
          )}
          {status === 'idle' && (
            <span className="text-[7px] text-text-muted/50 uppercase tracking-wider">Pending</span>
          )}
        </div>

        {/* Hover description */}
        <div className="absolute -bottom-8 left-1/2 -translate-x-1/2 opacity-0 group-hover:opacity-100 transition-opacity z-30 pointer-events-none">
          <div className="px-2 py-1 rounded bg-bg-deep border border-border-default shadow-lg whitespace-nowrap">
            <span className="text-[7px] text-text-secondary">{config.description}</span>
          </div>
        </div>
      </button>

      {/* Connector arrow */}
      {!isLast && (
        <div className="relative w-8 h-px shrink-0 mx-0.5">
          <div
            className={cn(
              'absolute inset-y-0 left-0 right-0 transition-all duration-500',
              status === 'complete' ? 'bg-green-500/50' : 'bg-border-subtle/50',
            )}
            style={status === 'complete' ? { boxShadow: `0 0 8px ${config.glowColor}` } : undefined}
          />
          <FlowingParticles active={status === 'active'} />
          <ArrowRight
            className={cn(
              'absolute -right-1.5 top-1/2 -translate-y-1/2 w-3 h-3 transition-colors duration-500',
              status === 'complete' ? 'text-green-400' : 'text-border-subtle/70',
            )}
          />
        </div>
      )}
    </div>
  )
}

// ── Event Particle Trail (animated dots showing recent events flowing) ────

function EventParticleTrail({ events, orderedIds }: { events: Map<string, EventLifecycle>; orderedIds: string[] }) {
  const recentActive = useMemo(() => {
    const active: { id: string; stageIdx: number; fraud: boolean }[] = []
    for (const id of orderedIds.slice(0, 30)) {
      const lc = events.get(id)
      if (!lc) continue
      const completedStages = new Set(lc.stages.map((s) => s.stage))
      let lastIdx = -1
      for (let i = PIPELINE_STAGES.length - 1; i >= 0; i--) {
        if (completedStages.has(PIPELINE_STAGES[i].key)) { lastIdx = i; break }
      }
      if (lastIdx >= 0 && lastIdx < PIPELINE_STAGES.length - 1) {
        active.push({ id, stageIdx: lastIdx, fraud: lc.fraudLabel > 0 })
      }
    }
    return active.slice(0, 8)
  }, [events, orderedIds])

  if (recentActive.length === 0) return null

  return (
    <div className="absolute inset-0 pointer-events-none z-20">
      {recentActive.map((ev, i) => {
        const pct = ((ev.stageIdx + 0.5) / PIPELINE_STAGES.length) * 100
        return (
          <div
            key={ev.id}
            className="absolute top-0 bottom-0 flex items-center"
            style={{
              left: `${pct}%`,
              animation: `pipeline-particle-drift 3s ease-in-out infinite`,
              animationDelay: `${i * 0.4}s`,
            }}
          >
            <div
              className={cn(
                'w-2 h-2 rounded-full',
                ev.fraud ? 'bg-red-400' : 'bg-accent-primary',
              )}
              style={{
                boxShadow: ev.fraud
                  ? '0 0 8px 3px rgba(248, 113, 113, 0.5)'
                  : '0 0 8px 3px oklch(0.55 0.14 250 / 0.5)',
                animation: 'pipeline-dot-pulse 1.2s ease-in-out infinite',
              }}
            />
          </div>
        )
      })}
    </div>
  )
}

// ── Stage Detail Panel (expanded info for selected stage) ─────────────────

function StageDetailPanel({
  lifecycle,
  stageKey,
  onClose,
}: {
  lifecycle: EventLifecycle
  stageKey: PipelineStage
  onClose: () => void
}) {
  const stageConfig = PIPELINE_STAGES.find((s) => s.key === stageKey)
  const stageDetail = lifecycle.stages.find((s) => s.stage === stageKey)
  if (!stageConfig) return null
  const Icon = stageConfig.icon

  return (
    <div className="animate-fade-in bg-bg-overlay/80 border border-border-default rounded-lg p-3 mt-2 backdrop-blur-sm">
      <div className="flex items-center justify-between mb-2">
        <div className="flex items-center gap-2">
          <div
            className="w-5 h-5 rounded flex items-center justify-center text-white shadow-sm"
            style={{ background: stageConfig.color }}
          >
            <Icon className="w-3 h-3" />
          </div>
          <span className="text-[10px] font-bold text-text-primary uppercase tracking-wider">
            {stageConfig.label}
          </span>
        </div>
        <button onClick={onClose} className="text-text-muted hover:text-text-primary">
          <X className="w-3.5 h-3.5" />
        </button>
      </div>

      <p className="text-[8px] text-text-secondary mb-2">{stageConfig.description}</p>

      {stageDetail && (
        <div className="space-y-1.5">
          <div className="flex items-center gap-2 text-[8px]">
            <Clock className="w-2.5 h-2.5 text-text-muted" />
            <span className="text-text-muted">Timestamp:</span>
            <span className="font-mono text-text-secondary">
              {new Date(stageDetail.timestamp * 1000).toLocaleTimeString()}
            </span>
          </div>
          {stageDetail.durationMs != null && (
            <div className="flex items-center gap-2 text-[8px]">
              <Zap className="w-2.5 h-2.5 text-text-muted" />
              <span className="text-text-muted">Duration:</span>
              <span className="font-mono text-accent-primary font-bold">{stageDetail.durationMs}ms</span>
            </div>
          )}
          {stageDetail.meta && Object.keys(stageDetail.meta).length > 0 && (
            <div className="mt-1 pt-1 border-t border-border-subtle/50 space-y-0.5">
              {Object.entries(stageDetail.meta).slice(0, 6).map(([k, v]) => (
                <div key={k} className="flex justify-between text-[7px]">
                  <span className="text-text-muted uppercase tracking-wider">{k.replace(/_/g, ' ')}</span>
                  <span className="font-mono text-text-secondary truncate max-w-[140px]">{String(v)}</span>
                </div>
              ))}
            </div>
          )}

          {/* Stage-specific enrichment */}
          {stageKey === 'ml_scored' && lifecycle.riskScore != null && (
            <div className="flex items-center gap-2 text-[8px] pt-1 border-t border-border-subtle/50">
              <Activity className="w-2.5 h-2.5 text-amber-400" />
              <span className="text-text-muted">Risk Score:</span>
              <span className={cn(
                'font-mono font-bold',
                lifecycle.riskScore > 0.7 ? 'text-red-400' : lifecycle.riskScore > 0.4 ? 'text-amber-400' : 'text-green-400',
              )}>
                {(lifecycle.riskScore * 100).toFixed(1)}%
              </span>
              {lifecycle.riskTier && (
                <span className="text-[7px] px-1 py-0.5 rounded bg-bg-deep text-text-muted uppercase">{lifecycle.riskTier}</span>
              )}
            </div>
          )}
          {stageKey === 'verdict' && lifecycle.verdict && (
            <div className="space-y-1 pt-1 border-t border-border-subtle/50">
              <div className="flex items-center gap-2 text-[8px]">
                <Scale className="w-2.5 h-2.5 text-accent-primary" />
                <span className="text-text-muted">Verdict:</span>
                <span className={cn(
                  'font-bold uppercase text-[8px]',
                  lifecycle.verdict === 'fraudulent' ? 'text-red-400' : lifecycle.verdict === 'suspicious' ? 'text-amber-400' : 'text-green-400',
                )}>
                  {lifecycle.verdict}
                </span>
                {lifecycle.confidence != null && (
                  <span className="font-mono text-text-secondary">({(lifecycle.confidence * 100).toFixed(0)}%)</span>
                )}
              </div>
              {lifecycle.fraudTypology && (
                <div className="text-[7px] text-text-muted">
                  Typology: <span className="text-text-secondary">{lifecycle.fraudTypology}</span>
                </div>
              )}
            </div>
          )}
          {stageKey === 'cb_evaluated' && lifecycle.consensusScores && (
            <div className="flex gap-3 pt-1 border-t border-border-subtle/50">
              {Object.entries(lifecycle.consensusScores).map(([model, score]) => (
                <div key={model} className="text-[7px]">
                  <span className="text-text-muted uppercase">{model}:</span>{' '}
                  <span className={cn(
                    'font-mono font-bold',
                    score > 0.7 ? 'text-red-400' : score > 0.4 ? 'text-amber-400' : 'text-green-400',
                  )}>
                    {(score * 100).toFixed(0)}%
                  </span>
                </div>
              ))}
            </div>
          )}
        </div>
      )}
      {!stageDetail && (
        <div className="text-[8px] text-text-muted/50 italic">Stage has not been reached yet</div>
      )}
    </div>
  )
}

// ── Global Pipeline Stats Bar ─────────────────────────────────────────────

function PipelineGlobalStats({ events, orderedIds }: { events: Map<string, EventLifecycle>; orderedIds: string[] }) {
  const stats = useMemo(() => {
    let processing = 0
    let completed = 0
    let fraudulent = 0

    for (const id of orderedIds.slice(0, 50)) {
      const lc = events.get(id)
      if (!lc) continue
      const stages = new Set(lc.stages.map((s) => s.stage))
      if (stages.has('verdict')) {
        completed++
        if (lc.verdict === 'fraudulent') fraudulent++
      } else if (stages.size > 0) {
        processing++
      }
    }

    return { processing, completed, fraudulent, total: orderedIds.length }
  }, [events, orderedIds])

  return (
    <div className="flex items-center gap-4 text-[8px]">
      <div className="flex items-center gap-1">
        <div className="w-1.5 h-1.5 rounded-full bg-accent-primary animate-pulse" />
        <span className="text-text-muted">In Pipeline:</span>
        <span className="font-mono font-bold text-accent-primary">{stats.processing}</span>
      </div>
      <div className="flex items-center gap-1">
        <div className="w-1.5 h-1.5 rounded-full bg-green-400" />
        <span className="text-text-muted">Completed:</span>
        <span className="font-mono font-bold text-green-400">{stats.completed}</span>
      </div>
      <div className="flex items-center gap-1">
        <div className="w-1.5 h-1.5 rounded-full bg-red-400" />
        <span className="text-text-muted">Flagged:</span>
        <span className="font-mono font-bold text-red-400">{stats.fraudulent}</span>
      </div>
      <div className="flex items-center gap-1 ml-auto">
        <span className="text-text-muted">Total tracked:</span>
        <span className="font-mono text-text-secondary">{stats.total}</span>
      </div>
    </div>
  )
}

// ── Event Selector (choose which event to track) ──────────────────────────

function EventSelector({
  events,
  orderedIds,
  selectedId,
  onSelect,
}: {
  events: Map<string, EventLifecycle>
  orderedIds: string[]
  selectedId: string | null
  onSelect: (id: string | null) => void
}) {
  const recentEvents = useMemo(() => {
    return orderedIds.slice(0, 12).map((id) => {
      const lc = events.get(id)!
      const stagesCompleted = lc.stages.length
      const totalStages = PIPELINE_STAGES.length
      return { id, lifecycle: lc, stagesCompleted, totalStages }
    }).filter((e) => e.lifecycle)
  }, [events, orderedIds])

  if (recentEvents.length === 0) {
    return (
      <div className="text-[9px] text-text-muted/50 italic py-2 text-center">
        No events in pipeline yet — inject a custom event or launch an attack
      </div>
    )
  }

  return (
    <div className="space-y-0.5 max-h-40 overflow-y-auto pr-1">
      {recentEvents.map((ev) => {
        const isSelected = ev.id === selectedId
        const hasVerdict = ev.lifecycle.stages.some((s) => s.stage === 'verdict')
        const progressPct = (ev.stagesCompleted / ev.totalStages) * 100

        return (
          <button
            key={ev.id}
            onClick={() => onSelect(isSelected ? null : ev.id)}
            className={cn(
              'w-full flex items-center gap-2 px-2 py-1.5 rounded-md text-left transition-all',
              isSelected
                ? 'bg-accent-primary/10 border border-accent-primary/30'
                : 'hover:bg-bg-overlay/50 border border-transparent',
            )}
          >
            {/* Progress ring */}
            <div className="relative w-5 h-5 shrink-0">
              <svg className="w-5 h-5 -rotate-90" viewBox="0 0 20 20">
                <circle cx="10" cy="10" r="8" fill="none" stroke="oklch(0.2 0.02 250)" strokeWidth="2" />
                <circle
                  cx="10" cy="10" r="8" fill="none"
                  stroke={hasVerdict ? '#22c55e' : 'oklch(0.55 0.14 250)'}
                  strokeWidth="2"
                  strokeDasharray={`${progressPct * 0.503} 50.3`}
                  strokeLinecap="round"
                  className="transition-all duration-500"
                />
              </svg>
              {!hasVerdict && ev.stagesCompleted > 0 && (
                <div className="absolute inset-0 flex items-center justify-center">
                  <div className="w-1 h-1 rounded-full bg-accent-primary animate-pulse" />
                </div>
              )}
            </div>

            {/* Event info */}
            <div className="flex-1 min-w-0 flex items-center gap-2">
              <code className="text-[8px] font-mono text-text-secondary truncate max-w-[100px]">{ev.id}</code>
              {ev.lifecycle.fraudLabel > 0 && (
                <span className="px-1 py-0.5 text-[6px] rounded bg-red-500/15 text-red-400 font-bold uppercase">Fraud</span>
              )}
              {ev.lifecycle.sender && (
                <span className="text-[7px] text-text-muted truncate">{ev.lifecycle.sender} → {ev.lifecycle.receiver}</span>
              )}
            </div>

            {/* Stage count */}
            <span className="text-[8px] font-mono text-text-muted shrink-0">
              {ev.stagesCompleted}/{ev.totalStages}
            </span>
          </button>
        )
      })}
    </div>
  )
}

// ── Main Component ──────────────────────────────────────────────────────────

interface PipelineMotionVisualizerProps {
  /** Specific event ID to track (e.g., from custom event injection) */
  trackedEventId?: string | null
  /** Compact mode — no event picker, just pipeline flow */
  compact?: boolean
  /** Additional CSS class */
  className?: string
}

export function PipelineMotionVisualizer({ trackedEventId, compact, className }: PipelineMotionVisualizerProps) {
  const events = useActivityStore((s) => s.events)
  const orderedIds = useActivityStore((s) => s.orderedIds)
  const [selectedId, setSelectedId] = useState<string | null>(null)
  const [selectedStage, setSelectedStage] = useState<PipelineStage | null>(null)
  const [expanded, setExpanded] = useState(!compact)
  const prevStagesRef = useRef<number>(0)

  // Use tracked event if provided, otherwise use selected
  const activeId = trackedEventId ?? selectedId
  const lifecycle = activeId ? events.get(activeId) : undefined

  // Auto-select most recent active event if nothing selected
  useEffect(() => {
    if (trackedEventId) return // External tracking takes priority
    if (selectedId && events.has(selectedId)) return // Already tracking valid event

    // Find the most recent event that's still being processed
    for (const id of orderedIds.slice(0, 20)) {
      const lc = events.get(id)
      if (!lc) continue
      const hasVerdict = lc.stages.some((s) => s.stage === 'verdict')
      if (!hasVerdict && lc.stages.length > 0) {
        setSelectedId(id)
        return
      }
    }
  }, [trackedEventId, selectedId, events, orderedIds])

  // Detect stage changes for animation triggers
  const currentStages = lifecycle?.stages.length ?? 0
  useEffect(() => {
    if (currentStages > prevStagesRef.current) {
      // A new stage completed — could trigger a celebration animation
      prevStagesRef.current = currentStages
    }
  }, [currentStages])

  // Auto-update tracked event from prop changes
  useEffect(() => {
    if (trackedEventId) {
      setSelectedId(null)
      setSelectedStage(null)
    }
  }, [trackedEventId])

  const stageStatuses = useMemo(() => getStageStatuses(lifecycle), [lifecycle])
  const completionPct = lifecycle
    ? (lifecycle.stages.filter((s) => PIPELINE_STAGES.some((ps) => ps.key === s.stage)).length / PIPELINE_STAGES.length) * 100
    : 0

  return (
    <div className={cn('bg-bg-elevated/95 border border-border-default rounded-lg overflow-hidden backdrop-blur-sm', className)}>
      {/* Header */}
      <div className="px-4 py-3 border-b border-border-subtle flex items-center justify-between">
        <div className="flex items-center gap-2">
          <div
            className="w-7 h-7 rounded-lg border flex items-center justify-center"
            style={{
              background: lifecycle ? 'oklch(0.55 0.14 250 / 0.1)' : 'oklch(0.15 0.02 250)',
              borderColor: lifecycle ? 'oklch(0.55 0.14 250 / 0.2)' : 'oklch(0.22 0.02 250)',
            }}
          >
            {lifecycle ? (
              <Radio className="w-3.5 h-3.5 text-accent-primary animate-pulse" />
            ) : (
              <Sparkles className="w-3.5 h-3.5 text-text-muted/50" />
            )}
          </div>
          <div>
            <h3 className="text-[11px] font-bold text-text-primary uppercase tracking-[0.12em]">
              Live Processing Pipeline
            </h3>
            <p className="text-[8px] text-text-muted mt-0.5">
              {lifecycle
                ? `Tracking: ${activeId?.slice(0, 16)} — ${lifecycle.stages.length}/${PIPELINE_STAGES.length} stages complete`
                : 'Real-time event flow visualization — select an event or inject one to track'
              }
            </p>
          </div>
        </div>
        <div className="flex items-center gap-2">
          {lifecycle && (
            <div className="flex items-center gap-1.5">
              {/* Completion progress */}
              <div className="w-16 h-1.5 bg-bg-deep rounded-full overflow-hidden">
                <div
                  className="h-full rounded-full transition-all duration-700 ease-out"
                  style={{
                    width: `${completionPct}%`,
                    background: completionPct === 100
                      ? 'linear-gradient(90deg, #22c55e, #4ade80)'
                      : 'linear-gradient(90deg, oklch(0.55 0.14 250), oklch(0.65 0.18 250))',
                  }}
                />
              </div>
              <span className="text-[8px] font-mono font-bold text-text-secondary">
                {completionPct.toFixed(0)}%
              </span>
            </div>
          )}
          {!compact && (
            <button
              onClick={() => setExpanded(!expanded)}
              className="text-text-muted hover:text-text-primary transition-colors"
            >
              {expanded ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
            </button>
          )}
        </div>
      </div>

      {/* Main pipeline flow */}
      <div className="relative p-4">
        <PipelineGlobalStats events={events} orderedIds={orderedIds} />

        <div className="relative flex items-center gap-0 mt-3">
          <EventParticleTrail events={events} orderedIds={orderedIds} />
          {PIPELINE_STAGES.map((stage, i) => (
            <StageNode
              key={stage.key}
              config={stage}
              status={stageStatuses.get(stage.key) ?? 'idle'}
              duration={getStageDuration(lifecycle, stage.key)}
              stageDetail={lifecycle?.stages.find((s) => s.stage === stage.key)}
              isLast={i === PIPELINE_STAGES.length - 1}
              onClick={() => setSelectedStage(selectedStage === stage.key ? null : stage.key)}
              selected={selectedStage === stage.key}
            />
          ))}
        </div>

        {/* Expanded stage detail */}
        {selectedStage && lifecycle && (
          <StageDetailPanel
            lifecycle={lifecycle}
            stageKey={selectedStage}
            onClose={() => setSelectedStage(null)}
          />
        )}
      </div>

      {/* Event picker panel (when expanded) */}
      {expanded && !compact && (
        <div className="border-t border-border-subtle px-4 py-3">
          <div className="flex items-center justify-between mb-2">
            <span className="text-[9px] font-semibold text-text-muted uppercase tracking-wider">
              Recent Pipeline Events
            </span>
            {activeId && (
              <button
                onClick={() => { setSelectedId(null); setSelectedStage(null) }}
                className="text-[8px] text-text-muted hover:text-accent-primary transition-colors uppercase tracking-wider"
              >
                Clear Selection
              </button>
            )}
          </div>
          <EventSelector
            events={events}
            orderedIds={orderedIds}
            selectedId={activeId ?? null}
            onSelect={(id) => { setSelectedId(id); setSelectedStage(null) }}
          />
        </div>
      )}
    </div>
  )
}

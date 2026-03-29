// ============================================================================
// Pipeline Transparency Panel — Full "X-ray" of every pipeline stage
// No black box: Shows ML features, Graph analysis, CB consensus, LLM reasoning
// ============================================================================

import { useMemo, useState } from 'react'
import { useActivityStore, type EventLifecycle, type PipelineStage } from '@/stores/use-activity-store'
import { cn } from '@/lib/utils'
import {
  Eye,
  Database,
  BrainCircuit,
  Network,
  ShieldCheck,
  Bot,
  Scale,
  ChevronDown,
  ChevronRight,
  Activity,
  BarChart3,
  GitBranch,
  Shield,
  Cpu,
  MessageSquare,
  FileText,
  AlertTriangle,
  CheckCircle2,
  Zap,
  TrendingUp,
  Layers,
  Wrench,
  Brain,
  Target,
  type LucideIcon,
} from 'lucide-react'

// ── Stage transparency configs ───────────────────────────────────────────

interface TransparencyStageConfig {
  key: PipelineStage
  label: string
  icon: LucideIcon
  color: string
  algorithmName: string
  algorithmDetail: string
  techStack: string[]
}

const TRANSPARENCY_STAGES: TransparencyStageConfig[] = [
  {
    key: 'ingested',
    label: 'Event Ingestion & Validation',
    icon: Database,
    color: 'oklch(0.65 0.18 200)',
    algorithmName: 'Schema Validator + CRC32',
    algorithmDetail: 'Pydantic schema validation with strict field typing, CRC32 checksum integrity verification, timestamp normalization, and amount-to-paisa conversion.',
    techStack: ['Pydantic V2', 'CRC32 Checksum', 'Field Normalizer'],
  },
  {
    key: 'ml_scored',
    label: 'ML Feature Engine + XGBoost',
    icon: BrainCircuit,
    color: 'oklch(0.65 0.18 280)',
    algorithmName: '36-Dim Feature Extraction → XGBoost Classifier',
    algorithmDetail: 'Extracts 36 behavioral & transactional features: velocity (1h/24h/7d), amount z-score, time-of-day encoding, sender/receiver historical patterns, device entropy, geo-distance anomaly, channel frequency deviation. Scored by XGBoost gradient-boosted trees trained on synthetic fraud corpus.',
    techStack: ['XGBoost', '36-Feature Vector', 'Velocity Engine', 'Z-Score Normalization'],
  },
  {
    key: 'graph_investigated',
    label: 'Graph Neural Analysis',
    icon: Network,
    color: 'oklch(0.65 0.18 150)',
    algorithmName: 'NetworkX Structural Pattern Scanner',
    algorithmDetail: 'Builds directed transaction graph. Detects mule chains (DFS path analysis), circular laundering (cycle detection), hub anomalies (betweenness/closeness centrality), community outliers (Louvain), and star pattern detection for collector nodes.',
    techStack: ['NetworkX', 'Cycle Detection', 'Betweenness Centrality', 'Louvain Communities', 'DFS Mule Chains'],
  },
  {
    key: 'cb_evaluated',
    label: 'Circuit Breaker — Multi-Model Consensus',
    icon: ShieldCheck,
    color: 'oklch(0.65 0.18 50)',
    algorithmName: 'Weighted Consensus Scoring (ML + GNN + Graph)',
    algorithmDetail: 'Combines ML risk score (XGBoost), GNN embedding similarity score, and structural graph evidence into a weighted consensus. If consensus exceeds threshold, issues a freeze order on the node and escalates to AI investigation. Implements exponential back-off to prevent alert fatigue.',
    techStack: ['Weighted Ensemble', 'Freeze Orders', 'Alert Dedup', 'Threshold Calibration'],
  },
  {
    key: 'llm_started',
    label: 'Qwen 3.5 AI Forensic Agent',
    icon: Bot,
    color: 'oklch(0.65 0.18 25)',
    algorithmName: 'LangGraph ReAct Agent (Qwen 3.5 via Ollama)',
    algorithmDetail: 'Runs a multi-step LangGraph ReAct loop: (1) Analyze transaction context, (2) Query graph neighbors, (3) Check historical patterns, (4) Cross-reference velocity data, (5) Generate forensic reasoning chain. Uses Ollama-hosted Qwen 3.5 (8B) with custom system prompt for Indian banking fraud patterns.',
    techStack: ['Qwen 3.5 (8B)', 'LangGraph ReAct', 'Ollama', 'Tool-Calling Agent', 'Chain-of-Thought'],
  },
  {
    key: 'verdict',
    label: 'Final Verdict + Blockchain Anchor',
    icon: Scale,
    color: 'oklch(0.55 0.18 250)',
    algorithmName: 'Classification → HITL Decision → Immutable Ledger',
    algorithmDetail: 'Aggregates all evidence into final verdict (legitimate / suspicious / fraudulent). High-confidence fraud triggers automatic freeze; borderline cases escalate to Human-in-the-Loop (HITL). Verdict + evidence hash anchored to append-only blockchain ledger with Ed25519 signatures.',
    techStack: ['Ed25519 Signatures', 'Append-Only Ledger', 'HITL Escalation', 'ZKP Anchoring'],
  },
]

// ── Helper: format duration ──────────────────────────────────────────────

function fmtDuration(ms?: number): string {
  if (ms == null) return '—'
  return ms < 1000 ? `${ms.toFixed(0)}ms` : `${(ms / 1000).toFixed(1)}s`
}

function fmtScore(score?: number): string {
  if (score == null) return '—'
  return `${(score * 100).toFixed(1)}%`
}

// ── Risk Score Gauge ─────────────────────────────────────────────────────

function RiskGauge({ score, tier }: { score: number; tier?: string }) {
  const pct = Math.min(score * 100, 100)
  const color = score > 0.7 ? '#ef4444' : score > 0.4 ? '#f59e0b' : '#22c55e'

  return (
    <div className="flex items-center gap-3">
      <div className="relative w-20 h-2 bg-bg-deep rounded-full overflow-hidden">
        <div
          className="absolute inset-y-0 left-0 rounded-full transition-all duration-700"
          style={{ width: `${pct}%`, background: color, boxShadow: `0 0 8px ${color}66` }}
        />
      </div>
      <span className="text-[10px] font-mono font-bold" style={{ color }}>{pct.toFixed(1)}%</span>
      {tier && (
        <span className="text-[8px] px-1.5 py-0.5 rounded-full font-bold uppercase tracking-wider" style={{
          color,
          background: `${color}15`,
          border: `1px solid ${color}30`,
        }}>{tier}</span>
      )}
    </div>
  )
}

// ── Consensus Score Bars ─────────────────────────────────────────────────

function ConsensusBar({ label, score, icon: Icon }: { label: string; score: number; icon: LucideIcon }) {
  const color = score > 0.7 ? '#ef4444' : score > 0.4 ? '#f59e0b' : '#22c55e'
  return (
    <div className="flex items-center gap-2">
      <Icon className="w-3 h-3 text-text-muted shrink-0" />
      <span className="text-[8px] font-semibold text-text-muted uppercase tracking-wider w-10">{label}</span>
      <div className="flex-1 h-1.5 bg-bg-deep rounded-full overflow-hidden">
        <div
          className="h-full rounded-full transition-all duration-500"
          style={{ width: `${score * 100}%`, background: color }}
        />
      </div>
      <span className="text-[9px] font-mono font-bold w-10 text-right" style={{ color }}>
        {(score * 100).toFixed(0)}%
      </span>
    </div>
  )
}

// ── Stage Transparency Card ──────────────────────────────────────────────

function StageTransparencyCard({
  config,
  lifecycle,
  isReached,
  isActive,
}: {
  config: TransparencyStageConfig
  lifecycle: EventLifecycle | undefined
  isReached: boolean
  isActive: boolean
}) {
  const [expanded, setExpanded] = useState(isActive)
  const Icon = config.icon
  const stageDetail = lifecycle?.stages.find((s) => s.stage === config.key)
  const duration = stageDetail?.durationMs
    ?? (stageDetail && lifecycle ? (() => {
        const idx = lifecycle.stages.findIndex(s => s.stage === config.key)
        if (idx > 0) return (lifecycle.stages[idx].timestamp - lifecycle.stages[idx - 1].timestamp) * 1000
        return undefined
      })() : undefined)

  return (
    <div
      className={cn(
        'border rounded-lg overflow-hidden transition-all duration-300',
        isActive && 'ring-1 ring-offset-0',
        isReached ? 'bg-bg-elevated/95 border-border-default' : 'bg-bg-overlay/40 border-border-subtle/50 opacity-50',
      )}
      style={isActive ? { outlineColor: `${config.color}60`, outlineWidth: '1px', outlineStyle: 'solid' } : undefined}
    >
      {/* Header — always visible */}
      <button
        onClick={() => setExpanded(!expanded)}
        className="w-full flex items-center gap-3 px-4 py-3 text-left hover:bg-bg-overlay/30 transition-colors"
      >
        {/* Stage icon */}
        <div
          className={cn(
            'w-8 h-8 rounded-lg flex items-center justify-center shrink-0 transition-all',
            isReached ? 'text-white' : 'bg-bg-deep/80 text-text-muted/50',
          )}
          style={isReached ? { background: config.color, boxShadow: `0 0 12px ${config.color}40` } : undefined}
        >
          <Icon className="w-4 h-4" />
        </div>

        {/* Label + algorithm */}
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2">
            <span className={cn(
              'text-[10px] font-bold uppercase tracking-wider',
              isReached ? 'text-text-primary' : 'text-text-muted/60',
            )}>
              {config.label}
            </span>
            {isActive && (
              <span className="flex items-center gap-1 text-[7px] px-1.5 py-0.5 rounded-full bg-accent-primary/15 text-accent-primary font-bold uppercase animate-pulse">
                <Activity className="w-2.5 h-2.5" /> Processing
              </span>
            )}
            {isReached && !isActive && (
              <CheckCircle2 className="w-3 h-3 text-green-400 shrink-0" />
            )}
          </div>
          <div className="text-[8px] text-text-muted mt-0.5 flex items-center gap-2">
            <Cpu className="w-2.5 h-2.5 shrink-0" />
            <span className="truncate">{config.algorithmName}</span>
            {duration != null && (
              <span className="flex items-center gap-0.5 shrink-0" style={{ color: config.color }}>
                <Zap className="w-2.5 h-2.5" />{fmtDuration(duration)}
              </span>
            )}
          </div>
        </div>

        {/* Expand arrow */}
        {expanded ? (
          <ChevronDown className="w-4 h-4 text-text-muted shrink-0" />
        ) : (
          <ChevronRight className="w-4 h-4 text-text-muted shrink-0" />
        )}
      </button>

      {/* Expanded detail */}
      {expanded && (
        <div className="px-4 pb-4 space-y-3 border-t border-border-subtle/50 pt-3 animate-fade-in">
          {/* Algorithm explanation */}
          <div className="space-y-1.5">
            <div className="flex items-center gap-1.5 text-[8px] font-semibold text-text-muted uppercase tracking-wider">
              <Brain className="w-3 h-3" /> How it works
            </div>
            <p className="text-[9px] text-text-secondary leading-relaxed">{config.algorithmDetail}</p>
          </div>

          {/* Tech stack pills */}
          <div className="flex flex-wrap gap-1">
            {config.techStack.map((tech) => (
              <span
                key={tech}
                className="px-1.5 py-0.5 rounded text-[7px] font-bold uppercase tracking-wider border"
                style={{
                  color: config.color,
                  borderColor: `${config.color}30`,
                  background: `${config.color}08`,
                }}
              >
                {tech}
              </span>
            ))}
          </div>

          {/* Stage-specific live data */}
          {config.key === 'ingested' && lifecycle && (
            <StageDataIngestion lifecycle={lifecycle} />
          )}
          {config.key === 'ml_scored' && lifecycle && (
            <StageDataML lifecycle={lifecycle} />
          )}
          {config.key === 'graph_investigated' && lifecycle && (
            <StageDataGraph stageDetail={stageDetail} />
          )}
          {config.key === 'cb_evaluated' && lifecycle && (
            <StageDataCircuitBreaker lifecycle={lifecycle} />
          )}
          {config.key === 'llm_started' && lifecycle && (
            <StageDataLLM lifecycle={lifecycle} />
          )}
          {config.key === 'verdict' && lifecycle && (
            <StageDataVerdict lifecycle={lifecycle} />
          )}

          {/* Raw meta dump */}
          {stageDetail?.meta && Object.keys(stageDetail.meta).length > 0 && (
            <details className="group">
              <summary className="flex items-center gap-1.5 text-[7px] font-semibold text-text-muted/60 uppercase tracking-wider cursor-pointer hover:text-text-muted transition-colors select-none list-none">
                <svg className="w-2.5 h-2.5 transition-transform group-open:rotate-90" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                </svg>
                Raw Stage Metadata
              </summary>
              <div className="mt-1.5 p-2 rounded bg-bg-deep/80 border border-border-subtle/30 space-y-0.5">
                {Object.entries(stageDetail.meta).map(([k, v]) => (
                  <div key={k} className="flex justify-between text-[7px] font-mono">
                    <span className="text-text-muted">{k}</span>
                    <span className="text-text-secondary truncate max-w-[200px] ml-4">{JSON.stringify(v)}</span>
                  </div>
                ))}
              </div>
            </details>
          )}
        </div>
      )}
    </div>
  )
}

// ── Stage-specific data renderers ────────────────────────────────────────

function StageDataIngestion({ lifecycle }: { lifecycle: EventLifecycle }) {
  return (
    <div className="p-2.5 rounded-md bg-bg-deep/60 border border-border-subtle/30 space-y-2">
      <div className="text-[8px] font-semibold text-text-muted uppercase tracking-wider flex items-center gap-1.5">
        <Database className="w-3 h-3" /> Ingested Event Summary
      </div>
      <div className="grid grid-cols-2 gap-x-4 gap-y-1">
        <KV label="Transaction ID" value={lifecycle.txnId} mono />
        <KV label="Sender" value={lifecycle.sender || '—'} mono />
        <KV label="Receiver" value={lifecycle.receiver || '—'} mono />
        <KV label="Amount" value={lifecycle.amountPaisa ? `₹${(lifecycle.amountPaisa / 100).toLocaleString()}` : '—'} />
        <KV label="Fraud Label" value={lifecycle.fraudLabel > 0 ? `⚠ ${lifecycle.fraudLabel}` : '✓ Clean'} />
        <KV label="First Seen" value={new Date(lifecycle.firstSeen * 1000).toLocaleTimeString()} />
      </div>
    </div>
  )
}

function StageDataML({ lifecycle }: { lifecycle: EventLifecycle }) {
  return (
    <div className="p-2.5 rounded-md bg-bg-deep/60 border border-border-subtle/30 space-y-2.5">
      <div className="text-[8px] font-semibold text-text-muted uppercase tracking-wider flex items-center gap-1.5">
        <BarChart3 className="w-3 h-3" /> ML Scoring Output
      </div>
      {lifecycle.riskScore != null ? (
        <>
          <RiskGauge score={lifecycle.riskScore} tier={lifecycle.riskTier} />
          {lifecycle.topFeatures && lifecycle.topFeatures.length > 0 && (
            <div className="space-y-1">
              <div className="text-[7px] font-semibold text-text-muted uppercase tracking-wider">Top Contributing Features</div>
              <div className="flex flex-wrap gap-1">
                {lifecycle.topFeatures.map((f, i) => (
                  <span key={i} className="px-1.5 py-0.5 rounded text-[7px] font-mono bg-purple-500/10 text-purple-300 border border-purple-500/20">
                    {f}
                  </span>
                ))}
              </div>
            </div>
          )}
        </>
      ) : (
        <WaitingIndicator label="Awaiting XGBoost scoring..." />
      )}
    </div>
  )
}

function StageDataGraph({ stageDetail }: { stageDetail?: { meta?: Record<string, unknown> } }) {
  const meta = stageDetail?.meta ?? {}
  return (
    <div className="p-2.5 rounded-md bg-bg-deep/60 border border-border-subtle/30 space-y-2">
      <div className="text-[8px] font-semibold text-text-muted uppercase tracking-wider flex items-center gap-1.5">
        <GitBranch className="w-3 h-3" /> Graph Analysis Results
      </div>
      {Object.keys(meta).length > 0 ? (
        <div className="grid grid-cols-2 gap-x-4 gap-y-1">
          {meta.mule_detected != null && <KV label="Mule Detected" value={meta.mule_detected ? '⚠ YES' : '✓ No'} />}
          {meta.cycle_found != null && <KV label="Cycles Found" value={meta.cycle_found ? '⚠ YES' : '✓ No'} />}
          {meta.centrality_score != null && <KV label="Centrality Score" value={fmtScore(meta.centrality_score as number)} />}
          {meta.community_id != null && <KV label="Community ID" value={String(meta.community_id)} />}
          {meta.degree != null && <KV label="Node Degree" value={String(meta.degree)} />}
          {meta.connected_fraud_nodes != null && <KV label="Connected Fraud Nodes" value={String(meta.connected_fraud_nodes)} />}
        </div>
      ) : (
        <div className="text-[8px] text-text-muted/50 italic">
          Graph structural analysis active — detecting mule chains, cycles, and centrality anomalies in the transaction network
        </div>
      )}
    </div>
  )
}

function StageDataCircuitBreaker({ lifecycle }: { lifecycle: EventLifecycle }) {
  return (
    <div className="p-2.5 rounded-md bg-bg-deep/60 border border-border-subtle/30 space-y-2.5">
      <div className="text-[8px] font-semibold text-text-muted uppercase tracking-wider flex items-center gap-1.5">
        <Shield className="w-3 h-3" /> Multi-Model Consensus
      </div>
      {lifecycle.consensusScores ? (
        <div className="space-y-1.5">
          <ConsensusBar label="ML" score={lifecycle.consensusScores.ml} icon={BrainCircuit} />
          <ConsensusBar label="GNN" score={lifecycle.consensusScores.gnn} icon={Layers} />
          <ConsensusBar label="Graph" score={lifecycle.consensusScores.graph} icon={Network} />
          <div className="pt-1.5 border-t border-border-subtle/30">
            <div className="flex items-center justify-between text-[8px]">
              <span className="text-text-muted font-semibold uppercase tracking-wider">Weighted Consensus</span>
              <span className="font-mono font-bold" style={{
                color: ((lifecycle.consensusScores.ml + lifecycle.consensusScores.gnn + lifecycle.consensusScores.graph) / 3) > 0.5
                  ? '#ef4444' : '#22c55e',
              }}>
                {(((lifecycle.consensusScores.ml + lifecycle.consensusScores.gnn + lifecycle.consensusScores.graph) / 3) * 100).toFixed(1)}%
              </span>
            </div>
          </div>
        </div>
      ) : (
        <WaitingIndicator label="Awaiting multi-model consensus evaluation..." />
      )}
    </div>
  )
}

function StageDataLLM({ lifecycle }: { lifecycle: EventLifecycle }) {
  return (
    <div className="p-2.5 rounded-md bg-bg-deep/60 border border-border-subtle/30 space-y-2.5">
      <div className="text-[8px] font-semibold text-text-muted uppercase tracking-wider flex items-center gap-1.5">
        <MessageSquare className="w-3 h-3" /> Qwen 3.5 AI Investigation
      </div>
      {lifecycle.thinkingSteps != null || lifecycle.toolsUsed ? (
        <div className="space-y-2">
          <div className="grid grid-cols-2 gap-x-4 gap-y-1">
            <KV label="Thinking Steps" value={String(lifecycle.thinkingSteps ?? '—')} />
            <KV label="Investigation Time" value={fmtDuration(lifecycle.totalDurationMs)} />
            <KV label="NLU Findings" value={String(lifecycle.nluFindingsCount ?? '—')} />
            <KV label="Escalated" value={lifecycle.nluEscalated ? '⚠ YES' : '✓ No'} />
          </div>
          {lifecycle.toolsUsed && lifecycle.toolsUsed.length > 0 && (
            <div className="space-y-1">
              <div className="text-[7px] font-semibold text-text-muted uppercase tracking-wider flex items-center gap-1">
                <Wrench className="w-2.5 h-2.5" /> Tools Called by Agent
              </div>
              <div className="flex flex-wrap gap-1">
                {lifecycle.toolsUsed.map((tool, i) => (
                  <span key={i} className="px-1.5 py-0.5 rounded text-[7px] font-mono bg-rose-500/10 text-rose-300 border border-rose-500/20">
                    {tool}
                  </span>
                ))}
              </div>
            </div>
          )}
          {lifecycle.reasoningSummary && (
            <div className="space-y-1">
              <div className="text-[7px] font-semibold text-text-muted uppercase tracking-wider flex items-center gap-1">
                <Brain className="w-2.5 h-2.5" /> AI Reasoning Chain
              </div>
              <p className="text-[8px] text-text-secondary leading-relaxed bg-bg-overlay/40 rounded p-2 border border-border-subtle/30 italic">
                "{lifecycle.reasoningSummary}"
              </p>
            </div>
          )}
        </div>
      ) : (
        <WaitingIndicator label="Qwen 3.5 agent active — running ReAct investigation loop..." />
      )}
    </div>
  )
}

function StageDataVerdict({ lifecycle }: { lifecycle: EventLifecycle }) {
  const verdictColor =
    lifecycle.verdict === 'fraudulent' ? '#ef4444'
    : lifecycle.verdict === 'suspicious' ? '#f59e0b'
    : lifecycle.verdict === 'legitimate' ? '#22c55e'
    : '#6b7280'

  return (
    <div className="p-2.5 rounded-md bg-bg-deep/60 border border-border-subtle/30 space-y-2.5">
      <div className="text-[8px] font-semibold text-text-muted uppercase tracking-wider flex items-center gap-1.5">
        <Target className="w-3 h-3" /> Final Verdict
      </div>
      {lifecycle.verdict ? (
        <div className="space-y-2">
          {/* Verdict badge */}
          <div className="flex items-center gap-3">
            <span
              className="px-3 py-1.5 rounded-md text-[11px] font-black uppercase tracking-wider"
              style={{
                color: verdictColor,
                background: `${verdictColor}15`,
                border: `1px solid ${verdictColor}40`,
                boxShadow: `0 0 12px ${verdictColor}20`,
              }}
            >
              {lifecycle.verdict}
            </span>
            {lifecycle.confidence != null && (
              <div className="flex items-center gap-1.5">
                <span className="text-[8px] text-text-muted">Confidence:</span>
                <span className="text-[10px] font-mono font-bold" style={{ color: verdictColor }}>
                  {(lifecycle.confidence * 100).toFixed(1)}%
                </span>
              </div>
            )}
          </div>

          {lifecycle.fraudTypology && (
            <div className="flex items-center gap-2 text-[8px]">
              <AlertTriangle className="w-3 h-3 text-amber-400" />
              <span className="text-text-muted">Typology:</span>
              <span className="font-semibold text-text-primary">{lifecycle.fraudTypology}</span>
            </div>
          )}

          {lifecycle.recommendedAction && (
            <div className="flex items-center gap-2 text-[8px]">
              <Shield className="w-3 h-3 text-accent-primary" />
              <span className="text-text-muted">Action:</span>
              <span className="font-semibold text-text-primary">{lifecycle.recommendedAction}</span>
            </div>
          )}

          {lifecycle.evidenceCited && lifecycle.evidenceCited.length > 0 && (
            <div className="space-y-1">
              <div className="text-[7px] font-semibold text-text-muted uppercase tracking-wider flex items-center gap-1">
                <FileText className="w-2.5 h-2.5" /> Evidence Cited
              </div>
              <ul className="space-y-0.5">
                {lifecycle.evidenceCited.map((ev, i) => (
                  <li key={i} className="flex items-start gap-1.5 text-[8px] text-text-secondary">
                    <span className="text-accent-primary mt-0.5">•</span>
                    <span>{ev}</span>
                  </li>
                ))}
              </ul>
            </div>
          )}
        </div>
      ) : (
        <WaitingIndicator label="Awaiting final classification verdict..." />
      )}
    </div>
  )
}

// ── Shared small components ──────────────────────────────────────────────

function KV({ label, value, mono }: { label: string; value: string; mono?: boolean }) {
  return (
    <div className="flex justify-between text-[8px]">
      <span className="text-text-muted uppercase tracking-wider">{label}</span>
      <span className={cn('text-text-secondary truncate max-w-[140px] ml-2', mono && 'font-mono')}>{value}</span>
    </div>
  )
}

function WaitingIndicator({ label }: { label: string }) {
  return (
    <div className="flex items-center gap-2 py-1.5 text-[8px] text-text-muted/60 italic">
      <div className="w-1.5 h-1.5 rounded-full bg-accent-primary/60 animate-pulse" />
      {label}
    </div>
  )
}

// ── Main Component ──────────────────────────────────────────────────────────

interface PipelineTransparencyProps {
  className?: string
}

export function PipelineTransparency({ className }: PipelineTransparencyProps) {
  const events = useActivityStore((s) => s.events)
  const orderedIds = useActivityStore((s) => s.orderedIds)
  const trackedEventId = useActivityStore((s) => s.trackedEventId)

  // Pick the event to display
  const activeId = trackedEventId ?? orderedIds[0] ?? null
  const lifecycle = activeId ? events.get(activeId) : undefined

  const completedStages = useMemo(() => {
    if (!lifecycle) return new Set<PipelineStage>()
    return new Set(lifecycle.stages.map((s) => s.stage))
  }, [lifecycle])

  // Determine active stage (the next one after the last completed)
  const activeStage = useMemo<PipelineStage | null>(() => {
    if (!lifecycle) return null
    let lastIdx = -1
    for (let i = TRANSPARENCY_STAGES.length - 1; i >= 0; i--) {
      if (completedStages.has(TRANSPARENCY_STAGES[i].key)) {
        lastIdx = i
        break
      }
    }
    if (lastIdx < TRANSPARENCY_STAGES.length - 1) {
      return TRANSPARENCY_STAGES[lastIdx + 1].key
    }
    return null
  }, [lifecycle, completedStages])

  // Overall progress
  const stagesReached = lifecycle ? lifecycle.stages.filter(s =>
    TRANSPARENCY_STAGES.some(ts => ts.key === s.stage)
  ).length : 0
  const totalStages = TRANSPARENCY_STAGES.length

  return (
    <div className={cn('bg-bg-elevated/95 border border-border-default rounded-lg overflow-hidden backdrop-blur-sm', className)}>
      {/* Header */}
      <div className="px-4 py-3 border-b border-border-subtle flex items-center justify-between">
        <div className="flex items-center gap-2">
          <div className="w-7 h-7 rounded-lg bg-gradient-to-br from-amber-500/15 to-rose-500/15 border border-amber-500/25 flex items-center justify-center">
            <Eye className="w-3.5 h-3.5 text-amber-400" />
          </div>
          <div>
            <h3 className="text-[11px] font-bold text-text-primary uppercase tracking-[0.12em]">
              Pipeline Transparency — Full X-Ray
            </h3>
            <p className="text-[8px] text-text-muted mt-0.5">
              {lifecycle
                ? `Tracking ${activeId?.slice(0, 16)} — ${stagesReached}/${totalStages} stages • Every algorithm, every decision, zero black boxes`
                : 'Inject an event to see the full processing pipeline with complete transparency'
              }
            </p>
          </div>
        </div>
        {lifecycle && (
          <div className="flex items-center gap-2">
            <div className="flex items-center gap-1.5 text-[8px]">
              <TrendingUp className="w-3 h-3 text-text-muted" />
              <span className="text-text-muted">Progress:</span>
              <span className="font-mono font-bold text-text-primary">{stagesReached}/{totalStages}</span>
            </div>
            <div className="w-16 h-1.5 bg-bg-deep rounded-full overflow-hidden">
              <div
                className="h-full rounded-full transition-all duration-700"
                style={{
                  width: `${(stagesReached / totalStages) * 100}%`,
                  background: stagesReached === totalStages
                    ? 'linear-gradient(90deg, #22c55e, #4ade80)'
                    : 'linear-gradient(90deg, #f59e0b, #f97316)',
                }}
              />
            </div>
          </div>
        )}
      </div>

      {/* Stage cards */}
      <div className="p-4 space-y-2">
        {!lifecycle ? (
          <div className="flex flex-col items-center justify-center py-10 text-center">
            <Eye className="w-8 h-8 text-text-muted/30 mb-3" />
            <p className="text-[11px] font-semibold text-text-muted">No Active Event</p>
            <p className="text-[9px] text-text-muted/60 mt-1 max-w-sm">
              Use the <span className="text-amber-400 font-semibold">Random Attack</span> button or inject a custom event
              to see the full pipeline processing with complete transparency
            </p>
          </div>
        ) : (
          TRANSPARENCY_STAGES.map((config) => (
            <StageTransparencyCard
              key={config.key}
              config={config}
              lifecycle={lifecycle}
              isReached={completedStages.has(config.key)}
              isActive={activeStage === config.key}
            />
          ))
        )}
      </div>
    </div>
  )
}

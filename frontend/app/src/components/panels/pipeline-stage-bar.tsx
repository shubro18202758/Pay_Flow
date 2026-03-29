// ============================================================================
// Pipeline Stage Bar -- 6-stage horizontal progress indicator
// IN → ML → GR → CB → AI → VD
// Enhanced: full stage names in expanded, latency coloring, stage descriptions
// ============================================================================

import type { PipelineStage, StageDetail } from '@/stores/use-activity-store'
import { cn } from '@/lib/utils'

const STAGES: { key: PipelineStage; label: string; full: string; desc: string }[] = [
  { key: 'ingested', label: 'IN', full: 'Ingestion', desc: 'Schema validation + CRC32 checksum' },
  { key: 'ml_scored', label: 'ML', full: 'ML Scoring', desc: 'XGBoost fraud classifier (GPU)' },
  { key: 'graph_investigated', label: 'GR', full: 'Graph Analysis', desc: 'NetworkX + GNN + mule/cycle detection' },
  { key: 'cb_evaluated', label: 'CB', full: 'Circuit Breaker', desc: 'Multi-model consensus (ML+GNN+structural)' },
  { key: 'llm_started', label: 'AI', full: 'Qwen AI Agent', desc: 'LLM forensic investigation via LangGraph' },
  { key: 'verdict', label: 'VD', full: 'Verdict', desc: 'Final classification + HITL escalation' },
]

// Latency thresholds for color coding
function latencyColor(ms: number): string {
  if (ms < 50) return 'text-green-400'
  if (ms < 200) return 'text-accent-primary'
  if (ms < 500) return 'text-amber-400'
  return 'text-red-400'
}

interface StageEntry {
  stage: string
  timestamp: number
  durationMs?: number
}

interface PipelineStageBarProps {
  completedStages: PipelineStage[]
  stageEntries?: StageDetail[]
  expanded?: boolean
}

export function PipelineStageBar({ completedStages, stageEntries, expanded }: PipelineStageBarProps) {
  const completed = new Set(completedStages)
  let lastCompletedIdx = -1
  for (let i = STAGES.length - 1; i >= 0; i--) {
    if (completed.has(STAGES[i].key)) {
      lastCompletedIdx = i
      break
    }
  }

  const stageMap = new Map<string, StageEntry>()
  if (stageEntries) {
    for (const se of stageEntries) stageMap.set(se.stage, se as StageEntry)
  }

  // Calculate total pipeline duration
  const totalDuration = stageEntries && stageEntries.length >= 2
    ? Math.max(0, stageEntries[stageEntries.length - 1].timestamp - stageEntries[0].timestamp)
    : null

  if (expanded && stageEntries && stageEntries.length > 0) {
    return (
      <div className="space-y-1">
        {STAGES.map((stage, i) => {
          const isCompleted = completed.has(stage.key)
          const isActive = i === lastCompletedIdx + 1 && lastCompletedIdx < STAGES.length - 1
          const entry = stageMap.get(stage.key)
          const prevEntry = i > 0 ? stageMap.get(STAGES[i - 1].key) : null
          const delta = entry && prevEntry ? Math.max(0, entry.timestamp - prevEntry.timestamp) : null
          const deltaMs = delta != null ? delta * 1000 : null
          const deltaStr = delta != null
            ? delta < 1 ? `${(delta * 1000).toFixed(0)}ms` : `${delta.toFixed(1)}s`
            : null
          const durationMs = entry?.durationMs

          return (
            <div key={stage.key} className="flex items-center gap-2 group" title={stage.desc}>
              <div
                className={cn(
                  'w-5 h-5 rounded-sm flex items-center justify-center text-[8px] font-mono font-bold transition-all duration-300 shrink-0',
                  isCompleted && 'bg-accent-primary text-white shadow-[0_0_6px_oklch(0.55_0.14_250_/_0.3)]',
                  isActive && 'bg-accent-primary/30 text-accent-primary border border-accent-primary/50 animate-data-pulse',
                  !isCompleted && !isActive && 'bg-bg-overlay/60 text-text-muted border border-border-subtle',
                )}
              >
                {stage.label}
              </div>
              <div className="flex-1 min-w-0">
                <div className="flex items-center gap-1.5">
                  <span className={cn(
                    'text-[9px] font-semibold tracking-wider',
                    isCompleted ? 'text-text-primary' : isActive ? 'text-accent-primary' : 'text-text-muted/50',
                  )}>
                    {stage.full}
                  </span>
                  {isActive && (
                    <span className="text-[7px] text-accent-primary/60 uppercase tracking-wider animate-pulse">Processing</span>
                  )}
                </div>
                <div className="h-1 bg-bg-overlay rounded-full overflow-hidden mt-0.5">
                  {isCompleted && (
                    <div className="h-full bg-accent-primary rounded-full animate-stage-fill" />
                  )}
                  {isActive && (
                    <div className="h-full bg-accent-primary/40 rounded-full animate-data-pulse" style={{ width: '60%' }} />
                  )}
                </div>
              </div>
              <div className="flex items-center gap-1.5 shrink-0">
                {deltaStr && (
                  <span className={cn('text-[8px] font-mono tabular-nums', deltaMs != null ? latencyColor(deltaMs) : 'text-text-muted')}>+{deltaStr}</span>
                )}
                {durationMs != null && (
                  <span className={cn('text-[8px] font-mono tabular-nums', latencyColor(durationMs))}>{durationMs}ms</span>
                )}
                {!isCompleted && !isActive && (
                  <span className="text-[8px] text-text-muted/40">&mdash;</span>
                )}
              </div>
            </div>
          )
        })}
        {totalDuration != null && lastCompletedIdx === STAGES.length - 1 && (
          <div className="flex items-center justify-end gap-1 pt-1 border-t border-border-subtle/50 mt-1">
            <span className="text-[7px] text-text-muted uppercase tracking-wider">Total pipeline</span>
            <span className={cn('text-[9px] font-mono font-bold tabular-nums', latencyColor(totalDuration * 1000))}>
              {totalDuration < 1 ? `${(totalDuration * 1000).toFixed(0)}ms` : `${totalDuration.toFixed(2)}s`}
            </span>
          </div>
        )}
      </div>
    )
  }

  // Compact mode (default) — enhanced with subtle glow on completed stages
  return (
    <div className="flex items-center gap-0.5">
      {STAGES.map((stage, i) => {
        const isCompleted = completed.has(stage.key)
        const isActive = i === lastCompletedIdx + 1 && lastCompletedIdx < STAGES.length - 1

        return (
          <div key={stage.key} className="flex items-center" title={`${stage.full}: ${stage.desc}`}>
            <div className="flex flex-col items-center">
              <div
                className={cn(
                  'w-4 h-4 rounded-sm flex items-center justify-center text-[7px] font-mono font-bold transition-all duration-300',
                  isCompleted && 'bg-accent-primary text-white shadow-[0_0_4px_oklch(0.55_0.14_250_/_0.25)]',
                  isActive && 'bg-accent-primary/30 text-accent-primary border border-accent-primary/50 animate-data-pulse',
                  !isCompleted && !isActive && 'bg-bg-overlay/60 text-text-muted border border-border-subtle',
                )}
              >
                {stage.label}
              </div>
            </div>
            {i < STAGES.length - 1 && (
              <div
                className={cn(
                  'w-1.5 h-px transition-colors duration-300',
                  i < lastCompletedIdx ? 'bg-accent-primary' : 'bg-border-subtle',
                )}
              />
            )}
          </div>
        )
      })}
    </div>
  )
}

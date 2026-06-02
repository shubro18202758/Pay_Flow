// ============================================================================
// Pipeline Stage Bar -- 6-stage horizontal progress indicator
// IN → ML → GR → CB → AI → VD
// Enhanced: full stage names in expanded, latency coloring, stage descriptions
// ============================================================================

import { useEffect, useMemo, useState } from 'react'
import type { PipelineStage, StageDetail } from '@/stores/use-activity-store'
import { cn } from '@/lib/utils'
import { useLLMStatus } from '@/hooks/use-api'
import { resolveLLMRuntime } from '@/lib/llm-runtime'
import { latencyToneClass } from '@/lib/union-bank-theme'

type StageDef = { key: PipelineStage; label: string; full: string; desc: string }

const BASE_STAGES: StageDef[] = [
  { key: 'ingested', label: 'IN', full: 'Ingestion', desc: 'Schema validation + CRC32 checksum' },
  { key: 'ml_scored', label: 'ML', full: 'ML Scoring', desc: '36-feature extraction + XGBoost risk scoring' },
  { key: 'graph_investigated', label: 'GR', full: 'Graph Analysis', desc: 'NetworkX mule, cycle, community, and centrality detection' },
  { key: 'cb_evaluated', label: 'CB', full: 'Circuit Breaker', desc: 'Available-model consensus from ML and graph evidence' },
  { key: 'llm_started', label: 'AI', full: 'LLM Agent', desc: 'Bounded forensic explanation via LangGraph' },
  { key: 'verdict', label: 'VD', full: 'Verdict', desc: 'Final classification + HITL escalation' },
]

function stagesForRuntime(status: ReturnType<typeof resolveLLMRuntime>): StageDef[] {
  return BASE_STAGES.map((stage) =>
    stage.key === 'llm_started'
      ? {
          ...stage,
          full: status.model !== 'configured LLM' ? `${status.model} Agent` : stage.full,
          desc: `Bounded forensic explanation via LangGraph (${status.model}, ${status.statusLabel}); ML, graph, rules, and ledger remain authoritative`,
        }
      : stage,
  )
}

// Latency thresholds for color coding
function latencyColor(ms: number): string {
  return latencyToneClass(ms)
}

function formatElapsed(seconds: number): string {
  if (seconds < 1) return `${(seconds * 1000).toFixed(0)}ms`
  if (seconds < 60) return `${seconds.toFixed(1)}s`
  return `${Math.floor(seconds / 60)}m ${Math.floor(seconds % 60)}s`
}

function hasBackendTimestamp(timestamp: number | null | undefined): timestamp is number {
  return typeof timestamp === 'number' && Number.isFinite(timestamp) && timestamp > 0
}

function hasDurationMs(ms: number | null | undefined): ms is number {
  return typeof ms === 'number' && Number.isFinite(ms) && ms >= 0
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
  const [nowSec, setNowSec] = useState(0)
  const { data: llmStatus, isLoading: llmStatusLoading, isError: llmStatusError } = useLLMStatus()
  const llmRuntime = useMemo(
    () => resolveLLMRuntime(llmStatus, { loading: llmStatusLoading, error: llmStatusError }),
    [llmStatus, llmStatusLoading, llmStatusError],
  )
  const stages = useMemo(
    () => stagesForRuntime(llmRuntime),
    [llmRuntime],
  )
  const completed = new Set(completedStages)
  let lastCompletedIdx = -1
  for (let i = stages.length - 1; i >= 0; i--) {
    if (completed.has(stages[i].key)) {
      lastCompletedIdx = i
      break
    }
  }

  const stageMap = new Map<string, StageEntry>()
  if (stageEntries) {
    for (const se of stageEntries) stageMap.set(se.stage, se as StageEntry)
  }
  const hasActiveStage = lastCompletedIdx < stages.length - 1
  const latestEntry = stageEntries?.reduce<StageDetail | null>(
    (latest, entry) => (!latest || entry.timestamp > latest.timestamp ? entry : latest),
    null,
  )
  const latestTimedEntry = latestEntry && hasBackendTimestamp(latestEntry.timestamp) ? latestEntry : null

  useEffect(() => {
    if (!expanded || !hasActiveStage || !latestTimedEntry) return
    const updateNow = () => setNowSec(Date.now() / 1000)
    updateNow()
    const interval = window.setInterval(updateNow, 1000)
    return () => window.clearInterval(interval)
  }, [expanded, hasActiveStage, latestTimedEntry])

  // Calculate total pipeline duration
  const firstEntry = stageEntries?.[0]
  const finalEntry = stageEntries?.[stageEntries.length - 1]
  const totalDuration = firstEntry && finalEntry && hasBackendTimestamp(firstEntry.timestamp) && hasBackendTimestamp(finalEntry.timestamp)
    ? Math.max(0, stageEntries[stageEntries.length - 1].timestamp - stageEntries[0].timestamp)
    : null

  if (expanded && stageEntries && stageEntries.length > 0) {
    return (
      <div className="space-y-1">
        {stages.map((stage, i) => {
          const isCompleted = completed.has(stage.key)
          const isActive = i === lastCompletedIdx + 1 && lastCompletedIdx < stages.length - 1
          const entry = stageMap.get(stage.key)
          const prevEntry = i > 0 ? stageMap.get(stages[i - 1].key) : null
          const delta = entry && prevEntry && hasBackendTimestamp(entry.timestamp) && hasBackendTimestamp(prevEntry.timestamp)
            ? Math.max(0, entry.timestamp - prevEntry.timestamp)
            : null
          const deltaMs = delta != null ? delta * 1000 : null
          const deltaStr = delta != null
            ? delta < 1 ? `${(delta * 1000).toFixed(0)}ms` : `${delta.toFixed(1)}s`
            : null
          const durationMs = hasDurationMs(entry?.durationMs) ? entry.durationMs : null
          const activeElapsed = isActive && latestTimedEntry && nowSec > 0
            ? Math.max(0, nowSec - latestTimedEntry.timestamp)
            : null

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
                    <span className="text-[7px] text-accent-primary/60 uppercase tracking-wider animate-pulse">
                      Awaiting backend event
                    </span>
                  )}
                </div>
                <div className="h-1 bg-bg-overlay rounded-full overflow-hidden mt-0.5">
                  {isCompleted && (
                    <div className="h-full bg-accent-primary rounded-full animate-stage-fill" />
                  )}
                  {isActive && (
                    <div className="h-full w-2 rounded-full bg-accent-primary/50 animate-data-pulse" />
                  )}
                </div>
              </div>
              <div className="flex items-center gap-1.5 shrink-0">
                {activeElapsed != null && (
                  <span className="text-[8px] font-mono tabular-nums text-accent-primary">
                    {formatElapsed(activeElapsed)}
                  </span>
                )}
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
        {totalDuration != null && lastCompletedIdx === stages.length - 1 && (
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
      {stages.map((stage, i) => {
        const isCompleted = completed.has(stage.key)
        const isActive = i === lastCompletedIdx + 1 && lastCompletedIdx < stages.length - 1

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
            {i < stages.length - 1 && (
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

import { useMemo } from 'react'
import {
  Activity,
  CheckCircle2,
  DatabaseZap,
  GitBranch,
  RadioTower,
  ShieldCheck,
  TimerReset,
} from 'lucide-react'
import { useActivityStore, type PipelineStage } from '@/stores/use-activity-store'
import { useUIStore } from '@/stores/use-ui-store'
import { cn, fmtPaisa, truncId } from '@/lib/utils'

const STAGES: Array<{ key: PipelineStage; label: string; detail: string }> = [
  { key: 'ingested', label: 'Ingest', detail: 'backend intake' },
  { key: 'ml_scored', label: 'ML', detail: 'feature score' },
  { key: 'graph_investigated', label: 'Graph', detail: 'network evidence' },
  { key: 'cb_evaluated', label: 'Control', detail: 'freeze/hold gate' },
  { key: 'llm_started', label: 'Qwen', detail: 'bounded explanation' },
  { key: 'verdict', label: 'Decision', detail: 'case outcome' },
  { key: 'pipeline_dispatched', label: 'Dispatch', detail: 'consumer fan-out' },
]

export function LivePipelineRibbon() {
  const events = useActivityStore((s) => s.events)
  const orderedIds = useActivityStore((s) => s.orderedIds)
  const eventLabRuns = useActivityStore((s) => s.eventLabRuns)
  const activeRunId = useActivityStore((s) => s.activeEventLabRunId)
  const latestReport = useActivityStore((s) => s.eventLabReports[0])
  const countermeasure = useActivityStore((s) => s.countermeasures[0])
  const resolution = useActivityStore((s) => s.resolutionFanouts[0])
  const connected = useUIStore((s) => s.connected)

  const latest = useMemo(() => {
    for (const id of orderedIds) {
      const lifecycle = events.get(id)
      if (lifecycle?.stages.length) return lifecycle
    }
    return null
  }, [events, orderedIds])

  const activeRun = activeRunId ? eventLabRuns[activeRunId] : null
  const completed = new Set(latest?.stages.map((stage) => stage.stage) ?? [])
  const completion = Math.round((completed.size / STAGES.length) * 100)

  return (
    <section className="shrink-0 border-b border-[#d7e3f1] bg-white px-4 py-2 shadow-[0_2px_10px_rgba(20,33,61,0.06)]">
      <div className="grid gap-2 2xl:grid-cols-[330px_minmax(0,1fr)_430px]">
        <div className="min-w-0 rounded-md border border-[#00579C]/20 bg-[#f7fbff] px-3 py-2">
          <div className="mb-1 flex items-center justify-between gap-2">
            <span className="inline-flex items-center gap-1.5 text-[9px] font-extrabold uppercase tracking-[0.14em] text-[#00579C]">
              <RadioTower className="h-3.5 w-3.5 text-[#DA251C]" />
              live backend pipeline
            </span>
            <span
              className={cn(
                'rounded-sm px-1.5 py-0.5 text-[8px] font-extrabold uppercase tracking-[0.1em]',
                connected ? 'bg-[#00579C]/8 text-[#00579C]' : 'bg-[#DA251C]/8 text-[#DA251C]',
              )}
            >
              {connected ? 'sse online' : 'connecting'}
            </span>
          </div>
          {latest ? (
            <div className="grid min-w-0 grid-cols-[1fr_auto] items-end gap-2">
              <div className="min-w-0">
                <div className="truncate font-mono text-[11px] font-black text-[#14213d]" title={latest.txnId}>
                  {truncId(latest.txnId, 18)}
                </div>
                <div className="mt-0.5 truncate text-[9px] font-semibold text-[#617189]">
                  {truncId(latest.sender || 'sender pending', 10)} -&gt; {truncId(latest.receiver || 'receiver pending', 10)}
                  {latest.amountPaisa > 0 ? ` | ${fmtPaisa(latest.amountPaisa)}` : ''}
                </div>
              </div>
              <div className="text-right">
                <div className="font-mono text-[13px] font-black text-[#00579C]">{completion}%</div>
                <div className="text-[7px] font-bold uppercase tracking-[0.12em] text-[#617189]">complete</div>
              </div>
            </div>
          ) : (
            <div className="grid grid-cols-[1fr_auto] items-end gap-2">
              <div>
                <div className="font-mono text-[11px] font-black text-[#14213d]">awaiting first event</div>
                <div className="mt-0.5 text-[9px] font-semibold text-[#617189]">
                  Launch an event drill or inject a transaction to populate live stages.
                </div>
              </div>
              <TimerReset className="h-5 w-5 text-[#617189]/55" />
            </div>
          )}
        </div>

        <div className="min-w-0 rounded-md border border-[#d7e3f1] bg-[#f7fbff] px-3 py-2">
          <div className="grid min-w-0 grid-cols-7 gap-1.5">
            {STAGES.map((stage) => {
              const entry = latest?.stages.find((item) => item.stage === stage.key)
              const done = Boolean(entry)
              return (
                <div
                  key={stage.key}
                  className={cn(
                    'min-w-0 rounded-md border px-2 py-1.5',
                    done
                      ? 'border-[#00579C]/25 bg-white text-[#00579C]'
                      : 'border-[#d7e3f1] bg-[#edf4fb] text-[#617189]',
                  )}
                  title={`${stage.label}: ${stage.detail}`}
                >
                  <div className="flex items-center justify-between gap-1">
                    <span className="truncate text-[8px] font-extrabold uppercase tracking-[0.1em]">{stage.label}</span>
                    {done ? <CheckCircle2 className="h-3 w-3 shrink-0 text-[#00579C]" /> : <Activity className="h-3 w-3 shrink-0 text-[#617189]/55" />}
                  </div>
                  <div className="mt-0.5 truncate text-[7px] font-semibold text-[#617189]">
                    {entry?.durationMs != null ? `${Math.round(entry.durationMs)} ms` : stage.detail}
                  </div>
                </div>
              )
            })}
          </div>
        </div>

        <div className="grid min-w-0 gap-2 lg:grid-cols-3 2xl:grid-cols-[1.2fr_1fr_1fr]">
          <RibbonFact
            icon={GitBranch}
            label="resolution fan-out"
            value={resolution ? `${resolution.status || 'decided'} -> ${resolution.nodeStatus || 'graph sync'}` : 'awaiting analyst decision'}
            detail={resolution ? `${resolution.queryGroups.slice(0, 4).join(' / ')}${resolution.queryGroups.length > 4 ? ' +' : ''}` : 'updates graph, queues, analytics and audit views'}
            active={Boolean(resolution)}
          />
          <RibbonFact
            icon={ShieldCheck}
            label="countermeasure"
            value={countermeasure ? `${countermeasure.action} ${countermeasure.status}` : 'no active proposal'}
            detail={countermeasure ? countermeasure.targets.slice(0, 2).join(' / ') || countermeasure.runId : 'analyst gated before execution'}
            active={Boolean(countermeasure)}
          />
          <RibbonFact
            icon={DatabaseZap}
            label="event lab run"
            value={latestReport ? `${latestReport.riskTier} report synced` : activeRun ? activeRun.status : 'not launched'}
            detail={latestReport ? `${truncId(latestReport.runId, 12)} | ${latestReport.eventCount} events | ${latestReport.routeLabel}` : activeRun ? `${activeRun.stages.length} stages | ${activeRun.templateTitle}` : 'pipeline stages will appear here live'}
            active={Boolean(activeRun || latestReport)}
          />
        </div>
      </div>
    </section>
  )
}

function RibbonFact({
  icon: Icon,
  label,
  value,
  detail,
  active,
}: {
  icon: typeof Activity
  label: string
  value: string
  detail: string
  active: boolean
}) {
  return (
    <div className="min-w-0 rounded-md border border-[#d7e3f1] bg-[#f7fbff] px-3 py-2">
      <div className="mb-1 flex items-center gap-1.5">
        <Icon className={cn('h-3.5 w-3.5 shrink-0', active ? 'text-[#DA251C]' : 'text-[#617189]/70')} />
        <span className="truncate text-[8px] font-extrabold uppercase tracking-[0.13em] text-[#00579C]">{label}</span>
      </div>
      <div className="truncate text-[10px] font-black text-[#14213d]" title={value}>{value}</div>
      <div className="mt-0.5 truncate text-[8px] font-semibold text-[#617189]" title={detail}>{detail}</div>
    </div>
  )
}

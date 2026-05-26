import { useDashboardStore } from '@/stores/use-dashboard-store'
import { useUIStore } from '@/stores/use-ui-store'
import { cn } from '@/lib/utils'
import { AlertCircle, CheckCircle2, Radio, Server, Wifi } from 'lucide-react'
import { useEffect, useState } from 'react'

function buildUiBase(): string {
  if (typeof window === 'undefined') return 'http://127.0.0.1:8010'
  return window.location.origin
}

function buildApiBase(): string {
  if (typeof window === 'undefined') return 'http://127.0.0.1:8010'
  return window.location.origin
}

export function RuntimeBanner() {
  const connected = useUIStore((s) => s.connected)
  const orchestrator = useDashboardStore((s) => s.orchestrator)
  const hardware = useDashboardStore((s) => s.hardware)
  const [warmup, setWarmup] = useState(true)

  useEffect(() => {
    if (connected || (orchestrator && hardware)) {
      setWarmup(false)
      return
    }
    const timer = window.setTimeout(() => setWarmup(false), 14_000)
    return () => window.clearTimeout(timer)
  }, [connected, orchestrator, hardware])

  if ((connected || !warmup) && orchestrator && hardware) {
    return null
  }

  const uiBase = buildUiBase()
  const apiBase = buildApiBase()

  const title = connected
    ? 'Live stream connected. Waiting for the next telemetry frame.'
    : warmup
      ? 'Starting live telemetry. REST snapshot is hydrating the prototype while SSE attaches.'
      : 'Live stream is reconnecting in the background. Current REST snapshot remains available.'

  const StatusIcon = connected ? CheckCircle2 : AlertCircle
  const tone = connected ? 'ready' : warmup ? 'warming' : 'snapshot'

  return (
    <section
      className={cn(
        'shrink-0 border-b border-border-default px-4 py-2.5 text-[11px] animate-fade-in',
        tone === 'ready'
          ? 'bg-linear-to-r from-bg-elevated via-bg-surface to-bg-elevated text-text-secondary'
          : tone === 'warming'
            ? 'bg-linear-to-r from-[#fff6dd] via-bg-surface to-[#e7f2ff] text-text-primary'
            : 'bg-linear-to-r from-[#fff6dd] via-bg-surface to-bg-elevated text-text-primary',
      )}
    >
      {/* Status line */}
      <div className="flex items-center gap-2">
        <StatusIcon
          className={cn(
            'w-3.5 h-3.5 shrink-0',
            connected ? 'text-emerald-500' : 'text-amber-600',
          )}
          strokeWidth={2}
        />
        <span className="font-semibold uppercase tracking-wider">Runtime</span>
        <span className="mx-1 text-border-default">|</span>
        <span className="text-text-muted">{title}</span>
      </div>

      {/* Endpoint details */}
      <div className="mt-1.5 flex flex-wrap items-center gap-4 font-mono text-[10px] text-text-muted">
        <span className="inline-flex items-center gap-1">
          <Server className="w-3 h-3 text-text-muted/70" strokeWidth={1.5} />
          UI: {uiBase}
        </span>
        <span className="inline-flex items-center gap-1">
          <Wifi className="w-3 h-3 text-text-muted/70" strokeWidth={1.5} />
          API: {apiBase}
        </span>
        <span className="inline-flex items-center gap-1">
          <Radio className="w-3 h-3 text-text-muted/70" strokeWidth={1.5} />
          SSE: {apiBase}/api/v1/stream/events
        </span>
      </div>
    </section>
  )
}

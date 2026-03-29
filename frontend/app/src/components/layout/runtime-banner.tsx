import { useDashboardStore } from '@/stores/use-dashboard-store'
import { useUIStore } from '@/stores/use-ui-store'
import { cn } from '@/lib/utils'
import { AlertCircle, CheckCircle2, Radio, Server, Wifi } from 'lucide-react'

function buildUiBase(): string {
  if (typeof window === 'undefined') return 'http://127.0.0.1:3000'
  const { protocol, hostname, port } = window.location
  if (!port || port === '3000') return `${protocol}//${hostname}:3000`
  return `${protocol}//${hostname}:${port}`
}

function buildApiBase(): string {
  if (typeof window === 'undefined') return 'http://127.0.0.1:8000'
  const { protocol, hostname } = window.location
  return `${protocol}//${hostname}:8000`
}

export function RuntimeBanner() {
  const connected = useUIStore((s) => s.connected)
  const orchestrator = useDashboardStore((s) => s.orchestrator)
  const hardware = useDashboardStore((s) => s.hardware)

  if (connected && orchestrator && hardware) {
    return null
  }

  const uiBase = buildUiBase()
  const apiBase = buildApiBase()

  const title = connected
    ? 'Connected to stream. Waiting for telemetry payload...'
    : 'Live feed disconnected. Start backend stream to hydrate dashboard.'

  const StatusIcon = connected ? CheckCircle2 : AlertCircle

  return (
    <section
      className={cn(
        'shrink-0 border-b border-border-default px-4 py-2.5 text-[11px] animate-fade-in',
        connected
          ? 'bg-linear-to-r from-bg-elevated via-bg-surface to-bg-elevated text-text-secondary'
          : 'bg-linear-to-r from-alert-critical/20 via-bg-surface to-alert-high/15 text-text-primary',
      )}
    >
      {/* Status line */}
      <div className="flex items-center gap-2">
        <StatusIcon
          className={cn(
            'w-3.5 h-3.5 shrink-0',
            connected ? 'text-green-400' : 'text-red-400',
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

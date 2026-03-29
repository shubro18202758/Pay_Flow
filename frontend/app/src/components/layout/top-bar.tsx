// ============================================================================
// Top Bar -- "PAYFLOW INTELLIGENCE CENTER" + UBI branding + institutional chrome
// ============================================================================

import { useEffect, useState } from 'react'
import { useUIStore } from '@/stores/use-ui-store'
import { ConnectionStatus } from '@/components/shared/connection-status'
import { Shield, Radio, Home } from 'lucide-react'

export function TopBar() {
  const [clock, setClock] = useState(formatClock())
  const [date, setDate] = useState(formatDate())

  useEffect(() => {
    const interval = setInterval(() => {
      setClock(formatClock())
      setDate(formatDate())
    }, 1000)
    return () => clearInterval(interval)
  }, [])

  const connected = useUIStore((s) => s.connected)

  return (
    <>
      {/* Accent gradient line at the very top */}
      <div className="h-[2px] top-accent-line shrink-0" />

      <header className="flex items-center justify-between px-5 h-11 bg-bg-surface/80 border-b border-border-default shrink-0 backdrop-blur-sm">
        {/* Left: Brand */}
        <div className="flex items-center gap-3">
          <a
            href="/landing"
            title="Landing Page"
            target="_self"
            className="flex items-center justify-center w-7 h-7 rounded-lg bg-accent-primary/10 border border-accent-primary/20 hover:bg-accent-primary/20 transition-colors"
          >
            <Shield className="w-4 h-4 text-accent-primary" />
          </a>
          <div className="flex flex-col">
            <span className="text-[11px] font-bold tracking-[0.2em] uppercase text-text-primary leading-tight">
              PayFlow <span className="text-gradient-brand">Intelligence Center</span>
            </span>
            <span className="text-[8px] font-medium tracking-[0.15em] uppercase text-text-muted leading-tight">
              Fraud Detection & Response Platform
            </span>
          </div>
        </div>

        {/* Center: Institutional Identity */}
        <div className="flex items-center gap-2">
          <div className="h-4 w-px bg-border-subtle" />
          <span className="text-[9px] font-semibold tracking-[0.2em] uppercase text-text-muted">
            Union Bank of India
          </span>
          <div className="h-4 w-px bg-border-subtle" />
        </div>

        {/* Right: Status + Clock */}
        <div className="flex items-center gap-4">
          <ConnectionStatus connected={connected} />
          <div className="h-4 w-px bg-border-subtle" />
          <div className="flex items-center gap-2">
            <Radio className="w-3 h-3 text-text-muted" />
            <div className="flex flex-col items-end">
              <span className="text-[11px] font-mono font-semibold text-text-primary tabular-nums leading-tight">
                {clock}
              </span>
              <span className="text-[8px] font-mono text-text-muted uppercase leading-tight">
                {date}
              </span>
            </div>
          </div>
        </div>
      </header>
    </>
  )
}

function formatClock(): string {
  return new Date().toLocaleTimeString('en-IN', {
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
    hour12: false,
  })
}

function formatDate(): string {
  return new Date().toLocaleDateString('en-IN', {
    day: '2-digit',
    month: 'short',
    year: 'numeric',
  })
}

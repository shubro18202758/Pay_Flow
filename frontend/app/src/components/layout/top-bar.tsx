// ============================================================================
// Top Bar -- Union Bank-facing institutional shell
// ============================================================================

import { useCallback, useEffect, useState } from 'react'
import { useUIStore } from '@/stores/use-ui-store'
import { ConnectionStatus } from '@/components/shared/connection-status'
import { ROLE_POLICIES, canAccessTab, rolePolicy, type PayflowRole } from '@/lib/rbac'
import { Building2, LockKeyhole, Radio, Search, ShieldCheck, UserRound } from 'lucide-react'

const PRIMARY_NAV = [
  { tab: 'pre-fraud-intel', label: 'Pre-Fraud Intel' },
  { tab: 'overview', label: 'Fund-Flow' },
  { tab: 'threat-sim', label: 'Adaptive Event Lab' },
  { tab: 'investigations', label: 'Investigator Workbench' },
  { tab: 'compliance', label: 'FIU Reporting' },
] as const

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
  const setActiveTab = useUIStore((s) => s.setActiveTab)
  const currentRole = useUIStore((s) => s.currentRole)
  const setCurrentRole = useUIStore((s) => s.setCurrentRole)
  const activePolicy = rolePolicy(currentRole)
  const skipToMain = useCallback(() => {
    const main = document.getElementById('main-content')
    if (main instanceof HTMLElement) {
      main.focus({ preventScroll: false })
      main.scrollIntoView({ block: 'start' })
    }
  }, [])

  return (
    <>
      <div className="ubi-access-strip hidden shrink-0 items-center justify-between px-5 text-[10px] font-medium lg:flex">
        <div className="flex items-center gap-4">
          <button
            onClick={skipToMain}
            className="font-semibold text-accent-primary underline-offset-4 hover:underline focus:outline-none focus:ring-2 focus:ring-accent-primary/30"
          >
            Skip to main content
          </button>
          <span className="h-3 w-px bg-border-subtle" />
          <span>Backend-synced fraud operations workspace</span>
        </div>
        <div className="flex items-center gap-3">
          <a href="/landing" className="rounded-full bg-accent-primary px-3 py-1 font-bold text-white">Landing Page</a>
          <a href="/docs" className="font-semibold text-accent-primary underline-offset-4 hover:underline">API Docs</a>
        </div>
      </div>

      <header className="ubi-official-header flex h-16 shrink-0 items-center justify-between border-b border-border-default px-5">
        <a href="/landing" title="Union Bank PayFlow portal" target="_self" className="flex min-w-0 items-center gap-3">
          <span className="ubi-brand-mark shrink-0">
            <span className="sr-only">Union Bank of India</span>
          </span>
          <span className="flex min-w-0 flex-col">
            <span className="ubi-wordmark-title truncate text-[16px] leading-tight">
              Union Bank <span>of India</span>
            </span>
            <span className="truncate text-[9px] font-semibold uppercase tracking-[0.12em] text-text-muted leading-tight">
              A Government of India Undertaking | PayFlow Fraud Intelligence
            </span>
          </span>
        </a>

        <div className="hidden min-w-0 flex-1 items-center justify-center px-4 2xl:flex">
          <div className="ubi-pill-menu flex items-center gap-4 px-5 py-2 text-[11px] font-bold">
            {PRIMARY_NAV.map((item) => {
              const allowed = canAccessTab(currentRole, item.tab)
              return (
                <button
                  key={item.tab}
                  type="button"
                  onClick={() => setActiveTab(item.tab)}
                  disabled={!allowed}
                  title={allowed ? item.label : `${activePolicy.label} cannot access ${item.label}`}
                  className="inline-flex items-center gap-1.5 disabled:cursor-not-allowed disabled:opacity-45"
                >
                  {!allowed && <LockKeyhole className="h-3 w-3" />}
                  {item.label}
                </button>
              )
            })}
          </div>
        </div>

        <div className="flex shrink-0 items-center gap-3">
          <a href="/docs" className="hidden items-center gap-2 rounded-full bg-bg-elevated px-3 py-1.5 text-[10px] font-semibold text-text-secondary lg:flex">
            <Search className="h-3.5 w-3.5" />
            API Docs
          </a>
          <div className="hidden items-center gap-2 md:flex">
            <Building2 className="h-3.5 w-3.5 text-accent-primary" />
            <span className="text-[9px] font-bold uppercase tracking-[0.16em] text-text-secondary">
              Good people to bank with
            </span>
          </div>
          <label
            title={`${activePolicy.domain}: ${activePolicy.escalationScope}`}
            className="hidden items-center gap-2 rounded-full border border-border-subtle bg-bg-elevated px-2.5 py-1 text-[9px] font-bold uppercase tracking-[0.12em] text-text-secondary xl:flex"
          >
            <UserRound className="h-3.5 w-3.5 text-accent-primary" />
            <span className="sr-only">Union Bank role</span>
            <select
              value={currentRole}
              onChange={(event) => setCurrentRole(event.target.value as PayflowRole)}
              className="max-w-[180px] bg-transparent text-[9px] font-bold uppercase tracking-[0.12em] text-text-primary outline-none"
            >
              {Object.values(ROLE_POLICIES).map((policy) => (
                <option key={policy.role} value={policy.role}>
                  {policy.label}
                </option>
              ))}
            </select>
          </label>
          <div className="hidden h-4 w-px bg-border-subtle md:block" />
          <ShieldCheck className="hidden h-3.5 w-3.5 text-alert-low md:block" />
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

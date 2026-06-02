// ============================================================================
// Tab Navigation -- Union Bank product navigation with live alert badges
// ============================================================================

import { useEffect, useCallback } from 'react'
import { useUIStore, type TabId } from '@/stores/use-ui-store'
import { useDashboardStore } from '@/stores/use-dashboard-store'
import { useCountermeasureProposals, useEscalations, useIntelTuningStatus } from '@/hooks/use-api'
import { cn } from '@/lib/utils'
import { canAccessTab, rolePolicy } from '@/lib/rbac'
import {
  LayoutDashboard,
  Crosshair,
  Scale,
  BrainCircuit,
  BarChart3,
  Cpu,
  ShieldCheck,
  Radar,
  LockKeyhole,
} from 'lucide-react'
import type { LucideIcon } from 'lucide-react'

const TABS: { id: TabId; label: string; shortLabel: string; icon: LucideIcon; key: string }[] = [
  { id: 'pre-fraud-intel', label: 'Pre-Fraud Intel', shortLabel: 'Pre-Fraud', icon: Radar, key: '1' },
  { id: 'overview', label: 'Fund-Flow Overview', shortLabel: 'Overview', icon: LayoutDashboard, key: '2' },
  { id: 'threat-sim', label: 'Adaptive Event Lab', shortLabel: 'Event Lab', icon: Crosshair, key: '3' },
  { id: 'investigations', label: 'Investigations', shortLabel: 'Investigate', icon: Scale, key: '4' },
  { id: 'intelligence', label: 'Intelligence & Integrity', shortLabel: 'Intel', icon: BrainCircuit, key: '5' },
  { id: 'analytics', label: 'Analytics', shortLabel: 'Analytics', icon: BarChart3, key: '6' },
  { id: 'compliance', label: 'Compliance & Regulatory', shortLabel: 'Comply', icon: ShieldCheck, key: '7' },
  { id: 'system', label: 'System', shortLabel: 'System', icon: Cpu, key: '8' },
]

export function TabNav() {
  const activeTab = useUIStore((s) => s.activeTab)
  const setActiveTab = useUIStore((s) => s.setActiveTab)
  const currentRole = useUIStore((s) => s.currentRole)
  const frozenCount = useDashboardStore((s) => s.frozenCount)
  const pendingAlerts = useDashboardStore((s) => s.pendingAlerts)
  const { data: intelStatus } = useIntelTuningStatus()
  const { data: escalations } = useEscalations()
  const { data: countermeasures } = useCountermeasureProposals()

  // Preserve direct keyboard navigation without rendering shortcut hints.
  const handleKeyDown = useCallback(
    (e: KeyboardEvent) => {
      if (e.altKey && !e.ctrlKey && !e.metaKey) {
        const tab = TABS.find((t) => t.key === e.key)
        if (tab) {
          e.preventDefault()
          setActiveTab(tab.id)
        }
      }
    },
    [setActiveTab],
  )

  useEffect(() => {
    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [handleKeyDown])

  // Badge counts per tab
  const badgeCounts: Partial<Record<TabId, number>> = {}
  const pendingEscalations = (escalations ?? []).filter((item) => item.status === 'pending_review').length
  const pendingCountermeasures = (countermeasures?.proposals ?? []).filter((item) => item.status === 'proposed').length
  if (frozenCount > 0 || pendingAlerts > 0) badgeCounts['overview'] = frozenCount + pendingAlerts
  if (pendingEscalations > 0) badgeCounts['investigations'] = pendingEscalations
  if (pendingCountermeasures > 0) badgeCounts['threat-sim'] = pendingCountermeasures
  if ((intelStatus?.active_playbooks ?? 0) > 0) badgeCounts['pre-fraud-intel'] = intelStatus?.active_playbooks ?? 0

  return (
    <nav className="flex shrink-0 items-center overflow-x-auto border-b border-border-default bg-bg-surface px-2 py-1.5 shadow-sm">
      {TABS.map((tab) => {
        const Icon = tab.icon
        const allowed = canAccessTab(currentRole, tab.id)
        const policy = rolePolicy(currentRole)
        const isActive = activeTab === tab.id
        const badge = badgeCounts[tab.id]
        return (
          <button
            key={tab.id}
            onClick={() => allowed && setActiveTab(tab.id)}
            disabled={!allowed}
            aria-label={allowed ? tab.label : `${tab.label} restricted for ${policy.label}`}
            aria-disabled={!allowed}
            aria-current={isActive ? 'page' : undefined}
            title={allowed ? tab.label : `${policy.label} cannot access ${tab.label}`}
            className={cn(
              'group relative flex shrink-0 items-center gap-2 px-3 py-2.5 text-[10px] font-bold uppercase tracking-[0.1em] transition-all duration-150',
              'relative rounded-full border',
              !allowed
                ? 'cursor-not-allowed border-transparent text-text-muted/35 opacity-60'
                : isActive
                ? 'border-accent-primary bg-accent-primary text-white shadow-sm'
                : 'border-transparent text-text-muted hover:border-border-subtle hover:bg-bg-elevated/70 hover:text-accent-primary',
            )}
          >
            <Icon className={cn(
              'w-3.5 h-3.5 transition-colors',
              isActive ? 'text-white' : 'text-text-muted group-hover:text-accent-primary',
            )} />
            <span className="hidden 2xl:inline">{tab.label}</span>
            <span className="2xl:hidden">{tab.shortLabel}</span>
            {!allowed && <LockKeyhole className="h-3 w-3 text-text-muted/45" />}
            {/* Alert badge */}
            {badge != null && badge > 0 && (
              <span className="flex h-[14px] min-w-[14px] items-center justify-center rounded-full bg-alert-critical px-1 font-mono text-[7px] font-bold text-white animate-data-pulse">
                {badge > 99 ? '99+' : badge}
              </span>
            )}
            {isActive && (
              <span className="absolute -bottom-1 left-1/2 h-0.5 w-8 -translate-x-1/2 rounded-full bg-alert-critical" />
            )}
          </button>
        )
      })}
    </nav>
  )
}

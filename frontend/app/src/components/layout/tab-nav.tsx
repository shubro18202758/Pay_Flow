// ============================================================================
// Tab Navigation -- SOC command center style navigation tabs with icons
// Enhanced: keyboard shortcuts, alert count badges, active indicator animation
// ============================================================================

import { useEffect, useCallback } from 'react'
import { useUIStore, type TabId } from '@/stores/use-ui-store'
import { useDashboardStore } from '@/stores/use-dashboard-store'
import { useIntelTuningStatus } from '@/hooks/use-api'
import { cn } from '@/lib/utils'
import {
  LayoutDashboard,
  Crosshair,
  Scale,
  BrainCircuit,
  BarChart3,
  Cpu,
  ShieldCheck,
  Radar,
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
  const frozenCount = useDashboardStore((s) => s.frozenCount)
  const pendingAlerts = useDashboardStore((s) => s.pendingAlerts)
  const agentLogLen = useDashboardStore((s) => s.agentLog.length)
  const { data: intelStatus } = useIntelTuningStatus()

  // Keyboard shortcut: Alt+1..5
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
  if (frozenCount > 0 || pendingAlerts > 0) badgeCounts['overview'] = frozenCount + pendingAlerts
  if (agentLogLen > 0) badgeCounts['investigations'] = agentLogLen
  if ((intelStatus?.active_playbooks ?? 0) > 0) badgeCounts['pre-fraud-intel'] = intelStatus?.active_playbooks ?? 0

  return (
    <nav className="flex shrink-0 items-center overflow-x-auto border-b border-border-default bg-bg-surface px-2 py-1.5 shadow-sm">
      {TABS.map((tab) => {
        const Icon = tab.icon
        const isActive = activeTab === tab.id
        const badge = badgeCounts[tab.id]
        return (
          <button
            key={tab.id}
            onClick={() => setActiveTab(tab.id)}
            aria-label={tab.label}
            aria-current={isActive ? 'page' : undefined}
            title={`${tab.label} (Alt+${tab.key})`}
            className={cn(
              'group relative flex shrink-0 items-center gap-2 px-3 py-2.5 text-[10px] font-bold uppercase tracking-[0.1em] transition-all duration-150',
              'relative rounded-full border',
              isActive
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
            {/* Keyboard hint */}
            <span className={cn(
              'text-[7px] font-mono px-1 py-0.5 rounded border leading-none ml-0.5 transition-colors',
              isActive
                ? 'border-white/40 bg-white/15 text-white/80'
                : 'border-border-subtle text-text-muted/40 group-hover:border-border-default group-hover:text-text-muted/60',
            )}>
              {tab.key}
            </span>
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

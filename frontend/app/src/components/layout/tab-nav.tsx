// ============================================================================
// Tab Navigation -- SOC command center style navigation tabs with icons
// Enhanced: keyboard shortcuts, alert count badges, active indicator animation
// ============================================================================

import { useEffect, useCallback } from 'react'
import { useUIStore, type TabId } from '@/stores/use-ui-store'
import { useDashboardStore } from '@/stores/use-dashboard-store'
import { cn } from '@/lib/utils'
import {
  LayoutDashboard,
  Crosshair,
  Scale,
  BrainCircuit,
  BarChart3,
  Cpu,
  ShieldCheck,
} from 'lucide-react'
import type { LucideIcon } from 'lucide-react'

const TABS: { id: TabId; label: string; shortLabel: string; icon: LucideIcon; key: string }[] = [
  { id: 'overview', label: 'Overview', shortLabel: 'Overview', icon: LayoutDashboard, key: '1' },
  { id: 'threat-sim', label: 'Threat Simulation', shortLabel: 'Threats', icon: Crosshair, key: '2' },
  { id: 'investigations', label: 'Investigations', shortLabel: 'Investigate', icon: Scale, key: '3' },
  { id: 'intelligence', label: 'Intelligence & Integrity', shortLabel: 'Intel', icon: BrainCircuit, key: '4' },
  { id: 'analytics', label: 'Analytics', shortLabel: 'Analytics', icon: BarChart3, key: '5' },
  { id: 'compliance', label: 'Compliance & Regulatory', shortLabel: 'Comply', icon: ShieldCheck, key: '6' },
  { id: 'system', label: 'System', shortLabel: 'System', icon: Cpu, key: '7' },
]

export function TabNav() {
  const activeTab = useUIStore((s) => s.activeTab)
  const setActiveTab = useUIStore((s) => s.setActiveTab)
  const frozenCount = useDashboardStore((s) => s.frozenCount)
  const pendingAlerts = useDashboardStore((s) => s.pendingAlerts)
  const agentLogLen = useDashboardStore((s) => s.agentLog.length)

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

  return (
    <nav className="flex items-center bg-bg-surface border-b border-border-default shrink-0">
      {TABS.map((tab) => {
        const Icon = tab.icon
        const isActive = activeTab === tab.id
        const badge = badgeCounts[tab.id]
        return (
          <button
            key={tab.id}
            onClick={() => setActiveTab(tab.id)}
            title={`${tab.label} (Alt+${tab.key})`}
            className={cn(
              'group flex items-center gap-2 px-4 py-2.5 text-[10px] font-semibold uppercase tracking-[0.12em] transition-all duration-150',
              'border-b-2 -mb-px relative',
              isActive
                ? 'text-text-primary border-accent-primary bg-bg-elevated/60'
                : 'text-text-muted border-transparent hover:text-text-secondary hover:bg-bg-elevated/30 hover:border-border-subtle',
            )}
          >
            <Icon className={cn(
              'w-3.5 h-3.5 transition-colors',
              isActive ? 'text-accent-primary' : 'text-text-muted group-hover:text-text-secondary',
            )} />
            {tab.label}
            {/* Keyboard hint */}
            <span className={cn(
              'text-[7px] font-mono px-1 py-0.5 rounded border leading-none ml-0.5 transition-colors',
              isActive
                ? 'border-accent-primary/30 text-accent-primary/60'
                : 'border-border-subtle text-text-muted/40 group-hover:border-border-default group-hover:text-text-muted/60',
            )}>
              {tab.key}
            </span>
            {/* Alert badge */}
            {badge != null && badge > 0 && (
              <span className="flex items-center justify-center min-w-[14px] h-[14px] text-[7px] font-bold font-mono rounded-full bg-alert-critical text-white px-1 animate-data-pulse">
                {badge > 99 ? '99+' : badge}
              </span>
            )}
            {isActive && (
              <span className="absolute bottom-0 left-1/2 -translate-x-1/2 w-6 h-px bg-accent-primary blur-sm" />
            )}
          </button>
        )
      })}
    </nav>
  )
}

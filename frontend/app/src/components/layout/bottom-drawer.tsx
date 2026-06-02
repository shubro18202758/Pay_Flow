// ============================================================================
// Bottom Drawer -- Expandable panels: Agent Evidence | Circuit Breaker | Metrics
// ============================================================================

import { useEffect } from 'react'
import { useUIStore } from '@/stores/use-ui-store'
import { AgentInvestigationTrace } from '@/components/panels/agent-cot'
import { CircuitBreakerPanel } from '@/components/panels/circuit-breaker'
import { SystemMetrics } from '@/components/panels/system-metrics'
import { EventDetailDrawer } from '@/components/panels/event-detail-drawer'
import { PreFraudIntelBrief } from '@/components/panels/pre-fraud-intel-brief'
import { cn } from '@/lib/utils'
import { useRoleAccess } from '@/hooks/use-rbac'
import type { Permission } from '@/lib/rbac'
import { BrainCircuit, Zap, Activity, Fingerprint, ChevronDown, ChevronRight, Radar, LockKeyhole } from 'lucide-react'

const DRAWERS = [
  { id: 'agent', label: 'Agent Evidence', icon: BrainCircuit, component: AgentInvestigationTrace, permission: 'explain:view' },
  { id: 'pre-fraud-intel', label: 'Pre-Fraud Intel', icon: Radar, component: PreFraudDrawerPanel, permission: 'intel:view' },
  { id: 'circuit-breaker', label: 'Circuit Breaker', icon: Zap, component: CircuitBreakerPanel, permission: 'circuit_breaker:trigger' },
  { id: 'system-metrics', label: 'System Metrics', icon: Activity, component: SystemMetrics, permission: 'system:view' },
  { id: 'event-inspector', label: 'Event Inspector', icon: Fingerprint, component: EventDetailDrawer, permission: 'ops:view' },
] as const

function PreFraudDrawerPanel() {
  return (
    <div className="h-full overflow-auto p-3">
      <PreFraudIntelBrief variant="drawer" />
    </div>
  )
}

export function BottomDrawer() {
  const expanded = useUIStore((s) => s.expandedDrawers)
  const toggle = useUIStore((s) => s.toggleDrawer)
  const selectedEventId = useUIStore((s) => s.selectedEventId)
  const access = useRoleAccess()

  const hasExpanded = DRAWERS.some((d) => expanded.has(d.id))

  useEffect(() => {
    const restrictedOpen = DRAWERS.some((drawer) => (
      expanded.has(drawer.id) && !access.can(drawer.permission as Permission)
    ))
    if (!restrictedOpen) return
    useUIStore.setState((state) => {
      const next = new Set(state.expandedDrawers)
      DRAWERS.forEach((drawer) => {
        if (!access.can(drawer.permission as Permission)) next.delete(drawer.id)
      })
      return { expandedDrawers: next }
    })
  }, [access, expanded])

  return (
    <div
      className={cn(
        'border-t border-border-default bg-bg-surface shrink-0 transition-[height] duration-300 ease-in-out relative',
        hasExpanded ? 'h-64' : 'h-9',
      )}
    >
      {/* Subtle top border gradient when expanded */}
      {hasExpanded && (
        <div className="absolute top-0 left-0 right-0 h-px bg-gradient-to-r from-accent-primary/0 via-accent-primary/40 to-accent-primary/0" />
      )}

      {/* Tab headers */}
      <div className="flex items-center h-9 border-b border-border-subtle gap-0">
        {DRAWERS.map((drawer) => {
          const Icon = drawer.icon
          const isOpen = expanded.has(drawer.id)
          const needsSelectedEvent = drawer.id === 'event-inspector'
          const permission = drawer.permission as Permission
          const restricted = !access.can(permission)
          const disabled = restricted || (needsSelectedEvent && !selectedEventId)
          const title = restricted
            ? `${access.policy.label} cannot open ${drawer.label}`
            : needsSelectedEvent && !selectedEventId
            ? 'Select an event from Live Activity first'
            : drawer.label
          return (
            <button
              key={drawer.id}
              onClick={() => {
                if (!disabled) toggle(drawer.id)
              }}
              disabled={disabled}
              title={title}
              className={cn(
                'flex items-center gap-1.5 px-3.5 h-full text-[10px] font-semibold uppercase tracking-[0.12em] transition-all duration-300',
                'border-r border-border-subtle',
                disabled
                  ? 'cursor-not-allowed text-text-muted/35 bg-bg-surface'
                  : isOpen
                  ? 'text-text-primary bg-bg-elevated shadow-[inset_0_-2px_0_0_theme(colors.accent.primary)]'
                  : 'text-text-muted hover:text-text-secondary hover:bg-bg-elevated/40',
              )}
            >
              {isOpen
                ? <ChevronDown className="w-3 h-3 text-accent-primary shrink-0" />
                : <ChevronRight className="w-3 h-3 shrink-0" />
              }
              {restricted ? (
                <LockKeyhole className="w-3.5 h-3.5 shrink-0 text-alert-critical/70" />
              ) : (
                <Icon className={cn(
                  'w-3.5 h-3.5 shrink-0 transition-colors duration-300',
                  isOpen ? 'text-accent-primary' : 'text-text-muted',
                )} />
              )}
              {drawer.label}
            </button>
          )
        })}
      </div>

      {/* Panel content */}
      {hasExpanded && (
        <div className="flex h-[calc(100%-2.25rem)] overflow-hidden animate-fade-in">
          {DRAWERS.filter((d) => expanded.has(d.id)).map((drawer) => {
            const Component = drawer.component
            return (
              <div
                key={drawer.id}
                className="flex-1 border-r border-border-subtle last:border-r-0 min-w-0 animate-slide-up"
              >
                <Component />
              </div>
            )
          })}
        </div>
      )}
    </div>
  )
}

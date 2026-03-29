// ============================================================================
// Bottom Drawer -- Expandable panels: Agent CoT | Circuit Breaker | Metrics
// ============================================================================

import { useUIStore } from '@/stores/use-ui-store'
import { AgentCoT } from '@/components/panels/agent-cot'
import { CircuitBreakerPanel } from '@/components/panels/circuit-breaker'
import { SystemMetrics } from '@/components/panels/system-metrics'
import { EventDetailDrawer } from '@/components/panels/event-detail-drawer'
import { cn } from '@/lib/utils'
import { BrainCircuit, Zap, Activity, Fingerprint, ChevronDown, ChevronRight } from 'lucide-react'

const DRAWERS = [
  { id: 'agent', label: 'Agent Investigation', icon: BrainCircuit, component: AgentCoT },
  { id: 'circuit-breaker', label: 'Circuit Breaker', icon: Zap, component: CircuitBreakerPanel },
  { id: 'system-metrics', label: 'System Metrics', icon: Activity, component: SystemMetrics },
  { id: 'event-inspector', label: 'Event Inspector', icon: Fingerprint, component: EventDetailDrawer },
] as const

export function BottomDrawer() {
  const expanded = useUIStore((s) => s.expandedDrawers)
  const toggle = useUIStore((s) => s.toggleDrawer)

  const hasExpanded = DRAWERS.some((d) => expanded.has(d.id))

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
          return (
            <button
              key={drawer.id}
              onClick={() => toggle(drawer.id)}
              className={cn(
                'flex items-center gap-1.5 px-3.5 h-full text-[10px] font-semibold uppercase tracking-[0.12em] transition-all duration-300',
                'border-r border-border-subtle',
                isOpen
                  ? 'text-text-primary bg-bg-elevated shadow-[inset_0_-2px_0_0_theme(colors.accent.primary)]'
                  : 'text-text-muted hover:text-text-secondary hover:bg-bg-elevated/40',
              )}
            >
              {isOpen
                ? <ChevronDown className="w-3 h-3 text-accent-primary shrink-0" />
                : <ChevronRight className="w-3 h-3 shrink-0" />
              }
              <Icon className={cn(
                'w-3.5 h-3.5 shrink-0 transition-colors duration-300',
                isOpen ? 'text-accent-primary' : 'text-text-muted',
              )} />
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

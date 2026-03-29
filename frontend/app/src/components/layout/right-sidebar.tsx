// ============================================================================
// Right Sidebar -- Collapsible: Live Activity Feed + Quick Stats
// ============================================================================

import { useUIStore } from '@/stores/use-ui-store'
import { LiveActivityFeed } from '@/components/panels/live-activity-feed'
import { QuickStats } from '@/components/panels/quick-stats'
import { cn } from '@/lib/utils'
import { PanelRightClose, PanelRightOpen } from 'lucide-react'

export function RightSidebar() {
  const collapsed = useUIStore((s) => s.sidebarCollapsed)
  const toggle = useUIStore((s) => s.toggleSidebar)

  return (
    <div
      className={cn(
        'flex flex-col border-l border-border-default bg-bg-surface transition-[width] duration-300 ease-in-out overflow-hidden shrink-0 relative',
        collapsed ? 'w-10' : 'w-80',
      )}
    >
      {/* Subtle left border accent */}
      <div className="absolute top-0 left-0 bottom-0 w-px bg-gradient-to-b from-accent-primary/0 via-accent-primary/20 to-accent-primary/0" />

      {/* Collapse toggle */}
      <button
        onClick={toggle}
        className={cn(
          'flex items-center justify-center h-8 border-b border-border-subtle shrink-0',
          'text-text-muted hover:text-text-primary hover:bg-bg-elevated/50 transition-all duration-300',
        )}
        title={collapsed ? 'Expand sidebar' : 'Collapse sidebar'}
      >
        {collapsed
          ? <PanelRightOpen className="w-3.5 h-3.5" />
          : <PanelRightClose className="w-3.5 h-3.5" />
        }
      </button>

      {/* Collapsed state indicator */}
      {collapsed && (
        <div className="flex flex-col items-center gap-3 pt-4 animate-fade-in">
          <div className="w-1.5 h-1.5 rounded-full bg-accent-primary/60" />
          <div className="w-1.5 h-1.5 rounded-full bg-accent-primary/40" />
          <div className="w-1.5 h-1.5 rounded-full bg-accent-primary/20" />
        </div>
      )}

      {!collapsed && (
        <div className="flex flex-col flex-1 min-h-0 animate-fade-in">
          {/* Live Activity Feed */}
          <div className="flex-1 min-h-0">
            <LiveActivityFeed />
          </div>

          {/* Section divider */}
          <div className="mx-3 h-px bg-gradient-to-r from-border-subtle/0 via-border-default to-border-subtle/0" />

          {/* Quick Stats */}
          <div className="shrink-0">
            <QuickStats />
          </div>
        </div>
      )}
    </div>
  )
}

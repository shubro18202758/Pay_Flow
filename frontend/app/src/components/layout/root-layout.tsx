// ============================================================================
// Root Layout -- Full-screen flex: top-bar + content + bottom-drawer
// ============================================================================

import { TopBar } from './top-bar'
import { TabNav } from './tab-nav'
import { BottomDrawer } from './bottom-drawer'
import { RuntimeBanner } from './runtime-banner'
import { RoleAccessBanner } from './role-access-banner'
import { OperationalRealityStrip } from './operational-reality-strip'
import { LivePipelineRibbon } from './live-pipeline-ribbon'
import { EventEvaluationSyncStrip } from './event-evaluation-sync-strip'
import { RoleContextSwitcher, type RoleContextPanel } from './role-context-switcher'
import { useUIStore, type TabId } from '@/stores/use-ui-store'
import { useState, type ReactNode } from 'react'

interface Props {
  children: ReactNode
}

export function RootLayout({ children }: Props) {
  const activeTab = useUIStore((s) => s.activeTab)
  const [contextState, setContextState] = useState<{ panel: RoleContextPanel; tab: TabId | null }>({
    panel: 'closed',
    tab: null,
  })
  const contextPanel = contextState.tab === activeTab ? contextState.panel : 'closed'
  const setContextPanel = (panel: RoleContextPanel) => setContextState({ panel, tab: activeTab })

  return (
    <div className="ubi-app-shell flex h-screen flex-col overflow-hidden bg-bg-deep">
      <TopBar />
      <TabNav />
      <RuntimeBanner />
      <RoleContextSwitcher panel={contextPanel} onPanelChange={setContextPanel} />
      {contextPanel !== 'closed' && (
        <div className="shrink-0 max-h-[315px] overflow-y-auto border-b border-[#c6d3e3] bg-[#edf5ff]">
          {contextPanel === 'access' ? <RoleAccessBanner /> : <OperationalRealityStrip />}
        </div>
      )}
      <LivePipelineRibbon />
      <EventEvaluationSyncStrip />
      <main
        id="main-content"
        tabIndex={-1}
        className="ubi-main-surface custom-scrollbar relative flex-1 min-h-0 overflow-y-auto overflow-x-hidden focus:outline-none"
      >
        {children}
      </main>
      <BottomDrawer />
    </div>
  )
}

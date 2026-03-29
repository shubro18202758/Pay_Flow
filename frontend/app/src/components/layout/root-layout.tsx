// ============================================================================
// Root Layout -- Full-screen flex: top-bar + content + bottom-drawer
// ============================================================================

import { TopBar } from './top-bar'
import { TabNav } from './tab-nav'
import { BottomDrawer } from './bottom-drawer'
import { RuntimeBanner } from './runtime-banner'
import type { ReactNode } from 'react'

interface Props {
  children: ReactNode
}

export function RootLayout({ children }: Props) {
  return (
    <div className="flex flex-col h-screen overflow-hidden bg-bg-deep">
      <TopBar />
      <TabNav />
      <RuntimeBanner />
      <main className="flex-1 min-h-0 overflow-hidden relative">
        {children}
      </main>
      <BottomDrawer />
    </div>
  )
}

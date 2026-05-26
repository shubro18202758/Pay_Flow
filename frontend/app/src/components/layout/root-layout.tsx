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
    <div className="ubi-app-shell flex h-screen flex-col overflow-hidden bg-bg-deep">
      <TopBar />
      <TabNav />
      <RuntimeBanner />
      <main className="ubi-main-surface flex-1 min-h-0 overflow-hidden relative">
        {children}
      </main>
      <BottomDrawer />
    </div>
  )
}

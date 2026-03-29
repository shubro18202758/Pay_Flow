// ============================================================================
// App.tsx -- Root component: QueryClientProvider + SSE + tab routing
// ============================================================================

import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { RootLayout } from '@/components/layout/root-layout'
import { useSSE } from '@/hooks/use-sse'
import { useDashboardHydration } from '@/hooks/use-dashboard-hydration'
import { useUIStore } from '@/stores/use-ui-store'
import { OverviewPage } from '@/pages/overview'
import { ThreatSimPage } from '@/pages/threat-sim'
import { InvestigationsPage } from '@/pages/investigations'
import { SystemPage } from '@/pages/system'
import { IntelligencePage } from '@/pages/intelligence'
import { AnalyticsPage } from '@/pages/analytics'
import { CompliancePage } from '@/pages/compliance'

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      retry: 2,
      refetchOnWindowFocus: false,
    },
  },
})

function AppContent() {
  // Initialize SSE connection
  useSSE()
  useDashboardHydration()

  const activeTab = useUIStore((s) => s.activeTab)

  return (
    <RootLayout>
      {activeTab === 'overview' && <OverviewPage />}
      {activeTab === 'threat-sim' && <ThreatSimPage />}
      {activeTab === 'investigations' && <InvestigationsPage />}
      {activeTab === 'intelligence' && <IntelligencePage />}
      {activeTab === 'analytics' && <AnalyticsPage />}
      {activeTab === 'compliance' && <CompliancePage />}
      {activeTab === 'system' && <SystemPage />}
    </RootLayout>
  )
}

export default function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <AppContent />
    </QueryClientProvider>
  )
}

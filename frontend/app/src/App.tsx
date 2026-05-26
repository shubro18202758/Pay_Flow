// ============================================================================
// App.tsx -- Root component: QueryClientProvider + SSE + tab routing
// ============================================================================

import { lazy, Suspense, useEffect, useRef } from 'react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { RootLayout } from '@/components/layout/root-layout'
import { useSSE } from '@/hooks/use-sse'
import { useDashboardHydration } from '@/hooks/use-dashboard-hydration'
import { useUIStore, type TabId } from '@/stores/use-ui-store'
import { PreFraudIntelPage } from '@/pages/pre-fraud-intel'
import { useLaunchPS3Scenario, useRefreshIntel } from '@/hooks/use-api'

const loadOverviewPage = () => import('@/pages/overview')
const loadThreatSimPage = () => import('@/pages/threat-sim')
const loadInvestigationsPage = () => import('@/pages/investigations')
const loadIntelligencePage = () => import('@/pages/intelligence')
const loadAnalyticsPage = () => import('@/pages/analytics')
const loadCompliancePage = () => import('@/pages/compliance')
const loadSystemPage = () => import('@/pages/system')

const OverviewPage = lazy(() => loadOverviewPage().then((m) => ({ default: m.OverviewPage })))
const ThreatSimPage = lazy(() => loadThreatSimPage().then((m) => ({ default: m.ThreatSimPage })))
const InvestigationsPage = lazy(() => loadInvestigationsPage().then((m) => ({ default: m.InvestigationsPage })))
const IntelligencePage = lazy(() => loadIntelligencePage().then((m) => ({ default: m.IntelligencePage })))
const AnalyticsPage = lazy(() => loadAnalyticsPage().then((m) => ({ default: m.AnalyticsPage })))
const CompliancePage = lazy(() => loadCompliancePage().then((m) => ({ default: m.CompliancePage })))
const SystemPage = lazy(() => loadSystemPage().then((m) => ({ default: m.SystemPage })))

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
  const setActiveTab = useUIStore((s) => s.setActiveTab)
  const setActiveCaseId = useUIStore((s) => s.setActiveCaseId)
  const refreshIntel = useRefreshIntel()
  const launchPS3 = useLaunchPS3Scenario()
  const handledDeepLink = useRef(false)

  useEffect(() => {
    if (handledDeepLink.current) return
    handledDeepLink.current = true
    const params = new URLSearchParams(window.location.search)
    const tab = params.get('tab') as TabId | null
    const action = params.get('action')
    const validTabs: TabId[] = ['pre-fraud-intel', 'overview', 'threat-sim', 'investigations', 'intelligence', 'analytics', 'compliance', 'system']
    if (tab && validTabs.includes(tab)) {
      setActiveTab(tab)
    }
    if (action === 'refresh_intel') {
      setActiveTab('pre-fraud-intel')
      void refreshIntel.mutateAsync(undefined)
    }
    if (action === 'launch_ps3_case' || action === 'open_evidence') {
      setActiveTab('investigations')
      void launchPS3
        .mutateAsync({ scenario: 'rapid_layering', intensity: 'demo', seed: 2026 })
        .then((response) => setActiveCaseId(response.primary_case_id))
    }
    if (tab || action) {
      window.history.replaceState(null, '', '/app')
    }
  }, [launchPS3, refreshIntel, setActiveCaseId, setActiveTab])

  useEffect(() => {
    if (activeTab !== 'pre-fraud-intel') return
    const preloadPlan = [
      { delay: 8_000, load: loadOverviewPage },
      { delay: 9_500, load: loadThreatSimPage },
      { delay: 11_000, load: loadInvestigationsPage },
      { delay: 12_500, load: loadIntelligencePage },
      { delay: 14_000, load: loadAnalyticsPage },
      { delay: 15_500, load: loadCompliancePage },
      { delay: 17_000, load: loadSystemPage },
    ]
    const idleWindow = window as Window & {
      requestIdleCallback?: (callback: () => void, options?: { timeout: number }) => number
      cancelIdleCallback?: (handle: number) => void
    }
    const timers = preloadPlan.map(({ delay, load }) => window.setTimeout(() => void load(), delay))
    if (idleWindow.requestIdleCallback) {
      const handle = idleWindow.requestIdleCallback(() => void loadOverviewPage(), { timeout: 8_000 })
      return () => {
        timers.forEach((timer) => window.clearTimeout(timer))
        idleWindow.cancelIdleCallback?.(handle)
      }
    }
    return () => timers.forEach((timer) => window.clearTimeout(timer))
  }, [activeTab])

  return (
    <RootLayout>
      <Suspense fallback={<TabLoadingFallback />}>
        {activeTab === 'overview' && <OverviewPage />}
        {activeTab === 'threat-sim' && <ThreatSimPage />}
        {activeTab === 'investigations' && <InvestigationsPage />}
        {activeTab === 'pre-fraud-intel' && <PreFraudIntelPage />}
        {activeTab === 'intelligence' && <IntelligencePage />}
        {activeTab === 'analytics' && <AnalyticsPage />}
        {activeTab === 'compliance' && <CompliancePage />}
        {activeTab === 'system' && <SystemPage />}
      </Suspense>
    </RootLayout>
  )
}

function TabLoadingFallback() {
  return (
    <div className="flex h-full items-center justify-center bg-bg-deep px-6">
      <div className="w-full max-w-3xl rounded-lg border border-border-subtle bg-bg-surface p-6 shadow-sm">
        <div className="mb-4 h-2 w-32 rounded-full bg-alert-critical" />
        <div className="text-sm font-bold uppercase tracking-[0.14em] text-text-primary">
          Loading Union Bank intelligence surface
        </div>
        <div className="mt-2 text-[11px] text-text-muted">
          Preparing the next tab from the local production bundle. Live data continues to hydrate in the background.
        </div>
        <div className="mt-5 grid grid-cols-3 gap-3">
          <div className="h-16 rounded-md border border-border-subtle bg-bg-elevated animate-pulse" />
          <div className="h-16 rounded-md border border-border-subtle bg-bg-elevated animate-pulse" />
          <div className="h-16 rounded-md border border-border-subtle bg-bg-elevated animate-pulse" />
        </div>
      </div>
    </div>
  )
}

export default function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <AppContent />
    </QueryClientProvider>
  )
}

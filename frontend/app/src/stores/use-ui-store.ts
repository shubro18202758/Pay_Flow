// ============================================================================
// UI Store -- Active tab, sidebar, drawer state, connection status
// ============================================================================

import { create } from 'zustand'

export type TabId = 'overview' | 'threat-sim' | 'investigations' | 'intelligence' | 'analytics' | 'compliance' | 'system'

interface UIState {
  activeTab: TabId
  sidebarCollapsed: boolean
  expandedDrawers: Set<string>
  connected: boolean
  selectedNodeId: string | null
  selectedEventId: string | null

  // Actions
  setActiveTab: (tab: TabId) => void
  toggleSidebar: () => void
  toggleDrawer: (id: string) => void
  setConnected: (connected: boolean) => void
  setSelectedNode: (nodeId: string | null) => void
  setSelectedEvent: (eventId: string | null) => void
}

export const useUIStore = create<UIState>((set) => ({
  activeTab: 'overview',
  sidebarCollapsed: false,
  expandedDrawers: new Set<string>(),
  connected: false,
  selectedNodeId: null,
  selectedEventId: null,

  setActiveTab: (tab) => set({ activeTab: tab }),

  toggleSidebar: () =>
    set((state) => ({ sidebarCollapsed: !state.sidebarCollapsed })),

  toggleDrawer: (id) =>
    set((state) => {
      const next = new Set(state.expandedDrawers)
      if (next.has(id)) {
        next.delete(id)
      } else {
        next.add(id)
      }
      return { expandedDrawers: next }
    }),

  setConnected: (connected) => set({ connected }),

  setSelectedNode: (nodeId) => set({ selectedNodeId: nodeId }),

  setSelectedEvent: (eventId) =>
    set((state) => {
      // Auto-open event inspector drawer when selecting an event
      const next = new Set(state.expandedDrawers)
      if (eventId) {
        next.add('event-inspector')
      } else {
        next.delete('event-inspector')
      }
      return { selectedEventId: eventId, expandedDrawers: next }
    }),
}))

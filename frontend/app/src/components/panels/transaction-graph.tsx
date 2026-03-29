// ============================================================================
// Transaction Graph -- Cytoscape.js wrapper with SOC styling
// ============================================================================

import { useEffect, useRef, useCallback } from 'react'
import cytoscape, { type Core } from 'cytoscape'
import { useDashboardStore } from '@/stores/use-dashboard-store'
import { useTopology } from '@/hooks/use-api'

// eslint-disable-next-line @typescript-eslint/no-explicit-any
const CYTO_STYLE: any[] = [
  {
    selector: 'node',
    style: {
      'label': '',
      'width': 'mapData(txn_count, 0, 50, 12, 36)' as unknown as number,
      'height': 'mapData(txn_count, 0, 50, 12, 36)' as unknown as number,
      'background-color': '#3d5a80',
      'border-width': 1,
      'border-color': '#2a3f5f',
    },
  },
  {
    selector: 'node[status = "frozen"]',
    style: {
      'background-color': '#c1121f',
      'border-width': 2,
      'border-color': '#e63946',
    },
  },
  {
    selector: 'node[status = "paused"]',
    style: {
      'background-color': '#e76f51',
      'border-width': 2,
      'border-color': '#f4a261',
      'border-style': 'dashed',
    },
  },
  {
    selector: 'node[status = "suspicious"]',
    style: {
      'background-color': '#e9c46a',
      'border-width': 2,
      'border-color': '#f4a261',
    },
  },
  {
    selector: 'edge',
    style: {
      'width': 1,
      'line-color': '#2a3f5f',
      'curve-style': 'bezier',
      'target-arrow-shape': 'triangle',
      'target-arrow-color': '#2a3f5f',
      'arrow-scale': 0.6,
      'opacity': 0.5,
    },
  },
  {
    selector: 'edge[fraud_label > 0]',
    style: {
      'line-color': '#c1121f',
      'target-arrow-color': '#c1121f',
      'width': 2,
      'opacity': 0.8,
    },
  },
]

const RELAYOUT_THRESHOLD = 300

export function TransactionGraph() {
  const containerRef = useRef<HTMLDivElement>(null)
  const cyRef = useRef<Core | null>(null)

  const graphNodes = useDashboardStore((s) => s.graphNodes)
  const graphEdges = useDashboardStore((s) => s.graphEdges)
  const shouldRelayout = useDashboardStore((s) => s.shouldRelayout)
  const clearRelayout = useDashboardStore((s) => s.clearRelayout)
  const setInitialTopology = useDashboardStore((s) => s.setInitialTopology)

  const { data: topology } = useTopology()

  // Initialize Cytoscape once
  useEffect(() => {
    if (!containerRef.current) return

    const cy = cytoscape({
      container: containerRef.current,
      style: CYTO_STYLE,
      layout: { name: 'preset' },
      minZoom: 0.1,
      maxZoom: 5,
      wheelSensitivity: 0.3,
    })

    cyRef.current = cy
    return () => {
      cy.destroy()
      cyRef.current = null
    }
  }, [])

  // Hydrate from initial topology fetch
  useEffect(() => {
    if (topology) {
      setInitialTopology(topology.nodes, topology.edges)
    }
  }, [topology, setInitialTopology])

  // Sync Cytoscape with Zustand store
  const syncGraph = useCallback(() => {
    const cy = cyRef.current
    if (!cy) return

    const elements = [
      ...graphNodes.map((n) => ({ group: 'nodes' as const, data: n.data })),
      ...graphEdges.map((e) => ({ group: 'edges' as const, data: e.data })),
    ]

    cy.json({ elements })

    if (shouldRelayout && graphNodes.length <= RELAYOUT_THRESHOLD && graphNodes.length > 0) {
      cy.layout({
        name: 'cose',
        animate: false,
        nodeRepulsion: () => 8000,
        idealEdgeLength: () => 60,
        randomize: false,
      }).run()
      clearRelayout()
    }
  }, [graphNodes, graphEdges, shouldRelayout, clearRelayout])

  useEffect(() => {
    syncGraph()
  }, [syncGraph])

  return (
    <div className="w-full h-full relative">
      <div ref={containerRef} className="absolute inset-0" />
      {graphNodes.length === 0 && (
        <div className="absolute inset-0 flex items-center justify-center">
          <p className="text-text-muted text-xs uppercase tracking-wider">
            Awaiting transaction data...
          </p>
        </div>
      )}
    </div>
  )
}

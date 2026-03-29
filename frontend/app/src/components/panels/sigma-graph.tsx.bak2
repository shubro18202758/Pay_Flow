// ============================================================================
// Sigma Graph -- Advanced real-time fraud network visualization
// ForceAtlas2 continuous physics + particle flows + heat diffusion +
// threat propagation ripples + minimap + advanced analytics
// ============================================================================

import { useEffect, useRef, useState, useCallback } from 'react'
import Graph from 'graphology'
import forceAtlas2 from 'graphology-layout-forceatlas2'
import {
  SigmaContainer,
  useRegisterEvents,
  useSigma,
  useSetSettings,
} from '@react-sigma/core'
import '@react-sigma/core/lib/style.css'

import { useDashboardStore } from '@/stores/use-dashboard-store'
import { useUIStore } from '@/stores/use-ui-store'
import { useTopology } from '@/hooks/use-api'
import { FRAUD_PATTERN_LABELS } from '@/lib/types'
import type { CytoNode, CytoEdge } from '@/lib/types'
import type { NodeDisplayData, EdgeDisplayData } from 'sigma/types'
import {
  Network,
  AlertTriangle,
  ZoomIn,
  ZoomOut,
  Maximize2,
  Tag,
  RotateCcw,
  Filter,
  Search,
  Crosshair,
  Download,
  Repeat,
  Type,
  Radio,
  Layers,
  Zap,
  TrendingUp,
  Target,
} from 'lucide-react'

// --- Layout constants ---
const MAX_NODES = 80
const NODE_COLOR_SAFE = '#3d4f5f'

// Refined palette — desaturated, harmonious dark-theme colors
const FRAUD_TYPE_COLORS: Record<number, string> = {
  0: '#3d4f5f', 1: '#4e8cd9', 2: '#9066d4', 3: '#2ea89e',
  4: '#7070d9', 5: '#d45e98', 6: '#d98044', 7: '#c93955', 8: '#2ea8c9',
}
const STATUS_BORDER_COLORS: Record<string, string> = {
  frozen: '#e05656', suspicious: '#d4ad2a', paused: '#d98044', normal: '#384858',
}
const COMMUNITY_COLORS = [
  '#3a5068','#4a6070','#3d5a64','#4d5868','#3f5560','#425a64','#485e68','#445862',
]

function dominantFraudType(nodeId: string, edges: CytoEdge[]): number {
  const counts = new Map<number, number>()
  for (const e of edges) {
    const fl = e.data.fraud_label ?? 0
    if (fl === 0) continue
    if (e.data.source === nodeId || e.data.target === nodeId)
      counts.set(fl, (counts.get(fl) ?? 0) + 1)
  }
  if (counts.size === 0) return 0
  let best = 0, bestCount = 0
  for (const [fl, c] of counts) { if (c > bestCount) { best = fl; bestCount = c } }
  return best
}

// --- Node selection & graph algorithms ---
function selectTopNodes(nodes: CytoNode[], edges: CytoEdge[]): CytoNode[] {
  if (nodes.length <= MAX_NODES) return nodes
  const degreeMap = new Map<string, number>()
  for (const e of edges) {
    degreeMap.set(e.data.source, (degreeMap.get(e.data.source) ?? 0) + 1)
    degreeMap.set(e.data.target, (degreeMap.get(e.data.target) ?? 0) + 1)
  }
  const statusPriority: Record<string, number> = { frozen: 4, suspicious: 3, paused: 2, normal: 0 }
  const scored = nodes.map(n => {
    const fraudCount = edges.filter(e =>
      (e.data.source === n.data.id || e.data.target === n.data.id) && (e.data.fraud_label ?? 0) > 0,
    ).length
    return {
      node: n,
      score: (statusPriority[n.data.status] ?? 0) * 100 + fraudCount * 10 + (degreeMap.get(n.data.id) ?? 0),
    }
  })
  scored.sort((a, b) => b.score - a.score)
  return scored.slice(0, MAX_NODES).map(s => s.node)
}

function detectCycles(g: Graph): string[][] {
  const cycles: string[][] = []
  const visited = new Set<string>()
  const stack = new Set<string>()
  const path: string[] = []
  function dfs(node: string) {
    if (cycles.length >= 8) return
    visited.add(node); stack.add(node); path.push(node)
    g.forEachOutNeighbor(node, (nb: string) => {
      if (cycles.length >= 8) return
      if (stack.has(nb)) { const idx = path.indexOf(nb); if (idx >= 0) cycles.push(path.slice(idx)) }
      else if (!visited.has(nb)) dfs(nb)
    })
    stack.delete(node); path.pop()
  }
  g.forEachNode((n: string) => { if (!visited.has(n) && cycles.length < 8) dfs(n) })
  return cycles
}

function computePageRank(g: Graph, iterations = 20, damping = 0.85): Map<string, number> {
  const nodes = g.nodes()
  const N = nodes.length
  if (N === 0) return new Map()
  let ranks = new Map<string, number>()
  const init = 1 / N
  nodes.forEach(n => ranks.set(n, init))
  for (let i = 0; i < iterations; i++) {
    const next = new Map<string, number>()
    const base = (1 - damping) / N
    nodes.forEach(n => next.set(n, base))
    nodes.forEach(n => {
      const outDeg = g.outDegree(n)
      if (outDeg === 0) {
        const share = (ranks.get(n) ?? 0) * damping / N
        nodes.forEach(m => next.set(m, (next.get(m) ?? 0) + share))
      } else {
        const share = (ranks.get(n) ?? 0) * damping / outDeg
        g.forEachOutNeighbor(n, (nb: string) => next.set(nb, (next.get(nb) ?? 0) + share))
      }
    })
    ranks = next
  }
  return ranks
}

function computeBetweenness(g: Graph): Map<string, number> {
  const nodes = g.nodes()
  const bc = new Map<string, number>()
  nodes.forEach(n => bc.set(n, 0))
  const samples = nodes.length > 15 ? nodes.sort(() => Math.random() - 0.5).slice(0, 15) : nodes
  for (const s of samples) {
    const stack: string[] = []
    const pred = new Map<string, string[]>()
    const sigma = new Map<string, number>()
    const dist = new Map<string, number>()
    nodes.forEach(n => { pred.set(n, []); sigma.set(n, 0); dist.set(n, -1) })
    sigma.set(s, 1); dist.set(s, 0)
    const queue = [s]
    while (queue.length > 0) {
      const v = queue.shift()!
      stack.push(v)
      g.forEachOutNeighbor(v, (w: string) => {
        if (dist.get(w) === -1) { dist.set(w, (dist.get(v) ?? 0) + 1); queue.push(w) }
        if (dist.get(w) === (dist.get(v) ?? 0) + 1) {
          sigma.set(w, (sigma.get(w) ?? 0) + (sigma.get(v) ?? 0))
          pred.get(w)!.push(v)
        }
      })
    }
    const delta = new Map<string, number>()
    nodes.forEach(n => delta.set(n, 0))
    while (stack.length > 0) {
      const w = stack.pop()!
      for (const v of pred.get(w) ?? []) {
        const d = ((sigma.get(v) ?? 0) / (sigma.get(w) ?? 1)) * (1 + (delta.get(w) ?? 0))
        delta.set(v, (delta.get(v) ?? 0) + d)
      }
      if (w !== s) bc.set(w, (bc.get(w) ?? 0) + (delta.get(w) ?? 0))
    }
  }
  // normalize
  let maxBC = 0
  bc.forEach(v => { if (v > maxBC) maxBC = v })
  if (maxBC > 0) bc.forEach((v, k) => bc.set(k, v / maxBC))
  return bc
}

function computeClusteringCoeff(g: Graph): number {
  let totalCoeff = 0, count = 0
  g.forEachNode((n: string) => {
    const neighbors = g.neighbors(n)
    const k = neighbors.length
    if (k < 2) return
    let triangles = 0
    for (let i = 0; i < k; i++)
      for (let j = i + 1; j < k; j++)
        if (g.hasEdge(neighbors[i], neighbors[j]) || g.hasEdge(neighbors[j], neighbors[i])) triangles++
    totalCoeff += (2 * triangles) / (k * (k - 1))
    count++
  })
  return count > 0 ? totalCoeff / count : 0
}

function computeThreatPaths(g: Graph): Map<string, number> {
  const levels = new Map<string, number>()
  g.forEachNode((n: string) => levels.set(n, 0))
  const sources: string[] = []
  g.forEachNode((n: string) => {
    const attrs = g.getNodeAttributes(n)
    if (attrs.status === 'frozen' || (attrs.fraudRatio ?? 0) > 0.4) sources.push(n)
  })
  for (const src of sources) {
    const queue: [string, number][] = [[src, 1.0]]
    const visited = new Set<string>([src])
    levels.set(src, Math.max(levels.get(src) ?? 0, 1.0))
    let depth = 0
    while (queue.length > 0 && depth < 3) {
      const next: [string, number][] = []
      for (const [node, level] of queue) {
        g.forEachOutNeighbor(node, (nb: string) => {
          if (!visited.has(nb)) {
            visited.add(nb)
            const propagated = level * 0.4
            if (propagated > 0.05) {
              levels.set(nb, Math.max(levels.get(nb) ?? 0, propagated))
              next.push([nb, propagated])
            }
          }
        })
      }
      queue.length = 0
      queue.push(...next)
      depth++
    }
  }
  return levels
}

// --- Activity tracking ---
const _newFraudNodes = new Map<string, number>()
const _activityHistory: number[] = []
const _activityTimestamps: number[] = []
let _prevEdgeCount = 0

function recordActivity(edgeCount: number) {
  const now = Date.now()
  const delta = Math.abs(edgeCount - _prevEdgeCount)
  _prevEdgeCount = edgeCount
  _activityHistory.push(delta)
  _activityTimestamps.push(now)
  if (_activityHistory.length > 60) { _activityHistory.shift(); _activityTimestamps.shift() }
}

// --- Component-aware force-directed layout ---
// Detects connected components, lays out each internally, then packs them
function applyCleanLayout(g: Graph) {
  if (g.order === 0) return

  // 1. Find connected components via BFS
  const visited = new Set<string>()
  const components: string[][] = []
  g.forEachNode((node: string) => {
    if (visited.has(node)) return
    const comp: string[] = []
    const queue = [node]
    while (queue.length > 0) {
      const n = queue.shift()!
      if (visited.has(n)) continue
      visited.add(n)
      comp.push(n)
      g.forEachNeighbor(n, (nb: string) => {
        if (!visited.has(nb)) queue.push(nb)
      })
    }
    components.push(comp)
  })
  components.sort((a, b) => b.length - a.length)

  // 2. Layout each component internally
  for (const comp of components) {
    if (comp.length === 1) {
      g.setNodeAttribute(comp[0], 'x', 0)
      g.setNodeAttribute(comp[0], 'y', 0)
    } else if (comp.length <= 4) {
      const r = comp.length * 12
      for (let j = 0; j < comp.length; j++) {
        const angle = (2 * Math.PI * j) / comp.length - Math.PI / 2
        g.setNodeAttribute(comp[j], 'x', r * Math.cos(angle))
        g.setNodeAttribute(comp[j], 'y', r * Math.sin(angle))
      }
    } else {
      // Circle initial positions then local FA2
      const r = Math.sqrt(comp.length) * 18
      for (let j = 0; j < comp.length; j++) {
        const angle = (2 * Math.PI * j) / comp.length - Math.PI / 2
        g.setNodeAttribute(comp[j], 'x', r * Math.cos(angle))
        g.setNodeAttribute(comp[j], 'y', r * Math.sin(angle))
      }
    }
  }

  // 3. Run FA2 independently per component to avoid gravity merging them
  //    We offset each component first, run FA2, then re-offset
  const compBounds: { cx: number; cy: number; r: number }[] = []
  for (const comp of components) {
    if (comp.length < 5) {
      // Small components: skip FA2, just compute bounds
      let sx = 0, sy = 0
      for (const n of comp) {
        sx += g.getNodeAttribute(n, 'x') as number
        sy += g.getNodeAttribute(n, 'y') as number
      }
      compBounds.push({ cx: sx / comp.length, cy: sy / comp.length, r: comp.length * 15 })
    } else {
      // Create a temporary subgraph for per-component FA2
      const subG = new Graph()
      for (const n of comp) {
        subG.addNode(n, {
          x: g.getNodeAttribute(n, 'x'),
          y: g.getNodeAttribute(n, 'y'),
          size: g.getNodeAttribute(n, 'size'),
        })
      }
      const compSet = new Set(comp)
      g.forEachEdge((edge: string, _attrs: Record<string, unknown>, src: string, tgt: string) => {
        if (compSet.has(src) && compSet.has(tgt) && !subG.hasEdge(edge)) {
          subG.addEdgeWithKey(edge, src, tgt)
        }
      })

      forceAtlas2.assign(subG, {
        iterations: 300,
        settings: {
          gravity: 0.5,
          scalingRatio: 80,
          barnesHutOptimize: comp.length > 20,
          barnesHutTheta: 0.5,
          slowDown: 5,
          strongGravityMode: false,
          edgeWeightInfluence: 0.5,
        },
      })

      // Copy positions back
      let sx = 0, sy = 0, maxR = 0
      for (const n of comp) {
        const x = subG.getNodeAttribute(n, 'x') as number
        const y = subG.getNodeAttribute(n, 'y') as number
        g.setNodeAttribute(n, 'x', x)
        g.setNodeAttribute(n, 'y', y)
        sx += x; sy += y
      }
      const cx = sx / comp.length, cy = sy / comp.length
      for (const n of comp) {
        const dx = (g.getNodeAttribute(n, 'x') as number) - cx
        const dy = (g.getNodeAttribute(n, 'y') as number) - cy
        const d = Math.sqrt(dx * dx + dy * dy)
        if (d > maxR) maxR = d
      }
      compBounds.push({ cx, cy, r: maxR + 20 })
    }
  }

  // 4. Pack components in a spiral arrangement (no overlaps)
  if (components.length > 1) {
    const placed: { x: number; y: number; r: number }[] = []
    for (let ci = 0; ci < components.length; ci++) {
      const comp = components[ci]
      const cr = compBounds[ci].r
      let bestX = 0, bestY = 0

      if (placed.length === 0) {
        bestX = 0; bestY = 0
      } else {
        // Try spiral positions until no overlap
        let angle = 0, dist = 0
        const step = 15
        let found = false
        for (let attempt = 0; attempt < 500 && !found; attempt++) {
          const tx = dist * Math.cos(angle)
          const ty = dist * Math.sin(angle)
          let overlap = false
          for (const p of placed) {
            const dx = tx - p.x, dy = ty - p.y
            const d = Math.sqrt(dx * dx + dy * dy)
            if (d < cr + p.r + 25) { overlap = true; break }
          }
          if (!overlap) { bestX = tx; bestY = ty; found = true }
          angle += 0.5
          dist += step / (2 * Math.PI)
        }
      }

      // Offset all nodes in this component
      const oldCx = compBounds[ci].cx, oldCy = compBounds[ci].cy
      for (const n of comp) {
        const x = g.getNodeAttribute(n, 'x') as number
        const y = g.getNodeAttribute(n, 'y') as number
        g.setNodeAttribute(n, 'x', x - oldCx + bestX)
        g.setNodeAttribute(n, 'y', y - oldCy + bestY)
      }
      placed.push({ x: bestX, y: bestY, r: cr })
    }
  }

  // 5. Center and normalize to fill viewport
  let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity
  g.forEachNode((node: string) => {
    const x = g.getNodeAttribute(node, 'x') as number
    const y = g.getNodeAttribute(node, 'y') as number
    if (x < minX) minX = x; if (x > maxX) maxX = x
    if (y < minY) minY = y; if (y > maxY) maxY = y
  })
  const w = maxX - minX || 1, h = maxY - minY || 1
  const cx = (minX + maxX) / 2, cy = (minY + maxY) / 2
  const scale = 900 / Math.max(w, h)
  g.forEachNode((node: string) => {
    const x = g.getNodeAttribute(node, 'x') as number
    const y = g.getNodeAttribute(node, 'y') as number
    g.setNodeAttribute(node, 'x', (x - cx) * scale)
    g.setNodeAttribute(node, 'y', (y - cy) * scale)
  })
}

// --- Sync graphology from store ---
function syncGraphology(
  g: Graph,
  nodes: CytoNode[],
  edges: CytoEdge[],
) {
  const top = selectTopNodes(nodes, edges)
  const topIds = new Set(top.map(n => n.data.id))

  recordActivity(edges.length)

  // clear graph
  g.clear()

  // Compute degrees within visible subgraph
  const degMap = new Map<string, number>()
  for (const e of edges) {
    if (!topIds.has(e.data.source) || !topIds.has(e.data.target)) continue
    degMap.set(e.data.source, (degMap.get(e.data.source) ?? 0) + 1)
    degMap.set(e.data.target, (degMap.get(e.data.target) ?? 0) + 1)
  }

  // Find max degree for relative sizing
  let maxDeg = 1
  degMap.forEach(d => { if (d > maxDeg) maxDeg = d })

  // Build nodes with proper sizing and labels
  for (const n of top) {
    const fraudEdges = edges.filter(e =>
      (e.data.source === n.data.id || e.data.target === n.data.id) && (e.data.fraud_label ?? 0) > 0,
    )
    const totalEdges = edges.filter(e => e.data.source === n.data.id || e.data.target === n.data.id)
    const fraudRatio = totalEdges.length > 0 ? fraudEdges.length / totalEdges.length : 0
    const dom = dominantFraudType(n.data.id, edges)
    const deg = degMap.get(n.data.id) ?? 0

    // Sizing: every node large enough to be clearly visible
    let sz: number
    if (deg >= 6) sz = 14 + (deg / maxDeg) * 6          // major hub: 14-20px
    else if (deg >= 3) sz = 10 + (deg / maxDeg) * 4     // hub: 10-14px
    else if (deg >= 1) sz = 7 + (deg / maxDeg) * 3      // connected: 7-10px
    else sz = 5                                          // isolated: 5px

    // Color: fraud=accent from palette, clean=muted slate
    const color = dom > 0 ? FRAUD_TYPE_COLORS[dom] ?? '#c0392b' : NODE_COLOR_SAFE

    // Label: every node with connections gets a label
    let label = ''
    if (deg >= 1) {
      label = n.data.id.length > 10 ? '…' + n.data.id.slice(-6) : n.data.id
    }

    const zDepth = 0.3 + (deg / maxDeg) * 0.7
    g.addNode(n.data.id, {
      label,
      size: sz,
      color,
      x: 0,
      y: 0,
      status: n.data.status,
      fraudRatio,
      dominantFraud: dom,
      degree: deg,
      zDepth,
      borderColor: STATUS_BORDER_COLORS[n.data.status] ?? '#384858',
      type: 'circle',
    })
  }

  // Add edges — fraud edges prominent, clean edges visible but subtle
  for (const e of edges) {
    if (!topIds.has(e.data.source) || !topIds.has(e.data.target)) continue
    if (e.data.source === e.data.target) continue
    const key = `${e.data.source}->${e.data.target}`
    if (g.hasEdge(key)) continue
    const fl = e.data.fraud_label ?? 0
    g.addEdgeWithKey(key, e.data.source, e.data.target, {
      color: fl > 0 ? 'rgba(230,90,80,0.7)' : 'rgba(140,170,200,0.3)',
      size: fl > 0 ? 1.5 : 0.8,
      type: 'line',
      fraud_label: fl,
      amount: e.data.amount_paisa ?? 0,
      channel: String(e.data.channel ?? 'unknown'),
    })
  }

  // Apply clean force-directed layout
  if (g.order > 0) {
    applyCleanLayout(g)
  }

  // Mark new fraud nodes
  const now = Date.now()
  for (const n of top) {
    const fraudEdges = edges.filter(e =>
      (e.data.source === n.data.id || e.data.target === n.data.id) && (e.data.fraud_label ?? 0) > 0,
    )
    if (fraudEdges.length > 0 && !_newFraudNodes.has(n.data.id)) {
      _newFraudNodes.set(n.data.id, now)
    }
  }
  // expire old entries
  for (const [id, ts] of _newFraudNodes) {
    if (now - ts > 5000) _newFraudNodes.delete(id)
  }
}

// --- Types ---
interface TooltipData { id: string; x: number; y: number; status: string; fraudRatio: number; dominantFraud: number; degree: number }
interface EdgeTooltipData { key: string; x: number; y: number; fraud_label: number; amount: number; channel: string; source: string; target: string }
type GraphFilter = 'none' | 'fraud-only' | 'high-risk' | 'cycles' | 'community' | 'threat-path'

// ============================================================================
// Particle Flow System -- animated dots flowing along fraud edges
// ============================================================================
interface Particle { x: number; y: number; progress: number; speed: number; edgeKey: string; color: string; size: number }

function useParticleFlow(sigma: ReturnType<typeof useSigma> | null, enabled: boolean) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null)
  const particlesRef = useRef<Particle[]>([])
  const rafRef = useRef<number>(0)

  useEffect(() => {
    if (!sigma || !enabled) {
      if (canvasRef.current) { canvasRef.current.remove(); canvasRef.current = null }
      cancelAnimationFrame(rafRef.current)
      return
    }
    const container = sigma.getContainer()
    if (!container) return
    let canvas = canvasRef.current
    if (!canvas) {
      canvas = document.createElement('canvas')
      canvas.style.cssText = 'position:absolute;top:0;left:0;width:100%;height:100%;pointer-events:none;z-index:5'
      container.style.position = 'relative'
      container.appendChild(canvas)
      canvasRef.current = canvas
    }
    const graph = sigma.getGraph()
    let spawnTimer = 0

    function animate() {
      if (!canvas || !sigma) return
      const w = container.clientWidth
      const h = container.clientHeight
      canvas.width = w * window.devicePixelRatio
      canvas.height = h * window.devicePixelRatio
      canvas.style.width = w + 'px'
      canvas.style.height = h + 'px'
      const ctx = canvas.getContext('2d')
      if (!ctx) return
      ctx.scale(window.devicePixelRatio, window.devicePixelRatio)
      ctx.clearRect(0, 0, w, h)

      // Spawn new particles
      spawnTimer++
      if (spawnTimer % 24 === 0) { // every ~400ms at 60fps
        graph.forEachEdge((edgeKey: string, attrs: Record<string, unknown>) => {
          if ((attrs.fraud_label as number) === 0) return
          if (Math.random() > 0.15) return
          const fl = attrs.fraud_label as number
          particlesRef.current.push({
            x: 0, y: 0,
            progress: 0,
            speed: 0.003 + Math.random() * 0.005,
            edgeKey,
            color: FRAUD_TYPE_COLORS[fl] ?? '#3b82f6',
            size: 1.5 + Math.random() * 1.5,
          })
        })
        if (particlesRef.current.length > 300) particlesRef.current = particlesRef.current.slice(-200)
      }

      // Update & draw
      const alive: Particle[] = []
      for (const p of particlesRef.current) {
        p.progress += p.speed
        if (p.progress >= 1) continue
        try {
          const src = graph.source(p.edgeKey)
          const tgt = graph.target(p.edgeKey)
          const srcCoords = sigma.graphToViewport(graph.getNodeAttributes(src) as {x:number,y:number})
          const tgtCoords = sigma.graphToViewport(graph.getNodeAttributes(tgt) as {x:number,y:number})
          p.x = srcCoords.x + (tgtCoords.x - srcCoords.x) * p.progress
          p.y = srcCoords.y + (tgtCoords.y - srcCoords.y) * p.progress
          const alpha = 1 - p.progress * 0.6
          ctx.beginPath()
          ctx.arc(p.x, p.y, p.size, 0, Math.PI * 2)
          ctx.fillStyle = p.color
          ctx.globalAlpha = alpha
          ctx.fill()
          // glow
          ctx.shadowColor = p.color
          ctx.shadowBlur = 6
          ctx.fill()
          ctx.shadowBlur = 0
          ctx.globalAlpha = 1
          alive.push(p)
        } catch { /* edge removed */ }
      }
      particlesRef.current = alive
      rafRef.current = requestAnimationFrame(animate)
    }
    rafRef.current = requestAnimationFrame(animate)
    return () => {
      cancelAnimationFrame(rafRef.current)
      if (canvasRef.current) { canvasRef.current.remove(); canvasRef.current = null }
    }
  }, [sigma, enabled])
}

// ============================================================================
// Threat Ripple System -- pulsing rings from high-risk nodes
// ============================================================================
interface Ripple { x: number; y: number; radius: number; maxRadius: number; color: string; born: number; lifetime: number }

function useThreatRipples(sigma: ReturnType<typeof useSigma> | null, enabled: boolean) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null)
  const ripplesRef = useRef<Ripple[]>([])
  const rafRef = useRef<number>(0)

  useEffect(() => {
    if (!sigma || !enabled) {
      if (canvasRef.current) { canvasRef.current.remove(); canvasRef.current = null }
      cancelAnimationFrame(rafRef.current)
      return
    }
    const container = sigma.getContainer()
    if (!container) return
    let canvas = canvasRef.current
    if (!canvas) {
      canvas = document.createElement('canvas')
      canvas.style.cssText = 'position:absolute;top:0;left:0;width:100%;height:100%;pointer-events:none;z-index:4'
      container.appendChild(canvas)
      canvasRef.current = canvas
    }
    const graph = sigma.getGraph()
    let spawnTimer = 0

    function animate() {
      if (!canvas || !sigma) return
      const w = container.clientWidth, h = container.clientHeight
      canvas.width = w * window.devicePixelRatio
      canvas.height = h * window.devicePixelRatio
      canvas.style.width = w + 'px'; canvas.style.height = h + 'px'
      const ctx = canvas.getContext('2d')
      if (!ctx) return
      ctx.scale(window.devicePixelRatio, window.devicePixelRatio)
      ctx.clearRect(0, 0, w, h)
      const now = Date.now()

      spawnTimer++
      if (spawnTimer % 90 === 0) { // every ~1.5s
        graph.forEachNode((_n: string, attrs: Record<string, unknown>) => {
          const fr = (attrs.fraudRatio as number) ?? 0
          if (fr < 0.4 && attrs.status !== 'frozen') return
          if (Math.random() > 0.1) return
          const coords = sigma.graphToViewport(attrs as {x:number,y:number})
          const color = FRAUD_TYPE_COLORS[(attrs.dominantFraud as number) ?? 0] ?? '#ef4444'
          ripplesRef.current.push({ x: coords.x, y: coords.y, radius: 0, maxRadius: 30 + fr * 40, color, born: now, lifetime: 2000 })
        })
        if (ripplesRef.current.length > 30) ripplesRef.current = ripplesRef.current.slice(-20)
      }

      const alive: Ripple[] = []
      for (const r of ripplesRef.current) {
        const age = now - r.born
        if (age > r.lifetime) continue
        const progress = age / r.lifetime
        r.radius = r.maxRadius * progress
        const alpha = 0.5 * (1 - progress)
        ctx.beginPath()
        ctx.arc(r.x, r.y, r.radius, 0, Math.PI * 2)
        ctx.strokeStyle = r.color
        ctx.globalAlpha = alpha
        ctx.lineWidth = 2
        ctx.stroke()
        ctx.globalAlpha = 1
        alive.push(r)
      }
      ripplesRef.current = alive
      rafRef.current = requestAnimationFrame(animate)
    }
    rafRef.current = requestAnimationFrame(animate)
    return () => {
      cancelAnimationFrame(rafRef.current)
      if (canvasRef.current) { canvasRef.current.remove(); canvasRef.current = null }
    }
  }, [sigma, enabled])
}

// ============================================================================
// Heat Diffusion Overlay -- radial gradient zones around high-risk nodes
// ============================================================================
function useHeatOverlay(sigma: ReturnType<typeof useSigma> | null, enabled: boolean) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null)
  const rafRef = useRef<number>(0)

  useEffect(() => {
    if (!sigma || !enabled) {
      if (canvasRef.current) { canvasRef.current.remove(); canvasRef.current = null }
      cancelAnimationFrame(rafRef.current)
      return
    }
    const container = sigma.getContainer()
    if (!container) return
    let canvas = canvasRef.current
    if (!canvas) {
      canvas = document.createElement('canvas')
      canvas.style.cssText = 'position:absolute;top:0;left:0;width:100%;height:100%;pointer-events:none;z-index:3'
      container.appendChild(canvas)
      canvasRef.current = canvas
    }
    const graph = sigma.getGraph()

    function draw() {
      if (!canvas || !sigma) return
      const w = container.clientWidth, h = container.clientHeight
      canvas.width = w * window.devicePixelRatio
      canvas.height = h * window.devicePixelRatio
      canvas.style.width = w + 'px'; canvas.style.height = h + 'px'
      const ctx = canvas.getContext('2d')
      if (!ctx) return
      ctx.scale(window.devicePixelRatio, window.devicePixelRatio)
      ctx.clearRect(0, 0, w, h)

      graph.forEachNode((_n: string, attrs: Record<string, unknown>) => {
        const fr = (attrs.fraudRatio as number) ?? 0
        if (fr < 0.2) return
        const coords = sigma.graphToViewport(attrs as {x:number,y:number})
        const radius = 30 + fr * 60
        const color = FRAUD_TYPE_COLORS[(attrs.dominantFraud as number) ?? 7] ?? '#e11d48'
        const grad = ctx.createRadialGradient(coords.x, coords.y, 0, coords.x, coords.y, radius)
        // Extract RGB from hex
        const r = parseInt(color.slice(1,3), 16), g2 = parseInt(color.slice(3,5), 16), b = parseInt(color.slice(5,7), 16)
        grad.addColorStop(0, `rgba(${r},${g2},${b},0.15)`)
        grad.addColorStop(1, `rgba(${r},${g2},${b},0)`)
        ctx.fillStyle = grad
        ctx.fillRect(coords.x - radius, coords.y - radius, radius * 2, radius * 2)
      })
      rafRef.current = requestAnimationFrame(draw)
    }
    rafRef.current = requestAnimationFrame(draw)
    return () => {
      cancelAnimationFrame(rafRef.current)
      if (canvasRef.current) { canvasRef.current.remove(); canvasRef.current = null }
    }
  }, [sigma, enabled])
}

// ============================================================================
// 3D Depth Overlay -- perspective grid, node glow, drop shadows, sphere highlights
// ============================================================================
function use3DDepthOverlay(sigma: ReturnType<typeof useSigma> | null, enabled: boolean) {
  const bgRef = useRef<HTMLCanvasElement | null>(null)
  const fgRef = useRef<HTMLCanvasElement | null>(null)
  const rafRef = useRef<number>(0)
  const frameRef = useRef(0)

  useEffect(() => {
    if (!sigma || !enabled) {
      if (bgRef.current) { bgRef.current.remove(); bgRef.current = null }
      if (fgRef.current) { fgRef.current.remove(); fgRef.current = null }
      cancelAnimationFrame(rafRef.current)
      return
    }
    const container = sigma.getContainer()
    if (!container) return
    container.style.position = 'relative'

    if (!bgRef.current) {
      const bg = document.createElement('canvas')
      bg.style.cssText = 'position:absolute;top:0;left:0;width:100%;height:100%;pointer-events:none'
      container.insertBefore(bg, container.firstChild)
      bgRef.current = bg
    }
    if (!fgRef.current) {
      const fg = document.createElement('canvas')
      fg.style.cssText = 'position:absolute;top:0;left:0;width:100%;height:100%;pointer-events:none;z-index:6'
      container.appendChild(fg)
      fgRef.current = fg
    }

    const graph = sigma.getGraph()

    function animate() {
      const bg = bgRef.current
      const fg = fgRef.current
      if (!bg || !fg || !sigma) return
      frameRef.current++
      const frame = frameRef.current
      const w = container.clientWidth
      const h = container.clientHeight
      const dpr = window.devicePixelRatio
      for (const c of [bg, fg]) {
        c.width = w * dpr; c.height = h * dpr
        c.style.width = w + 'px'; c.style.height = h + 'px'
      }

      // ---- Background: animated perspective grid ----
      const bgCtx = bg.getContext('2d')!
      bgCtx.scale(dpr, dpr)
      const t = frame * 0.008
      const sp = 50
      bgCtx.lineWidth = 0.5
      for (let gy = 0; gy <= h; gy += sp) {
        bgCtx.beginPath()
        bgCtx.strokeStyle = `rgba(80,160,240,${0.03 + Math.sin(gy * 0.02 + t) * 0.01})`
        for (let gx = 0; gx <= w; gx += 4) {
          const wy = gy + Math.sin(gx * 0.008 + t) * 3
          gx === 0 ? bgCtx.moveTo(gx, wy) : bgCtx.lineTo(gx, wy)
        }
        bgCtx.stroke()
      }
      for (let gx = 0; gx <= w; gx += sp) {
        bgCtx.beginPath()
        bgCtx.strokeStyle = `rgba(80,160,240,${0.03 + Math.sin(gx * 0.02 + t) * 0.01})`
        for (let gy2 = 0; gy2 <= h; gy2 += 4) {
          const wx = gx + Math.sin(gy2 * 0.008 + t * 0.7) * 3
          gy2 === 0 ? bgCtx.moveTo(wx, gy2) : bgCtx.lineTo(wx, gy2)
        }
        bgCtx.stroke()
      }
      const nebula = bgCtx.createRadialGradient(w / 2, h / 2, 0, w / 2, h / 2, Math.max(w, h) * 0.5)
      nebula.addColorStop(0, `rgba(30,70,140,${0.06 + Math.sin(t * 0.5) * 0.02})`)
      nebula.addColorStop(0.5, 'rgba(20,40,80,0.03)')
      nebula.addColorStop(1, 'rgba(15,23,42,0)')
      bgCtx.fillStyle = nebula
      bgCtx.fillRect(0, 0, w, h)

      // ---- Foreground: edge glow + node effects ----
      const fgCtx = fg.getContext('2d')!
      fgCtx.scale(dpr, dpr)
      const camRatio = sigma.getCamera().getState().ratio

      // Edge glow for fraud connections
      graph.forEachEdge((edgeKey: string, attrs: Record<string, unknown>) => {
        const fl = (attrs.fraud_label as number) ?? 0
        if (fl === 0) return
        try {
          const src = graph.source(edgeKey)
          const tgt = graph.target(edgeKey)
          const sv = sigma.graphToViewport(graph.getNodeAttributes(src) as { x: number; y: number })
          const tv = sigma.graphToViewport(graph.getNodeAttributes(tgt) as { x: number; y: number })
          const ec = FRAUD_TYPE_COLORS[fl] ?? '#e11d48'
          const er = parseInt(ec.slice(1, 3), 16) || 200
          const eg2 = parseInt(ec.slice(3, 5), 16) || 50
          const eb = parseInt(ec.slice(5, 7), 16) || 50
          fgCtx.beginPath()
          fgCtx.moveTo(sv.x, sv.y)
          fgCtx.lineTo(tv.x, tv.y)
          fgCtx.strokeStyle = `rgba(${er},${eg2},${eb},0.06)`
          fgCtx.lineWidth = 4
          fgCtx.stroke()
        } catch (_) { /* edge may reference removed node */ }
      })

      // Node glow + shadow + sphere
      graph.forEachNode((node: string, attrs: Record<string, unknown>) => {
        const vp = sigma.graphToViewport(attrs as { x: number; y: number })
        const baseSize = (attrs.size as number) ?? 5
        const ds = baseSize / camRatio
        if (ds < 1.5) return
        const color = (attrs.color as string) ?? '#3d4f5f'
        const zd = (attrs.zDepth as number) ?? 0.5
        const fr = (attrs.fraudRatio as number) ?? 0
        const cr = parseInt(color.slice(1, 3), 16) || 60
        const cg2 = parseInt(color.slice(3, 5), 16) || 80
        const cb = parseInt(color.slice(5, 7), 16) || 95

        // Drop shadow
        const sOff = 1.5 + ds * 0.12
        fgCtx.beginPath()
        fgCtx.arc(vp.x + sOff, vp.y + sOff, ds * 0.85, 0, Math.PI * 2)
        fgCtx.fillStyle = `rgba(0,0,0,${0.12 + zd * 0.12})`
        fgCtx.fill()

        // Outer glow halo
        const glowR = ds * (1.6 + fr * 1.4)
        const halo = fgCtx.createRadialGradient(vp.x, vp.y, ds * 0.4, vp.x, vp.y, glowR)
        halo.addColorStop(0, `rgba(${cr},${cg2},${cb},${0.12 + fr * 0.18})`)
        halo.addColorStop(1, `rgba(${cr},${cg2},${cb},0)`)
        fgCtx.beginPath()
        fgCtx.arc(vp.x, vp.y, glowR, 0, Math.PI * 2)
        fgCtx.fillStyle = halo
        fgCtx.fill()

        // Sphere specular highlight
        if (ds > 3) {
          const hx = vp.x - ds * 0.22
          const hy = vp.y - ds * 0.22
          const specular = fgCtx.createRadialGradient(hx, hy, 0, hx, hy, ds * 0.65)
          specular.addColorStop(0, `rgba(255,255,255,${0.10 + zd * 0.06})`)
          specular.addColorStop(1, 'rgba(255,255,255,0)')
          fgCtx.beginPath()
          fgCtx.arc(vp.x, vp.y, ds, 0, Math.PI * 2)
          fgCtx.fillStyle = specular
          fgCtx.fill()
        }

        // Pulsing ring for high-fraud nodes
        if (fr > 0.5 && ds > 4) {
          const seed = node.split('').reduce((a, c) => a + c.charCodeAt(0), 0)
          const pulse = 0.5 + Math.sin(frame * 0.05 + seed * 0.1) * 0.5
          fgCtx.beginPath()
          fgCtx.arc(vp.x, vp.y, ds * (1.3 + pulse * 0.6), 0, Math.PI * 2)
          fgCtx.strokeStyle = `rgba(${cr},${cg2},${cb},${0.1 + pulse * 0.12})`
          fgCtx.lineWidth = 1
          fgCtx.stroke()
        }
      })

      rafRef.current = requestAnimationFrame(animate)
    }

    rafRef.current = requestAnimationFrame(animate)
    return () => {
      cancelAnimationFrame(rafRef.current)
      if (bgRef.current) { bgRef.current.remove(); bgRef.current = null }
      if (fgRef.current) { fgRef.current.remove(); fgRef.current = null }
    }
  }, [sigma, enabled])
}

// ============================================================================
// Minimap Navigator
// ============================================================================
function Minimap() {
  const sigma = useSigma()
  const canvasRef = useRef<HTMLCanvasElement>(null)

  useEffect(() => {
    const iv = setInterval(() => {
      const canvas = canvasRef.current
      if (!canvas || !sigma) return
      const graph = sigma.getGraph()
      if (graph.order === 0) return
      const ctx = canvas.getContext('2d')
      if (!ctx) return
      const W = 180, H = 120
      canvas.width = W * window.devicePixelRatio
      canvas.height = H * window.devicePixelRatio
      ctx.scale(window.devicePixelRatio, window.devicePixelRatio)
      ctx.clearRect(0, 0, W, H)
      ctx.fillStyle = 'rgba(15,23,42,0.85)'
      ctx.fillRect(0, 0, W, H)

      // bounding box
      let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity
      graph.forEachNode((_: string, a: Record<string, unknown>) => {
        const x = a.x as number, y = a.y as number
        if (x < minX) minX = x; if (x > maxX) maxX = x
        if (y < minY) minY = y; if (y > maxY) maxY = y
      })
      const rangeX = maxX - minX || 1, rangeY = maxY - minY || 1
      const pad = 10
      const scaleX = (W - pad * 2) / rangeX, scaleY = (H - pad * 2) / rangeY

      // Draw edges
      graph.forEachEdge((_: string, attrs: Record<string, unknown>, src: string, tgt: string) => {
        const sa = graph.getNodeAttributes(src), ta = graph.getNodeAttributes(tgt)
        const x1 = pad + ((sa.x as number) - minX) * scaleX
        const y1 = pad + ((sa.y as number) - minY) * scaleY
        const x2 = pad + ((ta.x as number) - minX) * scaleX
        const y2 = pad + ((ta.y as number) - minY) * scaleY
        ctx.beginPath(); ctx.moveTo(x1, y1); ctx.lineTo(x2, y2)
        ctx.strokeStyle = (attrs.fraud_label as number) > 0 ? 'rgba(239,68,68,0.3)' : 'rgba(100,116,139,0.15)'
        ctx.lineWidth = 0.5; ctx.stroke()
      })

      // Draw nodes
      graph.forEachNode((_: string, attrs: Record<string, unknown>) => {
        const x = pad + ((attrs.x as number) - minX) * scaleX
        const y = pad + ((attrs.y as number) - minY) * scaleY
        ctx.beginPath(); ctx.arc(x, y, 2, 0, Math.PI * 2)
        ctx.fillStyle = attrs.color as string ?? '#64748b'; ctx.fill()
      })

      // camera viewport
      const cam = sigma.getCamera()
      const state = cam.getState()
      const vw = W / state.ratio, vh = H / state.ratio
      const cx = W / 2 - (state.x - 0.5) * W / state.ratio
      const cy = H / 2 - (state.y - 0.5) * H / state.ratio
      ctx.strokeStyle = 'rgba(59,130,246,0.6)'; ctx.lineWidth = 1.5
      ctx.strokeRect(cx - vw / 2, cy - vh / 2, vw, vh)
    }, 500)
    return () => clearInterval(iv)
  }, [sigma])

  return (
    <div className="absolute bottom-3 left-3 z-20 rounded-lg border border-slate-700/50 overflow-hidden shadow-lg">
      <canvas ref={canvasRef} style={{ width: 180, height: 120 }} />
    </div>
  )
}

// ============================================================================
// Activity Sparkline
// ============================================================================
function ActivitySparkline() {
  const canvasRef = useRef<HTMLCanvasElement>(null)

  useEffect(() => {
    const iv = setInterval(() => {
      const canvas = canvasRef.current
      if (!canvas) return
      const ctx = canvas.getContext('2d')
      if (!ctx) return
      const W = 80, H = 20
      canvas.width = W * window.devicePixelRatio
      canvas.height = H * window.devicePixelRatio
      ctx.scale(window.devicePixelRatio, window.devicePixelRatio)
      ctx.clearRect(0, 0, W, H)
      if (_activityHistory.length < 2) return
      const max = Math.max(..._activityHistory, 1)
      const step = W / (_activityHistory.length - 1)
      ctx.beginPath()
      _activityHistory.forEach((v, i) => {
        const x = i * step, y = H - (v / max) * (H - 2)
        i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y)
      })
      ctx.strokeStyle = '#3b82f6'; ctx.lineWidth = 1.5; ctx.stroke()
      // fill gradient
      const last = _activityHistory.length - 1
      ctx.lineTo(last * step, H); ctx.lineTo(0, H); ctx.closePath()
      const grad = ctx.createLinearGradient(0, 0, 0, H)
      grad.addColorStop(0, 'rgba(59,130,246,0.3)'); grad.addColorStop(1, 'rgba(59,130,246,0)')
      ctx.fillStyle = grad; ctx.fill()
      // dot on last point
      const lx = last * step, ly = H - (_activityHistory[last] / max) * (H - 2)
      ctx.beginPath(); ctx.arc(lx, ly, 2, 0, Math.PI * 2); ctx.fillStyle = '#60a5fa'; ctx.fill()
    }, 500)
    return () => clearInterval(iv)
  }, [])

  return <canvas ref={canvasRef} style={{ width: 80, height: 20 }} className="inline-block ml-2 align-middle" />
}

// ============================================================================
// Continuous FA2 Layout
// ============================================================================
function ContinuousLayout({ enabled: _enabled }: { enabled: boolean }) {
  // Disabled — one-shot FA2 in applyCleanLayout() produces final positions.
  // Continuous physics destroys the converged layout and causes drift.
  return null
}

// ============================================================================
// Graph Events -- click, hover, filter handlers
// ============================================================================
function GraphEvents({
  onNodeClick, onNodeHover, onNodeOut, onEdgeHover, onEdgeOut,
  filter, cycleNodes, searchTerm, hoveredNode, threatLevels,
}: {
  onNodeClick: (id: string) => void
  onNodeHover: (data: TooltipData) => void
  onNodeOut: () => void
  onEdgeHover: (data: EdgeTooltipData) => void
  onEdgeOut: () => void
  filter: GraphFilter
  cycleNodes: Set<string>
  searchTerm: string
  hoveredNode: string | null
  threatLevels: Map<string, number>
}) {
  const sigma = useSigma()
  const registerEvents = useRegisterEvents()
  const setSettings = useSetSettings()

  useEffect(() => {
    registerEvents({
      clickNode: (e: {node: string}) => onNodeClick(e.node),
      enterNode: (e: {node: string}) => {
        const graph = sigma.getGraph()
        const attrs = graph.getNodeAttributes(e.node)
        const coords = sigma.graphToViewport(attrs as {x:number,y:number})
        onNodeHover({
          id: e.node, x: coords.x, y: coords.y,
          status: attrs.status as string,
          fraudRatio: attrs.fraudRatio as number,
          dominantFraud: attrs.dominantFraud as number,
          degree: attrs.degree as number,
        })
      },
      leaveNode: () => onNodeOut(),
      enterEdge: (e: {edge: string}) => {
        const graph = sigma.getGraph()
        const attrs = graph.getEdgeAttributes(e.edge)
        const src = graph.source(e.edge), tgt = graph.target(e.edge)
        const srcA = graph.getNodeAttributes(src), tgtA = graph.getNodeAttributes(tgt)
        const srcC = sigma.graphToViewport(srcA as {x:number,y:number})
        const tgtC = sigma.graphToViewport(tgtA as {x:number,y:number})
        onEdgeHover({
          key: e.edge, x: (srcC.x + tgtC.x) / 2, y: (srcC.y + tgtC.y) / 2,
          fraud_label: attrs.fraud_label as number, amount: attrs.amount as number,
          channel: attrs.channel as string, source: src, target: tgt,
        })
      },
      leaveEdge: () => onEdgeOut(),
    })
  }, [registerEvents, sigma, onNodeClick, onNodeHover, onNodeOut, onEdgeHover, onEdgeOut])

  useEffect(() => {
    const graph = sigma.getGraph()
    const searchLower = searchTerm.toLowerCase()
    setSettings({
      nodeReducer: (node: string, data: Partial<NodeDisplayData>) => {
        const res = { ...data }
        const attrs = graph.getNodeAttributes(node)
        const sz = res.size ?? 5
        // search filter
        if (searchTerm && !node.toLowerCase().includes(searchLower)) {
          res.color = '#151d2a'; res.label = ''; res.size = sz * 0.3
          return res
        }
        // hover: highlight hovered + neighbors, dim rest
        if (hoveredNode) {
          if (node === hoveredNode) {
            res.size = sz * 1.4
            res.zIndex = 10
            res.label = node.length > 10 ? '…' + node.slice(-8) : node
          } else if (graph.hasEdge(hoveredNode, node) || graph.hasEdge(node, hoveredNode)) {
            res.size = sz * 1.1
            res.label = node.length > 10 ? '…' + node.slice(-6) : node
          } else {
            res.color = '#1a2436'; res.label = ''; res.size = sz * 0.4
          }
          return res
        }
        // filter modes
        if (filter === 'fraud-only') {
          if ((attrs.fraudRatio as number) === 0) { res.color = '#151d2a'; res.label = ''; res.size = sz * 0.2 }
        } else if (filter === 'high-risk') {
          if (attrs.status !== 'frozen' && attrs.status !== 'suspicious') { res.color = '#151d2a'; res.label = ''; res.size = sz * 0.2 }
        } else if (filter === 'cycles') {
          if (!cycleNodes.has(node)) { res.color = '#151d2a'; res.label = ''; res.size = sz * 0.2 }
          else { res.color = '#f59e0b'; res.size = sz * 1.3 }
        } else if (filter === 'community') {
          const neighbors = graph.neighbors(node)
          const communityId = neighbors.length % COMMUNITY_COLORS.length
          res.color = COMMUNITY_COLORS[communityId]
        } else if (filter === 'threat-path') {
          const level = threatLevels.get(node) ?? 0
          if (level === 0) { res.color = '#151d2a'; res.label = ''; res.size = sz * 0.2 }
          else { res.size = sz * (0.7 + level * 0.8) }
        }
        // new fraud pulse
        if (_newFraudNodes.has(node)) res.size = (res.size ?? sz) * 1.25
        return res
      },
      edgeReducer: (edge: string, data: Partial<EdgeDisplayData>) => {
        const res = { ...data }
        if (hoveredNode) {
          const src = graph.source(edge), tgt = graph.target(edge)
          if (src !== hoveredNode && tgt !== hoveredNode) res.hidden = true
        }
        if (filter === 'fraud-only') {
          const attrs = graph.getEdgeAttributes(edge)
          if ((attrs.fraud_label as number) === 0) res.hidden = true
        }
        if (filter === 'high-risk') {
          const src = graph.source(edge), tgt = graph.target(edge)
          const sa = graph.getNodeAttributes(src), ta = graph.getNodeAttributes(tgt)
          if (sa.status !== 'frozen' && sa.status !== 'suspicious' && ta.status !== 'frozen' && ta.status !== 'suspicious') res.hidden = true
        }
        if (filter === 'cycles') {
          const src = graph.source(edge), tgt = graph.target(edge)
          if (!cycleNodes.has(src) || !cycleNodes.has(tgt)) res.hidden = true
        }
        if (filter === 'threat-path') {
          const src = graph.source(edge), tgt = graph.target(edge)
          const sl = threatLevels.get(src) ?? 0, tl = threatLevels.get(tgt) ?? 0
          if (sl === 0 && tl === 0) res.hidden = true
        }
        return res
      },
    })
  }, [sigma, setSettings, filter, cycleNodes, searchTerm, hoveredNode, threatLevels])

  return null
}

// ============================================================================
// Tooltips
// ============================================================================
function NodeTooltip({ data }: { data: TooltipData }) {
  const riskLevel = data.fraudRatio > 0.6 ? 'CRITICAL' : data.fraudRatio > 0.3 ? 'HIGH' : data.fraudRatio > 0.1 ? 'MEDIUM' : 'LOW'
  const riskColor = { CRITICAL: '#ef4444', HIGH: '#f97316', MEDIUM: '#eab308', LOW: '#22c55e' }[riskLevel]
  return (
    <div className="fixed z-50 pointer-events-none animate-[fade-in_0.15s_ease-out]" style={{ left: data.x + 15, top: data.y - 10 }}>
      <div className="bg-slate-900/95 border border-slate-600/50 rounded-lg px-3 py-2 shadow-xl backdrop-blur-sm min-w-[180px]">
        <div className="flex items-center justify-between mb-1">
          <span className="text-xs font-mono text-slate-200 font-semibold">{data.id}</span>
          <span className="text-[9px] font-bold px-1.5 py-0.5 rounded-full" style={{ background: riskColor + '22', color: riskColor, border: `1px solid ${riskColor}44` }}>{riskLevel}</span>
        </div>
        <div className="flex items-center gap-2 text-[10px] text-slate-400 mb-1.5">
          <span className="px-1.5 py-0.5 rounded" style={{ background: STATUS_BORDER_COLORS[data.status] + '22', color: STATUS_BORDER_COLORS[data.status] }}>{data.status}</span>
          <span>{data.degree} connections</span>
        </div>
        <div className="text-[10px] text-slate-400 mb-1">Fraud: {(data.fraudRatio * 100).toFixed(0)}%</div>
        <div className="w-full h-1.5 bg-slate-700 rounded-full overflow-hidden">
          <div className="h-full rounded-full transition-all duration-500" style={{ width: `${data.fraudRatio * 100}%`, background: `linear-gradient(90deg, ${riskColor}88, ${riskColor})` }} />
        </div>
        {data.dominantFraud > 0 && (
          <div className="text-[10px] mt-1.5" style={{ color: FRAUD_TYPE_COLORS[data.dominantFraud] }}>
            {FRAUD_PATTERN_LABELS[data.dominantFraud] ?? `Type ${data.dominantFraud}`}
          </div>
        )}
      </div>
    </div>
  )
}

function EdgeTooltip({ data }: { data: EdgeTooltipData }) {
  return (
    <div className="fixed z-50 pointer-events-none animate-[fade-in_0.15s_ease-out]" style={{ left: data.x + 15, top: data.y - 10 }}>
      <div className="bg-slate-900/95 border border-slate-600/50 rounded-lg px-3 py-2 shadow-xl backdrop-blur-sm">
        <div className="text-[10px] text-slate-300 font-mono mb-1">{data.source} &rarr; {data.target}</div>
        <div className="flex items-center gap-2 text-[10px] text-slate-400">
          {data.fraud_label > 0 && (
            <span className="px-1.5 py-0.5 rounded" style={{ background: FRAUD_TYPE_COLORS[data.fraud_label] + '22', color: FRAUD_TYPE_COLORS[data.fraud_label] }}>
              {FRAUD_PATTERN_LABELS[data.fraud_label] ?? `Type ${data.fraud_label}`}
            </span>
          )}
          <span>${data.amount.toLocaleString()}</span>
          <span className="text-slate-500">{data.channel}</span>
        </div>
      </div>
    </div>
  )
}

// ============================================================================
// In-Graph Node Detail Panel -- enhanced with centrality metrics
// ============================================================================
function InGraphNodeDetailPanel({
  nodeId, graph, edges, onClose, pageRanks, betweenness,
}: {
  nodeId: string; graph: Graph; edges: CytoEdge[]; onClose: () => void
  pageRanks: Map<string, number>; betweenness: Map<string, number>
}) {
  if (!graph.hasNode(nodeId)) return null
  const attrs = graph.getNodeAttributes(nodeId)
  const nodeEdges = edges.filter(e => e.data.source === nodeId || e.data.target === nodeId)
  const fraudEdges = nodeEdges.filter(e => (e.data.fraud_label ?? 0) > 0)
  const totalAmount = nodeEdges.reduce((s, e) => s + (e.data.amount_paisa ?? 0), 0)
  const fraudRatio = nodeEdges.length > 0 ? fraudEdges.length / nodeEdges.length : 0
  const riskLevel = fraudRatio > 0.6 ? 'CRITICAL' : fraudRatio > 0.3 ? 'HIGH' : fraudRatio > 0.1 ? 'MEDIUM' : 'LOW'
  const riskColor = { CRITICAL: '#ef4444', HIGH: '#f97316', MEDIUM: '#eab308', LOW: '#22c55e' }[riskLevel]

  // Fraud breakdown by type
  const fraudBreakdown = new Map<number, number>()
  for (const e of fraudEdges) {
    const fl = e.data.fraud_label ?? 0
    fraudBreakdown.set(fl, (fraudBreakdown.get(fl) ?? 0) + 1)
  }

  // Flow direction
  const inflow = nodeEdges.filter(e => e.data.target === nodeId)
  const outflow = nodeEdges.filter(e => e.data.source === nodeId)
  const inflowAmt = inflow.reduce((s, e) => s + (e.data.amount_paisa ?? 0), 0)
  const outflowAmt = outflow.reduce((s, e) => s + (e.data.amount_paisa ?? 0), 0)
  const maxFlow = Math.max(inflowAmt, outflowAmt, 1)

  // PageRank & Betweenness
  const pr = pageRanks.get(nodeId) ?? 0
  const bc = betweenness.get(nodeId) ?? 0

  // Channel distribution
  const channels = new Map<string, number>()
  for (const e of nodeEdges) {
    const ch = String(e.data.channel ?? 'unknown')
    channels.set(ch, (channels.get(ch) ?? 0) + 1)
  }

  return (
    <div className="absolute top-3 right-3 z-30 w-72 bg-slate-900/95 border border-slate-700/50 rounded-xl shadow-2xl backdrop-blur-sm animate-[slide-in-right_0.3s_ease-out] overflow-hidden">
      <div className="px-4 py-3 border-b border-slate-700/50 flex items-center justify-between">
        <div className="flex items-center gap-2">
          <div className="w-3 h-3 rounded-full" style={{ background: attrs.color as string }} />
          <span className="text-sm font-mono text-slate-200 font-semibold truncate max-w-[160px]">{nodeId}</span>
        </div>
        <button onClick={onClose} className="text-slate-500 hover:text-slate-300 text-lg leading-none">&times;</button>
      </div>
      <div className="px-4 py-3 space-y-3 max-h-[400px] overflow-y-auto text-[11px]">
        {/* Risk Level */}
        <div>
          <div className="flex items-center justify-between mb-1">
            <span className="text-slate-400">Risk Level</span>
            <span className="font-bold px-2 py-0.5 rounded-full text-[10px]" style={{ background: riskColor + '22', color: riskColor, border: `1px solid ${riskColor}44` }}>{riskLevel}</span>
          </div>
          <div className="w-full h-2 bg-slate-700 rounded-full overflow-hidden">
            <div className="h-full rounded-full animate-[stage-fill_1s_ease-out_forwards]" style={{ width: `${fraudRatio * 100}%`, background: `linear-gradient(90deg, ${riskColor}88, ${riskColor})` }} />
          </div>
        </div>
        {/* Centrality Metrics */}
        <div className="grid grid-cols-2 gap-2">
          <div className="bg-slate-800/60 rounded-lg px-2 py-1.5">
            <div className="text-slate-500 text-[9px]">PageRank</div>
            <div className="text-blue-400 font-mono font-semibold">{pr.toFixed(4)}</div>
          </div>
          <div className="bg-slate-800/60 rounded-lg px-2 py-1.5">
            <div className="text-slate-500 text-[9px]">Betweenness</div>
            <div className="text-purple-400 font-mono font-semibold">{bc.toFixed(4)}</div>
          </div>
          <div className="bg-slate-800/60 rounded-lg px-2 py-1.5">
            <div className="text-slate-500 text-[9px]">Connections</div>
            <div className="text-slate-200 font-mono font-semibold">{nodeEdges.length}</div>
          </div>
          <div className="bg-slate-800/60 rounded-lg px-2 py-1.5">
            <div className="text-slate-500 text-[9px]">Volume</div>
            <div className="text-emerald-400 font-mono font-semibold">${totalAmount.toLocaleString()}</div>
          </div>
        </div>
        {/* Flow Direction */}
        <div>
          <div className="text-slate-500 text-[9px] mb-1">Flow Direction</div>
          <div className="space-y-1">
            <div className="flex items-center gap-2">
              <span className="text-green-400 w-10">IN</span>
              <div className="flex-1 h-1.5 bg-slate-700 rounded-full overflow-hidden">
                <div className="h-full bg-green-500/70 rounded-full" style={{ width: `${(inflowAmt / maxFlow) * 100}%` }} />
              </div>
              <span className="text-slate-400 text-[9px] w-16 text-right">${inflowAmt.toLocaleString()}</span>
            </div>
            <div className="flex items-center gap-2">
              <span className="text-red-400 w-10">OUT</span>
              <div className="flex-1 h-1.5 bg-slate-700 rounded-full overflow-hidden">
                <div className="h-full bg-red-500/70 rounded-full" style={{ width: `${(outflowAmt / maxFlow) * 100}%` }} />
              </div>
              <span className="text-slate-400 text-[9px] w-16 text-right">${outflowAmt.toLocaleString()}</span>
            </div>
          </div>
        </div>
        {/* Fraud Breakdown */}
        {fraudBreakdown.size > 0 && (
          <div>
            <div className="text-slate-500 text-[9px] mb-1">Fraud Breakdown</div>
            <div className="space-y-1">
              {[...fraudBreakdown.entries()].map(([fl, count]) => (
                <div key={fl} className="flex items-center gap-2">
                  <div className="w-2 h-2 rounded-full" style={{ background: FRAUD_TYPE_COLORS[fl] }} />
                  <span className="text-slate-300 flex-1">{FRAUD_PATTERN_LABELS[fl] ?? `Type ${fl}`}</span>
                  <span className="text-slate-400">{count}</span>
                </div>
              ))}
            </div>
          </div>
        )}
        {/* Channels */}
        {channels.size > 0 && (
          <div>
            <div className="text-slate-500 text-[9px] mb-1">Channels</div>
            <div className="flex flex-wrap gap-1">
              {[...channels.entries()].map(([ch, count]) => (
                <span key={ch} className="px-1.5 py-0.5 bg-slate-800/80 rounded text-slate-400 text-[9px]">{ch} ({count})</span>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  )
}

// ============================================================================
// VFX wrappers (inside SigmaContainer for sigma context access)
// ============================================================================
function VisualEffects({ particles, ripples, heat, depth3D }: { particles: boolean; ripples: boolean; heat: boolean; depth3D: boolean }) {
  const sigma = useSigma()
  useParticleFlow(sigma, particles)
  useThreatRipples(sigma, ripples)
  useHeatOverlay(sigma, heat)
  use3DDepthOverlay(sigma, depth3D)
  return null
}

function MinimapWrapper({ visible }: { visible: boolean }) {
  if (!visible) return null
  return <Minimap />
}

// ============================================================================
// VFX Toggle Switch
// ============================================================================
function VfxToggle({ label, enabled, onChange, shortcut }: { label: string; enabled: boolean; onChange: () => void; shortcut?: string }) {
  return (
    <button onClick={onChange} className="flex items-center justify-between w-full px-2 py-1 rounded hover:bg-slate-700/50 transition-colors text-[10px]">
      <span className="text-slate-300">{label}</span>
      <div className="flex items-center gap-1.5">
        {shortcut && <span className="text-slate-600 text-[8px] font-mono">{shortcut}</span>}
        <div className={`w-7 h-4 rounded-full transition-colors duration-200 relative ${enabled ? 'bg-blue-500' : 'bg-slate-600'}`}>
          <div className={`absolute top-0.5 w-3 h-3 rounded-full bg-white shadow-sm transition-transform duration-200 ${enabled ? 'left-3.5' : 'left-0.5'}`} />
        </div>
      </div>
    </button>
  )
}

// ============================================================================
// Graph Controls -- toolbar with filters, VFX panel, search
// ============================================================================
function GraphControls({
  filter, setFilter, showLabels, setShowLabels,
  searchTerm, setSearchTerm, onRestart, onCycles,
  particlesEnabled, setParticlesEnabled, ripplesEnabled, setRipplesEnabled,
  heatEnabled, setHeatEnabled, physicsEnabled, setPhysicsEnabled,
  minimapVisible, setMinimapVisible, depth3DEnabled, setDepth3DEnabled, graph,
}: {
  filter: GraphFilter; setFilter: (f: GraphFilter) => void
  showLabels: boolean; setShowLabels: (v: boolean) => void
  searchTerm: string; setSearchTerm: (s: string) => void
  onRestart: () => void; onCycles: () => void
  particlesEnabled: boolean; setParticlesEnabled: (v: boolean) => void
  ripplesEnabled: boolean; setRipplesEnabled: (v: boolean) => void
  heatEnabled: boolean; setHeatEnabled: (v: boolean) => void
  physicsEnabled: boolean; setPhysicsEnabled: (v: boolean) => void
  minimapVisible: boolean; setMinimapVisible: (v: boolean) => void
  depth3DEnabled: boolean; setDepth3DEnabled: (v: boolean) => void
  graph: Graph
}) {
  const sigma = useSigma()
  const [showVfx, setShowVfx] = useState(false)
  const [showSearch, setShowSearch] = useState(false)

  const zoom = (dir: 1 | -1) => {
    const cam = sigma.getCamera()
    cam.animatedZoom({ duration: 200, factor: dir === 1 ? 1 / 1.5 : 1.5 })
  }
  const fit = () => sigma.getCamera().animatedReset({ duration: 300 })

  const doCenterFraud = () => {
    let target: string | null = null
    graph.forEachNode((n: string, a: Record<string, unknown>) => {
      if (!target && (a.status === 'frozen' || Number(a.fraudRatio ?? 0) > 0.5)) target = n
    })
    if (target && graph.hasNode(target)) {
      const attrs = graph.getNodeAttributes(target)
      sigma.getCamera().animate({ x: attrs.x as number, y: attrs.y as number }, { duration: 600 })
    }
  }

  const doExport = () => {
    const layers = ['edges', 'nodes', 'labels'] as const
    const canvas = document.createElement('canvas')
    const ctx = canvas.getContext('2d')
    if (!ctx) return
    const w = sigma.getContainer().offsetWidth
    const h = sigma.getContainer().offsetHeight
    canvas.width = w; canvas.height = h
    ctx.fillStyle = '#0f172a'; ctx.fillRect(0, 0, w, h)
    layers.forEach(id => {
      const layer = sigma.getCanvases()[id]
      if (layer) ctx.drawImage(layer, 0, 0, w, h)
    })
    const link = document.createElement('a')
    link.download = `payflow-graph-${Date.now()}.png`
    link.href = canvas.toDataURL('image/png')
    link.click()
  }

  // keyboard shortcuts
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (e.target instanceof HTMLInputElement) return
      if (e.key === 'p' || e.key === 'P') setParticlesEnabled(!particlesEnabled)
      if (e.key === 'h' || e.key === 'H') setHeatEnabled(!heatEnabled)
      if (e.key === 'd' || e.key === 'D') setDepth3DEnabled(!depth3DEnabled)
    }
    window.addEventListener('keydown', handler)
    return () => window.removeEventListener('keydown', handler)
  }, [particlesEnabled, heatEnabled, depth3DEnabled, setParticlesEnabled, setHeatEnabled, setDepth3DEnabled])

  const filterButtons: { key: GraphFilter; icon: React.ReactNode; label: string }[] = [
    { key: 'none', icon: <Network size={13} />, label: 'All' },
    { key: 'fraud-only', icon: <AlertTriangle size={13} />, label: 'Fraud' },
    { key: 'high-risk', icon: <Filter size={13} />, label: 'High Risk' },
    { key: 'threat-path', icon: <Radio size={13} />, label: 'Threats' },
    { key: 'cycles', icon: <Repeat size={13} />, label: 'Cycles' },
    { key: 'community', icon: <Tag size={13} />, label: 'Community' },
  ]

  return (
    <div className="absolute top-3 left-3 z-20 flex flex-col gap-1.5">
      {/* Filter row */}
      <div className="flex items-center gap-1 bg-slate-900/90 border border-slate-700/50 rounded-lg px-1.5 py-1 shadow-lg backdrop-blur-sm">
        {filterButtons.map(fb => (
          <button key={fb.key} onClick={() => { if (fb.key === 'cycles') onCycles(); setFilter(fb.key) }}
            className={`flex items-center gap-1 px-2 py-1 rounded text-[10px] transition-all ${filter === fb.key ? 'bg-blue-500/20 text-blue-400 shadow-sm' : 'text-slate-400 hover:text-slate-200 hover:bg-slate-700/50'}`}
            title={fb.label}>
            {fb.icon}
            <span className="hidden sm:inline">{fb.label}</span>
          </button>
        ))}
      </div>
      {/* Tool row */}
      <div className="flex items-center gap-1 bg-slate-900/90 border border-slate-700/50 rounded-lg px-1.5 py-1 shadow-lg backdrop-blur-sm">
        <button onClick={() => zoom(1)} className="p-1.5 text-slate-400 hover:text-slate-200 hover:bg-slate-700/50 rounded transition-colors" title="Zoom in"><ZoomIn size={13} /></button>
        <button onClick={() => zoom(-1)} className="p-1.5 text-slate-400 hover:text-slate-200 hover:bg-slate-700/50 rounded transition-colors" title="Zoom out"><ZoomOut size={13} /></button>
        <button onClick={fit} className="p-1.5 text-slate-400 hover:text-slate-200 hover:bg-slate-700/50 rounded transition-colors" title="Fit"><Maximize2 size={13} /></button>
        <div className="w-px h-4 bg-slate-700/50 mx-0.5" />
        <button onClick={() => setShowLabels(!showLabels)} className={`p-1.5 rounded transition-colors ${showLabels ? 'text-blue-400 bg-blue-500/10' : 'text-slate-400 hover:text-slate-200 hover:bg-slate-700/50'}`} title="Labels"><Type size={13} /></button>
        <button onClick={doCenterFraud} className="p-1.5 text-slate-400 hover:text-slate-200 hover:bg-slate-700/50 rounded transition-colors" title="Center on fraud"><Crosshair size={13} /></button>
        <button onClick={onRestart} className="p-1.5 text-slate-400 hover:text-slate-200 hover:bg-slate-700/50 rounded transition-colors" title="Restart layout"><RotateCcw size={13} /></button>
        <button onClick={doExport} className="p-1.5 text-slate-400 hover:text-slate-200 hover:bg-slate-700/50 rounded transition-colors" title="Export PNG"><Download size={13} /></button>
        <div className="w-px h-4 bg-slate-700/50 mx-0.5" />
        <button onClick={() => setShowSearch(!showSearch)} className={`p-1.5 rounded transition-colors ${showSearch ? 'text-blue-400 bg-blue-500/10' : 'text-slate-400 hover:text-slate-200 hover:bg-slate-700/50'}`} title="Search"><Search size={13} /></button>
        <button onClick={() => setShowVfx(!showVfx)} className={`p-1.5 rounded transition-colors ${showVfx ? 'text-blue-400 bg-blue-500/10' : 'text-slate-400 hover:text-slate-200 hover:bg-slate-700/50'}`} title="Visual Effects"><Layers size={13} /></button>
      </div>
      {/* Search input */}
      {showSearch && (
        <div className="bg-slate-900/90 border border-slate-700/50 rounded-lg px-2 py-1.5 shadow-lg backdrop-blur-sm animate-[slide-up_0.2s_ease-out]">
          <input type="text" value={searchTerm} onChange={e => setSearchTerm(e.target.value)} placeholder="Search nodes..."
            className="w-full bg-transparent border-none outline-none text-xs text-slate-200 placeholder:text-slate-500" autoFocus />
        </div>
      )}
      {/* VFX panel */}
      {showVfx && (
        <div className="bg-slate-900/90 border border-slate-700/50 rounded-lg px-2 py-2 shadow-lg backdrop-blur-sm animate-[slide-up_0.2s_ease-out] min-w-[180px]">
          <div className="text-[9px] text-slate-500 font-semibold uppercase tracking-wider mb-1.5 flex items-center gap-1"><Zap size={10} /> Visual Effects</div>
          <div className="space-y-0.5">
            <VfxToggle label="Particle Flow" enabled={particlesEnabled} onChange={() => setParticlesEnabled(!particlesEnabled)} shortcut="P" />
            <VfxToggle label="Threat Ripples" enabled={ripplesEnabled} onChange={() => setRipplesEnabled(!ripplesEnabled)} />
            <VfxToggle label="Heat Diffusion" enabled={heatEnabled} onChange={() => setHeatEnabled(!heatEnabled)} shortcut="H" />
            <VfxToggle label="Live Physics" enabled={physicsEnabled} onChange={() => setPhysicsEnabled(!physicsEnabled)} />
            <VfxToggle label="Minimap" enabled={minimapVisible} onChange={() => setMinimapVisible(!minimapVisible)} />
            <VfxToggle label="3D Depth" enabled={depth3DEnabled} onChange={() => setDepth3DEnabled(!depth3DEnabled)} shortcut="D" />
          </div>
        </div>
      )}
    </div>
  )
}

// ============================================================================
// Graph Legend
// ============================================================================
function GraphLegend() {
  const [collapsed, setCollapsed] = useState(true)
  return (
    <div className="absolute bottom-3 right-3 z-20">
      <button onClick={() => setCollapsed(!collapsed)}
        className="bg-slate-900/90 border border-slate-700/50 rounded-lg px-2.5 py-1.5 text-[10px] text-slate-400 hover:text-slate-200 shadow-lg backdrop-blur-sm transition-colors flex items-center gap-1">
        <Network size={11} /> Legend {collapsed ? '+' : '−'}
      </button>
      {!collapsed && (
        <div className="mt-1 bg-slate-900/95 border border-slate-700/50 rounded-lg px-3 py-2.5 shadow-xl backdrop-blur-sm animate-[slide-up_0.2s_ease-out] min-w-[160px]">
          <div className="text-[9px] text-slate-500 font-semibold uppercase tracking-wider mb-2">Fraud Types</div>
          <div className="space-y-1 mb-3">
            {Object.entries(FRAUD_PATTERN_LABELS).filter(([k]) => Number(k) > 0).map(([k, label]) => (
              <div key={k} className="flex items-center gap-2 text-[10px]">
                <div className="w-2.5 h-2.5 rounded-full" style={{ background: FRAUD_TYPE_COLORS[Number(k)] }} />
                <span className="text-slate-300">{label}</span>
              </div>
            ))}
          </div>
          <div className="text-[9px] text-slate-500 font-semibold uppercase tracking-wider mb-2">Status</div>
          <div className="space-y-1 mb-3">
            {Object.entries(STATUS_BORDER_COLORS).map(([status, color]) => (
              <div key={status} className="flex items-center gap-2 text-[10px]">
                <div className="w-2.5 h-2.5 rounded-full border-2" style={{ borderColor: color, background: 'transparent' }} />
                <span className="text-slate-300 capitalize">{status}</span>
              </div>
            ))}
          </div>
          <div className="text-[9px] text-slate-500 font-semibold uppercase tracking-wider mb-1">Visual Key</div>
          <div className="space-y-0.5 text-[9px] text-slate-400">
            <div>Size = connection count + fraud ratio</div>
            <div>Glow = new fraud detection</div>
            <div>Particles = active fraud flow</div>
            <div>Ripples = high-threat source</div>
          </div>
        </div>
      )}
    </div>
  )
}

// ============================================================================
// Graph Stats Overlay -- enhanced with sparkline and key actors
// ============================================================================
function GraphStatsOverlay({
  graph, edges, clusteringCoeff, pageRanks, betweenness,
}: {
  graph: Graph; edges: CytoEdge[]; clusteringCoeff: number
  pageRanks: Map<string, number>; betweenness: Map<string, number>
}) {
  const nodeCount = graph.order
  const edgeCount = graph.size
  const fraudEdgeCount = edges.filter(e => (e.data.fraud_label ?? 0) > 0).length
  const fraudPercent = edgeCount > 0 ? ((fraudEdgeCount / edgeCount) * 100).toFixed(1) : '0'

  // Risk distribution
  let frozen = 0, suspicious = 0, paused = 0, normal = 0
  graph.forEachNode((_: string, a: Record<string, unknown>) => {
    if (a.status === 'frozen') frozen++
    else if (a.status === 'suspicious') suspicious++
    else if (a.status === 'paused') paused++
    else normal++
  })
  const total = frozen + suspicious + paused + normal || 1

  // Active fraud patterns
  const activePatterns = new Set<number>()
  edges.forEach(e => { const fl = e.data.fraud_label ?? 0; if (fl > 0) activePatterns.add(fl) })

  // Key actors
  let topPR = '', topPRVal = 0
  pageRanks.forEach((v, k) => { if (v > topPRVal) { topPR = k; topPRVal = v } })
  let topBC = '', topBCVal = 0
  betweenness.forEach((v, k) => { if (v > topBCVal) { topBC = k; topBCVal = v } })

  return (
    <div className="absolute top-3 right-3 z-20 bg-slate-900/90 border border-slate-700/50 rounded-lg shadow-lg backdrop-blur-sm w-56">
      <div className="px-3 py-2 border-b border-slate-700/50 flex items-center justify-between">
        <span className="text-[10px] text-slate-400 font-semibold uppercase tracking-wider flex items-center gap-1">
          <TrendingUp size={10} /> Network Stats
        </span>
        <ActivitySparkline />
      </div>
      <div className="px-3 py-2 space-y-2 text-[10px]">
        <div className="grid grid-cols-2 gap-x-4 gap-y-1">
          <div className="flex justify-between"><span className="text-slate-500">Nodes</span><span className="text-slate-200 font-mono">{nodeCount}</span></div>
          <div className="flex justify-between"><span className="text-slate-500">Edges</span><span className="text-slate-200 font-mono">{edgeCount}</span></div>
          <div className="flex justify-between"><span className="text-slate-500">Fraud</span><span className="text-red-400 font-mono">{fraudEdgeCount}</span></div>
          <div className="flex justify-between"><span className="text-slate-500">Ratio</span><span className="text-amber-400 font-mono">{fraudPercent}%</span></div>
          <div className="flex justify-between col-span-2"><span className="text-slate-500">Clustering</span><span className="text-cyan-400 font-mono">{clusteringCoeff.toFixed(3)}</span></div>
        </div>
        {/* Risk distribution bar */}
        <div>
          <div className="text-[9px] text-slate-500 mb-1">Risk Distribution</div>
          <div className="w-full h-2 rounded-full overflow-hidden flex">
            {frozen > 0 && <div style={{ width: `${(frozen/total)*100}%`, background: '#f87171' }} className="h-full" />}
            {suspicious > 0 && <div style={{ width: `${(suspicious/total)*100}%`, background: '#facc15' }} className="h-full" />}
            {paused > 0 && <div style={{ width: `${(paused/total)*100}%`, background: '#fb923c' }} className="h-full" />}
            {normal > 0 && <div style={{ width: `${(normal/total)*100}%`, background: '#475569' }} className="h-full" />}
          </div>
          <div className="flex justify-between mt-0.5 text-[8px] text-slate-500">
            <span>{frozen} frozen</span><span>{suspicious} susp</span><span>{normal} ok</span>
          </div>
        </div>
        {/* Key Actors */}
        {(topPR || topBC) && (
          <div>
            <div className="text-[9px] text-slate-500 mb-1">Key Actors</div>
            {topPR && (
              <div className="flex items-center gap-1 mb-0.5">
                <Target size={9} className="text-blue-400" />
                <span className="text-slate-400">PR:</span>
                <span className="text-blue-400 font-mono truncate max-w-[100px]">{topPR.length > 12 ? topPR.slice(0,6)+'..'+topPR.slice(-4) : topPR}</span>
              </div>
            )}
            {topBC && (
              <div className="flex items-center gap-1">
                <Zap size={9} className="text-purple-400" />
                <span className="text-slate-400">BC:</span>
                <span className="text-purple-400 font-mono truncate max-w-[100px]">{topBC.length > 12 ? topBC.slice(0,6)+'..'+topBC.slice(-4) : topBC}</span>
              </div>
            )}
          </div>
        )}
        {/* Active patterns */}
        {activePatterns.size > 0 && (
          <div className="flex flex-wrap gap-1">
            {[...activePatterns].map(fl => (
              <span key={fl} className="px-1.5 py-0.5 rounded text-[8px] font-medium" style={{ background: FRAUD_TYPE_COLORS[fl] + '22', color: FRAUD_TYPE_COLORS[fl] }}>
                {FRAUD_PATTERN_LABELS[fl] ?? `T${fl}`}
              </span>
            ))}
          </div>
        )}
      </div>
    </div>
  )
}

// ============================================================================
// Threat Radar -- animated scanning ring visible in threat-path mode
// ============================================================================
function ThreatRadar({ active }: { active: boolean }) {
  const [signalCount, setSignalCount] = useState(0)
  useEffect(() => {
    if (!active) return
    const iv = setInterval(() => setSignalCount(Math.floor(Math.random() * 12) + 1), 3000)
    return () => clearInterval(iv)
  }, [active])

  if (!active) return null

  return (
    <div className="absolute bottom-20 right-3 z-20 flex flex-col items-center gap-1">
      <div className="relative w-16 h-16">
        <svg viewBox="0 0 100 100" className="w-full h-full">
          {/* rings */}
          <circle cx="50" cy="50" r="45" fill="none" stroke="#334155" strokeWidth="0.5" />
          <circle cx="50" cy="50" r="30" fill="none" stroke="#334155" strokeWidth="0.5" />
          <circle cx="50" cy="50" r="15" fill="none" stroke="#334155" strokeWidth="0.5" />
          {/* crosshairs */}
          <line x1="50" y1="2" x2="50" y2="98" stroke="#334155" strokeWidth="0.5" />
          <line x1="2" y1="50" x2="98" y2="50" stroke="#334155" strokeWidth="0.5" />
          {/* scanning line -- animated */}
          <line x1="50" y1="50" x2="50" y2="5" stroke="#22d3ee" strokeWidth="1.5" opacity="0.7" className="origin-center animate-[radar-spin_3s_linear_infinite]" style={{ transformOrigin: '50px 50px' }} />
          {/* center dot */}
          <circle cx="50" cy="50" r="3" fill="#22d3ee" className="animate-pulse" />
        </svg>
      </div>
      <div className="bg-slate-900/80 border border-cyan-900/40 rounded px-2 py-0.5 text-[9px] text-cyan-400 font-mono backdrop-blur-sm flex items-center gap-1">
        <Radio size={8} className="animate-pulse" /> SCANNING · {signalCount} signals
      </div>
    </div>
  )
}

// ============================================================================
// Main SigmaGraph component
// ============================================================================
export default function SigmaGraph() {
  // -- Data fetching from dashboard store (populated via SSE + initial topology) --
  const graphNodes = useDashboardStore((s) => s.graphNodes)
  const graphEdges = useDashboardStore((s) => s.graphEdges)
  const setInitialTopology = useDashboardStore((s) => s.setInitialTopology)
  const { data: topology } = useTopology()
  const syncedRef = useRef(false)

  // Seed store from REST topology on first load
  useEffect(() => {
    if (topology && !syncedRef.current) {
      syncedRef.current = true
      setInitialTopology(topology.nodes, topology.edges)
    }
  }, [topology, setInitialTopology])

  const nodes: CytoNode[] = graphNodes
  const edges: CytoEdge[] = graphEdges

  // -- UI state --
  const [filter, setFilter] = useState<GraphFilter>('none')
  const [showLabels, setShowLabels] = useState(true)
  const [showEdgeLabels, _setShowEdgeLabels] = useState(false)
  const [searchTerm, setSearchTerm] = useState('')
  const [hoveredNode, setHoveredNode] = useState<string | null>(null)
  const [selectedNode, setSelectedNode] = useState<string | null>(null)
  const [tooltip, setTooltip] = useState<TooltipData | null>(null)
  const [edgeTooltip, setEdgeTooltip] = useState<EdgeTooltipData | null>(null)
  const [cycleNodes, setCycleNodes] = useState<Set<string>>(new Set())

  // VFX toggles
  const [particlesEnabled, setParticlesEnabled] = useState(true)
  const [ripplesEnabled, setRipplesEnabled] = useState(true)
  const [heatEnabled, setHeatEnabled] = useState(false)
  const [physicsEnabled, setPhysicsEnabled] = useState(false)
  const [minimapVisible, setMinimapVisible] = useState(true)
  const [depth3DEnabled, setDepth3DEnabled] = useState(true)

  // Advanced analytics
  const [pageRanks, setPageRanks] = useState<Map<string, number>>(new Map())
  const [betweennessMap, setBetweennessMap] = useState<Map<string, number>>(new Map())
  const [clusteringCoeff, setClusteringCoeff] = useState(0)
  const [threatLevels, setThreatLevels] = useState<Map<string, number>>(new Map())

  const graphRef = useRef<Graph>(new Graph())
  const [version, setVersion] = useState(0)
  const setSelectedNodeUI = useUIStore(s => s.setSelectedNode)

  // -- Sync graphology on topology changes --
  useEffect(() => {
    if (nodes.length === 0 && edges.length === 0) return
    syncGraphology(graphRef.current, nodes, edges)
    setVersion(v => v + 1)
    recordActivity(graphRef.current.size)
  }, [nodes, edges])

  // -- Recompute metrics (debounced) --
  useEffect(() => {
    const g = graphRef.current
    if (g.order === 0) return
    const timer = setTimeout(() => {
      setPageRanks(computePageRank(g))
      setBetweennessMap(computeBetweenness(g))
      setClusteringCoeff(computeClusteringCoeff(g))
      setThreatLevels(computeThreatPaths(g))
    }, 500)
    return () => clearTimeout(timer)
  }, [version])

  // -- Handlers --
  const handleNodeClick = useCallback((nodeId: string) => {
    setSelectedNode(prev => prev === nodeId ? null : nodeId)
    setSelectedNodeUI(nodeId)
  }, [setSelectedNodeUI])

  const handleRestart = useCallback(() => {
    setVersion(v => v + 1)
    setFilter('none')
    setCycleNodes(new Set())
    setSearchTerm('')
    setSelectedNode(null)
    setTooltip(null)
    setEdgeTooltip(null)
  }, [])

  const handleCycles = useCallback(() => {
    const cycles = detectCycles(graphRef.current)
    const nodeSet = new Set<string>()
    cycles.forEach(c => c.forEach(n => nodeSet.add(n)))
    setCycleNodes(nodeSet)
    if (nodeSet.size > 0) setFilter('cycles')
  }, [])

  const handleNodeHover = useCallback((data: TooltipData) => {
    setHoveredNode(data.id)
    setTooltip(data)
  }, [])

  const handleEdgeHover = useCallback((data: EdgeTooltipData) => {
    setEdgeTooltip(data)
  }, [])

  // -- Loading state --
  if (nodes.length === 0 && edges.length === 0) {
    return (
      <div className="flex items-center justify-center h-full bg-slate-950/50 rounded-xl border border-slate-800">
        <div className="text-center space-y-3">
          <Network size={32} className="text-slate-600 mx-auto animate-pulse" />
          <p className="text-xs text-slate-500">Waiting for graph topology…</p>
          <div className="w-32 h-1 bg-slate-800 rounded-full overflow-hidden mx-auto">
            <div className="h-full bg-gradient-to-r from-blue-500 to-cyan-500 animate-[shimmer_1.5s_ease-in-out_infinite] w-1/3" />
          </div>
        </div>
      </div>
    )
  }

  return (
    <div className="relative w-full h-full rounded-xl overflow-hidden border border-slate-800 bg-slate-950">
      <SigmaContainer
        graph={graphRef.current}
        style={{ width: '100%', height: '100%', background: '#0f172a' }}
        settings={{
          renderLabels: showLabels,
          renderEdgeLabels: showEdgeLabels,
          labelColor: { color: '#d0dde8' },
          labelFont: 'Inter, system-ui, sans-serif',
          labelSize: 12,
          labelWeight: '600',
          labelDensity: 1.0,
          labelGridCellSize: 80,
          labelRenderedSizeThreshold: 1,
          edgeLabelFont: 'Inter, system-ui, sans-serif',
          edgeLabelSize: 7,
          edgeLabelColor: { color: '#506070' },
          defaultEdgeType: 'line',
          defaultNodeColor: NODE_COLOR_SAFE,
          stagePadding: 30,
          zoomDuration: 250,
          inertiaDuration: 350,
          inertiaRatio: 0.55,
          minCameraRatio: 0.03,
          maxCameraRatio: 5,
          allowInvalidContainer: true,
          defaultEdgeColor: 'rgba(140,170,200,0.25)',
          zIndex: true,
        }}
      >
        <GraphEvents
          onNodeClick={handleNodeClick}
          onNodeHover={handleNodeHover}
          onNodeOut={() => { setHoveredNode(null); setTooltip(null) }}
          onEdgeHover={handleEdgeHover}
          onEdgeOut={() => setEdgeTooltip(null)}
          filter={filter}
          cycleNodes={cycleNodes}
          searchTerm={searchTerm}
          hoveredNode={hoveredNode}
          threatLevels={threatLevels}
        />
        <GraphControls
          filter={filter}
          setFilter={setFilter}
          showLabels={showLabels}
          setShowLabels={setShowLabels}
          searchTerm={searchTerm}
          setSearchTerm={setSearchTerm}
          onRestart={handleRestart}
          onCycles={handleCycles}
          particlesEnabled={particlesEnabled}
          setParticlesEnabled={setParticlesEnabled}
          ripplesEnabled={ripplesEnabled}
          setRipplesEnabled={setRipplesEnabled}
          heatEnabled={heatEnabled}
          setHeatEnabled={setHeatEnabled}
          physicsEnabled={physicsEnabled}
          setPhysicsEnabled={setPhysicsEnabled}
          minimapVisible={minimapVisible}
          setMinimapVisible={setMinimapVisible}
          depth3DEnabled={depth3DEnabled}
          setDepth3DEnabled={setDepth3DEnabled}
          graph={graphRef.current}
        />
        <ContinuousLayout enabled={physicsEnabled} />
        <VisualEffects particles={particlesEnabled} ripples={ripplesEnabled} heat={heatEnabled} depth3D={depth3DEnabled} />
        <MinimapWrapper visible={minimapVisible} />
      </SigmaContainer>

      {/* Overlays */}
      {tooltip && <NodeTooltip data={tooltip} />}
      {edgeTooltip && <EdgeTooltip data={edgeTooltip} />}
      {selectedNode && graphRef.current.hasNode(selectedNode) && (
        <InGraphNodeDetailPanel
          nodeId={selectedNode}
          graph={graphRef.current}
          edges={edges}
          onClose={() => setSelectedNode(null)}
          pageRanks={pageRanks}
          betweenness={betweennessMap}
        />
      )}
      <GraphLegend />
      <GraphStatsOverlay
        graph={graphRef.current}
        edges={edges}
        clusteringCoeff={clusteringCoeff}
        pageRanks={pageRanks}
        betweenness={betweennessMap}
      />
      <ThreatRadar active={filter === 'threat-path'} />

      {/* Keyboard hints */}
      <div className="absolute bottom-2 left-1/2 -translate-x-1/2 z-10 flex items-center gap-3 text-[9px] text-slate-600 select-none">
        <span><kbd className="px-1 py-0.5 bg-slate-800 rounded text-slate-500 font-mono">P</kbd> particles</span>
        <span><kbd className="px-1 py-0.5 bg-slate-800 rounded text-slate-500 font-mono">H</kbd> heatmap</span>
        <span><kbd className="px-1 py-0.5 bg-slate-800 rounded text-slate-500 font-mono">D</kbd> 3D depth</span>
      </div>
    </div>
  )
}

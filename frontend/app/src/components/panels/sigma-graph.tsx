// ============================================================================
// 3D Force Graph -- Interactive real-time fraud network visualization
// WebGL-powered Three.js 3D graph with full orbit controls
// Left-drag: rotate | Right-drag: pan | Scroll: zoom
// ============================================================================

import { useEffect, useRef, useState, useCallback, useMemo } from 'react'
import ForceGraph3D from 'react-force-graph-3d'
import * as THREE from 'three'
import SpriteText from 'three-spritetext'
import Graph from 'graphology'

import { useDashboardStore } from '@/stores/use-dashboard-store'
import { useUIStore } from '@/stores/use-ui-store'
import { useTopology } from '@/hooks/use-api'
import { FRAUD_PATTERN_LABELS } from '@/lib/types'
import type { CytoNode, CytoEdge } from '@/lib/types'
import {
  Network, AlertTriangle, ZoomIn, ZoomOut, Maximize2,
  Tag, RotateCcw, Filter, Search, Crosshair, Download,
  Repeat, Radio, Zap, TrendingUp, Target, SlidersHorizontal,
  Flame, Users, Layers, Shield,
} from 'lucide-react'

// --- Constants ---
const MAX_NODES = 300
const NODE_COLOR_SAFE = '#4a90d9'

// Degree-based color gradient for safe (non-fraud) nodes — visual hierarchy
const SAFE_DEGREE_COLORS: Record<string, string> = {
  hub:    '#38bdf8',  // sky-400 — high-degree hubs
  active: '#60a5fa',  // blue-400 — mid-degree
  normal: '#818cf8',  // indigo-400 — low-degree with connections
  leaf:   '#a78bfa',  // violet-400 — leaf/isolated nodes
}

const FRAUD_TYPE_COLORS: Record<number, string> = {
  0: '#4a90d9', 1: '#ff6b6b', 2: '#a855f7', 3: '#2dd4bf',
  4: '#818cf8', 5: '#f472b6', 6: '#fb923c', 7: '#ef4444', 8: '#22d3ee',
}

// Status-based node color overrides
const STATUS_NODE_COLORS: Record<string, string> = {
  frozen: '#ef4444',
  suspicious: '#f59e0b',
  paused: '#fb923c',
}
const STATUS_BORDER_COLORS: Record<string, string> = {
  frozen: '#e05656', suspicious: '#d4ad2a', paused: '#d98044', normal: '#384858',
}
const COMMUNITY_COLORS = [
  '#ff6b6b','#4ecdc4','#45b7d1','#96ceb4','#feca57','#ff9ff3','#54a0ff','#5f27cd',
]

const CHANNEL_LABELS: Record<number, string> = {
  0: 'Branch', 1: 'ATM', 2: 'Net Banking', 3: 'Mobile', 4: 'UPI',
  5: 'RTGS', 6: 'NEFT', 7: 'IMPS', 8: 'SWIFT', 9: 'POS',
}
const CHANNEL_COLORS: Record<number, string> = {
  0: '#94a3b8', 1: '#f59e0b', 2: '#3b82f6', 3: '#10b981', 4: '#8b5cf6',
  5: '#ef4444', 6: '#ec4899', 7: '#06b6d4', 8: '#f97316', 9: '#6366f1',
}

// Pre-index edges by node for O(N+E) lookups instead of O(N*E)
function buildEdgeIndex(edges: CytoEdge[]): Map<string, CytoEdge[]> {
  const idx = new Map<string, CytoEdge[]>()
  for (const e of edges) {
    const s = e.data.source, t = e.data.target
    if (!idx.has(s)) idx.set(s, [])
    idx.get(s)!.push(e)
    if (s !== t) {
      if (!idx.has(t)) idx.set(t, [])
      idx.get(t)!.push(e)
    }
  }
  return idx
}

function dominantFraudType(nodeId: string, edgeIdx: Map<string, CytoEdge[]>): number {
  const counts = new Map<number, number>()
  for (const e of edgeIdx.get(nodeId) ?? []) {
    const fl = e.data.fraud_label ?? 0
    if (fl === 0) continue
    counts.set(fl, (counts.get(fl) ?? 0) + 1)
  }
  if (counts.size === 0) return 0
  let best = 0, bestCount = 0
  for (const [fl, c] of counts) { if (c > bestCount) { best = fl; bestCount = c } }
  return best
}

// ============================================================================
// Graph algorithms
// ============================================================================
function selectTopNodes(nodes: CytoNode[], edges: CytoEdge[], edgeIdx?: Map<string, CytoEdge[]>): CytoNode[] {
  if (nodes.length <= MAX_NODES) return nodes
  const idx = edgeIdx ?? buildEdgeIndex(edges)
  const degreeMap = new Map<string, number>()
  for (const e of edges) {
    degreeMap.set(e.data.source, (degreeMap.get(e.data.source) ?? 0) + 1)
    degreeMap.set(e.data.target, (degreeMap.get(e.data.target) ?? 0) + 1)
  }
  const statusPriority: Record<string, number> = { frozen: 4, suspicious: 3, paused: 2, normal: 0 }
  const scored = nodes.map(n => {
    const nodeEdges = idx.get(n.data.id) ?? []
    const fraudCount = nodeEdges.filter(e => (e.data.fraud_label ?? 0) > 0).length
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
    const stackArr: string[] = []
    const pred = new Map<string, string[]>()
    const sigma = new Map<string, number>()
    const dist = new Map<string, number>()
    nodes.forEach(n => { pred.set(n, []); sigma.set(n, 0); dist.set(n, -1) })
    sigma.set(s, 1); dist.set(s, 0)
    const queue = [s]
    while (queue.length > 0) {
      const v = queue.shift()!
      stackArr.push(v)
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
    while (stackArr.length > 0) {
      const w = stackArr.pop()!
      for (const v of pred.get(w) ?? []) {
        const d = ((sigma.get(v) ?? 0) / (sigma.get(w) ?? 1)) * (1 + (delta.get(w) ?? 0))
        delta.set(v, (delta.get(v) ?? 0) + d)
      }
      if (w !== s) bc.set(w, (bc.get(w) ?? 0) + (delta.get(w) ?? 0))
    }
  }
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

// ============================================================================
// Mule Account Detection — fan-in/fan-out hub patterns (star topology)
// ============================================================================
function detectMuleAccounts(edges: CytoEdge[]): Set<string> {
  const mules = new Set<string>()
  const inSenders = new Map<string, Set<string>>()
  const outReceivers = new Map<string, Set<string>>()

  for (const e of edges) {
    if (!inSenders.has(e.data.target)) inSenders.set(e.data.target, new Set())
    inSenders.get(e.data.target)!.add(e.data.source)
    if (!outReceivers.has(e.data.source)) outReceivers.set(e.data.source, new Set())
    outReceivers.get(e.data.source)!.add(e.data.target)
  }

  const allNodes = new Set([...inSenders.keys(), ...outReceivers.keys()])
  for (const node of allNodes) {
    const iD = inSenders.get(node)?.size ?? 0
    const oD = outReceivers.get(node)?.size ?? 0
    if (iD + oD < 3) continue
    // Collector mule: many distinct senders → this account
    if (iD >= 4 && oD >= 1 && iD / Math.max(oD, 1) >= 2.5) mules.add(node)
    // Distributor mule: this account → many distinct receivers
    else if (oD >= 4 && iD >= 1 && oD / Math.max(iD, 1) >= 2.5) mules.add(node)
    // Passthrough mule: high throughput both directions
    else if (iD >= 3 && oD >= 3) mules.add(node)
  }
  return mules
}

// ============================================================================
// Layering Chain Detection — sequential fund movement paths (rapid layering)
// ============================================================================
function detectLayeringChains(edges: CytoEdge[]): Set<string> {
  const chainNodes = new Set<string>()
  const minChainLen = 3

  // Build directed adjacency
  const outAdj = new Map<string, Set<string>>()
  const inCount = new Map<string, number>()
  for (const e of edges) {
    if (!outAdj.has(e.data.source)) outAdj.set(e.data.source, new Set())
    outAdj.get(e.data.source)!.add(e.data.target)
    inCount.set(e.data.target, (inCount.get(e.data.target) ?? 0) + 1)
  }

  // Start from chain origins (low in-degree, has out-edges)
  const globalVisited = new Set<string>()
  for (const [startNode] of outAdj) {
    if ((inCount.get(startNode) ?? 0) > 1) continue
    if (globalVisited.has(startNode)) continue

    const path = [startNode]
    let current = startNode
    const pathVisited = new Set([startNode])

    while (path.length < 10) {
      const neighbors = outAdj.get(current)
      if (!neighbors) break
      const next = [...neighbors].find(n => !pathVisited.has(n))
      if (!next) break
      path.push(next)
      pathVisited.add(next)
      current = next
    }

    if (path.length >= minChainLen) {
      path.forEach(n => { chainNodes.add(n); globalVisited.add(n) })
    }
  }
  return chainNodes
}

// ============================================================================
// Structuring Detection — amounts near ₹10L reporting threshold
// ============================================================================
const STRUCTURING_THRESHOLD = 1_000_000 // ₹10 Lakh in amount units
const STRUCTURING_LOWER = STRUCTURING_THRESHOLD * 0.80

function detectStructuringEdges(edges: CytoEdge[]): Set<string> {
  const structuring = new Set<string>()
  for (const e of edges) {
    const amt = e.data.amount_paisa ?? 0
    if (amt >= STRUCTURING_LOWER && amt < STRUCTURING_THRESHOLD) {
      structuring.add(`${e.data.source}->${e.data.target}`)
    }
  }
  return structuring
}

// ============================================================================
// Activity tracking
// ============================================================================
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

// ============================================================================
// Sync graphology for analytics (no layout needed — ForceGraph3D handles it)
// ============================================================================
function syncGraphology(g: Graph, nodes: CytoNode[], edges: CytoEdge[]) {
  const edgeIdx = buildEdgeIndex(edges)
  const top = selectTopNodes(nodes, edges, edgeIdx)
  const topIds = new Set(top.map(n => n.data.id))
  recordActivity(edges.length)
  g.clear()

  const degMap = new Map<string, number>()
  for (const e of edges) {
    if (!topIds.has(e.data.source) || !topIds.has(e.data.target)) continue
    degMap.set(e.data.source, (degMap.get(e.data.source) ?? 0) + 1)
    degMap.set(e.data.target, (degMap.get(e.data.target) ?? 0) + 1)
  }

  for (const n of top) {
    const nodeEdges = edgeIdx.get(n.data.id) ?? []
    const fraudEdges = nodeEdges.filter(e => (e.data.fraud_label ?? 0) > 0)
    const fraudRatio = nodeEdges.length > 0 ? fraudEdges.length / nodeEdges.length : 0
    const dom = dominantFraudType(n.data.id, edgeIdx)
    const deg = degMap.get(n.data.id) ?? 0
    const color = dom > 0 ? FRAUD_TYPE_COLORS[dom] ?? '#c0392b' : NODE_COLOR_SAFE

    g.addNode(n.data.id, {
      status: n.data.status,
      fraudRatio,
      dominantFraud: dom,
      degree: deg,
      color,
    })
  }

  for (const e of edges) {
    if (!topIds.has(e.data.source) || !topIds.has(e.data.target)) continue
    if (e.data.source === e.data.target) continue
    const key = `${e.data.source}->${e.data.target}`
    if (g.hasEdge(key)) continue
    const fl = e.data.fraud_label ?? 0
    g.addEdgeWithKey(key, e.data.source, e.data.target, {
      fraud_label: fl,
      amount: e.data.amount_paisa ?? 0,
      channel: String(e.data.channel ?? 'unknown'),
    })
  }
}

// ============================================================================
// ForceGraph3D data types & builder
// ============================================================================
interface FGNode {
  id: string
  color: string
  val: number
  status: string
  fraudRatio: number
  dominantFraud: number
  degree: number
  borderColor: string
  channels: number[]
}

interface FGLink {
  source: string
  target: string
  color: string
  width: number
  fraud_label: number
  amount: number
  channel: string
}

type GraphFilter = 'none' | 'fraud-only' | 'high-risk' | 'cycles' | 'community' | 'threat-path' | 'mule' | 'layering'

// Fibonacci sphere — distributes N points uniformly across a sphere surface
function fibonacciSphere(n: number, radius: number): { x: number; y: number; z: number }[] {
  const pts: { x: number; y: number; z: number }[] = []
  const goldenAngle = Math.PI * (3 - Math.sqrt(5))
  for (let i = 0; i < n; i++) {
    const y = 1 - (i / (n - 1)) * 2 // -1 to 1
    const rSlice = Math.sqrt(1 - y * y)
    const theta = goldenAngle * i
    pts.push({ x: rSlice * Math.cos(theta) * radius, y: y * radius, z: rSlice * Math.sin(theta) * radius })
  }
  return pts
}

function buildForceGraphData(nodes: CytoNode[], edges: CytoEdge[]): { fgNodes: FGNode[]; fgLinks: FGLink[] } {
  const edgeIdx = buildEdgeIndex(edges)
  const top = selectTopNodes(nodes, edges, edgeIdx)
  const topIds = new Set(top.map(n => n.data.id))

  const degMap = new Map<string, number>()
  for (const e of edges) {
    if (!topIds.has(e.data.source) || !topIds.has(e.data.target)) continue
    degMap.set(e.data.source, (degMap.get(e.data.source) ?? 0) + 1)
    degMap.set(e.data.target, (degMap.get(e.data.target) ?? 0) + 1)
  }

  let maxDeg = 1
  degMap.forEach(d => { if (d > maxDeg) maxDeg = d })

  // Pre-compute Fibonacci sphere positions for uniform distribution
  const spherePositions = fibonacciSphere(top.length, 140)

  const fgNodes: FGNode[] = []
  for (const n of top) {
    const nodeEdges = edgeIdx.get(n.data.id) ?? []
    const fraudEdges = nodeEdges.filter(e => (e.data.fraud_label ?? 0) > 0)
    const fraudRatio = nodeEdges.length > 0 ? fraudEdges.length / nodeEdges.length : 0
    const dom = dominantFraudType(n.data.id, edgeIdx)
    const deg = degMap.get(n.data.id) ?? 0

    let sz: number
    // Size by importance — fraud/status nodes are larger for visual hierarchy
    const isFraud = dom > 0
    const isHighRisk = n.data.status === 'frozen' || n.data.status === 'suspicious'
    if (isHighRisk) sz = 4.5
    else if (isFraud && deg >= 4) sz = 3.8
    else if (isFraud) sz = 3.2
    else if (deg >= 6) sz = 3.0
    else if (deg >= 3) sz = 2.4
    else if (deg >= 1) sz = 2.0
    else sz = 1.6

    // Color classification: status > fraud > degree-based safe colors
    let color: string
    if (STATUS_NODE_COLORS[n.data.status]) color = STATUS_NODE_COLORS[n.data.status]
    else if (dom > 0) color = FRAUD_TYPE_COLORS[dom] ?? '#c0392b'
    else if (deg >= 6) color = SAFE_DEGREE_COLORS.hub
    else if (deg >= 3) color = SAFE_DEGREE_COLORS.active
    else if (deg >= 1) color = SAFE_DEGREE_COLORS.normal
    else color = SAFE_DEGREE_COLORS.leaf

    // Compute unique channels this node participates in
    const nodeChannels = new Set<number>()
    for (const e of nodeEdges) {
      nodeChannels.add(Number(e.data.channel ?? 0))
    }

    const pos = spherePositions[fgNodes.length] ?? { x: 0, y: 0, z: 0 }
    fgNodes.push({
      id: n.data.id,
      color,
      val: sz,
      status: n.data.status,
      fraudRatio,
      dominantFraud: dom,
      degree: deg,
      borderColor: STATUS_BORDER_COLORS[n.data.status] ?? '#384858',
      channels: [...nodeChannels],
      x: pos.x,
      y: pos.y,
      z: pos.z,
    } as FGNode)
  }

  const fgLinks: FGLink[] = []
  const addedEdges = new Set<string>()
  for (const e of edges) {
    if (!topIds.has(e.data.source) || !topIds.has(e.data.target)) continue
    if (e.data.source === e.data.target) continue
    const key = `${e.data.source}->${e.data.target}`
    if (addedEdges.has(key)) continue
    addedEdges.add(key)
    const fl = e.data.fraud_label ?? 0
    const ch = Number(e.data.channel ?? 0)
    const amt = e.data.amount_paisa ?? 0
    // Amount-proportional width (log-scaled) — large fund flows are visually prominent
    const logW = amt > 0 ? Math.log10(amt / 100 + 1) / 5 : 0.15
    const baseWidth = Math.max(0.15, Math.min(2.5, logW * 2))
    // Structuring indicator — amounts near ₹10L threshold get warning treatment
    const isStructuring = amt >= STRUCTURING_LOWER && amt < STRUCTURING_THRESHOLD
    const linkColor = isStructuring
      ? '#f59e0b'
      : fl > 0 ? (FRAUD_TYPE_COLORS[fl] ?? '#ff4757') : (CHANNEL_COLORS[ch] ?? '#3b82f680')
    fgLinks.push({
      source: e.data.source,
      target: e.data.target,
      color: linkColor,
      width: fl > 0 ? Math.max(baseWidth, 0.7) : baseWidth,
      fraud_label: fl,
      amount: amt,
      channel: String(e.data.channel ?? '0'),
    })
  }

  return { fgNodes, fgLinks }
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
function getLinkSourceId(link: any): string {
  return typeof link.source === 'object' ? link.source.id : link.source
}
// eslint-disable-next-line @typescript-eslint/no-explicit-any
function getLinkTargetId(link: any): string {
  return typeof link.target === 'object' ? link.target.id : link.target
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
      const last = _activityHistory.length - 1
      ctx.lineTo(last * step, H); ctx.lineTo(0, H); ctx.closePath()
      const grad = ctx.createLinearGradient(0, 0, 0, H)
      grad.addColorStop(0, 'rgba(59,130,246,0.3)'); grad.addColorStop(1, 'rgba(59,130,246,0)')
      ctx.fillStyle = grad; ctx.fill()
      const lx = last * step, ly = H - (_activityHistory[last] / max) * (H - 2)
      ctx.beginPath(); ctx.arc(lx, ly, 2, 0, Math.PI * 2); ctx.fillStyle = '#60a5fa'; ctx.fill()
    }, 2000)
    return () => clearInterval(iv)
  }, [])

  return <canvas ref={canvasRef} style={{ width: 80, height: 20 }} className="inline-block ml-2 align-middle" />
}

// ============================================================================
// In-Graph Node Detail Panel -- centrality metrics, flow, fraud breakdown
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

  const fraudBreakdown = new Map<number, number>()
  for (const e of fraudEdges) {
    const fl = e.data.fraud_label ?? 0
    fraudBreakdown.set(fl, (fraudBreakdown.get(fl) ?? 0) + 1)
  }

  const inflow = nodeEdges.filter(e => e.data.target === nodeId)
  const outflow = nodeEdges.filter(e => e.data.source === nodeId)
  const inflowAmt = inflow.reduce((s, e) => s + (e.data.amount_paisa ?? 0), 0)
  const outflowAmt = outflow.reduce((s, e) => s + (e.data.amount_paisa ?? 0), 0)
  const maxFlow = Math.max(inflowAmt, outflowAmt, 1)

  const pr = pageRanks.get(nodeId) ?? 0
  const bc = betweenness.get(nodeId) ?? 0

  const channels = new Map<number, number>()
  for (const e of nodeEdges) {
    const ch = Number(e.data.channel ?? 0)
    channels.set(ch, (channels.get(ch) ?? 0) + 1)
  }

  const fmtVol = (v: number) => v >= 1e7 ? `₹${(v / 1e7).toFixed(1)}Cr` : v >= 1e5 ? `₹${(v / 1e5).toFixed(1)}L` : `₹${(v / 100).toLocaleString()}`

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
            <div className="h-full rounded-full" style={{ width: `${fraudRatio * 100}%`, background: `linear-gradient(90deg, ${riskColor}88, ${riskColor})` }} />
          </div>
        </div>
        {/* Centrality */}
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
            <div className="text-emerald-400 font-mono font-semibold">{fmtVol(totalAmount)}</div>
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
              <span className="text-slate-400 text-[9px] w-16 text-right">{fmtVol(inflowAmt)}</span>
            </div>
            <div className="flex items-center gap-2">
              <span className="text-red-400 w-10">OUT</span>
              <div className="flex-1 h-1.5 bg-slate-700 rounded-full overflow-hidden">
                <div className="h-full bg-red-500/70 rounded-full" style={{ width: `${(outflowAmt / maxFlow) * 100}%` }} />
              </div>
              <span className="text-slate-400 text-[9px] w-16 text-right">{fmtVol(outflowAmt)}</span>
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
              {[...channels.entries()].sort((a, b) => b[1] - a[1]).map(([ch, count]) => (
                <span key={ch} className="px-1.5 py-0.5 rounded text-[9px] flex items-center gap-1" style={{ background: (CHANNEL_COLORS[ch] ?? '#94a3b8') + '22', color: CHANNEL_COLORS[ch] ?? '#94a3b8', border: `1px solid ${(CHANNEL_COLORS[ch] ?? '#94a3b8')}33` }}>
                  <span className="w-1.5 h-1.5 rounded-full inline-block" style={{ background: CHANNEL_COLORS[ch] ?? '#94a3b8' }} />
                  {CHANNEL_LABELS[ch] ?? `Ch ${ch}`} ({count})
                </span>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  )
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
// Graph 3D Controls -- toolbar with filters, zoom, VFX, search
// ============================================================================
function Graph3DControls({
  fgRef, filter, setFilter, searchTerm, setSearchTerm,
  onRestart, onCycles, onMules, onLayering, graphData, onFlyToNode,
  activeChannels, setActiveChannels, activeFraudTypes, setActiveFraudTypes,
  heatmapMode, setHeatmapMode,
  particlesEnabled, setParticlesEnabled,
  autoRotate, setAutoRotate,
  fogEnabled, setFogEnabled,
  glowEnabled, setGlowEnabled,
  labelsEnabled, setLabelsEnabled,
  starsEnabled, setStarsEnabled,
}: {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  fgRef: React.RefObject<any>
  filter: GraphFilter; setFilter: (f: GraphFilter) => void
  searchTerm: string; setSearchTerm: (s: string) => void
  onRestart: () => void; onCycles: () => void; onMules: () => void; onLayering: () => void
  graphData: { nodes: FGNode[]; links: FGLink[] }
  onFlyToNode: (nodeId: string) => void
  activeChannels: Set<number>; setActiveChannels: (s: Set<number>) => void
  activeFraudTypes: Set<number>; setActiveFraudTypes: (s: Set<number>) => void
  heatmapMode: boolean; setHeatmapMode: (v: boolean) => void
  particlesEnabled: boolean; setParticlesEnabled: (v: boolean) => void
  autoRotate: boolean; setAutoRotate: (v: boolean) => void
  fogEnabled: boolean; setFogEnabled: (v: boolean) => void
  glowEnabled: boolean; setGlowEnabled: (v: boolean) => void
  labelsEnabled: boolean; setLabelsEnabled: (v: boolean) => void
  starsEnabled: boolean; setStarsEnabled: (v: boolean) => void
}) {
  const [showVfx, setShowVfx] = useState(false)
  const [showSearch, setShowSearch] = useState(false)
  const [showFilters, setShowFilters] = useState(false)

  const zoom = (dir: 1 | -1) => {
    if (!fgRef.current) return
    const cam = fgRef.current.camera()
    const pos = cam.position
    const factor = dir === 1 ? 0.7 : 1.4
    fgRef.current.cameraPosition(
      { x: pos.x * factor, y: pos.y * factor, z: pos.z * factor },
      undefined,
      300,
    )
  }

  const fit = () => fgRef.current?.zoomToFit(400)

  const doCenterFraud = () => {
    if (!fgRef.current) return
    const data = fgRef.current.graphData()
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const fraudNode = data.nodes.find((n: any) => n.status === 'frozen' || (n.fraudRatio ?? 0) > 0.5)
    if (fraudNode) {
      const dist = 120
      const distRatio = 1 + dist / Math.hypot(fraudNode.x || 1, fraudNode.y || 1, fraudNode.z || 1)
      fgRef.current.cameraPosition(
        { x: (fraudNode.x || 0) * distRatio, y: (fraudNode.y || 0) * distRatio, z: (fraudNode.z || 0) * distRatio },
        fraudNode,
        1000,
      )
    }
  }

  const doExport = () => {
    if (!fgRef.current) return
    const renderer = fgRef.current.renderer()
    fgRef.current.renderer().render(fgRef.current.scene(), fgRef.current.camera())
    const link = document.createElement('a')
    link.download = `payflow-graph-3d-${Date.now()}.png`
    link.href = renderer.domElement.toDataURL('image/png')
    link.click()
  }

  // Keyboard shortcuts
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (e.target instanceof HTMLInputElement) return
      if (e.key === 'p' || e.key === 'P') setParticlesEnabled(!particlesEnabled)
      if (e.key === 'r' || e.key === 'R') setAutoRotate(!autoRotate)
      if (e.key === 'g' || e.key === 'G') setGlowEnabled(!glowEnabled)
      if (e.key === 'l' || e.key === 'L') setLabelsEnabled(!labelsEnabled)
      if (e.key === 'f' || e.key === 'F') setFogEnabled(!fogEnabled)
    }
    window.addEventListener('keydown', handler)
    return () => window.removeEventListener('keydown', handler)
  }, [particlesEnabled, autoRotate, glowEnabled, labelsEnabled, fogEnabled, setParticlesEnabled, setAutoRotate, setGlowEnabled, setLabelsEnabled, setFogEnabled])

  const filterButtons: { key: GraphFilter; icon: React.ReactNode; label: string }[] = [
    { key: 'none', icon: <Network size={13} />, label: 'All' },
    { key: 'fraud-only', icon: <AlertTriangle size={13} />, label: 'Fraud' },
    { key: 'high-risk', icon: <Filter size={13} />, label: 'High Risk' },
    { key: 'threat-path', icon: <Radio size={13} />, label: 'Threats' },
    { key: 'cycles', icon: <Repeat size={13} />, label: 'Cycles' },
    { key: 'mule', icon: <Users size={13} />, label: 'Mules' },
    { key: 'layering', icon: <Layers size={13} />, label: 'Layering' },
    { key: 'community', icon: <Tag size={13} />, label: 'Community' },
  ]

  return (
    <div className="absolute top-3 left-3 z-20 flex flex-col gap-1.5">
      {/* Filter row */}
      <div className="flex items-center gap-1 bg-slate-900/90 border border-slate-700/50 rounded-lg px-1.5 py-1 shadow-lg backdrop-blur-sm">
        {filterButtons.map(fb => (
          <button key={fb.key} onClick={() => { if (fb.key === 'cycles') onCycles(); if (fb.key === 'mule') onMules(); if (fb.key === 'layering') onLayering(); setFilter(fb.key) }}
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
        <button onClick={doCenterFraud} className="p-1.5 text-slate-400 hover:text-slate-200 hover:bg-slate-700/50 rounded transition-colors" title="Center on fraud"><Crosshair size={13} /></button>
        <button onClick={onRestart} className="p-1.5 text-slate-400 hover:text-slate-200 hover:bg-slate-700/50 rounded transition-colors" title="Restart layout"><RotateCcw size={13} /></button>
        <button onClick={doExport} className="p-1.5 text-slate-400 hover:text-slate-200 hover:bg-slate-700/50 rounded transition-colors" title="Export PNG"><Download size={13} /></button>
        <div className="w-px h-4 bg-slate-700/50 mx-0.5" />
        <button onClick={() => setShowSearch(!showSearch)} className={`p-1.5 rounded transition-colors ${showSearch ? 'text-blue-400 bg-blue-500/10' : 'text-slate-400 hover:text-slate-200 hover:bg-slate-700/50'}`} title="Search"><Search size={13} /></button>
        <button onClick={() => setShowVfx(!showVfx)} className={`p-1.5 rounded transition-colors ${showVfx ? 'text-blue-400 bg-blue-500/10' : 'text-slate-400 hover:text-slate-200 hover:bg-slate-700/50'}`} title="3D Effects"><Zap size={13} /></button>
        <button onClick={() => setShowFilters(!showFilters)} className={`p-1.5 rounded transition-colors ${showFilters ? 'text-amber-400 bg-amber-500/10' : 'text-slate-400 hover:text-slate-200 hover:bg-slate-700/50'}`} title="Advanced Filters"><SlidersHorizontal size={13} /></button>
        <button onClick={() => setHeatmapMode(!heatmapMode)} className={`p-1.5 rounded transition-colors ${heatmapMode ? 'text-orange-400 bg-orange-500/10' : 'text-slate-400 hover:text-slate-200 hover:bg-slate-700/50'}`} title="Heatmap Mode"><Flame size={13} /></button>
      </div>
      {/* Search input with fly-to results */}
      {showSearch && (
        <div className="bg-slate-900/90 border border-slate-700/50 rounded-lg px-2 py-1.5 shadow-lg backdrop-blur-sm min-w-[220px]">
          <div className="flex items-center gap-1.5 mb-1">
            <Search size={11} className="text-slate-500" />
            <input type="text" value={searchTerm} onChange={e => setSearchTerm(e.target.value)} placeholder="Search nodes by ID..."
              className="flex-1 bg-transparent border-none outline-none text-xs text-slate-200 placeholder:text-slate-500" autoFocus />
            {searchTerm && <button onClick={() => setSearchTerm('')} className="text-slate-500 hover:text-slate-300 text-xs">&times;</button>}
          </div>
          {searchTerm.length >= 2 && (() => {
            const sl = searchTerm.toLowerCase()
            const matches = graphData.nodes.filter(n => n.id.toLowerCase().includes(sl)).slice(0, 8)
            if (matches.length === 0) return <div className="text-[9px] text-slate-500 py-1">No matches found</div>
            return (
              <div className="border-t border-slate-700/40 pt-1 max-h-[180px] overflow-y-auto space-y-0.5">
                {matches.map(n => {
                  const riskLevel = n.fraudRatio > 0.6 ? 'CRIT' : n.fraudRatio > 0.3 ? 'HIGH' : n.fraudRatio > 0.1 ? 'MED' : 'LOW'
                  const riskColor = ({ CRIT: '#ef4444', HIGH: '#f97316', MED: '#eab308', LOW: '#22c55e' } as Record<string,string>)[riskLevel]
                  return (
                    <button key={n.id} onClick={() => { onFlyToNode(n.id); setSearchTerm('') }}
                      className="w-full flex items-center gap-2 px-1.5 py-1 rounded hover:bg-slate-700/40 transition-colors text-left group">
                      <div className="w-2.5 h-2.5 rounded-full shrink-0" style={{ background: n.color }} />
                      <span className="text-[10px] font-mono text-slate-300 flex-1 truncate group-hover:text-white">{n.id}</span>
                      <span className="text-[8px] font-semibold px-1 py-0.5 rounded" style={{ background: riskColor + '22', color: riskColor }}>{riskLevel}</span>
                      <Crosshair size={9} className="text-slate-600 group-hover:text-cyan-400 shrink-0" />
                    </button>
                  )
                })}
                <div className="text-[8px] text-slate-600 pt-0.5">Click to fly to node</div>
              </div>
            )
          })()}
        </div>
      )}
      {/* VFX panel */}
      {showVfx && (
        <div className="bg-slate-900/90 border border-slate-700/50 rounded-lg px-2 py-2 shadow-lg backdrop-blur-sm min-w-[180px]">
          <div className="text-[9px] text-slate-500 font-semibold uppercase tracking-wider mb-1.5 flex items-center gap-1"><Zap size={10} /> 3D Effects</div>
          <div className="space-y-0.5">
            <VfxToggle label="Particle Flow" enabled={particlesEnabled} onChange={() => setParticlesEnabled(!particlesEnabled)} shortcut="P" />
            <VfxToggle label="Auto Rotate" enabled={autoRotate} onChange={() => setAutoRotate(!autoRotate)} shortcut="R" />
            <VfxToggle label="Node Glow" enabled={glowEnabled} onChange={() => setGlowEnabled(!glowEnabled)} shortcut="G" />
            <VfxToggle label="Node Labels" enabled={labelsEnabled} onChange={() => setLabelsEnabled(!labelsEnabled)} shortcut="L" />
            <VfxToggle label="Fog Depth" enabled={fogEnabled} onChange={() => setFogEnabled(!fogEnabled)} shortcut="F" />
            <VfxToggle label="Star Field" enabled={starsEnabled} onChange={() => setStarsEnabled(!starsEnabled)} />
          </div>
        </div>
      )}
      {/* Advanced Filter panel */}
      {showFilters && (
        <div className="bg-slate-900/90 border border-slate-700/50 rounded-lg px-2 py-2 shadow-lg backdrop-blur-sm min-w-[220px] max-h-[400px] overflow-y-auto">
          <div className="text-[9px] text-slate-500 font-semibold uppercase tracking-wider mb-1.5 flex items-center gap-1"><SlidersHorizontal size={10} /> Advanced Filters</div>
          {/* Channel filters */}
          <div className="mb-2">
            <div className="flex items-center justify-between mb-1">
              <span className="text-[9px] text-slate-400 font-medium">Channels</span>
              <div className="flex gap-1">
                <button onClick={() => setActiveChannels(new Set([0,1,2,3,4,5,6,7,8,9]))} className="text-[8px] text-cyan-400 hover:underline">All</button>
                <button onClick={() => setActiveChannels(new Set())} className="text-[8px] text-slate-500 hover:underline">None</button>
              </div>
            </div>
            <div className="grid grid-cols-2 gap-x-2 gap-y-0.5">
              {CHANNEL_LABELS.map((label, i) => {
                const on = activeChannels.has(i)
                return (
                  <label key={i} className="flex items-center gap-1.5 cursor-pointer group" title={label}>
                    <input type="checkbox" checked={on} onChange={() => {
                      const next = new Set(activeChannels)
                      on ? next.delete(i) : next.add(i)
                      setActiveChannels(next)
                    }} className="hidden" />
                    <div className={`w-2.5 h-2.5 rounded-sm border transition-all ${on ? 'border-transparent' : 'border-slate-600 bg-slate-800'}`}
                      style={on ? { background: CHANNEL_COLORS[i] } : {}} />
                    <span className={`text-[9px] truncate transition-colors ${on ? 'text-slate-300' : 'text-slate-600'}`}>{label}</span>
                  </label>
                )
              })}
            </div>
          </div>
          {/* Fraud type filters */}
          <div>
            <div className="flex items-center justify-between mb-1">
              <span className="text-[9px] text-slate-400 font-medium">Fraud Types</span>
              <div className="flex gap-1">
                <button onClick={() => setActiveFraudTypes(new Set([0,1,2,3,4,5,6,7,8]))} className="text-[8px] text-cyan-400 hover:underline">All</button>
                <button onClick={() => setActiveFraudTypes(new Set())} className="text-[8px] text-slate-500 hover:underline">None</button>
              </div>
            </div>
            <div className="space-y-0.5">
              {FRAUD_PATTERN_LABELS.map((label, i) => {
                const on = activeFraudTypes.has(i)
                const color = FRAUD_TYPE_COLORS[i] ?? '#888'
                return (
                  <label key={i} className="flex items-center gap-1.5 cursor-pointer group" title={label}>
                    <input type="checkbox" checked={on} onChange={() => {
                      const next = new Set(activeFraudTypes)
                      on ? next.delete(i) : next.add(i)
                      setActiveFraudTypes(next)
                    }} className="hidden" />
                    <div className={`w-2.5 h-2.5 rounded-sm border transition-all ${on ? 'border-transparent' : 'border-slate-600 bg-slate-800'}`}
                      style={on ? { background: color } : {}} />
                    <span className={`text-[9px] truncate transition-colors ${on ? 'text-slate-300' : 'text-slate-600'}`}>{label}</span>
                  </label>
                )
              })}
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

// ============================================================================
// Graph Legend — multi-section, rich color key
// ============================================================================
function GraphLegend({ edges }: { edges: CytoEdge[] }) {
  const [collapsed, setCollapsed] = useState(true)
  const [section, setSection] = useState<'fraud' | 'channel' | 'status' | 'size'>('fraud')

  // Compute channel distribution for the legend
  const channelCounts = useMemo(() => {
    const counts = new Map<number, number>()
    for (const e of edges) {
      const ch = Number(e.data.channel ?? 0)
      counts.set(ch, (counts.get(ch) ?? 0) + 1)
    }
    return counts
  }, [edges])

  const fraudCounts = useMemo(() => {
    const counts = new Map<number, number>()
    for (const e of edges) {
      const fl = e.data.fraud_label ?? 0
      if (fl > 0) counts.set(fl, (counts.get(fl) ?? 0) + 1)
    }
    return counts
  }, [edges])

  const tabs = [
    { key: 'fraud' as const, label: 'Fraud' },
    { key: 'channel' as const, label: 'Channels' },
    { key: 'status' as const, label: 'Status' },
    { key: 'size' as const, label: 'Guide' },
  ]

  return (
    <div className="absolute bottom-3 right-3 z-20">
      <button onClick={() => setCollapsed(!collapsed)}
        className="bg-slate-900/90 border border-slate-700/50 rounded-lg px-2.5 py-1.5 text-[10px] text-slate-400 hover:text-slate-200 shadow-lg backdrop-blur-sm transition-colors flex items-center gap-1">
        <Network size={11} /> Legend {collapsed ? '+' : '−'}
      </button>
      {!collapsed && (
        <div className="mt-1 bg-slate-900/95 border border-slate-700/50 rounded-xl px-3 py-2.5 shadow-xl backdrop-blur-sm min-w-[200px] max-w-[240px]">
          {/* Tabs */}
          <div className="flex gap-0.5 mb-2 border-b border-slate-700/40 pb-1.5">
            {tabs.map(t => (
              <button key={t.key} onClick={() => setSection(t.key)}
                className={`px-2 py-0.5 rounded text-[9px] font-medium transition-colors ${
                  section === t.key ? 'bg-blue-500/20 text-blue-400' : 'text-slate-500 hover:text-slate-300'
                }`}>
                {t.label}
              </button>
            ))}
          </div>

          {/* Fraud Types */}
          {section === 'fraud' && (
            <div className="space-y-1">
              {Object.entries(FRAUD_PATTERN_LABELS).filter(([k]) => Number(k) > 0).map(([k, label]) => {
                const count = fraudCounts.get(Number(k)) ?? 0
                return (
                  <div key={k} className="flex items-center gap-2 text-[10px]">
                    <div className="w-2.5 h-2.5 rounded-full shrink-0" style={{ background: FRAUD_TYPE_COLORS[Number(k)] }} />
                    <span className="text-slate-300 flex-1 truncate">{label}</span>
                    {count > 0 && <span className="text-slate-500 font-mono text-[9px]">{count}</span>}
                  </div>
                )
              })}
              <div className="flex items-center gap-2 text-[10px] mt-1 pt-1 border-t border-slate-700/30">
                <div className="w-2.5 h-2.5 rounded-full shrink-0" style={{ background: NODE_COLOR_SAFE }} />
                <span className="text-slate-400">Clean / No Fraud</span>
              </div>
            </div>
          )}

          {/* Channels */}
          {section === 'channel' && (
            <div className="space-y-1">
              {Object.entries(CHANNEL_LABELS).map(([k, label]) => {
                const count = channelCounts.get(Number(k)) ?? 0
                if (count === 0) return null
                const total = edges.length || 1
                const pct = ((count / total) * 100).toFixed(0)
                return (
                  <div key={k} className="flex items-center gap-2 text-[10px]">
                    <div className="w-5 h-[3px] rounded-full shrink-0" style={{ background: CHANNEL_COLORS[Number(k)] }} />
                    <span className="text-slate-300 flex-1 truncate">{label}</span>
                    <span className="text-slate-500 font-mono text-[9px]">{pct}%</span>
                  </div>
                )
              })}
            </div>
          )}

          {/* Status */}
          {section === 'status' && (
            <div className="space-y-1.5">
              {Object.entries(STATUS_BORDER_COLORS).map(([status, color]) => (
                <div key={status} className="flex items-center gap-2 text-[10px]">
                  <div className="w-3 h-3 rounded-full border-2 shrink-0" style={{ borderColor: color, background: status === 'frozen' ? color + '22' : 'transparent' }} />
                  <span className="text-slate-300 capitalize flex-1">{status}</span>
                  <span className="text-slate-600 text-[8px]">{status === 'frozen' ? 'ring + fill' : status === 'suspicious' ? 'ring' : '—'}</span>
                </div>
              ))}
              <div className="mt-1 pt-1 border-t border-slate-700/30 text-[9px] text-slate-500">
                Ring border = node status indicator
              </div>
            </div>
          )}

          {/* Size Guide */}
          {section === 'size' && (
            <div className="space-y-1.5">
              <div className="text-[9px] text-slate-500 mb-1">Node Size = Importance</div>
              {[
                { sz: 10, label: 'Frozen / Suspicious', desc: 'Highest risk' },
                { sz: 8, label: 'High-degree Fraud', desc: '≥4 connections' },
                { sz: 6, label: 'Fraud Node', desc: 'Any fraud type' },
                { sz: 4, label: 'Hub', desc: '≥3 connections' },
                { sz: 3, label: 'Normal', desc: 'Clean node' },
              ].map(({ sz, label, desc }) => (
                <div key={label} className="flex items-center gap-2 text-[10px]">
                  <div className="w-5 flex items-center justify-center shrink-0">
                    <div className="rounded-full bg-blue-400/50" style={{ width: sz, height: sz }} />
                  </div>
                  <span className="text-slate-300 flex-1">{label}</span>
                  <span className="text-slate-600 text-[8px]">{desc}</span>
                </div>
              ))}
              <div className="mt-1.5 pt-1 border-t border-slate-700/30 text-[9px] text-slate-500">
                Link arrows = fraud direction<br />
                Particles = transaction flow
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  )
}

// ============================================================================
// Network Risk Score — animated SVG gauge showing composite risk
// ============================================================================
function NetworkRiskGauge({
  graph, edges, clusteringCoeff,
}: {
  graph: Graph; edges: CytoEdge[]; clusteringCoeff: number
}) {
  const nodeCount = graph.order || 1
  const edgeCount = graph.size || 1
  const fraudEdgeCount = edges.filter(e => (e.data.fraud_label ?? 0) > 0).length
  const fraudRatio = edgeCount > 0 ? fraudEdgeCount / edgeCount : 0

  let frozen = 0, suspicious = 0
  graph.forEachNode((_: string, a: Record<string, unknown>) => {
    if (a.status === 'frozen') frozen++
    else if (a.status === 'suspicious') suspicious++
  })
  const statusRatio = (frozen * 1.0 + suspicious * 0.5) / nodeCount

  // Composite risk score (0-100): 40% fraud ratio, 30% status ratio, 20% clustering, 10% density
  const density = (2 * edgeCount) / (nodeCount * (nodeCount - 1) || 1)
  const riskScore = Math.min(100, Math.round(
    fraudRatio * 40 * 2.5 +      // fraud ratio weight (multiply to scale ~0-1 to meaningful range)
    statusRatio * 30 * 2 +        // status weight
    (1 - clusteringCoeff) * 20 +  // low clustering = more distributed = higher risk 
    density * 10 * 5              // density weight
  ))

  const riskLabel = riskScore >= 75 ? 'CRITICAL' : riskScore >= 50 ? 'HIGH' : riskScore >= 25 ? 'ELEVATED' : 'LOW'
  const riskColor = riskScore >= 75 ? '#ef4444' : riskScore >= 50 ? '#f97316' : riskScore >= 25 ? '#eab308' : '#22c55e'

  // SVG arc parameters
  const radius = 38
  const circumference = Math.PI * radius // half circle
  const offset = circumference * (1 - riskScore / 100)

  return (
    <div className="absolute bottom-3 left-3 z-20">
      <div className="bg-slate-900/95 border border-slate-700/50 rounded-xl px-3 py-2.5 shadow-xl backdrop-blur-sm w-[130px]">
        <div className="text-[9px] text-slate-500 font-semibold uppercase tracking-wider mb-1 text-center">Network Risk</div>
        <div className="relative flex justify-center">
          <svg width="100" height="58" viewBox="0 0 100 58">
            {/* Background arc */}
            <path d="M 10 52 A 38 38 0 0 1 90 52" fill="none" stroke="#334155" strokeWidth="5" strokeLinecap="round" />
            {/* Risk arc */}
            <path d="M 10 52 A 38 38 0 0 1 90 52" fill="none" stroke={riskColor} strokeWidth="5" strokeLinecap="round"
              strokeDasharray={`${circumference}`} strokeDashoffset={offset}
              style={{ transition: 'stroke-dashoffset 1s ease-out, stroke 0.5s ease' }} />
            {/* Glow arc */}
            <path d="M 10 52 A 38 38 0 0 1 90 52" fill="none" stroke={riskColor} strokeWidth="8" strokeLinecap="round"
              strokeDasharray={`${circumference}`} strokeDashoffset={offset} opacity="0.15"
              style={{ transition: 'stroke-dashoffset 1s ease-out' }} />
            {/* Score text */}
            <text x="50" y="42" textAnchor="middle" fill={riskColor} fontSize="18" fontWeight="bold" fontFamily="monospace">{riskScore}</text>
            <text x="50" y="54" textAnchor="middle" fill="#64748b" fontSize="7" fontFamily="system-ui">/100</text>
          </svg>
        </div>
        <div className="text-center mt-0.5">
          <span className="text-[10px] font-bold px-2 py-0.5 rounded-full" style={{ background: riskColor + '22', color: riskColor, border: `1px solid ${riskColor}33` }}>
            {riskLabel}
          </span>
        </div>
        <div className="mt-1.5 grid grid-cols-3 gap-1 text-center">
          <div>
            <div className="text-[8px] text-slate-500">Fraud</div>
            <div className="text-[9px] font-mono text-red-400">{(fraudRatio * 100).toFixed(0)}%</div>
          </div>
          <div>
            <div className="text-[8px] text-slate-500">Frozen</div>
            <div className="text-[9px] font-mono text-yellow-400">{frozen}</div>
          </div>
          <div>
            <div className="text-[8px] text-slate-500">Susp</div>
            <div className="text-[9px] font-mono text-orange-400">{suspicious}</div>
          </div>
        </div>
      </div>
    </div>
  )
}

// ============================================================================
// Graph Stats Overlay — enriched metrics panel
// ============================================================================
function GraphStatsOverlay({
  graph, edges, clusteringCoeff, pageRanks, betweenness,
  muleCount, layeringCount, structuringCount,
}: {
  graph: Graph; edges: CytoEdge[]; clusteringCoeff: number
  pageRanks: Map<string, number>; betweenness: Map<string, number>
  muleCount: number; layeringCount: number; structuringCount: number
}) {
  const nodeCount = graph.order
  const edgeCount = graph.size
  const fraudEdgeCount = edges.filter(e => (e.data.fraud_label ?? 0) > 0).length
  const fraudPercent = edgeCount > 0 ? ((fraudEdgeCount / edgeCount) * 100).toFixed(1) : '0'

  let frozen = 0, suspicious = 0, paused = 0, normal = 0
  graph.forEachNode((_: string, a: Record<string, unknown>) => {
    if (a.status === 'frozen') frozen++
    else if (a.status === 'suspicious') suspicious++
    else if (a.status === 'paused') paused++
    else normal++
  })
  const total = frozen + suspicious + paused + normal || 1

  // Fraud pattern counts
  const patternCounts = new Map<number, number>()
  edges.forEach(e => { const fl = e.data.fraud_label ?? 0; if (fl > 0) patternCounts.set(fl, (patternCounts.get(fl) ?? 0) + 1) })
  const sortedPatterns = [...patternCounts.entries()].sort((a, b) => b[1] - a[1])

  // Channel distribution
  const channelCounts = new Map<number, number>()
  edges.forEach(e => { const ch = Number(e.data.channel ?? 0); channelCounts.set(ch, (channelCounts.get(ch) ?? 0) + 1) })
  const sortedChannels = [...channelCounts.entries()].sort((a, b) => b[1] - a[1]).slice(0, 5)

  // Volume metrics
  const totalVolume = edges.reduce((s, e) => s + (e.data.amount_paisa ?? 0), 0)
  const fraudVolume = edges.filter(e => (e.data.fraud_label ?? 0) > 0).reduce((s, e) => s + (e.data.amount_paisa ?? 0), 0)
  const avgTxn = edgeCount > 0 ? totalVolume / edgeCount : 0

  // Avg degree
  const avgDegree = nodeCount > 0 ? ((edgeCount * 2) / nodeCount).toFixed(1) : '0'

  let topPR = '', topPRVal = 0
  pageRanks.forEach((v, k) => { if (v > topPRVal) { topPR = k; topPRVal = v } })
  let topBC = '', topBCVal = 0
  betweenness.forEach((v, k) => { if (v > topBCVal) { topBC = k; topBCVal = v } })

  const fmtVol = (v: number) => v >= 1e7 ? `₹${(v / 1e7).toFixed(1)}Cr` : v >= 1e5 ? `₹${(v / 1e5).toFixed(1)}L` : `₹${(v / 100).toFixed(0)}`

  return (
    <div className="absolute top-3 right-3 z-20 bg-slate-900/90 border border-slate-700/50 rounded-xl shadow-lg backdrop-blur-sm w-60">
      <div className="px-3 py-2 border-b border-slate-700/50 flex items-center justify-between">
        <span className="text-[10px] text-slate-400 font-semibold uppercase tracking-wider flex items-center gap-1">
          <TrendingUp size={10} /> Network Stats
        </span>
        <ActivitySparkline />
      </div>
      <div className="px-3 py-2 space-y-2 text-[10px]">
        {/* Core metrics grid */}
        <div className="grid grid-cols-3 gap-x-2 gap-y-1">
          <div className="flex flex-col items-center bg-slate-800/40 rounded px-1 py-1">
            <span className="text-slate-200 font-mono font-semibold">{nodeCount}</span>
            <span className="text-[8px] text-slate-500">Nodes</span>
          </div>
          <div className="flex flex-col items-center bg-slate-800/40 rounded px-1 py-1">
            <span className="text-slate-200 font-mono font-semibold">{edgeCount}</span>
            <span className="text-[8px] text-slate-500">Edges</span>
          </div>
          <div className="flex flex-col items-center bg-slate-800/40 rounded px-1 py-1">
            <span className="text-red-400 font-mono font-semibold">{fraudPercent}%</span>
            <span className="text-[8px] text-slate-500">Fraud</span>
          </div>
        </div>

        {/* Volume & degree */}
        <div className="grid grid-cols-2 gap-x-3 gap-y-1">
          <div className="flex justify-between"><span className="text-slate-500">Volume</span><span className="text-emerald-400 font-mono">{fmtVol(totalVolume)}</span></div>
          <div className="flex justify-between"><span className="text-slate-500">Fraud Vol</span><span className="text-red-400 font-mono">{fmtVol(fraudVolume)}</span></div>
          <div className="flex justify-between"><span className="text-slate-500">Avg Txn</span><span className="text-slate-300 font-mono">{fmtVol(avgTxn)}</span></div>
          <div className="flex justify-between"><span className="text-slate-500">Avg Deg</span><span className="text-cyan-400 font-mono">{avgDegree}</span></div>
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

        {/* Top fraud patterns with mini bars */}
        {sortedPatterns.length > 0 && (
          <div>
            <div className="text-[9px] text-slate-500 mb-1">Top Fraud Patterns</div>
            <div className="space-y-1">
              {sortedPatterns.slice(0, 4).map(([fl, count]) => {
                const pct = edgeCount > 0 ? (count / edgeCount) * 100 : 0
                return (
                  <div key={fl} className="flex items-center gap-1.5">
                    <div className="w-1.5 h-1.5 rounded-full shrink-0" style={{ background: FRAUD_TYPE_COLORS[fl] }} />
                    <span className="text-slate-400 flex-1 truncate text-[9px]">{FRAUD_PATTERN_LABELS[fl] ?? `T${fl}`}</span>
                    <div className="w-12 h-1 bg-slate-700 rounded-full overflow-hidden">
                      <div className="h-full rounded-full" style={{ width: `${Math.min(pct * 3, 100)}%`, background: FRAUD_TYPE_COLORS[fl] }} />
                    </div>
                    <span className="text-slate-500 font-mono text-[8px] w-5 text-right">{count}</span>
                  </div>
                )
              })}
            </div>
          </div>
        )}

        {/* Channel breakdown */}
        {sortedChannels.length > 0 && (
          <div>
            <div className="text-[9px] text-slate-500 mb-1">Channel Mix</div>
            <div className="w-full h-2 rounded-full overflow-hidden flex">
              {sortedChannels.map(([ch, count]) => (
                <div key={ch} style={{ width: `${(count / (edgeCount || 1)) * 100}%`, background: CHANNEL_COLORS[ch] }} className="h-full" title={CHANNEL_LABELS[ch]} />
              ))}
            </div>
            <div className="flex flex-wrap gap-x-2 mt-0.5">
              {sortedChannels.slice(0, 3).map(([ch]) => (
                <span key={ch} className="flex items-center gap-0.5 text-[8px] text-slate-500">
                  <span className="w-1 h-1 rounded-full inline-block" style={{ background: CHANNEL_COLORS[ch] }} />
                  {CHANNEL_LABELS[ch]}
                </span>
              ))}
            </div>
          </div>
        )}

        {/* AML Indicators */}
        {(muleCount > 0 || layeringCount > 0 || structuringCount > 0) && (
          <div>
            <div className="text-[9px] text-slate-500 mb-1 flex items-center gap-1"><Shield size={9} className="text-amber-400" /> AML Indicators</div>
            <div className="grid grid-cols-3 gap-x-2 gap-y-1">
              <div className="flex flex-col items-center bg-red-950/30 border border-red-800/20 rounded px-1 py-1">
                <span className="text-red-400 font-mono font-semibold">{muleCount}</span>
                <span className="text-[8px] text-slate-500">Mules</span>
              </div>
              <div className="flex flex-col items-center bg-purple-950/30 border border-purple-800/20 rounded px-1 py-1">
                <span className="text-purple-400 font-mono font-semibold">{layeringCount}</span>
                <span className="text-[8px] text-slate-500">Layering</span>
              </div>
              <div className="flex flex-col items-center bg-amber-950/30 border border-amber-800/20 rounded px-1 py-1">
                <span className="text-amber-400 font-mono font-semibold">{structuringCount}</span>
                <span className="text-[8px] text-slate-500">Structuring</span>
              </div>
            </div>
          </div>
        )}

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
          <circle cx="50" cy="50" r="45" fill="none" stroke="#334155" strokeWidth="0.5" />
          <circle cx="50" cy="50" r="30" fill="none" stroke="#334155" strokeWidth="0.5" />
          <circle cx="50" cy="50" r="15" fill="none" stroke="#334155" strokeWidth="0.5" />
          <line x1="50" y1="2" x2="50" y2="98" stroke="#334155" strokeWidth="0.5" />
          <line x1="2" y1="50" x2="98" y2="50" stroke="#334155" strokeWidth="0.5" />
          <line x1="50" y1="50" x2="50" y2="5" stroke="#22d3ee" strokeWidth="1.5" opacity="0.7" className="origin-center animate-[radar-spin_3s_linear_infinite]" style={{ transformOrigin: '50px 50px' }} />
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
// Main SigmaGraph component -- now powered by ForceGraph3D
// ============================================================================
export default function SigmaGraph() {
  // -- Data fetching --
  const graphNodes = useDashboardStore((s) => s.graphNodes)
  const graphEdges = useDashboardStore((s) => s.graphEdges)
  const setInitialTopology = useDashboardStore((s) => s.setInitialTopology)
  const { data: topology } = useTopology()
  const syncedRef = useRef(false)

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
  const [searchTerm, setSearchTerm] = useState('')
  const [hoveredNode, setHoveredNode] = useState<string | null>(null)
  const [selectedNode, setSelectedNode] = useState<string | null>(null)
  const [cycleNodes, setCycleNodes] = useState<Set<string>>(new Set())
  const [particlesEnabled, setParticlesEnabled] = useState(false)
  const [autoRotate, setAutoRotate] = useState(false)
  const [fogEnabled, setFogEnabled] = useState(true)
  const [glowEnabled, setGlowEnabled] = useState(true)
  const [labelsEnabled, setLabelsEnabled] = useState(true)
  const [starsEnabled, setStarsEnabled] = useState(true)
  // Advanced filters
  const [activeChannels, setActiveChannels] = useState<Set<number>>(new Set([0,1,2,3,4,5,6,7,8,9]))
  const [activeFraudTypes, setActiveFraudTypes] = useState<Set<number>>(new Set([0,1,2,3,4,5,6,7,8]))
  const [heatmapMode, setHeatmapMode] = useState(false)

  // -- AML Detection state --
  const [muleNodes, setMuleNodes] = useState<Set<string>>(new Set())
  const [layeringNodes, setLayeringNodes] = useState<Set<string>>(new Set())
  const [structuringEdgeKeys, setStructuringEdgeKeys] = useState<Set<string>>(new Set())

  // -- Analytics --
  const [pageRanks, setPageRanks] = useState<Map<string, number>>(new Map())
  const [betweennessMap, setBetweennessMap] = useState<Map<string, number>>(new Map())
  const [clusteringCoeff, setClusteringCoeff] = useState(0)
  const [threatLevels, setThreatLevels] = useState<Map<string, number>>(new Map())

  const graphRef = useRef<Graph>(new Graph())
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const fgRef = useRef<any>(null)
  const [version, setVersion] = useState(0)
  const setSelectedNodeUI = useUIStore(s => s.setSelectedNode)

  // -- Container sizing --
  const containerRef = useRef<HTMLDivElement>(null)
  const [dimensions, setDimensions] = useState({ width: 800, height: 600 })

  useEffect(() => {
    if (!containerRef.current) return
    // Measure after layout settles to get true container size
    const measure = () => {
      if (!containerRef.current) return
      const rect = containerRef.current.getBoundingClientRect()
      if (rect.width > 0 && rect.height > 0) setDimensions({ width: Math.floor(rect.width), height: Math.floor(rect.height) })
    }
    requestAnimationFrame(measure)
    // Also re-measure after a short delay for late layout shifts
    const t = setTimeout(measure, 200)
    const observer = new ResizeObserver(entries => {
      const { width, height } = entries[0].contentRect
      if (width > 0 && height > 0) setDimensions({ width: Math.floor(width), height: Math.floor(height) })
    })
    observer.observe(containerRef.current)
    return () => { observer.disconnect(); clearTimeout(t) }
  }, [])

  // -- Sync graphology for analytics (throttled — runs at most every 3s) --
  const lastSyncRef = useRef(0)
  const syncTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  useEffect(() => {
    if (nodes.length === 0 && edges.length === 0) return
    const doSync = () => {
      lastSyncRef.current = Date.now()
      syncGraphology(graphRef.current, nodes, edges)
      setVersion(v => v + 1)
      recordActivity(edges.length)
    }
    const elapsed = Date.now() - lastSyncRef.current
    if (elapsed >= 3000) {
      doSync()
    } else if (!syncTimerRef.current) {
      syncTimerRef.current = setTimeout(() => {
        syncTimerRef.current = null
        doSync()
      }, 3000 - elapsed)
    }
    return () => {
      if (syncTimerRef.current) { clearTimeout(syncTimerRef.current); syncTimerRef.current = null }
    }
  }, [nodes, edges])

  // -- Build ForceGraph3D data (throttled via version to avoid recalc on every SSE tick) --
  const graphData = useMemo(() => {
    void version // depend on throttled version counter instead of raw nodes/edges
    if (nodes.length === 0 && edges.length === 0) return { nodes: [] as FGNode[], links: [] as FGLink[] }
    const { fgNodes, fgLinks } = buildForceGraphData(nodes, edges)
    return { nodes: fgNodes, links: fgLinks }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [version])

  // -- Recompute analytics (debounced — 3s after last graph sync) --
  useEffect(() => {
    const g = graphRef.current
    if (g.order === 0) return
    const timer = setTimeout(() => {
      setPageRanks(computePageRank(g))
      setBetweennessMap(computeBetweenness(g))
      setClusteringCoeff(computeClusteringCoeff(g))
      setThreatLevels(computeThreatPaths(g))
      setMuleNodes(detectMuleAccounts(edges))
      setLayeringNodes(detectLayeringChains(edges))
      setStructuringEdgeKeys(detectStructuringEdges(edges))
    }, 3000)
    return () => clearTimeout(timer)
  }, [version])

  // -- Neighbor set for hover highlighting --
  const neighborSet = useMemo(() => {
    if (!hoveredNode) return new Set<string>()
    const set = new Set<string>()
    for (const link of graphData.links) {
      const src = getLinkSourceId(link)
      const tgt = getLinkTargetId(link)
      if (src === hoveredNode) set.add(tgt)
      if (tgt === hoveredNode) set.add(src)
    }
    return set
  }, [hoveredNode, graphData.links])

  // -- Scene setup: lighting & fog --
  const sceneInitialized = useRef(false)
  useEffect(() => {
    if (!fgRef.current || sceneInitialized.current) return
    sceneInitialized.current = true
    const scene = fgRef.current.scene()
    // Strong lighting so 3D spheres are clearly lit from all angles
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.7)
    scene.add(ambientLight)
    const hemiLight = new THREE.HemisphereLight(0xffffff, 0x444466, 1.0)
    scene.add(hemiLight)
    const ptLight = new THREE.PointLight(0xffffff, 0.8, 2000)
    ptLight.position.set(0, 300, 200)
    scene.add(ptLight)
    const ptLight2 = new THREE.PointLight(0x6688ff, 0.5, 2000)
    ptLight2.position.set(-200, -100, 150)
    scene.add(ptLight2)
    // Rim light for dramatic sphere silhouette
    const rimLight = new THREE.PointLight(0x22d3ee, 0.35, 2000)
    rimLight.position.set(150, -200, -150)
    scene.add(rimLight)
    // Subtle wireframe sphere shell — reinforces the sphere surface structure
    const wireGeo = new THREE.SphereGeometry(140, 32, 24)
    const wireMat = new THREE.MeshBasicMaterial({
      color: 0x1e3a5f,
      wireframe: true,
      transparent: true,
      opacity: 0.04,
    })
    scene.add(new THREE.Mesh(wireGeo, wireMat))
    // Inner glow sphere — faint solid sphere for depth perception
    const innerGeo = new THREE.SphereGeometry(137, 32, 24)
    const innerMat = new THREE.MeshBasicMaterial({
      color: 0x0f172a,
      transparent: true,
      opacity: 0.15,
      side: THREE.BackSide,
    })
    scene.add(new THREE.Mesh(innerGeo, innerMat))

    // Exponential fog — distant objects fade for depth perception
    scene.fog = new THREE.FogExp2(0x0f172a, 0.0018)

    // Star field backdrop — scattered points in a large outer sphere
    const starCount = 600
    const starPositions = new Float32Array(starCount * 3)
    const starSizes = new Float32Array(starCount)
    for (let i = 0; i < starCount; i++) {
      const theta = Math.random() * Math.PI * 2
      const phi = Math.acos(2 * Math.random() - 1)
      const r = 450 + Math.random() * 500
      starPositions[i * 3] = r * Math.sin(phi) * Math.cos(theta)
      starPositions[i * 3 + 1] = r * Math.sin(phi) * Math.sin(theta)
      starPositions[i * 3 + 2] = r * Math.cos(phi)
      starSizes[i] = 0.3 + Math.random() * 1.2
    }
    const starGeo = new THREE.BufferGeometry()
    starGeo.setAttribute('position', new THREE.BufferAttribute(starPositions, 3))
    starGeo.setAttribute('size', new THREE.BufferAttribute(starSizes, 1))
    const starMat = new THREE.PointsMaterial({
      color: 0x6b8db5,
      size: 0.6,
      transparent: true,
      opacity: 0.5,
      sizeAttenuation: true,
    })
    const stars = new THREE.Points(starGeo, starMat)
    stars.name = '__payflow_stars'
    scene.add(stars)

    // Latitude guide rings — equator and ±35° latitude lines
    const ringMaterial = new THREE.LineBasicMaterial({ color: 0x1e3a5f, transparent: true, opacity: 0.12 })
    for (const lat of [0, 35, -35]) {
      const latRad = (lat * Math.PI) / 180
      const ringR = 140 * Math.cos(latRad)
      const ringY = 140 * Math.sin(latRad)
      const ringPts: THREE.Vector3[] = []
      for (let i = 0; i <= 64; i++) {
        const angle = (i / 64) * Math.PI * 2
        ringPts.push(new THREE.Vector3(ringR * Math.cos(angle), ringY, ringR * Math.sin(angle)))
      }
      const ringLineGeo = new THREE.BufferGeometry().setFromPoints(ringPts)
      scene.add(new THREE.Line(ringLineGeo, ringMaterial))
    }
    // Longitude guide rings — 3 great circles at 0°, 60°, 120° longitude
    for (const lon of [0, 60, 120]) {
      const lonRad = (lon * Math.PI) / 180
      const lonPts: THREE.Vector3[] = []
      for (let i = 0; i <= 64; i++) {
        const angle = (i / 64) * Math.PI * 2
        lonPts.push(new THREE.Vector3(140 * Math.cos(angle) * Math.cos(lonRad), 140 * Math.sin(angle), 140 * Math.cos(angle) * Math.sin(lonRad)))
      }
      const lonLineGeo = new THREE.BufferGeometry().setFromPoints(lonPts)
      scene.add(new THREE.Line(lonLineGeo, ringMaterial))
    }

    // Configure forces — nodes distributed ON the surface of a sphere
    // Strong charge repels nodes evenly across the sphere surface
    fgRef.current.d3Force('charge').strength(-80).distanceMax(350)
    // Links connect nodes but are weak enough not to overpower radial force
    fgRef.current.d3Force('link').distance(40).strength(0.15)
    // Custom radial force — pins every node TO the sphere surface (not inside)
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const radialForce = (() => {
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      let _nodes: any[] = []
      const sphereRadius = 140
      function force(alpha: number) {
        for (const nd of _nodes) {
          const x = nd.x || 0, y = nd.y || 0, z = nd.z || 0
          const r = Math.sqrt(x * x + y * y + z * z) || 1
          // Very strong pull TO the exact sphere surface — not inside, not outside
          const k = ((sphereRadius - r) / r) * 3.5 * alpha
          nd.vx = (nd.vx || 0) + x * k
          nd.vy = (nd.vy || 0) + y * k
          nd.vz = (nd.vz || 0) + z * k
        }
      }
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      force.initialize = (n: any[]) => { _nodes = n }
      return force
    })()
    fgRef.current.d3Force('radial', radialForce)
    // Keep center force to anchor the sphere at origin
    fgRef.current.d3Force('center', null)
    // Zoom to fit after simulation settles
    setTimeout(() => fgRef.current?.zoomToFit(800, 40), 2000)
    setTimeout(() => fgRef.current?.zoomToFit(800, 40), 5000)
  }, [graphData, dimensions])

  // -- Auto-rotate --
  useEffect(() => {
    if (!fgRef.current) return
    const controls = fgRef.current.controls()
    if (controls) {
      controls.autoRotate = autoRotate
      controls.autoRotateSpeed = 1.5
    }
  }, [autoRotate])

  // -- Fog toggle --
  useEffect(() => {
    if (!fgRef.current) return
    const scene = fgRef.current.scene()
    if (scene) {
      scene.fog = fogEnabled ? new THREE.FogExp2(0x0f172a, 0.0018) : null
    }
  }, [fogEnabled])

  // -- Stars toggle --
  useEffect(() => {
    if (!fgRef.current) return
    const scene = fgRef.current.scene()
    if (!scene) return
    const stars = scene.getObjectByName('__payflow_stars')
    if (stars) stars.visible = starsEnabled
  }, [starsEnabled])

  // -- Node color accessor (handles filter/hover/search) --
  const getNodeColor = useCallback((node: FGNode) => {
    const searchLower = searchTerm.toLowerCase()
    if (searchTerm && !node.id.toLowerCase().includes(searchLower)) return '#2a3a4d'
    if (hoveredNode) {
      if (node.id === hoveredNode) return node.color
      if (neighborSet.has(node.id)) return node.color
      return '#2a3a4d'
    }
    if (filter === 'fraud-only' && node.fraudRatio === 0) return '#2a3a4d'
    if (filter === 'high-risk' && node.status !== 'frozen' && node.status !== 'suspicious') return '#2a3a4d'
    if (filter === 'cycles') {
      if (!cycleNodes.has(node.id)) return '#2a3a4d'
      return '#f59e0b'
    }
    if (filter === 'community') return COMMUNITY_COLORS[node.degree % COMMUNITY_COLORS.length]
    if (filter === 'threat-path' && (threatLevels.get(node.id) ?? 0) === 0) return '#2a3a4d'
    if (filter === 'mule') {
      if (!muleNodes.has(node.id)) return '#2a3a4d'
      return '#ff6b6b'
    }
    if (filter === 'layering') {
      if (!layeringNodes.has(node.id)) return '#2a3a4d'
      return '#a855f7'
    }
    // Heatmap mode: green → yellow → red based on fraudRatio
    if (heatmapMode) {
      const r = Math.min(1, node.fraudRatio ?? 0)
      const red = Math.round(255 * Math.min(1, r * 2))
      const green = Math.round(255 * Math.min(1, (1 - r) * 2))
      return `rgb(${red},${green},40)`
    }
    return node.color
  }, [searchTerm, hoveredNode, neighborSet, filter, cycleNodes, threatLevels, heatmapMode, muleNodes, layeringNodes])

  // -- Node size accessor --
  const getNodeVal = useCallback((node: FGNode) => {
    if (searchTerm && !node.id.toLowerCase().includes(searchTerm.toLowerCase())) return node.val * 0.2
    if (hoveredNode) {
      if (node.id === hoveredNode) return node.val * 2
      if (neighborSet.has(node.id)) return node.val * 1.3
      return node.val * 0.2
    }
    if (filter === 'cycles' && cycleNodes.has(node.id)) return node.val * 1.5
    if (filter === 'threat-path') {
      const level = threatLevels.get(node.id) ?? 0
      if (level === 0) return node.val * 0.2
      return node.val * (0.8 + level)
    }
    if (filter === 'mule') {
      if (!muleNodes.has(node.id)) return node.val * 0.2
      return node.val * 1.8
    }
    if (filter === 'layering') {
      if (!layeringNodes.has(node.id)) return node.val * 0.2
      return node.val * 1.4
    }
    return node.val
  }, [searchTerm, hoveredNode, neighborSet, filter, cycleNodes, threatLevels, muleNodes, layeringNodes])

  // -- Link visibility --
  const getLinkVis = useCallback((link: FGLink) => {
    const src = getLinkSourceId(link)
    const tgt = getLinkTargetId(link)
    // Advanced channel/fraud-type filters
    if (link.channel != null && !activeChannels.has(Number(link.channel))) return false
    if (link.fraud_label != null && !activeFraudTypes.has(link.fraud_label)) return false
    if (hoveredNode) return src === hoveredNode || tgt === hoveredNode
    if (filter === 'fraud-only') return link.fraud_label > 0
    if (filter === 'high-risk') {
      const srcNode = graphData.nodes.find(n => n.id === src)
      const tgtNode = graphData.nodes.find(n => n.id === tgt)
      if (!srcNode || !tgtNode) return false
      return srcNode.status === 'frozen' || srcNode.status === 'suspicious' || tgtNode.status === 'frozen' || tgtNode.status === 'suspicious'
    }
    if (filter === 'cycles') return cycleNodes.has(src) && cycleNodes.has(tgt)
    if (filter === 'threat-path') {
      const sl = threatLevels.get(src) ?? 0
      const tl = threatLevels.get(tgt) ?? 0
      return sl > 0 || tl > 0
    }
    if (filter === 'mule') {
      return muleNodes.has(src) || muleNodes.has(tgt)
    }
    if (filter === 'layering') {
      return layeringNodes.has(src) && layeringNodes.has(tgt)
    }
    return true
  }, [hoveredNode, filter, cycleNodes, threatLevels, graphData.nodes, activeChannels, activeFraudTypes, muleNodes, layeringNodes])

  // -- Node tooltip (rich HTML) --
  const getNodeLabel = useCallback((node: FGNode) => {
    const riskLevel = node.fraudRatio > 0.6 ? 'CRITICAL' : node.fraudRatio > 0.3 ? 'HIGH' : node.fraudRatio > 0.1 ? 'MEDIUM' : 'LOW'
    const riskColor = ({ CRITICAL: '#ef4444', HIGH: '#f97316', MEDIUM: '#eab308', LOW: '#22c55e' } as Record<string,string>)[riskLevel]
    return `<div style="background:rgba(15,23,42,0.95);border:1px solid rgba(71,85,105,0.5);border-radius:8px;padding:8px 12px;min-width:180px;font-family:system-ui;backdrop-filter:blur(8px);">
      <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:4px;">
        <span style="font-size:12px;font-family:monospace;color:#e2e8f0;font-weight:600;">${node.id}</span>
        <span style="font-size:9px;font-weight:700;padding:2px 6px;border-radius:999px;background:${riskColor}22;color:${riskColor};border:1px solid ${riskColor}44;">${riskLevel}</span>
      </div>
      <div style="display:flex;align-items:center;gap:8px;font-size:10px;color:#94a3b8;margin-bottom:6px;">
        <span style="padding:2px 6px;border-radius:4px;background:${STATUS_BORDER_COLORS[node.status]}22;color:${STATUS_BORDER_COLORS[node.status]}">${node.status}</span>
        <span>${node.degree} connections</span>
      </div>
      <div style="font-size:10px;color:#94a3b8;margin-bottom:4px;">Fraud: ${(node.fraudRatio * 100).toFixed(0)}%</div>
      <div style="width:100%;height:6px;background:#334155;border-radius:999px;overflow:hidden;">
        <div style="height:100%;width:${node.fraudRatio * 100}%;background:linear-gradient(90deg,${riskColor}88,${riskColor});border-radius:999px;"></div>
      </div>
      ${node.dominantFraud > 0 ? `<div style="font-size:10px;margin-top:6px;color:${FRAUD_TYPE_COLORS[node.dominantFraud]}">${FRAUD_PATTERN_LABELS[node.dominantFraud] ?? 'Type ' + node.dominantFraud}</div>` : ''}
    </div>`
  }, [])

  // -- Link tooltip (enriched) --
  const getLinkLabel = useCallback((link: FGLink) => {
    const src = getLinkSourceId(link)
    const tgt = getLinkTargetId(link)
    const fmtAmt = link.amount >= 1e7 ? `₹${(link.amount / 1e7).toFixed(2)} Cr` : link.amount >= 1e5 ? `₹${(link.amount / 1e5).toFixed(2)} L` : `₹${(link.amount / 100).toLocaleString()}`
    const chLabel = CHANNEL_LABELS[link.channel] ?? `Ch ${link.channel}`
    const chColor = CHANNEL_COLORS[link.channel] ?? '#94a3b8'
    const isFraud = link.fraud_label > 0
    const borderClr = isFraud ? FRAUD_TYPE_COLORS[link.fraud_label] + '66' : 'rgba(71,85,105,0.5)'
    return `<div style="background:rgba(15,23,42,0.97);border:1px solid ${borderClr};border-radius:10px;padding:10px 14px;min-width:200px;font-family:system-ui;backdrop-filter:blur(12px);box-shadow:0 8px 32px rgba(0,0,0,0.4);">
      <div style="display:flex;align-items:center;gap:6px;margin-bottom:6px;">
        <span style="font-size:11px;font-family:monospace;color:#e2e8f0;font-weight:600;">${src.length > 14 ? src.slice(0,7)+'..'+src.slice(-5) : src}</span>
        <span style="color:#475569;font-size:13px;">→</span>
        <span style="font-size:11px;font-family:monospace;color:#e2e8f0;font-weight:600;">${tgt.length > 14 ? tgt.slice(0,7)+'..'+tgt.slice(-5) : tgt}</span>
      </div>
      <div style="display:flex;align-items:center;gap:6px;font-size:10px;margin-bottom:6px;">
        <span style="padding:2px 8px;border-radius:999px;background:${chColor}22;color:${chColor};border:1px solid ${chColor}33;font-weight:500;">${chLabel}</span>
        ${isFraud ? `<span style="padding:2px 8px;border-radius:999px;background:${FRAUD_TYPE_COLORS[link.fraud_label]}22;color:${FRAUD_TYPE_COLORS[link.fraud_label]};border:1px solid ${FRAUD_TYPE_COLORS[link.fraud_label]}33;font-weight:600;">⚠ ${FRAUD_PATTERN_LABELS[link.fraud_label] ?? 'Type ' + link.fraud_label}</span>` : '<span style="padding:2px 8px;border-radius:999px;background:#22c55e22;color:#22c55e;font-size:9px;">Clean</span>'}
      </div>
      <div style="display:flex;align-items:baseline;gap:4px;">
        <span style="font-size:14px;font-weight:700;color:${isFraud ? '#f87171' : '#34d399'};font-family:monospace;">${fmtAmt}</span>
      </div>
    </div>`
  }, [])

  // -- Custom node objects with Three.js --
  const nodeThreeObject = useCallback((node: FGNode) => {
    const size = Math.cbrt(getNodeVal(node)) * 2.0
    const color = getNodeColor(node)
    const isDimmed = color === '#2a3a4d'

    // Smooth lit sphere
    const geo = new THREE.SphereGeometry(size, 16, 16)
    const mat = new THREE.MeshPhongMaterial({
      color: new THREE.Color(color),
      transparent: true,
      opacity: isDimmed ? 0.06 : 0.92,
      emissive: new THREE.Color(color),
      emissiveIntensity: isDimmed ? 0 : 0.45,
      shininess: 80,
      specular: new THREE.Color(0x444444),
    })
    const sphere = new THREE.Mesh(geo, mat)

    // All visible (non-dimmed) nodes get a group for labels; only notable ones get effects
    if (isDimmed) return sphere

    const isNotable = (
      node.status === 'frozen' ||
      node.status === 'suspicious' ||
      node.degree >= 4 ||
      node.fraudRatio > 0.2
    )

    const group = new THREE.Group()
    group.add(sphere)

    // Cycle node highlight — spinning octahedron wireframe
    if (filter === 'cycles' && cycleNodes.has(node.id)) {
      const wireGeo = new THREE.OctahedronGeometry(size * 2.8, 0)
      const wireMat = new THREE.MeshBasicMaterial({
        color: 0xfbbf24,
        wireframe: true,
        transparent: true,
        opacity: 0.5,
      })
      const wireMesh = new THREE.Mesh(wireGeo, wireMat)
      wireMesh.onBeforeRender = () => {
        wireMesh.rotation.y += 0.02
        wireMesh.rotation.x += 0.01
      }
      group.add(wireMesh)
    }

    // Outer glow halo — larger transparent sphere for bloom effect
    if (glowEnabled && node.fraudRatio > 0.15) {
      const glowGeo = new THREE.SphereGeometry(size * 2.2, 12, 12)
      const glowMat = new THREE.MeshBasicMaterial({
        color: new THREE.Color(color),
        transparent: true,
        opacity: Math.min(0.18, node.fraudRatio * 0.35),
        blending: THREE.AdditiveBlending,
        depthWrite: false,
      })
      group.add(new THREE.Mesh(glowGeo, glowMat))
    }

    // Status ring for frozen/suspicious nodes
    if (node.status === 'frozen' || node.status === 'suspicious') {
      const ringGeo = new THREE.RingGeometry(size * 1.7, size * 2.1, 28)
      const ringMat = new THREE.MeshBasicMaterial({
        color: new THREE.Color(node.borderColor),
        transparent: true,
        opacity: 0.55,
        side: THREE.DoubleSide,
      })
      group.add(new THREE.Mesh(ringGeo, ringMat))

      // Animated pulse ring — expands and fades in a loop
      const pulseGeo = new THREE.RingGeometry(size * 1.5, size * 1.8, 32)
      const pulseMat = new THREE.MeshBasicMaterial({
        color: new THREE.Color(node.borderColor),
        transparent: true,
        opacity: 0.4,
        side: THREE.DoubleSide,
        depthWrite: false,
      })
      const pulseRing = new THREE.Mesh(pulseGeo, pulseMat)
      const startTime = performance.now() + Math.random() * 2000
      pulseRing.onBeforeRender = () => {
        const elapsed = ((performance.now() - startTime) % 2000) / 2000
        const scale = 1 + elapsed * 1.5
        pulseRing.scale.set(scale, scale, 1)
        pulseMat.opacity = 0.4 * (1 - elapsed)
      }
      group.add(pulseRing)
    }

    // Mule account indicator — spinning icosahedron wireframe (distinct from cycle octahedron)
    if (filter === 'mule' && muleNodes.has(node.id)) {
      const muleGeo = new THREE.IcosahedronGeometry(size * 2.5, 0)
      const muleMat = new THREE.MeshBasicMaterial({
        color: 0xff6b6b,
        wireframe: true,
        transparent: true,
        opacity: 0.6,
      })
      const muleMesh = new THREE.Mesh(muleGeo, muleMat)
      muleMesh.onBeforeRender = () => {
        muleMesh.rotation.y += 0.015
        muleMesh.rotation.z += 0.008
      }
      group.add(muleMesh)
    }

    // Layering chain indicator — double ring helix for chain nodes
    if (filter === 'layering' && layeringNodes.has(node.id)) {
      const layerRing1 = new THREE.RingGeometry(size * 2.0, size * 2.3, 24)
      const layerMat = new THREE.MeshBasicMaterial({
        color: 0xa855f7,
        transparent: true,
        opacity: 0.45,
        side: THREE.DoubleSide,
      })
      const ring1 = new THREE.Mesh(layerRing1, layerMat)
      ring1.rotation.x = Math.PI / 3
      group.add(ring1)
      const layerRing2 = new THREE.RingGeometry(size * 2.0, size * 2.3, 24)
      const ring2 = new THREE.Mesh(layerRing2, layerMat.clone())
      ring2.rotation.x = -Math.PI / 3
      ring2.rotation.y = Math.PI / 2
      group.add(ring2)
    }

    // Multi-channel diversity indicator — colored dots orbiting multi-channel nodes
    if (node.channels.length >= 3) {
      const chCount = Math.min(node.channels.length, 8)
      const chRadius = size * 2.8
      for (let ci = 0; ci < chCount; ci++) {
        const ch = node.channels[ci]
        const angle = (ci / chCount) * Math.PI * 2
        const dotGeo = new THREE.SphereGeometry(0.25, 6, 6)
        const dotMat = new THREE.MeshBasicMaterial({
          color: new THREE.Color(CHANNEL_COLORS[ch] ?? '#94a3b8'),
          transparent: true,
          opacity: 0.8,
        })
        const dot = new THREE.Mesh(dotGeo, dotMat)
        dot.position.set(chRadius * Math.cos(angle), chRadius * Math.sin(angle), 0)
        group.add(dot)
      }
    }

    // --- Informative labelling with classification ---
    if (labelsEnabled) {
      const idStr = node.id.length > 12
        ? node.id.slice(0, 6) + '..' + node.id.slice(-4)
        : node.id
      const fraudLabel = node.dominantFraud > 0 ? FRAUD_PATTERN_LABELS[node.dominantFraud] : ''
      const riskLevel = node.fraudRatio > 0.6 ? 'CRIT' : node.fraudRatio > 0.3 ? 'HIGH' : node.fraudRatio > 0.1 ? 'MED' : ''
      const statusTag = node.status !== 'normal' ? node.status.toUpperCase() : ''

      // Primary label: account ID + degree
      const primaryText = `${idStr}  [${node.degree}]`
      const primaryColor = node.dominantFraud > 0 ? (FRAUD_TYPE_COLORS[node.dominantFraud] ?? color) : color
      const primary = new SpriteText(primaryText, 1.6, primaryColor)
      primary.fontFace = 'monospace'
      primary.backgroundColor = node.fraudRatio > 0.3
        ? 'rgba(127,29,29,0.6)'
        : node.status === 'frozen' ? 'rgba(127,29,29,0.5)'
        : node.status === 'suspicious' ? 'rgba(120,80,0,0.5)'
        : 'rgba(15,23,42,0.7)'
      primary.borderRadius = 3
      primary.padding = [1, 2]
      primary.position.y = size + 3.5
      group.add(primary)

      // Secondary label: classification badges (fraud type + risk + status)
      if (isNotable) {
        const badges: string[] = []
        if (fraudLabel) badges.push(fraudLabel)
        if (riskLevel) badges.push(`Risk:${riskLevel}`)
        if (statusTag) badges.push(statusTag)
        if (badges.length > 0) {
          const badgeText = badges.join(' · ')
          const badgeColor = node.status === 'frozen' ? '#ef4444'
            : node.status === 'suspicious' ? '#f59e0b'
            : node.dominantFraud > 0 ? (FRAUD_TYPE_COLORS[node.dominantFraud] ?? '#94a3b8')
            : '#94a3b8'
          const badge = new SpriteText(badgeText, 1.15, badgeColor)
          badge.fontFace = 'monospace'
          badge.backgroundColor = 'rgba(15,23,42,0.65)'
          badge.borderRadius = 3
          badge.padding = [0.6, 1.5]
          badge.position.y = size + 6.5
          group.add(badge)
        }
      }
    }

    return group
  }, [getNodeColor, getNodeVal, glowEnabled, labelsEnabled, filter, cycleNodes, heatmapMode, muleNodes, layeringNodes])

  // -- Handlers --
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const handleNodeClick = useCallback((node: any) => {
    if (!node) return
    setSelectedNode(prev => prev === node.id ? null : node.id)
    setSelectedNodeUI(node.id)
    if (fgRef.current) {
      const dist = 120
      const distRatio = 1 + dist / Math.hypot(node.x || 1, node.y || 1, node.z || 1)
      fgRef.current.cameraPosition(
        { x: (node.x || 0) * distRatio, y: (node.y || 0) * distRatio, z: (node.z || 0) * distRatio },
        node,
        1000,
      )
    }
  }, [setSelectedNodeUI])

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const handleNodeHover = useCallback((node: any) => {
    setHoveredNode(node?.id ?? null)
    if (containerRef.current) {
      containerRef.current.style.cursor = node ? 'pointer' : 'default'
    }
  }, [])

  const handleRestart = useCallback(() => {
    setVersion(v => v + 1)
    setFilter('none')
    setCycleNodes(new Set())
    setSearchTerm('')
    setSelectedNode(null)
    setHoveredNode(null)
    if (fgRef.current) {
      fgRef.current.d3ReheatSimulation()
      setTimeout(() => fgRef.current?.zoomToFit(400), 500)
    }
  }, [])

  const handleCycles = useCallback(() => {
    const cycles = detectCycles(graphRef.current)
    const nodeSet = new Set<string>()
    cycles.forEach(c => c.forEach(n => nodeSet.add(n)))
    setCycleNodes(nodeSet)
    if (nodeSet.size > 0) setFilter('cycles')
  }, [])

  const handleMules = useCallback(() => {
    if (muleNodes.size > 0) setFilter('mule')
  }, [muleNodes])

  const handleLayering = useCallback(() => {
    if (layeringNodes.size > 0) setFilter('layering')
  }, [layeringNodes])

  // -- Fly camera to a specific node by ID --
  const handleFlyToNode = useCallback((nodeId: string) => {
    if (!fgRef.current) return
    const data = fgRef.current.graphData()
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const node = data.nodes.find((n: any) => n.id === nodeId)
    if (!node) return
    setSelectedNode(nodeId)
    setSelectedNodeUI(nodeId)
    const dist = 100
    const distRatio = 1 + dist / Math.hypot(node.x || 1, node.y || 1, node.z || 1)
    fgRef.current.cameraPosition(
      { x: (node.x || 0) * distRatio, y: (node.y || 0) * distRatio, z: (node.z || 0) * distRatio },
      node,
      1200,
    )
  }, [setSelectedNodeUI])

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
    <div ref={containerRef} className="relative w-full h-full rounded-xl overflow-hidden border border-slate-800 bg-slate-950">
      <ForceGraph3D
          ref={fgRef}
          width={dimensions.width}
          height={dimensions.height}
          graphData={graphData}
          backgroundColor="#0f172a"
          nodeThreeObject={nodeThreeObject}
          nodeThreeObjectExtend={false}
          nodeLabel={getNodeLabel}
          linkColor={(link: FGLink) => {
            if (filter === 'cycles') {
              const src = getLinkSourceId(link)
              const tgt = getLinkTargetId(link)
              if (cycleNodes.has(src) && cycleNodes.has(tgt)) return '#f59e0b'
            }
            if (filter === 'mule') {
              const src = getLinkSourceId(link)
              const tgt = getLinkTargetId(link)
              if (muleNodes.has(src) || muleNodes.has(tgt)) return '#ff6b6b'
            }
            if (filter === 'layering') {
              const src = getLinkSourceId(link)
              const tgt = getLinkTargetId(link)
              if (layeringNodes.has(src) && layeringNodes.has(tgt)) return '#a855f7'
            }
            // Structuring threshold warning color
            const edgeKey = `${getLinkSourceId(link)}->${getLinkTargetId(link)}`
            if (structuringEdgeKeys.has(edgeKey)) return '#f59e0b'
            return link.color
          }}
          linkWidth={(link: FGLink) => {
            if (filter === 'cycles') {
              const src = getLinkSourceId(link)
              const tgt = getLinkTargetId(link)
              if (cycleNodes.has(src) && cycleNodes.has(tgt)) return 2.5
            }
            if (filter === 'mule') {
              const src = getLinkSourceId(link)
              const tgt = getLinkTargetId(link)
              if (muleNodes.has(src) || muleNodes.has(tgt)) return Math.max(link.width, 1.2)
            }
            if (filter === 'layering') {
              const src = getLinkSourceId(link)
              const tgt = getLinkTargetId(link)
              if (layeringNodes.has(src) && layeringNodes.has(tgt)) return Math.max(link.width, 1.0)
            }
            return link.width
          }}
          linkOpacity={0.35}
          linkCurvature={0.12}
          linkCurveRotation={0}
          linkVisibility={getLinkVis}
          linkDirectionalArrowLength={(link: FGLink) => link.fraud_label > 0 ? 3.0 : 1.5}
          linkDirectionalArrowRelPos={0.85}
          linkDirectionalArrowColor={(link: FGLink) => link.color}
          linkDirectionalParticles={particlesEnabled ? ((link: FGLink) => {
            if (filter === 'cycles') {
              const src = getLinkSourceId(link)
              const tgt = getLinkTargetId(link)
              if (cycleNodes.has(src) && cycleNodes.has(tgt)) return 6
            }
            if (filter === 'mule') {
              const src = getLinkSourceId(link)
              const tgt = getLinkTargetId(link)
              if (muleNodes.has(src) || muleNodes.has(tgt)) return 5
            }
            if (filter === 'layering') {
              const src = getLinkSourceId(link)
              const tgt = getLinkTargetId(link)
              if (layeringNodes.has(src) && layeringNodes.has(tgt)) return 5
            }
            // High-amount links get more particles
            if (link.amount > 500000) return link.fraud_label > 0 ? 5 : 2
            return link.fraud_label > 0 ? 4 : 1
          }) : 0}
          linkDirectionalParticleWidth={(link: FGLink) => {
            if (filter === 'cycles') {
              const src = getLinkSourceId(link)
              const tgt = getLinkTargetId(link)
              if (cycleNodes.has(src) && cycleNodes.has(tgt)) return 3
            }
            // Amount-proportional particle size
            if (link.amount > 500000) return link.fraud_label > 0 ? 3.0 : 1.5
            return link.fraud_label > 0 ? 2.4 : 0.8
          }}
          linkDirectionalParticleSpeed={(link: FGLink) => {
            if (filter === 'cycles') {
              const src = getLinkSourceId(link)
              const tgt = getLinkTargetId(link)
              if (cycleNodes.has(src) && cycleNodes.has(tgt)) return 0.015
            }
            return link.fraud_label > 0 ? 0.008 : 0.003
          }}
          linkDirectionalParticleColor={(link: FGLink) => {
            if (filter === 'cycles') {
              const src = getLinkSourceId(link)
              const tgt = getLinkTargetId(link)
              if (cycleNodes.has(src) && cycleNodes.has(tgt)) return '#fbbf24'
            }
            if (filter === 'mule') return '#ff6b6b'
            if (filter === 'layering') return '#a855f7'
            // Structuring gets amber particles
            const edgeKey = `${getLinkSourceId(link)}->${getLinkTargetId(link)}`
            if (structuringEdgeKeys.has(edgeKey)) return '#f59e0b'
            return FRAUD_TYPE_COLORS[link.fraud_label] ?? '#3b82f6'
          }}
          linkLabel={getLinkLabel}
          onNodeClick={handleNodeClick}
          onNodeHover={handleNodeHover}
          enableNodeDrag={true}
          warmupTicks={200}
          cooldownTicks={400}
          d3AlphaDecay={0.025}
          d3VelocityDecay={0.4}
        />

      {/* Controls overlay */}
      <Graph3DControls
        fgRef={fgRef}
        filter={filter}
        setFilter={setFilter}
        searchTerm={searchTerm}
        setSearchTerm={setSearchTerm}
        onRestart={handleRestart}
        onCycles={handleCycles}
        onMules={handleMules}
        onLayering={handleLayering}
        graphData={graphData}
        onFlyToNode={handleFlyToNode}
        activeChannels={activeChannels}
        setActiveChannels={setActiveChannels}
        activeFraudTypes={activeFraudTypes}
        setActiveFraudTypes={setActiveFraudTypes}
        heatmapMode={heatmapMode}
        setHeatmapMode={setHeatmapMode}
        particlesEnabled={particlesEnabled}
        setParticlesEnabled={setParticlesEnabled}
        autoRotate={autoRotate}
        setAutoRotate={setAutoRotate}
        fogEnabled={fogEnabled}
        setFogEnabled={setFogEnabled}
        glowEnabled={glowEnabled}
        setGlowEnabled={setGlowEnabled}
        labelsEnabled={labelsEnabled}
        setLabelsEnabled={setLabelsEnabled}
        starsEnabled={starsEnabled}
        setStarsEnabled={setStarsEnabled}
      />

      {/* Selected node detail panel */}
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

      <GraphLegend edges={edges} />
      <GraphStatsOverlay
        graph={graphRef.current}
        edges={edges}
        clusteringCoeff={clusteringCoeff}
        pageRanks={pageRanks}
        betweenness={betweennessMap}
        muleCount={muleNodes.size}
        layeringCount={layeringNodes.size}
        structuringCount={structuringEdgeKeys.size}
      />
      <NetworkRiskGauge
        graph={graphRef.current}
        edges={edges}
        clusteringCoeff={clusteringCoeff}
      />
      <ThreatRadar active={filter === 'threat-path'} />

      {/* 3D navigation hints */}
      <div className="absolute bottom-2 left-1/2 -translate-x-1/2 z-10 flex items-center gap-3 text-[9px] text-slate-600 select-none">
        <span>Left-drag: <b className="text-slate-500">rotate</b></span>
        <span>Right-drag: <b className="text-slate-500">pan</b></span>
        <span>Scroll: <b className="text-slate-500">zoom</b></span>
        <span><kbd className="px-1 py-0.5 bg-slate-800 rounded text-slate-500 font-mono">P</kbd> particles</span>
        <span><kbd className="px-1 py-0.5 bg-slate-800 rounded text-slate-500 font-mono">R</kbd> rotate</span>
        <span><kbd className="px-1 py-0.5 bg-slate-800 rounded text-slate-500 font-mono">G</kbd> glow</span>
        <span><kbd className="px-1 py-0.5 bg-slate-800 rounded text-slate-500 font-mono">L</kbd> labels</span>
        <span><kbd className="px-1 py-0.5 bg-slate-800 rounded text-slate-500 font-mono">F</kbd> fog</span>
      </div>
    </div>
  )
}

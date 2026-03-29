// ============================================================================
// Cryptographic Audit Trail -- Scrolling block ledger with ZKP badges
// ============================================================================

import { useRef, useEffect, useState, useMemo } from 'react'
import { Lock, Check, KeyRound, GitMerge, ChevronDown, Blocks } from 'lucide-react'
import { useRecentBlocks } from '@/hooks/use-api'
import { cn, fmtTimestamp, fmtNum } from '@/lib/utils'
import type { LedgerBlock } from '@/lib/types'

const EVENT_TYPE_STYLES: Record<string, { label: string; color: string }> = {
  SYSTEM_STATE:     { label: 'SYSTEM',   color: 'text-blue-400'    },
  ALERT:            { label: 'ALERT',    color: 'text-red-400'     },
  INVESTIGATION:    { label: 'INVEST',   color: 'text-amber-400'   },
  MODEL_UPDATE:     { label: 'MODEL',    color: 'text-violet-400'  },
  ZKP_VERIFICATION: { label: 'ZKP',      color: 'text-emerald-400' },
  CIRCUIT_BREAKER:  { label: 'BREAKER',  color: 'text-orange-400'  },
  AGENT_VERDICT:    { label: 'VERDICT',  color: 'text-cyan-400'    },
}

export function CryptographicAuditTrail() {
  const { data } = useRecentBlocks(80)
  const scrollRef = useRef<HTMLDivElement>(null)
  const [autoScroll, setAutoScroll] = useState(true)
  const prevCountRef = useRef(0)

  const blocks = useMemo(() => data?.blocks ?? [], [data])
  const stats = data?.stats ?? null

  // Auto-scroll when new blocks arrive
  useEffect(() => {
    if (blocks.length > prevCountRef.current && autoScroll && scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight
    }
    prevCountRef.current = blocks.length
  }, [blocks, autoScroll])

  function handleScroll() {
    const el = scrollRef.current
    if (!el) return
    const atBottom = el.scrollHeight - el.scrollTop - el.clientHeight < 40
    setAutoScroll(atBottom)
  }

  return (
    <div className="flex flex-col h-full bg-bg-deep rounded-lg border border-border-subtle overflow-hidden">
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-2.5 border-b border-border-subtle shrink-0 bg-bg-surface">
        <div className="flex items-center gap-2.5">
          <Lock className="w-3.5 h-3.5 text-emerald-400" />
          <span className="text-[10px] font-semibold uppercase tracking-[0.12em] text-text-secondary">
            Cryptographic Audit Trail
          </span>
        </div>
        {stats && (
          <div className="flex items-center gap-3 text-[9px] font-mono text-text-muted tabular-nums">
            <span>{fmtNum(stats.total_blocks)} blocks</span>
            <span>{fmtNum(stats.checkpoints)} checkpoints</span>
          </div>
        )}
      </div>

      {/* Block stream */}
      <div
        ref={scrollRef}
        onScroll={handleScroll}
        className="flex-1 overflow-y-auto p-3 space-y-2"
      >
        {blocks.length === 0 && (
          <div className="flex items-center justify-center h-full animate-fade-in">
            <div className="text-center space-y-3">
              <Blocks className="w-8 h-8 text-text-muted/30 mx-auto" />
              <p className="text-text-muted text-[10px] uppercase tracking-[0.12em]">
                No blocks yet...
              </p>
              <p className="text-text-muted/50 text-[9px] max-w-[240px] mx-auto leading-relaxed">
                Blocks are anchored as the AI processes events
              </p>
            </div>
          </div>
        )}

        {blocks.map((block) => (
          <BlockCard key={block.index} block={block} />
        ))}
      </div>

      {/* Scroll-to-bottom button */}
      {!autoScroll && (
        <button
          onClick={() => {
            setAutoScroll(true)
            scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: 'smooth' })
          }}
          className="absolute bottom-10 right-4 flex items-center gap-1.5 text-[9px] bg-accent-primary/90 text-white px-2.5 py-1 rounded-md hover:bg-accent-primary transition-colors shadow-lg shadow-accent-primary/20"
        >
          <ChevronDown className="w-3 h-3" />
          Latest
        </button>
      )}

      {/* Chain integrity footer */}
      {stats && (
        <div className="flex items-center justify-between px-4 py-2 border-t border-border-subtle bg-bg-surface shrink-0">
          <div className="flex items-center gap-2">
            <span className="relative flex h-2 w-2">
              <span className="absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-50 animate-ping" />
              <span className="relative inline-flex rounded-full h-2 w-2 bg-emerald-400" />
            </span>
            <span className="text-[9px] text-emerald-400 font-semibold uppercase tracking-[0.12em]">
              Chain Intact
            </span>
          </div>
          <span className="text-[9px] text-text-muted font-mono truncate max-w-[220px]">
            HEAD: {stats.latest_hash.slice(0, 16)}...
          </span>
        </div>
      )}
    </div>
  )
}

// -- Single block card --

function BlockCard({ block }: { block: LedgerBlock }) {
  const [expanded, setExpanded] = useState(false)
  const evtStyle = EVENT_TYPE_STYLES[block.event_type] ?? { label: block.event_type, color: 'text-text-muted' }
  const isZkp = block.event_type === 'ZKP_VERIFICATION'
  const isHighConfidence = block.event_type === 'AGENT_VERDICT' || block.event_type === 'CIRCUIT_BREAKER'

  return (
    <div
      onClick={() => setExpanded(!expanded)}
      className={cn(
        'card-hover rounded-lg border px-3 py-2 cursor-pointer transition-all',
        isHighConfidence
          ? 'border-accent-primary/30 bg-accent-primary/[0.04]'
          : 'border-border-subtle/50 bg-bg-surface/50',
      )}
    >
      {/* Top row: index + event type + timestamp */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <span className="text-[10px] font-mono text-text-muted/50 tabular-nums">#{block.index}</span>
          <span className={cn('text-[10px] font-bold uppercase tracking-wider', evtStyle.color)}>
            {evtStyle.label}
          </span>
          {(isZkp || isHighConfidence) && <ZKPBadge />}
          {block.has_signature && <SigBadge />}
          {block.merkle_root && <MerkleBadge />}
        </div>
        <span className="text-[9px] text-text-muted/50 font-mono tabular-nums">
          {fmtTimestamp(block.timestamp)}
        </span>
      </div>

      {/* Hash row */}
      <div className="mt-1.5 flex items-center gap-1.5">
        <span className="text-[9px] text-text-muted/40 uppercase tracking-wider">SHA-256:</span>
        <span className="text-[9px] font-mono text-text-secondary/70 truncate">
          {block.block_hash}
        </span>
      </div>

      {/* Prev hash linkage */}
      <div className="flex items-center gap-1.5">
        <span className="text-[9px] text-text-muted/40 uppercase tracking-wider">PREV:</span>
        <span className="text-[9px] font-mono text-text-muted/40 truncate">
          {block.prev_hash.slice(0, 32)}...
        </span>
      </div>

      {/* Expanded payload */}
      {expanded && (
        <div className="mt-2 pt-2 border-t border-border-subtle/30 animate-fade-in">
          <pre className="text-[9px] font-mono text-text-muted/70 whitespace-pre-wrap break-all max-h-40 overflow-y-auto bg-bg-deep/60 rounded-md p-2.5 border border-border-subtle/20">
            {JSON.stringify(block.payload, null, 2)}
          </pre>
        </div>
      )}
    </div>
  )
}

// -- Badges --

function ZKPBadge() {
  return (
    <span className="inline-flex items-center gap-1 px-1.5 py-0.5 rounded-full bg-emerald-500/15 border border-emerald-500/30">
      <Check className="w-2.5 h-2.5 text-emerald-400" strokeWidth={3} />
      <span className="text-[8px] font-bold text-emerald-400 uppercase tracking-wider">
        ZKP Verified
      </span>
    </span>
  )
}

function SigBadge() {
  return (
    <span className="inline-flex items-center gap-1 px-1.5 py-0.5 rounded-full bg-blue-500/10 border border-blue-500/20">
      <KeyRound className="w-2.5 h-2.5 text-blue-400" strokeWidth={2.5} />
      <span className="text-[8px] font-bold text-blue-400 uppercase tracking-wider">
        Ed25519
      </span>
    </span>
  )
}

function MerkleBadge() {
  return (
    <span className="inline-flex items-center gap-1 px-1.5 py-0.5 rounded-full bg-violet-500/10 border border-violet-500/20">
      <GitMerge className="w-2.5 h-2.5 text-violet-400" strokeWidth={2.5} />
      <span className="text-[8px] font-bold text-violet-400 uppercase tracking-wider">
        Merkle
      </span>
    </span>
  )
}

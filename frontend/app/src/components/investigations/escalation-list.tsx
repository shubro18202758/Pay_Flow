// ============================================================================
// Escalation List -- HITL escalations from analyst API
// ============================================================================

import { UserCheck, AlertCircle, Loader2 } from 'lucide-react'
import { useEscalations } from '@/hooks/use-api'
import { SeverityBadge } from '@/components/shared/severity-badge'
import { truncId } from '@/lib/utils'

export function EscalationList() {
  const { data: escalations, isLoading } = useEscalations()

  return (
    <div className="flex flex-col h-full">
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-2.5 border-b border-border-subtle shrink-0 bg-bg-surface">
        <div className="flex items-center gap-2.5">
          <UserCheck className="w-3.5 h-3.5 text-amber-400" />
          <span className="text-[10px] font-semibold uppercase tracking-[0.12em] text-text-secondary">
            HITL Escalations
          </span>
        </div>
        <span className="text-[9px] font-mono text-text-muted tabular-nums">
          {escalations?.length ?? 0}
        </span>
      </div>

      {/* List */}
      <div className="flex-1 overflow-y-auto">
        {isLoading ? (
          <div className="flex items-center justify-center h-full animate-fade-in">
            <div className="text-center space-y-3">
              <Loader2 className="w-6 h-6 text-text-muted/40 mx-auto animate-spin" />
              <p className="text-text-muted text-[10px] uppercase tracking-[0.12em]">
                Loading...
              </p>
            </div>
          </div>
        ) : !escalations || escalations.length === 0 ? (
          <div className="flex items-center justify-center h-full animate-fade-in">
            <div className="text-center space-y-3">
              <AlertCircle className="w-8 h-8 text-text-muted/30 mx-auto" />
              <p className="text-text-muted text-[10px] uppercase tracking-[0.12em]">
                No escalations pending
              </p>
              <p className="text-text-muted/50 text-[9px] max-w-[200px] mx-auto leading-relaxed">
                Escalations appear when AI confidence is below threshold or analyst review is required
              </p>
            </div>
          </div>
        ) : (
          <div className="p-3 space-y-2">
            {escalations.map((esc) => (
              <div
                key={esc.ack_id}
                className="card-hover bg-bg-elevated rounded-lg p-3 border border-border-subtle animate-fade-in cursor-pointer"
              >
                <div className="flex items-center gap-2 mb-1.5">
                  <SeverityBadge severity="escalated" />
                  <span className="text-[10px] font-mono font-bold text-text-primary tracking-wide">
                    {truncId(esc.ack_id, 10)}
                  </span>
                </div>
                <div className="text-[9px] text-text-muted font-mono leading-relaxed bg-bg-deep/60 rounded-md p-2 border border-border-subtle/20">
                  {JSON.stringify(esc.payload).slice(0, 120)}
                  {JSON.stringify(esc.payload).length > 120 && '...'}
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  )
}

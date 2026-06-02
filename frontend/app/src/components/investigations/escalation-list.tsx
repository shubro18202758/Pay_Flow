// ============================================================================
// Escalation List -- HITL escalations from analyst API
// ============================================================================

import { UserCheck, AlertCircle, Loader2, CheckCircle2, XCircle, ArrowUpRight } from 'lucide-react'
import { useDecideEscalation, useEscalations } from '@/hooks/use-api'
import { useRoleAccess } from '@/hooks/use-rbac'
import { SeverityBadge } from '@/components/shared/severity-badge'
import { cn, fmtOptionalTimestamp, truncId } from '@/lib/utils'
import type { Escalation } from '@/lib/types'

function prioritySeverity(priority?: string) {
  if (priority === 'critical') return 'critical' as const
  if (priority === 'high') return 'high' as const
  if (priority === 'low') return 'low' as const
  return 'medium' as const
}

function statusClass(status?: string) {
  if (status === 'approved') return 'bg-alert-low/10 text-alert-low border-alert-low/25'
  if (status === 'rejected') return 'bg-text-muted/10 text-text-muted border-border-subtle'
  if (status === 'escalated') return 'bg-alert-escalated/10 text-alert-escalated border-alert-escalated/25'
  return 'bg-alert-medium/10 text-alert-medium border-alert-medium/25'
}

function pct(value?: number) {
  return typeof value === 'number' && Number.isFinite(value) ? `${Math.round(value * 100)}%` : 'n/a'
}

function evidenceLine(esc: Escalation) {
  const summary = esc.evidence_summary
  if (!summary) return 'Evidence summary pending'
  const pieces = [
    `${summary.reasoning_steps ?? 0} rationale steps`,
    `${summary.subgraph_nodes ?? 0} nodes`,
    `${summary.subgraph_edges ?? 0} edges`,
  ]
  if (summary.mule_network_detected) pieces.push('mule pattern')
  if ((summary.cycles_found ?? 0) > 0) pieces.push(`${summary.cycles_found} cycles`)
  return pieces.join(' · ')
}

export function EscalationList() {
  const access = useRoleAccess()
  const { data: escalations, isLoading } = useEscalations()
  const decide = useDecideEscalation()
  const pendingAck = decide.variables?.ackId
  const canDecide = access.can('case:decide')

  return (
    <div className="flex flex-col h-full">
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-2.5 border-b border-border-subtle shrink-0 bg-bg-surface">
        <div className="flex items-center gap-2.5">
          <UserCheck className="w-3.5 h-3.5 text-[#DA251C]" />
          <span className="text-[10px] font-semibold uppercase tracking-[0.12em] text-text-secondary">
            HITL Escalations
          </span>
        </div>
        <span className="text-[9px] font-mono text-text-muted tabular-nums">
          {access.policy.label} | {escalations?.length ?? 0}
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
                className="card-hover bg-bg-elevated rounded-lg p-3 border border-border-subtle animate-fade-in"
              >
                <div className="flex items-start gap-2">
                  <SeverityBadge severity={prioritySeverity(esc.priority)} label={esc.priority ?? 'review'} />
                  <div className="min-w-0 flex-1">
                    <div className="flex min-w-0 items-center gap-2">
                      <span className="truncate text-[10px] font-mono font-bold text-text-primary tracking-wide">
                        {esc.case_id ?? truncId(esc.ack_id, 10)}
                      </span>
                      <span className={cn('rounded-full border px-1.5 py-0.5 text-[7px] font-bold uppercase tracking-wide', statusClass(esc.status))}>
                        {(esc.status ?? 'pending_review').replace(/_/g, ' ')}
                      </span>
                    </div>
                    <div className="mt-1 flex flex-wrap items-center gap-2 text-[8px] font-mono text-text-muted">
                      <span>{truncId(esc.txn_id || String(esc.payload.txn_id ?? 'unknown'), 14)}</span>
                      <span>{esc.detected_typology ?? String(esc.payload.detected_typology ?? 'typology pending')}</span>
                      <span className="ml-auto">{fmtOptionalTimestamp(esc.received_at)}</span>
                    </div>
                  </div>
                </div>

                <div className="mt-2 grid grid-cols-3 gap-1.5">
                  <div className="rounded-md border border-border-subtle/30 bg-bg-deep/40 p-1.5">
                    <div className="text-[7px] uppercase tracking-wide text-text-muted">Agent confidence</div>
                    <div className="font-mono text-[10px] font-bold text-text-primary">{pct(esc.agent_confidence)}</div>
                  </div>
                  <div className="rounded-md border border-border-subtle/30 bg-bg-deep/40 p-1.5">
                    <div className="text-[7px] uppercase tracking-wide text-text-muted">ML score</div>
                    <div className="font-mono text-[10px] font-bold text-text-primary">{pct(esc.ml_score)}</div>
                  </div>
                  <div className="rounded-md border border-border-subtle/30 bg-bg-deep/40 p-1.5">
                    <div className="text-[7px] uppercase tracking-wide text-text-muted">Threshold</div>
                    <div className="font-mono text-[10px] font-bold text-text-primary">{pct(esc.confidence_threshold)}</div>
                  </div>
                </div>

                <div className="mt-2 rounded-md border border-border-subtle/20 bg-bg-deep/60 p-2 text-[9px] leading-relaxed text-text-secondary">
                  {evidenceLine(esc)}
                </div>

                {esc.audit_hash && (
                  <div className="mt-1.5 font-mono text-[8px] text-text-muted">
                    audit {truncId(esc.audit_hash, 12)}
                  </div>
                )}

                {esc.status === 'pending_review' && (
                  <div className="mt-2 grid grid-cols-3 gap-1.5">
                    <button
                      type="button"
                      disabled={decide.isPending || !canDecide}
                      title={!canDecide ? `${access.policy.label} cannot approve escalations` : 'Approve escalation'}
                      onClick={() => void decide.mutateAsync({ ackId: esc.ack_id, decision: 'approve', reason: 'analyst_approved_after_evidence_review' })}
                      className="inline-flex items-center justify-center gap-1 rounded-md border border-alert-low/30 bg-alert-low/10 px-2 py-1.5 text-[8px] font-bold uppercase tracking-wide text-alert-low disabled:opacity-40"
                    >
                      {pendingAck === esc.ack_id && decide.isPending ? <Loader2 className="h-3 w-3 animate-spin" /> : <CheckCircle2 className="h-3 w-3" />}
                      Approve
                    </button>
                    <button
                      type="button"
                      disabled={decide.isPending || !canDecide}
                      title={!canDecide ? `${access.policy.label} cannot reject escalations` : 'Reject escalation'}
                      onClick={() => void decide.mutateAsync({ ackId: esc.ack_id, decision: 'reject', reason: 'analyst_rejected_insufficient_evidence' })}
                      className="inline-flex items-center justify-center gap-1 rounded-md border border-border-subtle bg-bg-surface px-2 py-1.5 text-[8px] font-bold uppercase tracking-wide text-text-secondary disabled:opacity-40"
                    >
                      <XCircle className="h-3 w-3" />
                      Reject
                    </button>
                    <button
                      type="button"
                      disabled={decide.isPending || !canDecide}
                      title={!canDecide ? `${access.policy.label} cannot escalate cases to FIU` : 'Escalate to FIU queue'}
                      onClick={() => void decide.mutateAsync({ ackId: esc.ack_id, decision: 'escalate', reason: 'analyst_escalated_to_fiu_queue' })}
                      className="inline-flex items-center justify-center gap-1 rounded-md border border-alert-escalated/30 bg-alert-escalated/10 px-2 py-1.5 text-[8px] font-bold uppercase tracking-wide text-alert-escalated disabled:opacity-40"
                    >
                      <ArrowUpRight className="h-3 w-3" />
                      FIU
                    </button>
                  </div>
                )}
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  )
}

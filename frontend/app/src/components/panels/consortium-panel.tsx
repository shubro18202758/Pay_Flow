// ============================================================================
// Consortium Intelligence Panel -- Cross-bank ZKP-backed fraud intelligence
// ============================================================================

import { useState } from 'react'
import {
  Globe, Shield, ShieldCheck, ShieldAlert, Search, Loader2,
  Users, AlertTriangle, CheckCircle, Lock, Clock,
} from 'lucide-react'
import { cn } from '@/lib/utils'
import { useConsortiumStatus, useConsortiumAlerts, useCheckConsortiumAccount } from '@/hooks/use-api'
import { useRoleAccess } from '@/hooks/use-rbac'

const SEVERITY_COLORS: Record<string, { text: string; bg: string; border: string }> = {
  LOW:      { text: 'text-[#00579C]',   bg: 'bg-[#00579C]/15',   border: 'border-[#00579C]/30' },
  MEDIUM:   { text: 'text-[#DA251C]',  bg: 'bg-[#DA251C]/15',  border: 'border-[#DA251C]/30' },
  HIGH:     { text: 'text-[#DA251C]', bg: 'bg-[#DA251C]/15', border: 'border-[#DA251C]/30' },
  CRITICAL: { text: 'text-[#DA251C]',    bg: 'bg-[#DA251C]/15',    border: 'border-[#DA251C]/30' },
}

export function ConsortiumPanel() {
  const { data: status, isLoading: statusLoading } = useConsortiumStatus()
  const { data: alertsData, isLoading: alertsLoading } = useConsortiumAlerts()
  const checkAccount = useCheckConsortiumAccount()
  const access = useRoleAccess()
  const [accountInput, setAccountInput] = useState('')
  const [activeTab, setActiveTab] = useState<'overview' | 'alerts' | 'lookup'>('overview')
  const canCheckConsortium = access.can('cfr:check') || access.can('consortium:publish') || access.can('audit:review')
  const verifiedCount = status && !('error' in status)
    ? status.verified_proofs ?? status.verified_alerts ?? 0
    : 0

  function getSeverityStyle(severity: string) {
    return SEVERITY_COLORS[severity] ?? SEVERITY_COLORS.LOW
  }

  function submitLookup() {
    if (!accountInput.trim() || !canCheckConsortium) return
    checkAccount.mutate(accountInput.trim())
  }

  return (
    <div className="flex flex-col h-full bg-bg-deep rounded-lg border border-border-subtle overflow-hidden">
      {/* Header */}
      <div className="flex items-center gap-2 px-4 py-3 border-b border-border-subtle bg-bg-surface/50">
        <Globe className="w-4 h-4 text-[#00579C]" />
        <span className="text-xs font-semibold text-text-primary tracking-wide">
          Consortium Intelligence
        </span>
        <Lock className="w-3 h-3 text-text-muted/50 ml-auto" aria-label="Consortium privacy proof channel" />
      </div>

      {/* Tab bar */}
      <div className="flex border-b border-border-subtle bg-bg-surface/30 px-2">
        {(['overview', 'alerts', 'lookup'] as const).map((tab) => (
          <button
            key={tab}
            onClick={() => setActiveTab(tab)}
            className={cn(
              'text-[9px] px-3 py-2 uppercase tracking-wider transition-colors border-b-2',
              activeTab === tab
                ? 'text-[#00579C] border-[#00579C]'
                : 'text-text-muted border-transparent hover:text-text-secondary',
            )}
          >
            {tab}
          </button>
        ))}
      </div>

      {/* Content */}
      <div className="flex-1 min-h-0 overflow-y-auto custom-scrollbar p-4">
        {activeTab === 'overview' && (
          statusLoading ? (
            <div className="flex items-center justify-center h-full">
              <Loader2 className="w-5 h-5 animate-spin text-[#00579C]" />
            </div>
          ) : !status || 'error' in status ? (
            <div className="flex flex-col items-center justify-center h-full text-text-muted text-[10px] gap-2 animate-fade-in">
              <Globe className="w-8 h-8 opacity-30" />
              <p>Consortium hub not available</p>
            </div>
          ) : (
            <div className="space-y-4 animate-fade-in">
              {/* Stats grid */}
              <div className="grid grid-cols-2 gap-2">
                <StatCard icon={Users} label="Member Banks" value={status.member_banks} color="text-[#00579C]" />
                <StatCard icon={AlertTriangle} label="Total Alerts" value={status.total_alerts} color="text-[#DA251C]" />
                <StatCard icon={ShieldCheck} label="Active Alerts" value={status.active_alerts} color="text-[#00579C]" />
                <StatCard icon={Shield} label="ZKP Verified" value={verifiedCount} color="text-[#00579C]" />
              </div>

              {/* Rejected proofs */}
              {status.rejected_proofs > 0 && (
                <div className="flex items-center gap-2 bg-[#DA251C]/10 border border-[#DA251C]/20 rounded-md px-3 py-2">
                  <ShieldAlert className="w-3.5 h-3.5 text-[#DA251C]" />
                  <span className="text-[10px] text-[#DA251C]">{status.rejected_proofs} rejected ZKP proofs</span>
                </div>
              )}

              {/* Alerts by type */}
              {status.alerts_by_type && Object.keys(status.alerts_by_type).length > 0 && (
                <div>
                  <p className="text-[9px] text-text-muted uppercase tracking-wider mb-2">Alerts by Fraud Type</p>
                  <div className="space-y-1">
                    {Object.entries(status.alerts_by_type).map(([type, count]) => (
                      <div key={type} className="flex items-center justify-between text-[10px]">
                        <span className="text-text-secondary">{type}</span>
                        <span className="text-text-primary font-mono">{count}</span>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Member banks */}
              {status.members && Object.keys(status.members).length > 0 && (
                <div>
                  <p className="text-[9px] text-text-muted uppercase tracking-wider mb-2">Peer Banks</p>
                  <div className="space-y-1.5">
                    {Object.entries(status.members).map(([bankId, info]) => (
                      <div key={bankId} className="flex items-center gap-2 bg-bg-surface/40 rounded-md px-3 py-2 border border-border-subtle">
                        <span className="text-[10px] text-text-primary font-mono flex-1">{bankId}</span>
                        <span className="text-[9px] text-text-muted">{info.alerts_published ?? info.alerts_shared ?? 0} shared</span>
                        {info.alerts_received != null && (
                          <span className="text-[9px] text-text-muted">{info.alerts_received} received</span>
                        )}
                        <span className={cn(
                          'text-[9px] font-mono',
                          info.trust_score >= 0.9 ? 'text-[#00579C]' : info.trust_score >= 0.7 ? 'text-[#DA251C]' : 'text-[#DA251C]',
                        )}>
                          {(info.trust_score * 100).toFixed(0)}%
                        </span>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          )
        )}

        {activeTab === 'alerts' && (
          alertsLoading ? (
            <div className="flex items-center justify-center h-full">
              <Loader2 className="w-5 h-5 animate-spin text-[#00579C]" />
            </div>
          ) : !alertsData || alertsData.count === 0 ? (
            <div className="flex flex-col items-center justify-center h-full text-text-muted text-[10px] gap-2 animate-fade-in">
              <CheckCircle className="w-8 h-8 opacity-30 text-[#00579C]" />
              <p>No active consortium alerts</p>
            </div>
          ) : (
            <div className="space-y-2 animate-fade-in">
              <p className="text-[9px] text-text-muted uppercase tracking-wider mb-2">
                {alertsData.count} Alert{alertsData.count !== 1 ? 's' : ''} from Peer Banks
              </p>
              {alertsData.alerts.map((alert) => {
                const sev = getSeverityStyle(alert.severity)
                return (
                  <div
                    key={alert.alert_id}
                    className={cn(
                      'rounded-md border px-3 py-2.5 space-y-1',
                      alert.expired ? 'opacity-50 border-border-subtle' : `${sev.border} ${sev.bg}`,
                    )}
                  >
                    <div className="flex items-center gap-2">
                      <span className={cn('text-[10px] font-semibold', sev.text)}>{alert.severity}</span>
                      <span className="text-[9px] text-text-muted">{alert.fraud_type}</span>
                      {alert.zkp_verified && <ShieldCheck className="w-3 h-3 text-[#00579C] ml-auto" aria-label="ZKP verified" />}
                      {alert.expired && <span className="text-[8px] text-[#DA251C] ml-auto">EXPIRED</span>}
                    </div>
                    <div className="flex items-center gap-3 text-[9px] text-text-muted">
                      <span className="font-mono">{alert.originating_bank}</span>
                      <span>Risk: <span className="text-[#DA251C]">{(alert.risk_score * 100).toFixed(0)}%</span></span>
                      <span className="flex items-center gap-1">
                        <Clock className="w-2.5 h-2.5" />
                        {alert.ttl_hours}h TTL
                      </span>
                    </div>
                    <div className="text-[8px] text-text-muted font-mono truncate">
                      Hash: {alert.account_hash}
                    </div>
                  </div>
                )
              })}
            </div>
          )
        )}

        {activeTab === 'lookup' && (
          <div className="space-y-4 animate-fade-in">
            <p className="text-[10px] text-text-secondary">
              Check if an account has been flagged by any consortium member bank.
            </p>
            <div className="flex items-center gap-2">
              <input
                type="text"
                value={accountInput}
                onChange={(e) => setAccountInput(e.target.value)}
                onKeyDown={(e) => e.key === 'Enter' && submitLookup()}
                placeholder="Account ID"
                disabled={!canCheckConsortium}
                title={!canCheckConsortium ? `${access.policy.label} cannot run Central Fraud Registry lookups` : 'Account ID'}
                className="flex-1 bg-bg-surface border border-border-subtle rounded-md px-3 py-1.5
                  text-[11px] text-text-primary placeholder:text-text-muted/50
                  focus:outline-none focus:border-[#00579C]/40 disabled:cursor-not-allowed disabled:opacity-45 transition-colors"
              />
              <button
                onClick={submitLookup}
                disabled={checkAccount.isPending || !accountInput.trim() || !canCheckConsortium}
                title={!canCheckConsortium ? `${access.policy.label} cannot run Central Fraud Registry lookups` : 'Check consortium account'}
                className="flex items-center justify-center w-7 h-7 rounded-md
                  bg-[#00579C]/15 border border-[#00579C]/30 text-[#00579C]
                  hover:bg-[#00579C]/25 disabled:opacity-30 disabled:cursor-not-allowed transition-colors"
              >
                {checkAccount.isPending ? (
                  <Loader2 className="w-3.5 h-3.5 animate-spin" />
                ) : (
                  <Search className="w-3.5 h-3.5" />
                )}
              </button>
            </div>
            {!canCheckConsortium && (
              <div className="rounded-md border border-[#DA251C]/25 bg-[#DA251C]/10 px-3 py-2 text-[10px] font-semibold leading-5 text-[#DA251C]">
                {access.policy.label} can view shared intelligence, but account-level CFR checks are locked by RBAC.
              </div>
            )}

            {checkAccount.data && (
              <div className="animate-slide-up">
                {checkAccount.data.flagged ? (
                  <div className="bg-[#DA251C]/10 border border-[#DA251C]/20 rounded-md p-3 space-y-2">
                    <div className="flex items-center gap-2">
                      <ShieldAlert className="w-4 h-4 text-[#DA251C]" />
                      <span className="text-[11px] font-semibold text-[#DA251C]">Account Flagged</span>
                      <span className="text-[9px] text-text-muted ml-auto">
                        {checkAccount.data.alert_count} alert{checkAccount.data.alert_count !== 1 ? 's' : ''}
                      </span>
                    </div>
                    {checkAccount.data.alerts.map((alert) => (
                      <div key={alert.alert_id} className="text-[9px] text-text-secondary bg-bg-deep/50 rounded px-2 py-1">
                        <span className="font-mono">{alert.originating_bank}</span> — {alert.fraud_type} ({alert.severity})
                      </div>
                    ))}
                  </div>
                ) : (
                  <div className="bg-[#00579C]/10 border border-[#00579C]/20 rounded-md p-3 flex items-center gap-2">
                    <CheckCircle className="w-4 h-4 text-[#00579C]" />
                    <span className="text-[11px] text-[#00579C]">Account not flagged in consortium</span>
                  </div>
                )}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  )
}

/* ── Small stat card ── */
function StatCard({ icon: Icon, label, value, color }: { icon: React.ComponentType<{ className?: string }>; label: string; value: number; color: string }) {
  return (
    <div className="bg-bg-surface/40 rounded-md border border-border-subtle px-3 py-2 flex items-center gap-2">
      <Icon className={cn('w-4 h-4', color)} />
      <div>
        <p className="text-[12px] font-mono font-semibold text-text-primary">{value}</p>
        <p className="text-[8px] text-text-muted uppercase tracking-wider">{label}</p>
      </div>
    </div>
  )
}

// ============================================================================
// Compliance & Regulatory Intelligence Page
// Fraud Intelligence Layer: Rule Engine, STR/CTR/FMR, CFR-RBI, AML Stages,
// FIU-IND, Investigation Management, Mule Detection, Victim Fund Tracing
// ============================================================================

import { useState, useCallback } from 'react'
import { useQuery, useQueryClient } from '@tanstack/react-query'
import {
  Shield, ShieldCheck, ShieldAlert, FileCheck, Scale, AlertTriangle,
  Landmark, Search, Eye, Layers, GitBranch, Users, Fingerprint,
  Activity,
  ToggleLeft, ToggleRight, Gavel, FileText,
  Network, Banknote, UserX, Route, Brain, Radio,
  LockKeyhole,
} from 'lucide-react'
import { cn } from '@/lib/utils'
import { useRoleAccess } from '@/hooks/use-rbac'
import {
  fetchRules, fetchRuleStats, toggleRule,
  fetchGateStats, fetchReports, fetchCFRStats,
  fetchAMLStats, fetchFIUStats, fetchFIUHighRisk,
  fetchInvestigationStats, fetchMuleChains, fetchMuleStats,
  fetchSuspectedMules, fetchVictimStats, fetchAnomalyStats,
  fetchClusters, fetchIntermediaries,
} from '@/lib/api-client'
import type {
  RuleInfo, GateStatsResponse,
  FIUStatsResponse, InvestigationStatsResponse,
} from '@/lib/types'
import type { Permission } from '@/lib/rbac'

// ============================================================================
// Constants
// ============================================================================

const REFETCH_INTERVAL = 8_000

const QUERY_PERMISSIONS: Record<string, Permission[]> = {
  rules: ['rules:toggle', 'analytics:view', 'audit:review'],
  gate: ['alert:hold', 'analytics:view', 'audit:review'],
  reports: ['regulatory:file', 'audit:review', 'fraud:fmr:file'],
  cfr: ['cfr:check', 'regulatory:file', 'audit:review'],
  investigation: ['case:view', 'case:decide', 'regulatory:file', 'fraud:fmr:file', 'audit:review'],
  aml: ['aml:cdd', 'aml:str:draft', 'regulatory:file', 'audit:review'],
  fiu: ['regulatory:file', 'fiu:disseminate', 'aml:str:authorize', 'audit:review'],
  mule: ['case:view', 'aml:cdd', 'analytics:view'],
  anomaly: ['model:feedback', 'analytics:view', 'risk:view', 'audit:review'],
  centrality: ['case:view', 'analytics:view', 'aml:cdd'],
  victim: ['case:view', 'customer:contact', 'regulatory:file', 'analytics:view'],
}

const ACCENT = {
  emerald: 'text-[#00579C] bg-[#00579C]/10 border-[#00579C]/20',
  cyan: 'text-[#00579C] bg-[#00579C]/10 border-[#00579C]/20',
  amber: 'text-[#DA251C] bg-[#DA251C]/10 border-[#DA251C]/20',
  rose: 'text-[#DA251C] bg-[#DA251C]/10 border-[#DA251C]/20',
  violet: 'text-[#00579C] bg-[#00579C]/10 border-[#00579C]/20',
  blue: 'text-[#00579C] bg-[#00579C]/10 border-[#00579C]/20',
  orange: 'text-[#DA251C] bg-[#DA251C]/10 border-[#DA251C]/20',
  pink: 'text-[#00579C] bg-[#00579C]/10 border-[#00579C]/20',
  indigo: 'text-[#00579C] bg-[#00579C]/10 border-[#00579C]/20',
  teal: 'text-[#00579C] bg-[#00579C]/10 border-[#00579C]/20',
}

type MetricSource = Record<string, unknown> | undefined

function usePermissionGroup(permissions: Permission[]) {
  const access = useRoleAccess()
  return {
    access,
    currentRole: access.currentRole,
    enabled: permissions.some((permission) => access.can(permission)),
  }
}

function RoleLockedState({ area }: { area: string }) {
  const access = useRoleAccess()
  return (
    <div className="flex min-h-[140px] flex-col items-center justify-center gap-2 rounded-md border border-[#d7e3f1] bg-[#f4f8fc] px-4 py-5 text-center">
      <LockKeyhole className="h-5 w-5 text-[#617189]" />
      <p className="text-[10px] font-semibold leading-5 text-[#4b5d76]">
        {area} is role-gated for {access.policy.label}; backend requests are not sent for this role.
      </p>
    </div>
  )
}

function metricNumber(source: MetricSource, keys: string[]): number | null {
  for (const key of keys) {
    const value = Number(source?.[key])
    if (Number.isFinite(value)) return value
  }
  return null
}

function alertTotal(source: MetricSource, keys: string[]): number | null {
  const direct = metricNumber(source, ['alerts_raised'])
  if (direct != null) return direct
  const values = keys
    .map((key) => Number(source?.[key]))
    .filter(Number.isFinite)
  return values.length > 0 ? values.reduce((sum, value) => sum + value, 0) : null
}

// ============================================================================
// Metric Card
// ============================================================================

function MetricCard({ icon: Icon, label, value, accent, pulse, sub }: {
  icon: typeof Activity
  label: string
  value: string | number
  accent: string
  pulse?: boolean
  sub?: string
}) {
  return (
    <div className={cn(
      'relative flex items-center gap-2.5 px-3 py-2.5 rounded-lg border backdrop-blur-sm',
      'bg-bg-surface/70 border-border-subtle/40 hover:border-border-subtle/70',
      'transition-all duration-300 group min-w-0',
    )}>
      <div className={cn(
        'flex items-center justify-center w-8 h-8 rounded-md border shrink-0',
        accent,
      )}>
        <Icon className={cn('w-4 h-4', pulse && 'animate-pulse')} />
      </div>
      <div className="min-w-0">
        <p className="text-[10px] text-text-muted truncate uppercase tracking-wider">{label}</p>
        <p className="text-sm font-semibold text-text-primary tabular-nums">{value}</p>
        {sub && <p className="text-[9px] text-text-muted/70 truncate">{sub}</p>}
      </div>
    </div>
  )
}

// ============================================================================
// Panel Shell
// ============================================================================

function Panel({ title, icon: Icon, accent, children, className }: {
  title: string
  icon: typeof Activity
  accent: string
  children: React.ReactNode
  className?: string
}) {
  return (
    <div className={cn(
      'flex flex-col rounded-lg border backdrop-blur-sm overflow-hidden',
      'bg-bg-surface/60 border-border-subtle/40',
      className,
    )}>
      <div className="flex items-center gap-2 px-3.5 py-2.5 border-b border-border-subtle/30 bg-bg-elevated/30 shrink-0">
        <div className={cn(
          'flex items-center justify-center w-6 h-6 rounded-md border',
          accent,
        )}>
          <Icon className="w-3.5 h-3.5" />
        </div>
        <h3 className="text-xs font-semibold text-text-primary tracking-wide">{title}</h3>
      </div>
      <div className="flex-1 overflow-auto custom-scrollbar p-3">
        {children}
      </div>
    </div>
  )
}

// ============================================================================
// Rule Engine Panel
// ============================================================================

function RuleEnginePanel() {
  const { access, currentRole, enabled: canViewRules } = usePermissionGroup(QUERY_PERMISSIONS.rules)
  const canToggleRules = access.can('rules:toggle')
  const roleLabel = access.policy.label
  const queryClient = useQueryClient()
  const { data: rulesData } = useQuery({
    queryKey: ['fraud', 'rules', currentRole],
    queryFn: fetchRules,
    enabled: canViewRules,
    refetchInterval: canViewRules ? REFETCH_INTERVAL : false,
  })
  const { data: ruleStats } = useQuery({
    queryKey: ['fraud', 'rules', 'stats', currentRole],
    queryFn: fetchRuleStats,
    enabled: canViewRules,
    refetchInterval: canViewRules ? REFETCH_INTERVAL : false,
  })

  const [toggling, setToggling] = useState<string | null>(null)
  const getRuleId = useCallback((rule: RuleInfo) => rule.rule_id || rule.id || rule.name, [])

  const handleToggle = useCallback(async (rule: RuleInfo) => {
    if (!canToggleRules) return
    const id = getRuleId(rule)
    setToggling(id)
    try {
      await toggleRule(id, !rule.enabled)
      queryClient.invalidateQueries({ queryKey: ['fraud', 'rules'] })
    } finally {
      setToggling(null)
    }
  }, [canToggleRules, getRuleId, queryClient])

  const rules = rulesData?.rules ?? []

  return (
    <Panel title="Rule Engine" icon={Scale} accent={ACCENT.emerald}>
      {!canViewRules ? <RoleLockedState area="Rule engine metrics" /> : (
      <>
      {/* Stats strip */}
      <div className="flex gap-2 mb-3">
        <div className="flex-1 rounded-md bg-bg-elevated/50 border border-border-subtle/25 px-2.5 py-1.5 text-center">
          <p className="text-[9px] text-text-muted uppercase tracking-wider">Evaluations</p>
          <p className="text-xs font-bold text-[#00579C] tabular-nums">{ruleStats?.total_evaluations ?? 0}</p>
        </div>
        <div className="flex-1 rounded-md bg-bg-elevated/50 border border-border-subtle/25 px-2.5 py-1.5 text-center">
          <p className="text-[9px] text-text-muted uppercase tracking-wider">Rules Active</p>
          <p className="text-xs font-bold text-[#00579C] tabular-nums">{rules.filter(r => r.enabled).length}/{rules.length}</p>
        </div>
        <div className="flex-1 rounded-md bg-bg-elevated/50 border border-border-subtle/25 px-2.5 py-1.5 text-center">
          <p className="text-[9px] text-text-muted uppercase tracking-wider">Avg Eval</p>
          <p className="text-xs font-bold text-[#00579C] tabular-nums">{(ruleStats?.last_evaluation_ms ?? 0).toFixed(1)}ms</p>
        </div>
      </div>

      {/* Rules list */}
      <div className="space-y-1.5">
        {rules.map(rule => {
          const id = getRuleId(rule)
          return (
          <div
            key={id}
            className={cn(
              'flex items-center gap-2 px-2.5 py-2 rounded-md border transition-all duration-200',
              rule.enabled
                ? 'bg-[#00579C]/5 border-[#00579C]/15 hover:border-[#00579C]/30'
                : 'bg-bg-elevated/30 border-border-subtle/20 opacity-60 hover:opacity-80',
            )}
          >
            <button
              onClick={() => handleToggle(rule)}
              disabled={toggling === id || !canToggleRules}
              title={!canToggleRules ? `${roleLabel} cannot toggle fraud rules` : 'Toggle fraud rule'}
              className="shrink-0 transition-transform active:scale-90"
            >
              {rule.enabled
                ? <ToggleRight className="w-5 h-5 text-[#00579C]" />
                : <ToggleLeft className="w-5 h-5 text-text-muted" />
              }
            </button>
            <div className="min-w-0 flex-1">
              <p className="text-[11px] font-medium text-text-primary truncate">{rule.name}</p>
              <p className="text-[9px] text-text-muted truncate">{rule.description || `Threshold: ${rule.threshold}`}</p>
            </div>
            <span className={cn(
              'shrink-0 text-[9px] font-mono px-1.5 py-0.5 rounded border',
              rule.category === 'velocity' ? ACCENT.amber :
              rule.category === 'amount' ? ACCENT.rose :
              rule.category === 'pattern' ? ACCENT.violet :
              ACCENT.cyan,
            )}>
              {rule.category}
            </span>
          </div>
          )
        })}
        {rules.length === 0 && (
          <p className="text-[10px] text-text-muted text-center py-4">No rules configured</p>
        )}
      </div>
      </>
      )}
    </Panel>
  )
}

// ============================================================================
// Regulatory Reports Panel
// ============================================================================

function RegulatoryReportsPanel() {
  const { currentRole, enabled: canViewReports } = usePermissionGroup(QUERY_PERMISSIONS.reports)
  const { data: reportsData } = useQuery({
    queryKey: ['fraud', 'reports', currentRole],
    queryFn: () => fetchReports(undefined, 30),
    enabled: canViewReports,
    refetchInterval: canViewReports ? REFETCH_INTERVAL : false,
  })

  const reports = reportsData?.reports ?? []

  const typeIcon = (t: string) => {
    if (t === 'STR') return <AlertTriangle className="w-3 h-3 text-[#DA251C]" />
    if (t === 'CTR') return <Banknote className="w-3 h-3 text-[#DA251C]" />
    if (t === 'FMR') return <FileText className="w-3 h-3 text-[#00579C]" />
    return <FileCheck className="w-3 h-3 text-text-muted" />
  }

  const statusColor = (s: string) => {
    if (s === 'filed') return 'text-[#00579C] bg-[#00579C]/10 border-[#00579C]/20'
    if (s === 'pending') return 'text-[#DA251C] bg-[#DA251C]/10 border-[#DA251C]/20'
    return 'text-[#DA251C] bg-[#DA251C]/10 border-[#DA251C]/20'
  }

  return (
    <Panel title="Regulatory Filings" icon={FileCheck} accent={ACCENT.amber}>
      {!canViewReports ? <RoleLockedState area="Regulatory filings" /> : (
      <>
      {/* Summary counts */}
      <div className="flex gap-2 mb-3">
        {(['STR', 'CTR', 'FMR'] as const).map(type => {
          const count = reports.filter(r => r.report_type === type).length
          return (
            <div key={type} className="flex-1 rounded-md bg-bg-elevated/50 border border-border-subtle/25 px-2.5 py-1.5 text-center">
              <p className="text-[9px] text-text-muted uppercase tracking-wider">{type}</p>
              <p className={cn('text-xs font-bold tabular-nums',
                type === 'STR' ? 'text-[#DA251C]' : type === 'CTR' ? 'text-[#DA251C]' : 'text-[#00579C]'
              )}>{count}</p>
            </div>
          )
        })}
      </div>

      <div className="space-y-1">
        {reports.slice(0, 20).map(report => (
          <div
            key={report.report_id}
            className="flex items-center gap-2 px-2.5 py-1.5 rounded-md bg-bg-elevated/20 border border-border-subtle/15 hover:border-border-subtle/30 transition-all"
          >
            {typeIcon(report.report_type)}
            <span className="text-[10px] font-mono text-text-secondary min-w-[32px]">{report.report_type}</span>
            <span className="text-[10px] text-text-muted flex-1 truncate font-mono">{report.report_id.slice(0, 12)}</span>
            <span className={cn('text-[9px] font-medium px-1.5 py-0.5 rounded border', statusColor(report.status))}>
              {report.status}
            </span>
          </div>
        ))}
        {reports.length === 0 && (
          <p className="text-[10px] text-text-muted text-center py-4">No regulatory filings yet</p>
        )}
      </div>
      </>
      )}
    </Panel>
  )
}

// ============================================================================
// AML Stage Detection Panel
// ============================================================================

function AMLStagePanel() {
  const { currentRole, enabled: canViewAml } = usePermissionGroup(QUERY_PERMISSIONS.aml)
  const { enabled: canViewMule } = usePermissionGroup(QUERY_PERMISSIONS.mule)
  const { data: amlStats } = useQuery({
    queryKey: ['fraud', 'aml', 'stats', currentRole],
    queryFn: fetchAMLStats,
    enabled: canViewAml,
    refetchInterval: canViewAml ? REFETCH_INTERVAL : false,
  })
  const { data: muleStats } = useQuery({
    queryKey: ['fraud', 'mule', 'stats', currentRole],
    queryFn: fetchMuleStats,
    enabled: canViewMule,
    refetchInterval: canViewMule ? REFETCH_INTERVAL : false,
  })
  const { data: chainsData } = useQuery({
    queryKey: ['fraud', 'mule', 'chains', currentRole],
    queryFn: fetchMuleChains,
    enabled: canViewMule,
    refetchInterval: canViewMule ? REFETCH_INTERVAL : false,
  })

  const chainStats = muleStats?.chain_detector
  const placementEvals = metricNumber(amlStats?.placement, ['total_evaluations', 'evaluations'])
  const placementAlerts = alertTotal(amlStats?.placement, ['cash_alerts', 'structuring_alerts', 'multi_channel_alerts', 'round_amount_alerts'])
  const layeringEvals = metricNumber(chainStats, ['total_scans'])
  const layeringAlerts = metricNumber(chainStats, ['total_chains_detected']) ?? chainsData?.count ?? null
  const integrationEvals = metricNumber(amlStats?.integration, ['total_evaluations', 'evaluations'])
  const integrationAlerts = alertTotal(amlStats?.integration, ['asset_purchase_alerts', 'investment_alerts', 'rapid_withdrawal_alerts', 'round_trip_alerts'])

  const stages = [
    {
      name: 'Placement',
      desc: 'Initial deposit of illicit funds',
      icon: Layers,
      color: 'text-[#DA251C]',
      bg: 'bg-[#DA251C]/10 border-[#DA251C]/20',
      evals: placementEvals,
      alerts: placementAlerts,
    },
    {
      name: 'Layering',
      desc: 'Complex fund movement via shell entities',
      icon: GitBranch,
      color: 'text-[#DA251C]',
      bg: 'bg-[#DA251C]/10 border-[#DA251C]/20',
      evals: layeringEvals,
      alerts: layeringAlerts,
      note: layeringEvals == null
        ? 'Backend chain detector is configured; scan counters are not exposed by this endpoint.'
        : null,
    },
    {
      name: 'Integration',
      desc: 'Re-entry of laundered funds into economy',
      icon: Network,
      color: 'text-[#00579C]',
      bg: 'bg-[#00579C]/10 border-[#00579C]/20',
      evals: integrationEvals,
      alerts: integrationAlerts,
    },
  ]

  return (
    <Panel title="AML Stage Detection" icon={Layers} accent={ACCENT.rose}>
      {!canViewAml && !canViewMule ? <RoleLockedState area="AML and mule-chain metrics" /> : (
      <div className="space-y-2">
        {stages.map(stage => {
          const alertRate = stage.evals == null || stage.alerts == null
            ? null
            : stage.evals > 0
              ? ((stage.alerts / stage.evals) * 100).toFixed(1)
              : '0.0'
          return (
            <div key={stage.name} className={cn(
              'rounded-md border px-3 py-2.5 transition-all',
              stage.bg,
            )}>
              <div className="flex items-center gap-2 mb-1.5">
                <stage.icon className={cn('w-4 h-4', stage.color)} />
                <span className="text-[11px] font-semibold text-text-primary">{stage.name}</span>
                {(stage.alerts ?? 0) > 0 && (
                  <span className="ml-auto flex items-center gap-1 text-[9px] text-[#DA251C] font-bold">
                    <Radio className="w-3 h-3 animate-pulse" /> {stage.alerts} alerts
                  </span>
                )}
              </div>
              <p className="text-[9px] text-text-muted mb-1.5">{stage.desc}</p>
              <div className="flex flex-wrap gap-3 text-[9px]">
                <span className="text-text-muted">
                  Evaluations: <span className="text-text-secondary font-medium">{stage.evals ?? 'not exposed'}</span>
                </span>
                <span className="text-text-muted">
                  Alert Rate:{' '}
                  <span className={cn('font-medium', alertRate != null && parseFloat(alertRate) > 10 ? 'text-[#DA251C]' : 'text-[#00579C]')}>
                    {alertRate == null ? 'not exposed' : `${alertRate}%`}
                  </span>
                </span>
              </div>
              {stage.note && (
                <p className="mt-1.5 text-[8px] leading-relaxed text-text-muted/75">{stage.note}</p>
              )}
            </div>
          )
        })}
      </div>
      )}
    </Panel>
  )
}

// ============================================================================
// CFR-RBI Registry Panel
// ============================================================================

function CFRRegistryPanel() {
  const { currentRole, enabled: canViewCfr } = usePermissionGroup(QUERY_PERMISSIONS.cfr)
  const { data: cfrStats } = useQuery({
    queryKey: ['fraud', 'cfr', 'stats', currentRole],
    queryFn: fetchCFRStats,
    enabled: canViewCfr,
    refetchInterval: canViewCfr ? REFETCH_INTERVAL : false,
  })

  const registry = cfrStats?.registry
  const categories = registry?.categories ?? {}

  return (
    <Panel title="Central Fraud Registry (RBI)" icon={Landmark} accent={ACCENT.indigo}>
      {!canViewCfr ? <RoleLockedState area="Central Fraud Registry metrics" /> : (
      <>
      <div className="grid grid-cols-2 gap-2 mb-3">
        <div className="rounded-md bg-bg-elevated/50 border border-border-subtle/25 px-2.5 py-1.5 text-center">
          <p className="text-[9px] text-text-muted uppercase tracking-wider">Records</p>
          <p className="text-xs font-bold text-[#00579C] tabular-nums">{registry?.total_records ?? 0}</p>
        </div>
        <div className="rounded-md bg-bg-elevated/50 border border-border-subtle/25 px-2.5 py-1.5 text-center">
          <p className="text-[9px] text-text-muted uppercase tracking-wider">Accounts</p>
          <p className="text-xs font-bold text-[#00579C] tabular-nums">{registry?.unique_accounts ?? 0}</p>
        </div>
      </div>

      {/* Category breakdown */}
      {Object.entries(categories).length > 0 && (
        <div className="space-y-1">
          <p className="text-[9px] text-text-muted uppercase tracking-wider mb-1.5">Fraud Categories</p>
          {Object.entries(categories).map(([cat, count]) => (
            <div key={cat} className="flex items-center gap-2 px-2 py-1.5 rounded bg-bg-elevated/30 border border-border-subtle/15">
              <div className="w-1.5 h-1.5 rounded-full bg-[#00579C] shrink-0" />
              <span className="text-[10px] text-text-secondary flex-1 truncate capitalize">{cat.replace(/_/g, ' ')}</span>
              <span className="text-[10px] font-mono font-bold text-text-primary tabular-nums">{count as number}</span>
            </div>
          ))}
        </div>
      )}
      {Object.entries(categories).length === 0 && (
        <p className="text-[10px] text-text-muted text-center py-3">Registry empty — no reports filed yet</p>
      )}
      </>
      )}
    </Panel>
  )
}

// ============================================================================
// FIU Intelligence Panel
// ============================================================================

function FIUIntelligencePanel() {
  const { currentRole, enabled: canViewFiu } = usePermissionGroup(QUERY_PERMISSIONS.fiu)
  const { data: fiuStats } = useQuery({
    queryKey: ['fraud', 'fiu', 'stats', currentRole],
    queryFn: fetchFIUStats,
    enabled: canViewFiu,
    refetchInterval: canViewFiu ? REFETCH_INTERVAL : false,
  })
  const { data: highRisk } = useQuery({
    queryKey: ['fraud', 'fiu', 'high-risk', currentRole],
    queryFn: fetchFIUHighRisk,
    enabled: canViewFiu,
    refetchInterval: canViewFiu ? REFETCH_INTERVAL : false,
  })

  const stats = fiuStats ?? {} as FIUStatsResponse

  return (
    <Panel title="FIU-IND Intelligence" icon={Eye} accent={ACCENT.teal}>
      {!canViewFiu ? <RoleLockedState area="FIU-IND intelligence" /> : (
      <>
      {/* KPI tiles */}
      <div className="grid grid-cols-3 gap-1.5 mb-3">
        {[
          { label: 'STR Collected', val: stats.str_collected ?? 0, color: 'text-[#DA251C]' },
          { label: 'Alerts', val: stats.alerts_collected ?? 0, color: 'text-[#DA251C]' },
          { label: 'Packages', val: stats.packages_prepared ?? 0, color: 'text-[#00579C]' },
          { label: 'Disseminated', val: stats.packages_disseminated ?? 0, color: 'text-[#00579C]' },
          { label: 'High Risk', val: stats.high_risk_accounts ?? 0, color: 'text-[#DA251C]' },
          { label: 'Total Entries', val: stats.total_entries ?? 0, color: 'text-[#00579C]' },
        ].map(kpi => (
          <div key={kpi.label} className="rounded-md bg-bg-elevated/50 border border-border-subtle/25 px-1.5 py-1.5 text-center">
            <p className="text-[8px] text-text-muted uppercase tracking-wider leading-tight">{kpi.label}</p>
            <p className={cn('text-xs font-bold tabular-nums', kpi.color)}>{kpi.val}</p>
          </div>
        ))}
      </div>

      {/* High-Risk accounts */}
      {(highRisk?.count ?? 0) > 0 && (
        <div>
          <p className="text-[9px] text-text-muted uppercase tracking-wider mb-1.5 flex items-center gap-1">
            <ShieldAlert className="w-3 h-3 text-[#DA251C]" /> High-Risk Accounts
          </p>
          <div className="space-y-1 max-h-[140px] overflow-auto custom-scrollbar">
            {highRisk!.high_risk_accounts.map(acc => (
              <div key={acc} className="flex items-center gap-2 px-2 py-1 rounded bg-[#DA251C]/5 border border-[#DA251C]/10">
                <Fingerprint className="w-3 h-3 text-[#DA251C] shrink-0" />
                <span className="text-[10px] font-mono text-[#DA251C] truncate">{acc}</span>
              </div>
            ))}
          </div>
        </div>
      )}
      </>
      )}
    </Panel>
  )
}

// ============================================================================
// Investigation Management Panel
// ============================================================================

function InvestigationPanel() {
  const { currentRole, enabled: canViewInvestigation } = usePermissionGroup(QUERY_PERMISSIONS.investigation)
  const { data: invStats } = useQuery({
    queryKey: ['fraud', 'investigation', 'stats', currentRole],
    queryFn: fetchInvestigationStats,
    enabled: canViewInvestigation,
    refetchInterval: canViewInvestigation ? REFETCH_INTERVAL : false,
  })

  const stats = invStats ?? {} as InvestigationStatsResponse

  const stages = [
    { label: 'Open Cases', val: stats.open_cases ?? 0, icon: Search, color: 'text-[#00579C]', bg: 'bg-[#00579C]/10 border-[#00579C]/15' },
    { label: 'Referred to LEA', val: stats.referred_cases ?? 0, icon: Gavel, color: 'text-[#DA251C]', bg: 'bg-[#DA251C]/10 border-[#DA251C]/15' },
    { label: 'Legal Proceedings', val: stats.legal_proceedings ?? 0, icon: Scale, color: 'text-[#DA251C]', bg: 'bg-[#DA251C]/10 border-[#DA251C]/15' },
  ]

  return (
    <Panel title="Investigation Management" icon={Gavel} accent={ACCENT.orange}>
      {!canViewInvestigation ? <RoleLockedState area="Investigation management" /> : (
      <>
      <div className="rounded-md bg-bg-elevated/50 border border-border-subtle/25 px-3 py-2 mb-3 text-center">
        <p className="text-[9px] text-text-muted uppercase tracking-wider">Total Cases</p>
        <p className="text-lg font-bold text-[#DA251C] tabular-nums">{stats.total_cases ?? 0}</p>
      </div>

      <div className="space-y-2">
        {stages.map(stage => (
          <div key={stage.label} className={cn('flex items-center gap-2.5 px-3 py-2 rounded-md border', stage.bg)}>
            <stage.icon className={cn('w-4 h-4 shrink-0', stage.color)} />
            <span className="text-[10px] text-text-secondary flex-1">{stage.label}</span>
            <span className={cn('text-sm font-bold tabular-nums', stage.color)}>{stage.val}</span>
          </div>
        ))}
      </div>

      {(stats.total_fraud_amount_paisa ?? 0) > 0 && (
        <div className="mt-3 rounded-md bg-[#DA251C]/5 border border-[#DA251C]/10 px-3 py-2 text-center">
          <p className="text-[9px] text-text-muted uppercase tracking-wider">Total Fraud Amount</p>
          <p className="text-sm font-bold text-[#DA251C] tabular-nums">
            ₹{((stats.total_fraud_amount_paisa ?? 0) / 100).toLocaleString('en-IN', { minimumFractionDigits: 2 })}
          </p>
        </div>
      )}
      </>
      )}
    </Panel>
  )
}

// ============================================================================
// Mule Detection Panel
// ============================================================================

function MuleDetectionPanel() {
  const { currentRole, enabled: canViewMule } = usePermissionGroup(QUERY_PERMISSIONS.mule)
  const { data: muleStats } = useQuery({
    queryKey: ['fraud', 'mule', 'stats', currentRole],
    queryFn: fetchMuleStats,
    enabled: canViewMule,
    refetchInterval: canViewMule ? REFETCH_INTERVAL : false,
  })
  const { data: chainsData } = useQuery({
    queryKey: ['fraud', 'mule', 'chains', currentRole],
    queryFn: fetchMuleChains,
    enabled: canViewMule,
    refetchInterval: canViewMule ? REFETCH_INTERVAL : false,
  })
  const { data: suspectedData } = useQuery({
    queryKey: ['fraud', 'mule', 'suspected', currentRole],
    queryFn: () => fetchSuspectedMules(0.5),
    enabled: canViewMule,
    refetchInterval: canViewMule ? REFETCH_INTERVAL : false,
  })

  const chains = chainsData?.chains ?? []
  const mules = suspectedData?.mules ?? []
  const accountStats = muleStats?.account_scorer
  const detectedChainCount = chainsData?.count ?? chains.length

  return (
    <Panel title="Mule Detection (Carbanak)" icon={UserX} accent={ACCENT.pink}>
      {!canViewMule ? <RoleLockedState area="Mule detection" /> : (
      <>
      {/* KPI strip */}
      <div className="flex gap-2 mb-3">
        <div className="flex-1 rounded-md bg-bg-elevated/50 border border-border-subtle/25 px-2 py-1.5 text-center">
          <p className="text-[8px] text-text-muted uppercase tracking-wider">Chains</p>
          <p className="text-xs font-bold text-[#00579C] tabular-nums">{detectedChainCount}</p>
        </div>
        <div className="flex-1 rounded-md bg-bg-elevated/50 border border-border-subtle/25 px-2 py-1.5 text-center">
          <p className="text-[8px] text-text-muted uppercase tracking-wider">Suspected</p>
          <p className="text-xs font-bold text-[#DA251C] tabular-nums">{accountStats?.suspected_mules ?? 0}</p>
        </div>
        <div className="flex-1 rounded-md bg-bg-elevated/50 border border-border-subtle/25 px-2 py-1.5 text-center">
          <p className="text-[8px] text-text-muted uppercase tracking-wider">Scored</p>
          <p className="text-xs font-bold text-[#00579C] tabular-nums">{accountStats?.total_scored ?? 0}</p>
        </div>
      </div>

      {/* Detected chains */}
      {chains.length > 0 && (
        <div className="mb-3">
          <p className="text-[9px] text-text-muted uppercase tracking-wider mb-1.5 flex items-center gap-1">
            <Route className="w-3 h-3 text-[#00579C]" /> Detected Chains
          </p>
          <div className="space-y-1.5 max-h-[120px] overflow-auto custom-scrollbar">
            {chains.slice(0, 8).map((chain, i) => (
              <div key={i} className="flex items-center gap-2 px-2 py-1.5 rounded-md bg-[#00579C]/5 border border-[#00579C]/10">
                <GitBranch className="w-3.5 h-3.5 text-[#00579C] shrink-0" />
                <div className="min-w-0 flex-1">
                  <p className="text-[10px] font-mono text-text-secondary truncate">
                    {chain.origin_node.slice(0, 8)} → {chain.terminal_node.slice(0, 8)}
                  </p>
                  <p className="text-[9px] text-text-muted">
                    {chain.chain_length} hops · ₹{(chain.total_amount_paisa / 100).toLocaleString('en-IN')}
                  </p>
                </div>
                <span className="text-[9px] font-mono text-[#00579C]/70 shrink-0">{chain.detection_time_ms.toFixed(0)}ms</span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Suspected mule accounts */}
      {mules.length > 0 && (
        <div>
          <p className="text-[9px] text-text-muted uppercase tracking-wider mb-1.5 flex items-center gap-1">
            <UserX className="w-3 h-3 text-[#DA251C]" /> Suspected Accounts
          </p>
          <div className="space-y-1 max-h-[120px] overflow-auto custom-scrollbar">
            {mules.slice(0, 8).map(mule => (
              <div key={mule.account_id} className="flex items-center gap-2 px-2 py-1.5 rounded-md bg-bg-elevated/30 border border-border-subtle/15">
                <Fingerprint className="w-3 h-3 text-[#DA251C] shrink-0" />
                <span className="text-[10px] font-mono text-text-secondary truncate flex-1">{mule.account_id.slice(0, 12)}</span>
                <div className="flex gap-1 shrink-0">
                  {mule.indicators.newly_opened > 0 && <span className="w-1.5 h-1.5 rounded-full bg-[#DA251C]" title="Newly opened" />}
                  {mule.indicators.high_frequency > 0 && <span className="w-1.5 h-1.5 rounded-full bg-[#DA251C]" title="High frequency" />}
                  {mule.indicators.rapid_forward > 0 && <span className="w-1.5 h-1.5 rounded-full bg-[#00579C]" title="Rapid forward" />}
                  {mule.indicators.large_cashout > 0 && <span className="w-1.5 h-1.5 rounded-full bg-[#00579C]" title="Large cashout" />}
                </div>
                <span className={cn(
                  'text-[9px] font-bold tabular-nums shrink-0',
                  mule.mule_score > 0.8 ? 'text-[#DA251C]' : mule.mule_score > 0.6 ? 'text-[#DA251C]' : 'text-[#00579C]',
                )}>{(mule.mule_score * 100).toFixed(0)}%</span>
              </div>
            ))}
          </div>
        </div>
      )}

      {chains.length === 0 && mules.length === 0 && (
        <p className="text-[10px] text-text-muted text-center py-4">No mule activity detected yet</p>
      )}
      </>
      )}
    </Panel>
  )
}

// ============================================================================
// Anomaly Detection Panel
// ============================================================================

function AnomalyDetectionPanel() {
  const { currentRole, enabled: canViewAnomaly } = usePermissionGroup(QUERY_PERMISSIONS.anomaly)
  const { data: anomalyStats } = useQuery({
    queryKey: ['fraud', 'anomaly', 'stats', currentRole],
    queryFn: fetchAnomalyStats,
    enabled: canViewAnomaly,
    refetchInterval: canViewAnomaly ? REFETCH_INTERVAL : false,
  })
  const { data: clustersData } = useQuery({
    queryKey: ['fraud', 'clusters', currentRole],
    queryFn: fetchClusters,
    enabled: canViewAnomaly,
    refetchInterval: canViewAnomaly ? REFETCH_INTERVAL : false,
  })

  const iforest = anomalyStats?.isolation_forest
  const autoenc = anomalyStats?.autoencoder
  const clusters = clustersData?.clusters ?? []

  return (
    <Panel title="Anomaly Detection" icon={Brain} accent={ACCENT.violet}>
      {!canViewAnomaly ? <RoleLockedState area="Anomaly detection" /> : (
      <>
      {/* Model stats */}
      <div className="space-y-1.5 mb-3">
        {[
          { label: 'Isolation Forest', scored: iforest?.total_scored ?? 0, anomalies: iforest?.anomalies_detected ?? 0, color: 'text-[#00579C]', bg: 'bg-[#00579C]/10 border-[#00579C]/15' },
          { label: 'Autoencoder', scored: autoenc?.total_scored ?? 0, anomalies: autoenc?.anomalies_detected ?? 0, color: 'text-[#00579C]', bg: 'bg-[#00579C]/10 border-[#00579C]/15' },
        ].map(model => (
          <div key={model.label} className={cn('flex items-center gap-2.5 px-3 py-2 rounded-md border', model.bg)}>
            <Brain className={cn('w-4 h-4 shrink-0', model.color)} />
            <div className="flex-1 min-w-0">
              <p className="text-[10px] font-medium text-text-primary">{model.label}</p>
              <p className="text-[9px] text-text-muted">
                {model.scored} scored · <span className={model.anomalies > 0 ? 'text-[#DA251C] font-medium' : ''}>
                  {model.anomalies} anomalies
                </span>
              </p>
            </div>
          </div>
        ))}
      </div>

      {/* Suspicious clusters */}
      {clusters.length > 0 && (
        <div>
          <p className="text-[9px] text-text-muted uppercase tracking-wider mb-1.5 flex items-center gap-1">
            <Users className="w-3 h-3 text-[#00579C]" /> Suspicious Communities
          </p>
          <div className="space-y-1 max-h-[130px] overflow-auto custom-scrollbar">
            {clusters.slice(0, 8).map(cluster => (
              <div key={cluster.community_id} className="flex items-center gap-2 px-2 py-1.5 rounded-md bg-bg-elevated/30 border border-border-subtle/15">
                <Network className="w-3 h-3 text-[#00579C] shrink-0" />
                <span className="text-[10px] text-text-secondary flex-1">
                  Community #{cluster.community_id} · {cluster.node_count} nodes
                </span>
                <span className={cn(
                  'text-[9px] font-bold tabular-nums',
                  cluster.anomaly_score > 0.7 ? 'text-[#DA251C]' : cluster.anomaly_score > 0.4 ? 'text-[#DA251C]' : 'text-[#00579C]',
                )}>{(cluster.anomaly_score * 100).toFixed(0)}%</span>
              </div>
            ))}
          </div>
        </div>
      )}
      </>
      )}
    </Panel>
  )
}

// ============================================================================
// Centrality Intermediaries Panel
// ============================================================================

function CentralityPanel() {
  const { currentRole, enabled: canViewCentrality } = usePermissionGroup(QUERY_PERMISSIONS.centrality)
  const { data: intermediariesData } = useQuery({
    queryKey: ['fraud', 'centrality', 'intermediaries', currentRole],
    queryFn: () => fetchIntermediaries(15),
    enabled: canViewCentrality,
    refetchInterval: canViewCentrality ? REFETCH_INTERVAL : false,
  })

  const intermediaries = intermediariesData?.intermediaries ?? []

  return (
    <Panel title="Centrality Analysis" icon={Network} accent={ACCENT.blue}>
      {!canViewCentrality ? <RoleLockedState area="Centrality analysis" /> : (
      <>
      {intermediaries.length === 0 ? (
        <p className="text-[10px] text-text-muted text-center py-4">No intermediaries detected yet</p>
      ) : (
        <div className="space-y-1.5">
          {intermediaries.slice(0, 12).map((node, i) => {
            const barWidth = Math.min(100, node.betweenness_centrality * 100)
            return (
              <div key={node.node_id} className="relative px-2.5 py-2 rounded-md bg-bg-elevated/30 border border-border-subtle/15 overflow-hidden">
                {/* Background bar */}
                <div
                  className="absolute inset-y-0 left-0 bg-[#00579C]/5 rounded-md"
                  style={{ width: `${barWidth}%` }}
                />
                <div className="relative flex items-center gap-2">
                  <span className="text-[9px] text-text-muted font-mono w-4 shrink-0">#{i + 1}</span>
                  <Fingerprint className="w-3 h-3 text-[#00579C] shrink-0" />
                  <span className="text-[10px] font-mono text-text-secondary truncate flex-1">{node.node_id.slice(0, 14)}</span>
                  <div className="flex gap-2 shrink-0 text-[9px] text-text-muted tabular-nums">
                    <span title="Betweenness">BC: <span className="text-[#00579C] font-medium">{node.betweenness_centrality.toFixed(3)}</span></span>
                    <span title="PageRank">PR: <span className="text-[#00579C] font-medium">{node.pagerank.toFixed(4)}</span></span>
                  </div>
                </div>
              </div>
            )
          })}
        </div>
      )}
      </>
      )}
    </Panel>
  )
}

// ============================================================================
// Victim Fund Tracing Panel
// ============================================================================

function VictimTracingPanel() {
  const { currentRole, enabled: canViewVictims } = usePermissionGroup(QUERY_PERMISSIONS.victim)
  const { data: victimStats } = useQuery({
    queryKey: ['fraud', 'victim', 'stats', currentRole],
    queryFn: fetchVictimStats,
    enabled: canViewVictims,
    refetchInterval: canViewVictims ? REFETCH_INTERVAL : false,
  })

  return (
    <Panel title="Victim Fund Tracing" icon={Route} accent={ACCENT.cyan}>
      {!canViewVictims ? <RoleLockedState area="Victim fund tracing" /> : (
      <div className="flex flex-col items-center justify-center py-6 gap-3">
        <div className="flex items-center justify-center w-14 h-14 rounded-full bg-[#00579C]/10 border border-[#00579C]/20">
          <Route className="w-7 h-7 text-[#00579C]" />
        </div>
        <div className="text-center">
          <p className="text-2xl font-bold text-[#00579C] tabular-nums">
            {victimStats?.total_victims_traced ?? 0}
          </p>
          <p className="text-[10px] text-text-muted uppercase tracking-wider mt-1">Victims Traced</p>
        </div>
        <p className="text-[9px] text-text-muted text-center max-w-[220px] leading-relaxed">
          Downstream fund-flow mapping tracks the complete movement of stolen funds through mule chains to identify all victim accounts.
        </p>
      </div>
      )}
    </Panel>
  )
}

// ============================================================================
// Pre-Approval Gate Panel
// ============================================================================

function GatePanel() {
  const { currentRole, enabled: canViewGate } = usePermissionGroup(QUERY_PERMISSIONS.gate)
  const { data: gateStats } = useQuery({
    queryKey: ['fraud', 'gate', 'stats', currentRole],
    queryFn: fetchGateStats,
    enabled: canViewGate,
    refetchInterval: canViewGate ? REFETCH_INTERVAL : false,
  })

  const stats = gateStats ?? {} as GateStatsResponse
  const total = stats.total_evaluations ?? 0

  const segments = [
    { label: 'Approved', count: stats.approved ?? 0, color: 'bg-[#00579C]', text: 'text-[#00579C]' },
    { label: 'Held', count: stats.held ?? 0, color: 'bg-[#DA251C]', text: 'text-[#DA251C]' },
    { label: 'Blocked', count: stats.blocked ?? 0, color: 'bg-[#DA251C]', text: 'text-[#DA251C]' },
  ]

  return (
    <Panel title="Pre-Approval Gate" icon={Shield} accent={ACCENT.emerald}>
      {!canViewGate ? <RoleLockedState area="Pre-approval gate metrics" /> : (
      <>
      {/* Total evaluations */}
      <div className="text-center mb-3">
        <p className="text-2xl font-bold text-text-primary tabular-nums">{total}</p>
        <p className="text-[9px] text-text-muted uppercase tracking-wider">Transactions Evaluated</p>
      </div>

      {/* Proportion bar */}
      {total > 0 && (
        <div className="flex h-2.5 rounded-full overflow-hidden mb-3 bg-bg-elevated/50">
          {segments.map(seg => {
            const pct = (seg.count / total) * 100
            return pct > 0 ? (
              <div key={seg.label} className={cn('h-full transition-all', seg.color)} style={{ width: `${pct}%` }} />
            ) : null
          })}
        </div>
      )}

      {/* Breakdown */}
      <div className="flex gap-2">
        {segments.map(seg => (
          <div key={seg.label} className="flex-1 rounded-md bg-bg-elevated/50 border border-border-subtle/25 px-2 py-1.5 text-center">
            <p className="text-[8px] text-text-muted uppercase tracking-wider">{seg.label}</p>
            <p className={cn('text-sm font-bold tabular-nums', seg.text)}>{seg.count}</p>
          </div>
        ))}
      </div>

      {(stats.avg_evaluation_ms ?? 0) > 0 && (
        <div className="mt-2 text-center">
          <p className="text-[9px] text-text-muted">
            Avg evaluation: <span className="text-[#00579C] font-medium">{(stats.avg_evaluation_ms ?? 0).toFixed(1)}ms</span>
          </p>
        </div>
      )}
      </>
      )}
    </Panel>
  )
}

// ============================================================================
// Main Compliance Page
// ============================================================================

export function CompliancePage() {
  const access = useRoleAccess()
  const currentRole = access.currentRole
  const canQuery = (permissions: Permission[]) => permissions.some((permission) => access.can(permission))
  const canViewGate = canQuery(QUERY_PERMISSIONS.gate)
  const canViewRules = canQuery(QUERY_PERMISSIONS.rules)
  const canViewCfr = canQuery(QUERY_PERMISSIONS.cfr)
  const canViewAml = canQuery(QUERY_PERMISSIONS.aml)
  const canViewFiu = canQuery(QUERY_PERMISSIONS.fiu)
  const canViewInvestigation = canQuery(QUERY_PERMISSIONS.investigation)
  const canViewMule = canQuery(QUERY_PERMISSIONS.mule)
  const canViewVictims = canQuery(QUERY_PERMISSIONS.victim)
  // Aggregate stats for the hero metrics strip
  const { data: gateStats } = useQuery({ queryKey: ['fraud', 'gate', 'stats', currentRole], queryFn: fetchGateStats, enabled: canViewGate, refetchInterval: canViewGate ? REFETCH_INTERVAL : false })
  const { data: ruleStats } = useQuery({ queryKey: ['fraud', 'rules', 'stats', currentRole], queryFn: fetchRuleStats, enabled: canViewRules, refetchInterval: canViewRules ? REFETCH_INTERVAL : false })
  const { data: cfrStats } = useQuery({ queryKey: ['fraud', 'cfr', 'stats', currentRole], queryFn: fetchCFRStats, enabled: canViewCfr, refetchInterval: canViewCfr ? REFETCH_INTERVAL : false })
  const { data: amlStats } = useQuery({ queryKey: ['fraud', 'aml', 'stats', currentRole], queryFn: fetchAMLStats, enabled: canViewAml, refetchInterval: canViewAml ? REFETCH_INTERVAL : false })
  const { data: fiuStats } = useQuery({ queryKey: ['fraud', 'fiu', 'stats', currentRole], queryFn: fetchFIUStats, enabled: canViewFiu, refetchInterval: canViewFiu ? REFETCH_INTERVAL : false })
  const { data: invStats } = useQuery({ queryKey: ['fraud', 'investigation', 'stats', currentRole], queryFn: fetchInvestigationStats, enabled: canViewInvestigation, refetchInterval: canViewInvestigation ? REFETCH_INTERVAL : false })
  const { data: muleStats } = useQuery({ queryKey: ['fraud', 'mule', 'stats', currentRole], queryFn: fetchMuleStats, enabled: canViewMule, refetchInterval: canViewMule ? REFETCH_INTERVAL : false })
  const { data: muleChainsData } = useQuery({ queryKey: ['fraud', 'mule', 'chains', currentRole], queryFn: fetchMuleChains, enabled: canViewMule, refetchInterval: canViewMule ? REFETCH_INTERVAL : false })
  const { data: victimStats } = useQuery({ queryKey: ['fraud', 'victim', 'stats', currentRole], queryFn: fetchVictimStats, enabled: canViewVictims, refetchInterval: canViewVictims ? REFETCH_INTERVAL : false })
  const placementAlerts = alertTotal(amlStats?.placement, ['cash_alerts', 'structuring_alerts', 'multi_channel_alerts', 'round_amount_alerts']) ?? 0
  const layeringAlerts = metricNumber(muleStats?.chain_detector, ['total_chains_detected']) ?? muleChainsData?.count ?? 0
  const integrationAlerts = alertTotal(amlStats?.integration, ['asset_purchase_alerts', 'investment_alerts', 'rapid_withdrawal_alerts', 'round_trip_alerts']) ?? 0
  const amlAlertTotal = placementAlerts + layeringAlerts + integrationAlerts

  return (
    <div className="flex min-h-full flex-col">
      {/* ---- Page header ---- */}
      <div className="shrink-0 px-5 pt-5 pb-3 animate-fade-in">
        <div className="flex items-center gap-3 mb-1.5">
          <div className="flex items-center justify-center w-9 h-9 rounded-lg bg-[#00579C]/10 border border-[#00579C]/20">
            <ShieldCheck className="w-5 h-5 text-[#00579C]" />
          </div>
          <div>
            <h1 className="text-sm font-semibold text-text-primary tracking-wide flex items-center gap-2">
              Compliance & Regulatory Intelligence
              <Landmark className="w-4 h-4 text-[#00579C]/80" />
            </h1>
            <p className="text-[10px] text-text-muted leading-relaxed mt-0.5">
              {access.policy.label} scope: {access.policy.escalationScope}
            </p>
          </div>
        </div>
      </div>

      {/* ---- Metrics Strip ---- */}
      <div className="shrink-0 px-5 pb-3 animate-fade-in" style={{ animationDelay: '30ms' }}>
        <div className="grid grid-cols-4 xl:grid-cols-8 gap-2">
          <MetricCard icon={Shield} label="Gate Evals" value={gateStats?.total_evaluations ?? 0} accent={ACCENT.emerald} sub={`${gateStats?.blocked ?? 0} blocked`} />
          <MetricCard icon={Scale} label="Rule Evals" value={ruleStats?.total_evaluations ?? 0} accent={ACCENT.cyan} />
          <MetricCard icon={Landmark} label="CFR Records" value={cfrStats?.registry?.total_records ?? 0} accent={ACCENT.indigo} />
          <MetricCard
            icon={Layers}
            label="AML Alerts"
            value={amlAlertTotal}
            accent={ACCENT.rose}
            pulse={amlAlertTotal > 0}
          />
          <MetricCard icon={Eye} label="FIU STRs" value={fiuStats?.str_collected ?? 0} accent={ACCENT.teal} sub={`${fiuStats?.packages_prepared ?? 0} packages`} />
          <MetricCard icon={Gavel} label="Cases" value={invStats?.total_cases ?? 0} accent={ACCENT.orange} sub={`${invStats?.open_cases ?? 0} open`} />
          <MetricCard icon={UserX} label="Mule Chains" value={muleChainsData?.count ?? metricNumber(muleStats?.chain_detector, ['total_chains_detected']) ?? 0} accent={ACCENT.pink} />
          <MetricCard icon={Route} label="Victims Traced" value={victimStats?.total_victims_traced ?? 0} accent={ACCENT.cyan} />
        </div>
      </div>

      {/* ---- Content Grid ---- */}
      <div className="custom-scrollbar space-y-3 px-5 pb-5">
        {/* Row 1: Gate + Rule Engine + Regulatory Reports */}
        <div className="flex gap-3 animate-fade-in" style={{ animationDelay: '60ms' }}>
          <div className="w-[280px] shrink-0">
            <GatePanel />
          </div>
          <div className="flex-1 min-w-0">
            <RuleEnginePanel />
          </div>
          <div className="w-[320px] shrink-0">
            <RegulatoryReportsPanel />
          </div>
        </div>

        {/* Row 2: AML Stages + CFR-RBI + FIU Intelligence */}
        <div className="flex gap-3 animate-fade-in" style={{ animationDelay: '120ms' }}>
          <div className="flex-1 min-w-0">
            <AMLStagePanel />
          </div>
          <div className="flex-1 min-w-0">
            <CFRRegistryPanel />
          </div>
          <div className="flex-1 min-w-0">
            <FIUIntelligencePanel />
          </div>
        </div>

        {/* Row 3: Investigation + Mule Detection + Anomaly + Centrality */}
        <div className="flex gap-3 animate-fade-in" style={{ animationDelay: '180ms' }}>
          <div className="w-[260px] shrink-0">
            <InvestigationPanel />
          </div>
          <div className="flex-1 min-w-0">
            <MuleDetectionPanel />
          </div>
          <div className="flex-1 min-w-0">
            <AnomalyDetectionPanel />
          </div>
        </div>

        {/* Row 4: Centrality + Victim Tracing */}
        <div className="flex gap-3 animate-fade-in" style={{ animationDelay: '240ms' }}>
          <div className="flex-[2] min-w-0">
            <CentralityPanel />
          </div>
          <div className="w-[280px] shrink-0">
            <VictimTracingPanel />
          </div>
        </div>
      </div>
    </div>
  )
}

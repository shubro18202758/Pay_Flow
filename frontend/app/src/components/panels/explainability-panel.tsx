// ============================================================================
// SHAP Explainability Panel -- Global feature importance + per-txn explanations
// ============================================================================

import { useEffect, useMemo, useState } from 'react'
import {
  Eye,
  BarChart3,
  ArrowUp,
  ArrowDown,
  Loader2,
  Search,
  BrainCircuit,
  DatabaseZap,
  Timer,
} from 'lucide-react'
import type { LucideIcon } from 'lucide-react'
import { cn, fmtOptionalTimestamp, truncId } from '@/lib/utils'
import { useGlobalImportance, useLLMStatus } from '@/hooks/use-api'
import { fetchTransactionExplanation } from '@/lib/api-client'
import { resolveLLMRuntime } from '@/lib/llm-runtime'
import { contributionToneClass, UBI_TONE } from '@/lib/union-bank-theme'
import { useActivityStore } from '@/stores/use-activity-store'
import { useDashboardStore } from '@/stores/use-dashboard-store'
import { useUIStore } from '@/stores/use-ui-store'
import type { ExplainResponse } from '@/lib/types'

function formatFeatureLabel(name: string) {
  return name.replace(/^ubi_/, '').replace(/^ext_/, '').replace(/_/g, ' ')
}

function formatFeatureValue(value: number) {
  if (Number.isInteger(value)) return String(value)
  return value.toFixed(Math.abs(value) >= 10 ? 1 : 3)
}

export function ExplainabilityPanel() {
  const { data: globalData, isLoading: globalLoading } = useGlobalImportance()
  const {
    data: llmStatus,
    isLoading: llmStatusLoading,
    isError: llmStatusError,
  } = useLLMStatus()
  const selectedEventId = useUIStore((state) => state.selectedEventId)
  const activityEvents = useActivityStore((state) => state.events)
  const orderedActivityIds = useActivityStore((state) => state.orderedIds)
  const graphEdges = useDashboardStore((state) => state.graphEdges)
  const [txnExplanation, setTxnExplanation] = useState<ExplainResponse | null>(null)
  const [txnLoading, setTxnLoading] = useState(false)
  const [txnError, setTxnError] = useState<string | null>(null)
  const [txnInput, setTxnInput] = useState('')
  const [activeView, setActiveView] = useState<'global' | 'transaction'>('global')
  const globalSnapshot = (globalData?.snapshot ?? {}) as Record<string, unknown>
  const llmRuntime = resolveLLMRuntime(llmStatus, {
    loading: llmStatusLoading,
    error: llmStatusError,
    fallbackModel: 'qwen3.5:4b',
  })

  const liveCandidates = useMemo(() => {
    const ids: string[] = []

    if (selectedEventId) ids.push(selectedEventId)

    ids.push(...orderedActivityIds.slice(0, 20))

    const recentGraphIds = [...graphEdges]
      .sort((a, b) => (b.data.timestamp ?? 0) - (a.data.timestamp ?? 0))
      .slice(0, 40)
      .map((edge) => edge.data.id)

    ids.push(...recentGraphIds)

    const uniqueIds = Array.from(new Set(ids.filter(Boolean))).slice(0, 12)

    return uniqueIds.map((id) => {
      const lifecycle = activityEvents.get(id)
      const edge = graphEdges.find((item) => item.data.id === id)
      const amountPaisa = lifecycle?.amountPaisa ?? edge?.data.amount_paisa ?? 0
      const riskScore = lifecycle?.riskScore
      const source = lifecycle ? 'activity feed' : edge ? 'graph topology' : 'selected event'
      const timestamp = lifecycle?.firstSeen ?? edge?.data.timestamp

      return {
        id,
        amountPaisa,
        riskScore,
        source,
        timestamp,
        fraudLabel: lifecycle?.fraudLabel ?? edge?.data.fraud_label ?? 0,
      }
    })
  }, [activityEvents, graphEdges, orderedActivityIds, selectedEventId])

  const activeDomainFeatures = useMemo(() => (
    Object.entries(txnExplanation?.domain_features ?? {})
      .filter(([, value]) => Number(value) > 0)
      .sort(([, a], [, b]) => Number(b) - Number(a))
      .slice(0, 8)
  ), [txnExplanation?.domain_features])

  useEffect(() => {
    if (activeView !== 'transaction' || txnInput || !selectedEventId) return
    setTxnInput(selectedEventId)
  }, [activeView, selectedEventId, txnInput])

  async function handleExplainTxn(targetId?: string) {
    const txnId = (targetId ?? txnInput).trim()
    if (!txnId) return
    setTxnInput(txnId)
    setTxnLoading(true)
    setTxnError(null)
    try {
      const result = await fetchTransactionExplanation(txnId)
      if (result.error) {
        setTxnExplanation(null)
        setTxnError(result.error)
        return
      }
      setTxnExplanation(result)
    } catch (error) {
      setTxnExplanation(null)
      setTxnError(error instanceof Error ? error.message : 'Unable to retrieve transaction explanation')
    } finally {
      setTxnLoading(false)
    }
  }

  // Prepare sorted global importance bars
  const importanceEntries = globalData?.feature_importance
    ? Object.entries(globalData.feature_importance)
        .sort(([, a], [, b]) => b - a)
        .slice(0, 20)
    : []

  const maxImportance = importanceEntries.length > 0
    ? Math.max(...importanceEntries.map(([, v]) => v))
    : 1

  return (
    <div className="flex flex-col h-full bg-bg-deep rounded-lg border border-border-subtle overflow-hidden">
      {/* Header */}
      <div className="flex flex-wrap items-center gap-2 px-4 py-3 border-b border-border-subtle bg-bg-surface/50">
        <Eye className="w-4 h-4 text-accent-primary" />
        <div className="min-w-0">
          <span className="block text-xs font-semibold text-text-primary tracking-wide">
            Fraud AI Reasoning
          </span>
          <span className="block text-[8px] font-bold uppercase tracking-[0.16em] text-text-muted">
            live feature cache + model attribution + Union Bank controls
          </span>
        </div>
        <div className="ml-auto flex min-w-0 flex-wrap items-center gap-1.5">
          <span className="inline-flex h-6 items-center gap-1 rounded-md border border-accent-primary/25 bg-accent-primary/10 px-2 text-[8px] font-bold uppercase tracking-[0.1em] text-accent-primary">
            <BrainCircuit className="h-3 w-3" />
            {llmRuntime.model}
          </span>
          <span
            className={cn(
              'inline-flex h-6 items-center rounded-md border px-2 text-[8px] font-bold uppercase tracking-[0.1em]',
              llmRuntime.running
                ? 'border-alert-low/30 bg-alert-low/10 text-alert-low'
                : 'border-alert-medium/30 bg-alert-medium/10 text-alert-medium',
            )}
            title={llmRuntime.statusDetail}
          >
            {llmRuntime.statusLabel}
          </span>
        </div>
        <div className="flex items-center gap-1">
          <button
            onClick={() => setActiveView('global')}
            className={cn(
              'text-[9px] px-2 py-1 rounded transition-colors',
              activeView === 'global'
                ? 'bg-accent-primary/15 text-accent-primary border border-accent-primary/30'
                : 'text-text-muted hover:text-text-secondary',
            )}
          >
            Global
          </button>
          <button
            onClick={() => setActiveView('transaction')}
            className={cn(
              'text-[9px] px-2 py-1 rounded transition-colors',
              activeView === 'transaction'
                ? 'bg-accent-primary/15 text-accent-primary border border-accent-primary/30'
                : 'text-text-muted hover:text-text-secondary',
            )}
          >
            Per-Txn
          </button>
        </div>
      </div>

      {/* Content */}
      <div className="flex-1 min-h-0 overflow-y-auto custom-scrollbar p-4">
        {activeView === 'global' ? (
          // Global feature importance view
          globalLoading ? (
            <div className="flex items-center justify-center h-full">
              <Loader2 className="w-5 h-5 animate-spin text-accent-primary" />
            </div>
          ) : importanceEntries.length === 0 ? (
            <div className="flex flex-col items-center justify-center h-full text-text-muted text-[10px] gap-2 animate-fade-in">
              <BarChart3 className="w-8 h-8 opacity-30" />
              <p>
                {globalSnapshot.model_ready === false
                  ? 'Model training has not completed yet.'
                  : 'Waiting for cached feature vectors for global importance.'}
              </p>
              {globalSnapshot.feature_source ? (
                <p className="text-[9px]">
                  Source: {String(globalSnapshot.feature_source)}
                </p>
              ) : null}
            </div>
          ) : (
            <div className="space-y-1.5 animate-fade-in">
              <p className="text-[9px] text-text-muted mb-3 uppercase tracking-wider">
                Top {importanceEntries.length} Features by SHAP Importance
              </p>
              {importanceEntries.map(([name, value], i) => (
                <div key={name} className="flex items-center gap-2 group" style={{ animationDelay: `${i * 20}ms` }}>
                  <span className="text-[9px] text-text-secondary w-32 truncate shrink-0 text-right font-mono">
                    {name}
                  </span>
                  <div className="flex-1 h-3 bg-bg-surface rounded-sm overflow-hidden">
                    <div
                      className="h-full bg-gradient-to-r from-accent-primary/70 to-alert-critical/60 rounded-sm transition-all duration-500"
                      style={{ width: `${(value / maxImportance) * 100}%` }}
                    />
                  </div>
                  <span className="text-[9px] text-text-muted font-mono w-12 text-right">
                    {value.toFixed(4)}
                  </span>
                </div>
              ))}
              {globalData?.snapshot && (
                <div className="mt-4 pt-3 border-t border-border-subtle text-[9px] text-text-muted space-y-1">
                  <p>Explained: {String(globalSnapshot.explanations_generated ?? 0)} transactions</p>
                  <p>Method: {String(globalSnapshot.method ?? 'TreeSHAP')}</p>
                  <p>Source: {String(globalSnapshot.feature_source ?? 'feature_cache')} ({String(globalSnapshot.sample_count ?? importanceEntries.length)} samples)</p>
                </div>
              )}
            </div>
          )
        ) : (
          // Per-transaction explanation view
          <div className="space-y-4 animate-fade-in">
            {/* Search input */}
            <div className="space-y-1.5">
              <label htmlFor="transaction-explain-id" className="text-[9px] font-bold uppercase tracking-[0.14em] text-text-muted">
                Transaction ID from live activity or graph cache
              </label>
            </div>
            <div className="flex items-center gap-2">
              <input
                id="transaction-explain-id"
                type="text"
                value={txnInput}
                onChange={(e) => setTxnInput(e.target.value)}
                onKeyDown={(e) => e.key === 'Enter' && handleExplainTxn()}
                aria-label="Transaction ID for fraud explanation"
                className="flex-1 bg-bg-surface border border-border-subtle rounded-md px-3 py-1.5
                  text-[11px] text-text-primary
                  focus:outline-none focus:border-accent-primary/40 transition-colors"
              />
              <button
                onClick={() => handleExplainTxn()}
                disabled={txnLoading || !txnInput.trim()}
                className="flex items-center justify-center w-7 h-7 rounded-md
                  bg-accent-primary/15 border border-accent-primary/30 text-accent-primary
                  hover:bg-accent-primary/25 disabled:opacity-30 disabled:cursor-not-allowed transition-colors"
              >
                {txnLoading ? <Loader2 className="w-3.5 h-3.5 animate-spin" /> : <Search className="w-3.5 h-3.5" />}
              </button>
            </div>

            {/* Live candidates */}
            <div className="rounded-md border border-border-subtle bg-bg-surface/45 p-3">
              <div className="flex items-center justify-between gap-2 mb-2">
                <span className="text-[9px] text-text-muted uppercase tracking-wider">
                  Live Transaction Candidates
                </span>
                {selectedEventId && (
                  <span className="text-[9px] text-accent-primary font-mono">
                    selected {truncId(selectedEventId, 10)}
                  </span>
                )}
              </div>
              {liveCandidates.length === 0 ? (
                <p className="text-[10px] text-text-muted leading-relaxed">
                  No explainable transactions yet. Launch or ingest events first so the backend can cache feature vectors.
                </p>
              ) : (
                <div className="grid gap-1.5">
                  {liveCandidates.map((candidate) => (
                    <button
                      key={candidate.id}
                      type="button"
                      data-testid="xai-candidate"
                      onClick={() => handleExplainTxn(candidate.id)}
                      disabled={txnLoading}
                      className={cn(
                        'group flex items-center justify-between gap-2 rounded-md border px-2.5 py-2 text-left transition-colors',
                        candidate.id === txnInput
                          ? 'border-accent-primary/40 bg-accent-primary/10'
                          : 'border-border-subtle bg-bg-deep/40 hover:border-accent-primary/25 hover:bg-accent-primary/5',
                        txnLoading && 'cursor-wait opacity-70',
                      )}
                    >
                      <div className="min-w-0">
                        <div className="flex items-center gap-2">
                          <span className="text-[10px] text-text-primary font-mono truncate">
                            {truncId(candidate.id, 18)}
                          </span>
                          {candidate.fraudLabel > 0 && (
                            <span className={cn('rounded-full px-1.5 py-0.5 text-[8px] font-semibold', UBI_TONE.redBadge)}>
                              FRAUD
                            </span>
                          )}
                        </div>
                        <p className="mt-0.5 text-[9px] text-text-muted">
                          {candidate.source}
                          {' - '}
                          {fmtOptionalTimestamp(candidate.timestamp)}
                        </p>
                      </div>
                      <div className="shrink-0 text-right">
                        <p className="text-[10px] font-mono text-text-secondary">
                          INR {(candidate.amountPaisa / 100).toLocaleString('en-IN', { maximumFractionDigits: 0 })}
                        </p>
                        <p className="text-[9px] font-mono text-[#DA251C]">
                          {candidate.riskScore !== undefined ? `${(candidate.riskScore * 100).toFixed(0)}% risk` : 'cached'}
                        </p>
                      </div>
                    </button>
                  ))}
                </div>
              )}
            </div>

            {/* Explanation result */}
            {txnExplanation && (
              <div className="space-y-3 animate-slide-up">
                {/* Summary */}
                <div className="bg-bg-surface rounded-md border border-border-subtle p-3">
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-[10px] text-text-muted font-mono">{txnExplanation.txn_id}</span>
                    <span className={cn(
                      'text-[9px] font-semibold px-2 py-0.5 rounded-full',
                      txnExplanation.verdict === 'FRAUD' || txnExplanation.verdict === 'SUSPICIOUS'
                        ? UBI_TONE.redBadgeStrong
                        : UBI_TONE.blueBadge,
                    )}>
                      {txnExplanation.verdict}
                    </span>
                  </div>
                  <div className="text-[11px] text-text-secondary leading-relaxed">
                    {txnExplanation.narrative}
                  </div>
                  <div className="mt-3 grid gap-2 sm:grid-cols-3">
                    <ReasonMetric
                      icon={BrainCircuit}
                      label="classifier verdict"
                      value={`${(txnExplanation.risk_score * 100).toFixed(1)}% risk`}
                    />
                    <ReasonMetric
                      icon={DatabaseZap}
                      label="feature source"
                      value={
                        txnExplanation.feature_source === 'feature_cache'
                          ? `${txnExplanation.feature_count ?? 0} cached features`
                          : `${txnExplanation.feature_count ?? 0} submitted features`
                      }
                    />
                    <ReasonMetric
                      icon={Timer}
                      label="attribution"
                      value={`${txnExplanation.attribution_method ?? 'model'}${txnExplanation.explanation_ms != null ? ` / ${txnExplanation.explanation_ms} ms` : ''}`}
                    />
                  </div>
                </div>

                {txnExplanation.model_reasoning && (
                  <div className="rounded-md border border-border-subtle bg-bg-surface/70 p-3">
                    <div className="mb-2 flex items-center gap-2">
                      <BrainCircuit className="h-3.5 w-3.5 text-accent-primary" />
                      <p className="text-[9px] font-bold uppercase tracking-[0.14em] text-text-muted">
                        Model-Derived Reasoning
                      </p>
                    </div>
                    <p className="text-[10px] leading-relaxed text-text-secondary">
                      {txnExplanation.model_reasoning.summary}
                    </p>
                    <div className="mt-2 grid gap-2 sm:grid-cols-2">
                      <ReasonList title="Risk drivers" values={txnExplanation.model_reasoning.risk_drivers} tone="risk" />
                      <ReasonList title="Mitigating drivers" values={txnExplanation.model_reasoning.protective_factors} tone="protective" />
                    </div>
                  </div>
                )}

                {txnExplanation.domain_feature_count != null && txnExplanation.domain_feature_count > 0 && (
                  <div className="rounded-md border border-accent-primary/25 bg-accent-primary/10 p-3">
                    <div className="mb-2 flex items-center justify-between gap-2">
                      <p className="text-[9px] font-bold uppercase tracking-[0.14em] text-accent-primary">
                        Union Bank Domain Checks
                      </p>
                      <span className="font-mono text-[9px] text-text-muted">
                        {txnExplanation.domain_feature_count} controls
                      </span>
                    </div>
                    {txnExplanation.domain_controls?.length ? (
                      <div className="space-y-1.5">
                        {txnExplanation.domain_controls.slice(0, 6).map((control) => (
                          <div key={control} className="rounded border border-border-subtle bg-bg-surface/70 px-2 py-1.5 text-[9px] leading-relaxed text-text-secondary">
                            {control}
                          </div>
                        ))}
                      </div>
                    ) : (
                      <div className="text-[9px] text-text-muted">
                        No RBI/FIU, beneficiary, MFA, CFR, or pass-through control threshold fired for this transaction.
                      </div>
                    )}
                    {activeDomainFeatures.length > 0 && (
                      <div className="mt-3 grid gap-1.5 sm:grid-cols-2">
                        {activeDomainFeatures.map(([name, value]) => (
                          <div key={name} className="flex items-center justify-between gap-2 rounded border border-accent-primary/15 bg-bg-deep/50 px-2 py-1.5">
                            <span className="truncate text-[8px] font-bold uppercase tracking-[0.08em] text-text-secondary" title={name}>
                              {formatFeatureLabel(name)}
                            </span>
                            <span className="font-mono text-[9px] text-accent-primary">
                              {formatFeatureValue(Number(value))}
                            </span>
                          </div>
                        ))}
                      </div>
                    )}
                  </div>
                )}

                {/* Feature contributions */}
                <p className="text-[9px] text-text-muted uppercase tracking-wider">Key Feature Contributions</p>
                <div className="space-y-1.5">
                  {txnExplanation.top_features.map((feat, i) => (
                    <div
                      key={i}
                      className="flex items-center gap-2 bg-bg-surface/50 rounded-md px-3 py-2 border border-border-subtle"
                    >
                      {feat.direction === 'increases_risk' ? (
                        <ArrowUp className="w-3 h-3 text-[#DA251C] shrink-0" />
                      ) : (
                        <ArrowDown className="w-3 h-3 text-[#00579C] shrink-0" />
                      )}
                      <div className="flex-1 min-w-0">
                        <span className="text-[10px] text-text-primary font-mono">{feat.name}</span>
                        <p className="text-[9px] text-text-muted truncate">{feat.description}</p>
                      </div>
                      <span className={cn(
                        'text-[10px] font-mono shrink-0',
                        contributionToneClass(feat.contribution),
                      )}>
                        {feat.contribution > 0 ? '+' : ''}{feat.contribution.toFixed(4)}
                      </span>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {txnError && (
              <div className="rounded-md border border-[#DA251C]/25 bg-[#DA251C]/10 px-3 py-2 text-[10px] text-[#DA251C]">
                {txnError}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  )
}

function ReasonMetric({ icon: Icon, label, value }: { icon: LucideIcon; label: string; value: string }) {
  return (
    <div className="min-w-0 rounded-md border border-border-subtle bg-bg-deep/45 px-2.5 py-2">
      <div className="flex items-center gap-1.5 text-[8px] font-bold uppercase tracking-[0.12em] text-text-muted">
        <Icon className="h-3 w-3 text-accent-primary" />
        {label}
      </div>
      <div className="mt-1 truncate font-mono text-[10px] font-semibold text-text-primary" title={value}>
        {value}
      </div>
    </div>
  )
}

function ReasonList({
  title,
  values,
  tone,
}: {
  title: string
  values: string[]
  tone: 'risk' | 'protective'
}) {
  const color = tone === 'risk' ? 'text-alert-critical' : 'text-alert-low'
  return (
    <div className="rounded-md border border-border-subtle bg-bg-deep/40 p-2">
      <div className={cn('mb-1 text-[8px] font-bold uppercase tracking-[0.12em]', color)}>
        {title}
      </div>
      {values.length ? (
        <div className="flex flex-wrap gap-1">
          {values.map((value) => (
            <span key={value} className="rounded-sm border border-border-subtle bg-bg-surface px-1.5 py-1 text-[8px] text-text-secondary">
              {formatFeatureLabel(value)}
            </span>
          ))}
        </div>
      ) : (
        <div className="text-[9px] text-text-muted">No dominant signals in the current attribution set.</div>
      )}
    </div>
  )
}

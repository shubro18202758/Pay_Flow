// ============================================================================
// SHAP Explainability Panel -- Global feature importance + per-txn explanations
// ============================================================================

import { useState } from 'react'
import { Eye, BarChart3, ArrowUp, ArrowDown, Loader2, Search } from 'lucide-react'
import { cn } from '@/lib/utils'
import { useGlobalImportance } from '@/hooks/use-api'
import { fetchExplanation } from '@/lib/api-client'
import type { ExplainResponse } from '@/lib/types'

export function ExplainabilityPanel() {
  const { data: globalData, isLoading: globalLoading } = useGlobalImportance()
  const [txnExplanation, setTxnExplanation] = useState<ExplainResponse | null>(null)
  const [txnLoading, setTxnLoading] = useState(false)
  const [txnInput, setTxnInput] = useState('')
  const [activeView, setActiveView] = useState<'global' | 'transaction'>('global')

  async function handleExplainTxn() {
    if (!txnInput.trim()) return
    setTxnLoading(true)
    try {
      // Generate a simple 42-dim feature vector for demonstration
      const features = Array.from({ length: 42 }, () => Math.random())
      const result = await fetchExplanation(features, txnInput.trim())
      setTxnExplanation(result)
    } catch {
      setTxnExplanation(null)
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
      <div className="flex items-center gap-2 px-4 py-3 border-b border-border-subtle bg-bg-surface/50">
        <Eye className="w-4 h-4 text-violet-400" />
        <span className="text-xs font-semibold text-text-primary tracking-wide">
          SHAP Explainability
        </span>
        <div className="ml-auto flex items-center gap-1">
          <button
            onClick={() => setActiveView('global')}
            className={cn(
              'text-[9px] px-2 py-1 rounded transition-colors',
              activeView === 'global'
                ? 'bg-violet-500/20 text-violet-300 border border-violet-500/30'
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
                ? 'bg-violet-500/20 text-violet-300 border border-violet-500/30'
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
              <Loader2 className="w-5 h-5 animate-spin text-violet-400" />
            </div>
          ) : importanceEntries.length === 0 ? (
            <div className="flex flex-col items-center justify-center h-full text-text-muted text-[10px] gap-2 animate-fade-in">
              <BarChart3 className="w-8 h-8 opacity-30" />
              <p>Model not trained yet — SHAP importance unavailable</p>
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
                      className="h-full bg-gradient-to-r from-violet-500/60 to-violet-400/40 rounded-sm transition-all duration-500"
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
                  <p>Explained: {String((globalData.snapshot as Record<string,unknown>).explanations_generated ?? 0)} transactions</p>
                  <p>Method: {String((globalData.snapshot as Record<string,unknown>).method ?? 'TreeSHAP')}</p>
                </div>
              )}
            </div>
          )
        ) : (
          // Per-transaction explanation view
          <div className="space-y-4 animate-fade-in">
            {/* Search input */}
            <div className="flex items-center gap-2">
              <input
                type="text"
                value={txnInput}
                onChange={(e) => setTxnInput(e.target.value)}
                onKeyDown={(e) => e.key === 'Enter' && handleExplainTxn()}
                placeholder="Enter transaction ID..."
                className="flex-1 bg-bg-surface border border-border-subtle rounded-md px-3 py-1.5
                  text-[11px] text-text-primary placeholder:text-text-muted/50
                  focus:outline-none focus:border-violet-500/40 transition-colors"
              />
              <button
                onClick={handleExplainTxn}
                disabled={txnLoading || !txnInput.trim()}
                className="flex items-center justify-center w-7 h-7 rounded-md
                  bg-violet-500/15 border border-violet-500/30 text-violet-400
                  hover:bg-violet-500/25 disabled:opacity-30 disabled:cursor-not-allowed transition-colors"
              >
                {txnLoading ? <Loader2 className="w-3.5 h-3.5 animate-spin" /> : <Search className="w-3.5 h-3.5" />}
              </button>
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
                      txnExplanation.verdict === 'FRAUD' ? 'bg-red-500/20 text-red-400'
                        : txnExplanation.verdict === 'SUSPICIOUS' ? 'bg-amber-500/20 text-amber-400'
                        : 'bg-green-500/20 text-green-400',
                    )}>
                      {txnExplanation.verdict}
                    </span>
                  </div>
                  <div className="text-[11px] text-text-secondary leading-relaxed">
                    {txnExplanation.narrative}
                  </div>
                  <div className="mt-2 text-[10px] text-text-muted">
                    Risk Score: <span className="font-mono text-amber-400">{(txnExplanation.risk_score * 100).toFixed(1)}%</span>
                  </div>
                </div>

                {/* Feature contributions */}
                <p className="text-[9px] text-text-muted uppercase tracking-wider">Key Feature Contributions</p>
                <div className="space-y-1.5">
                  {txnExplanation.top_features.map((feat, i) => (
                    <div
                      key={i}
                      className="flex items-center gap-2 bg-bg-surface/50 rounded-md px-3 py-2 border border-border-subtle"
                    >
                      {feat.direction === 'increases_risk' ? (
                        <ArrowUp className="w-3 h-3 text-red-400 shrink-0" />
                      ) : (
                        <ArrowDown className="w-3 h-3 text-green-400 shrink-0" />
                      )}
                      <div className="flex-1 min-w-0">
                        <span className="text-[10px] text-text-primary font-mono">{feat.name}</span>
                        <p className="text-[9px] text-text-muted truncate">{feat.description}</p>
                      </div>
                      <span className={cn(
                        'text-[10px] font-mono shrink-0',
                        feat.contribution > 0 ? 'text-red-400' : 'text-green-400',
                      )}>
                        {feat.contribution > 0 ? '+' : ''}{feat.contribution.toFixed(4)}
                      </span>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  )
}

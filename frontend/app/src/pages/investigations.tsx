// ============================================================================
// Investigations Page -- Union Bank fund-flow case workbench
// ============================================================================

import { useState } from 'react'
import {
  Activity,
  BadgeCheck,
  BrainCircuit,
  FileText,
  GitBranch,
  Play,
  Printer,
  Scale,
  ShieldCheck,
  TimerReset,
} from 'lucide-react'
import { PreFraudIntelBrief } from '@/components/panels/pre-fraud-intel-brief'
import { EscalationList } from '@/components/investigations/escalation-list'
import { useRoleAccess } from '@/hooks/use-rbac'
import {
  useCaseTrace,
  useCreateEvidencePackage,
  useLaunchPS3Scenario,
  useLLMStatus,
  usePS3Readiness,
  usePS3Scenarios,
} from '@/hooks/use-api'
import { resolveLLMRuntime } from '@/lib/llm-runtime'
import { cn } from '@/lib/utils'
import { useUIStore } from '@/stores/use-ui-store'
import type {
  CaseTraceResponse,
  EvidencePackageResponse,
  PS3ScenarioId,
} from '@/lib/types'

function shortId(value: string, keep = 12) {
  if (!value) return 'n/a'
  return value.length > keep ? `${value.slice(0, keep)}...` : value
}

function pct(value: number) {
  return `${Math.round(value * 100)}%`
}

function metricOrNA(value: number | string | null | undefined, suffix = '') {
  if (value == null || value === '') return 'n/a'
  if (typeof value === 'number' && !Number.isFinite(value)) return 'n/a'
  return `${value}${suffix}`
}

function openPrintablePackage(pkg: EvidencePackageResponse) {
  const win = window.open('', '_blank', 'noopener,noreferrer')
  if (!win) return
  win.document.write(pkg.printable_html)
  win.document.close()
}

function StatusBadge({ status }: { status: string }) {
  const ready = status === 'ready' || status === 'completed' || status === 'running'
  return (
    <span
      className={cn(
        'inline-flex items-center gap-1 rounded-md border px-2 py-1 text-[9px] font-bold uppercase tracking-[0.12em]',
        ready
          ? 'border-[#00579C]/30 bg-[#00579C]/10 text-[#00579C]'
          : 'border-[#DA251C]/30 bg-[#DA251C]/10 text-[#DA251C]',
      )}
    >
      <BadgeCheck className="h-3 w-3" />
      {status}
    </span>
  )
}

function ReadinessPanel() {
  const { data } = usePS3Readiness()
  const requirements = data?.requirements ?? []

  return (
    <section className="rounded-lg border border-border-subtle bg-bg-deep p-3">
      <div className="mb-3 flex items-center justify-between">
        <div className="flex items-center gap-2 text-[10px] font-semibold uppercase tracking-[0.14em] text-text-secondary">
          <ShieldCheck className="h-4 w-4 text-[#00579C]" />
          Union Bank Fund-Flow Readiness
        </div>
        <span className="rounded-md border border-[#00579C]/25 bg-[#00579C]/10 px-2 py-1 text-[9px] font-bold uppercase tracking-[0.12em] text-[#00579C]">
          {data ? `${requirements.filter((r) => r.status === 'ready').length}/${requirements.length} ready` : 'syncing'}
        </span>
      </div>
      <div className="space-y-2">
        {requirements.map((item) => (
          <div key={item.id} className="rounded-md border border-border-subtle bg-bg-elevated/60 p-2">
            <div className="mb-1 flex items-center justify-between gap-2">
              <span className="text-[10px] font-semibold text-text-primary">{item.label}</span>
              <StatusBadge status={item.status} />
            </div>
            <p className="text-[9px] leading-relaxed text-text-muted">{item.evidence}</p>
          </div>
        ))}
      </div>
    </section>
  )
}

function ScenarioGallery({
  selected,
  onSelect,
  onLaunch,
  launching,
  canLaunchRole,
  roleLabel,
}: {
  selected: PS3ScenarioId
  onSelect: (id: PS3ScenarioId) => void
  onLaunch: () => void
  launching: boolean
  canLaunchRole: boolean
  roleLabel: string
}) {
  const { data, isLoading, isError } = usePS3Scenarios()
  const scenarios = data?.scenarios ?? []
  const canLaunch = scenarios.length > 0 && !isLoading && !isError

  return (
    <section className="rounded-lg border border-border-subtle bg-bg-deep p-3">
      <div className="mb-3 flex items-center justify-between">
        <div className="flex items-center gap-2 text-[10px] font-semibold uppercase tracking-[0.14em] text-text-secondary">
          <GitBranch className="h-4 w-4 text-accent-primary" />
          Fund-Flow Case Drill Gallery
        </div>
        <button
          onClick={onLaunch}
          disabled={launching || !canLaunch || !canLaunchRole}
          title={!canLaunchRole ? `${roleLabel} cannot launch fund-flow case drills` : 'Launch case drill'}
          className={cn(
            'inline-flex items-center gap-1.5 rounded-md border border-accent-primary/60 px-3 py-1.5',
            'text-[9px] font-bold uppercase tracking-[0.12em] text-accent-primary transition-colors',
            'hover:bg-accent-primary hover:text-bg-deep disabled:cursor-not-allowed disabled:opacity-50',
          )}
        >
          <Play className="h-3.5 w-3.5" />
          {launching ? 'Launching' : 'Launch Case Drill'}
        </button>
      </div>
      <div className="grid grid-cols-1 gap-2">
        {isLoading && (
          <div className="rounded-md border border-border-subtle bg-bg-elevated/45 p-3">
            <div className="mb-2 h-3 w-44 animate-pulse rounded bg-border-default" />
            <div className="h-2 w-64 animate-pulse rounded bg-border-subtle" />
          </div>
        )}
        {!isLoading && (isError || scenarios.length === 0) && (
          <div className="rounded-md border border-dashed border-border-subtle bg-bg-elevated/45 p-4 text-[10px] leading-relaxed text-text-muted">
            Scenario definitions are not available from the backend right now. Case drills stay disabled until `/api/v1/simulation/fund-flow/scenarios` returns live configuration.
          </div>
        )}
        {scenarios.map((scenario) => {
          const active = scenario.id === selected
          return (
            <button
              key={scenario.id}
              onClick={() => onSelect(scenario.id)}
              className={cn(
                'rounded-md border p-3 text-left transition-colors',
                active
                  ? 'border-accent-primary/60 bg-accent-primary/10'
                  : 'border-border-subtle bg-bg-elevated/50 hover:border-border-default',
              )}
            >
              <div className="mb-1 flex items-center justify-between gap-2">
                <span className="text-[10px] font-bold uppercase tracking-[0.1em] text-text-primary">
                  {scenario.label}
                </span>
                <span className="rounded bg-bg-overlay px-1.5 py-0.5 text-[8px] font-semibold text-text-muted">
                  {scenario.typologies[0]}
                </span>
              </div>
              <p className="text-[9px] leading-relaxed text-text-muted">
                {scenario.expected_indicators.slice(0, 2).join(' | ')}
              </p>
            </button>
          )
        })}
      </div>
    </section>
  )
}

function CaseTracePanel({ trace }: { trace: CaseTraceResponse | undefined }) {
  const timeline = trace?.timeline ?? []

  return (
    <section className="flex min-h-0 flex-1 flex-col rounded-lg border border-border-subtle bg-bg-deep">
      <div className="shrink-0 border-b border-border-subtle p-4">
        <div className="mb-2 flex items-center justify-between gap-3">
          <div>
            <div className="mb-1 flex items-center gap-2 text-[10px] font-semibold uppercase tracking-[0.14em] text-text-secondary">
              <Scale className="h-4 w-4 text-accent-primary" />
              Case Workbench
            </div>
            <h2 className="text-sm font-semibold text-text-primary">
              {trace?.scenario_label ?? 'Awaiting fund-flow case replay'}
            </h2>
          </div>
          {trace && <StatusBadge status={trace.status} />}
        </div>
        <div className="grid grid-cols-4 gap-2">
          <Metric label="Case" value={trace?.case_id ?? 'n/a'} />
          <Metric label="Focus Txn" value={shortId(trace?.focus_txn_id ?? '')} />
          <Metric label="Graph Evidence" value={trace ? pct(trace.risk_scores.graph_evidence_score) : 'n/a'} />
          <Metric label="Value" value={trace?.risk_scores.total_amount_display ?? 'n/a'} />
        </div>
        <PreFraudIntelBrief
          variant="case"
          context={trace?.pre_fraud_intelligence}
          className="mt-3"
        />
      </div>

      <div className="grid min-h-0 flex-1 grid-cols-1 gap-0 2xl:grid-cols-[1.15fr_0.85fr]">
        <div className="min-h-0 overflow-auto p-4">
          <div className="mb-3 flex items-center gap-2 text-[10px] font-semibold uppercase tracking-[0.14em] text-text-secondary">
            <TimerReset className="h-4 w-4 text-[#DA251C]" />
            Transaction Timeline
          </div>
          <div className="space-y-2">
            {timeline.map((entry) => (
              <div key={`${entry.step}-${entry.txn_id}`} className="rounded-md border border-border-subtle bg-bg-elevated/55 p-3">
                <div className="mb-2 flex items-center justify-between gap-2">
                  <span className="text-[10px] font-semibold text-text-primary">
                    {entry.step}. {entry.title}
                  </span>
                  <span className="font-mono text-[9px] text-text-muted">{shortId(entry.txn_id)}</span>
                </div>
                <div className="mb-2 flex flex-wrap gap-1.5">
                  <span className="rounded bg-bg-overlay px-2 py-0.5 text-[8px] font-semibold text-text-secondary">
                    {entry.amount_display}
                  </span>
                  <span className="rounded bg-bg-overlay px-2 py-0.5 text-[8px] font-semibold text-text-secondary">
                    {entry.channel}
                  </span>
                  <span className="rounded bg-bg-overlay px-2 py-0.5 text-[8px] font-semibold text-text-secondary">
                    {entry.evidence_id}
                  </span>
                </div>
                <p className="text-[9px] leading-relaxed text-text-muted">{entry.indicator}</p>
              </div>
            ))}
            {timeline.length === 0 && (
              <div className="flex h-60 items-center justify-center rounded-md border border-dashed border-border-subtle text-[10px] uppercase tracking-[0.12em] text-text-muted">
                Launch a fund-flow scenario
              </div>
            )}
          </div>
        </div>

        <div className="min-h-0 overflow-auto border-t border-border-subtle p-4 2xl:border-l 2xl:border-t-0">
          <div className="mb-3 flex items-center gap-2 text-[10px] font-semibold uppercase tracking-[0.14em] text-text-secondary">
            <GitBranch className="h-4 w-4 text-[#00579C]" />
            Fund Path
          </div>
          <div className="space-y-2">
            {(trace?.account_roles ?? []).map((role) => (
              <div key={`${role.position}-${role.account_id}`} className="rounded-md border border-border-subtle bg-bg-elevated/55 p-2">
                <div className="mb-1 flex items-center justify-between gap-2">
                  <span className="font-mono text-[10px] text-text-primary">{shortId(role.account_id, 16)}</span>
                  <span className="rounded bg-bg-overlay px-1.5 py-0.5 text-[8px] font-semibold uppercase text-text-muted">
                    {role.role.replace(/_/g, ' ')}
                  </span>
                </div>
                <div className="h-1 overflow-hidden rounded-full bg-bg-overlay">
                  <div
                    className="h-full rounded-full bg-accent-primary"
                    style={{ width: `${Math.min(100, Number(role.position) * 16)}%` }}
                  />
                </div>
              </div>
            ))}
          </div>

          <div className="mt-5">
            <div className="mb-2 flex items-center gap-2 text-[10px] font-semibold uppercase tracking-[0.14em] text-text-secondary">
              <ShieldCheck className="h-4 w-4 text-alert-high" />
              Suspicious Indicators
            </div>
            <div className="space-y-2">
              {(trace?.expected_indicators ?? []).map((indicator) => (
                <div key={indicator} className="rounded-md border border-border-subtle bg-bg-elevated/55 p-2 text-[9px] leading-relaxed text-text-muted">
                  {indicator}
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </section>
  )
}

function EvidencePanel({
  caseId,
  packageData,
  onGenerate,
  generating,
  canGenerateRole,
  roleLabel,
}: {
  caseId: string | null
  packageData: EvidencePackageResponse | null
  onGenerate: () => void
  generating: boolean
  canGenerateRole: boolean
  roleLabel: string
}) {
  return (
    <section className="rounded-lg border border-border-subtle bg-bg-deep p-3">
      <div className="mb-3 flex items-center justify-between gap-2">
        <div className="flex items-center gap-2 text-[10px] font-semibold uppercase tracking-[0.14em] text-text-secondary">
          <FileText className="h-4 w-4 text-[#DA251C]" />
          Evidence Package v2
        </div>
        <button
          onClick={onGenerate}
          disabled={!caseId || generating || !canGenerateRole}
          title={!canGenerateRole ? `${roleLabel} cannot generate FIU evidence packages` : 'Generate FIU evidence package'}
          className={cn(
            'inline-flex items-center gap-1.5 rounded-md border border-[#DA251C]/50 px-2.5 py-1.5',
            'text-[9px] font-bold uppercase tracking-[0.12em] text-[#DA251C] transition-colors',
            'hover:bg-[#DA251C] hover:text-bg-deep disabled:cursor-not-allowed disabled:opacity-40',
          )}
        >
          <FileText className="h-3.5 w-3.5" />
          {generating ? 'Generating' : 'Generate'}
        </button>
      </div>

      {packageData ? (
        <div className="space-y-3">
          <PreFraudIntelBrief
            variant="evidence"
            context={packageData.pre_fraud_intelligence}
          />
          <div className="rounded-md border border-border-subtle bg-bg-elevated/55 p-3">
            <div className="mb-1 font-mono text-[10px] text-text-primary">{packageData.package_id}</div>
            <p className="text-[9px] leading-relaxed text-text-muted">{packageData.fiu_summary}</p>
          </div>
          <div className="grid grid-cols-2 gap-2">
            <Metric label="Model" value={packageData.model_metadata.model} />
            <Metric label="Audit Hash" value={shortId(packageData.audit_hashes.package_hash, 10)} />
          </div>
          <button
            onClick={() => openPrintablePackage(packageData)}
            className="inline-flex w-full items-center justify-center gap-1.5 rounded-md border border-border-default px-3 py-2 text-[9px] font-bold uppercase tracking-[0.12em] text-text-secondary transition-colors hover:border-accent-primary hover:text-accent-primary"
          >
            <Printer className="h-3.5 w-3.5" />
            Open Printable Package
          </button>
        </div>
      ) : (
        <div className="rounded-md border border-dashed border-border-subtle p-4 text-[10px] leading-relaxed text-text-muted">
          FIU-ready JSON and printable HTML will appear after case generation.
        </div>
      )}
    </section>
  )
}

function ScalePanel() {
  const { data } = usePS3Readiness()
  const metrics = data?.scale_metrics

  return (
    <section className="rounded-lg border border-border-subtle bg-bg-deep p-3">
      <div className="mb-3 flex items-center gap-2 text-[10px] font-semibold uppercase tracking-[0.14em] text-text-secondary">
        <Activity className="h-4 w-4 text-accent-primary" />
        Scale Proof
      </div>
      <div className="grid grid-cols-2 gap-2">
        <Metric label="Events" value={metricOrNA(metrics?.events_ingested)} />
        <Metric label="EPS" value={metricOrNA(metrics?.events_per_sec)} />
        <Metric label="Graph Nodes" value={metricOrNA(metrics?.graph_nodes)} />
        <Metric label="Graph Edges" value={metricOrNA(metrics?.graph_edges)} />
        <Metric label="Free VRAM" value={metricOrNA(metrics?.gpu_vram_free_mb, ' MB')} />
        <Metric label="LLM Tokens" value={metricOrNA(metrics?.llm_tokens_total)} />
      </div>
      <div className="mt-3 space-y-2">
        {(data?.pilot_architecture ?? []).slice(0, 4).map((item) => (
          <div key={item} className="rounded-md border border-border-subtle bg-bg-elevated/45 p-2 text-[9px] leading-relaxed text-text-muted">
            {item}
          </div>
        ))}
      </div>
    </section>
  )
}

function Metric({ label, value }: { label: string; value: string }) {
  return (
    <div className="min-w-0 rounded-md border border-border-subtle bg-bg-elevated/55 p-2">
      <div className="mb-1 truncate text-[8px] font-semibold uppercase tracking-[0.12em] text-text-muted">{label}</div>
      <div className="truncate font-mono text-[10px] font-semibold text-text-primary" title={value}>
        {value}
      </div>
    </div>
  )
}

export function InvestigationsPage() {
  const access = useRoleAccess()
  const [selectedScenario, setSelectedScenario] = useState<PS3ScenarioId>('rapid_layering')
  const activeCaseId = useUIStore((s) => s.activeCaseId)
  const setActiveCaseId = useUIStore((s) => s.setActiveCaseId)
  const latestEvidencePackage = useUIStore((s) => s.latestEvidencePackage)
  const setLatestEvidencePackage = useUIStore((s) => s.setLatestEvidencePackage)
  const caseId = activeCaseId
  const packageData: EvidencePackageResponse | null =
    latestEvidencePackage?.case_id === caseId ? latestEvidencePackage : null
  const launchPS3 = useLaunchPS3Scenario()
  const evidence = useCreateEvidencePackage()
  const { data: trace } = useCaseTrace(caseId)
  const { data: llmStatus, isLoading: llmStatusLoading, isError: llmStatusError } = useLLMStatus()
  const llmRuntime = resolveLLMRuntime(llmStatus, {
    loading: llmStatusLoading,
    error: llmStatusError,
  })

  const typologyLine = trace?.ps3_typologies?.length
    ? trace.ps3_typologies.join(' / ')
    : 'fund-flow case'

  async function launchCaseDrill() {
    if (!access.can('case:launch')) return
    setLatestEvidencePackage(null)
    const response = await launchPS3.mutateAsync({
      scenario: selectedScenario,
      intensity: 'scale',
      seed: Date.now() % 1_000_000,
    })
    setActiveCaseId(response.primary_case_id)
  }

  async function generatePackage() {
    if (!caseId || !access.can('evidence:package')) return
    const pkg = await evidence.mutateAsync(caseId)
    setLatestEvidencePackage(pkg)
  }

  return (
    <div className="flex min-h-full flex-col">
      <div className="ubi-page-band shrink-0 border-b px-5 py-4">
        <div className="flex items-center justify-between gap-4">
          <div className="flex min-w-0 items-center gap-3">
            <div className="flex h-9 w-9 items-center justify-center rounded-lg border border-accent-primary/20 bg-accent-primary/10">
              <BrainCircuit className="h-5 w-5 text-accent-primary" />
            </div>
            <div className="min-w-0">
              <h1 className="truncate text-base font-bold tracking-wide text-text-primary">
                Union Bank Fund-Flow Case Workbench
              </h1>
              <p className="mt-0.5 truncate text-[10px] text-text-muted" title={`LLM runtime: ${llmRuntime.model} (${llmRuntime.statusLabel})`}>
                {access.policy.label} scope: {access.policy.escalationScope} | {typologyLine} | {llmRuntime.model} advisory copilot {llmRuntime.statusLabel}
              </p>
            </div>
          </div>
          <div className="hidden items-center gap-2 xl:flex">
            {trace && <StatusBadge status={trace.status} />}
            <span className="rounded-md border border-border-default px-2 py-1 text-[9px] font-semibold uppercase tracking-[0.12em] text-text-muted">
              {caseId ?? 'No active case'}
            </span>
          </div>
        </div>
      </div>

      <div className="grid min-h-0 grid-cols-1 gap-3 px-5 pb-5 xl:grid-cols-[320px_minmax(0,1fr)_320px]">
        <div className="min-h-0 space-y-3">
          <PreFraudIntelBrief variant="case" />
          <ReadinessPanel />
          <ScenarioGallery
            selected={selectedScenario}
            onSelect={setSelectedScenario}
            onLaunch={() => void launchCaseDrill()}
            launching={launchPS3.isPending}
            canLaunchRole={access.can('case:launch')}
            roleLabel={access.policy.label}
          />
        </div>

        <CaseTracePanel trace={trace} />

        <div className="min-h-0 space-y-3">
          <EvidencePanel
            caseId={caseId}
            packageData={packageData}
            onGenerate={() => void generatePackage()}
            generating={evidence.isPending}
            canGenerateRole={access.can('evidence:package')}
            roleLabel={access.policy.label}
          />
          <section className="h-[360px] overflow-hidden rounded-lg border border-border-subtle bg-bg-deep">
            <EscalationList />
          </section>
          <ScalePanel />
        </div>
      </div>
    </div>
  )
}

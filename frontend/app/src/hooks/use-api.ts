// ============================================================================
// TanStack Query Hooks -- REST endpoint wrappers
// ============================================================================

import { useQuery, useMutation, useQueryClient, type QueryClient } from '@tanstack/react-query'
import {
  fetchSnapshot,
  fetchTopology,
  fetchCircuitBreakerStatus,
  fetchVerdicts,
  fetchAttackTypes,
  fetchPS3Scenarios,
  launchPS3Scenario,
  launchAttack,
  stopAttack,
  stopAllAttacks,
  fetchActiveScenarios,
  fetchHistory,
  fetchEscalations,
  decideEscalation,
  fetchRecentBlocks,
  fetchEnums,
  injectEvent,
  fetchEventLabTemplates,
  previewEventLabRun,
  createEventLabRun,
  fetchEventLabRun,
  fetchEventLabExplainability,
  fetchCountermeasureProposals,
  approveCountermeasure,
  rejectCountermeasure,
  fetchInvestigation,
  fetchRiskDistribution,
  fetchFraudTypology,
  fetchVelocityTrends,
  fetchTemporalHeatmap,
  fetchThreatSummary,
  fetchGlobalImportance,
  fetchDriftStatus,
  fetchNLQuery,
  fetchLLMStatus,
  fetchConsortiumStatus,
  fetchConsortiumAlerts,
  publishConsortiumAlert,
  checkConsortiumAccount,
  fetchCaseTrace,
  createEvidencePackage,
  fetchPS3Readiness,
  fetchIntelSources,
  fetchIntelSignals,
  fetchIntelTrends,
  fetchIntelPlaybooks,
  fetchIntelCockpit,
  fetchIntelMedia,
  fetchIntelTuningStatus,
  refreshIntel,
  simulateIntelSignal,
} from '@/lib/api-client'
import type {
  CountermeasureProposal,
  CountermeasureProposalsResponse,
  Escalation,
  LaunchRequest,
  InjectEventRequest,
  PS3LaunchRequest,
  EventLabRequest,
  EventLabRunRequest,
} from '@/lib/types'
import { useActivityStore } from '@/stores/use-activity-store'
import { useUIStore } from '@/stores/use-ui-store'
import { hasPermission, type Permission } from '@/lib/rbac'

const LIVE_FRAUD_QUERY_KEYS = [
  ['snapshot'],
  ['topology'],
  ['circuit-breaker'],
  ['verdicts'],
  ['escalations'],
  ['recent-blocks'],
  ['case-trace'],
  ['active-scenarios'],
  ['scenario-history'],
  ['ps3-readiness'],
  ['risk-distribution'],
  ['fraud-typology'],
  ['velocity-trends'],
  ['temporal-heatmap'],
  ['threat-summary'],
  ['global-importance'],
  ['drift-status'],
  ['fraud'],
] as const

function invalidateLiveFraudQueries(qc: QueryClient) {
  LIVE_FRAUD_QUERY_KEYS.forEach((queryKey) => {
    void qc.invalidateQueries({ queryKey })
  })
}

function invalidateCountermeasureDecisionQueries(qc: QueryClient, runId?: string | null) {
  void qc.invalidateQueries({ queryKey: ['countermeasure-proposals'] })
  void qc.invalidateQueries({ queryKey: runId ? ['event-lab-run', runId] : ['event-lab-run'] })
  void qc.invalidateQueries({ queryKey: runId ? ['event-lab-explainability', runId] : ['event-lab-explainability'] })
  invalidateLiveFraudQueries(qc)
}

function useHasRolePermission(permission: Permission) {
  const currentRole = useUIStore((s) => s.currentRole)
  return {
    currentRole,
    enabled: hasPermission(currentRole, permission),
  }
}

function useHasAnyRolePermission(permissions: Permission[]) {
  const currentRole = useUIStore((s) => s.currentRole)
  return {
    currentRole,
    enabled: permissions.some((permission) => hasPermission(currentRole, permission)),
  }
}

function cacheAnalystDecision(qc: QueryClient, analystCase: Escalation) {
  qc.setQueryData<Escalation[]>(['escalations'], (current) => {
    if (!current?.length) return [analystCase]

    const matchesCase = (item: Escalation) =>
      item.ack_id === analystCase.ack_id ||
      (item.case_id !== undefined && item.case_id === analystCase.case_id) ||
      (item.escalation_id !== undefined && item.escalation_id === analystCase.escalation_id)

    const found = current.some(matchesCase)
    if (!found) return [analystCase, ...current]
    return current.map((item) => (matchesCase(item) ? analystCase : item))
  })
}

function replaceCountermeasureProposal(
  current: CountermeasureProposalsResponse | undefined,
  proposal: CountermeasureProposal,
): CountermeasureProposalsResponse {
  const existing = current?.proposals ?? []
  const matchesProposal = (item: CountermeasureProposal) => item.proposal_id === proposal.proposal_id
  const found = existing.some(matchesProposal)
  const proposals = found
    ? existing.map((item) => (matchesProposal(item) ? proposal : item))
    : [proposal, ...existing]
  return {
    count: proposals.length,
    proposals,
    generated_at: proposal.updated_at ?? current?.generated_at ?? Date.now() / 1000,
  }
}

function cacheCountermeasureDecision(qc: QueryClient, proposal: CountermeasureProposal) {
  const update = (current: CountermeasureProposalsResponse | undefined) =>
    replaceCountermeasureProposal(current, proposal)
  qc.setQueryData<CountermeasureProposalsResponse>(['countermeasure-proposals', 'all'], update)
  qc.setQueryData<CountermeasureProposalsResponse>(['countermeasure-proposals', proposal.run_id], update)
  qc.setQueriesData<CountermeasureProposalsResponse>(
    { queryKey: ['countermeasure-proposals'] },
    update,
  )
}

// -- Dashboard hydration --

export function useSnapshot() {
  return useQuery({
    queryKey: ['snapshot'],
    queryFn: fetchSnapshot,
    staleTime: 5_000,
    refetchInterval: 10_000,
    refetchOnWindowFocus: false,
  })
}

export function useTopology(limit = 300) {
  return useQuery({
    queryKey: ['topology', limit],
    queryFn: () => fetchTopology(limit),
    staleTime: 30_000,
    refetchOnWindowFocus: false,
  })
}

export function useCircuitBreakerStatus() {
  return useQuery({
    queryKey: ['circuit-breaker'],
    queryFn: fetchCircuitBreakerStatus,
    staleTime: 5_000,
    refetchOnWindowFocus: false,
  })
}

export function useVerdicts(limit = 20) {
  return useQuery({
    queryKey: ['verdicts', limit],
    queryFn: () => fetchVerdicts(limit),
    staleTime: 10_000,
    refetchOnWindowFocus: false,
  })
}

// -- Simulation --

export function useAttackTypes() {
  return useQuery({
    queryKey: ['attack-types'],
    queryFn: fetchAttackTypes,
    staleTime: 60_000,
  })
}

export function usePS3Scenarios() {
  return useQuery({
    queryKey: ['ps3-scenarios'],
    queryFn: fetchPS3Scenarios,
    staleTime: 300_000,
  })
}

export function useLaunchPS3Scenario() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (body: PS3LaunchRequest) => launchPS3Scenario(body),
    onSuccess: (data) => {
      invalidateLiveFraudQueries(qc)
      void qc.invalidateQueries({ queryKey: ['case-trace', data.primary_case_id] })
    },
  })
}

export function useLaunchAttack() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (body: LaunchRequest) => launchAttack(body),
    onSuccess: () => {
      invalidateLiveFraudQueries(qc)
    },
  })
}

export function useStopAttack() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (scenarioId: string) => stopAttack(scenarioId),
    onSuccess: () => {
      invalidateLiveFraudQueries(qc)
    },
  })
}

export function useStopAllAttacks() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: stopAllAttacks,
    onSuccess: () => {
      invalidateLiveFraudQueries(qc)
    },
  })
}

export function useActiveScenarios(enabled = true) {
  return useQuery({
    queryKey: ['active-scenarios'],
    queryFn: fetchActiveScenarios,
    refetchInterval: enabled ? 8_000 : false,
    enabled,
  })
}

export function useScenarioHistory() {
  return useQuery({
    queryKey: ['scenario-history'],
    queryFn: fetchHistory,
    staleTime: 5_000,
  })
}

// -- Analyst --

export function useEscalations() {
  return useQuery({
    queryKey: ['escalations'],
    queryFn: fetchEscalations,
    staleTime: 10_000,
    refetchInterval: 15_000,
    refetchOnWindowFocus: false,
  })
}

export function useDecideEscalation() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (body: { ackId: string; decision: 'approve' | 'reject' | 'escalate'; reason: string }) =>
      decideEscalation(body.ackId, {
        decision: body.decision,
        reason: body.reason,
      }),
    onSuccess: (analystCase) => {
      cacheAnalystDecision(qc, analystCase)
      useActivityStore.getState().onAnalystActivity({
        type: 'escalation_decided',
        case: analystCase,
      })
      void qc.invalidateQueries({ queryKey: ['escalations'] })
      invalidateLiveFraudQueries(qc)
    },
  })
}

// -- Blockchain --

export function useRecentBlocks(limit = 50) {
  return useQuery({
    queryKey: ['recent-blocks', limit],
    queryFn: () => fetchRecentBlocks(limit),
    staleTime: 5_000,
    refetchInterval: 8_000,
    refetchOnWindowFocus: false,
  })
}

// -- Enums / Custom Event Injection --

export function useEnums() {
  return useQuery({
    queryKey: ['enums'],
    queryFn: fetchEnums,
    staleTime: 300_000,
  })
}

export function useInjectEvent() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (body: InjectEventRequest) => injectEvent(body),
    onSuccess: () => {
      invalidateLiveFraudQueries(qc)
    },
  })
}

export function useEventLabTemplates() {
  return useQuery({
    queryKey: ['event-lab-templates'],
    queryFn: fetchEventLabTemplates,
    staleTime: 10_000,
    refetchInterval: 15_000,
    refetchOnWindowFocus: false,
  })
}

export function usePreviewEventLabRun() {
  return useMutation({
    mutationFn: (body: EventLabRequest) => previewEventLabRun(body),
  })
}

export function useCreateEventLabRun() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (body: EventLabRunRequest) => createEventLabRun(body),
    onSuccess: (run) => {
      invalidateCountermeasureDecisionQueries(qc, run.run_id)
    },
  })
}

export function useEventLabRun(runId: string | null) {
  return useQuery({
    queryKey: ['event-lab-run', runId],
    queryFn: () => fetchEventLabRun(runId!),
    enabled: !!runId,
    staleTime: 1_000,
    refetchInterval: runId ? 2_000 : false,
    refetchOnWindowFocus: false,
  })
}

export function useEventLabExplainability(runId: string | null) {
  return useQuery({
    queryKey: ['event-lab-explainability', runId],
    queryFn: () => fetchEventLabExplainability(runId!),
    enabled: !!runId,
    staleTime: 1_000,
    refetchInterval: runId ? 2_000 : false,
    refetchOnWindowFocus: false,
  })
}

export function useCountermeasureProposals(runId?: string | null) {
  return useQuery({
    queryKey: ['countermeasure-proposals', runId ?? 'all'],
    queryFn: () => fetchCountermeasureProposals(runId ?? undefined),
    staleTime: 1_000,
    refetchInterval: 8_000,
    refetchOnWindowFocus: false,
  })
}

export function useApproveCountermeasure() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (proposalId: string) => approveCountermeasure(proposalId),
    onSuccess: (proposal) => {
      cacheCountermeasureDecision(qc, proposal)
      useActivityStore.getState().onCountermeasureActivity({
        type: proposal.status === 'executed' ? 'action_executed' : 'proposal_approved',
        proposal,
      })
      invalidateCountermeasureDecisionQueries(qc, proposal.run_id)
    },
  })
}

export function useRejectCountermeasure() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (proposalId: string) => rejectCountermeasure(proposalId),
    onSuccess: (proposal) => {
      cacheCountermeasureDecision(qc, proposal)
      useActivityStore.getState().onCountermeasureActivity({
        type: 'proposal_rejected',
        proposal,
      })
      invalidateCountermeasureDecisionQueries(qc, proposal.run_id)
    },
  })
}

// -- Investigation --

export function useInvestigation(txnId: string | null) {
  return useQuery({
    queryKey: ['investigation', txnId],
    queryFn: () => fetchInvestigation(txnId!),
    enabled: !!txnId,
    staleTime: 30_000,
  })
}

export function useCaseTrace(caseId: string | null) {
  return useQuery({
    queryKey: ['case-trace', caseId],
    queryFn: () => fetchCaseTrace(caseId!),
    enabled: !!caseId,
    staleTime: 2_000,
    refetchInterval: caseId ? 2_500 : false,
    refetchOnWindowFocus: false,
  })
}

export function useCreateEvidencePackage() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (caseId: string) => createEvidencePackage(caseId),
    onSuccess: (_package, caseId) => {
      void qc.invalidateQueries({ queryKey: ['case-trace', caseId] })
      void qc.invalidateQueries({ queryKey: ['case-trace'] })
      invalidateLiveFraudQueries(qc)
    },
  })
}

export function usePS3Readiness() {
  return useQuery({
    queryKey: ['ps3-readiness'],
    queryFn: fetchPS3Readiness,
    staleTime: 5_000,
    refetchInterval: 10_000,
    refetchOnWindowFocus: false,
  })
}

// -- Pre-Fraud Intelligence --

export function useIntelSources() {
  return useQuery({
    queryKey: ['intel-sources'],
    queryFn: fetchIntelSources,
    staleTime: 60_000,
    refetchOnWindowFocus: false,
  })
}

export function useIntelSignals() {
  return useQuery({
    queryKey: ['intel-signals'],
    queryFn: () => fetchIntelSignals({ min_trust: 0.0 }),
    staleTime: 5_000,
    refetchInterval: 10_000,
    refetchOnWindowFocus: false,
  })
}

export function useIntelTrends() {
  return useQuery({
    queryKey: ['intel-trends'],
    queryFn: fetchIntelTrends,
    staleTime: 5_000,
    refetchInterval: 10_000,
    refetchOnWindowFocus: false,
  })
}

export function useIntelPlaybooks() {
  return useQuery({
    queryKey: ['intel-playbooks'],
    queryFn: fetchIntelPlaybooks,
    staleTime: 5_000,
    refetchInterval: 10_000,
    refetchOnWindowFocus: false,
  })
}

export function useIntelCockpit() {
  return useQuery({
    queryKey: ['intel-cockpit'],
    queryFn: fetchIntelCockpit,
    staleTime: 2_000,
    refetchInterval: 5_000,
    refetchOnWindowFocus: false,
  })
}

export function useIntelMedia() {
  return useQuery({
    queryKey: ['intel-media'],
    queryFn: fetchIntelMedia,
    staleTime: 5_000,
    refetchInterval: 12_000,
    refetchOnWindowFocus: false,
  })
}

export function useIntelTuningStatus() {
  return useQuery({
    queryKey: ['intel-tuning-status'],
    queryFn: fetchIntelTuningStatus,
    staleTime: 5_000,
    refetchInterval: 10_000,
    refetchOnWindowFocus: false,
  })
}

export function useRefreshIntel() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (seed?: number) => refreshIntel(seed),
    onSuccess: () => {
      void qc.invalidateQueries({ queryKey: ['intel-sources'] })
      void qc.invalidateQueries({ queryKey: ['intel-signals'] })
      void qc.invalidateQueries({ queryKey: ['intel-trends'] })
      void qc.invalidateQueries({ queryKey: ['intel-playbooks'] })
      void qc.invalidateQueries({ queryKey: ['intel-cockpit'] })
      void qc.invalidateQueries({ queryKey: ['intel-media'] })
      void qc.invalidateQueries({ queryKey: ['intel-tuning-status'] })
      void qc.invalidateQueries({ queryKey: ['ps3-readiness'] })
    },
  })
}

export function useSimulateIntelSignal() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (scenario?: string) => simulateIntelSignal(scenario ?? 'digital_arrest_mule'),
    onSuccess: () => {
      void qc.invalidateQueries({ queryKey: ['intel-sources'] })
      void qc.invalidateQueries({ queryKey: ['intel-signals'] })
      void qc.invalidateQueries({ queryKey: ['intel-trends'] })
      void qc.invalidateQueries({ queryKey: ['intel-playbooks'] })
      void qc.invalidateQueries({ queryKey: ['intel-cockpit'] })
      void qc.invalidateQueries({ queryKey: ['intel-media'] })
      void qc.invalidateQueries({ queryKey: ['intel-tuning-status'] })
      void qc.invalidateQueries({ queryKey: ['ps3-readiness'] })
    },
  })
}

// -- Analytics --

export function useRiskDistribution() {
  const { currentRole, enabled } = useHasRolePermission('analytics:view')
  return useQuery({
    queryKey: ['risk-distribution', currentRole],
    queryFn: fetchRiskDistribution,
    enabled,
    staleTime: 5_000,
    refetchInterval: 8_000,
    refetchOnWindowFocus: false,
  })
}

export function useFraudTypology() {
  const { currentRole, enabled } = useHasRolePermission('analytics:view')
  return useQuery({
    queryKey: ['fraud-typology', currentRole],
    queryFn: fetchFraudTypology,
    enabled,
    staleTime: 5_000,
    refetchInterval: 8_000,
    refetchOnWindowFocus: false,
  })
}

export function useVelocityTrends(windowMinutes = 30, topN = 10) {
  const { currentRole, enabled } = useHasRolePermission('analytics:view')
  return useQuery({
    queryKey: ['velocity-trends', currentRole, windowMinutes, topN],
    queryFn: () => fetchVelocityTrends(windowMinutes, topN),
    enabled,
    staleTime: 10_000,
    refetchInterval: 15_000,
    refetchOnWindowFocus: false,
  })
}

export function useTemporalHeatmap(bucketSeconds = 60, lookbackMinutes = 30) {
  const { currentRole, enabled } = useHasRolePermission('analytics:view')
  return useQuery({
    queryKey: ['temporal-heatmap', currentRole, bucketSeconds, lookbackMinutes],
    queryFn: () => fetchTemporalHeatmap(bucketSeconds, lookbackMinutes),
    enabled,
    staleTime: 10_000,
    refetchInterval: 15_000,
    refetchOnWindowFocus: false,
  })
}

export function useThreatSummary() {
  const { currentRole, enabled } = useHasRolePermission('analytics:view')
  return useQuery({
    queryKey: ['threat-summary', currentRole],
    queryFn: fetchThreatSummary,
    enabled,
    staleTime: 3_000,
    refetchInterval: 5_000,
    refetchOnWindowFocus: false,
  })
}

// -- Intelligence --

export function useGlobalImportance() {
  const { currentRole, enabled } = useHasAnyRolePermission(['explain:view', 'model:feedback', 'audit:review'])
  return useQuery({
    queryKey: ['global-importance', currentRole],
    queryFn: fetchGlobalImportance,
    enabled,
    staleTime: 30_000,
    refetchOnWindowFocus: false,
  })
}

export function useDriftStatus() {
  const { currentRole, enabled } = useHasAnyRolePermission(['model:feedback', 'risk:view', 'system:view', 'audit:review'])
  return useQuery({
    queryKey: ['drift-status', currentRole],
    queryFn: fetchDriftStatus,
    enabled,
    staleTime: 10_000,
    refetchInterval: enabled ? 15_000 : false,
    refetchOnWindowFocus: false,
  })
}

export function useNLQuery() {
  return useMutation({
    mutationFn: (question: string) => fetchNLQuery(question),
  })
}

export function useLLMStatus() {
  return useQuery({
    queryKey: ['llm-status'],
    queryFn: fetchLLMStatus,
    staleTime: 5_000,
    refetchInterval: 10_000,
    refetchOnWindowFocus: false,
  })
}

export function useConsortiumStatus() {
  const { currentRole, enabled } = useHasAnyRolePermission(['cfr:check', 'consortium:publish', 'audit:review'])
  return useQuery({
    queryKey: ['consortium-status', currentRole],
    queryFn: fetchConsortiumStatus,
    enabled,
    staleTime: 10_000,
    refetchInterval: enabled ? 20_000 : false,
    refetchOnWindowFocus: false,
  })
}

export function useConsortiumAlerts(fraudType?: number, severityMin = 1, limit = 50) {
  const { currentRole, enabled } = useHasAnyRolePermission(['cfr:check', 'consortium:publish', 'audit:review'])
  return useQuery({
    queryKey: ['consortium-alerts', currentRole, fraudType, severityMin, limit],
    queryFn: () => fetchConsortiumAlerts(fraudType, severityMin, limit),
    enabled,
    staleTime: 10_000,
    refetchInterval: enabled ? 20_000 : false,
    refetchOnWindowFocus: false,
  })
}

export function usePublishConsortiumAlert() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: publishConsortiumAlert,
    onSuccess: () => {
      void qc.invalidateQueries({ queryKey: ['consortium-alerts'] })
      void qc.invalidateQueries({ queryKey: ['consortium-status'] })
    },
  })
}

export function useCheckConsortiumAccount() {
  return useMutation({
    mutationFn: (accountId: string) => checkConsortiumAccount(accountId),
  })
}

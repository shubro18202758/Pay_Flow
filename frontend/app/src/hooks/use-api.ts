// ============================================================================
// TanStack Query Hooks -- REST endpoint wrappers
// ============================================================================

import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
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
import type { LaunchRequest, InjectEventRequest, PS3LaunchRequest, EventLabRequest, EventLabRunRequest } from '@/lib/types'

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
      void qc.invalidateQueries({ queryKey: ['active-scenarios'] })
      void qc.invalidateQueries({ queryKey: ['scenario-history'] })
      void qc.invalidateQueries({ queryKey: ['case-trace', data.primary_case_id] })
      void qc.invalidateQueries({ queryKey: ['ps3-readiness'] })
    },
  })
}

export function useLaunchAttack() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (body: LaunchRequest) => launchAttack(body),
    onSuccess: () => {
      void qc.invalidateQueries({ queryKey: ['active-scenarios'] })
      void qc.invalidateQueries({ queryKey: ['scenario-history'] })
    },
  })
}

export function useStopAttack() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (scenarioId: string) => stopAttack(scenarioId),
    onSuccess: () => {
      void qc.invalidateQueries({ queryKey: ['active-scenarios'] })
      void qc.invalidateQueries({ queryKey: ['scenario-history'] })
    },
  })
}

export function useStopAllAttacks() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: stopAllAttacks,
    onSuccess: () => {
      void qc.invalidateQueries({ queryKey: ['active-scenarios'] })
      void qc.invalidateQueries({ queryKey: ['scenario-history'] })
    },
  })
}

export function useActiveScenarios(enabled = true) {
  return useQuery({
    queryKey: ['active-scenarios'],
    queryFn: fetchActiveScenarios,
    refetchInterval: enabled ? 2_000 : false,
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
  return useMutation({
    mutationFn: (body: InjectEventRequest) => injectEvent(body),
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
      void qc.invalidateQueries({ queryKey: ['event-lab-run', run.run_id] })
      void qc.invalidateQueries({ queryKey: ['event-lab-explainability', run.run_id] })
      void qc.invalidateQueries({ queryKey: ['countermeasure-proposals'] })
      void qc.invalidateQueries({ queryKey: ['active-scenarios'] })
      void qc.invalidateQueries({ queryKey: ['scenario-history'] })
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
    refetchInterval: 3_000,
    refetchOnWindowFocus: false,
  })
}

export function useApproveCountermeasure() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (proposalId: string) => approveCountermeasure(proposalId),
    onSuccess: (proposal) => {
      void qc.invalidateQueries({ queryKey: ['countermeasure-proposals'] })
      void qc.invalidateQueries({ queryKey: ['event-lab-run', proposal.run_id] })
      void qc.invalidateQueries({ queryKey: ['event-lab-explainability', proposal.run_id] })
      void qc.invalidateQueries({ queryKey: ['circuit-breaker'] })
      void qc.invalidateQueries({ queryKey: ['snapshot'] })
    },
  })
}

export function useRejectCountermeasure() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (proposalId: string) => rejectCountermeasure(proposalId),
    onSuccess: (proposal) => {
      void qc.invalidateQueries({ queryKey: ['countermeasure-proposals'] })
      void qc.invalidateQueries({ queryKey: ['event-lab-run', proposal.run_id] })
      void qc.invalidateQueries({ queryKey: ['event-lab-explainability', proposal.run_id] })
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
  return useMutation({
    mutationFn: (caseId: string) => createEvidencePackage(caseId),
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
  return useQuery({
    queryKey: ['risk-distribution'],
    queryFn: fetchRiskDistribution,
    staleTime: 5_000,
    refetchInterval: 8_000,
    refetchOnWindowFocus: false,
  })
}

export function useFraudTypology() {
  return useQuery({
    queryKey: ['fraud-typology'],
    queryFn: fetchFraudTypology,
    staleTime: 5_000,
    refetchInterval: 8_000,
    refetchOnWindowFocus: false,
  })
}

export function useVelocityTrends(windowMinutes = 30, topN = 10) {
  return useQuery({
    queryKey: ['velocity-trends', windowMinutes, topN],
    queryFn: () => fetchVelocityTrends(windowMinutes, topN),
    staleTime: 10_000,
    refetchInterval: 15_000,
    refetchOnWindowFocus: false,
  })
}

export function useTemporalHeatmap(bucketSeconds = 60, lookbackMinutes = 30) {
  return useQuery({
    queryKey: ['temporal-heatmap', bucketSeconds, lookbackMinutes],
    queryFn: () => fetchTemporalHeatmap(bucketSeconds, lookbackMinutes),
    staleTime: 10_000,
    refetchInterval: 15_000,
    refetchOnWindowFocus: false,
  })
}

export function useThreatSummary() {
  return useQuery({
    queryKey: ['threat-summary'],
    queryFn: fetchThreatSummary,
    staleTime: 3_000,
    refetchInterval: 5_000,
    refetchOnWindowFocus: false,
  })
}

// -- Intelligence --

export function useGlobalImportance() {
  return useQuery({
    queryKey: ['global-importance'],
    queryFn: fetchGlobalImportance,
    staleTime: 30_000,
    refetchOnWindowFocus: false,
  })
}

export function useDriftStatus() {
  return useQuery({
    queryKey: ['drift-status'],
    queryFn: fetchDriftStatus,
    staleTime: 10_000,
    refetchInterval: 15_000,
    refetchOnWindowFocus: false,
  })
}

export function useNLQuery() {
  return useMutation({
    mutationFn: (question: string) => fetchNLQuery(question),
  })
}

export function useConsortiumStatus() {
  return useQuery({
    queryKey: ['consortium-status'],
    queryFn: fetchConsortiumStatus,
    staleTime: 10_000,
    refetchInterval: 20_000,
    refetchOnWindowFocus: false,
  })
}

export function useConsortiumAlerts(fraudType?: number, severityMin = 1, limit = 50) {
  return useQuery({
    queryKey: ['consortium-alerts', fraudType, severityMin, limit],
    queryFn: () => fetchConsortiumAlerts(fraudType, severityMin, limit),
    staleTime: 10_000,
    refetchInterval: 20_000,
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

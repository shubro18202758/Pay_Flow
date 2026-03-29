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
  launchAttack,
  stopAttack,
  stopAllAttacks,
  fetchActiveScenarios,
  fetchHistory,
  fetchEscalations,
  fetchRecentBlocks,
  fetchEnums,
  injectEvent,
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
} from '@/lib/api-client'
import type { LaunchRequest, InjectEventRequest } from '@/lib/types'

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

export function useTopology(limit = 500) {
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

// -- Investigation --

export function useInvestigation(txnId: string | null) {
  return useQuery({
    queryKey: ['investigation', txnId],
    queryFn: () => fetchInvestigation(txnId!),
    enabled: !!txnId,
    staleTime: 30_000,
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

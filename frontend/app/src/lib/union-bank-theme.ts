import type { LucideIcon } from 'lucide-react'

export const UBI_BLUE = '#00579C'
export const UBI_RED = '#DA251C'
export const UBI_MUTED = '#617189'

export const UBI_TONE = {
  blueText: 'text-[#00579C]',
  redText: 'text-[#DA251C]',
  mutedText: 'text-text-muted',
  blueBadge: 'border-[#00579C]/25 bg-[#00579C]/10 text-[#00579C]',
  redBadge: 'border-[#DA251C]/25 bg-[#DA251C]/10 text-[#DA251C]',
  redBadgeStrong: 'border-[#DA251C]/35 bg-[#DA251C]/15 text-[#DA251C]',
  neutralBadge: 'border-border-subtle bg-bg-overlay/60 text-text-muted',
  blueBar: 'bg-[#00579C]',
  redBar: 'bg-[#DA251C]',
  mutedBar: 'bg-[#617189]',
} as const

export function riskTierBadgeClass(tier: string | undefined): string {
  if (tier === 'critical' || tier === 'high') return UBI_TONE.redBadgeStrong
  if (tier === 'medium') return UBI_TONE.redBadge
  return UBI_TONE.blueBadge
}

export function riskScoreBarClass(score: number): string {
  if (score > 0.5) return UBI_TONE.redBar
  return UBI_TONE.blueBar
}

export function contributionToneClass(value: number): string {
  return value > 0 ? UBI_TONE.redText : UBI_TONE.blueText
}

export function latencyToneClass(ms: number): string {
  if (ms < 200) return UBI_TONE.blueText
  return UBI_TONE.redText
}

export function analystStatusBadgeClass(status: string | undefined): string {
  if (status === 'approved' || status === 'escalated') return UBI_TONE.redBadge
  if (status === 'rejected') return UBI_TONE.neutralBadge
  return UBI_TONE.blueBadge
}

export function verdictToneClass(verdict: string | undefined): string {
  if (verdict === 'FRAUDULENT' || verdict === 'FRAUD' || verdict === 'SUSPICIOUS') return UBI_TONE.redText
  return UBI_TONE.blueText
}

export type ConsensusKey = 'ml' | 'gnn' | 'graph'

export const UBI_CONSENSUS_SERIES: Array<{
  key: ConsensusKey
  label: string
  color: string
  text: string
}> = [
  { key: 'ml', label: 'ML', color: UBI_TONE.blueBar, text: UBI_TONE.blueText },
  { key: 'gnn', label: 'GNN', color: UBI_TONE.redBar, text: UBI_TONE.redText },
  { key: 'graph', label: 'Graph', color: UBI_TONE.mutedBar, text: UBI_TONE.mutedText },
]

export interface ToneIcon {
  icon: LucideIcon
  label: string
}

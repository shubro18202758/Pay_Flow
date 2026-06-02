export type Severity = 'critical' | 'high' | 'medium' | 'low' | 'escalated'

export function verdictToSeverity(verdict: string): Severity {
  const v = verdict.toLowerCase()
  if (v === 'fraudulent') return 'critical'
  if (v === 'suspicious') return 'high'
  if (v.includes('escalat')) return 'escalated'
  if (v === 'legitimate') return 'low'
  return 'medium'
}

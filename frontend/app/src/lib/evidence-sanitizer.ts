// ============================================================================
// Evidence Sanitizer -- shared guards for public fraud-rationale surfaces
// ============================================================================

export function sanitizeUnavailableGnnText(value: string): string {
  return value
    .replace(
      /(?:the\s+)?gnn score(?:\s+of)?\s*\(?-1(?:\.0)?\)?[^.]*\./gi,
      'The GNN score was unavailable and is not used as risk evidence.',
    )
    .replace(
      /\bgnn_score\s*:\s*-1(?:\.0)?\b/gi,
      'GNN unavailable',
    )
    .replace(
      /\bGNN Score\s*:\s*-1(?:\.0)?\b/g,
      'GNN unavailable',
    )
}

export function sanitizePublicTraceText(value: unknown): string {
  const text = String(value ?? '').trim()
  if (!text) return 'Evidence-rationale update recorded.'

  const blockedPrefixes = [
    'thought:',
    'chain-of-thought',
    'hidden reasoning',
    'let me think',
    'i think',
  ] as const
  const blockedPatterns = [
    /\bchain[-\s]?of[-\s]?thought\b/i,
    /\bhidden\s+reasoning\b/i,
    /\bprivate\s+reasoning\b/i,
  ]

  const publicLines: string[] = []
  let publicLength = 0
  for (const rawLine of text.replace(/\r/g, '\n').split('\n')) {
    const line = rawLine.trim().replace(/^-+\s*/, '')
    if (!line || line.startsWith('```')) continue
    const lowered = line.toLowerCase()
    if (blockedPrefixes.some((prefix) => lowered.startsWith(prefix))) continue
    if (blockedPatterns.some((pattern) => pattern.test(line))) continue
    publicLines.push(line)
    publicLength += line.length + 1
    if (publicLength >= 500) break
  }

  const publicText = publicLines.join(' ').replace(/\s+/g, ' ').trim()
  return publicText ? publicText.slice(0, 500) : 'Evidence-rationale update recorded.'
}

export function sanitizeOptionalEvidenceText(value: string | undefined): string | undefined {
  return value ? sanitizeUnavailableGnnText(sanitizePublicTraceText(value)) : value
}

export function sanitizeEvidenceList(value: string[] | undefined): string[] | undefined {
  return value?.map((item) => sanitizeUnavailableGnnText(sanitizePublicTraceText(item)))
}

import type { LLMStatusResponse } from './types'

export interface LLMRuntimeSummary {
  model: string
  statusLabel: string
  statusDetail: string
  running: boolean
  installed: boolean
  reachable: boolean
}

function nonEmptyString(value: unknown): string | null {
  return typeof value === 'string' && value.trim().length > 0 ? value.trim() : null
}

function firstString(values: unknown): string | null {
  return Array.isArray(values) ? values.map(nonEmptyString).find(Boolean) ?? null : null
}

function hasModelName(names: unknown, model: string): boolean {
  return Array.isArray(names) && names.some((name) => {
    const value = nonEmptyString(name)
    return value === model || Boolean(value?.startsWith(model))
  })
}

export function resolveLLMRuntime(
  status: LLMStatusResponse | undefined,
  options: {
    lifecycleModel?: string | null
    loading?: boolean
    error?: boolean
    fallbackModel?: string
  } = {},
): LLMRuntimeSummary {
  const model =
    nonEmptyString(options.lifecycleModel) ??
    nonEmptyString(status?.resolved_model) ??
    nonEmptyString(status?.target_model) ??
    nonEmptyString(status?.model) ??
    firstString(status?.running_models) ??
    firstString(status?.installed_models) ??
    options.fallbackModel ??
    'configured LLM'

  if (options.loading) {
    return {
      model,
      statusLabel: 'checking',
      statusDetail: 'LLM runtime status check is still in progress.',
      running: false,
      installed: false,
      reachable: false,
    }
  }

  if (options.error || status?.reachable === false) {
    return {
      model,
      statusLabel: 'status unavailable',
      statusDetail: 'Backend could not confirm the Ollama runtime; only existing lifecycle evidence is shown.',
      running: false,
      installed: false,
      reachable: false,
    }
  }

  const running = Boolean(status?.target_running || status?.running || hasModelName(status?.running_models, model))
  const installed = Boolean(status?.target_installed || hasModelName(status?.installed_models, model))

  if (running) {
    return {
      model,
      statusLabel: 'running',
      statusDetail: 'Backend reports the configured Ollama model is resident and serving requests.',
      running,
      installed: true,
      reachable: true,
    }
  }

  if (installed) {
    return {
      model,
      statusLabel: 'installed',
      statusDetail: 'Backend reports the configured Ollama model is installed but not currently resident.',
      running,
      installed,
      reachable: true,
    }
  }

  return {
    model,
    statusLabel: status ? 'not loaded' : 'unknown',
    statusDetail: status
      ? 'Backend did not report the configured model as running or installed.'
      : 'LLM runtime status has not been loaded yet.',
    running,
    installed,
    reachable: Boolean(status),
  }
}

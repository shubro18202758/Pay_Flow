import { useCallback, useEffect, useMemo, useRef, useState, type FormEvent } from 'react'
import {
  Bot,
  ClipboardList,
  Loader2,
  Radio,
  RotateCcw,
  Search,
  SendHorizontal,
  ShieldCheck,
  Sparkles,
  X,
} from 'lucide-react'
import { useLLMStatus, useNLQuery } from '@/hooks/use-api'
import { resolveLLMRuntime } from '@/lib/llm-runtime'
import { rolePolicy } from '@/lib/rbac'
import type { NLQueryMessage, NLQueryResponse } from '@/lib/types'
import { cn } from '@/lib/utils'
import { useUIStore } from '@/stores/use-ui-store'

type ChatMessage = NLQueryMessage & {
  id: string
  meta?: string
  intent?: string
  confidence?: number
  processingMs?: number
  error?: boolean
}

const STARTER_QUERIES = [
  'Where should I go to create and evaluate a custom UPI mule fraud event?',
  'Explain the PS3 proof of concept in Union Bank operating terms.',
  'What changed after the latest fraud event run across graph, analytics and reports?',
  'How does qwen3.5:4b help PayFlow without taking decision authority?',
  'Which role can approve a freeze and which role can only package evidence?',
]

const createIntroMessage = (): ChatMessage => ({
  id: 'intro',
  role: 'assistant',
  content:
    'Ask me to find a PayFlow page, explain a role gate, summarize the PS3 proof of concept, inspect live fraud metrics, or trace how Qwen, ML, graph and heuristics work together.',
  meta: 'PayFlow context loaded',
})

function formatResultMeta(result: NLQueryResponse) {
  const confidence = Math.round((result.confidence ?? 0) * 100)
  return `${result.model_used} | ${result.intent} | ${confidence}% confidence | ${Math.round(result.processing_ms)} ms`
}

export function PayFlowCopilotOverlay() {
  const open = useUIStore((s) => s.copilotOpen)
  const seed = useUIStore((s) => s.copilotSeed)
  const autoRun = useUIStore((s) => s.copilotAutoRun)
  const openSeq = useUIStore((s) => s.copilotOpenSeq)
  const closeCopilot = useUIStore((s) => s.closeCopilot)
  const openCopilot = useUIStore((s) => s.openCopilot)
  const activeTab = useUIStore((s) => s.activeTab)
  const currentRole = useUIStore((s) => s.currentRole)
  const policy = rolePolicy(currentRole)

  const nlQuery = useNLQuery()
  const llmStatus = useLLMStatus()
  const [draft, setDraft] = useState('')
  const [messages, setMessages] = useState<ChatMessage[]>(() => [createIntroMessage()])
  const inputRef = useRef<HTMLInputElement | null>(null)
  const scrollRef = useRef<HTMLDivElement | null>(null)
  const lastAutoRunSeq = useRef(0)
  const messagesRef = useRef<ChatMessage[]>(messages)
  const queryPendingRef = useRef(false)
  const runNLQuery = nlQuery.mutateAsync

  useEffect(() => {
    messagesRef.current = messages
  }, [messages])

  useEffect(() => {
    queryPendingRef.current = nlQuery.isPending
  }, [nlQuery.isPending])

  const qwenRuntime = useMemo(
    () =>
      resolveLLMRuntime(llmStatus.data, {
        loading: llmStatus.isLoading,
        error: llmStatus.isError,
        fallbackModel: 'qwen3.5:4b',
      }),
    [llmStatus.data, llmStatus.isError, llmStatus.isLoading],
  )

  const modelLabel = qwenRuntime.model
  const runtimeLabel = `${modelLabel} ${qwenRuntime.statusLabel}`
  const qwenReady = qwenRuntime.running || qwenRuntime.installed

  const sendQuestion = useCallback(
    async (rawQuestion: string) => {
      const question = rawQuestion.trim()
      if (!question || queryPendingRef.current) return

      const userMessage: ChatMessage = {
        id: `user-${Date.now()}`,
        role: 'user',
        content: question,
      }
      const conversation = messagesRef.current
        .filter((message) => message.id !== 'intro')
        .slice(-8)
        .map(({ role, content }) => ({ role, content }))

      setMessages((current) => [...current, userMessage])
      setDraft('')

      try {
        const result = await runNLQuery({
          question,
          surface: 'global_payflow_copilot',
          active_tab: activeTab,
          conversation,
        })
        setMessages((current) => [
          ...current,
          {
            id: `assistant-${Date.now()}`,
            role: 'assistant',
            content: result.answer,
            meta: formatResultMeta(result),
            intent: result.intent,
            confidence: result.confidence,
            processingMs: result.processing_ms,
          },
        ])
      } catch (error) {
        setMessages((current) => [
          ...current,
          {
            id: `assistant-error-${Date.now()}`,
            role: 'assistant',
            content:
              error instanceof Error
                ? `The Qwen query failed: ${error.message}`
                : 'The Qwen query failed before the backend returned a response.',
            meta: 'backend query error',
            error: true,
          },
        ])
      }
    },
    [activeTab, runNLQuery],
  )

  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === 'k') {
        event.preventDefault()
        openCopilot('', false)
      }
      if (event.key === 'Escape' && open) {
        closeCopilot()
      }
    }
    window.addEventListener('keydown', onKeyDown)
    return () => window.removeEventListener('keydown', onKeyDown)
  }, [closeCopilot, open, openCopilot])

  useEffect(() => {
    if (!open) return
    setDraft(seed)
    const focusTimer = window.setTimeout(() => inputRef.current?.focus(), 40)
    return () => window.clearTimeout(focusTimer)
  }, [open, openSeq, seed])

  useEffect(() => {
    if (!open) return
    if (autoRun && seed.trim() && lastAutoRunSeq.current !== openSeq) {
      lastAutoRunSeq.current = openSeq
      const runTimer = window.setTimeout(() => void sendQuestion(seed), 60)
      return () => window.clearTimeout(runTimer)
    }
  }, [autoRun, open, openSeq, seed, sendQuestion])

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: 'smooth' })
  }, [messages, nlQuery.isPending])

  const onSubmit = (event: FormEvent) => {
    event.preventDefault()
    void sendQuestion(draft)
  }

  if (!open) return null

  return (
    <div className="fixed inset-0 z-[120] bg-[#06192d]/78 backdrop-blur-sm">
      <div className="flex h-full items-center justify-center px-4 py-5">
        <section className="flex h-[min(820px,calc(100vh-34px))] w-full max-w-6xl flex-col overflow-hidden rounded-lg border border-[#7fb6e8]/55 bg-[#f6fbff] shadow-[0_28px_80px_rgba(0,20,45,0.5)]">
          <header className="flex shrink-0 items-center justify-between gap-4 border-b border-[#c8d8e8] bg-[#00579C] px-5 py-4 text-white">
            <div className="flex min-w-0 items-center gap-3">
              <div className="flex h-11 w-11 shrink-0 items-center justify-center rounded-md bg-white text-[#00579C] shadow-sm">
                <Bot className="h-6 w-6" />
              </div>
              <div className="min-w-0">
                <div className="text-[18px] font-black tracking-normal">PayFlow Qwen Search and Copilot</div>
                <div className="mt-1 truncate text-[11px] font-semibold uppercase tracking-[0.12em] text-white/72">
                  {policy.label} | {activeTab} | X-Payflow-Role: {currentRole}
                </div>
              </div>
            </div>
            <div className="flex shrink-0 items-center gap-2">
              <RuntimeChip active={qwenReady} label={runtimeLabel} />
              <RuntimeChip active label="Ctrl+K" />
              <button
                type="button"
                onClick={closeCopilot}
                className="inline-flex h-10 items-center gap-2 rounded-md border border-white/18 bg-white/12 px-3 text-[11px] font-extrabold uppercase tracking-[0.12em] text-white hover:bg-white/18"
              >
                <X className="h-4 w-4" />
                Close
              </button>
            </div>
          </header>

          <div className="grid min-h-0 flex-1 grid-cols-1 lg:grid-cols-[330px_minmax(0,1fr)]">
            <aside className="hidden min-h-0 border-r border-[#c8d8e8] bg-[#eaf4ff] p-4 lg:flex lg:flex-col">
              <div className="rounded-md border border-[#bdd5eb] bg-white p-4 shadow-sm">
                <div className="flex items-center gap-2 text-[11px] font-extrabold uppercase tracking-[0.14em] text-[#00579C]">
                  <ShieldCheck className="h-4 w-4" />
                  Active Context
                </div>
                <div className="mt-3 space-y-2">
                  <ContextLine label="Role" value={policy.label} />
                  <ContextLine label="Domain" value={policy.domain} />
                  <ContextLine label="Tab" value={activeTab} />
                  <ContextLine label="Model" value={modelLabel} />
                </div>
              </div>

              <div className="mt-4 rounded-md border border-[#bdd5eb] bg-white p-4 shadow-sm">
                <div className="flex items-center gap-2 text-[11px] font-extrabold uppercase tracking-[0.14em] text-[#DA251C]">
                  <ClipboardList className="h-4 w-4" />
                  Ask Fast
                </div>
                <div className="mt-3 space-y-2">
                  {STARTER_QUERIES.map((query) => (
                    <button
                      key={query}
                      type="button"
                      onClick={() => void sendQuestion(query)}
                      className="w-full rounded-md border border-[#d5e3f0] bg-[#f5f9fd] px-3 py-2 text-left text-[11px] font-semibold leading-5 text-[#24364f] hover:border-[#00579C] hover:bg-white"
                    >
                      {query}
                    </button>
                  ))}
                </div>
              </div>

              <div className="mt-4 rounded-md border border-[#bdd5eb] bg-[#003f73] p-4 text-white shadow-sm">
                <div className="flex items-center gap-2 text-[11px] font-extrabold uppercase tracking-[0.14em]">
                  <Radio className="h-4 w-4" />
                  Live Query Path
                </div>
                <div className="mt-3 space-y-2 font-mono text-[10px] text-white/72">
                  <div>POST /api/v1/intelligence/query</div>
                  <div>model={modelLabel}</div>
                  <div>surface=global_payflow_copilot</div>
                </div>
              </div>
            </aside>

            <div className="flex min-h-0 flex-col bg-white">
              <div
                ref={scrollRef}
                className="custom-scrollbar min-h-0 flex-1 overflow-y-auto bg-[linear-gradient(180deg,#f8fbff_0%,#ffffff_42%,#eff7ff_100%)] p-4"
              >
                <div className="mx-auto flex max-w-3xl flex-col gap-3">
                  {messages.map((message) => (
                    <ChatBubble key={message.id} message={message} />
                  ))}
                  {nlQuery.isPending && (
                    <div className="self-start rounded-md border border-[#bdd5eb] bg-white px-4 py-3 shadow-sm">
                      <div className="flex items-center gap-2 text-[12px] font-bold text-[#00579C]">
                        <Loader2 className="h-4 w-4 animate-spin" />
                        qwen3.5:4b is reading PayFlow context and live runtime state
                      </div>
                    </div>
                  )}
                </div>
              </div>

              <form onSubmit={onSubmit} className="shrink-0 border-t border-[#c8d8e8] bg-[#f4f9ff] p-4">
                <div className="mx-auto flex max-w-3xl items-center gap-2 rounded-full border border-[#bfd4ea] bg-white p-2 shadow-[0_8px_22px_rgba(0,70,130,0.08)] focus-within:border-[#00579C] focus-within:ring-4 focus-within:ring-[#00579C]/10">
                  <Search className="ml-3 h-5 w-5 shrink-0 text-[#00579C]" />
                  <input
                    ref={inputRef}
                    value={draft}
                    onChange={(event) => setDraft(event.target.value)}
                    placeholder="Search PayFlow or ask Qwen about any page, role, event run, report, graph or proof of concept"
                    className="h-11 min-w-0 flex-1 bg-transparent text-[14px] font-semibold text-[#111827] outline-none placeholder:text-[#7b8ba0]"
                  />
                  <button
                    type="button"
                    onClick={() => {
                      setMessages([createIntroMessage()])
                      setDraft('')
                    }}
                    className="hidden h-10 items-center gap-2 rounded-full border border-[#d5e3f0] px-3 text-[10px] font-extrabold uppercase tracking-[0.1em] text-[#00579C] hover:bg-[#eaf4ff] sm:inline-flex"
                  >
                    <RotateCcw className="h-3.5 w-3.5" />
                    Reset
                  </button>
                  <button
                    type="submit"
                    disabled={!draft.trim() || nlQuery.isPending}
                    className="inline-flex h-10 items-center gap-2 rounded-full bg-[#DA251C] px-4 text-[11px] font-extrabold uppercase tracking-[0.12em] text-white shadow-[0_8px_18px_rgba(218,37,28,0.22)] disabled:cursor-not-allowed disabled:opacity-55"
                  >
                    {nlQuery.isPending ? <Loader2 className="h-4 w-4 animate-spin" /> : <SendHorizontal className="h-4 w-4" />}
                    Ask
                  </button>
                </div>
              </form>
            </div>
          </div>
        </section>
      </div>
    </div>
  )
}

function RuntimeChip({ active, label }: { active: boolean; label: string }) {
  return (
    <span
      className={cn(
        'hidden rounded-full border px-3 py-2 font-mono text-[10px] font-extrabold uppercase tracking-[0.08em] sm:inline-flex',
        active
          ? 'border-emerald-300/40 bg-emerald-400/14 text-emerald-100'
          : 'border-[#ffb2a9]/45 bg-[#DA251C]/22 text-[#ffe6e2]',
      )}
    >
      {label}
    </span>
  )
}

function ContextLine({ label, value }: { label: string; value: string }) {
  return (
    <div className="grid grid-cols-[74px_minmax(0,1fr)] gap-2 rounded-md bg-[#f5f9fd] px-3 py-2">
      <div className="text-[9px] font-extrabold uppercase tracking-[0.12em] text-[#7b8ba0]">{label}</div>
      <div className="truncate text-[11px] font-bold text-[#24364f]" title={value}>
        {value}
      </div>
    </div>
  )
}

function ChatBubble({ message }: { message: ChatMessage }) {
  const user = message.role === 'user'
  const lines = message.content.split('\n').filter(Boolean)
  return (
    <article
      className={cn(
        'max-w-[88%] rounded-lg border px-4 py-3 shadow-sm',
        user
          ? 'self-end border-[#00579C] bg-[#00579C] text-white'
          : message.error
            ? 'self-start border-[#f3b4af] bg-[#fff5f4] text-[#682019]'
            : 'self-start border-[#c8d8e8] bg-white text-[#24364f]',
      )}
    >
      <div className="mb-2 flex items-center gap-2 text-[10px] font-extrabold uppercase tracking-[0.13em] opacity-80">
        {user ? <Search className="h-3.5 w-3.5" /> : <Sparkles className="h-3.5 w-3.5 text-[#00579C]" />}
        {user ? 'Search Query' : 'Qwen Response'}
      </div>
      <div className="space-y-1 text-[13px] font-medium leading-6">
        {lines.map((line, index) => (
          <p key={`${message.id}-${index}`}>{line}</p>
        ))}
      </div>
      {message.meta && (
        <div
          className={cn(
            'mt-3 rounded-md px-2 py-1 font-mono text-[10px]',
            user ? 'bg-white/12 text-white/70' : 'bg-[#edf5ff] text-[#617189]',
          )}
        >
          {message.meta}
        </div>
      )}
    </article>
  )
}

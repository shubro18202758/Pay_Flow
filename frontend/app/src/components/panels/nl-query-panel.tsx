// ============================================================================
// NL Query Panel -- Natural language analyst queries powered by Qwen 3.5
// ============================================================================

import { useState, useRef, useEffect, useCallback } from 'react'
import { MessageSquare, Send, Loader2, Sparkles, Clock, Target, Bot } from 'lucide-react'
import { cn } from '@/lib/utils'
import { useNLQuery } from '@/hooks/use-api'
import type { NLQueryResponse } from '@/lib/types'

interface ChatMessage {
  id: string
  role: 'user' | 'assistant'
  content: string
  meta?: {
    intent: string
    confidence: number
    processing_ms: number
    model_used: string
    sources: string[]
  }
  timestamp: number
}

const SUGGESTED_QUERIES = [
  'How many mule networks have been detected?',
  'What is the current system risk level?',
  'Show me the highest-risk accounts',
  'How many transactions have been frozen?',
  'Explain the latest fraud patterns detected',
  'What is the model performance status?',
]

export function NLQueryPanel() {
  const [input, setInput] = useState('')
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const scrollRef = useRef<HTMLDivElement>(null)
  const inputRef = useRef<HTMLInputElement>(null)
  const nlQuery = useNLQuery()
  const idCounter = useRef(0)

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: 'smooth' })
  }, [messages])

  const handleSubmit = useCallback(function handleSubmit(question?: string) {
    const q = (question ?? input).trim()
    if (!q || nlQuery.isPending) return

    idCounter.current += 1
    const uid = idCounter.current
    const userMsg: ChatMessage = {
      id: `user-${uid}`,
      role: 'user',
      content: q,
      timestamp: uid,
    }
    setMessages((prev) => [...prev, userMsg])
    setInput('')

    nlQuery.mutate(q, {
      onSuccess: (result: NLQueryResponse) => {
        idCounter.current += 1
        const assistantMsg: ChatMessage = {
          id: `assistant-${idCounter.current}`,
          role: 'assistant',
          content: result.answer,
          meta: {
            intent: result.intent,
            confidence: result.confidence,
            processing_ms: result.processing_ms,
            model_used: result.model_used,
            sources: result.sources,
          },
          timestamp: idCounter.current,
        }
        setMessages((prev) => [...prev, assistantMsg])
      },
      onError: (err) => {
        idCounter.current += 1
        const errMsg: ChatMessage = {
          id: `error-${idCounter.current}`,
          role: 'assistant',
          content: `Error: ${err instanceof Error ? err.message : 'Query failed'}`,
          timestamp: idCounter.current,
        }
        setMessages((prev) => [...prev, errMsg])
      },
    })
  }, [input, nlQuery])

  return (
    <div className="flex flex-col h-full bg-bg-deep rounded-lg border border-border-subtle overflow-hidden">
      {/* Header */}
      <div className="flex items-center gap-2 px-4 py-3 border-b border-border-subtle bg-bg-surface/50">
        <MessageSquare className="w-4 h-4 text-accent-primary" />
        <span className="text-xs font-semibold text-text-primary tracking-wide">
          AI Analyst Query
        </span>
        <Sparkles className="w-3 h-3 text-amber-400/60 ml-auto" />
        <span className="text-[9px] text-text-muted font-mono">Qwen 3.5</span>
      </div>

      {/* Messages area */}
      <div ref={scrollRef} className="flex-1 min-h-0 overflow-y-auto p-4 space-y-3 custom-scrollbar">
        {messages.length === 0 ? (
          <div className="flex flex-col items-center justify-center h-full gap-4 animate-fade-in">
            <Bot className="w-10 h-10 text-accent-primary/30" />
            <p className="text-[11px] text-text-muted text-center max-w-[240px]">
              Ask questions about fraud patterns, risk levels, account activity, or system status in natural language.
            </p>
            <div className="grid grid-cols-2 gap-2 w-full max-w-[400px]">
              {SUGGESTED_QUERIES.map((q) => (
                <button
                  key={q}
                  onClick={() => handleSubmit(q)}
                  className="text-[9px] text-left px-2.5 py-2 rounded-md border border-border-subtle
                    bg-bg-surface/40 text-text-secondary hover:border-accent-primary/40 hover:text-text-primary
                    transition-colors truncate"
                  disabled={nlQuery.isPending}
                >
                  {q}
                </button>
              ))}
            </div>
          </div>
        ) : (
          messages.map((msg) => (
            <div
              key={msg.id}
              className={cn(
                'flex gap-2 animate-fade-in',
                msg.role === 'user' ? 'justify-end' : 'justify-start',
              )}
            >
              <div
                className={cn(
                  'max-w-[85%] rounded-lg px-3 py-2',
                  msg.role === 'user'
                    ? 'bg-accent-primary/15 border border-accent-primary/20 text-text-primary'
                    : 'bg-bg-surface border border-border-subtle text-text-secondary',
                )}
              >
                <p className="text-[11px] leading-relaxed whitespace-pre-wrap">{msg.content}</p>
                {msg.meta && (
                  <div className="flex flex-wrap items-center gap-2 mt-2 pt-2 border-t border-border-subtle">
                    <span className="flex items-center gap-1 text-[8px] text-text-muted">
                      <Target className="w-2.5 h-2.5" />
                      {msg.meta.intent}
                    </span>
                    <span className="flex items-center gap-1 text-[8px] text-text-muted">
                      <Clock className="w-2.5 h-2.5" />
                      {msg.meta.processing_ms}ms
                    </span>
                    <span className="text-[8px] text-text-muted font-mono">
                      {(msg.meta.confidence * 100).toFixed(0)}% conf
                    </span>
                    {msg.meta.sources.length > 0 && (
                      <span className="text-[8px] text-text-muted">
                        {msg.meta.sources.length} source{msg.meta.sources.length > 1 ? 's' : ''}
                      </span>
                    )}
                  </div>
                )}
              </div>
            </div>
          ))
        )}

        {nlQuery.isPending && (
          <div className="flex items-center gap-2 text-[10px] text-text-muted animate-fade-in">
            <Loader2 className="w-3 h-3 animate-spin text-accent-primary" />
            Analyzing with Qwen 3.5...
          </div>
        )}
      </div>

      {/* Input bar */}
      <div className="shrink-0 border-t border-border-subtle bg-bg-surface/50 p-3">
        <div className="flex items-center gap-2">
          <input
            ref={inputRef}
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && handleSubmit()}
            placeholder="Ask about fraud patterns, risk levels, accounts..."
            className="flex-1 bg-bg-deep border border-border-subtle rounded-md px-3 py-1.5
              text-[11px] text-text-primary placeholder:text-text-muted/50
              focus:outline-none focus:border-accent-primary/40 transition-colors"
            disabled={nlQuery.isPending}
          />
          <button
            onClick={() => handleSubmit()}
            disabled={!input.trim() || nlQuery.isPending}
            className="flex items-center justify-center w-7 h-7 rounded-md
              bg-accent-primary/15 border border-accent-primary/30
              text-accent-primary hover:bg-accent-primary/25
              disabled:opacity-30 disabled:cursor-not-allowed transition-colors"
          >
            {nlQuery.isPending ? (
              <Loader2 className="w-3.5 h-3.5 animate-spin" />
            ) : (
              <Send className="w-3.5 h-3.5" />
            )}
          </button>
        </div>
      </div>
    </div>
  )
}

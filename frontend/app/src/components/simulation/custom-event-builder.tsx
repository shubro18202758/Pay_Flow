// ============================================================================
// Custom Event Builder -- Full-featured dynamic event injection interface
// ============================================================================

import { useState, useMemo, useCallback } from 'react'
import { useEnums, useInjectEvent, useEventLabTemplates, usePreviewEventLabRun } from '@/hooks/use-api'
import { useRoleAccess } from '@/hooks/use-rbac'
import { useDashboardStore } from '@/stores/use-dashboard-store'
import { useActivityStore } from '@/stores/use-activity-store'
import { cn } from '@/lib/utils'
import {
  ArrowRightLeft,
  Shield,
  Building2,
  Send,
  CheckCircle2,
  XCircle,
  Loader2,
  Fingerprint,
  MapPin,
  User,
  CreditCard,
  Hash,
  Globe,
  Wifi,
  Lock,
  Unlock,
  Copy,
  Clock,
  Zap,
  Tag,
} from 'lucide-react'
import type { EventLabGeneratedEvent, InjectEventRequest } from '@/lib/types'

type EventTab = 'transaction' | 'auth' | 'interbank'

// -- Reusable field components --

function FieldGroup({ label, required, hint, children }: {
  label: string
  required?: boolean
  hint?: string
  children: React.ReactNode
}) {
  return (
    <div className="space-y-1">
      <label className="text-[9px] font-semibold text-text-primary uppercase tracking-wider flex items-center gap-1">
        {label}
        {required && <span className="text-[#DA251C]">*</span>}
      </label>
      {children}
      {hint && <p className="text-[8px] text-text-muted/60 leading-tight">{hint}</p>}
    </div>
  )
}

function TextInput({
  value,
  onChange,
  placeholder,
  icon: Icon,
  suggestions,
}: {
  value: string
  onChange: (v: string) => void
  placeholder?: string
  icon?: typeof User
  suggestions?: string[]
}) {
  const [showSuggestions, setShowSuggestions] = useState(false)
  const filtered = suggestions?.filter(s =>
    s.toLowerCase().includes(value.toLowerCase()) && s !== value
  ).slice(0, 6) ?? []

  return (
    <div className="relative">
      <div className="relative">
        {Icon && (
          <Icon className="absolute left-2 top-1/2 -translate-y-1/2 w-3 h-3 text-text-muted/50" />
        )}
        <input
          type="text"
          value={value}
          onChange={(e) => { onChange(e.target.value); setShowSuggestions(true) }}
          onFocus={() => setShowSuggestions(true)}
          onBlur={() => setTimeout(() => setShowSuggestions(false), 200)}
          placeholder={placeholder}
          className={cn(
            'w-full bg-bg-deep border border-border-subtle rounded-md py-1.5 text-[10px] font-mono text-text-primary placeholder:text-text-muted/40 focus:border-accent-primary/50 focus:ring-1 focus:ring-accent-primary/20 focus:outline-none transition-all',
            Icon ? 'pl-7 pr-2' : 'px-2.5',
          )}
        />
      </div>
      {showSuggestions && filtered.length > 0 && (
        <div className="absolute z-20 top-full left-0 right-0 mt-0.5 bg-bg-surface border border-border-default rounded-md shadow-lg max-h-32 overflow-y-auto">
          {filtered.map((s) => (
            <button
              key={s}
              type="button"
              onMouseDown={(e) => { e.preventDefault(); onChange(s); setShowSuggestions(false) }}
              className="w-full text-left px-2.5 py-1 text-[9px] font-mono text-text-secondary hover:bg-accent-primary/10 hover:text-accent-primary transition-colors"
            >
              {s}
            </button>
          ))}
        </div>
      )}
    </div>
  )
}

function NumberInput({
  value,
  onChange,
  placeholder,
  step,
  prefix,
  icon: Icon,
}: {
  value: number | ''
  onChange: (v: number | '') => void
  placeholder?: string
  step?: number
  prefix?: string
  icon?: typeof CreditCard
}) {
  return (
    <div className="relative">
      {Icon && (
        <Icon className="absolute left-2 top-1/2 -translate-y-1/2 w-3 h-3 text-text-muted/50" />
      )}
      {prefix && (
        <span className="absolute left-2 top-1/2 -translate-y-1/2 text-[10px] font-mono font-bold text-accent-primary/70">{prefix}</span>
      )}
      <input
        type="number"
        value={value}
        onChange={(e) => onChange(e.target.value === '' ? '' : Number(e.target.value))}
        placeholder={placeholder}
        step={step}
        className={cn(
          'w-full bg-bg-deep border border-border-subtle rounded-md py-1.5 text-[10px] font-mono text-text-primary placeholder:text-text-muted/40 focus:border-accent-primary/50 focus:ring-1 focus:ring-accent-primary/20 focus:outline-none transition-all',
          prefix ? 'pl-6 pr-2' : Icon ? 'pl-7 pr-2' : 'px-2.5',
        )}
      />
    </div>
  )
}

function SelectInput({
  value,
  onChange,
  options,
  placeholder,
  icon: Icon,
}: {
  value: string
  onChange: (v: string) => void
  options: { value: string; label: string }[]
  placeholder?: string
  icon?: typeof Tag
}) {
  return (
    <div className="relative">
      {Icon && (
        <Icon className="absolute left-2 top-1/2 -translate-y-1/2 w-3 h-3 text-text-muted/50 pointer-events-none" />
      )}
      <select
        value={value}
        onChange={(e) => onChange(e.target.value)}
        className={cn(
          'w-full bg-bg-deep border border-border-subtle rounded-md py-1.5 text-[10px] font-mono text-text-primary focus:border-accent-primary/50 focus:ring-1 focus:ring-accent-primary/20 focus:outline-none transition-all appearance-none cursor-pointer',
          Icon ? 'pl-7 pr-6' : 'px-2.5 pr-6',
        )}
      >
        {placeholder && <option value="">{placeholder}</option>}
        {options.map((o) => (
          <option key={o.value} value={o.value}>{o.label}</option>
        ))}
      </select>
      <div className="absolute right-2 top-1/2 -translate-y-1/2 pointer-events-none">
        <svg className="w-3 h-3 text-text-muted" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
        </svg>
      </div>
    </div>
  )
}

// -- Result display --

interface InjectionResult {
  success: boolean
  message: string
  eventId?: string
  eventType?: string
  timestamp?: number
  details?: Record<string, unknown>
}

function ResultDisplay({ result, onDismiss }: { result: InjectionResult; onDismiss: () => void }) {
  return (
    <div className={cn(
      'rounded-md border p-3 animate-fade-in',
      result.success
        ? 'bg-[#00579C]/5 border-[#00579C]/20'
        : 'bg-[#DA251C]/5 border-[#DA251C]/20',
    )}>
      <div className="flex items-start gap-2">
        {result.success ? (
          <CheckCircle2 className="w-4 h-4 text-[#00579C] shrink-0 mt-0.5" />
        ) : (
          <XCircle className="w-4 h-4 text-[#DA251C] shrink-0 mt-0.5" />
        )}
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 mb-1">
            <span className={cn(
              'text-[10px] font-bold uppercase tracking-wider',
              result.success ? 'text-[#00579C]' : 'text-[#DA251C]',
            )}>
              {result.success ? 'Event Injected Successfully' : 'Injection Failed'}
            </span>
            <button onClick={onDismiss} className="ml-auto text-text-muted hover:text-text-primary transition-colors">
              <XCircle className="w-3 h-3" />
            </button>
          </div>
          {result.eventId && (
            <div className="flex items-center gap-1.5 mb-1">
              <Hash className="w-2.5 h-2.5 text-text-muted" />
              <code className="text-[9px] font-mono text-accent-primary">{result.eventId}</code>
              <button
                onClick={() => navigator.clipboard.writeText(result.eventId!)}
                className="text-text-muted hover:text-text-primary transition-colors"
                title="Copy event ID"
              >
                <Copy className="w-2.5 h-2.5" />
              </button>
            </div>
          )}
          {result.details && (
            <div className="grid grid-cols-2 gap-x-4 gap-y-0.5 mt-1.5">
              {Object.entries(result.details).map(([k, v]) => (
                <div key={k} className="flex justify-between text-[8px]">
                  <span className="text-text-muted uppercase tracking-wider">{k.replace(/_/g, ' ')}</span>
                  <span className="font-mono text-text-secondary">{String(v)}</span>
                </div>
              ))}
            </div>
          )}
          {!result.success && (
            <p className="text-[9px] text-[#DA251C]/80 mt-1">{result.message}</p>
          )}
        </div>
      </div>
    </div>
  )
}

function stringValue(value: unknown): string {
  return typeof value === 'string' ? value : ''
}

function numberValue(value: unknown): number | '' {
  const parsed = Number(value)
  return Number.isFinite(parsed) ? parsed : ''
}

function accountTypeFor(accountId: string, fallback = 'SAVINGS') {
  if (!accountId) return fallback
  if (accountId.startsWith('SHELL') || accountId.startsWith('MULE')) return 'CURRENT'
  if (accountId.startsWith('DORM')) return 'SAVINGS'
  return fallback
}

// -- Main component --

export function CustomEventBuilder() {
  const access = useRoleAccess()
  const { data: enums } = useEnums()
  const { data: templatesData, isLoading: templatesLoading } = useEventLabTemplates()
  const preview = usePreviewEventLabRun()
  const inject = useInjectEvent()
  const setTrackedEventId = useActivityStore((s) => s.setTrackedEventId)
  const appendTerminalEntry = useActivityStore((s) => s.appendTerminalEntry)
  const [tab, setTab] = useState<EventTab>('transaction')
  const [result, setResult] = useState<InjectionResult | null>(null)

  // Get node IDs from graph for autocomplete suggestions
  const graphNodes = useDashboardStore((s) => s.graphNodes)
  const nodeIds = useMemo(() => graphNodes.map((n) => n.data.id), [graphNodes])

  // Transaction fields
  const [senderId, setSenderId] = useState('')
  const [receiverId, setReceiverId] = useState('')
  const [amountInr, setAmountInr] = useState<number | ''>('')
  const [channel, setChannel] = useState('')
  const [senderAcctType, setSenderAcctType] = useState('')
  const [receiverAcctType, setReceiverAcctType] = useState('')

  // Auth fields
  const [accountId, setAccountId] = useState('')
  const [authAction, setAuthAction] = useState('')
  const [ipAddress, setIpAddress] = useState('')
  const [authSuccess, setAuthSuccess] = useState(true)

  // Interbank fields
  const [senderIfsc, setSenderIfsc] = useState('')
  const [receiverIfsc, setReceiverIfsc] = useState('')
  const [ibAmount, setIbAmount] = useState<number | ''>('')
  const [msgType, setMsgType] = useState('')
  const [ibChannel, setIbChannel] = useState('')

  // Common
  const [deviceFp, setDeviceFp] = useState('')
  const [geoLat, setGeoLat] = useState<number | ''>('')
  const [geoLon, setGeoLon] = useState<number | ''>('')

  const channelOptions = (enums?.channels ?? []).map((e) => ({ value: e.name, label: e.name }))
  const acctTypeOptions = (enums?.account_types ?? []).map((e) => ({ value: e.name, label: e.name }))
  const authActionOptions = (enums?.auth_actions ?? []).map((e) => ({ value: e.name, label: e.name }))
  const msgTypeOptions = (enums?.message_types ?? []).map((t) => ({ value: t, label: t }))

  const templates = useMemo(() => templatesData?.templates ?? [], [templatesData?.templates])
  const [selectedTemplateId, setSelectedTemplateId] = useState('')
  const [lastPrimedTemplate, setLastPrimedTemplate] = useState<{
    title: string
    description: string
    eventId: string
  } | null>(null)
  const selectedTemplate = useMemo(() => {
    if (!templates.length) return undefined
    return templates.find((template) => template.template_id === selectedTemplateId) ?? templates[0]
  }, [selectedTemplateId, templates])
  const templateOptions = templates.map((template) => ({
    value: template.template_id,
    label: template.title,
  }))

  const applyGeneratedEvent = useCallback((event: EventLabGeneratedEvent) => {
    const record = event as Record<string, unknown>
    const eventType = event.type
    setTab(eventType)

    const device = stringValue(event.device_fingerprint)
    const lat = numberValue(record.geo_lat)
    const lon = numberValue(record.geo_lon)
    setDeviceFp(device)
    setGeoLat(lat)
    setGeoLon(lon)

    if (eventType === 'transaction') {
      const sender = stringValue(event.sender)
      const receiver = stringValue(event.receiver)
      setSenderId(sender)
      setReceiverId(receiver)
      setAmountInr(event.amount_paisa ? Math.round(event.amount_paisa / 100) : '')
      setChannel(stringValue(event.channel))
      setSenderAcctType(accountTypeFor(sender))
      setReceiverAcctType(accountTypeFor(receiver))
      return
    }

    if (eventType === 'auth') {
      setAccountId(stringValue(event.account))
      setAuthAction(stringValue(event.action))
      setIpAddress(stringValue(event.ip))
      setAuthSuccess(Boolean(event.success))
      return
    }

    setSenderIfsc(stringValue(event.sender_ifsc))
    setReceiverIfsc(stringValue(event.receiver_ifsc))
    setIbAmount(event.amount_paisa ? Math.round(event.amount_paisa / 100) : '')
    setMsgType(stringValue(event.message_type))
    setIbChannel(stringValue(event.channel))
  }, [])

  const primeFromTemplate = useCallback(async () => {
    if (!selectedTemplate) return
    setResult(null)
    try {
      const response = await preview.mutateAsync({
        template_id: selectedTemplate.template_id,
        playbook_id: selectedTemplate.linked_playbooks?.[0]?.playbook_id ?? null,
        mode: selectedTemplate.default_mode,
        intensity: 'scale',
        seed: null,
      })
      const event =
        response.run_preview.events.find((item) => item.type === tab) ??
        response.run_preview.events[0]
      if (!event) {
        setResult({ success: false, message: 'Selected backend template did not return a generated event' })
        return
      }
      applyGeneratedEvent(event)
      setLastPrimedTemplate({
        title: selectedTemplate.title,
        description: selectedTemplate.description,
        eventId: event.event_id,
      })
    } catch (err) {
      setResult({ success: false, message: String(err) })
    }
  }, [applyGeneratedEvent, preview, selectedTemplate, tab])

  async function handleInject() {
    if (!access.can('simulation:write')) {
      setResult({
        success: false,
        message: `${access.policy.label} cannot inject events into the live pipeline`,
      })
      return
    }
    setResult(null)
    const body: InjectEventRequest = { event_type: tab }

    if (tab === 'transaction') {
      if (!senderId || !receiverId || !amountInr) {
        setResult({ success: false, message: 'Sender ID, Receiver ID, and Amount are required' })
        return
      }
      body.sender_id = senderId
      body.receiver_id = receiverId
      body.amount_inr = amountInr as number
      if (channel) body.channel = channel
      if (senderAcctType) body.sender_account_type = senderAcctType
      if (receiverAcctType) body.receiver_account_type = receiverAcctType
    } else if (tab === 'auth') {
      if (!accountId) {
        setResult({ success: false, message: 'Account ID is required' })
        return
      }
      body.account_id = accountId
      if (authAction) body.action = authAction
      if (ipAddress) body.ip_address = ipAddress
      body.success = authSuccess
    } else {
      if (!senderIfsc || !receiverIfsc) {
        setResult({ success: false, message: 'Sender IFSC and Receiver IFSC are required' })
        return
      }
      body.sender_ifsc = senderIfsc
      body.receiver_ifsc = receiverIfsc
      if (ibAmount !== '') body.amount_inr = ibAmount as number
      if (msgType) body.message_type = msgType
      if (ibChannel) body.channel = ibChannel
    }

    if (deviceFp) body.device_fingerprint = deviceFp
    if (geoLat !== '') body.geo_lat = geoLat as number
    if (geoLon !== '') body.geo_lon = geoLon as number

    appendTerminalEntry({
      timestamp: Date.now() / 1000,
      source: 'custom',
      tone: 'warn',
      title: `custom ${tab} injection submitted`,
      detail: [
        body.sender_id ? `sender=${body.sender_id}` : '',
        body.receiver_id ? `receiver=${body.receiver_id}` : '',
        body.account_id ? `account=${body.account_id}` : '',
        body.sender_ifsc ? `sender_ifsc=${body.sender_ifsc}` : '',
        body.receiver_ifsc ? `receiver_ifsc=${body.receiver_ifsc}` : '',
        body.amount_inr != null ? `amount_inr=${body.amount_inr}` : '',
        body.channel ? `channel=${body.channel}` : '',
        body.action ? `action=${body.action}` : '',
      ].filter(Boolean).join(' | '),
      stage: 'custom_injection_submitted',
    })

    try {
      const res = await inject.mutateAsync(body)
      const evt = (res.event ?? {}) as Record<string, unknown>
      const evtId = String(evt.txn_id ?? evt.event_id ?? evt.msg_id ?? 'unknown')
      const details: Record<string, unknown> = {
        type: String(evt.type ?? tab),
        pipeline: 'accepted by live ingestion',
      }
      const addDetail = (key: string, value: unknown) => {
        if (value !== undefined && value !== null && value !== '') {
          details[key] = value
        }
      }
      addDetail('sender', evt.sender)
      addDetail('receiver', evt.receiver)
      addDetail('account', evt.account)
      if (evt.amount_paisa !== undefined && evt.amount_paisa !== null) {
        const amountPaisa = Number(evt.amount_paisa)
        if (Number.isFinite(amountPaisa)) {
          details.amount = `₹${(amountPaisa / 100).toLocaleString()}`
        }
      }
      addDetail('channel', evt.channel)
      addDetail('action', evt.action)
      addDetail('success', evt.success)
      addDetail('sender_ifsc', evt.sender_ifsc)
      addDetail('receiver_ifsc', evt.receiver_ifsc)
      appendTerminalEntry({
        timestamp: res.timestamp ?? Date.now() / 1000,
        source: 'custom',
        tone: 'success',
        title: `custom ${tab} accepted by live ingestion`,
        detail: [
          `event=${evtId}`,
          ...Object.entries(details)
            .slice(0, 8)
            .map(([key, value]) => `${key}=${String(value)}`),
        ].join(' | '),
        txnId: evtId,
        stage: 'custom_injection_accepted',
      })
      setTrackedEventId(evtId)
      setResult({
        success: true,
        message: 'Event injected into pipeline',
        eventId: evtId,
        eventType: tab,
        timestamp: res.timestamp,
        details,
      })
    } catch (err) {
      appendTerminalEntry({
        timestamp: Date.now() / 1000,
        source: 'custom',
        tone: 'danger',
        title: `custom ${tab} injection failed`,
        detail: err instanceof Error ? err.message : String(err),
        stage: 'custom_injection_failed',
      })
      setResult({ success: false, message: String(err) })
    }
  }

  const TABS: { key: EventTab; label: string; icon: typeof ArrowRightLeft; desc: string }[] = [
    { key: 'transaction', label: 'Transaction', icon: ArrowRightLeft, desc: 'UPI / NEFT / RTGS payment' },
    { key: 'auth', label: 'Auth Event', icon: Shield, desc: 'Login / logout / password change' },
    { key: 'interbank', label: 'Interbank', icon: Building2, desc: 'SWIFT / NEFT interbank message' },
  ]

  return (
    <div className="bg-bg-elevated/95 border border-border-default rounded-lg overflow-hidden backdrop-blur-sm shadow-[inset_0_1px_0_0_rgba(255,255,255,0.03)]">
      {/* Header */}
      <div className="px-4 pt-4 pb-3 border-b border-border-subtle">
        <div className="flex items-center gap-2">
          <div className="w-7 h-7 rounded-lg bg-accent-primary/10 border border-accent-primary/20 flex items-center justify-center">
            <Send className="w-3.5 h-3.5 text-accent-primary" />
          </div>
          <div>
            <h3 className="text-[11px] font-bold text-text-primary uppercase tracking-[0.12em]">
              Custom Event Builder
            </h3>
            <p className="text-[9px] text-text-muted mt-0.5">
              {access.policy.label} scope: {access.policy.escalationScope}
            </p>
          </div>
        </div>
      </div>

      {/* Tab bar */}
      <div className="flex border-b border-border-subtle">
        {TABS.map(({ key, label, icon: TabIcon, desc }) => (
          <button
            key={key}
            onClick={() => { setTab(key); setResult(null) }}
            className={cn(
              'flex-1 flex flex-col items-center gap-1 py-3 px-3 transition-all relative',
              tab === key
                ? 'bg-accent-primary/5 text-accent-primary'
                : 'text-text-muted hover:text-text-secondary hover:bg-bg-overlay/30',
            )}
          >
            <div className="flex items-center gap-1.5">
              <TabIcon className="w-3.5 h-3.5" />
              <span className="text-[10px] font-bold uppercase tracking-wider">{label}</span>
            </div>
            <span className="text-[8px] opacity-60">{desc}</span>
            {tab === key && (
              <div className="absolute bottom-0 left-2 right-2 h-0.5 bg-accent-primary rounded-full" />
            )}
          </button>
        ))}
      </div>

      <div className="p-4 space-y-4">
        <div className="rounded-md border border-border-subtle bg-bg-overlay/35 p-3">
          <div className="mb-2 flex flex-wrap items-start justify-between gap-2">
            <div>
              <div className="flex items-center gap-1.5 text-[9px] font-bold uppercase tracking-[0.14em] text-accent-primary">
                <Zap className="h-3 w-3" />
                Backend Typology Primer
              </div>
              <p className="mt-1 text-[9px] leading-relaxed text-text-muted">
                Prime the form from the same intel-linked Event Lab generator used for adaptive runs.
              </p>
            </div>
            {selectedTemplate && (
              <div className="flex max-w-[260px] flex-wrap justify-end gap-1">
                {selectedTemplate.typologies.slice(0, 3).map((typology) => (
                  <span
                    key={typology}
                    className="rounded-full border border-accent-primary/20 bg-accent-primary/5 px-1.5 py-0.5 text-[7px] font-bold uppercase tracking-wide text-accent-primary/80"
                  >
                    {typology.replace(/_/g, ' ')}
                  </span>
                ))}
              </div>
            )}
          </div>
          <div className="grid gap-2 md:grid-cols-[minmax(0,1fr)_auto]">
            <SelectInput
              value={selectedTemplate?.template_id ?? ''}
              onChange={(value) => {
                setSelectedTemplateId(value)
                setLastPrimedTemplate(null)
              }}
              options={templateOptions}
              placeholder={templatesLoading ? 'Loading templates from backend' : 'Choose backend template'}
              icon={Tag}
            />
            <button
              type="button"
              onClick={() => void primeFromTemplate()}
              disabled={!selectedTemplate || preview.isPending || inject.isPending}
              className={cn(
                'flex items-center justify-center gap-2 rounded-md border px-4 py-2 text-[10px] font-bold uppercase tracking-wider transition-all duration-200',
                'border-[#DA251C]/40 bg-[#DA251C]/10 text-[#DA251C] hover:bg-[#DA251C]/20 hover:shadow-[0_0_16px_rgba(245,158,11,0.18)]',
                'disabled:cursor-not-allowed disabled:opacity-40 disabled:hover:bg-[#DA251C]/10 disabled:hover:shadow-none',
              )}
            >
              {preview.isPending ? (
                <Loader2 className="h-3.5 w-3.5 animate-spin" />
              ) : (
                <Zap className="h-3.5 w-3.5" />
              )}
              {preview.isPending ? 'Priming...' : 'Prime Fields'}
            </button>
          </div>
          {selectedTemplate && (
            <p className="mt-2 line-clamp-2 text-[8px] leading-relaxed text-text-muted">
              {selectedTemplate.description}
            </p>
          )}
        </div>
        {/* Transaction Fields */}
        {tab === 'transaction' && (
          <div className="space-y-3 animate-fade-in">
            <div className="grid grid-cols-2 gap-3">
              <FieldGroup label="Sender Account ID" required hint="Select from graph or type new">
                <TextInput value={senderId} onChange={setSenderId} placeholder="Sender account ID" icon={User} suggestions={nodeIds} />
              </FieldGroup>
              <FieldGroup label="Receiver Account ID" required hint="Select from graph or type new">
                <TextInput value={receiverId} onChange={setReceiverId} placeholder="Receiver account ID" icon={User} suggestions={nodeIds} />
              </FieldGroup>
            </div>

            <div className="grid grid-cols-2 gap-3">
              <FieldGroup label="Amount (INR)" required hint="Converted to paisa internally">
                <NumberInput value={amountInr} onChange={setAmountInr} placeholder="Transaction amount" step={1000} prefix="₹" />
              </FieldGroup>
              <FieldGroup label="Payment Channel">
                <SelectInput value={channel} onChange={setChannel} options={channelOptions} placeholder="Use backend channel policy" icon={Wifi} />
              </FieldGroup>
            </div>
            <div className="flex items-center gap-2 px-2.5 py-1.5 rounded-md bg-accent-primary/5 border border-accent-primary/15 text-[8px] text-accent-primary/80">
              <Shield className="w-3 h-3 shrink-0" />
              <span>Fraud detection is backend-led — ML, graph analysis, rules, and ledger actions classify the event; the LLM layer only adds bounded forensic explanation when available</span>
            </div>

            <div className="grid grid-cols-2 gap-3">
              <FieldGroup label="Sender Account Type">
                <SelectInput value={senderAcctType} onChange={setSenderAcctType} options={acctTypeOptions} placeholder="Use sender account policy" icon={CreditCard} />
              </FieldGroup>
              <FieldGroup label="Receiver Account Type">
                <SelectInput value={receiverAcctType} onChange={setReceiverAcctType} options={acctTypeOptions} placeholder="Use receiver account policy" icon={CreditCard} />
              </FieldGroup>
            </div>
          </div>
        )}

        {/* Auth Event Fields */}
        {tab === 'auth' && (
          <div className="space-y-3 animate-fade-in">
            <div className="grid grid-cols-2 gap-3">
              <FieldGroup label="Account ID" required hint="Account performing auth action">
                <TextInput value={accountId} onChange={setAccountId} placeholder="Account ID" icon={User} suggestions={nodeIds} />
              </FieldGroup>
              <FieldGroup label="Auth Action">
                <SelectInput value={authAction} onChange={setAuthAction} options={authActionOptions} placeholder="Use auth action policy" icon={Shield} />
              </FieldGroup>
            </div>

            <div className="grid grid-cols-2 gap-3">
              <FieldGroup label="IP Address" hint="Generated by backend if empty">
                <TextInput value={ipAddress} onChange={setIpAddress} placeholder="IP address" icon={Globe} />
              </FieldGroup>
              <FieldGroup label="Auth Result">
                <div className="flex gap-2">
                  <button
                    onClick={() => setAuthSuccess(true)}
                    className={cn(
                      'flex-1 flex items-center justify-center gap-1.5 py-2 rounded-md text-[9px] font-bold uppercase tracking-wider border transition-all',
                      authSuccess
                        ? 'bg-[#00579C]/15 text-[#00579C] border-[#00579C]/30 shadow-[0_0_8px_rgba(0,87,156,0.1)]'
                        : 'text-text-muted border-border-subtle hover:border-border-default',
                    )}
                  >
                    <Unlock className="w-3 h-3" />
                    Success
                  </button>
                  <button
                    onClick={() => setAuthSuccess(false)}
                    className={cn(
                      'flex-1 flex items-center justify-center gap-1.5 py-2 rounded-md text-[9px] font-bold uppercase tracking-wider border transition-all',
                      !authSuccess
                        ? 'bg-[#DA251C]/15 text-[#DA251C] border-[#DA251C]/30 shadow-[0_0_8px_rgba(218,37,28,0.1)]'
                        : 'text-text-muted border-border-subtle hover:border-border-default',
                    )}
                  >
                    <Lock className="w-3 h-3" />
                    Failure
                  </button>
                </div>
              </FieldGroup>
            </div>
          </div>
        )}

        {/* Interbank Fields */}
        {tab === 'interbank' && (
          <div className="space-y-3 animate-fade-in">
            <div className="grid grid-cols-2 gap-3">
              <FieldGroup label="Sender IFSC" required hint="Sender bank IFSC code">
                <TextInput value={senderIfsc} onChange={setSenderIfsc} placeholder="Sender IFSC" icon={Building2} />
              </FieldGroup>
              <FieldGroup label="Receiver IFSC" required hint="Receiver bank IFSC code">
                <TextInput value={receiverIfsc} onChange={setReceiverIfsc} placeholder="Receiver IFSC" icon={Building2} />
              </FieldGroup>
            </div>

            <div className="grid grid-cols-3 gap-3">
              <FieldGroup label="Amount (INR)">
                <NumberInput value={ibAmount} onChange={setIbAmount} placeholder="Transfer amount" step={10000} prefix="₹" />
              </FieldGroup>
              <FieldGroup label="Message Type" hint="SWIFT / NEFT msg type">
                <SelectInput value={msgType} onChange={setMsgType} options={msgTypeOptions} placeholder="Use message policy" />
              </FieldGroup>
              <FieldGroup label="Channel">
                <SelectInput value={ibChannel} onChange={setIbChannel} options={channelOptions} placeholder="Use interbank channel policy" icon={Wifi} />
              </FieldGroup>
            </div>
          </div>
        )}

        {/* Common fields - collapsible */}
        <details className="group">
          <summary className="flex items-center gap-1.5 text-[9px] font-semibold text-text-muted uppercase tracking-wider cursor-pointer hover:text-text-secondary transition-colors select-none list-none">
            <svg className="w-3 h-3 transition-transform group-open:rotate-90" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
            </svg>
            Advanced Fields (Device, Geo-location)
          </summary>
          <div className="grid grid-cols-3 gap-3 mt-3 pt-3 border-t border-border-subtle/50">
            <FieldGroup label="Device Fingerprint" hint="Generated by backend if empty">
              <TextInput value={deviceFp} onChange={setDeviceFp} placeholder="Device fingerprint" icon={Fingerprint} />
            </FieldGroup>
            <FieldGroup label="Geo Latitude" hint="India range: 8 - 35">
              <NumberInput value={geoLat} onChange={setGeoLat} placeholder="Latitude" step={0.001} icon={MapPin} />
            </FieldGroup>
            <FieldGroup label="Geo Longitude" hint="India range: 69 - 97">
              <NumberInput value={geoLon} onChange={setGeoLon} placeholder="Longitude" step={0.001} icon={MapPin} />
            </FieldGroup>
          </div>
        </details>

        {/* Inject controls */}
        <div className="space-y-2.5 pt-2 border-t border-border-subtle">
          {lastPrimedTemplate && (
            <div className="flex items-center gap-2 px-2.5 py-1.5 rounded-md bg-[#DA251C]/5 border border-[#DA251C]/15 animate-fade-in">
              <Zap className="w-3 h-3 text-[#DA251C] shrink-0" />
              <span className="text-[9px] font-bold text-[#DA251C] uppercase tracking-wider">
                Event Lab Primed: {lastPrimedTemplate.title}
              </span>
              <span className="min-w-0 truncate text-[8px] text-[#DA251C]/60">
                {lastPrimedTemplate.description}
              </span>
              <span className="ml-auto shrink-0 font-mono text-[8px] text-[#DA251C]/80">
                {lastPrimedTemplate.eventId}
              </span>
            </div>
          )}
          <div className="flex items-center gap-3">
            <button
              onClick={() => void handleInject()}
              disabled={inject.isPending || !access.can('simulation:write')}
              title={!access.can('simulation:write') ? `${access.policy.label} cannot inject events into the live pipeline` : 'Inject event into pipeline'}
              className={cn(
                'flex items-center gap-2 px-5 py-2.5 rounded-md text-[10px] font-bold uppercase tracking-wider transition-all duration-200',
                'bg-accent-primary text-bg-deep hover:shadow-[0_0_20px_rgba(0,87,156,0.26)] hover:scale-[1.02]',
                'disabled:opacity-40 disabled:cursor-not-allowed disabled:hover:scale-100 disabled:hover:shadow-none',
              )}
            >
              {inject.isPending ? (
                <Loader2 className="w-3.5 h-3.5 animate-spin" />
              ) : (
                <Send className="w-3.5 h-3.5" />
              )}
              {inject.isPending ? 'Injecting...' : 'Inject into Pipeline'}
            </button>
            <div className="flex items-center gap-1 text-[8px] text-text-muted">
              <Clock className="w-2.5 h-2.5" />
              <span>Events are processed in real-time through all pipeline stages</span>
            </div>
          </div>
        </div>

        {/* Result */}
        {result && (
          <ResultDisplay result={result} onDismiss={() => setResult(null)} />
        )}
      </div>
    </div>
  )
}

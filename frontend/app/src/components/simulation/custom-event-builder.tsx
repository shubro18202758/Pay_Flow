// ============================================================================
// Custom Event Builder -- Full-featured dynamic event injection interface
// ============================================================================

import { useState, useMemo, useCallback } from 'react'
import { useEnums, useInjectEvent } from '@/hooks/use-api'
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
  Dices,
  Zap,
} from 'lucide-react'
import type { InjectEventRequest } from '@/lib/types'

// -- Random attack data pools --

const INDIAN_CITIES = [
  { name: 'Mumbai', lat: 19.076, lon: 72.878 },
  { name: 'Delhi', lat: 28.704, lon: 77.103 },
  { name: 'Bangalore', lat: 12.972, lon: 77.595 },
  { name: 'Chennai', lat: 13.083, lon: 80.271 },
  { name: 'Kolkata', lat: 22.573, lon: 88.364 },
  { name: 'Hyderabad', lat: 17.385, lon: 78.487 },
  { name: 'Pune', lat: 18.520, lon: 73.857 },
  { name: 'Ahmedabad', lat: 23.023, lon: 72.571 },
  { name: 'Jaipur', lat: 26.912, lon: 75.787 },
  { name: 'Lucknow', lat: 26.847, lon: 80.946 },
  { name: 'Surat', lat: 21.170, lon: 72.831 },
  { name: 'Nagpur', lat: 21.146, lon: 79.088 },
  { name: 'Patna', lat: 25.609, lon: 85.138 },
  { name: 'Guwahati', lat: 26.145, lon: 91.736 },
  { name: 'Bhopal', lat: 23.260, lon: 77.413 },
  { name: 'Varanasi', lat: 25.318, lon: 83.011 },
  { name: 'Chandigarh', lat: 30.734, lon: 76.779 },
  { name: 'Coimbatore', lat: 11.017, lon: 76.956 },
]

const IFSC_PREFIXES = ['UBIN', 'SBIN', 'HDFC', 'ICIC', 'PUNB', 'BARB', 'CNRB', 'BKID', 'IOBA', 'UCBA']

interface FraudScenario {
  label: string
  eventType: 'transaction' | 'auth' | 'interbank'
  senderPrefix: string
  receiverPrefix: string
  amountRange: [number, number]
  channel: string
  acctType?: string
  desc: string
}

const RANDOM_FRAUD_SCENARIOS: FraudScenario[] = [
  { label: 'Dormant Account Drain', eventType: 'transaction', senderPrefix: 'DORMANT_ACCT', receiverPrefix: 'MULE_RECV', amountRange: [300000, 900000], channel: 'IMPS', desc: 'Suddenly active dormant savings account transferring to mule' },
  { label: 'Micro-Split Ring', eventType: 'transaction', senderPrefix: 'SPLIT_SRC', receiverPrefix: 'SPLIT_DST', amountRange: [7500, 9999], channel: 'UPI', desc: 'Structuring transactions just below ₹10K reporting threshold' },
  { label: 'Corporate Shell Transfer', eventType: 'transaction', senderPrefix: 'RETAIL_VICTIM', receiverPrefix: 'SHELL_CORP', amountRange: [1000000, 5000000], channel: 'NEFT', acctType: 'CURRENT', desc: 'High-value transfer to newly created shell company account' },
  { label: 'UPI Velocity Burst', eventType: 'transaction', senderPrefix: 'RAPID_UPI', receiverPrefix: 'RAPID_RECV', amountRange: [500, 15000], channel: 'UPI', desc: 'Rapid-fire small UPI payments — velocity far exceeds pattern' },
  { label: 'NRI Layering', eventType: 'transaction', senderPrefix: 'NRI_LAYERED', receiverPrefix: 'DOMESTIC_PASS', amountRange: [2000000, 10000000], channel: 'RTGS', acctType: 'NRE', desc: 'Cross-border layered transfer through NRE intermediate account' },
  { label: 'Mule Collector', eventType: 'transaction', senderPrefix: 'BENAMI_SRC', receiverPrefix: 'COLLECTOR', amountRange: [50000, 500000], channel: 'IMPS', desc: 'Benami account funneling to centralized mule collector' },
  { label: 'Late-Night NEFT', eventType: 'transaction', senderPrefix: 'NOCTURNAL', receiverPrefix: 'OFFSHORE_LINK', amountRange: [500000, 3000000], channel: 'NEFT', desc: 'Unusual late-night large transfer to offshore-linked account' },
  { label: 'Phishing Drain', eventType: 'transaction', senderPrefix: 'PHISH_VICTIM', receiverPrefix: 'PHISHER_ACCT', amountRange: [20000, 200000], channel: 'UPI', desc: 'Compromised credentials draining victim account via rapid UPI' },
  { label: 'Credential Stuffing', eventType: 'auth', senderPrefix: 'BRUTE_ACCT', receiverPrefix: '', amountRange: [0, 0], channel: '', desc: 'Multiple failed login attempts from rotating IPs — brute force' },
  { label: 'Geo-Impossible Login', eventType: 'auth', senderPrefix: 'GEO_JUMP', receiverPrefix: '', amountRange: [0, 0], channel: '', desc: 'Login from geographically impossible location within minutes' },
  { label: 'SWIFT Heist Probe', eventType: 'interbank', senderPrefix: '', receiverPrefix: '', amountRange: [5000000, 50000000], channel: 'SWIFT', desc: 'Suspicious SWIFT message mimicking MT103 single customer transfer' },
  { label: 'Interbank Round-Trip', eventType: 'interbank', senderPrefix: '', receiverPrefix: '', amountRange: [1000000, 8000000], channel: 'NEFT', desc: 'Interbank round-trip pattern suggesting circular laundering' },
]

function pick<T>(arr: T[]): T { return arr[Math.floor(Math.random() * arr.length)] }
function randInt(min: number, max: number) { return Math.floor(Math.random() * (max - min + 1)) + min }
function randIfsc() { return `${pick(IFSC_PREFIXES)}0${String(randInt(100000, 999999))}` }
function randDeviceFp() { return Array.from({ length: 16 }, () => '0123456789abcdef'[randInt(0, 15)]).join('') }

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
        {required && <span className="text-red-400">*</span>}
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
        ? 'bg-green-500/5 border-green-500/20'
        : 'bg-red-500/5 border-red-500/20',
    )}>
      <div className="flex items-start gap-2">
        {result.success ? (
          <CheckCircle2 className="w-4 h-4 text-green-400 shrink-0 mt-0.5" />
        ) : (
          <XCircle className="w-4 h-4 text-red-400 shrink-0 mt-0.5" />
        )}
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 mb-1">
            <span className={cn(
              'text-[10px] font-bold uppercase tracking-wider',
              result.success ? 'text-green-400' : 'text-red-400',
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
            <p className="text-[9px] text-red-300/80 mt-1">{result.message}</p>
          )}
        </div>
      </div>
    </div>
  )
}

// -- Main component --

export function CustomEventBuilder() {
  const { data: enums } = useEnums()
  const inject = useInjectEvent()
  const setTrackedEventId = useActivityStore((s) => s.setTrackedEventId)
  const [tab, setTab] = useState<EventTab>('transaction')
  const [result, setResult] = useState<InjectionResult | null>(null)

  // Get node IDs from graph for autocomplete suggestions
  const graphNodes = useDashboardStore((s) => s.graphNodes)
  const nodeIds = useMemo(() => graphNodes.map((n) => n.data.id), [graphNodes])

  // Transaction fields
  const [senderId, setSenderId] = useState('')
  const [receiverId, setReceiverId] = useState('')
  const [amountInr, setAmountInr] = useState<number | ''>(50000)
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
  const [ibAmount, setIbAmount] = useState<number | ''>(100000)
  const [msgType, setMsgType] = useState('')
  const [ibChannel, setIbChannel] = useState('')

  // Common
  const [deviceFp, setDeviceFp] = useState('')
  const [geoLat, setGeoLat] = useState<number | ''>('')
  const [geoLon, setGeoLon] = useState<number | ''>('')

  const [isRandomizing, setIsRandomizing] = useState(false)
  const [lastScenarioLabel, setLastScenarioLabel] = useState<string | null>(null)

  const channelOptions = (enums?.channels ?? []).map((e) => ({ value: e.name, label: e.name }))
  const acctTypeOptions = (enums?.account_types ?? []).map((e) => ({ value: e.name, label: e.name }))
  const authActionOptions = (enums?.auth_actions ?? []).map((e) => ({ value: e.name, label: e.name }))
  const msgTypeOptions = (enums?.message_types ?? []).map((t) => ({ value: t, label: t }))

  const generateRandomAttack = useCallback(async () => {
    setIsRandomizing(true)
    setResult(null)

    const scenario = pick(RANDOM_FRAUD_SCENARIOS)
    setLastScenarioLabel(scenario.label)
    const city = pick(INDIAN_CITIES)
    const jitter = () => +(Math.random() * 0.04 - 0.02).toFixed(4)

    // Switch to the correct tab
    setTab(scenario.eventType)

    // Small delay for visual "filling" effect
    await new Promise((r) => setTimeout(r, 150))

    if (scenario.eventType === 'transaction') {
      const suffix = () => `_${randInt(100, 999)}`
      setSenderId(`${scenario.senderPrefix}${suffix()}`)
      setReceiverId(`${scenario.receiverPrefix}${suffix()}`)
      setAmountInr(randInt(scenario.amountRange[0], scenario.amountRange[1]))
      setChannel(scenario.channel)
      const accts = ['SAVINGS', 'CURRENT', 'NRE', 'NRO']
      setSenderAcctType(scenario.acctType ?? pick(accts))
      setReceiverAcctType(pick(accts))
    } else if (scenario.eventType === 'auth') {
      setAccountId(`${scenario.senderPrefix}_${randInt(100, 999)}`)
      const actions = ['LOGIN', 'PASSWORD_CHANGE', 'MFA_ENROLL', 'LOGOUT']
      if (scenario.label.includes('Credential')) {
        setAuthAction('LOGIN')
        setAuthSuccess(false)
      } else {
        setAuthAction(pick(actions))
        setAuthSuccess(Math.random() > 0.6)
      }
      setIpAddress(`${randInt(103, 223)}.${randInt(0, 255)}.${randInt(0, 255)}.${randInt(1, 254)}`)
    } else {
      setSenderIfsc(randIfsc())
      setReceiverIfsc(randIfsc())
      setIbAmount(randInt(scenario.amountRange[0], scenario.amountRange[1]))
      setMsgType(pick(['MT103', 'MT202', 'N01', 'N02', 'N06']))
      setIbChannel(scenario.channel)
    }

    setDeviceFp(randDeviceFp())
    setGeoLat(+(city.lat + jitter()).toFixed(4))
    setGeoLon(+(city.lon + jitter()).toFixed(4))

    // Brief pause before auto-injection
    await new Promise((r) => setTimeout(r, 300))
    setIsRandomizing(false)
  }, [])

  async function handleInject() {
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
      body.amount_inr = ibAmount as number || 100000
      if (msgType) body.message_type = msgType
      if (ibChannel) body.channel = ibChannel
    }

    if (deviceFp) body.device_fingerprint = deviceFp
    if (geoLat !== '') body.geo_lat = geoLat as number
    if (geoLon !== '') body.geo_lon = geoLon as number

    try {
      const res = await inject.mutateAsync(body)
      const evt = res.event ?? {}
      const evtId = evt.txn_id ?? evt.event_id ?? evt.msg_id ?? 'unknown'
      setTrackedEventId(evtId)
      setResult({
        success: true,
        message: 'Event injected into pipeline',
        eventId: evtId,
        eventType: tab,
        timestamp: res.timestamp,
        details: {
          type: evt.type ?? tab,
          ...(evt.sender && { sender: evt.sender }),
          ...(evt.receiver && { receiver: evt.receiver }),
          ...(evt.account && { account: evt.account }),
          ...(evt.amount_paisa && { amount: `₹${(evt.amount_paisa / 100).toLocaleString()}` }),
          ...(evt.channel && { channel: evt.channel }),
          ...(evt.action && { action: evt.action }),
          ...(evt.success !== undefined && { success: evt.success }),
          ...(evt.sender_ifsc && { sender_ifsc: evt.sender_ifsc }),
          ...(evt.receiver_ifsc && { receiver_ifsc: evt.receiver_ifsc }),
          pipeline: 'Processing...',
        },
      })
    } catch (err) {
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
              Inject real-world banking events into the live pipeline — ML + Graph + AI scores each event independently
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
        {/* Quick-fill scenario templates */}
        {tab === 'transaction' && (
          <div className="space-y-1.5 animate-fade-in">
            <div className="text-[8px] font-semibold text-text-muted uppercase tracking-wider">Quick-fill Scenarios</div>
            <div className="flex flex-wrap gap-1.5">
              {[
                { label: 'Small UPI', sender: 'UPI_ACCT_001', receiver: 'UPI_ACCT_002', amount: 2500, ch: 'UPI' },
                { label: 'High-Value NEFT', sender: 'CORP_SENDER_01', receiver: 'CORP_RECV_01', amount: 1500000, ch: 'NEFT' },
                { label: 'Suspicious Dormant', sender: 'DORMANT_ACCT_99', receiver: 'MULE_ACCT_01', amount: 495000, ch: 'IMPS' },
                { label: 'Micro-Split', sender: 'SPLIT_SRC_01', receiver: 'SPLIT_DST_01', amount: 9900, ch: 'UPI' },
                { label: 'Cross-Border RTGS', sender: 'NRI_ACCT_01', receiver: 'DOMESTIC_RECV_01', amount: 5000000, ch: 'RTGS' },
              ].map((t) => (
                <button
                  key={t.label}
                  type="button"
                  onClick={() => {
                    setSenderId(t.sender)
                    setReceiverId(t.receiver)
                    setAmountInr(t.amount)
                    setChannel(t.ch)
                  }}
                  className="px-2 py-1 rounded text-[8px] font-semibold uppercase tracking-wider border border-border-subtle bg-bg-overlay/40 text-text-secondary hover:border-accent-primary/40 hover:text-accent-primary hover:bg-accent-primary/5 transition-all"
                >
                  {t.label}
                </button>
              ))}
            </div>
          </div>
        )}
        {tab === 'interbank' && (
          <div className="space-y-1.5 animate-fade-in">
            <div className="text-[8px] font-semibold text-text-muted uppercase tracking-wider">Quick-fill Scenarios</div>
            <div className="flex flex-wrap gap-1.5">
              {[
                { label: 'UBI → SBI NEFT', sIfsc: 'UBIN0531456', rIfsc: 'SBIN0001234', amount: 250000, ch: 'NEFT' },
                { label: 'PNB → HDFC RTGS', sIfsc: 'PUNB0123400', rIfsc: 'HDFC0000123', amount: 10000000, ch: 'RTGS' },
                { label: 'Small SWIFT', sIfsc: 'UBIN0500001', rIfsc: 'CITI0000001', amount: 50000, ch: 'SWIFT' },
              ].map((t) => (
                <button
                  key={t.label}
                  type="button"
                  onClick={() => {
                    setSenderIfsc(t.sIfsc)
                    setReceiverIfsc(t.rIfsc)
                    setIbAmount(t.amount)
                    setIbChannel(t.ch)
                  }}
                  className="px-2 py-1 rounded text-[8px] font-semibold uppercase tracking-wider border border-border-subtle bg-bg-overlay/40 text-text-secondary hover:border-accent-primary/40 hover:text-accent-primary hover:bg-accent-primary/5 transition-all"
                >
                  {t.label}
                </button>
              ))}
            </div>
          </div>
        )}
        {/* Transaction Fields */}
        {tab === 'transaction' && (
          <div className="space-y-3 animate-fade-in">
            <div className="grid grid-cols-2 gap-3">
              <FieldGroup label="Sender Account ID" required hint="Select from graph or type new">
                <TextInput value={senderId} onChange={setSenderId} placeholder="e.g. ACCT001" icon={User} suggestions={nodeIds} />
              </FieldGroup>
              <FieldGroup label="Receiver Account ID" required hint="Select from graph or type new">
                <TextInput value={receiverId} onChange={setReceiverId} placeholder="e.g. ACCT002" icon={User} suggestions={nodeIds} />
              </FieldGroup>
            </div>

            <div className="grid grid-cols-2 gap-3">
              <FieldGroup label="Amount (INR)" required hint="Converted to paisa internally">
                <NumberInput value={amountInr} onChange={setAmountInr} placeholder="50000" step={1000} prefix="₹" />
              </FieldGroup>
              <FieldGroup label="Payment Channel">
                <SelectInput value={channel} onChange={setChannel} options={channelOptions} placeholder="Default: UPI" icon={Wifi} />
              </FieldGroup>
            </div>
            <div className="flex items-center gap-2 px-2.5 py-1.5 rounded-md bg-accent-primary/5 border border-accent-primary/15 text-[8px] text-accent-primary/80">
              <Shield className="w-3 h-3 shrink-0" />
              <span>Fraud detection is automatic — the ML pipeline, graph analysis, and Qwen AI will independently classify this event in real-time</span>
            </div>

            <div className="grid grid-cols-2 gap-3">
              <FieldGroup label="Sender Account Type">
                <SelectInput value={senderAcctType} onChange={setSenderAcctType} options={acctTypeOptions} placeholder="Default: SAVINGS" icon={CreditCard} />
              </FieldGroup>
              <FieldGroup label="Receiver Account Type">
                <SelectInput value={receiverAcctType} onChange={setReceiverAcctType} options={acctTypeOptions} placeholder="Default: SAVINGS" icon={CreditCard} />
              </FieldGroup>
            </div>
          </div>
        )}

        {/* Auth Event Fields */}
        {tab === 'auth' && (
          <div className="space-y-3 animate-fade-in">
            <div className="grid grid-cols-2 gap-3">
              <FieldGroup label="Account ID" required hint="Account performing auth action">
                <TextInput value={accountId} onChange={setAccountId} placeholder="e.g. ACCT001" icon={User} suggestions={nodeIds} />
              </FieldGroup>
              <FieldGroup label="Auth Action">
                <SelectInput value={authAction} onChange={setAuthAction} options={authActionOptions} placeholder="Default: LOGIN" icon={Shield} />
              </FieldGroup>
            </div>

            <div className="grid grid-cols-2 gap-3">
              <FieldGroup label="IP Address" hint="Auto-generated if empty">
                <TextInput value={ipAddress} onChange={setIpAddress} placeholder="Auto-generated" icon={Globe} />
              </FieldGroup>
              <FieldGroup label="Auth Result">
                <div className="flex gap-2">
                  <button
                    onClick={() => setAuthSuccess(true)}
                    className={cn(
                      'flex-1 flex items-center justify-center gap-1.5 py-2 rounded-md text-[9px] font-bold uppercase tracking-wider border transition-all',
                      authSuccess
                        ? 'bg-green-500/15 text-green-400 border-green-500/30 shadow-[0_0_8px_rgba(34,197,94,0.1)]'
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
                        ? 'bg-red-500/15 text-red-400 border-red-500/30 shadow-[0_0_8px_rgba(239,68,68,0.1)]'
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
                <TextInput value={senderIfsc} onChange={setSenderIfsc} placeholder="e.g. UBIN0000001" icon={Building2} />
              </FieldGroup>
              <FieldGroup label="Receiver IFSC" required hint="Receiver bank IFSC code">
                <TextInput value={receiverIfsc} onChange={setReceiverIfsc} placeholder="e.g. SBIN0000001" icon={Building2} />
              </FieldGroup>
            </div>

            <div className="grid grid-cols-3 gap-3">
              <FieldGroup label="Amount (INR)">
                <NumberInput value={ibAmount} onChange={setIbAmount} placeholder="100000" step={10000} prefix="₹" />
              </FieldGroup>
              <FieldGroup label="Message Type" hint="SWIFT / NEFT msg type">
                <SelectInput value={msgType} onChange={setMsgType} options={msgTypeOptions} placeholder="Default: N06" />
              </FieldGroup>
              <FieldGroup label="Channel">
                <SelectInput value={ibChannel} onChange={setIbChannel} options={channelOptions} placeholder="Default: NEFT" icon={Wifi} />
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
            <FieldGroup label="Device Fingerprint" hint="Auto-generated if empty">
              <TextInput value={deviceFp} onChange={setDeviceFp} placeholder="Auto-generated" icon={Fingerprint} />
            </FieldGroup>
            <FieldGroup label="Geo Latitude" hint="India range: 8 - 35">
              <NumberInput value={geoLat} onChange={setGeoLat} placeholder="Auto (India)" step={0.001} icon={MapPin} />
            </FieldGroup>
            <FieldGroup label="Geo Longitude" hint="India range: 69 - 97">
              <NumberInput value={geoLon} onChange={setGeoLon} placeholder="Auto (India)" step={0.001} icon={MapPin} />
            </FieldGroup>
          </div>
        </details>

        {/* Inject + Random Attack Buttons */}
        <div className="space-y-2.5 pt-2 border-t border-border-subtle">
          {/* Scenario label banner */}
          {lastScenarioLabel && (
            <div className="flex items-center gap-2 px-2.5 py-1.5 rounded-md bg-amber-500/5 border border-amber-500/15 animate-fade-in">
              <Zap className="w-3 h-3 text-amber-400 shrink-0" />
              <span className="text-[9px] font-bold text-amber-300 uppercase tracking-wider">
                Random Scenario: {lastScenarioLabel}
              </span>
              <span className="text-[8px] text-amber-400/60 ml-auto">
                {RANDOM_FRAUD_SCENARIOS.find(s => s.label === lastScenarioLabel)?.desc}
              </span>
            </div>
          )}
          <div className="flex items-center gap-3">
            <button
              onClick={() => void generateRandomAttack()}
              disabled={isRandomizing || inject.isPending}
              className={cn(
                'flex items-center gap-2 px-4 py-2.5 rounded-md text-[10px] font-bold uppercase tracking-wider transition-all duration-200 border',
                'border-amber-500/40 bg-amber-500/10 text-amber-300 hover:bg-amber-500/20 hover:shadow-[0_0_20px_rgba(245,158,11,0.2)] hover:scale-[1.02]',
                'disabled:opacity-40 disabled:cursor-not-allowed disabled:hover:scale-100 disabled:hover:shadow-none',
              )}
            >
              {isRandomizing ? (
                <Dices className="w-3.5 h-3.5 animate-spin" />
              ) : (
                <Dices className="w-3.5 h-3.5" />
              )}
              {isRandomizing ? 'Generating...' : 'Random Attack'}
            </button>
            <button
              onClick={() => void handleInject()}
              disabled={inject.isPending}
              className={cn(
                'flex items-center gap-2 px-5 py-2.5 rounded-md text-[10px] font-bold uppercase tracking-wider transition-all duration-200',
                'bg-accent-primary text-bg-deep hover:shadow-[0_0_20px_oklch(0.55_0.14_250_/_0.35)] hover:scale-[1.02]',
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

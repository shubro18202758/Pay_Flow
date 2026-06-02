import {
  CheckCircle2,
  ClipboardCheck,
  FileCheck2,
  KeyRound,
  LockKeyhole,
  Play,
  Radio,
  ShieldCheck,
  UserRound,
} from 'lucide-react'
import {
  ROLE_POLICIES,
  ROLE_ORDER,
  canAccessTab,
  type Permission,
} from '@/lib/rbac'
import { useUIStore, TAB_IDS } from '@/stores/use-ui-store'
import { cn } from '@/lib/utils'
import type { TabId } from '@/stores/use-ui-store'

const ACCESS_CHECKS: { label: string; permission: Permission; icon: typeof ShieldCheck }[] = [
  { label: 'Intel write', permission: 'intel:write', icon: Radio },
  { label: 'Case launch', permission: 'case:launch', icon: Play },
  { label: 'Payment hold', permission: 'alert:hold', icon: ShieldCheck },
  { label: 'Card hotlist', permission: 'card:hotlist', icon: ShieldCheck },
  { label: 'Decide case', permission: 'case:decide', icon: CheckCircle2 },
  { label: 'Approve freeze', permission: 'countermeasure:decide', icon: ShieldCheck },
  { label: 'AML draft', permission: 'aml:str:draft', icon: ClipboardCheck },
  { label: 'FIU filing', permission: 'regulatory:file', icon: FileCheck2 },
  { label: 'Rule toggle', permission: 'rules:toggle', icon: KeyRound },
  { label: 'SOC isolate', permission: 'soc:isolate', icon: ShieldCheck },
  { label: 'Model feedback', permission: 'model:feedback', icon: ClipboardCheck },
  { label: 'Audit review', permission: 'audit:review', icon: FileCheck2 },
]

const TAB_LABELS: Record<TabId, string> = {
  overview: 'Overview',
  'threat-sim': 'Event Lab',
  investigations: 'Investigations',
  'pre-fraud-intel': 'Pre-Fraud Intel',
  intelligence: 'Intelligence',
  analytics: 'Analytics',
  compliance: 'Compliance',
  system: 'System',
}

export function RoleAccessBanner() {
  const currentRole = useUIStore((s) => s.currentRole)
  const setCurrentRole = useUIStore((s) => s.setCurrentRole)
  const activeTab = useUIStore((s) => s.activeTab)
  const policy = ROLE_POLICIES[currentRole]
  const deniedTabs = TAB_IDS.filter((tab) => !canAccessTab(currentRole, tab))
  const allowedTabs = TAB_IDS.filter((tab) => canAccessTab(currentRole, tab))
  const deniedTabLabels = deniedTabs.map((tab) => TAB_LABELS[tab])
  const allowedChecks = ACCESS_CHECKS.filter((item) => policy.permissions.includes(item.permission)).length
  const blockedChecks = ACCESS_CHECKS.length - allowedChecks
  const activeTabAllowed = canAccessTab(currentRole, activeTab)

  return (
    <section className="shrink-0 border-b border-[#00579C]/30 bg-[#003f75] shadow-[0_5px_18px_rgba(0,39,73,0.18)]">
      <div className="grid gap-0 xl:grid-cols-[440px_minmax(0,1fr)_320px]">
        <div className="border-b border-white/15 bg-[#00579C] px-4 py-3 text-white xl:border-b-0 xl:border-r xl:border-white/20">
          <div className="flex items-start gap-3">
            <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-md bg-white text-[#00579C]">
              <UserRound className="h-5 w-5" />
            </div>
            <div className="min-w-0">
              <div className="text-[10px] font-extrabold uppercase tracking-[0.16em] text-white/75">
                Prototype-Wide RBAC Active
              </div>
              <div className="mt-0.5 truncate text-base font-black">{policy.label}</div>
              <div className="truncate text-[11px] font-semibold text-white/78">{policy.domain}</div>
            </div>
          </div>
          <div className="mt-2 grid grid-cols-3 gap-2">
            <BannerMetric label="open tabs" value={`${allowedTabs.length}/${TAB_IDS.length}`} />
            <BannerMetric label="write gates" value={`${allowedChecks}/${ACCESS_CHECKS.length}`} />
            <BannerMetric label="locked" value={String(deniedTabs.length)} tone={deniedTabs.length ? 'red' : 'green'} />
          </div>
          <div className="mt-2 rounded-md border border-white/15 bg-white/10 px-2.5 py-2">
            <div className="text-[7px] font-extrabold uppercase tracking-[0.14em] text-white/62">decision authority</div>
            <div className="mt-1 line-clamp-2 text-[10px] font-semibold leading-4 text-white/90" title={policy.decisionAuthority}>
              {policy.decisionAuthority}
            </div>
          </div>
        </div>

        <div className="min-w-0 bg-white px-4 py-2.5">
          <div className="mb-2 flex flex-wrap items-center justify-between gap-2">
            <div className="min-w-0">
              <div className="truncate text-[10px] font-bold uppercase tracking-[0.12em] text-[#24364f]">
                {activeTabAllowed ? 'Current page allowed' : 'Current page redirected by policy'}
              </div>
              <div className="truncate text-[10px] text-[#617189]" title={policy.escalationScope}>
                {policy.escalationScope}
              </div>
            </div>
            <div className="inline-flex items-center gap-1.5 rounded-md border border-[#00579C]/20 bg-[#f4f8fc] px-2.5 py-1.5 text-[9px] font-extrabold uppercase tracking-[0.12em] text-[#00579C]">
              <KeyRound className="h-3.5 w-3.5" />
              X-Payflow-Role: {currentRole}
            </div>
          </div>

          <div className="flex max-h-[74px] flex-wrap items-center gap-1.5 overflow-y-auto pr-1">
            {ROLE_ORDER.map((role) => {
              const item = ROLE_POLICIES[role]
              const active = role === currentRole
              return (
                <button
                  key={role}
                  type="button"
                  onClick={() => setCurrentRole(role)}
                  title={`${item.label}: ${item.escalationScope}`}
                  className={cn(
                    'h-8 rounded-md border px-2.5 text-[9px] font-extrabold uppercase tracking-[0.1em] transition-colors',
                    active
                      ? 'border-[#DA251C] bg-[#DA251C] text-white shadow-sm'
                      : 'border-[#d7e3f1] bg-white text-[#00579C] hover:border-[#00579C] hover:bg-[#dff2ff]',
                  )}
                >
                  {item.label}
                </button>
              )
            })}
          </div>

          <div className="mt-2 flex flex-wrap gap-1.5">
            {ACCESS_CHECKS.map((item) => {
              const allowed = policy.permissions.includes(item.permission)
              const Icon = item.icon
              return (
                <span
                  key={item.permission}
                  className={cn(
                    'inline-flex h-7 items-center gap-1.5 rounded-md border px-2 text-[8px] font-extrabold uppercase tracking-[0.09em]',
                    allowed
                      ? 'border-[#00579C]/25 bg-[#00579C]/5 text-[#00579C]'
                      : 'border-[#DA251C]/20 bg-[#DA251C]/5 text-[#DA251C]',
                  )}
                  title={allowed ? `${policy.label} can ${item.label}` : `${policy.label} cannot ${item.label}`}
                >
                  {allowed ? <Icon className="h-3 w-3" /> : <LockKeyhole className="h-3 w-3" />}
                  {item.label}
                </span>
              )
            })}
          </div>

          <div className="mt-2 grid gap-1.5 lg:grid-cols-3">
            <BannerInfo label="shift" value={policy.shift} />
            <BannerInfo label="reports to" value={policy.reportingLine} />
            <BannerInfo label="tools" value={policy.toolStack.slice(0, 3).join(' / ')} />
          </div>
        </div>

        <div className="border-t border-white/15 bg-[#f4f8fc] px-4 py-3 xl:border-l xl:border-t-0 xl:border-[#d7e3f1]">
          <div className="text-[10px] font-extrabold uppercase tracking-[0.15em] text-[#DA251C]">
            live enforcement map
          </div>
          <div className="mt-2 grid gap-2">
            <EnforcementRow label="navigation" value={`${deniedTabs.length} locked tabs`} />
            <EnforcementRow label="backend" value={`X-Payflow-Role = ${currentRole}`} />
            <EnforcementRow label="actions" value={`${blockedChecks} write controls blocked`} />
          </div>
          <div className="mt-2 rounded-md border border-[#d7e3f1] bg-white px-2.5 py-2">
            <div className="text-[7px] font-extrabold uppercase tracking-[0.14em] text-[#617189]">
              restricted for current role
            </div>
            <div className="mt-1.5 flex max-h-[46px] flex-wrap gap-1 overflow-y-auto">
              {deniedTabLabels.length > 0 ? (
                deniedTabLabels.map((label) => (
                  <span
                    key={label}
                    className="inline-flex items-center gap-1 rounded-sm border border-[#DA251C]/20 bg-[#DA251C]/10 px-1.5 py-0.5 text-[8px] font-extrabold uppercase tracking-[0.08em] text-[#DA251C]"
                  >
                    <LockKeyhole className="h-2.5 w-2.5" />
                    {label}
                  </span>
                ))
              ) : (
                <span className="text-[9px] font-bold text-[#00579C]">All console areas open for this authority.</span>
              )}
            </div>
          </div>
        </div>
      </div>
    </section>
  )
}

function BannerMetric({ label, value, tone = 'blue' }: { label: string; value: string; tone?: 'blue' | 'red' | 'green' }) {
  return (
    <div className="rounded-md border border-white/18 bg-white/10 px-2 py-1.5">
      <div className="text-[7px] font-extrabold uppercase tracking-[0.14em] text-white/62">{label}</div>
      <div
        className={cn(
          'font-mono text-[12px] font-black',
          tone === 'red' ? 'text-[#ffebe9]' : tone === 'green' ? 'text-[#d7ffe7]' : 'text-white',
        )}
      >
        {value}
      </div>
    </div>
  )
}

function BannerInfo({ label, value }: { label: string; value: string }) {
  return (
    <div className="min-w-0 rounded-md border border-[#d7e3f1] bg-[#f7fbff] px-2 py-1.5">
      <div className="text-[7px] font-extrabold uppercase tracking-[0.14em] text-[#617189]">{label}</div>
      <div className="mt-0.5 truncate text-[10px] font-bold text-[#24364f]" title={value}>
        {value}
      </div>
    </div>
  )
}

function EnforcementRow({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex min-w-0 items-center justify-between gap-3 rounded-md border border-[#d7e3f1] bg-white px-2.5 py-1.5">
      <span className="text-[8px] font-extrabold uppercase tracking-[0.13em] text-[#617189]">{label}</span>
      <span className="truncate font-mono text-[9px] font-black text-[#00579C]" title={value}>
        {value}
      </span>
    </div>
  )
}

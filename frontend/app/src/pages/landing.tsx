import { useMemo, useState } from 'react'
import {
  ArrowRight,
  BadgeCheck,
  BellRing,
  Building2,
  CheckCircle2,
  ChevronRight,
  ClipboardCheck,
  FileCheck2,
  KeyRound,
  Landmark,
  LockKeyhole,
  Radio,
  Search,
  ShieldAlert,
  ShieldCheck,
  Wifi,
} from 'lucide-react'
import {
  OPERATIONAL_WORKFLOWS,
  ROLE_ORDER,
  ROLE_POLICIES,
  canAccessTab,
  defaultTabForRole,
  storeRole,
  type PayflowRole,
  type Permission,
} from '@/lib/rbac'
import { cn } from '@/lib/utils'

const SERVICE_NAV = [
  'EFRMS Monitoring',
  'Cyber Cell',
  'Branch Review',
  'FIU Reporting',
  'CFR Registry',
  'Digital Banking',
  'RBI Returns',
  'Case Evidence',
] as const

const ONLINE_SERVICES = [
  { label: 'SOC Queue', body: 'Live EFRMS alerts, device risk, velocity bursts', role: 'soc_analyst' as PayflowRole },
  { label: 'Cyber IR', body: 'Phishing, malware, endpoint containment', role: 'soc_l2_incident_responder' as PayflowRole },
  { label: 'Fund-Flow Case', body: 'Mule chain, layering, round-tripping drill', role: 'fraud_analyst' as PayflowRole },
  { label: 'Payments Desk', body: 'Hold, hotlist, beneficiary, customer exposure', role: 'transaction_officer' as PayflowRole },
  { label: 'EFRMS Rules', body: 'Scenario tuning, threshold drift, false positives', role: 'efrms_specialist' as PayflowRole },
  { label: 'AML Mule Network', body: 'CDD, structuring, STR draft and watchlists', role: 'aml_analyst' as PayflowRole },
  { label: 'FIU Package', body: 'STR/CTR/FMR evidence and audit hashes', role: 'compliance_officer' as PayflowRole },
  { label: 'Committee Gate', body: 'Freeze, countermeasure and policy approvals', role: 'fraud_committee' as PayflowRole },
] as const

const ACTION_MATRIX: { label: string; permission: Permission; tab: string }[] = [
  { label: 'Refresh Intel', permission: 'intel:write', tab: 'pre-fraud-intel' },
  { label: 'Launch Case', permission: 'case:launch', tab: 'investigations' },
  { label: 'Payment Hold', permission: 'alert:hold', tab: 'investigations' },
  { label: 'Card Hotlist', permission: 'card:hotlist', tab: 'investigations' },
  { label: 'Approve Freeze', permission: 'countermeasure:decide', tab: 'threat-sim' },
  { label: 'SOC Isolation', permission: 'soc:isolate', tab: 'system' },
  { label: 'AML Draft', permission: 'aml:str:draft', tab: 'compliance' },
  { label: 'FIU/RBI Filing', permission: 'regulatory:file', tab: 'compliance' },
  { label: 'Toggle Rule', permission: 'rules:toggle', tab: 'compliance' },
  { label: 'Audit Review', permission: 'audit:review', tab: 'compliance' },
] as const

const STAGE_TILES = [
  ['01', 'EFRMS + SOC', 'Alerts, devices, sessions, IOCs'],
  ['02', 'FRM + Payments', 'Cases, holds, hotlists, graph review'],
  ['03', 'AML + FIU', 'CDD, STR, CTR, FMR, CFR, DAKSH'],
  ['04', 'Audit + Risk', 'Evidence hash, approvals, model feedback'],
] as const

function openApp(role: PayflowRole, tab = 'pre-fraud-intel') {
  storeRole(role)
  const landingTab = canAccessTab(role, tab) ? tab : defaultTabForRole(role)
  window.location.href = `/app?tab=${landingTab}`
}

function roleCan(role: PayflowRole, permission: Permission) {
  return ROLE_POLICIES[role].permissions.includes(permission)
}

export function LandingPage() {
  const [selectedRole, setSelectedRole] = useState<PayflowRole>('fraud_analyst')
  const selectedPolicy = ROLE_POLICIES[selectedRole]
  const roleRows = useMemo(() => ROLE_ORDER.map((role) => ROLE_POLICIES[role]), [])
  const allowedActions = ACTION_MATRIX.filter((item) => roleCan(selectedRole, item.permission)).length

  return (
    <main className="h-screen overflow-y-auto bg-[#003f73] text-[#111827]">
      <div className="h-1.5 bg-[#4b2a1f]" />

      <div className="border-b border-[#d9e2ef] bg-[#f8fbff] text-[12px] font-semibold text-[#24364f]">
        <div className="mx-auto flex max-w-7xl flex-wrap items-center justify-between gap-3 px-5 py-2">
          <div className="flex flex-wrap items-center gap-4">
            <a className="font-bold text-[#00579C]" href="#main-content">Skip to main content</a>
            <span className="h-4 w-px bg-[#d9e2ef]" />
            <span>Screen reader</span>
            <span>A-</span>
            <span>A</span>
            <span>A+</span>
          </div>
          <div className="flex flex-wrap items-center gap-3">
            <span>Contact Us</span>
            <a className="rounded-full bg-[#00579C] px-4 py-1.5 font-extrabold text-white" href="/app">
              PayFlow Portal
            </a>
            <a className="font-bold text-[#00579C]" href="/api/v1/rbac/roles">RBAC API</a>
            <span>English</span>
            <span>हिंदी</span>
          </div>
        </div>
      </div>

      <header className="bg-white shadow-[0_2px_14px_rgba(20,33,61,0.14)]">
        <div className="mx-auto flex max-w-7xl flex-wrap items-center justify-between gap-5 px-5 py-4">
          <div className="flex min-w-0 items-center gap-4">
            <span className="ubi-brand-mark h-14 w-14 rounded-xl" />
            <div>
              <div className="text-3xl font-extrabold leading-none text-[#DA251C]">
                Union Bank <span className="text-[#00579C]">of India</span>
              </div>
              <div className="mt-1 text-[12px] font-bold uppercase tracking-[0.2em] text-[#617189]">
                A Government of India Undertaking | PayFlow Fraud Intelligence
              </div>
            </div>
          </div>
          <div className="flex flex-wrap items-center gap-3">
            <button className="inline-flex h-10 items-center gap-2 rounded-full border border-[#d7e3f1] bg-[#f4f8fc] px-4 text-[12px] font-bold text-[#24364f]">
              <Search className="h-4 w-4 text-[#00579C]" />
              Looking for something specific?
            </button>
            <button
              type="button"
              onClick={() => openApp(selectedRole)}
              className="inline-flex h-11 items-center gap-2 rounded-md bg-[#00579C] px-5 text-[12px] font-extrabold uppercase tracking-[0.13em] text-white shadow-[0_8px_18px_rgba(0,87,156,0.26)] transition-colors hover:bg-[#00477f]"
            >
              Internet Banking Style Login
              <ArrowRight className="h-4 w-4" />
            </button>
          </div>
        </div>
      </header>

      <nav className="bg-[#00579C] text-white shadow-[0_8px_20px_rgba(0,87,156,0.18)]">
        <div className="mx-auto flex max-w-7xl items-center gap-1 overflow-x-auto px-5">
          {SERVICE_NAV.map((item) => (
            <button
              key={item}
              type="button"
              className="shrink-0 border-r border-white/15 px-4 py-3 text-[12px] font-extrabold uppercase tracking-[0.08em] hover:bg-white/12"
            >
              {item}
            </button>
          ))}
        </div>
      </nav>

      <section id="main-content" className="relative overflow-hidden bg-[#00579C]">
        <div className="absolute inset-y-0 right-0 hidden w-[38%] bg-[#003f75] xl:block" />
        <div className="absolute inset-y-0 left-[24%] hidden w-32 skew-x-[-14deg] bg-[#DA251C] xl:block" />
        <div className="absolute inset-x-0 top-0 h-px bg-white/35" />
        <div className="absolute inset-x-0 bottom-0 h-2 bg-[#DA251C]" />
        <div className="relative mx-auto grid max-w-7xl gap-0 px-5 py-10 xl:min-h-[calc(100vh-190px)] xl:grid-cols-[310px_minmax(0,1fr)_380px]">
          <aside className="border border-white/20 bg-white shadow-[0_16px_36px_rgba(0,23,52,0.22)]">
            <div className="bg-[#DA251C] px-4 py-3 text-[12px] font-extrabold uppercase tracking-[0.14em] text-white">
              Online Services
            </div>
            {ONLINE_SERVICES.map((item) => (
              <button
                key={item.label}
                type="button"
                onClick={() => setSelectedRole(item.role)}
                className={cn(
                  'flex w-full items-start gap-3 border-b border-[#d7e3f1] px-4 py-4 text-left transition-colors',
                  selectedRole === item.role ? 'bg-white text-[#00579C]' : 'bg-[#f4f8fc] text-[#24364f] hover:bg-white',
                )}
              >
                <ShieldCheck className="mt-0.5 h-5 w-5 shrink-0 text-[#00579C]" />
                <span>
                  <span className="block text-[13px] font-extrabold">{item.label}</span>
                  <span className="mt-1 block text-[11px] leading-5 text-[#617189]">{item.body}</span>
                </span>
              </button>
            ))}
            <button
              type="button"
              onClick={() => openApp(selectedRole)}
              className="m-4 inline-flex h-10 w-[calc(100%-2rem)] items-center justify-center gap-2 rounded-md bg-[#00579C] text-[11px] font-extrabold uppercase tracking-[0.12em] text-white"
            >
              Open Service
              <ChevronRight className="h-4 w-4" />
            </button>
          </aside>

          <div className="min-w-0 bg-[#004b86] px-7 py-8 text-white xl:min-h-[500px]">
            <div className="mb-4 inline-flex items-center gap-2 rounded-sm bg-white px-3 py-1.5 text-[11px] font-extrabold uppercase tracking-[0.16em] text-[#DA251C]">
              <Landmark className="h-4 w-4" />
              Fraud Operations Command Portal
            </div>
            <h1 className="max-w-3xl text-5xl font-black leading-[1.04] tracking-normal text-white">
              PayFlow for Union Bank fraud, cyber, FIU and branch teams.
            </h1>
            <p className="mt-4 max-w-2xl text-[15px] leading-7 text-white/86">
              A Union Bank-style operations entry point for EFRMS, SOC, digital payments, AML, FIU reporting, fraud
              investigation, model governance, risk oversight and audit control review.
            </p>

            <div className="mt-6 grid gap-3 sm:grid-cols-4">
              {STAGE_TILES.map(([num, title, body]) => (
                <div key={title} className="border border-white/18 bg-white/10 p-3 shadow-sm">
                  <div className="font-mono text-[11px] font-extrabold text-white">{num}</div>
                  <div className="mt-1 text-[11px] font-extrabold uppercase tracking-[0.1em] text-white">{title}</div>
                  <div className="mt-1 text-[10px] leading-4 text-white/72">{body}</div>
                </div>
              ))}
            </div>

            <div className="mt-6 grid gap-2 border border-white/18 bg-white/10 p-3 lg:grid-cols-3">
              <HeroControl label="role header" value={`X-Payflow-Role: ${selectedRole}`} />
              <HeroControl label="tabs available" value={`${selectedPolicy.tabs.length}/8`} />
              <HeroControl label="actions enabled" value={`${allowedActions}/${ACTION_MATRIX.length}`} />
            </div>

            <div className="mt-6 flex flex-wrap gap-2">
              <button
                type="button"
                onClick={() => openApp(selectedRole, 'overview')}
                className="inline-flex h-11 items-center gap-2 rounded-md border-2 border-white bg-white px-4 text-[12px] font-extrabold uppercase tracking-[0.12em] text-[#00579C]"
              >
                <Building2 className="h-4 w-4" />
                Open Fund-Flow Overview
              </button>
              <button
                type="button"
                onClick={() => openApp(selectedRole, 'threat-sim')}
                className="inline-flex h-11 items-center gap-2 rounded-md bg-[#DA251C] px-4 text-[12px] font-extrabold uppercase tracking-[0.12em] text-white shadow-[0_10px_22px_rgba(117,16,10,0.3)]"
              >
                <ShieldAlert className="h-4 w-4" />
                Test RBAC In Event Lab
              </button>
            </div>
          </div>

          <aside className="bg-[#003f75] p-5 text-white shadow-[0_16px_36px_rgba(0,23,52,0.22)]">
            <div className="mb-4 flex items-center justify-between gap-3 border-b border-white/20 pb-3">
              <div>
                <div className="text-[11px] font-extrabold uppercase tracking-[0.18em] text-white/75">Selected Role</div>
                <div className="mt-1 text-xl font-extrabold">{selectedPolicy.label}</div>
              </div>
              <KeyRound className="h-8 w-8 text-white/85" />
            </div>
            <div className="space-y-3">
              <InfoMetric label="Domain" value={selectedPolicy.domain} />
              <InfoMetric label="Tabs Open" value={`${selectedPolicy.tabs.length}/8`} />
              <InfoMetric label="Write Actions" value={`${allowedActions}/${ACTION_MATRIX.length}`} />
              <InfoMetric label="Shift / Queue" value={selectedPolicy.shift} />
              <InfoMetric label="Reports To" value={selectedPolicy.reportingLine} />
            </div>
            <p className="mt-4 text-[12px] leading-6 text-white/82">{selectedPolicy.escalationScope}</p>
            <div className="mt-4 border border-white/18 bg-white/10 p-3">
              <div className="text-[9px] font-extrabold uppercase tracking-[0.16em] text-white/65">tool stack</div>
              <div className="mt-2 flex flex-wrap gap-1.5">
                {selectedPolicy.toolStack.slice(0, 5).map((tool) => (
                  <span key={tool} className="rounded-sm bg-white px-2 py-1 text-[9px] font-bold uppercase tracking-[0.08em] text-[#00579C]">
                    {tool}
                  </span>
                ))}
              </div>
            </div>
            <div className="mt-4 grid grid-cols-2 gap-2">
              {ACTION_MATRIX.map((item) => {
                const allowed = roleCan(selectedRole, item.permission)
                return (
                  <button
                    key={item.label}
                    type="button"
                    onClick={() => allowed && openApp(selectedRole, item.tab)}
                    disabled={!allowed}
                    className={cn(
                      'flex min-h-14 items-center gap-2 rounded-md border px-2 text-left text-[10px] font-extrabold uppercase tracking-[0.08em]',
                      allowed
                        ? 'border-white/30 bg-white text-[#00579C]'
                        : 'border-white/15 bg-white/8 text-white/45',
                    )}
                  >
                    {allowed ? <CheckCircle2 className="h-4 w-4 shrink-0 text-[#00579C]" /> : <LockKeyhole className="h-4 w-4 shrink-0" />}
                    {item.label}
                  </button>
                )
              })}
            </div>
          </aside>
        </div>
      </section>

      <section className="border-y border-[#d7e3f1] bg-[#f4f8fc]">
        <div className="mx-auto grid max-w-7xl gap-4 px-5 py-5 lg:grid-cols-[1fr_1fr_1fr_1fr]">
          <PortalBlock icon={Wifi} title="Digital Banking Services" value="UPI / IMPS / Cards / Net Banking" />
          <PortalBlock icon={BellRing} title="EFRMS / SOC Watch" value="Velocity, MFA, device, malware, phishing" />
          <PortalBlock icon={FileCheck2} title="RBI / FIU Workbench" value="STR, CTR, FMR, CFR and audit trail" />
          <PortalBlock icon={Radio} title="Live Backend" value="SSE + role header + permission guards" />
        </div>
      </section>

      <section className="border-b border-[#d7e3f1] bg-white">
        <div className="mx-auto max-w-7xl px-5 py-7">
          <div className="mb-4 flex flex-wrap items-end justify-between gap-3">
            <div>
              <div className="inline-flex items-center gap-2 bg-[#DA251C] px-3 py-1.5 text-[10px] font-extrabold uppercase tracking-[0.16em] text-white">
                <ClipboardCheck className="h-4 w-4" />
                Indian bank fraud operating model
              </div>
              <h2 className="mt-3 text-2xl font-black text-[#111827]">Operational workflow authority matrix</h2>
            </div>
            <div className="rounded-md bg-[#00579C] px-4 py-2 text-[11px] font-extrabold uppercase tracking-[0.13em] text-white">
              {ROLE_ORDER.length} bank roles wired
            </div>
          </div>

          <div className="grid gap-4 lg:grid-cols-2">
            {OPERATIONAL_WORKFLOWS.map((workflow) => (
              <article key={workflow.id} className="border border-[#d7e3f1] bg-[#f7fbff] shadow-sm">
                <div className="border-b border-[#d7e3f1] bg-white px-4 py-3">
                  <div className="text-[15px] font-black text-[#111827]">{workflow.title}</div>
                  <div className="mt-1 text-[11px] leading-5 text-[#4b5d76]">{workflow.trigger}</div>
                </div>
                <div className="divide-y divide-[#d7e3f1]">
                  {workflow.stages.map((stage, index) => {
                    const owner = ROLE_POLICIES[stage.owner]
                    const active = stage.owner === selectedRole
                    return (
                      <button
                        key={`${workflow.id}-${stage.owner}-${index}`}
                        type="button"
                        onClick={() => setSelectedRole(stage.owner)}
                        className={cn(
                          'grid w-full gap-3 px-4 py-3 text-left sm:grid-cols-[124px_minmax(0,1fr)]',
                          active ? 'bg-[#00579C] text-white' : 'bg-[#f7fbff] text-[#24364f] hover:bg-white',
                        )}
                      >
                        <span
                          className={cn(
                            'inline-flex h-8 items-center justify-center rounded-sm px-2 text-[9px] font-extrabold uppercase tracking-[0.08em]',
                            active ? 'bg-white text-[#00579C]' : 'bg-[#00579C] text-white',
                          )}
                        >
                          {owner.label}
                        </span>
                        <span className={cn('text-[12px] font-semibold leading-5', active ? 'text-white' : 'text-[#4b5d76]')}>
                          {stage.action}
                        </span>
                      </button>
                    )
                  })}
                </div>
              </article>
            ))}
          </div>
        </div>
      </section>

      <section className="mx-auto max-w-7xl px-5 py-7">
        <div className="mb-4 flex flex-wrap items-center justify-between gap-3">
          <div>
            <h2 className="text-2xl font-black text-[#111827]">Choose Prototype Role</h2>
            <p className="mt-1 text-[13px] text-[#4b5d76]">
              Role authority spans tab access, write controls, backend permission checks, case actions, evidence packages and reporting gates.
            </p>
          </div>
          <div className="inline-flex items-center gap-2 rounded-md bg-[#00579C] px-4 py-2 text-[11px] font-extrabold uppercase tracking-[0.12em] text-white">
            <Radio className="h-4 w-4" />
            Live local console
          </div>
        </div>

        <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-3">
          {roleRows.map((policy) => {
            const active = policy.role === selectedRole
            const allowed = ACTION_MATRIX.filter((item) => roleCan(policy.role, item.permission)).length
            return (
              <article
                key={policy.role}
                className={cn(
                  'border bg-white shadow-[0_10px_24px_rgba(20,33,61,0.08)] transition-colors',
                  active ? 'border-[#00579C] ring-2 ring-[#00579C]/20' : 'border-[#d7e3f1]',
                )}
              >
                <button type="button" onClick={() => setSelectedRole(policy.role)} className="block w-full p-4 text-left">
                  <div className="flex items-start justify-between gap-3">
                    <div>
                      <div className="text-base font-black text-[#111827]">{policy.label}</div>
                      <div className="mt-1 text-[11px] font-extrabold uppercase tracking-[0.14em] text-[#00579C]">{policy.domain}</div>
                    </div>
                    {active ? <BadgeCheck className="h-5 w-5 text-[#00579C]" /> : <LockKeyhole className="h-4 w-4 text-[#617189]" />}
                  </div>
                  <p className="mt-3 min-h-12 text-[12px] leading-6 text-[#4b5d76]">{policy.summary}</p>
                  <div className="mt-3 border-l-4 border-[#DA251C] bg-[#f7fbff] px-3 py-2">
                    <div className="text-[8px] font-extrabold uppercase tracking-[0.14em] text-[#617189]">authority</div>
                    <div className="mt-1 line-clamp-2 text-[11px] font-semibold leading-5 text-[#24364f]">
                      {policy.decisionAuthority}
                    </div>
                  </div>
                  <div className="mt-3 grid grid-cols-3 gap-2">
                    <RoleStat label="tabs" value={`${policy.tabs.length}/8`} />
                    <RoleStat label="actions" value={`${allowed}/${ACTION_MATRIX.length}`} />
                    <RoleStat label="perms" value={String(policy.permissions.length)} />
                  </div>
                </button>
                <div className="flex border-t border-[#d7e3f1]">
                  <button
                    type="button"
                    onClick={() => setSelectedRole(policy.role)}
                    className="flex-1 px-3 py-3 text-[11px] font-extrabold uppercase tracking-[0.12em] text-[#00579C] hover:bg-[#dff2ff]"
                  >
                    Preview Access
                  </button>
                  <button
                    type="button"
                    onClick={() => openApp(policy.role)}
                    className="flex-1 bg-[#00579C] px-3 py-3 text-[11px] font-extrabold uppercase tracking-[0.12em] text-white hover:bg-[#00477f]"
                  >
                    Launch as Role
                  </button>
                </div>
              </article>
            )
          })}
        </div>
      </section>
    </main>
  )
}

function InfoMetric({ label, value }: { label: string; value: string }) {
  return (
    <div className="border border-white/18 bg-white/10 px-3 py-2">
      <div className="text-[9px] font-extrabold uppercase tracking-[0.15em] text-white/65">{label}</div>
      <div className="mt-1 truncate text-[13px] font-extrabold text-white" title={value}>{value}</div>
    </div>
  )
}

function HeroControl({ label, value }: { label: string; value: string }) {
  return (
    <div className="min-w-0 border-l-4 border-[#DA251C] bg-white px-3 py-2">
      <div className="text-[8px] font-extrabold uppercase tracking-[0.14em] text-[#617189]">{label}</div>
      <div className="mt-1 truncate font-mono text-[11px] font-black text-[#00579C]" title={value}>
        {value}
      </div>
    </div>
  )
}

function PortalBlock({ icon: Icon, title, value }: { icon: typeof Radio; title: string; value: string }) {
  return (
    <div className="flex items-center gap-3 border border-[#d7e3f1] bg-white p-4 shadow-sm">
      <div className="flex h-11 w-11 shrink-0 items-center justify-center bg-[#00579C] text-white">
        <Icon className="h-5 w-5" />
      </div>
      <div className="min-w-0">
        <div className="text-[11px] font-extrabold uppercase tracking-[0.12em] text-[#DA251C]">{title}</div>
        <div className="mt-1 truncate text-[12px] font-bold text-[#24364f]" title={value}>{value}</div>
      </div>
    </div>
  )
}

function RoleStat({ label, value }: { label: string; value: string }) {
  return (
    <div className="bg-[#f4f8fc] px-2 py-2">
      <div className="text-[8px] font-extrabold uppercase tracking-[0.12em] text-[#617189]">{label}</div>
      <div className="mt-1 font-mono text-[12px] font-extrabold text-[#00579C]">{value}</div>
    </div>
  )
}

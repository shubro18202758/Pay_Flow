import { Building2, CheckCircle2, ChevronDown, ChevronUp, KeyRound, LockKeyhole, ShieldCheck, UserRound } from 'lucide-react'
import { canAccessTab, rolePolicy } from '@/lib/rbac'
import { TAB_IDS, useUIStore, type TabId } from '@/stores/use-ui-store'
import { cn } from '@/lib/utils'

export type RoleContextPanel = 'closed' | 'access' | 'reality'

const TAB_LABELS: Record<TabId, string> = {
  overview: 'Fund-Flow Overview',
  'threat-sim': 'Adaptive Event Lab',
  investigations: 'Investigations',
  'pre-fraud-intel': 'Pre-Fraud Intel',
  intelligence: 'Intelligence',
  analytics: 'Analytics',
  compliance: 'Compliance',
  system: 'System',
}

interface Props {
  panel: RoleContextPanel
  onPanelChange: (panel: RoleContextPanel) => void
}

export function RoleContextSwitcher({ panel, onPanelChange }: Props) {
  const currentRole = useUIStore((s) => s.currentRole)
  const activeTab = useUIStore((s) => s.activeTab)
  const policy = rolePolicy(currentRole)
  const deniedTabs = TAB_IDS.filter((tab) => !canAccessTab(currentRole, tab))
  const activeTabAllowed = canAccessTab(currentRole, activeTab)

  const togglePanel = (target: Exclude<RoleContextPanel, 'closed'>) => {
    onPanelChange(panel === target ? 'closed' : target)
  }

  return (
    <section className="shrink-0 border-b border-[#d7e3f1] bg-white px-4 py-2 shadow-[0_1px_0_rgba(0,87,156,0.08)]">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div className="flex min-w-0 items-center gap-2">
          <div className="flex h-9 w-9 shrink-0 items-center justify-center rounded-md bg-[#00579C] text-white">
            <UserRound className="h-4 w-4" />
          </div>
          <div className="min-w-0">
            <div className="flex flex-wrap items-center gap-2">
              <span className="truncate text-[12px] font-black text-[#111827]">{policy.label}</span>
              <span
                className={cn(
                  'inline-flex items-center gap-1 rounded-sm px-2 py-0.5 text-[8px] font-extrabold uppercase tracking-[0.1em]',
                  activeTabAllowed ? 'bg-[#00579C]/10 text-[#00579C]' : 'bg-[#DA251C]/10 text-[#DA251C]',
                )}
              >
                {activeTabAllowed ? <CheckCircle2 className="h-3 w-3" /> : <LockKeyhole className="h-3 w-3" />}
                {TAB_LABELS[activeTab]}
              </span>
              <span className="hidden font-mono text-[9px] font-bold text-[#617189] md:inline">
                X-Payflow-Role: {currentRole}
              </span>
            </div>
            <div className="truncate text-[10px] font-semibold text-[#617189]" title={policy.escalationScope}>
              {policy.escalationScope}
            </div>
          </div>
        </div>

        <div className="flex flex-wrap items-center gap-2">
          <RolePill icon={ShieldCheck} label="open tabs" value={`${TAB_IDS.length - deniedTabs.length}/${TAB_IDS.length}`} />
          <RolePill icon={LockKeyhole} label="locked" value={String(deniedTabs.length)} tone={deniedTabs.length ? 'red' : 'blue'} />
          <button
            type="button"
            onClick={() => togglePanel('access')}
            className={cn(
              'inline-flex h-8 items-center gap-1.5 rounded-md border px-3 text-[10px] font-extrabold uppercase tracking-[0.1em] transition-colors',
              panel === 'access'
                ? 'border-[#00579C] bg-[#00579C] text-white'
                : 'border-[#d7e3f1] bg-[#f7fbff] text-[#00579C] hover:border-[#00579C]',
            )}
          >
            <KeyRound className="h-3.5 w-3.5" />
            Role Matrix
            {panel === 'access' ? <ChevronUp className="h-3.5 w-3.5" /> : <ChevronDown className="h-3.5 w-3.5" />}
          </button>
          <button
            type="button"
            onClick={() => togglePanel('reality')}
            className={cn(
              'inline-flex h-8 items-center gap-1.5 rounded-md border px-3 text-[10px] font-extrabold uppercase tracking-[0.1em] transition-colors',
              panel === 'reality'
                ? 'border-[#DA251C] bg-[#DA251C] text-white'
                : 'border-[#d7e3f1] bg-[#f7fbff] text-[#00579C] hover:border-[#00579C]',
            )}
          >
            <Building2 className="h-3.5 w-3.5" />
            Bank Reality
            {panel === 'reality' ? <ChevronUp className="h-3.5 w-3.5" /> : <ChevronDown className="h-3.5 w-3.5" />}
          </button>
        </div>
      </div>
    </section>
  )
}

function RolePill({
  icon: Icon,
  label,
  value,
  tone = 'blue',
}: {
  icon: typeof ShieldCheck
  label: string
  value: string
  tone?: 'blue' | 'red'
}) {
  return (
    <span
      className={cn(
        'hidden h-8 items-center gap-1.5 rounded-md border px-2.5 text-[9px] font-extrabold uppercase tracking-[0.1em] sm:inline-flex',
        tone === 'red'
          ? 'border-[#DA251C]/20 bg-[#DA251C]/5 text-[#DA251C]'
          : 'border-[#00579C]/20 bg-[#00579C]/5 text-[#00579C]',
      )}
    >
      <Icon className="h-3.5 w-3.5" />
      {label}
      <span className="font-mono">{value}</span>
    </span>
  )
}

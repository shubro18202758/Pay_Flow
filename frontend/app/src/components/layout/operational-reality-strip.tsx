import { useQuery } from '@tanstack/react-query'
import {
  Building2,
  ClipboardCheck,
  Gavel,
  Landmark,
  Layers,
  Network,
  ShieldCheck,
} from 'lucide-react'
import { rolePolicy } from '@/lib/rbac'
import { useUIStore } from '@/stores/use-ui-store'
import { cn } from '@/lib/utils'

interface OperatingUnit {
  id: string
  label: string
  mandate: string
  authority: string
  reporting_line: string
  tools?: string[]
}

interface RegulatoryObligation {
  id: string
  label: string
  detail: string
  owner_unit: string
}

interface ScenarioHandoff {
  workflow_id: string
  workflow_title: string
  trigger: string
  step: number
  total_steps: number
  receives_from: string
  action: string
  hands_to: string
}

interface HierarchyLevel {
  level: string
  entities: string[]
  core_function: string
}

interface OperatingModelPayload {
  source: string
  role: string
  primary_units: OperatingUnit[]
  authority_boundaries: string[]
  workflow_touchpoints: ScenarioHandoff[]
  regulatory_obligations: RegulatoryObligation[]
  organizational_hierarchy: HierarchyLevel[]
  all_operating_units_count: number
  reality_gap_note: string
}

interface RealityCheckPayload {
  source: string
  profile: {
    role: string
    label: string
    domain: string
    reporting_line: string
    decision_authority: string
    shift: string
  }
  scenario_handoffs: ScenarioHandoff[]
  operating_model: OperatingModelPayload
  organizational_hierarchy: HierarchyLevel[]
  floor_level_owner: boolean
  governance_note: string
}

async function fetchRealityCheck(role: string): Promise<RealityCheckPayload> {
  const response = await fetch('/api/v1/rbac/reality-check', {
    headers: { 'X-Payflow-Role': role },
  })
  if (!response.ok) {
    throw new Error(`Reality-check request failed with ${response.status}`)
  }
  return response.json() as Promise<RealityCheckPayload>
}

export function OperationalRealityStrip() {
  const currentRole = useUIStore((s) => s.currentRole)
  const policy = rolePolicy(currentRole)
  const { data, isError } = useQuery({
    queryKey: ['rbac-reality-check', currentRole],
    queryFn: () => fetchRealityCheck(currentRole),
    staleTime: 60_000,
  })

  const model = data?.operating_model
  const primaryUnit = model?.primary_units[0]
  const handoffs = data?.scenario_handoffs ?? []
  const boundaries = model?.authority_boundaries ?? [policy.decisionAuthority]
  const obligations = model?.regulatory_obligations ?? []
  const hierarchy = data?.organizational_hierarchy ?? model?.organizational_hierarchy ?? []
  const visibleHierarchy = hierarchy.slice(0, 3)
  const realityNote = data?.governance_note ?? model?.reality_gap_note ?? 'AI is advisory; bank authority remains role-gated.'

  return (
    <section className="shrink-0 border-b border-[#c6d3e3] bg-[#edf5ff] px-4 py-2">
      <div className="grid gap-2 xl:grid-cols-[340px_minmax(0,1fr)_430px]">
        <div className="min-w-0 rounded-md border border-[#c6d3e3] bg-white shadow-sm">
          <div className="flex items-center gap-3 border-b border-[#d7e3f1] px-3 py-2">
            <div className="flex h-9 w-9 shrink-0 items-center justify-center rounded-md bg-[#DA251C] text-white">
              <Building2 className="h-4 w-4" />
            </div>
            <div className="min-w-0">
              <div className="text-[9px] font-extrabold uppercase tracking-[0.16em] text-[#DA251C]">
                Indian Bank Operating Reality
              </div>
              <div className="truncate text-[11px] font-black text-[#24364f]" title={policy.reportingLine}>
                {policy.label} {'->'} {policy.reportingLine}
              </div>
            </div>
          </div>
          <div className="grid gap-2 p-3">
            <RealityMetric label="unit" value={primaryUnit?.label ?? policy.domain} />
            <RealityMetric label="shift pressure" value={data?.profile.shift ?? policy.shift} />
            <RealityMetric label="bank units mapped" value={String(model?.all_operating_units_count ?? 'backend pending')} />
            <EvidenceNote label="mandate" value={primaryUnit?.mandate ?? policy.summary} />
            <EvidenceNote label="authority" value={primaryUnit?.authority ?? policy.decisionAuthority} />
          </div>
        </div>

        <div className="min-w-0 rounded-md border border-[#c6d3e3] bg-white shadow-sm">
          <div className="flex items-center justify-between gap-3 border-b border-[#d7e3f1] px-3 py-2">
            <div className="inline-flex min-w-0 items-center gap-2">
              <Network className="h-4 w-4 shrink-0 text-[#00579C]" />
              <span className="truncate text-[10px] font-extrabold uppercase tracking-[0.14em] text-[#00579C]">
                incident handoff map from uploaded bank-operations doc
              </span>
            </div>
            <span
              className={cn(
                'shrink-0 rounded-sm px-2 py-1 text-[8px] font-extrabold uppercase tracking-[0.1em]',
                isError ? 'bg-[#DA251C]/10 text-[#DA251C]' : 'bg-[#00579C]/10 text-[#00579C]',
              )}
            >
              {isError ? 'local fallback' : 'backend synced'}
            </span>
          </div>

          <div className="grid gap-2 p-3 lg:grid-cols-2 2xl:grid-cols-3">
            {handoffs.length > 0 ? (
              <>
                {handoffs.slice(0, 3).map((handoff) => (
                  <article key={`${handoff.workflow_id}-${handoff.step}`} className="min-w-0 rounded-md border border-[#d7e3f1] bg-[#f7fbff] p-2">
                    <div className="flex items-center justify-between gap-2">
                      <span className="truncate text-[9px] font-extrabold uppercase tracking-[0.11em] text-[#00579C]" title={handoff.workflow_title}>
                        {handoff.workflow_title}
                      </span>
                      <span className="rounded-sm bg-white px-1.5 py-0.5 font-mono text-[8px] font-black text-[#DA251C]">
                        {handoff.step}/{handoff.total_steps}
                      </span>
                    </div>
                    <div className="mt-2 grid gap-1.5 text-[9px] font-semibold leading-4 text-[#40516b]">
                      <HandoffLine label="receives" value={handoff.receives_from} />
                      <HandoffLine label="does" value={handoff.action} emphasis />
                      <HandoffLine label="hands to" value={handoff.hands_to} />
                    </div>
                  </article>
                ))}
                <ContextCard title="scenario trigger" value={handoffs[0].trigger} />
                <ContextCard title="AI boundary" value={realityNote} />
              </>
            ) : (
              <div className="rounded-md border border-[#d7e3f1] bg-[#f7fbff] p-3 lg:col-span-2 2xl:col-span-3">
                <div className="inline-flex items-center gap-2 text-[10px] font-extrabold uppercase tracking-[0.13em] text-[#00579C]">
                  <Layers className="h-4 w-4" />
                  governance / support role
                </div>
                <p className="mt-2 text-[11px] font-semibold leading-5 text-[#40516b]">
                  {policy.label} is not a floor-level incident owner in the active workflows. Its value is governance,
                  audit, model, risk, or platform control, so operational actions remain with the mapped banking teams.
                </p>
              </div>
            )}
          </div>
        </div>

        <div className="min-w-0 rounded-md border border-[#c6d3e3] bg-white shadow-sm">
          <div className="flex items-center justify-between gap-3 border-b border-[#d7e3f1] px-3 py-2">
            <div className="inline-flex items-center gap-2 text-[10px] font-extrabold uppercase tracking-[0.14em] text-[#DA251C]">
              <Landmark className="h-4 w-4" />
              authority, regulation, hierarchy
            </div>
            <span className="rounded-sm bg-[#00579C]/10 px-2 py-1 text-[8px] font-extrabold uppercase tracking-[0.1em] text-[#00579C]">
              reality check
            </span>
          </div>
          <div className="grid gap-2 p-3">
            <RealityList
              icon={ShieldCheck}
              title="authority boundaries"
              tone="red"
              items={boundaries.slice(0, 2)}
            />
            <RealityList
              icon={ClipboardCheck}
              title="regulatory gates"
              tone="blue"
              items={obligations.length > 0 ? obligations.slice(0, 2).map((item) => `${item.label}: ${item.detail}`) : [realityNote]}
            />
            <div className="grid gap-1.5 rounded-md border border-[#d7e3f1] bg-[#f7fbff] p-2">
              <div className="inline-flex items-center gap-1.5 text-[8px] font-extrabold uppercase tracking-[0.14em] text-[#00579C]">
                <Gavel className="h-3.5 w-3.5 text-[#DA251C]" />
                hierarchy context
              </div>
              <div className="flex flex-wrap gap-1.5">
                {visibleHierarchy.map((item) => (
                  <span
                    key={item.level}
                    className="rounded-sm border border-[#00579C]/20 bg-white px-1.5 py-1 text-[8px] font-extrabold uppercase tracking-[0.08em] text-[#24364f]"
                    title={`${item.entities.join(' / ')}: ${item.core_function}`}
                  >
                    {item.level}
                  </span>
                ))}
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  )
}

function RealityMetric({ label, value }: { label: string; value: string }) {
  return (
    <div className="min-w-0 rounded-md border border-[#d7e3f1] bg-[#f7fbff] px-2.5 py-2">
      <div className="text-[7px] font-extrabold uppercase tracking-[0.14em] text-[#617189]">{label}</div>
      <div className="mt-1 truncate text-[10px] font-black text-[#24364f]" title={value}>
        {value}
      </div>
    </div>
  )
}

function EvidenceNote({ label, value }: { label: string; value: string }) {
  return (
    <div className="min-w-0 rounded-md border border-[#d7e3f1] bg-[#f7fbff] px-2.5 py-2">
      <div className="text-[7px] font-extrabold uppercase tracking-[0.14em] text-[#DA251C]">{label}</div>
      <div className="mt-1 line-clamp-2 text-[9px] font-bold leading-4 text-[#40516b]" title={value}>
        {value}
      </div>
    </div>
  )
}

function ContextCard({ title, value }: { title: string; value: string }) {
  return (
    <article className="min-w-0 rounded-md border border-[#d7e3f1] bg-white p-2">
      <div className="text-[8px] font-extrabold uppercase tracking-[0.14em] text-[#DA251C]">{title}</div>
      <div className="mt-1 line-clamp-4 text-[9px] font-semibold leading-4 text-[#40516b]" title={value}>
        {value}
      </div>
    </article>
  )
}

function HandoffLine({ label, value, emphasis = false }: { label: string; value: string; emphasis?: boolean }) {
  return (
    <div className="grid min-w-0 grid-cols-[64px_minmax(0,1fr)] items-center gap-1">
      <span className="text-[7px] font-extrabold uppercase tracking-[0.12em] text-[#617189]">{label}</span>
      <span className={cn('truncate', emphasis ? 'font-black text-[#24364f]' : 'text-[#40516b]')} title={value}>
        {value}
      </span>
    </div>
  )
}

function RealityList({
  icon: Icon,
  title,
  tone,
  items,
}: {
  icon: typeof ShieldCheck
  title: string
  tone: 'red' | 'blue'
  items: string[]
}) {
  return (
    <div className="rounded-md border border-[#d7e3f1] bg-[#f7fbff] p-2">
      <div
        className={cn(
          'inline-flex items-center gap-1.5 text-[8px] font-extrabold uppercase tracking-[0.14em]',
          tone === 'red' ? 'text-[#DA251C]' : 'text-[#00579C]',
        )}
      >
        <Icon className="h-3.5 w-3.5" />
        {title}
      </div>
      <div className="mt-1.5 grid gap-1">
        {items.map((item) => (
          <div key={item} className="line-clamp-2 text-[9px] font-semibold leading-4 text-[#40516b]" title={item}>
            {item}
          </div>
        ))}
      </div>
    </div>
  )
}

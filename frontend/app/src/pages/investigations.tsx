// ============================================================================
// Investigations Page -- Verdicts + Escalations
// ============================================================================

import { Scale, FileSearch } from 'lucide-react'
import { VerdictList } from '@/components/investigations/verdict-list'
import { EscalationList } from '@/components/investigations/escalation-list'

export function InvestigationsPage() {
  return (
    <div className="flex flex-col h-full">
      {/* ---- Page header ---- */}
      <div className="shrink-0 px-5 pt-5 pb-4 animate-fade-in">
        <div className="flex items-center gap-3 mb-1.5">
          <div className="flex items-center justify-center w-9 h-9 rounded-lg bg-accent-primary/10 border border-accent-primary/20">
            <Scale className="w-5 h-5 text-accent-primary" />
          </div>
          <div>
            <h1 className="text-sm font-semibold text-text-primary tracking-wide flex items-center gap-2">
              Investigations
              <FileSearch className="w-4 h-4 text-text-muted/60" />
            </h1>
            <p className="text-[10px] text-text-muted leading-relaxed mt-0.5">
              AI-generated verdicts and human-in-the-loop escalations awaiting analyst review.
            </p>
          </div>
        </div>
      </div>

      {/* ---- Content ---- */}
      <div className="flex flex-1 min-h-0 gap-3 px-5 pb-5 animate-slide-up">
        <div className="flex-1 min-w-0 bg-bg-deep rounded-lg border border-border-subtle overflow-hidden">
          <VerdictList />
        </div>
        <div className="w-[340px] shrink-0 bg-bg-deep rounded-lg border border-border-subtle overflow-hidden">
          <EscalationList />
        </div>
      </div>
    </div>
  )
}

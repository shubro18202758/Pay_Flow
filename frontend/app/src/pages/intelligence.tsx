// ============================================================================
// Intelligence & Integrity Page -- AI Investigation Stream + Cryptographic Audit Trail
// + Risk Heatmap + Velocity Sparklines + Forensic Evidence Chain
// + NL Query + SHAP Explainability + Drift Monitor + Consortium Intelligence
// ============================================================================

import { BrainCircuit, ShieldCheck } from 'lucide-react'
import { AIInvestigatorStream } from '@/components/panels/ai-investigator-stream'
import { CryptographicAuditTrail } from '@/components/panels/cryptographic-audit-trail'
import { RiskHeatmapTimeline } from '@/components/panels/risk-heatmap-timeline'
import { VelocitySparklines } from '@/components/panels/velocity-sparklines'
import { ForensicEvidenceChain } from '@/components/panels/forensic-evidence-chain'
import { NLQueryPanel } from '@/components/panels/nl-query-panel'
import { ExplainabilityPanel } from '@/components/panels/explainability-panel'
import { DriftMonitorPanel } from '@/components/panels/drift-monitor-panel'
import { ConsortiumPanel } from '@/components/panels/consortium-panel'

export function IntelligencePage() {
  return (
    <div className="flex min-h-full flex-col">
      {/* ---- Page header ---- */}
      <div className="shrink-0 px-5 pt-5 pb-4 animate-fade-in">
        <div className="flex items-center gap-3 mb-1.5">
          <div className="flex items-center justify-center w-9 h-9 rounded-lg bg-accent-primary/10 border border-accent-primary/20">
            <BrainCircuit className="w-5 h-5 text-accent-primary" />
          </div>
          <div>
            <h1 className="text-sm font-semibold text-text-primary tracking-wide flex items-center gap-2">
              Intelligence & Integrity
              <ShieldCheck className="w-4 h-4 text-[#00579C]/80" />
            </h1>
            <p className="text-[10px] text-text-muted leading-relaxed mt-0.5">
              Real-time evidence rationale, explainability, model health, cross-bank consortium intelligence, and cryptographic audit trail.
            </p>
          </div>
        </div>
      </div>

      {/* ---- Row 1: Analytics (Risk Heatmap + Velocity Sparklines) ---- */}
      <div className="flex gap-3 px-5 pb-3 animate-fade-in" style={{ animationDelay: '50ms' }}>
        <div className="flex-1 min-w-0">
          <RiskHeatmapTimeline bucketSeconds={120} lookbackMinutes={30} />
        </div>
        <div className="flex-1 min-w-0">
          <VelocitySparklines windowMinutes={30} topN={6} />
        </div>
      </div>

      {/* ---- Row 2: NL Query + SHAP Explainability + Model Drift + Consortium ---- */}
      <div className="flex gap-3 px-5 pb-3 animate-fade-in" style={{ animationDelay: '100ms' }}>
        <div className="flex-1 min-w-0 h-[320px]">
          <NLQueryPanel />
        </div>
        <div className="flex-1 min-w-0 h-[320px]">
          <ExplainabilityPanel />
        </div>
        <div className="w-[280px] shrink-0 h-[320px]">
          <DriftMonitorPanel />
        </div>
        <div className="w-[300px] shrink-0 h-[320px]">
          <ConsortiumPanel />
        </div>
      </div>

      {/* ---- Row 3: AI Evidence Stream + Evidence Chain + Audit Trail ---- */}
      <div className="flex min-h-[430px] gap-3 px-5 pb-5 animate-slide-up" style={{ animationDelay: '150ms' }}>
        {/* Pane 1: AI Evidence Stream */}
        <div className="relative h-[430px] flex-1 min-w-0">
          <AIInvestigatorStream />
        </div>

        {/* Pane 2: Forensic Evidence Chain */}
        <div className="h-[430px] flex-1 min-w-0">
          <ForensicEvidenceChain />
        </div>

        {/* Pane 3: Cryptographic Audit Trail */}
        <div className="h-[430px] flex-1 min-w-0">
          <CryptographicAuditTrail />
        </div>
      </div>
    </div>
  )
}

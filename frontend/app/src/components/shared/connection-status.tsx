// ============================================================================
// Connection Status -- Animated indicator with icon
// ============================================================================

import { cn } from '@/lib/utils'
import { Wifi, WifiOff } from 'lucide-react'

interface Props {
  connected: boolean
}

export function ConnectionStatus({ connected }: Props) {
  return (
    <div className="flex items-center gap-2">
      <div className="relative">
        <div
          className={cn(
            'w-2 h-2 rounded-full',
            connected ? 'bg-emerald-400' : 'bg-alert-critical',
          )}
        />
        {connected && (
          <div className="absolute inset-0 w-2 h-2 rounded-full bg-emerald-400 animate-ping opacity-75" />
        )}
      </div>
      {connected ? (
        <Wifi className="w-3 h-3 text-emerald-400" />
      ) : (
        <WifiOff className="w-3 h-3 text-alert-critical" />
      )}
      <span
        className={cn(
          'text-[9px] font-bold uppercase tracking-[0.15em]',
          connected ? 'text-emerald-400' : 'text-alert-critical',
        )}
      >
        {connected ? 'Live Stream' : 'Reconnecting'}
      </span>
    </div>
  )
}

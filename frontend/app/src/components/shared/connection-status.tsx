// ============================================================================
// Connection Status -- Animated indicator with icon
// ============================================================================

import { cn } from '@/lib/utils'
import { useEffect, useState } from 'react'
import { Wifi, WifiOff } from 'lucide-react'

interface Props {
  connected: boolean
}

export function ConnectionStatus({ connected }: Props) {
  const [warmup, setWarmup] = useState(true)

  useEffect(() => {
    if (connected) {
      return
    }
    const timer = window.setTimeout(() => setWarmup(false), 14_000)
    return () => window.clearTimeout(timer)
  }, [connected])

  const neutral = !connected && warmup
  const statusColor = connected ? 'text-[#00579C]' : neutral ? 'text-[#DA251C]' : 'text-[#DA251C]'
  const dotColor = connected ? 'bg-[#00579C]' : neutral ? 'bg-[#DA251C]' : 'bg-[#DA251C]'
  const label = connected ? 'Live Stream' : neutral ? 'Connecting' : 'Snapshot Mode'

  return (
    <div className="flex items-center gap-2">
      <div className="relative">
        <div
          className={cn(
            'w-2 h-2 rounded-full',
            dotColor,
          )}
        />
        {connected && (
          <div className="absolute inset-0 w-2 h-2 rounded-full bg-[#00579C] animate-ping opacity-75" />
        )}
      </div>
      {connected ? (
        <Wifi className="w-3 h-3 text-[#00579C]" />
      ) : (
        <WifiOff className={cn('w-3 h-3', statusColor)} />
      )}
      <span
        className={cn(
          'text-[9px] font-bold uppercase tracking-[0.15em]',
          statusColor,
        )}
      >
        {label}
      </span>
    </div>
  )
}

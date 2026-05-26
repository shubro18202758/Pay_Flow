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
      setWarmup(false)
      return
    }
    const timer = window.setTimeout(() => setWarmup(false), 14_000)
    return () => window.clearTimeout(timer)
  }, [connected])

  const neutral = !connected && warmup
  const statusColor = connected ? 'text-emerald-500' : neutral ? 'text-amber-600' : 'text-amber-700'
  const dotColor = connected ? 'bg-emerald-500' : neutral ? 'bg-amber-500' : 'bg-amber-600'
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
          <div className="absolute inset-0 w-2 h-2 rounded-full bg-emerald-500 animate-ping opacity-75" />
        )}
      </div>
      {connected ? (
        <Wifi className="w-3 h-3 text-emerald-500" />
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

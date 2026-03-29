// ============================================================================
// SSE Manager -- EventSource with auto-reconnect + exponential backoff
// ============================================================================

import type { SSEEnvelope } from './types'

type SSECallback = (event: SSEEnvelope) => void
type StatusCallback = (connected: boolean) => void

export class SSEManager {
  private eventSource: EventSource | null = null
  private readonly url: string
  private readonly onEvent: SSECallback
  private readonly onStatusChange: StatusCallback
  private reconnectAttempts = 0
  private readonly maxReconnectDelay = 30_000
  private reconnectTimer: ReturnType<typeof setTimeout> | null = null

  constructor(url: string, onEvent: SSECallback, onStatusChange: StatusCallback) {
    this.url = url
    this.onEvent = onEvent
    this.onStatusChange = onStatusChange
  }

  connect(): void {
    if (this.eventSource) {
      this.eventSource.close()
    }

    this.eventSource = new EventSource(this.url)

    this.eventSource.onopen = () => {
      this.reconnectAttempts = 0
      this.onStatusChange(true)
    }

    this.eventSource.onmessage = (e: MessageEvent) => {
      try {
        const event = JSON.parse(e.data as string) as SSEEnvelope
        this.onEvent(event)
      } catch {
        // Ignore malformed events (keepalive comments, etc.)
      }
    }

    this.eventSource.onerror = () => {
      this.onStatusChange(false)
      this.eventSource?.close()
      this.eventSource = null
      this.scheduleReconnect()
    }
  }

  private scheduleReconnect(): void {
    const delay = Math.min(
      1000 * Math.pow(2, this.reconnectAttempts),
      this.maxReconnectDelay,
    )
    this.reconnectAttempts++
    this.reconnectTimer = setTimeout(() => this.connect(), delay)
  }

  disconnect(): void {
    if (this.reconnectTimer) {
      clearTimeout(this.reconnectTimer)
      this.reconnectTimer = null
    }
    if (this.eventSource) {
      this.eventSource.close()
      this.eventSource = null
    }
  }

  get isConnected(): boolean {
    return this.eventSource?.readyState === EventSource.OPEN
  }
}

import { clsx, type ClassValue } from 'clsx'
import { twMerge } from 'tailwind-merge'

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs))
}

export function fmtNum(n: number): string {
  return n.toLocaleString('en-IN')
}

export function fmtPaisa(paisa: number): string {
  const rupees = paisa / 100
  return `\u20B9${rupees.toLocaleString('en-IN', { minimumFractionDigits: 2 })}`
}

export function fmtDuration(sec: number): string {
  if (sec < 60) return `${sec.toFixed(1)}s`
  if (sec < 3600) return `${Math.floor(sec / 60)}m ${Math.floor(sec % 60)}s`
  return `${Math.floor(sec / 3600)}h ${Math.floor((sec % 3600) / 60)}m`
}

export function fmtTimestamp(unix: number): string {
  return new Date(unix * 1000).toLocaleTimeString('en-IN', {
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
    hour12: false,
  })
}

export function truncId(id: string, len = 12): string {
  return id.length > len ? id.slice(0, len) + '...' : id
}

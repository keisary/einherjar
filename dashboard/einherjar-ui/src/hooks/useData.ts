import { useEffect, useState } from 'react'
import type { Account, BrokerStatus, Einher, EquityPoint, ExposureData, JournalEntry, Metric, Position, Signal } from '@/types'

const API_BASE = 'http://localhost:8000/api'
const POLL_MS = 5_000
const STALE_MS = 30_000

async function fetchJson<T>(path: string): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`)
  if (!response.ok) throw new Error(`HTTP ${response.status}`)
  return response.json() as Promise<T>
}

function usePolling<T>(load: () => Promise<T>, initial: T): T {
  const [data, setData] = useState<T>(initial)
  useEffect(() => {
    let active = true
    const refresh = () => load().then(value => active && setData(value)).catch(() => undefined)
    refresh()
    const timer = window.setInterval(refresh, POLL_MS)
    return () => {
      active = false
      window.clearInterval(timer)
    }
  }, [load])
  return data
}

export function useApiFreshness(): 'live' | 'stale' {
  const [lastSuccess, setLastSuccess] = useState(0)
  useEffect(() => {
    let active = true
    const refresh = () => fetchJson('/health').then(() => active && setLastSuccess(Date.now())).catch(() => undefined)
    refresh()
    const timer = window.setInterval(refresh, POLL_MS)
    return () => {
      active = false
      window.clearInterval(timer)
    }
  }, [])
  const [now, setNow] = useState(Date.now())
  useEffect(() => {
    const timer = window.setInterval(() => setNow(Date.now()), 1_000)
    return () => window.clearInterval(timer)
  }, [])
  return lastSuccess && now - lastSuccess < STALE_MS ? 'live' : 'stale'
}

export function useKillSwitch(): [boolean, () => Promise<void>] {
  const [enabled, setEnabled] = useState(false)
  useEffect(() => {
    fetchJson<{ status: string }>('/health').then(data => setEnabled(data.status === 'paused')).catch(() => undefined)
  }, [])
  const toggle = async () => {
    const next = !enabled
    const response = await fetch(`${API_BASE}/kill_switch?enabled=${next}`, { method: 'POST' })
    if (!response.ok) throw new Error(`HTTP ${response.status}`)
    setEnabled(next)
  }
  return [enabled, toggle]
}

export function useMetrics(): Metric[] {
  return usePolling(() => fetchJson<{ metrics: Metric[] }>('/overview').then(data => data.metrics), [])
}

export function usePositions(): Position[] {
  return usePolling(() => fetchJson<Position[]>('/positions'), [])
}

export function useSignals(): Signal[] {
  return usePolling(() => fetchJson<Signal[]>('/forming'), [])
}

export function useEinhers(): Einher[] {
  return usePolling(() => fetchJson<{ einhers: Einher[] }>('/performance').then(data => data.einhers), [])
}

export function useJournal(): JournalEntry[] {
  return usePolling(() => fetchJson<JournalEntry[]>('/journal'), [])
}

export function useAccount(): Account | null {
  return usePolling(() => fetchJson<Account>('/account'), null)
}

export function useBrokers(): BrokerStatus[] {
  return usePolling(
    () => fetchJson<{ components: { ctrader: { connected: boolean; host: string | null; circuitState: string } } }>('/health')
      .then(data => [{
        name: 'CTRADER',
        lastUpdate: new Date().toISOString(),
        latency: 0,
        status: data.components.ctrader.connected ? 'healthy' : 'critical',
      }]),
    [],
  )
}

export function useEquityData(): EquityPoint[] {
  return usePolling(() => fetchJson<{ equity: EquityPoint[] }>('/overview').then(data => data.equity), [])
}

export function useExposure(): ExposureData[] {
  return usePolling(() => fetchJson<{ exposure: ExposureData[] }>('/overview').then(data => data.exposure), [])
}

export function useLiveClock(): string {
  const [time, setTime] = useState(getUTCTime())
  useEffect(() => {
    const timer = window.setInterval(() => setTime(getUTCTime()), 1_000)
    return () => window.clearInterval(timer)
  }, [])
  return time
}

function getUTCTime(): string {
  return new Date().toISOString().slice(11, 19) + ' UTC'
}

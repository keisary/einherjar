import { useCallback, useEffect, useState } from 'react'
import type { Account, BrokerStatus, EinherPage, EquityPoint, ExposureData, JournalEntry, Metric, Position, Signal } from '@/types'

const API_BASE = 'http://localhost:8000/api'
const POLL_MS = 5_000
const STALE_MS = 30_000

/** Etat initial : aucun einher connu tant que le serveur n'a pas repondu. */
const EMPTY_EINHER_PAGE: EinherPage = {
  total: 0,
  returned: 0,
  einhers: [],
  universes: [],
  classes: {},
  summary: {
    sharpeMedian: null,
    winRateMedian: null,
    avgReturnMedian: null,
    totalReturnMedian: null,
    alphaMedian: null,
    pValueMedian: null,
    tradesTotal: 0,
  },
}

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

export function useEinhers(params: { asset?: string; timeframe?: string; limit?: number } = {}): EinherPage {
  const { asset, timeframe, limit } = params
  const load = useCallback(() => {
    const query = new URLSearchParams()
    if (asset) query.set('asset', asset)
    if (timeframe) query.set('timeframe', timeframe)
    if (limit) query.set('limit', String(limit))
    const suffix = query.toString() ? `?${query.toString()}` : ''
    // Le corpus (recherche) est la source des einhers ; le serveur y fusionne les
    // statistiques live quand elles existent.
    return fetchJson<EinherPage>(`/performance${suffix}`)
  }, [asset, timeframe, limit])
  return usePolling(load, EMPTY_EINHER_PAGE)
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

export interface EnvironmentState {
  environment: 'demo' | 'live' | null
  brokerConnected: boolean
  host: string | null
  circuitState: string | null
  corpusEinhers: number
  corpusUnivers: number
}

/**
 * Environnement reel du serveur, source unique de verite pour le badge DEMO/LIVE.
 * Vient de /api/health (`environment` = compte cTrader utilise, `components.ctrader.connected`).
 * Aucun etat local ne doit affirmer un mode de trading.
 */
export function useEnvironment(): EnvironmentState {
  return usePolling(
    () =>
      fetchJson<{
        environment: string | null
        components: {
          ctrader: { connected: boolean; host: string | null; circuitState: string | null }
          corpusEinhers?: number
          corpusUnivers?: number
        }
      }>('/health').then(data => ({
        environment:
          data.environment === 'live' || data.environment === 'demo'
            ? (data.environment as 'demo' | 'live')
            : null,
        brokerConnected: Boolean(data.components?.ctrader?.connected),
        host: data.components?.ctrader?.host ?? null,
        circuitState: data.components?.ctrader?.circuitState ?? null,
        corpusEinhers: data.components?.corpusEinhers ?? 0,
        corpusUnivers: data.components?.corpusUnivers ?? 0,
      })),
    { environment: null, brokerConnected: false, host: null, circuitState: null, corpusEinhers: 0, corpusUnivers: 0 },
  )
}

function getUTCTime(): string {
  return new Date().toISOString().slice(11, 19) + ' UTC'
}

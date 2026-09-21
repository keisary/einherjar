import { useCallback, useEffect, useState } from 'react'
import type { Account, BrokerStatus, EinherPage, EquityPoint, ExposureData, JournalEntry, Metric, Position, Signal } from '@/types'

/**
 * Base de l'API en RELATIF : le SPA est servi par le meme FastAPI (local comme sur
 * un hebergeur). Une URL absolue `http://localhost:8000/api` rendait le dashboard
 * muet des que l'application etait ouverte depuis une autre adresse (deploiement).
 * `VITE_API_BASE` permet de pointer ailleurs si besoin.
 */
const API_BASE = (import.meta.env.VITE_API_BASE as string | undefined) ?? '/api'
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

/**
 * Derniere mesure d'API, partagee par tous les hooks.
 *
 * `latence` est un aller-retour REELLEMENT mesure cote navigateur (aucune valeur
 * inventee) ; `erreur` porte la derniere cause d'echec pour que l'interface puisse
 * la montrer au lieu d'afficher un etat optimiste.
 */
interface MesureApi {
  latence: number | null
  erreur: string | null
  succes: number | null
}

const mesureApi: MesureApi = { latence: null, erreur: null, succes: null }
const abonnes = new Set<() => void>()

function publier(patch: Partial<MesureApi>): void {
  Object.assign(mesureApi, patch)
  abonnes.forEach((notifier) => notifier())
}

async function fetchJson<T>(path: string): Promise<T> {
  const debut = performance.now()
  try {
    const response = await fetch(`${API_BASE}${path}`)
    if (!response.ok) {
      // 401 = session expiree : on renvoie vers la page de connexion.
      if (response.status === 401 && typeof window !== 'undefined' && !window.location.pathname.startsWith('/login')) {
        window.location.href = '/login'
      }
      throw new Error(`HTTP ${response.status}`)
    }
    const data = (await response.json()) as T
    publier({ latence: Math.round(performance.now() - debut), erreur: null, succes: Date.now() })
    return data
  } catch (erreur) {
    publier({ erreur: erreur instanceof Error ? erreur.message : String(erreur) })
    throw erreur
  }
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

function useMesureApi(): MesureApi {
  const [etat, setEtat] = useState<MesureApi>({ ...mesureApi })
  useEffect(() => {
    const notifier = () => setEtat({ ...mesureApi })
    abonnes.add(notifier)
    const timer = window.setInterval(notifier, 1_000)
    return () => {
      abonnes.delete(notifier)
      window.clearInterval(timer)
    }
  }, [])
  return etat
}

export interface EtatApi {
  etat: 'live' | 'stale'
  /** Aller-retour mesure du dernier appel reussi, en ms (null si jamais mesure). */
  latence: number | null
  /** Derniere cause d'echec (HTTP 401/503, reseau) ou null. */
  erreur: string | null
  dernierSucces: number | null
}

export function useApiFreshness(): EtatApi {
  const mesure = useMesureApi()
  const vivant = mesure.succes !== null && Date.now() - mesure.succes < STALE_MS
  return {
    etat: vivant ? 'live' : 'stale',
    latence: mesure.latence,
    erreur: mesure.erreur,
    dernierSucces: mesure.succes,
  }
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

export interface EtatBoucle {
  running: boolean
  cycles: number | null
  lastCycleAt: string | null
  assets: number | null
  signals: number | null
  orders: number | null
  closed: number | null
  errors: number | null
}

export interface EtatCouverture {
  couplesVerifies: number | null
  couplesEcartes: number | null
  detail: Record<string, string[]>
}

interface ReponseHealth {
  status?: string
  timestamp?: string
  environment?: string | null
  components?: {
    ctrader?: { connected: boolean; host: string | null; circuitState: string | null; lastError?: string | null }
    corpusEinhers?: number
    corpusUnivers?: number
    loop?: Partial<EtatBoucle> | null
    couverture?: { couples_verifies?: number; couples_ecartes?: number; detail?: Record<string, string[]> } | null
  }
}

function versBoucle(brut: Partial<EtatBoucle> | null | undefined): EtatBoucle | null {
  if (!brut) return null
  return {
    running: Boolean(brut.running),
    cycles: brut.cycles ?? null,
    lastCycleAt: brut.lastCycleAt ?? null,
    assets: brut.assets ?? null,
    signals: brut.signals ?? null,
    orders: brut.orders ?? null,
    closed: brut.closed ?? null,
    errors: brut.errors ?? null,
  }
}

function versCouverture(
  brut: { couples_verifies?: number; couples_ecartes?: number; detail?: Record<string, string[]> } | null | undefined,
): EtatCouverture | null {
  if (!brut) return null
  return {
    couplesVerifies: brut.couples_verifies ?? null,
    couplesEcartes: brut.couples_ecartes ?? null,
    detail: brut.detail ?? {},
  }
}

export function useBrokers(): BrokerStatus[] {
  return usePolling(
    () =>
      fetchJson<ReponseHealth>('/health').then(data => {
        const ctrader = data.components?.ctrader
        const mesure = { ...mesureApi }
        return [{
          name: 'CTRADER',
          // Horodatage du SERVEUR (et non l'heure du navigateur) et latence mesuree.
          lastUpdate: data.timestamp ?? '',
          latency: mesure.latence ?? 0,
          status: ctrader?.connected ? 'healthy' : 'critical',
          detail: ctrader?.lastError ?? undefined,
        }]
      }),
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
  lastError: string | null
  corpusEinhers: number
  corpusUnivers: number
  boucle: EtatBoucle | null
  couverture: EtatCouverture | null
}

/**
 * Environnement reel du serveur, source unique de verite pour le badge DEMO/LIVE.
 * Vient de /api/health (`environment` = compte cTrader utilise, `components.ctrader.connected`).
 * Aucun etat local ne doit affirmer un mode de trading : si le serveur ne dit rien,
 * l'interface affiche un etat inconnu plutot qu'une valeur inventee.
 */
export function useEnvironment(): EnvironmentState {
  return usePolling(
    () =>
      fetchJson<ReponseHealth>('/health').then(data => ({
        environment:
          data.environment === 'live' || data.environment === 'demo'
            ? (data.environment as 'demo' | 'live')
            : null,
        brokerConnected: Boolean(data.components?.ctrader?.connected),
        host: data.components?.ctrader?.host ?? null,
        circuitState: data.components?.ctrader?.circuitState ?? null,
        lastError: data.components?.ctrader?.lastError ?? null,
        corpusEinhers: data.components?.corpusEinhers ?? 0,
        corpusUnivers: data.components?.corpusUnivers ?? 0,
        boucle: versBoucle(data.components?.loop),
        couverture: versCouverture(data.components?.couverture),
      })),
    {
      environment: null,
      brokerConnected: false,
      host: null,
      circuitState: null,
      lastError: null,
      corpusEinhers: 0,
      corpusUnivers: 0,
      boucle: null,
      couverture: null,
    },
  )
}

function getUTCTime(): string {
  return new Date().toISOString().slice(11, 19) + ' UTC'
}

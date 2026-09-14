import { useState } from 'react'
import { motion } from 'framer-motion'
import { Layers, SlidersHorizontal } from 'lucide-react'
import { DataTable } from '@/components/DataTable'
import { RuneDivider } from '@/components/RuneDivider'
import { FrostGlow } from '@/components/FrostGlow'
import { cn, formatNumber, formatPercent } from '@/lib/utils'
import { useEinhers } from '@/hooks/useData'
import type { Einher } from '@/types'

/** Affiche "—" pour toute valeur absente (jamais 0, jamais une valeur inventee). */
function chiffre(valeur: number | null, format: 'nombre' | 'pourcent' = 'nombre', decimales = 2) {
  if (valeur === null || valeur === undefined || Number.isNaN(valeur)) {
    return <span className="text-textMuted">—</span>
  }
  return <>{format === 'pourcent' ? formatPercent(valeur) : formatNumber(valeur, decimales)}</>
}

export function PerformancePage() {
  const [selection, setSelection] = useState('')
  const [asset, timeframe] = selection ? selection.split('|') : ['', '']
  const page = useEinhers({ asset, timeframe, limit: 250 })

  const columns = [
    {
      key: 'name' as const,
      header: 'Einher',
      width: '26%',
      render: (row: Einher) => (
        <div>
          <div className="font-mono text-textPrimary truncate">{row.name}</div>
          <div className="text-[10px] text-textMuted truncate" title={row.description}>
            {row.description || '—'}
          </div>
        </div>
      ),
      sortable: true,
    },
    {
      key: 'asset' as const,
      header: 'Univers',
      render: (row: Einher) => (
        <span className="font-mono text-[11px] text-textSecondary">
          {row.asset ?? '—'}
          <span className="text-textMuted"> / {row.timeframe ?? '—'}</span>
        </span>
      ),
      sortable: true,
    },
    {
      key: 'direction' as const,
      header: 'Sens',
      render: (row: Einher) => (
        <span
          className={cn(
            'px-2 py-0.5 text-[9px] font-mono uppercase tracking-wider',
            row.direction === 'BUY' && 'text-positive bg-positive/10',
            row.direction === 'SELL' && 'text-negative bg-negative/10'
          )}
        >
          {row.direction === 'BUY' ? 'LONG' : row.direction === 'SELL' ? 'SHORT' : '—'}
        </span>
      ),
    },
    {
      key: 'sharpe' as const,
      header: 'Sharpe',
      render: (row: Einher) => (
        <span
          className={cn(
            'font-mono text-[11px]',
            row.sharpe === null
              ? 'text-textMuted'
              : row.sharpe >= 2
                ? 'text-positive'
                : row.sharpe >= 1.5
                  ? 'text-textSecondary'
                  : 'text-negative'
          )}
        >
          {chiffre(row.sharpe)}
        </span>
      ),
      sortable: true,
    },
    {
      key: 'winRate' as const,
      header: 'Win Rate',
      render: (row: Einher) => (
        <span
          className={cn(
            'font-mono text-[11px]',
            row.winRate === null
              ? 'text-textMuted'
              : row.winRate >= 0.6
                ? 'text-positive'
                : row.winRate >= 0.5
                  ? 'text-textSecondary'
                  : 'text-negative'
          )}
        >
          {chiffre(row.winRate === null ? null : row.winRate * 100, 'pourcent')}
        </span>
      ),
      sortable: true,
    },
    {
      key: 'totalTrades' as const,
      header: 'Trades',
      render: (row: Einher) => (
        <span className="font-mono text-[11px] text-textSecondary">{chiffre(row.totalTrades, 'nombre', 0)}</span>
      ),
      sortable: true,
    },
    {
      key: 'totalReturn' as const,
      header: 'Retour total',
      render: (row: Einher) => (
        <span
          className={cn(
            'font-mono text-[11px]',
            row.totalReturn === null ? 'text-textMuted' : row.totalReturn >= 0 ? 'text-positive' : 'text-negative'
          )}
        >
          {row.totalReturn === null ? '—' : formatPercent(row.totalReturn * 100)}
        </span>
      ),
      sortable: true,
    },
    {
      key: 'alpha' as const,
      header: 'Alpha / p',
      render: (row: Einher) => (
        <span className="font-mono text-[11px] text-textSecondary">
          {row.alpha === null ? '—' : formatPercent(row.alpha * 100)}
          <span className="text-textMuted">
            {' / '}
            {row.pValue === null ? '—' : row.pValue.toFixed(3)}
          </span>
        </span>
      ),
      sortable: true,
    },
  ]

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      transition={{ duration: 0.4 }}
      className="p-6 space-y-6"
    >
      <div className="flex flex-wrap items-end justify-between gap-4">
        <div>
          <div className="text-[10px] font-cinzel tracking-[0.2em] text-textMuted uppercase">
            Einher Performance
          </div>
          <div className="font-mono text-2xl text-textPrimary mt-1">
            {page.total} <span className="text-textMuted text-sm">einhers du corpus</span>
            <span className="text-textMuted text-sm">
              {' '}
              · {page.universes.length} couples surveilles
            </span>
          </div>
          <div className="text-[10px] text-textMuted mt-1">
            {page.returned < page.total
              ? `${page.returned} affiches (tries par Sharpe) — filtres serveur sur ${page.total}`
              : 'Toutes les lignes du filtre courant'}
          </div>
        </div>
        <div className="flex items-center gap-2">
          <SlidersHorizontal size={14} className="text-textMuted" />
          <select
            value={selection}
            onChange={(e) => setSelection(e.target.value)}
            className="bg-background border border-border rounded px-3 py-2 text-xs text-textPrimary focus:outline-none focus:border-frost"
          >
            <option value="">Tous les couples ({page.universes.length})</option>
            {page.universes.map((u) => (
              <option key={`${u.asset}|${u.timeframe}`} value={`${u.asset}|${u.timeframe}`}>
                {u.asset} / {u.timeframe} — {u.einhers} einhers
              </option>
            ))}
          </select>
        </div>
      </div>

      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        <FrostGlow className="bg-surface border border-border p-4">
          <div className="text-[10px] font-cinzel tracking-[0.2em] text-textMuted uppercase">Sharpe median</div>
          <div className="font-mono text-xl text-textPrimary mt-1">{chiffre(page.summary.sharpeMedian)}</div>
        </FrostGlow>
        <FrostGlow className="bg-surface border border-border p-4">
          <div className="text-[10px] font-cinzel tracking-[0.2em] text-textMuted uppercase">Win rate median</div>
          <div className="font-mono text-xl text-textPrimary mt-1">
            {chiffre(page.summary.winRateMedian === null ? null : page.summary.winRateMedian * 100, 'pourcent')}
          </div>
        </FrostGlow>
        <FrostGlow className="bg-surface border border-border p-4">
          <div className="text-[10px] font-cinzel tracking-[0.2em] text-textMuted uppercase">Rendement moyen median</div>
          <div className="font-mono text-xl text-textPrimary mt-1">
            {chiffre(
              page.summary.avgReturnMedian === null ? null : page.summary.avgReturnMedian * 100,
              'pourcent'
            )}
          </div>
        </FrostGlow>
        <FrostGlow className="bg-surface border border-border p-4">
          <div className="text-[10px] font-cinzel tracking-[0.2em] text-textMuted uppercase">Trades (recherche)</div>
          <div className="font-mono text-xl text-textPrimary mt-1">
            {chiffre(page.summary.tradesTotal, 'nombre', 0)}
          </div>
        </FrostGlow>
      </div>

      <RuneDivider runes="ᚠᚢᚦᚨᚱ" />

      <div className="bg-surface border border-border">
        <DataTable columns={columns} data={page.einhers} rowKey={(row) => row.id} />
      </div>

      <div className="flex items-center gap-2 text-[10px] text-textMuted">
        <Layers size={12} />
        <span>
          Statistiques de recherche du corpus (validation hors ligne). Le statut runtime
          (ACTIVE/PROBATION) n'apparait qu'une fois l'einher evalue en live.
        </span>
      </div>
    </motion.div>
  )
}

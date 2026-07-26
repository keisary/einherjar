import { motion } from 'framer-motion'
import { DataTable } from '@/components/DataTable'
import { RuneDivider } from '@/components/RuneDivider'
import { cn, formatPercent, formatNumber } from '@/lib/utils'
import { useEinhers } from '@/hooks/useData'
import type { Einher } from '@/types'

export function PerformancePage() {
  const einhers = useEinhers()

  const columns = [
    {
      key: 'name' as const,
      header: 'Einher',
      render: (row: Einher) => (
        <div>
          <div className="font-mono text-textPrimary">{row.name}</div>
          <div className="text-[10px] text-textMuted">{row.description}</div>
        </div>
      ),
      sortable: true,
    },
    {
      key: 'status' as const,
      header: 'Status',
      render: (row: Einher) => (
        <span
          className={cn(
            'px-2 py-0.5 text-[9px] font-mono uppercase tracking-wider',
            row.status === 'ACTIVE' && 'text-positive bg-positive/10',
            row.status === 'PROBATION' && 'text-warning bg-warning/10',
            row.status === 'DISABLED' && 'text-textMuted bg-textMuted/10'
          )}
        >
          {row.status}
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
            row.winRate >= 0.6 ? 'text-positive' : row.winRate >= 0.5 ? 'text-textSecondary' : 'text-negative'
          )}
        >
          {formatPercent(row.winRate * 100)}
        </span>
      ),
      sortable: true,
    },
    {
      key: 'totalTrades' as const,
      header: 'Trades',
      render: (row: Einher) => (
        <span className="font-mono text-[11px] text-textSecondary">{row.totalTrades}</span>
      ),
      sortable: true,
    },
    {
      key: 'avgReturn' as const,
      header: 'Avg Return',
      render: (row: Einher) => (
        <span className="font-mono text-[11px] text-textSecondary">
          {formatPercent(row.avgReturn)}
        </span>
      ),
      sortable: true,
    },
    {
      key: 'sharpe' as const,
      header: 'Sharpe',
      render: (row: Einher) => (
        <span
          className={cn(
            'font-mono text-[11px]',
            row.sharpe >= 1.5 ? 'text-positive' : row.sharpe >= 1.0 ? 'text-textSecondary' : 'text-negative'
          )}
        >
          {formatNumber(row.sharpe)}
        </span>
      ),
      sortable: true,
    },
    {
      key: 'lastSignal' as const,
      header: 'Last Signal',
      render: (row: Einher) => (
        <span className="text-[10px] text-textMuted">{row.lastSignal}</span>
      ),
    },
  ]

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      transition={{ duration: 0.4 }}
      className="p-6"
    >
      <div className="mb-6">
        <div className="text-[10px] font-cinzel tracking-[0.2em] text-textMuted uppercase">
          Einher Performance
        </div>
        <div className="font-mono text-2xl text-textPrimary mt-1">
          {einhers.length} <span className="text-textMuted text-sm">strategies</span>
        </div>
      </div>

      <RuneDivider runes="ᚠᚢᚦᚨᚱ" />

      <div className="mt-6 bg-surface border border-border">
        <DataTable
          columns={columns}
          data={einhers}
          rowKey={(row) => row.id}
        />
      </div>
    </motion.div>
  )
}

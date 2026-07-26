import { useState } from 'react'
import { motion } from 'framer-motion'
import { PositionRow } from '@/components/PositionRow'
import { RuneDivider } from '@/components/RuneDivider'
import { cn, formatCurrency } from '@/lib/utils'
import { usePositions } from '@/hooks/useData'

const filters = ['ALL', 'CRYPTO', 'FOREX', 'EQUITY', 'COMMODITY']
const directions = ['ALL', 'LONG', 'SHORT']

export function PositionsPage() {
  const positions = usePositions()
  const [assetFilter, setAssetFilter] = useState('ALL')
  const [dirFilter, setDirFilter] = useState('ALL')

  const filtered = positions.filter((p) => {
    const matchAsset = assetFilter === 'ALL' || p.assetClass === assetFilter
    const matchDir = dirFilter === 'ALL' || p.direction === dirFilter
    return matchAsset && matchDir
  })

  const totalPnl = filtered.reduce((sum, p) => sum + p.pnl, 0)

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      transition={{ duration: 0.4 }}
      className="p-6"
    >
      <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4 mb-6">
        <div>
          <div className="text-[10px] font-cinzel tracking-[0.2em] text-textMuted uppercase">
            Total P&L
          </div>
          <div
            className={cn(
              'font-mono text-2xl mt-1',
              totalPnl >= 0 ? 'text-positive' : 'text-negative'
            )}
          >
            {totalPnl >= 0 ? '+' : ''}
            {formatCurrency(totalPnl)}
          </div>
        </div>

        <div className="flex flex-wrap gap-2">
          {filters.map((f) => (
            <button
              key={f}
              onClick={() => setAssetFilter(f)}
              className={cn(
                'px-3 py-1.5 text-[10px] font-mono uppercase tracking-wider border transition-colors',
                assetFilter === f
                  ? 'border-frost text-frost bg-frost/5'
                  : 'border-border text-textMuted hover:text-textSecondary hover:border-borderHighlight'
              )}
            >
              {f}
            </button>
          ))}
        </div>
      </div>

      <div className="flex gap-2 mb-6">
        {directions.map((d) => (
          <button
            key={d}
            onClick={() => setDirFilter(d)}
            className={cn(
              'px-3 py-1 text-[10px] font-mono uppercase tracking-wider border transition-colors',
              dirFilter === d
                ? d === 'LONG'
                  ? 'border-positive text-positive bg-positive/5'
                  : d === 'SHORT'
                  ? 'border-negative text-negative bg-negative/5'
                  : 'border-frost text-frost bg-frost/5'
                : 'border-border text-textMuted hover:text-textSecondary'
            )}
          >
            {d}
          </button>
        ))}
      </div>

      <RuneDivider runes="ᛈᛉᛊᛏᛒ" />

      <div className="space-y-3 mt-6">
        {filtered.map((pos) => (
          <PositionRow key={pos.id} position={pos} />
        ))}
      </div>
    </motion.div>
  )
}

import { motion } from 'framer-motion'
import { cn, formatCurrency, formatPercent } from '@/lib/utils'
import type { Position } from '@/types'
import { ArrowUp, ArrowDown, Clock } from 'lucide-react'

interface PositionRowProps {
  position: Position
}

export function PositionRow({ position }: PositionRowProps) {
  const isLong = position.direction === 'LONG'
  const isProfit = position.pnl >= 0

  const tpDistance = Math.abs(position.tpPrice - position.entryPrice)
  const currentDistance = Math.abs(position.currentPrice - position.entryPrice)
  const progress = Math.min((currentDistance / tpDistance) * 100, 100)

  return (
    <motion.div
      initial={{ opacity: 0, x: -10 }}
      animate={{ opacity: 1, x: 0 }}
      transition={{ duration: 0.3 }}
      className="bg-surface border border-border p-4 hover:border-borderHighlight transition-colors duration-300 frost-glow"
    >
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-3">
          <span className="font-mono text-sm text-textPrimary">{position.asset}</span>
          <span className="text-[10px] text-textMuted">{position.assetClass}</span>
          <div
            className={cn(
              'flex items-center gap-1 px-1.5 py-0.5 text-[9px] font-mono uppercase',
              isLong
                ? 'text-positive bg-positive/10'
                : 'text-negative bg-negative/10'
            )}
          >
            {isLong ? <ArrowUp size={8} /> : <ArrowDown size={8} />}
            {position.direction}
          </div>
        </div>
        <div className="flex items-center gap-1 text-textMuted">
          <Clock size={10} />
          <span className="text-[10px] font-mono">{position.timeInPosition}</span>
        </div>
      </div>

      <div className="grid grid-cols-3 gap-4 mb-3">
        <div>
          <div className="text-[9px] text-textMuted uppercase tracking-wider mb-0.5">Entry</div>
          <div className="text-[11px] font-mono text-textSecondary">{formatCurrency(position.entryPrice)}</div>
        </div>
        <div>
          <div className="text-[9px] text-textMuted uppercase tracking-wider mb-0.5">Current</div>
          <div className="text-[11px] font-mono text-textSecondary">{formatCurrency(position.currentPrice)}</div>
        </div>
        <div className="text-right">
          <div className="text-[9px] text-textMuted uppercase tracking-wider mb-0.5">P&L</div>
          <div
            className={cn(
              'text-[13px] font-mono',
              isProfit ? 'text-positive' : 'text-negative'
            )}
          >
            {isProfit ? '+' : ''}
            {formatCurrency(position.pnl)}
          </div>
          <div className={cn('text-[10px] font-mono', isProfit ? 'text-positive/70' : 'text-negative/70')}>
            {formatPercent(position.pnlPercent)}
          </div>
        </div>
      </div>

      <div className="relative">
        <div className="flex justify-between text-[9px] text-textMuted mb-1">
          <span>SL {formatCurrency(position.slPrice)}</span>
          <span>TP {formatCurrency(position.tpPrice)}</span>
        </div>
        <div className="h-1 w-full bg-border overflow-hidden">
          <motion.div
            initial={{ width: 0 }}
            animate={{ width: `${progress}%` }}
            transition={{ duration: 0.8, ease: 'easeOut' }}
            className={cn(
              'h-full',
              isProfit ? 'bg-positive' : 'bg-negative'
            )}
          />
        </div>
      </div>

      <div className="mt-2 text-[9px] text-textMuted font-mono">{position.einher}</div>
    </motion.div>
  )
}

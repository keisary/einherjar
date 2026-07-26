import { motion } from 'framer-motion'
import { cn } from '@/lib/utils'
import type { Signal } from '@/types'
import { ArrowUp, ArrowDown } from 'lucide-react'

interface SignalCardProps {
  signal: Signal
}

export function SignalCard({ signal }: SignalCardProps) {
  const metCount = signal.conditions.filter((c) => c.met).length
  const totalCount = signal.conditions.length
  const isTriggered = metCount === totalCount

  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.95 }}
      animate={{ opacity: 1, scale: 1 }}
      whileHover={{ scale: 1.01 }}
      transition={{ duration: 0.3 }}
      className={cn(
        'bg-surface border p-5 transition-all duration-500',
        isTriggered
          ? 'border-frost animate-pulse-slow'
          : 'border-border hover:border-borderHighlight frost-glow'
      )}
    >
      <div className="flex items-start justify-between mb-4">
        <div>
          <div className="text-[10px] font-cinzel tracking-wider text-textMuted uppercase">
            {signal.einher}
          </div>
          <div className="flex items-center gap-2 mt-1">
            <span className="font-mono text-lg text-textPrimary">{signal.asset}</span>
            <span className="text-[10px] text-textMuted">{signal.timeframe}</span>
          </div>
        </div>
        <div
          className={cn(
            'flex items-center gap-1 px-2 py-1 text-[10px] font-mono uppercase tracking-wider',
            signal.direction === 'LONG'
              ? 'text-positive bg-positive/10'
              : 'text-negative bg-negative/10'
          )}
        >
          {signal.direction === 'LONG' ? <ArrowUp size={10} /> : <ArrowDown size={10} />}
          {signal.direction}
        </div>
      </div>

      <div className="space-y-2 mb-4">
        {signal.conditions.map((condition, index) => (
          <div key={index} className="flex items-center gap-2">
            <div
              className={cn(
                'w-1.5 h-1.5',
                condition.met ? 'bg-positive' : 'bg-textMuted'
              )}
            />
            <span
              className={cn(
                'text-[11px]',
                condition.met ? 'text-textSecondary' : 'text-textMuted'
              )}
            >
              {condition.name}
            </span>
          </div>
        ))}
      </div>

      <div className="flex items-center justify-between pt-3 border-t border-border">
        <div className="flex items-center gap-2">
          <span className="text-[10px] text-textMuted uppercase tracking-wider">Confidence</span>
          <span className="text-[11px] font-mono text-frost">
            {Math.round(signal.confidence * 100)}%
          </span>
        </div>
        <div className="flex items-center gap-1.5">
          {Array.from({ length: totalCount }).map((_, i) => (
            <div
              key={i}
              className={cn(
                'w-4 h-1',
                i < metCount ? 'bg-frost' : 'bg-borderHighlight'
              )}
            />
          ))}
        </div>
      </div>
    </motion.div>
  )
}

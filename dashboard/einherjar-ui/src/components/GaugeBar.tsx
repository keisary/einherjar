import { motion } from 'framer-motion'
import { cn } from '@/lib/utils'

interface GaugeBarProps {
  value: number
  max?: number
  label?: string
  showValue?: boolean
  className?: string
}

export function GaugeBar({ value, max = 100, label, showValue = true, className }: GaugeBarProps) {
  const percentage = Math.min((value / max) * 100, 100)

  const getBarColor = () => {
    if (percentage <= 40) return 'bg-frost'
    if (percentage <= 60) return 'bg-ice'
    if (percentage <= 80) return 'bg-warning'
    return 'bg-negative'
  }

  return (
    <div className={cn('w-full', className)}>
      {label && (
        <div className="flex justify-between items-center mb-1.5">
          <span className="text-[10px] font-cinzel tracking-wider text-textMuted uppercase">
            {label}
          </span>
          {showValue && (
            <span className="text-[11px] font-mono text-textSecondary">
              {value}/{max}
            </span>
          )}
        </div>
      )}
      <div className="h-1 w-full bg-border overflow-hidden">
        <motion.div
          initial={{ width: 0 }}
          animate={{ width: `${percentage}%` }}
          transition={{ duration: 1, ease: 'easeOut' }}
          className={cn('h-full', getBarColor())}
        />
      </div>
    </div>
  )
}

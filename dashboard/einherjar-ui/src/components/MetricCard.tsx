import { useEffect, useState } from 'react'
import { motion } from 'framer-motion'
import { cn, formatCurrency, formatPercent, formatNumber } from '@/lib/utils'

interface MetricCardProps {
  label: string
  value: number
  change?: number
  format: 'currency' | 'percent' | 'number'
  className?: string
}

export function MetricCard({ label, value, change, format, className }: MetricCardProps) {
  const [displayValue, setDisplayValue] = useState(0)

  useEffect(() => {
    const duration = 800
    const steps = 30
    const stepDuration = duration / steps
    const increment = value / steps
    let current = 0
    let step = 0

    const timer = setInterval(() => {
      step++
      current += increment
      if (step >= steps) {
        setDisplayValue(value)
        clearInterval(timer)
      } else {
        setDisplayValue(current)
      }
    }, stepDuration)

    return () => clearInterval(timer)
  }, [value])

  const formattedValue =
    format === 'currency'
      ? formatCurrency(displayValue)
      : format === 'percent'
      ? formatPercent(displayValue)
      : formatNumber(displayValue)

  const isPositive = change !== undefined && change >= 0
  const isNegative = change !== undefined && change < 0

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
      className={cn(
        'bg-surface border border-border p-5 transition-all duration-500 hover:border-borderHighlight frost-glow',
        className
      )}
    >
      <div className="text-[10px] font-cinzel tracking-[0.2em] text-textMuted uppercase mb-3">
        {label}
      </div>
      <div className="font-mono text-2xl text-textPrimary tracking-tight">
        {formattedValue}
      </div>
      {change !== undefined && (
        <div
          className={cn(
            'text-[11px] font-mono mt-2',
            isPositive && 'text-positive',
            isNegative && 'text-negative',
            !isPositive && !isNegative && 'text-textMuted'
          )}
        >
          {isPositive ? '+' : ''}
          {format === 'currency' ? formatCurrency(change) : formatPercent(change)}
        </div>
      )}
    </motion.div>
  )
}

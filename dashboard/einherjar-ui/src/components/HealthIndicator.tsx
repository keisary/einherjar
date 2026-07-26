import { cn } from '@/lib/utils'

interface HealthIndicatorProps {
  label: string
  value: string
  status: 'healthy' | 'warning' | 'critical'
}

export function HealthIndicator({ label, value, status }: HealthIndicatorProps) {
  const dotColor =
    status === 'healthy'
      ? 'bg-positive'
      : status === 'warning'
      ? 'bg-warning'
      : 'bg-negative'

  return (
    <div className="flex items-center gap-3">
      <span className={cn('relative flex h-2 w-2', dotColor)}>
        {status === 'critical' && (
          <span className={cn('animate-ping absolute inline-flex h-full w-full opacity-75', dotColor)} />
        )}
        <span className={cn('relative inline-flex h-2 w-2', dotColor)} />
      </span>
      <div>
        <div className="text-[10px] text-textMuted uppercase tracking-wider">{label}</div>
        <div className="text-[11px] font-mono text-textSecondary">{value}</div>
      </div>
    </div>
  )
}

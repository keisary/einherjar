import { motion } from 'framer-motion'
import { MetricCard } from '@/components/MetricCard'
import { HealthIndicator } from '@/components/HealthIndicator'
import { RuneDivider } from '@/components/RuneDivider'
import { FrostGlow } from '@/components/FrostGlow'
import { useBrokers, useAccount } from '@/hooks/useData'
import { useSettings } from '@/contexts/SettingsContext'
import { RuneCrumble } from '@/components/RuneCrumble'

export function HealthPage() {
  const brokers = useBrokers()
  const account = useAccount()
  const { mode } = useSettings()

  const healthyCount = brokers.filter((b) => b.status === 'healthy').length
  const warningCount = brokers.filter((b) => b.status === 'warning').length
  const criticalCount = brokers.filter((b) => b.status === 'critical').length
  const avgLatency = brokers.length > 0
    ? Math.round(brokers.reduce((s, b) => s + b.latency, 0) / brokers.length)
    : 0

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      transition={{ duration: 0.4 }}
      className="p-6 space-y-6"
    >
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        <MetricCard label="Healthy Brokers" value={healthyCount} format="number" />
        <MetricCard label="Warnings" value={warningCount} format="number" />
        <MetricCard label="Critical" value={criticalCount} format="number" />
        <MetricCard label="Avg Latency" value={avgLatency} format="number" />
      </div>

      <RuneDivider runes="ᚺᛁᛃᛇᛈ" />

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <FrostGlow className="bg-surface border border-border p-5 relative overflow-hidden">
          <RuneCrumble density="low" />
          <div className="text-[10px] font-cinzel tracking-[0.2em] text-textMuted uppercase mb-4">
            cTrader Connection
          </div>
          <div className="space-y-4 relative z-10">
            {brokers.length === 0 && (
              <div className="text-xs text-textMuted">No broker data available.</div>
            )}
            {brokers.map((broker) => (
              <HealthIndicator
                key={broker.name}
                label={broker.name}
                value={`${broker.latency}ms · ${broker.lastUpdate.slice(11, 19)}`}
                status={broker.status}
              />
            ))}
          </div>
        </FrostGlow>

        <FrostGlow className="bg-surface border border-border p-5">
          <div className="text-[10px] font-cinzel tracking-[0.2em] text-textMuted uppercase mb-4">
            Account & Scheduler Status
          </div>
          <div className="space-y-3">
            <div className="flex items-center justify-between py-2 border-b border-border">
              <span className="text-[11px] text-textMuted uppercase tracking-wider">Mode</span>
              <span className={`font-mono text-[12px] ${mode === 'live' ? 'text-danger' : 'text-frost'}`}>
                {mode.toUpperCase()}
              </span>
            </div>
            <div className="flex items-center justify-between py-2 border-b border-border">
              <span className="text-[11px] text-textMuted uppercase tracking-wider">cTrader Connected</span>
              <span className={`font-mono text-[12px] ${account?.connected ? 'text-success' : 'text-danger'}`}>
                {account?.connected ? 'YES' : 'NO'}
              </span>
            </div>
            {account?.connected && (
              <>
                <div className="flex items-center justify-between py-2 border-b border-border">
                  <span className="text-[11px] text-textMuted uppercase tracking-wider">Equity</span>
                  <span className="font-mono text-[12px] text-textPrimary">
                    ${account.equity.toLocaleString()} {account.currency}
                  </span>
                </div>
                <div className="flex items-center justify-between py-2 border-b border-border">
                  <span className="text-[11px] text-textMuted uppercase tracking-wider">Leverage</span>
                  <span className="font-mono text-[12px] text-textPrimary">
                    {account.leverage}:1
                  </span>
                </div>
              </>
            )}
            <div className="flex items-center justify-between py-2 border-b border-border">
              <span className="text-[11px] text-textMuted uppercase tracking-wider">Inference Loop</span>
              <span className="font-mono text-[12px] text-textMuted">STOPPED</span>
            </div>
            <div className="flex items-center justify-between py-2 border-b border-border">
              <span className="text-[11px] text-textMuted uppercase tracking-wider">Last Cycle</span>
              <span className="font-mono text-[12px] text-textMuted">--</span>
            </div>
          </div>
          <div className="mt-4 text-[9px] text-textMuted">
            Start the scheduler via main.py to see live metrics.
          </div>
        </FrostGlow>
      </div>
    </motion.div>
  )
}

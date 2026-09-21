import { motion } from 'framer-motion'
import { MetricCard } from '@/components/MetricCard'
import { HealthIndicator } from '@/components/HealthIndicator'
import { RuneDivider } from '@/components/RuneDivider'
import { FrostGlow } from '@/components/FrostGlow'
import { useApiFreshness, useBrokers, useAccount, useEnvironment } from '@/hooks/useData'
import { RuneCrumble } from '@/components/RuneCrumble'

export function HealthPage() {
  const brokers = useBrokers()
  const account = useAccount()
  const env = useEnvironment()

  const sante = useApiFreshness()

  const healthyCount = brokers.filter((b) => b.status === 'healthy').length
  const warningCount = brokers.filter((b) => b.status === 'warning').length
  const criticalCount = brokers.filter((b) => b.status === 'critical').length
  // Latence REELLEMENT mesuree cote navigateur ; `null` (jamais 0) quand aucune
  // mesure n'a encore abouti, pour que la carte affiche « — ».
  const avgLatency = sante.latence

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
                value={
                  broker.lastUpdate
                    ? `${broker.latency ? `${broker.latency}ms` : '—'} · ${broker.lastUpdate.slice(11, 19)} UTC`
                    : '—'
                }
                status={broker.status}
              />
            ))}
            {brokers[0]?.detail && (
              <div className="text-[10px] font-mono text-danger break-words">
                {brokers[0].detail}
              </div>
            )}
            {!brokers[0]?.detail && sante.erreur && (
              <div className="text-[10px] font-mono text-danger break-words">
                Dernier appel API en echec : {sante.erreur}
              </div>
            )}
          </div>
        </FrostGlow>

        <FrostGlow className="bg-surface border border-border p-5">
          <div className="text-[10px] font-cinzel tracking-[0.2em] text-textMuted uppercase mb-4">
            Account & Scheduler Status
          </div>
          <div className="space-y-3">
            <div className="flex items-center justify-between py-2 border-b border-border">
              <span className="text-[11px] text-textMuted uppercase tracking-wider">Mode</span>
              <span className={`font-mono text-[12px] ${env.environment === 'live' ? 'text-danger' : 'text-frost'}`}>
                {(env.environment ?? 'inconnu').toUpperCase()}
              </span>
            </div>
            <div className="flex items-center justify-between py-2 border-b border-border">
              <span className="text-[11px] text-textMuted uppercase tracking-wider">Einhers suivis</span>
              <span className="font-mono text-[12px] text-textPrimary">{env.corpusEinhers}</span>
            </div>
            <div className="flex items-center justify-between py-2 border-b border-border">
              <span className="text-[11px] text-textMuted uppercase tracking-wider">Couples surveilles</span>
              <span className="font-mono text-[12px] text-textPrimary">{env.corpusUnivers}</span>
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
                    {account.equity !== undefined
                      ? `$${account.equity.toLocaleString()} ${account.currency ?? ''}`
                      : '—'}
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
              <span
                className={`font-mono text-[12px] ${
                  env.boucle === null
                    ? 'text-textMuted'
                    : env.boucle.running
                      ? 'text-success'
                      : 'text-danger'
                }`}
              >
                {env.boucle === null ? '—' : env.boucle.running ? 'RUNNING' : 'STOPPED'}
              </span>
            </div>
            <div className="flex items-center justify-between py-2 border-b border-border">
              <span className="text-[11px] text-textMuted uppercase tracking-wider">Last Cycle</span>
              <span className="font-mono text-[12px] text-textMuted">
                {env.boucle?.lastCycleAt
                  ? `${env.boucle.lastCycleAt.slice(11, 19)} UTC · ${env.boucle.cycles ?? 0} cycle(s)`
                  : '—'}
              </span>
            </div>
            {env.boucle && (
              <div className="flex items-center justify-between py-2 border-b border-border">
                <span className="text-[11px] text-textMuted uppercase tracking-wider">Dernier cycle</span>
                <span className="font-mono text-[12px] text-textPrimary">
                  {env.boucle.assets ?? 0} couples · {env.boucle.signals ?? 0} signaux ·{' '}
                  {env.boucle.orders ?? 0} ordres · {env.boucle.errors ?? 0} erreur(s)
                </span>
              </div>
            )}
            {env.couverture && (
              <div className="flex items-center justify-between py-2 border-b border-border">
                <span className="text-[11px] text-textMuted uppercase tracking-wider">Couverture features</span>
                <span
                  className={`font-mono text-[12px] ${
                    env.couverture.couplesEcartes ? 'text-danger' : 'text-success'
                  }`}
                >
                  {env.couverture.couplesVerifies ?? 0} verifies
                  {env.couverture.couplesEcartes ? ` · ${env.couverture.couplesEcartes} ecarte(s)` : ''}
                </span>
              </div>
            )}
          </div>
          {!env.boucle && (
            <div className="mt-4 text-[9px] text-textMuted">
              Aucun cycle publie : la boucle d'inference n'est pas demarree (main.py).
            </div>
          )}
        </FrostGlow>
      </div>
    </motion.div>
  )
}

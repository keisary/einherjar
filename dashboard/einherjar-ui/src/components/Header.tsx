import { motion } from 'framer-motion'
import { useLiveClock, useAccount, useApiFreshness, useEnvironment, useKillSwitch } from '@/hooks/useData'
import { RuneDivider } from './RuneDivider'
import { Link } from 'react-router-dom'
import { Settings } from 'lucide-react'

export function Header() {
  const time = useLiveClock()
  const account = useAccount()
  const freshness = useApiFreshness()
  const env = useEnvironment()
  const [killSwitchEnabled, toggleKillSwitch] = useKillSwitch()

  // Le badge reflete l'ETAT DU SERVEUR (compte cTrader connecte), jamais un
  // reglage local : pas d'API -> STALE, broker non connecte -> HORS LIGNE,
  // sinon DEMO/LIVE selon `environment`.
  const isLive = env.brokerConnected && env.environment === 'live'
  // Aucun mode n'est INVENTE : serveur muet -> INCONNU, broker absent -> HORS LIGNE.
  const badge = freshness.etat === 'stale'
    ? 'HORS LIGNE'
    : !env.brokerConnected
      ? 'HORS LIGNE'
      : env.environment
        ? env.environment.toUpperCase()
        : 'INCONNU'
  const badgeColor = freshness.etat === 'stale' || !env.brokerConnected
    ? 'text-textMuted'
    : isLive ? 'text-danger' : 'text-frost'
  // Classes ecrites en clair : Tailwind ne genere que les classes litterales.
  const dotColor = freshness.etat === 'stale' || !env.brokerConnected
    ? 'bg-textMuted'
    : isLive ? 'bg-danger' : 'bg-frost'

  return (
    <motion.header
      initial={{ opacity: 0, y: -20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.6 }}
      className="border-b border-border"
    >
      <div className="px-6 py-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-4">
            <h1 className="font-cinzel text-xl tracking-widest text-textPrimary">
              EINHERJAR
            </h1>
            <div className="flex items-center gap-2">
              <span className="relative flex h-2 w-2">
                <span
                  className={`animate-ping absolute inline-flex h-full w-full rounded-full opacity-75 ${dotColor}`}
                />
                <span
                  className={`relative inline-flex rounded-full h-2 w-2 ${dotColor}`}
                />
              </span>
              <span
                className={`text-[10px] font-mono uppercase tracking-wider ${badgeColor}`}
                title={freshness.erreur ?? 'Etat fourni par le serveur'}
              >
                {badge}
              </span>
            </div>
          </div>

          <div className="hidden md:flex items-center gap-6 text-[11px] font-mono text-textMuted">
            <div className="flex items-center gap-2">
              <span className="text-frostDark">ᚲ</span>
              <span>{time}</span>
            </div>
            {isLive && account?.connected && (
              <div className="flex items-center gap-3">
                <div className="flex items-center gap-1">
                  <span className="text-frostDark">ᛒ</span>
                  <span className="text-textPrimary">
                    {account.equity !== undefined ? `$${account.equity.toLocaleString()}` : '—'}
                  </span>
                </div>
                <div className="flex items-center gap-1">
                  <span className="text-frostDark">ᛗ</span>
                  <span>
                    {account.marginFree !== undefined ? `$${account.marginFree.toLocaleString()}` : '—'}
                  </span>
                </div>
                <div className="flex items-center gap-1">
                  <span className="text-frostDark">ᛚ</span>
                  <span>{account.leverage}x</span>
                </div>
              </div>
            )}
            <button
              type="button"
              onClick={() => { void toggleKillSwitch() }}
              className={killSwitchEnabled ? 'text-danger' : 'text-textMuted hover:text-danger'}
            >
              {killSwitchEnabled ? 'RESUME' : 'KILL SWITCH'}
            </button>
            <Link
              to="/settings"
              className="flex items-center gap-1 text-textMuted hover:text-frost transition-colors"
            >
              <Settings size={13} />
              <span>Settings</span>
            </Link>
            <a
              href="/logout"
              className="text-textMuted hover:text-danger transition-colors"
              title="Fermer la session"
            >
              Logout
            </a>
          </div>
        </div>
      </div>
      <RuneDivider className="opacity-30" />
    </motion.header>
  )
}

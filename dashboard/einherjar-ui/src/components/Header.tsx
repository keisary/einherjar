import { motion } from 'framer-motion'
import { useLiveClock, useAccount, useApiFreshness, useKillSwitch } from '@/hooks/useData'
import { useSettings } from '@/contexts/SettingsContext'
import { RuneDivider } from './RuneDivider'
import { Link } from 'react-router-dom'
import { Settings } from 'lucide-react'

export function Header() {
  const time = useLiveClock()
  const { mode, activeDemoAccount } = useSettings()
  const account = useAccount()
  const freshness = useApiFreshness()
  const [killSwitchEnabled, toggleKillSwitch] = useKillSwitch()

  const isLive = mode === 'live'

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
                  className={`animate-ping absolute inline-flex h-full w-full rounded-full opacity-75 ${
                    isLive ? 'bg-danger' : 'bg-frost'
                  }`}
                />
                <span
                  className={`relative inline-flex rounded-full h-2 w-2 ${
                    isLive ? 'bg-danger' : 'bg-frost'
                  }`}
                />
              </span>
              <span
                className={`text-[10px] font-mono uppercase tracking-wider ${
                  isLive ? 'text-danger' : 'text-frost'
                }`}
              >
                {freshness === 'stale' ? 'STALE' : isLive ? 'LIVE' : 'DEMO'}
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
                  <span className="text-textPrimary">${account.equity.toLocaleString()}</span>
                </div>
                <div className="flex items-center gap-1">
                  <span className="text-frostDark">ᛗ</span>
                  <span>${account.marginFree.toLocaleString()}</span>
                </div>
                <div className="flex items-center gap-1">
                  <span className="text-frostDark">ᛚ</span>
                  <span>{account.leverage}x</span>
                </div>
              </div>
            )}
            {!isLive && activeDemoAccount && (
              <div className="flex items-center gap-2">
                <span className="text-frostDark">ᚦ</span>
                <span>
                  {activeDemoAccount.name} : {activeDemoAccount.balance.toLocaleString()} {activeDemoAccount.currency}
                </span>
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
          </div>
        </div>
      </div>
      <RuneDivider className="opacity-30" />
    </motion.header>
  )
}

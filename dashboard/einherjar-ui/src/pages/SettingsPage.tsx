import { useState } from 'react'
import { motion } from 'framer-motion'
import { Settings, Plus, Trash2, User, AlertTriangle, Save } from 'lucide-react'
import { useSettings, type Mode } from '@/contexts/SettingsContext'
import { RuneCrumble } from '@/components/RuneCrumble'

interface CTraderConfig {
  client_id: string
  client_secret: string
  access_token: string
  account_id: string
  host: string
  port: string
  broker_name: string
}

function loadCtraderConfig(): CTraderConfig {
  try {
    const raw = localStorage.getItem('einherjar_ctrader_config')
    if (raw) return JSON.parse(raw)
  } catch {}
  return {
    client_id: '',
    client_secret: '',
    access_token: '',
    account_id: '',
    host: 'demo.ctraderapi.com',
    port: '5035',
    broker_name: 'ic_markets',
  }
}

function saveCtraderConfig(cfg: CTraderConfig) {
  localStorage.setItem('einherjar_ctrader_config', JSON.stringify(cfg))
}

export function SettingsPage() {
  const {
    mode,
    setMode,
    demoAccounts,
    activeDemoAccount,
    createDemoAccount,
    switchDemoAccount,
    deleteDemoAccount,
  } = useSettings()

  const [newName, setNewName] = useState('')
  const [newBalance, setNewBalance] = useState(100000)
  const [newCurrency, setNewCurrency] = useState('USD')
  const [showCreate, setShowCreate] = useState(false)

  const [ctrader, setCtrader] = useState<CTraderConfig>(loadCtraderConfig)
  const [saved, setSaved] = useState(false)

  const updateCtrader = (field: keyof CTraderConfig, value: string) => {
    setCtrader((prev) => ({ ...prev, [field]: value }))
    setSaved(false)
  }

  const handleSaveCtrader = () => {
    saveCtraderConfig(ctrader)
    setSaved(true)
    setTimeout(() => setSaved(false), 2000)
  }

  const handleCreate = () => {
    if (!newName.trim()) return
    createDemoAccount(newName.trim(), newBalance, newCurrency)
    setNewName('')
    setNewBalance(100000)
    setShowCreate(false)
  }

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      className="p-6 max-w-4xl mx-auto space-y-8"
    >
      {/* Header */}
      <div className="relative">
        <RuneCrumble density="low" />
        <h1 className="text-2xl font-cinzel tracking-widest text-textPrimary flex items-center gap-3">
          <Settings size={24} className="text-frost" />
          Parameters
        </h1>
        <p className="text-textMuted text-xs mt-1 tracking-wide uppercase">
          Configure the realm of Einherjar
        </p>
      </div>

      {/* Mode Switch */}
      <section className="border border-border bg-surface rounded-lg p-6">
        <h2 className="text-sm font-cinzel tracking-widest text-textSecondary mb-4 uppercase">
          Operation Mode
        </h2>
        <div className="flex items-center gap-4">
          <ModeToggle mode={mode} onChange={setMode} value="live" label="LIVE" />
          <ModeToggle mode={mode} onChange={setMode} value="demo" label="DEMO" />
        </div>
        {mode === 'live' && (
          <div className="mt-4 flex items-center gap-2 text-warning text-xs">
            <AlertTriangle size={14} />
            <span>Live mode connects to cTrader. Trades are real.</span>
          </div>
        )}
        {mode === 'demo' && (
          <div className="mt-4 text-textMuted text-xs">
            Demo mode uses virtual accounts. No real money at risk.
          </div>
        )}
      </section>

      {/* cTrader Configuration (visible in both modes, but especially useful for live) */}
      <section className="border border-border bg-surface rounded-lg p-6">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-sm font-cinzel tracking-widest text-textSecondary uppercase">
            cTrader API Configuration
          </h2>
          <button
            onClick={handleSaveCtrader}
            className={`flex items-center gap-1 text-xs px-3 py-1 rounded transition-colors ${
              saved
                ? 'bg-success text-background'
                : 'bg-frost text-background hover:bg-white'
            }`}
          >
            <Save size={12} />
            {saved ? 'Saved' : 'Save'}
          </button>
        </div>
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
          <div>
            <label className="block text-[10px] uppercase tracking-wider text-textMuted mb-1">Client ID</label>
            <input
              type="text"
              value={ctrader.client_id}
              onChange={(e) => updateCtrader('client_id', e.target.value)}
              className="w-full bg-background border border-border rounded px-3 py-2 text-xs text-textPrimary placeholder:text-textMuted focus:outline-none focus:border-frost"
              placeholder="54 chars"
            />
          </div>
          <div>
            <label className="block text-[10px] uppercase tracking-wider text-textMuted mb-1">Client Secret</label>
            <input
              type="password"
              value={ctrader.client_secret}
              onChange={(e) => updateCtrader('client_secret', e.target.value)}
              className="w-full bg-background border border-border rounded px-3 py-2 text-xs text-textPrimary placeholder:text-textMuted focus:outline-none focus:border-frost"
              placeholder="50 chars"
            />
          </div>
          <div>
            <label className="block text-[10px] uppercase tracking-wider text-textMuted mb-1">Access Token</label>
            <input
              type="password"
              value={ctrader.access_token}
              onChange={(e) => updateCtrader('access_token', e.target.value)}
              className="w-full bg-background border border-border rounded px-3 py-2 text-xs text-textPrimary placeholder:text-textMuted focus:outline-none focus:border-frost"
              placeholder="43 chars"
            />
          </div>
          <div>
            <label className="block text-[10px] uppercase tracking-wider text-textMuted mb-1">Account ID</label>
            <input
              type="text"
              value={ctrader.account_id}
              onChange={(e) => updateCtrader('account_id', e.target.value)}
              className="w-full bg-background border border-border rounded px-3 py-2 text-xs text-textPrimary placeholder:text-textMuted focus:outline-none focus:border-frost"
              placeholder="ctidTraderAccountId"
            />
          </div>
          <div>
            <label className="block text-[10px] uppercase tracking-wider text-textMuted mb-1">Host</label>
            <select
              value={ctrader.host}
              onChange={(e) => updateCtrader('host', e.target.value)}
              className="w-full bg-background border border-border rounded px-3 py-2 text-xs text-textPrimary focus:outline-none focus:border-frost"
            >
              <option value="demo.ctraderapi.com">demo.ctraderapi.com</option>
              <option value="live.ctraderapi.com">live.ctraderapi.com</option>
            </select>
          </div>
          <div>
            <label className="block text-[10px] uppercase tracking-wider text-textMuted mb-1">Broker</label>
            <select
              value={ctrader.broker_name}
              onChange={(e) => updateCtrader('broker_name', e.target.value)}
              className="w-full bg-background border border-border rounded px-3 py-2 text-xs text-textPrimary focus:outline-none focus:border-frost"
            >
              <option value="ic_markets">IC Markets</option>
              <option value="pepperstone">Pepperstone</option>
            </select>
          </div>
        </div>
        <p className="mt-3 text-[10px] text-textMuted">
          Credentials are stored locally in your browser. See docs/GUIDE_CTRADER.md for setup instructions.
        </p>
      </section>

      {/* Demo Accounts */}
      {mode === 'demo' && (
        <section className="border border-border bg-surface rounded-lg p-6">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-sm font-cinzel tracking-widest text-textSecondary uppercase">
              Demo Accounts
            </h2>
            <button
              onClick={() => setShowCreate(!showCreate)}
              className="flex items-center gap-1 text-xs text-frost hover:text-white transition-colors"
            >
              <Plus size={14} />
              New Account
            </button>
          </div>

          {showCreate && (
            <motion.div
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: 'auto' }}
              className="mb-4 p-4 border border-border bg-surfaceHighlight rounded"
            >
              <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
                <input
                  type="text"
                  placeholder="Account name"
                  value={newName}
                  onChange={(e) => setNewName(e.target.value)}
                  className="bg-background border border-border rounded px-3 py-2 text-xs text-textPrimary placeholder:text-textMuted focus:outline-none focus:border-frost"
                />
                <input
                  type="number"
                  placeholder="Balance"
                  value={newBalance}
                  onChange={(e) => setNewBalance(Number(e.target.value))}
                  className="bg-background border border-border rounded px-3 py-2 text-xs text-textPrimary placeholder:text-textMuted focus:outline-none focus:border-frost"
                />
                <select
                  value={newCurrency}
                  onChange={(e) => setNewCurrency(e.target.value)}
                  className="bg-background border border-border rounded px-3 py-2 text-xs text-textPrimary focus:outline-none focus:border-frost"
                >
                  <option value="USD">USD</option>
                  <option value="EUR">EUR</option>
                  <option value="USDT">USDT</option>
                </select>
              </div>
              <div className="flex justify-end mt-3 gap-2">
                <button
                  onClick={() => setShowCreate(false)}
                  className="px-3 py-1 text-xs text-textMuted hover:text-textPrimary transition-colors"
                >
                  Cancel
                </button>
                <button
                  onClick={handleCreate}
                  className="px-3 py-1 text-xs bg-frost text-background rounded hover:bg-white transition-colors"
                >
                  Create
                </button>
              </div>
            </motion.div>
          )}

          <div className="space-y-2">
            {demoAccounts.map((acc) => (
              <div
                key={acc.id}
                className={`flex items-center justify-between p-3 rounded border transition-colors ${
                  activeDemoAccount?.id === acc.id
                    ? 'border-frost bg-surfaceHighlight'
                    : 'border-border hover:border-textMuted'
                }`}
              >
                <div className="flex items-center gap-3">
                  <User size={16} className={activeDemoAccount?.id === acc.id ? 'text-frost' : 'text-textMuted'} />
                  <div>
                    <div className="text-xs font-medium text-textPrimary">{acc.name}</div>
                    <div className="text-[10px] text-textMuted">
                      {acc.balance.toLocaleString()} {acc.currency}
                    </div>
                  </div>
                </div>
                <div className="flex items-center gap-2">
                  {activeDemoAccount?.id !== acc.id && (
                    <button
                      onClick={() => switchDemoAccount(acc.id)}
                      className="text-[10px] px-2 py-1 border border-border rounded text-textMuted hover:text-frost hover:border-frost transition-colors"
                    >
                      Activate
                    </button>
                  )}
                  {demoAccounts.length > 1 && (
                    <button
                      onClick={() => deleteDemoAccount(acc.id)}
                      className="text-textMuted hover:text-danger transition-colors"
                    >
                      <Trash2 size={14} />
                    </button>
                  )}
                </div>
              </div>
            ))}
          </div>
        </section>
      )}
    </motion.div>
  )
}

function ModeToggle({
  mode,
  onChange,
  value,
  label,
}: {
  mode: Mode
  onChange: (m: Mode) => void
  value: Mode
  label: string
}) {
  const active = mode === value
  return (
    <button
      onClick={() => onChange(value)}
      className={`relative px-6 py-2 text-xs font-cinzel tracking-widest uppercase rounded transition-all duration-300 ${
        active
          ? 'bg-frost text-background shadow-[0_0_15px_rgba(180,200,220,0.3)]'
          : 'bg-surfaceHighlight text-textMuted border border-border hover:border-textMuted'
      }`}
    >
      {active && (
        <motion.div
          layoutId="mode-pill"
          className="absolute inset-0 bg-frost rounded -z-10"
          transition={{ type: 'spring', stiffness: 300, damping: 30 }}
        />
      )}
      {label}
    </button>
  )
}

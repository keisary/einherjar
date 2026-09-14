import { motion } from 'framer-motion'
import { AlertTriangle, KeyRound, Server, Settings, ShieldCheck } from 'lucide-react'
import { RuneCrumble } from '@/components/RuneCrumble'
import { useAccount, useEnvironment } from '@/hooks/useData'

/** Une ligne de statut (meme style que la page Sante). */
function Ligne({ label, valeur, classe }: { label: string; valeur: string; classe?: string }) {
  return (
    <div className="flex items-center justify-between py-2 border-b border-border">
      <span className="text-[11px] text-textMuted uppercase tracking-wider">{label}</span>
      <span className={`font-mono text-[12px] ${classe ?? 'text-textPrimary'}`}>{valeur}</span>
    </div>
  )
}

export function SettingsPage() {
  const env = useEnvironment()
  const account = useAccount()

  const environnement =
    env.environment === 'live' ? 'LIVE' : env.environment === 'demo' ? 'DEMO' : 'NON CONFIGURE'
  const couleurEnv =
    env.environment === 'live' ? 'text-danger' : env.environment === 'demo' ? 'text-frost' : 'text-warning'
  const compte =
    account?.accountId !== undefined && account?.accountId !== null
      ? `••••${String(account.accountId).slice(-4)}`
      : '—'

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
          Etat du systeme — determine par le serveur
        </p>
      </div>

      {/* Environnement : lecture seule */}
      <section className="border border-border bg-surface rounded-lg p-6">
        <h2 className="text-sm font-cinzel tracking-widest text-textSecondary mb-4 uppercase flex items-center gap-2">
          <ShieldCheck size={14} className="text-frost" />
          Operation Mode
        </h2>
        <Ligne label="Environnement" valeur={environnement} classe={couleurEnv} />
        <Ligne
          label="Compte cTrader"
          valeur={account?.connected ? 'CONNECTE' : 'NON CONNECTE'}
          classe={account?.connected ? 'text-success' : 'text-danger'}
        />
        <Ligne label="Identifiant compte" valeur={compte} />
        <div className="mt-4 flex items-start gap-2 text-textMuted text-xs">
          <AlertTriangle size={14} className="mt-0.5 shrink-0 text-warning" />
          <span>
            L'environnement vient des identifiants du serveur (<code className="font-mono">config/credentials.json</code>) :
            le mode demo utilise le compte demo cTrader, il n'existe pas de compte virtuel cote interface.
            Aucun reglage du navigateur ne peut changer l'environnement de trading.
          </span>
        </div>
        {env.environment === null && (
          <p className="mt-3 text-xs text-warning">
            Aucun identifiant charge : renseigner <code className="font-mono">config/credentials.json</code> pour
            activer le compte demo cTrader.
          </p>
        )}
      </section>

      {/* broker : lecture seule */}
      <section className="border border-border bg-surface rounded-lg p-6">
        <h2 className="text-sm font-cinzel tracking-widest text-textSecondary mb-4 uppercase flex items-center gap-2">
          <Server size={14} className="text-frost" />
          Broker cTrader
        </h2>
        <Ligne label="Hote" valeur={env.host ?? '—'} />
        <Ligne
          label="Circuit breaker"
          valeur={env.circuitState ?? '—'}
          classe={env.circuitState === 'CLOSED' ? 'text-success' : 'text-warning'}
        />
        <Ligne label="Einhers du corpus" valeur={String(env.corpusEinhers)} />
        <Ligne label="Couples surveilles" valeur={String(env.corpusUnivers)} />
        <div className="mt-4 flex items-start gap-2 text-textMuted text-xs">
          <KeyRound size={14} className="mt-0.5 shrink-0" />
          <span>
            Les identifiants (client_id, client_secret, access_token, account_id) restent sur le serveur,
            dans <code className="font-mono">config/credentials.json</code> (modele versionne :
            <code className="font-mono"> credentials.example.json</code>). Ils ne transitent jamais par le
            navigateur.
          </span>
        </div>
      </section>
    </motion.div>
  )
}

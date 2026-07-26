import { useState } from 'react'
import { motion } from 'framer-motion'
import { SignalCard } from '@/components/SignalCard'
import { RuneDivider } from '@/components/RuneDivider'
import { cn } from '@/lib/utils'
import { useSignals } from '@/hooks/useData'

export function FormingPage() {
  const signals = useSignals()
  const [filter, setFilter] = useState<'all' | '1' | '2' | '3'>('all')

  const filtered = signals.filter((s) => {
    const metCount = s.conditions.filter((c) => c.met).length
    if (filter === 'all') return true
    return String(metCount) === filter
  })

  const counts = {
    all: signals.length,
    '1': signals.filter((s) => s.conditions.filter((c) => c.met).length === 1).length,
    '2': signals.filter((s) => s.conditions.filter((c) => c.met).length === 2).length,
    '3': signals.filter((s) => s.conditions.filter((c) => c.met).length === 3).length,
  }

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      transition={{ duration: 0.4 }}
      className="p-6"
    >
      <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4 mb-6">
        <div>
          <div className="text-[10px] font-cinzel tracking-[0.2em] text-textMuted uppercase">
            Forming Signals
          </div>
          <div className="font-mono text-2xl text-textPrimary mt-1">
            {filtered.length} <span className="text-textMuted text-sm">active</span>
          </div>
        </div>

        <div className="flex gap-2">
          {(['all', '1', '2', '3'] as const).map((f) => (
            <button
              key={f}
              onClick={() => setFilter(f)}
              className={cn(
                'px-3 py-1.5 text-[10px] font-mono uppercase tracking-wider border transition-colors',
                filter === f
                  ? 'border-frost text-frost bg-frost/5'
                  : 'border-border text-textMuted hover:text-textSecondary hover:border-borderHighlight'
              )}
            >
              {f === 'all' ? 'All' : `${f}/3`} ({counts[f]})
            </button>
          ))}
        </div>
      </div>

      <RuneDivider runes="ᛖᛗᛚᛜᛞ" />

      <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-4 mt-6">
        {filtered.map((sig, i) => (
          <motion.div
            key={sig.id}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.3, delay: i * 0.05 }}
          >
            <SignalCard signal={sig} />
          </motion.div>
        ))}
      </div>
    </motion.div>
  )
}

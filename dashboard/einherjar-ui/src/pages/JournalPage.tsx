import { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { RuneDivider } from '@/components/RuneDivider'
import { cn } from '@/lib/utils'
import { useJournal } from '@/hooks/useData'
import type { JournalEntry } from '@/types'
import { ChevronDown, ChevronUp } from 'lucide-react'

const typeColors: Record<string, string> = {
  ORDER: 'text-positive bg-positive/10',
  SIGNAL: 'text-frost bg-frost/10',
  CLOSE: 'text-ice bg-ice/10',
  REJECT: 'text-negative bg-negative/10',
  FORMING: 'text-warning bg-warning/10',
}

export function JournalPage() {
  const journal = useJournal()
  const [expandedId, setExpandedId] = useState<string | null>(null)

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      transition={{ duration: 0.4 }}
      className="p-6"
    >
      <div className="mb-6">
        <div className="text-[10px] font-cinzel tracking-[0.2em] text-textMuted uppercase">
          System Journal
        </div>
        <div className="font-mono text-2xl text-textPrimary mt-1">
          {journal.length} <span className="text-textMuted text-sm">entries</span>
        </div>
      </div>

      <RuneDivider runes="ᚲᚷᚹᚺᚾ" />

      <div className="mt-6 space-y-2">
        {journal.map((entry) => (
          <JournalItem
            key={entry.id}
            entry={entry}
            expanded={expandedId === entry.id}
            onToggle={() => setExpandedId(expandedId === entry.id ? null : entry.id)}
          />
        ))}
      </div>
    </motion.div>
  )
}

function JournalItem({
  entry,
  expanded,
  onToggle,
}: {
  entry: JournalEntry
  expanded: boolean
  onToggle: () => void
}) {
  return (
    <div className="bg-surface border border-border hover:border-borderHighlight transition-colors">
      <button
        onClick={onToggle}
        className="w-full flex items-center gap-4 px-4 py-3 text-left"
      >
        <span
          className={cn(
            'px-2 py-0.5 text-[9px] font-mono uppercase tracking-wider shrink-0',
            typeColors[entry.type] ?? 'text-textMuted bg-textMuted/10'
          )}
        >
          {entry.type}
        </span>
        <span className="text-[10px] font-mono text-textMuted shrink-0 w-20">
          {entry.timestamp.slice(11, 16)}
        </span>
        <span className="font-mono text-[12px] text-textPrimary shrink-0 w-16">
          {entry.asset}
        </span>
        <span className="text-[11px] text-textSecondary flex-1 truncate">{entry.details}</span>
        {entry.pnl !== undefined && (
          <span
            className={cn(
              'text-[11px] font-mono shrink-0',
              entry.pnl >= 0 ? 'text-positive' : 'text-negative'
            )}
          >
            {entry.pnl >= 0 ? '+' : ''}${entry.pnl.toFixed(2)}
          </span>
        )}
        {expanded ? (
          <ChevronUp size={14} className="text-textMuted shrink-0" />
        ) : (
          <ChevronDown size={14} className="text-textMuted shrink-0" />
        )}
      </button>

      <AnimatePresence>
        {expanded && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.2 }}
            className="overflow-hidden"
          >
            <div className="px-4 pb-3 pt-1 border-t border-border">
              <div className="grid grid-cols-2 gap-4 text-[11px]">
                <div>
                  <span className="text-textMuted uppercase tracking-wider">Einher</span>
                  <p className="font-mono text-textSecondary mt-0.5">{entry.einher}</p>
                </div>
                <div>
                  <span className="text-textMuted uppercase tracking-wider">Timestamp</span>
                  <p className="font-mono text-textSecondary mt-0.5">{entry.timestamp}</p>
                </div>
                <div className="col-span-2">
                  <span className="text-textMuted uppercase tracking-wider">Details</span>
                  <p className="text-textSecondary mt-0.5">{entry.details}</p>
                </div>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}

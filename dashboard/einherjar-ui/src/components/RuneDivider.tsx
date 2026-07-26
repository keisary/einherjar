import { motion } from 'framer-motion'
import { cn } from '@/lib/utils'

interface RuneDividerProps {
  className?: string
  runes?: string
}

export function RuneDivider({ className, runes = 'ᚠᚢᚦᚨᚱᚲᚷᚹ' }: RuneDividerProps) {
  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ duration: 1.5 }}
      className={cn(
        'flex items-center justify-center gap-3 py-2 rune-pulse select-none',
        className
      )}
    >
      <div className="h-[1px] flex-1 bg-border" />
      <span className="text-[10px] text-textMuted tracking-[0.3em] font-mono">
        {runes}
      </span>
      <div className="h-[1px] flex-1 bg-border" />
    </motion.div>
  )
}

import { motion } from 'framer-motion'
import { cn } from '@/lib/utils'

interface FrostGlowProps {
  children: React.ReactNode
  className?: string
}

export function FrostGlow({ children, className }: FrostGlowProps) {
  return (
    <motion.div
      whileHover={{ scale: 1.005 }}
      transition={{ duration: 0.3 }}
      className={cn(
        'transition-all duration-500 hover:border-borderHighlight frost-glow',
        className
      )}
    >
      {children}
    </motion.div>
  )
}

import { useState } from 'react'
import { Link, useLocation } from 'react-router-dom'
import { motion } from 'framer-motion'
import { Menu, X } from 'lucide-react'
import { cn } from '@/lib/utils'

const navItems = [
  { path: '/', label: 'Overview', rune: 'ᚢ' },
  { path: '/positions', label: 'Positions', rune: 'ᚦ' },
  { path: '/forming', label: 'Forming', rune: 'ᚨ' },
  { path: '/performance', label: 'Performance', rune: 'ᚱ' },
  { path: '/journal', label: 'Journal', rune: 'ᚲ' },
  { path: '/health', label: 'Health', rune: 'ᚷ' },
  { path: '/settings', label: 'Settings', rune: 'ᛟ' },
]


export function Navigation() {
  const location = useLocation()
  const [mobileOpen, setMobileOpen] = useState(false)

  return (
    <nav className="border-b border-border">
      <div className="px-6">
        {/* Desktop nav */}
        <div className="hidden md:flex items-center gap-0">
          {navItems.map((item, index) => (
            <div key={item.path} className="flex items-center">
              {index > 0 && (
                <span className="text-textMuted text-[10px] px-3 select-none">
                  {item.rune}
                </span>
              )}
              <Link
                to={item.path}
                className={cn(
                  'relative py-3 px-4 text-[11px] font-cinzel tracking-widest uppercase transition-colors duration-300',
                  location.pathname === item.path
                    ? 'text-frost'
                    : 'text-textMuted hover:text-textSecondary'
                )}
              >
                {item.label}
                {location.pathname === item.path && (
                  <motion.div
                    layoutId="nav-underline"
                    className="absolute bottom-0 left-0 right-0 h-[1px] bg-frost"
                    transition={{ type: 'spring', stiffness: 300, damping: 30 }}
                  />
                )}
              </Link>
            </div>
          ))}
        </div>

        {/* Mobile toggle */}
        <div className="flex md:hidden items-center justify-between py-3">
          <span className="text-[11px] font-cinzel text-textMuted tracking-widest uppercase">
            {navItems.find((i) => i.path === location.pathname)?.label ?? 'Menu'}
          </span>
          <button
            onClick={() => setMobileOpen(!mobileOpen)}
            className="text-textSecondary hover:text-textPrimary transition-colors"
          >
            {mobileOpen ? <X size={18} /> : <Menu size={18} />}
          </button>
        </div>
      </div>

      {/* Mobile menu */}
      {mobileOpen && (
        <motion.div
          initial={{ opacity: 0, height: 0 }}
          animate={{ opacity: 1, height: 'auto' }}
          exit={{ opacity: 0, height: 0 }}
          className="md:hidden border-t border-border"
        >
          {navItems.map((item) => (
            <Link
              key={item.path}
              to={item.path}
              onClick={() => setMobileOpen(false)}
              className={cn(
                'block py-3 px-6 text-[11px] font-cinzel tracking-widest uppercase border-b border-border transition-colors',
                location.pathname === item.path
                  ? 'text-frost bg-surfaceHighlight'
                  : 'text-textMuted hover:text-textSecondary hover:bg-surfaceHighlight'
              )}
            >
              <span className="text-textMuted mr-3">{item.rune}</span>
              {item.label}
            </Link>
          ))}
        </motion.div>
      )}
    </nav>
  )
}

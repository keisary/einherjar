import { createContext, useContext, useState, useCallback, type ReactNode } from 'react'

export type Mode = 'live' | 'demo'

export interface DemoAccount {
  id: string
  name: string
  balance: number
  currency: string
  createdAt: string
}

interface SettingsContextType {
  mode: Mode
  setMode: (m: Mode) => void
  demoAccounts: DemoAccount[]
  activeDemoAccount: DemoAccount | null
  createDemoAccount: (name: string, balance: number, currency: string) => void
  switchDemoAccount: (id: string) => void
  deleteDemoAccount: (id: string) => void
}

const SettingsContext = createContext<SettingsContextType | null>(null)

export function SettingsProvider({ children }: { children: ReactNode }) {
  const [mode, setMode] = useState<Mode>('demo')
  const [demoAccounts, setDemoAccounts] = useState<DemoAccount[]>([
    { id: 'demo-1', name: 'Valhalla Test', balance: 100000, currency: 'USD', createdAt: new Date().toISOString() },
  ])
  const [activeDemoAccount, setActiveDemoAccount] = useState<DemoAccount | null>(demoAccounts[0])

  const createDemoAccount = useCallback((name: string, balance: number, currency: string) => {
    const acc: DemoAccount = {
      id: `demo-${Date.now()}`,
      name,
      balance,
      currency,
      createdAt: new Date().toISOString(),
    }
    setDemoAccounts((prev) => [...prev, acc])
    setActiveDemoAccount(acc)
  }, [])

  const switchDemoAccount = useCallback((id: string) => {
    setDemoAccounts((prev) => {
      const found = prev.find((a) => a.id === id) ?? null
      setActiveDemoAccount(found)
      return prev
    })
  }, [])

  const deleteDemoAccount = useCallback((id: string) => {
    setDemoAccounts((prev) => {
      const next = prev.filter((a) => a.id !== id)
      setActiveDemoAccount((cur) => (cur?.id === id ? (next[0] ?? null) : cur))
      return next
    })
  }, [])

  return (
    <SettingsContext.Provider
      value={{
        mode,
        setMode,
        demoAccounts,
        activeDemoAccount,
        createDemoAccount,
        switchDemoAccount,
        deleteDemoAccount,
      }}
    >
      {children}
    </SettingsContext.Provider>
  )
}

export function useSettings() {
  const ctx = useContext(SettingsContext)
  if (!ctx) throw new Error('useSettings must be used within SettingsProvider')
  return ctx
}

import { Routes, Route } from 'react-router-dom'
import { AnimatePresence } from 'framer-motion'
import { Header } from '@/components/Header'
import { Navigation } from '@/components/Navigation'
import { OverviewPage } from '@/pages/OverviewPage'
import { PositionsPage } from '@/pages/PositionsPage'
import { FormingPage } from '@/pages/FormingPage'
import { PerformancePage } from '@/pages/PerformancePage'
import { JournalPage } from '@/pages/JournalPage'
import { HealthPage } from '@/pages/HealthPage'
import { SettingsPage } from '@/pages/SettingsPage'

export default function App() {
  return (
    <div className="min-h-screen bg-background text-textPrimary">
      <Header />
      <Navigation />
      <main>
        <AnimatePresence mode="wait">
          <Routes>
            <Route path="/" element={<OverviewPage />} />
            <Route path="/positions" element={<PositionsPage />} />
            <Route path="/forming" element={<FormingPage />} />
            <Route path="/performance" element={<PerformancePage />} />
            <Route path="/journal" element={<JournalPage />} />
            <Route path="/health" element={<HealthPage />} />
            <Route path="/settings" element={<SettingsPage />} />
          </Routes>
        </AnimatePresence>
      </main>
    </div>
  )
}

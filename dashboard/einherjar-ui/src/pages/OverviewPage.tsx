import { motion } from 'framer-motion'
import { MetricCard } from '@/components/MetricCard'
import { PositionRow } from '@/components/PositionRow'
import { SignalCard } from '@/components/SignalCard'
import { GaugeBar } from '@/components/GaugeBar'
import { RuneDivider } from '@/components/RuneDivider'
import { FrostGlow } from '@/components/FrostGlow'
import { useMetrics, usePositions, useSignals, useEquityData, useExposure } from '@/hooks/useData'
import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
} from 'recharts'

export function OverviewPage() {
  const metrics = useMetrics()
  const positions = usePositions()
  const signals = useSignals()
  const equityData = useEquityData()
  const exposure = useExposure()

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      transition={{ duration: 0.4 }}
      className="p-6 space-y-6"
    >
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        {metrics.map((m) => (
          <MetricCard
            key={m.label}
            label={m.label}
            value={m.value}
            change={m.change}
            format={m.format}
          />
        ))}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <FrostGlow className="lg:col-span-2 bg-surface border border-border p-5">
          <div className="text-[10px] font-cinzel tracking-[0.2em] text-textMuted uppercase mb-4">
            Equity Curve
          </div>
          <div className="h-[280px]">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={equityData}>
                <defs>
                  <linearGradient id="equityGradient" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stopColor="#B0C4DE" stopOpacity={0.15} />
                    <stop offset="100%" stopColor="#B0C4DE" stopOpacity={0} />
                  </linearGradient>
                </defs>
                <XAxis
                  dataKey="time"
                  tick={{ fill: '#4A4A4A', fontSize: 10, fontFamily: 'JetBrains Mono' }}
                  axisLine={{ stroke: '#1A1A1A' }}
                  tickLine={false}
                />
                <YAxis
                  tick={{ fill: '#4A4A4A', fontSize: 10, fontFamily: 'JetBrains Mono' }}
                  axisLine={false}
                  tickLine={false}
                  domain={['auto', 'auto']}
                  tickFormatter={(v) => `$${(v / 1000).toFixed(0)}k`}
                />
                <Tooltip
                  contentStyle={{
                    backgroundColor: '#0A0A0A',
                    border: '1px solid #1A1A1A',
                    fontSize: 11,
                    fontFamily: 'JetBrains Mono',
                    color: '#E8E8E8',
                  }}
                  formatter={(value: number) => [`$${value.toLocaleString()}`, 'Equity']}
                />
                <Area
                  type="monotone"
                  dataKey="value"
                  stroke="#B0C4DE"
                  strokeWidth={1}
                  fill="url(#equityGradient)"
                />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </FrostGlow>

        <div className="space-y-4">
          <FrostGlow className="bg-surface border border-border p-5">
            <div className="text-[10px] font-cinzel tracking-[0.2em] text-textMuted uppercase mb-4">
              Exposure by Class
            </div>
            <div className="space-y-4">
              {exposure.map((e) => (
                <GaugeBar
                  key={e.class}
                  label={e.class}
                  value={e.value}
                  max={e.max}
                />
              ))}
            </div>
          </FrostGlow>
        </div>
      </div>

      <RuneDivider runes="ᚺᚾᛁᛃᛇ" />

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div>
          <div className="text-[10px] font-cinzel tracking-[0.2em] text-textMuted uppercase mb-4">
            Active Positions
          </div>
          <div className="space-y-3">
            {positions.slice(0, 4).map((pos) => (
              <PositionRow key={pos.id} position={pos} />
            ))}
          </div>
        </div>

        <div>
          <div className="text-[10px] font-cinzel tracking-[0.2em] text-textMuted uppercase mb-4">
            Forming Signals
          </div>
          <div className="space-y-3">
            {signals.slice(0, 3).map((sig) => (
              <SignalCard key={sig.id} signal={sig} />
            ))}
          </div>
        </div>
      </div>
    </motion.div>
  )
}

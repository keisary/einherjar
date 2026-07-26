export interface Position {
  id: string;
  asset: string;
  assetClass: string;
  direction: 'LONG' | 'SHORT';
  entryPrice: number;
  currentPrice: number;
  quantity: number;
  tpPrice: number;
  slPrice: number;
  pnl: number;
  pnlPercent: number;
  timeInPosition: string;
  einher: string;
}

export interface Signal {
  id: string;
  einher: string;
  asset: string;
  timeframe: string;
  direction: 'LONG' | 'SHORT';
  confidence: number;
  conditions: {
    name: string;
    met: boolean;
  }[];
  triggered: boolean;
}

export interface Einher {
  id: string;
  name: string;
  description: string;
  status: 'ACTIVE' | 'PROBATION' | 'DISABLED';
  winRate: number;
  totalTrades: number;
  avgReturn: number;
  sharpe: number;
  lastSignal: string;
}

export interface JournalEntry {
  id: string;
  timestamp: string;
  type: 'ORDER' | 'SIGNAL' | 'CLOSE' | 'REJECT' | 'FORMING';
  asset: string;
  einher: string;
  details: string;
  pnl?: number;
}

export interface HealthStatus {
  label: string;
  value: string;
  status: 'healthy' | 'warning' | 'critical';
}

export interface BrokerStatus {
  name: string;
  lastUpdate: string;
  latency: number;
  status: 'healthy' | 'warning' | 'critical';
}

export interface Metric {
  label: string;
  value: number;
  change?: number;
  format: 'currency' | 'percent' | 'number';
}

export interface ExposureData {
  class: string;
  value: number;
  max: number;
}

export interface EquityPoint {
  time: string;
  value: number;
}

export interface Account {
  balance: number;
  equity: number;
  margin: number;
  marginFree: number;
  leverage: number;
  currency: string;
  connected: boolean;
  accountId: number;
}

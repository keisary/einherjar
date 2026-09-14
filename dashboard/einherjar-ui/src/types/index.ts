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
  /** Condition lisible (issue de l'arbre de conditions du corpus). */
  description: string;
  /** null = einher jamais evalue en live : afficher "—", jamais un etat invente. */
  status: 'ACTIVE' | 'PROBATION' | 'DISABLED' | null;
  winRate: number | null;
  totalTrades: number | null;
  avgReturn: number | null;
  sharpe: number | null;
  lastSignal: string | null;
  // --- Champs de recherche issus du corpus (source de verite des einhers) ---
  asset: string | null;
  assetClass: string | null;
  timeframe: string | null;
  horizon: string | null;
  direction: 'BUY' | 'SELL' | null;
  amplitudeBars: number | null;
  tpPct: number | null;
  slPct: number | null;
  totalReturn: number | null;
  maxDrawdown: number | null;
  profitFactor: number | null;
  avgHoldingBars: number | null;
  tpHitRate: number | null;
  alpha: number | null;
  pValue: number | null;
  model: string | null;
}

/** Un couple (actif, timeframe) et le nombre d'einhers du corpus qui le surveillent. */
export interface UniverseCount {
  asset: string;
  timeframe: string;
  assetClass: string;
  einhers: number;
}

/** Agregats reels calcules par le serveur sur la selection courante. */
export interface EinherSummary {
  sharpeMedian: number | null;
  winRateMedian: number | null;
  avgReturnMedian: number | null;
  totalReturnMedian: number | null;
  alphaMedian: number | null;
  pValueMedian: number | null;
  tradesTotal: number;
}

export interface EinherPage {
  total: number;
  returned: number;
  einhers: Einher[];
  summary: EinherSummary;
  universes: UniverseCount[];
  classes: Record<string, number>;
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
  /** null = donnee indisponible (ex. aucune courbe d'equity) : afficher "—", jamais 0 */
  value: number | null;
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
  /** Le serveur renvoie { connected: false, reason } tant qu'aucun broker n'est connecte. */
  connected: boolean;
  reason?: string;
  balance?: number;
  equity?: number;
  margin?: number;
  marginFree?: number;
  leverage?: number;
  currency?: string;
  accountId?: number;
}

"""Broker de test local : rejeu d'OHLCV reel + compte papier.

Utilise UNIQUEMENT par les scripts de test et les tests unitaires :
- `scripts/smoke_demo_cycle.py` (cycle complet hors broker reel)
- tests d'orchestration de l'InferenceLoop

Le chemin de production est unique : `einherjar.brokers.ctrader_adapter.CTraderAdapter`
(host `demo.ctraderapi.com` ou live selon `environment`). Aucun mode "demo local"
n'existe dans `main.py` : sans credentials cTrader valides, le systeme s'arrete.

Les bougies servies sont les CSV locaux (`technical_agent_dataset_brut`), donc un
REJEU, pas un flux temps reel ; les bougies synthetiques ne sont qu'un repli pour
les actifs absents du disque.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

import polars as pl

logger = logging.getLogger(__name__)


class LocalReplayBroker:
    """Broker de test : rejoue l'OHLCV local et tient un compte papier.

Doublure de TEST uniquement (scripts de smoke, tests unitaires). Le systeme
en production n'a QU'UN point d'entree de donnees : `CTraderAdapter` (compte
demo ou live selon `environment` dans config/credentials.json). Ce module ne
doit jamais etre selectionne par `main.py`.
"""

    name = "mock"

    def __init__(self, cash: float = 100_000.0) -> None:
        self._prices: dict[str, float] = {}
        self._seed_prices()
        # Etat de compte simule : la demo affiche des positions/equity reelles
        # (prix reels, execution et tenue de position simulees) au lieu de pages vides.
        self._cash = float(cash)
        self._positions: dict[str, Any] = {}
        self._last_prices: dict[str, float] = dict(self._prices)

    def _mark(self, asset: str, price: float) -> None:
        """Enregistre le dernier prix connu (sert au calcul du P&L latent)."""
        if price > 0:
            self._last_prices[asset] = float(price)

    def _pnl(self, position: Any) -> float:
        """P&L latent d'une position au dernier prix connu."""
        prix = self._last_prices.get(position.asset, position.avg_entry_price)
        sens = 1.0 if str(position.direction).endswith("long") else -1.0
        return (prix - position.avg_entry_price) * position.quantity * sens

    def _seed_prices(self) -> None:
        from einherjar.brokers.broker_utils import ASSET_CLASS_MAP
        defaults = {
            "EURUSD": 1.0850, "GBPUSD": 1.2650, "USDJPY": 149.50,
            "AUDUSD": 0.6520, "USDCAD": 1.3520, "USDCHF": 0.8820,
            "EURGBP": 0.8570, "NZDUSD": 0.6120,
            "BTCUSD": 67500.0, "ETHUSD": 3520.0, "ADAUSD": 0.48,
            "BCHUSD": 230.0, "LTCUSD": 72.0,
            "AAPL": 185.0, "MSFT": 420.0, "NVDA": 880.0,
            "AMZN": 180.0, "GOOGL": 175.0, "TSLA": 175.0,
            "JPM": 195.0, "XOM": 105.0,
            "SP500": 5200.0, "NASDAQ100": 18500.0,
            "DOWJONES": 39000.0, "DAX40": 18200.0,
            "XAUUSD": 2320.0, "WTIUSD": 78.5,
            "BRENT": 82.0, "COPPER": 4.35,
        }
        for asset in ASSET_CLASS_MAP.keys():
            self._prices[asset] = defaults.get(asset, 100.0)

    async def get_ohlcv(self, asset: str, timeframe: str, since: int | None = None, limit: int = 500) -> pl.DataFrame:
        # Requete d'historique OU derniere bougie : on sert les prix reels locaux si
        # l'actif en a (demo = prix reels, execution simulee). Les bougies
        # synthetiques ne sont qu'un repli pour les actifs absents du disque.
        reel = self._load_real_history(asset, timeframe, max(limit, 2))
        if reel is not None:
            return reel
        import random
        base = self._prices.get(asset, 100.0)
        # Genere 2 bougies synthetiques
        rows = []
        now = datetime.now(timezone.utc)
        for i in range(2):
            ts = int((now.timestamp() - (1 - i) * 300) * 1000)
            noise = (random.random() - 0.5) * base * 0.002
            close = base + noise
            open_p = close - (random.random() - 0.5) * base * 0.001
            high = max(open_p, close) + random.random() * base * 0.001
            low = min(open_p, close) - random.random() * base * 0.001
            rows.append([ts, open_p, high, low, close, 1000.0])
        return pl.DataFrame({
            "timestamp": [r[0] for r in rows],
            "open": [r[1] for r in rows],
            "high": [r[2] for r in rows],
            "low": [r[3] for r in rows],
            "close": [r[4] for r in rows],
            "volume": [r[5] for r in rows],
        })

    def _load_real_history(self, asset: str, timeframe: str, limit: int) -> pl.DataFrame | None:
        """Charge l'historique OHLCV reel local pour l'amorcage (mode demo).

        Les CSV bruts sont ranges par classe large (`crypto`, `forex`, `stocks`,
        `indices`, `commodities`) alors que le corpus distingue `stocks_tech`,
        `stocks_value`, `stocks_growth` : on ramene la classe a sa racine.

        Args:
            asset: Symbole.
            timeframe: Timeframe.
            limit: Nombre de bougies souhaitees.

        Returns:
            DataFrame OHLCV ou None si les donnees locales sont indisponibles.
        """
        try:
            from einherjar.brokers.broker_utils import ASSET_CLASS_MAP
            from einherjar.research.data.ohlcv import OhlcvProvider

            classe = ASSET_CLASS_MAP.get(asset)
            nom = getattr(classe, "value", None) or str(classe) if classe else None
            candidats: list[str] = []
            if nom:
                candidats.append("stocks" if nom.startswith("stocks") else nom)
            # Actifs du corpus absents de ASSET_CLASS_MAP (ex. NVDA, XOM) : on essaie
            # les classes larges du disque avant de renoncer.
            candidats += [c for c in ("crypto", "forex", "stocks", "indices", "commodities")
                          if c not in candidats]
            provider = OhlcvProvider()
            for candidat in candidats:
                try:
                    frame = provider.load(asset, timeframe, "v1", asset_class=candidat)
                except Exception:
                    continue
                return frame.df.tail(limit)
            return None
        except Exception as exc:
            logger.debug("Historique reel indisponible %s %s : %s", asset, timeframe, exc)
            return None

    async def subscribe_live(self, assets: list[str], callback: callable) -> None:
        pass

    async def place_order(self, order: Any) -> Any:
        """Execute l'ordre en simule : fill + ouverture de position (mode demo).

        Permet a la demo de montrer des positions/equity reels dans le dashboard
        (prix reels, execution et tenue de position simulees).
        """
        from einherjar.brokers.broker_utils import ASSET_CLASS_MAP
        from einherjar.core.models import Fill, Position

        prix = float(order.entry_price or self._last_prices.get(order.asset, 100.0))
        self._mark(order.asset, prix)
        classe = ASSET_CLASS_MAP.get(order.asset)
        self._positions[order.order_id] = Position(
            position_id=f"POS_{order.order_id}",
            asset=order.asset,
            direction=order.direction,
            quantity=float(order.quantity),
            avg_entry_price=prix,
            tp_price=getattr(order, "tp_price", None),
            sl_price=getattr(order, "sl_price", None),
            einher_name=getattr(order, "einher_name", "") or "",
            **({"asset_class": classe} if classe is not None else {}),
        )
        return Fill(
            fill_id=f"MOCK_FILL_{order.order_id}",
            order_id=order.order_id,
            asset=order.asset,
            filled_qty=order.quantity,
            filled_price=prix,
            fee=0.0,
        )

    async def cancel_order(self, order_id: str) -> bool:
        return True

    async def get_positions(self) -> list[Any]:
        """Positions ouvertes par la demo, P&L actualise au dernier prix connu."""
        for position in self._positions.values():
            position.unrealized_pnl = self._pnl(position)
        return list(self._positions.values())

    async def get_account(self) -> Any:
        """Compte simule : cash de reference + P&L latent des positions ouvertes."""
        from einherjar.core.models import AccountState

        pnl = sum(self._pnl(position) for position in self._positions.values())
        notionnel = sum(
            abs(position.avg_entry_price * position.quantity)
            for position in self._positions.values()
        )
        equity = self._cash + pnl
        return AccountState(
            cash=self._cash,
            equity=equity,
            margin_used=notionnel,
            margin_available=max(0.0, equity - notionnel),
            leverage=100,
        )

    def get_fees(self, asset: str) -> dict[str, Any]:
        return {"spread_pct": 0.0001, "commission_per_lot": 0.0, "swap_long": 0.0, "swap_short": 0.0}

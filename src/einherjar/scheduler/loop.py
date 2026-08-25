"""InferenceLoop — Ordonnanceur asyncio du cycle d'inference live.

Architecture du cycle (Section 1.1 CDC):
1. Detection cloture bougie (+ marge 10s)
2. Fetch derniere bougie via CTraderAdapter
3. Append incremental au LiveDataStore
4. Recalcul cible des features (FeatureEngine.compute_incremental)
5. Evaluation des Einhers (EinherEngine.evaluate)
6. Passage au Risk Manager (sizing + limites)
7. Execution via CTraderAdapter
8. Journalisation DuckDB (DataStore)

Contraintes:
- Cycle complet par actif/TF < 1 s.
- Parallelisation par actif (asyncio.gather).
- Aucune dependance reseau pendant le calcul.

Reference : Section 1.1, 1.2 du CDC EINHERJAR.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import UTC, datetime, timedelta
from typing import Any

from einherjar.brokers.adapter import BrokerAdapter
from einherjar.brokers.broker_utils import ASSET_CLASS_MAP
from einherjar.core.config import SystemConfig
from einherjar.core.confluence import ConfluenceEngine
from einherjar.core.enums import AssetClass
from einherjar.core.models import Order
from einherjar.data.live_store import LiveDataStore
from einherjar.data.store import DataStore
from einherjar.signals.einher_engine import EinherEngine
from einherjar.signals.feature_engine import FeatureEngine

logger = logging.getLogger(__name__)

MARGIN_SECONDS = 10

TF_MINUTES: dict[str, int] = {
    "5m": 5,
    "15m": 15,
    "1h": 60,
    "4h": 240,
    "1d": 1440,
}


class MarketCalendar:
    """Calendrier simplifie de marche pour l'ordonnanceur."""

    def __init__(self) -> None:
        """__init__."""
        self.tz = UTC

    def is_open(self, asset: str, dt: datetime) -> bool:
        """Verifie si le marche est ouvert pour un actif a un instant donne."""
        asset_class = ASSET_CLASS_MAP.get(asset, AssetClass.CRYPTO)

        if asset_class == AssetClass.CRYPTO:
            return True

        if asset_class in (AssetClass.FOREX, AssetClass.METAL):
            weekday = dt.weekday()
            if weekday == 5:
                return False
            if weekday == 4 and dt.hour >= 21:
                return False
            if weekday == 6 and dt.hour < 21:
                return False
            return True

        if asset_class in (AssetClass.STOCK_US, AssetClass.INDEX):
            weekday = dt.weekday()
            if weekday >= 5:
                return False
            hour_min = dt.hour * 60 + dt.minute
            open_min = 14 * 60 + 30
            close_min = 21 * 60
            return open_min <= hour_min < close_min

        return True


def next_close_timestamp(now: datetime, timeframe: str) -> datetime:
    """Calcule la prochaine cloture alignee sur le timeframe."""
    minutes = TF_MINUTES.get(timeframe, 5)

    if minutes >= 1440:
        tomorrow = now.date() + timedelta(days=1)
        return datetime(tomorrow.year, tomorrow.month, tomorrow.day, tzinfo=UTC)

    total_min = now.hour * 60 + now.minute
    next_min = ((total_min // minutes) + 1) * minutes
    next_dt = now.replace(second=0, microsecond=0)

    if next_min >= 1440:
        next_dt = next_dt + timedelta(days=1)
        next_min = next_min % 1440

    next_hour = next_min // 60
    next_minute = next_min % 60
    return next_dt.replace(hour=next_hour, minute=next_minute)


class InferenceLoop:
    """Boucle d'inference live orchestrant tout le cycle."""

    def __init__(
        self,
        broker: BrokerAdapter,
        assets_timeframes: list[tuple[str, str]],
        feature_engine: FeatureEngine,
        einher_engine: EinherEngine,
        risk_manager: Any | None,
        live_store: LiveDataStore,
        data_store: DataStore,
        config: SystemConfig | None = None,
        confluence_engine: ConfluenceEngine | None = None,
    ) -> None:
        """Initialise la boucle.

        Args:
                confluence_engine: TODO: documenter.

        Args:
            broker: Adapter cTrader unique.
            assets_timeframes: Liste de tuples (asset, timeframe).
            feature_engine: Instance FeatureEngine.
            einher_engine: Instance EinherEngine.
            risk_manager: Instance RiskManager (optionnel).
            live_store: Instance LiveDataStore.
            data_store: Instance DataStore.
            config: Configuration systeme.
        """
        self.broker = broker
        self.assets_timeframes = assets_timeframes
        self.feature_engine = feature_engine
        self.einher_engine = einher_engine
        self.risk_manager = risk_manager
        self.live_store = live_store
        self.data_store = data_store
        self.config = config or SystemConfig(
            risk_limits=__import__("einherjar.core.config", fromlist=["RiskLimits"]).RiskLimits(),
            validation_config=__import__("einherjar.core.config", fromlist=["ValidationConfig"]).ValidationConfig(),
        )
        self.confluence_engine = confluence_engine or ConfluenceEngine()
        self.calendar = MarketCalendar()
        self.running = False
        self._tasks: set[asyncio.Task] = set()

    async def _fetch_last_candle(
        self, asset: str, timeframe: str
    ) -> dict[str, Any] | None:
        """Fetch la derniere bougie cloturee via cTrader."""
        try:
            df = await self.broker.get_ohlcv(asset, timeframe, limit=2)
            if len(df) == 0:
                return None
            row = df.to_dicts()[-1]
            return row
        except Exception as exc:
            logger.warning("Fetch bougie echoue %s %s: %s", asset, timeframe, exc)
            return None

    async def _process_asset_tf(
        self, asset: str, timeframe: str
    ) -> dict[str, Any]:
        """Execute un cycle complet d'inference pour un (asset, tf)."""
        result: dict[str, Any] = {
            "asset": asset,
            "tf": timeframe,
            "signals_count": 0,
            "forming_count": 0,
            "orders_count": 0,
            "signals": [],
            "error": None,
        }

        try:
            candle = await self._fetch_last_candle(asset, timeframe)
            if candle is None:
                result["error"] = "fetch_none"
                return result

            last_ts = self.live_store.get_last_timestamp(asset, timeframe)
            candle_ts = candle.get("timestamp")
            if isinstance(candle_ts, datetime) and last_ts == candle_ts:
                result["error"] = "already_known"
                return result

            df_history = self.live_store.get_window(asset, timeframe, n=self.feature_engine.max_lookback)
            self.live_store.append(asset, timeframe, candle)

            df_enriched = self.feature_engine.compute_incremental(df_history, candle)
            if "feature_placeholder" not in df_enriched.columns or len(df_enriched.columns) > 6:
                last_row = df_enriched.to_dicts()[-1]
                self.live_store.append(asset, timeframe, last_row)

            from einherjar.core.enums import TimeFrame as TFEnum

            tf_enum = TFEnum(timeframe)
            signals, forming = self.einher_engine.evaluate(df_enriched, asset, tf_enum)
            result["signals_count"] = len(signals)
            result["forming_count"] = len(forming)

            for sig in signals:
                self.data_store.append_signal(sig)
                logger.info(
                    "SIGNAL %s %s %s %s TP=%.4f SL=%.4f",
                    sig.asset,
                    sig.timeframe.value,
                    sig.einher_name,
                    sig.direction.value,
                    sig.tp_price,
                    sig.sl_price,
                )

            result["signals"] = signals

        except Exception as exc:
            logger.exception("Cycle inference echoue %s %s", asset, timeframe)
            result["error"] = str(exc)

        return result

    async def _run_cycle(self, now: datetime) -> None:
        """Execute un cycle d'inference pour tous les (asset, tf)."""
        tasks = []
        for asset, tf in self.assets_timeframes:
            if not self.calendar.is_open(asset, now):
                continue
            tasks.append(self._process_asset_tf(asset, tf))

        if not tasks:
            return

        results = await asyncio.gather(*tasks, return_exceptions=True)
        total_signals = sum(r.get("signals_count", 0) for r in results if isinstance(r, dict))
        total_orders = 0
        raw_signals = [
            signal
            for result in results
            if isinstance(result, dict)
            for signal in result.get("signals", [])
        ]
        if self.data_store.kill_switch_enabled():
            logger.warning("Kill switch actif: aucune nouvelle execution")
        elif self.risk_manager is not None and raw_signals:
            account = await self.broker.get_account()
            positions = await self.broker.get_positions()
            for cluster in self.confluence_engine.aggregate(raw_signals):
                signal = cluster.to_signal()
                order_or_rejection = self.risk_manager.evaluate(signal, account, positions)
                if isinstance(order_or_rejection, Order):
                    self.data_store.append_order(order_or_rejection)
                    try:
                        fill = await self.broker.place_order(order_or_rejection)
                        self.data_store.append_fill(fill)
                        total_orders += 1
                        logger.info(
        "EXEC %s %s contributors=%d", fill.asset, fill.order_id, len(cluster.contributing_einhers)
    )
                    except Exception as exc:
                        logger.error("Execution ordre echoue %s: %s", order_or_rejection.order_id, exc)
                else:
                    self.data_store.append_rejection(order_or_rejection)
                    logger.info("REJECT %s: %s", signal.asset, order_or_rejection.reason)
        errors = sum(1 for r in results if isinstance(r, Exception))
        logger.info(
            "Cycle %s | assets=%d | signals=%d | orders=%d | errors=%d",
            now.isoformat(),
            len(tasks),
            total_signals,
            total_orders,
            errors,
        )

    async def _sleep_until_next_close(self) -> datetime:
        """Calcule et attend la prochaine cloture la plus proche."""
        now = datetime.now(UTC)
        next_closes: list[datetime] = []

        for asset, tf in self.assets_timeframes:
            if not self.calendar.is_open(asset, now):
                continue
            nc = next_close_timestamp(now, tf)
            next_closes.append(nc)

        if not next_closes:
            await asyncio.sleep(60)
            return await self._sleep_until_next_close()

        closest = min(next_closes)
        wake_at = closest + timedelta(seconds=MARGIN_SECONDS)
        sleep_sec = (wake_at - now).total_seconds()

        if sleep_sec > 0:
            logger.debug("Prochain reveil a %s (dans %.1f s)", wake_at.isoformat(), sleep_sec)
            await asyncio.sleep(sleep_sec)

        return wake_at

    async def run(self) -> None:
        """Boucle principale d'inference."""
        self.running = True
        logger.info("InferenceLoop demarre avec %d actifs/TF", len(self.assets_timeframes))

        while self.running:
            try:
                wake_at = await self._sleep_until_next_close()
                if not self.running:
                    break
                await self._run_cycle(wake_at)
            except asyncio.CancelledError:
                logger.info("InferenceLoop annule")
                break
            except Exception as exc:
                logger.exception("Erreur dans la boucle principale: %s", exc)
                await asyncio.sleep(5)

        logger.info("InferenceLoop arrete")

    def stop(self) -> None:
        """Demande l'arret de la boucle."""
        self.running = False
        for task in self._tasks:
            task.cancel()
        self._tasks.clear()

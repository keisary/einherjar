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
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from einherjar.brokers.adapter import BrokerAdapter
from einherjar.brokers.broker_utils import ASSET_CLASS_MAP
from einherjar.core.config import SystemConfig
from einherjar.core.confluence import ConfluenceEngine
from einherjar.core.enums import AssetClass
from einherjar.core.models import Order
from einherjar.data.live_store import LiveDataStore
from einherjar.data.store import DataStore
from einherjar.risk.exits import doit_fermer
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
        self._cycles = 0
        self._demarre_le = datetime.now(UTC).isoformat()
        self._tasks: set[asyncio.Task] = set()
        # Features reellement requises par couple (calcul cible) : sans cela chaque
        # cycle paierait les 107 patterns et les 76 colonnes de facteurs dont
        # l'evaluation des einhers n'a pas besoin.
        self._needed: dict[tuple[str, str], set[str]] = {}
        corpus_path = Path(__file__).resolve().parents[3] / "outputs" / "corpus.jsonl"
        if corpus_path.exists():
            from einherjar.signals.corpus_bridge import required_features_by_universe

            self._needed = required_features_by_universe(corpus_path)
            logger.info("Calcul cible : %d couples avec besoin de features connu", len(self._needed))

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

            df_enriched = self.feature_engine.compute_incremental(
                df_history,
                candle,
                asset=asset,
                timeframe=timeframe,
                needed=self._needed.get((asset, timeframe)),
            )
            # Le store live ne contient QUE l'OHLCV (une seule bougie ajoutee plus
            # haut) : y ecrire la ligne enrichie melangerait 271 colonnes de features
            # a un schema OHLCV et ferait echouer le prochain concat. Les features sont
            # recalculees a chaque cycle, les signaux sont persistes par le DataStore.

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
        # Sorties : le broker gere les TP/SL (transmis a l'ouverture), EINHERJAR
        # ferme les positions dont la duree de tenue maximale est depassee.
        closed = await self._gerer_sorties(now)

        # Etat publie pour l'API et le dashboard : sans cela, l'etat de la boucle
        # n'est observable qu'en lisant les logs (le dashboard affichait un etat fige).
        self._cycles += 1
        self.data_store.set_state(
            "loop",
            {
                "running": True,
                "cycles": self._cycles,
                "lastCycleAt": now.isoformat(),
                "assets": len(tasks),
                "signals": total_signals,
                "orders": total_orders,
                "closed": closed,
                "errors": errors,
                "startedAt": self._demarre_le,
            },
        )
        logger.info(
            "Cycle %s | assets=%d | signals=%d | orders=%d | closed=%d | errors=%d",
            now.isoformat(),
            len(tasks),
            total_signals,
            total_orders,
            closed,
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

    async def bootstrap_history(self, limit: int | None = None) -> dict[str, int]:
        """Amorce le LiveDataStore avec l'historique broker avant le premier cycle.

        Sans amorcage, le premier cycle ne dispose que de la derniere bougie : toutes
        les features a fenetre (RSI, EMA, patterns, quant) sont NaN et aucun einher ne
        peut se declencher — la boucle tournerait a vide.

        Args:
            limit: Nombre de bougies demandees par (asset, tf).
                Defaut : `max_lookback` du pipeline de features.

        Returns:
            Dict {"ASSET|tf": bougies ajoutees}.
        """
        objectif = limit or getattr(self.feature_engine, "max_lookback", 500)
        ajouts: dict[str, int] = {}
        now = datetime.now(UTC)
        for asset, timeframe in self.assets_timeframes:
            cle = f"{asset}|{timeframe}"
            if not self.calendar.is_open(asset, now):
                ajouts[cle] = 0
                continue
            deja = self.live_store.get_window(asset, timeframe)
            if len(deja) >= objectif:
                ajouts[cle] = 0
                continue
            try:
                df = await self.broker.get_ohlcv(asset, timeframe, limit=objectif)
            except Exception as exc:
                logger.warning("Amorcage %s echoue : %s", cle, exc)
                ajouts[cle] = 0
                continue
            if df is None or len(df) == 0:
                logger.warning("Amorcage %s : le broker n'a renvoye aucune bougie", cle)
                ajouts[cle] = 0
                continue
            rows = df.tail(objectif).to_dicts()
            self.live_store.bulk_append(asset, timeframe, rows)
            ajouts[cle] = len(rows)
            if len(rows) < 200:
                logger.warning(
                    "Amorcage %s insuffisant (%d bougies) : les features a fenetre "
                    "resteront NaN sur les premieres bougies",
                    cle,
                    len(rows),
                )
        total = sum(ajouts.values())
        logger.info("Amorcage historique : %d couples, %d bougies chargees", len(ajouts), total)
        return ajouts

    def warmup_features(self, rows: int = 250) -> dict[str, float]:
        """Compile le code Numba a l'avance (un profil de besoin a la fois).

        Mesure : un cycle a froid coute ~85 s (compilation JIT des enrichisseurs)
        alors qu'un cycle a chaud coute ~0,5 s. On paie donc le JIT explicitement au
        demarrage, une fois par profil de features, au lieu de le payer au milieu
        d'une fenetre de marche.

        Args:
            rows: Taille de la frame d'echauffement (le cout est la compilation,
                pas le nombre de lignes).

        Returns:
            Dict {profil: duree_s}.
        """
        import numpy as np
        import polars as pl

        profils: dict[str, set[str]] = {}
        for refs in (self._needed.values() or [set()]):
            cle = f"patterns={any(r.startswith('pattern_') for r in refs)}|" \
                  f"quant={any(r.startswith('quant_') for r in refs)}|" \
                  f"facteurs={any(r.startswith('Factor_') or r.endswith(('_signal', '_norm')) for r in refs)}"
            profils.setdefault(cle, set()).update(refs)
        if not profils:
            profils["complet"] = set()

        # Frame d'echauffement : OHLCV reel si disponible, sinon synthetique
        frame = None
        for asset, timeframe in self.assets_timeframes:
            fenetre = self.live_store.get_window(asset, timeframe)
            if fenetre.height >= 50:
                frame = fenetre.tail(rows).select(["timestamp", "open", "high", "low", "close", "volume"])
                break
        if frame is None:
            rng = np.random.default_rng(0)
            close = np.cumsum(rng.normal(0, 1, rows)) + 100.0
            frame = pl.DataFrame({
                "timestamp": pl.datetime_range(
                    datetime(2024, 1, 1), datetime(2025, 1, 1), interval="1h", eager=True
                ).head(rows),
                "open": close, "high": close + 1.0, "low": close - 1.0, "close": close,
                "volume": np.full(rows, 1000.0),
            })

        from einherjar.signals.feature_pipeline import FeaturePipeline

        durees: dict[str, float] = {}
        manquantes_globales: set[str] = set()
        for cle, refs in profils.items():
            debut = time.perf_counter()
            try:
                enrichi = self.feature_engine.compute(
                    frame, asset="WARMUP", timeframe="1h",
                    needed=(refs or None),
                )
            except Exception as exc:
                logger.warning("Echauffement %s echoue : %s", cle, exc)
                durees[cle] = -1.0
                continue
            durees[cle] = round(time.perf_counter() - debut, 2)
            # Le calcul vient d'etre fait : on verifie GRATUITEMENT que chaque
            # feature_ref du corpus est bien produite. Un calcul cible qui oublie une
            # dependance laisse la colonne absente, l'evaluation de l'einher leve une
            # exception avalee et l'einher devient muet sans aucune alerte.
            if refs:
                absentes = FeaturePipeline.missing_refs(enrichi, sorted(refs))
                if absentes:
                    manquantes_globales.update(absentes)
                    logger.error(
                        "Couverture du profil %s : %d feature(s) du corpus NON produites : %s",
                        cle,
                        len(absentes),
                        ", ".join(absentes),
                    )
        self._features_manquantes = manquantes_globales
        logger.info("Echauffement features (JIT) : %s", durees)
        return durees

    async def warmup_features_async(self, rows: int = 250) -> dict[str, float]:
        """Echauffement JIT hors boucle d'evenements (l'API reste disponible).

        Args:
            rows: Taille de la frame d'echauffement.

        Returns:
            Dict {profil: duree_s}.
        """
        # `warmup_features` est SYNCHRONE : appelee telle quelle dans la boucle
        # d'evenements, elle gelait l'API pendant tout l'echauffement (~90 s).
        return await asyncio.to_thread(self.warmup_features, rows)

    def appliquer_couverture_features(self) -> dict[str, list[str]]:
        """Ecartee les couples dont une feature du corpus n'est pas produite.

        La liste des features manquantes vient de l'echauffement (ou chaque profil de
        besoin est reellement calcule) : aucun calcul supplementaire n'est fait ici.
        Un couple dont une `feature_ref` manque est retire de la boucle, car ses
        einhers seraient muets pour toujours (`_eval_condition` rend False en
        silence sur une colonne absente).

        Returns:
            Dict {"ASSET|tf": [refs manquantes]} des couples ecartes.
        """
        manquantes = getattr(self, "_features_manquantes", set())
        if not manquantes:
            self.data_store.set_state(
                "couverture",
                {"couples_verifies": len(self.assets_timeframes), "couples_ecartes": 0, "detail": {}},
            )
            logger.info(
                "Couverture des features : %d couples, aucune feature du corpus manquante",
                len(self.assets_timeframes),
            )
            return {}

        incomplets: dict[str, list[str]] = {}
        gardes: list[tuple[str, str]] = []
        for asset, timeframe in self.assets_timeframes:
            besoin = self._needed.get((asset, timeframe)) or set()
            fautives = sorted(besoin & manquantes)
            if fautives:
                incomplets[f"{asset}|{timeframe}"] = fautives
            else:
                gardes.append((asset, timeframe))

        self.assets_timeframes = gardes
        for cle, fautives in sorted(incomplets.items()):
            logger.error(
                "Couple %s ecarte : feature(s) du corpus non produites -> %s "
                "(einhers muets sinon)",
                cle,
                ", ".join(fautives),
            )
        logger.warning(
            "Couverture des features : %d couples conserves, %d ecarte(s)",
            len(gardes),
            len(incomplets),
        )
        self.data_store.set_state(
            "couverture",
            {
                "couples_verifies": len(gardes) + len(incomplets),
                "couples_ecartes": len(incomplets),
                "detail": incomplets,
            },
        )
        return incomplets

    async def run(self) -> None:
        """Boucle principale d'inference."""
        self.running = True
        logger.info("InferenceLoop demarre avec %d actifs/TF", len(self.assets_timeframes))
        try:
            await self.bootstrap_history()
        except Exception as exc:  # l'amorcage ne doit jamais empecher la boucle de tourner
            logger.exception("Amorcage historique echoue : %s", exc)
        try:
            await self.warmup_features_async()
        except Exception as exc:
            logger.exception("Echauffement des features echoue : %s", exc)
        try:
            self.appliquer_couverture_features()
        except Exception as exc:  # noqa: BLE001 - ne doit jamais bloquer la boucle
            logger.exception("Application de la couverture des features echouee : %s", exc)

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

    async def _gerer_sorties(self, maintenant: datetime | None = None) -> int:
        """Ferme les positions dont la duree de tenue maximale est depassee.

        Les TP/SL sont transmis au broker a l'ouverture (il gere les sorties de
        prix) ; cette passe applique la regle de duree du corpus
        (`Einher.max_holding`, issue de `amplitude_bars`), sans laquelle une
        position qui n'atteint ni TP ni SL resterait ouverte indefiniment.

        Args:
            maintenant: Instant de reference (defaut : maintenant UTC).

        Returns:
            Nombre de positions fermees.
        """
        try:
            positions = await self.broker.get_positions()
        except Exception as exc:  # noqa: BLE001 - une lecture ratee ne tue pas le cycle
            logger.warning("Sorties: positions illisibles (%s)", exc)
            return 0
        if not positions:
            return 0

        index = {einher.name: einher for einher in self.einher_engine.einhers}
        fermees = 0
        for position in positions:
            nom = getattr(position, "einher_name", None) or ""
            motif = doit_fermer(position, index.get(nom), maintenant)
            if motif is None:
                continue

            identifiant: str | int | None = getattr(position, "position_id", None)
            if isinstance(identifiant, str) and identifiant.isdigit():
                identifiant = int(identifiant)  # cTrader attend un positionId entier
            if identifiant is None:
                continue
            try:
                if await self.broker.close_position(identifiant):
                    fermees += 1
                    self.data_store.remove_position(str(getattr(position, "position_id", "")))
                    logger.info(
                        "CLOSE %s %s (%s, einher=%s)",
                        getattr(position, "asset", "?"),
                        identifiant,
                        motif,
                        nom,
                    )
                else:
                    logger.warning("CLOSE refuse par le broker : %s", identifiant)
            except Exception as exc:  # noqa: BLE001 - une fermeture ratee ne tue pas le cycle
                logger.error("CLOSE echoue %s: %s", identifiant, exc)
        return fermees

    def stop(self) -> None:
        """Demande l'arret de la boucle et force l'ecriture des donnees en attente."""
        self.running = False
        for task in self._tasks:
            task.cancel()
        self._tasks.clear()
        # Les ecritures disque sont throttlees : on vide le tampon a l'arret pour
        # ne pas perdre les bougies accumulees depuis la derniere sauvegarde.
        try:
            ecrits = self.live_store.flush()
            if ecrits:
                logger.info("LiveStore: %d fichiers ecrits a l'arret", ecrits)
        except Exception as exc:  # noqa: BLE001 - l'arret ne doit jamais lever
            logger.warning("LiveStore: flush a l'arret echoue (%s)", exc)

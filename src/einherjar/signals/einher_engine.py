"""EinherEngine — Evaluation des strategies sur features enrichies.

Cycle d'inference (Section 1.1 CDC):
1. Cloture detectee -> 2. Append bougie -> 3. Recalcul cible -> 4. Evaluation Einhers
5. Formation signaux -> 6. Confluence -> 7. Risk Manager -> 8. Execution

Cette classe couvre les etapes 4 et 5 : evaluation des Einhers et formation des signaux.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import polars as pl

from einherjar.core.enums import Direction, EinherState, TimeFrame
from einherjar.core.models import Einher, Signal
from einherjar.signals.midas_bridge import PatternBridge

logger = logging.getLogger(__name__)


class EinherEngine:
    """Moteur d'evaluation des Einhers sur un DataFrame enrichi.

    Attributs:
        einhers: Liste des Einhers actifs.
        _feature_map: Cache du mapping nom corpus -> nom MIDAS.
    """

    def __init__(self, einhers: list[Einher] | None = None) -> None:
        """__init__.

        Args:
            einhers: TODO document.
        """
        self.einhers = einhers or []
        self._feature_map: dict[str, str] = {}

    def load_corpus(self, path: str) -> None:
        """Charge un corpus depuis un fichier JSON.

        Le corpus JSON contient soit une liste directe d'Einhers,
        soit un dict avec une cle 'einhers'. Chaque element peut
        avoir les champs du dataclass en ligne ou imbriques sous
        'definition'.
        """
        import json
        with open(path, encoding="utf-8") as f:
            raw = json.load(f)
        items = raw.get("einhers", raw) if isinstance(raw, dict) else raw
        self.einhers = []
        for item in items:
            # Extraire la definition si elle est imbriquee
            data = item.get("definition", item)
            try:
                self.einhers.append(Einher.from_dict(data))
            except Exception as exc:
                logger.warning("Corpus item ignore: %s", exc)

    def evaluate(
        self,
        df: pl.DataFrame,
        asset: str,
        timeframe: TimeFrame,
    ) -> tuple[list[Signal], list[Einher]]:
        """Evalue tous les Einhers sur le DataFrame enrichi.

        Args:
            df: DataFrame OHLCV + features.
            asset: Symbole evalue.
            timeframe: Timeframe courant.

        Returns:
            Tuple (signaux_triggered, einhers_forming).
        """
        signals: list[Signal] = []
        forming: list[Einher] = []

        for einher in self.einhers:
            if einher.direction not in (Direction.LONG.value, Direction.SHORT.value):
                logger.warning("Einher ignore: direction invalide %s pour %s", einher.direction, einher.name)
                continue
            if timeframe.value not in einher.timeframes:
                continue
            if not self._asset_matches(einher, asset):
                continue

            result = self._evaluate_einher(einher, df, asset, timeframe)

            if result["triggered"]:
                signal = self._build_signal(einher, df, asset, timeframe, result)
                signals.append(signal)
                einher.state = EinherState.TRIGGERED
            elif result["forming"]:
                einher.state = EinherState.FORMING
                forming.append(einher)
            else:
                einher.state = EinherState.IDLE

        return signals, forming

    def _asset_matches(self, einher: Einher, asset: str) -> bool:
        """Verifie si un Einher couvre l'actif donne."""
        if einher.assets == "all":
            return True
        if isinstance(einher.assets, list):
            return asset in einher.assets or "all" in einher.assets
        return asset in str(einher.assets).split(",")

    def _evaluate_einher(
        self,
        einher: Einher,
        df: pl.DataFrame,
        asset: str,
        timeframe: TimeFrame,
    ) -> dict[str, Any]:
        """Evalue un seul Einher sur le DataFrame.

        Returns:
            Dict {triggered: bool, forming: bool, context: dict}.
        """
        if len(df) == 0:
            return {"triggered": False, "forming": False, "context": {}}

        # Evaluation du trigger sur la derniere ligne
        trigger_hit = self._eval_condition(df, einher.trigger)

        # Evaluation des filtres (AND)
        n_filters = len(einher.filters)
        n_filters_pass = 0
        filters_pass = True

        for filt in einher.filters:
            cond = filt.get("expr", "") if isinstance(filt, dict) else str(filt)
            if cond:
                ok = self._eval_condition(df, cond)
                if ok:
                    n_filters_pass += 1
                else:
                    filters_pass = False

        # Forming : trigger OK mais pas tous les filtres, OU trigger partiel
        forming = False
        if trigger_hit and n_filters > 0 and n_filters_pass > 0 and not filters_pass:
            forming = True
        elif not trigger_hit and n_filters_pass >= max(1, n_filters // 2):
            # Conditions partiellement remplies sans trigger
            forming = True

        triggered = trigger_hit and filters_pass

        # Contexte pour le signal
        last_close = float(df["close"][-1]) if len(df) > 0 else 0.0
        last_atr = 0.0
        if "atr_14" in df.columns and len(df) > 0:
            last_atr = float(df["atr_14"][-1])

        context = {
            "trigger_condition": einher.trigger,
            "filters_conditions": einher.filters,
            "last_close": last_close,
            "last_atr": last_atr,
            "filters_passed": n_filters_pass,
            "filters_total": n_filters,
            "domain": einher.domain,
            "structure": PatternBridge.structure_levels(df, einher.direction),
        }

        return {
            "triggered": triggered,
            "forming": forming,
            "context": context,
        }

    def _eval_condition(self, df: pl.DataFrame, condition: str) -> bool:
        """Evalue une condition sur la derniere ligne du DataFrame.

        Args:
            df: DataFrame avec les features.
            condition: String de condition, ex: "pattern_rectangle == 1",
                "quant_permutation_entropy < 0.3", "di_minus > di_plus",
                "bb_percent < 0.05 AND close < bb_middle".

        Returns:
            True si la condition est vraie sur la derniere ligne.
        """
        if not condition:
            return True

        # Normaliser les operateurs logiques vers Python
        expr = (
            condition
            .replace(" AND ", " and ")
            .replace(" OR ", " or ")
            .replace("&&", " and ")
            .replace("||", " or ")
        )

        # Construire le dictionnaire local avec les valeurs de la derniere ligne
        local: dict[str, float | int] = {}
        for col in df.columns:
            try:
                val = df[col][-1]
                if hasattr(val, "item"):
                    local[col] = val.item()
                else:
                    local[col] = float(val) if isinstance(val, int | float | complex) else 0.0  # pyright: ignore[reportArgumentType]
            except Exception:
                local[col] = 0.0

        try:
            result = eval(expr, {"__builtins__": {}}, local)
            return bool(result)
        except Exception as exc:
            logger.debug("Eval condition echoue: %s | expr=%s", exc, expr)
            return False

    def _map_condition(self, condition: str, columns: list[str]) -> str:
        """Mappe les noms de features du corpus vers les noms MIDAS.

        Le corpus utilise deja les noms corrects (pattern_, quant_, indicator_),
        mais cette methode conserve le mapping legacy col_ -> pattern_.
        """
        if condition.startswith("col_"):
            for col in columns:
                corpus_name = condition.split()[0]
                midas_name = corpus_name.replace("col_", "pattern_")
                if midas_name in columns:
                    return condition.replace(corpus_name, midas_name)
        return condition

    def _eval_numpy_fallback(self, df: pl.DataFrame, condition: str) -> bool:
        """Fallback d'evaluation via numpy si polars echoue."""
        try:
            local_dict = {c: df[c].to_numpy() for c in df.columns}
            safe_dict = {"__builtins__": None, "np": np, "True": True, "False": False}
            safe_dict.update(local_dict)
            result = eval(condition, {"__builtins__": {}}, safe_dict)
            if isinstance(result, np.ndarray):
                return bool(result[-1])
            return bool(result)
        except Exception:
            return False

    def _build_signal(
        self,
        einher: Einher,
        df: pl.DataFrame,
        asset: str,
        timeframe: TimeFrame,
        result: dict[str, Any],
    ) -> Signal:
        """Construit un Signal a partir d'un Einher declenche."""
        last_close = float(df["close"][-1])
        last_atr = float(df["atr_14"][-1]) if "atr_14" in df.columns else last_close * 0.01

        tp_price = self._calculate_tp(einher, last_close, last_atr, result["context"])
        sl_price = self._calculate_sl(einher, last_close, last_atr, result["context"])

        direction = Direction(einher.direction)

        # Confiance basee sur le nombre de filtres passes
        n_total = result["context"].get("filters_total", 0)
        n_pass = result["context"].get("filters_passed", n_total)
        confidence = 0.5 + (0.5 * n_pass / max(n_total, 1))

        return Signal(
            asset=asset,
            direction=direction,
            timeframe=timeframe,
            einher_name=einher.name,
            entry_price=last_close,
            tp_price=tp_price,
            sl_price=sl_price,
            confidence=round(min(confidence, 1.0), 2),
            context=result["context"],
        )

    def _calculate_tp(
        self, einher: Einher, entry: float, atr: float, context: dict[str, Any]
    ) -> float:
        """Calcule le prix de take-profit."""
        tp_rule = einher.tp_rule
        if not tp_rule:
            mult = 2.5
            return entry + (mult * atr) if einher.direction != "short" else entry - (mult * atr)

        rule_type = tp_rule.get("type", "atr_multiple")
        if rule_type == "atr_multiple":
            mult = tp_rule.get("value", 2.5)
            return entry + (mult * atr) if einher.direction != "short" else entry - (mult * atr)
        elif rule_type == "pattern_height":
            height = float(tp_rule.get("value") or context["structure"]["pattern_height"])
            return entry + height if einher.direction != "short" else entry - height
        elif rule_type == "fibonacci_ratio":
            level = float(tp_rule.get("value", 0.618))
            height = float(context["structure"]["pattern_height"])
            distance = height * level
            return entry + distance if einher.direction != "short" else entry - distance
        elif rule_type == "mfe_calibrated":
            distance = entry * float(tp_rule.get("value", 0.0))
            return entry + distance if einher.direction != "short" else entry - distance
        else:
            return entry + (2.5 * atr)

    def _calculate_sl(
        self, einher: Einher, entry: float, atr: float, context: dict[str, Any]
    ) -> float:
        """Calcule le prix de stop-loss."""
        sl_rule = einher.sl_rule
        if not sl_rule:
            return entry - (1.5 * atr) if einher.direction != "short" else entry + (1.5 * atr)

        rule_type = sl_rule.get("type", "atr_multiple")
        if rule_type == "atr_multiple":
            mult = sl_rule.get("value", 1.5)
            return entry - (mult * atr) if einher.direction != "short" else entry + (mult * atr)
        elif rule_type == "beyond_structure":
            level = sl_rule.get("value")
            if level is None:
                return float(context["structure"]["invalidation"])
            return float(level)
        elif rule_type == "mae_calibrated":
            distance = entry * float(sl_rule.get("value", 0.0))
            return entry - distance if einher.direction != "short" else entry + distance
        else:
            return entry - (1.5 * atr)

"""
Moteur de calibration backtest pour EINHERJAR.

Principe :
  1. Groupe les Einhers par (asset, tf) pour charger les donnees une seule fois.
  2. Pour chaque groupe, evalue les conditions trigger + filters sur tout le dataset.
  3. Simule les trades avec SL/TP natif ou fallback ATR.
  4. Collecte les metriques de performance.

Contrainte RAM : un seul (asset, tf) en memoire a la fois.
"""

import json
import numpy as np
import polars as pl
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field

from backtest.data_source import load_ohlcv, map_feature_name
from backtest import metrics

EINHERJAR_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_CONFIG = {
    "slippage_pct": 0.05,
    "fallback_tp_atr_mult": 2.5,
    "fallback_sl_atr_mult": 1.5,
    "max_holding_bars": {"5m": 36, "15m": 24, "1h": 12, "4h": 8, "1d": 5},
    "min_trades_for_metrics": 10,
}


TF_MINUTES = {"5m": 5, "15m": 15, "1h": 60, "4h": 240, "1d": 1440}


def _parse_max_holding(value, tf: str) -> int:
    """Convertit une duree texte en nombre de barres selon le TF."""
    if isinstance(value, (int, float)):
        return int(value)
    bar_min = TF_MINUTES.get(tf, 60)
    val_str = str(value).strip().lower()
    if val_str.endswith("d"):
        total_min = int(val_str[:-1]) * 1440
    elif val_str.endswith("h"):
        total_min = int(val_str[:-1]) * 60
    elif val_str.endswith("m"):
        total_min = int(val_str[:-1])
    else:
        try:
            return int(val_str)
        except ValueError:
            return 12
    return max(1, total_min // bar_min)


def _load_fees(asset_class: str) -> dict:
    fee_file = EINHERJAR_ROOT / "config" / f"fees_{asset_class}.json"
    if not fee_file.exists():
        return {"maker": 0.0, "taker": 0.001}
    with open(fee_file, "r", encoding="utf-8") as f:
        return json.load(f)


def _get_exit_multipliers(pattern_name: str, native_exits: dict) -> tuple[float, float]:
    key = pattern_name.replace("pattern_", "").replace("col_", "")
    harmonic_map = {
        "gartley_bull": "harmonic_gartley", "gartley_bear": "harmonic_gartley",
        "butterfly_bull": "harmonic_butterfly", "butterfly_bear": "harmonic_butterfly",
        "bat_bull": "harmonic_bat", "bat_bear": "harmonic_bat",
        "crab_bull": "harmonic_crab", "crab_bear": "harmonic_crab",
    }
    lookup = harmonic_map.get(key, key)
    rule = native_exits.get(lookup, native_exits.get("default"))

    tp_mult = rule.get("tp_atr_mult", 2.5)
    sl_mult = rule.get("sl_atr_mult", 1.5)

    if "tp_atr_mult" not in rule:
        if rule.get("tp_type") == "pattern_height":
            tp_mult = 2.5
        elif rule.get("tp_type") == "fibonacci_ratio":
            tp_mult = 1.5
        elif rule.get("tp_type") == "atr_multiple":
            tp_mult = 2.0
        else:
            tp_mult = 2.5
    if "sl_atr_mult" not in rule:
        sl_mult = 1.5
    return float(tp_mult), float(sl_mult)


def _eval_expression(df: pl.DataFrame, expr: str) -> np.ndarray:
    words = set(expr.replace("(", " ").replace(")", " ").replace("&", " ")
                .replace("|", " ").replace("==", " ").replace("!=", " ")
                .replace("<", " ").replace(">", " ").replace("<=", " ")
                .replace(">=", " ").split())
    mapped_expr = expr
    for w in words:
        mapped = map_feature_name(w)
        if mapped in df.columns:
            mapped_expr = mapped_expr.replace(w, mapped)
    try:
        result = df.select(pl.sql_expr(mapped_expr)).to_numpy().flatten()
    except Exception:
        result = _eval_expression_numpy(df, mapped_expr)
    return result.astype(bool)


def _eval_expression_numpy(df: pl.DataFrame, expr: str) -> np.ndarray:
    local_dict = {c: df[c].to_numpy() for c in df.columns}
    safe_dict = {"__builtins__": None}
    safe_dict.update(local_dict)
    safe_dict.update({"np": np, "abs": abs, "max": max, "min": min, "True": True, "False": False})
    try:
        result = eval(expr, {"__builtins__": {}}, safe_dict)
        if isinstance(result, np.ndarray):
            return result
        return np.full(len(df), bool(result))
    except Exception:
        return np.zeros(len(df), dtype=bool)


@dataclass
class EinherBacktestResult:
    einher_id: str
    einher_name: str
    asset: str
    timeframe: str
    direction: str
    n_trades: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    max_drawdown: float = 0.0
    avg_trade: float = 0.0
    expectancy: float = 0.0
    total_return: float = 0.0
    trades_per_month: float = 0.0
    returns: list = field(default_factory=list)
    equity: list = field(default_factory=list)
    entry_timestamps: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "einher_id": self.einher_id, "einher_name": self.einher_name,
            "asset": self.asset, "timeframe": self.timeframe, "direction": self.direction,
            "n_trades": self.n_trades, "win_rate": round(self.win_rate, 4),
            "profit_factor": round(self.profit_factor, 4),
            "sharpe_ratio": round(self.sharpe_ratio, 4),
            "sortino_ratio": round(self.sortino_ratio, 4),
            "max_drawdown": round(self.max_drawdown, 4),
            "avg_trade": round(self.avg_trade, 6),
            "expectancy": round(self.expectancy, 6),
            "total_return": round(self.total_return, 6),
            "trades_per_month": round(self.trades_per_month, 2),
        }


class Calibrator:
    def __init__(self, corpus_path: Path, native_exits_path: Path, config: Optional[dict] = None):
        with open(corpus_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
            self.corpus = raw.get("einhers", raw)
        with open(native_exits_path, "r", encoding="utf-8") as f:
            self.native_exits = json.load(f)
        self.config = {**DEFAULT_CONFIG, **(config or {})}

    def _group_by_asset_tf(self) -> dict:
        assets_path = EINHERJAR_ROOT / "config" / "assets_v1.json"
        all_assets = []
        if assets_path.exists():
            with open(assets_path, "r", encoding="utf-8") as f:
                all_assets = json.load(f).get("assets", [])

        groups: dict = {}
        for einher in self.corpus:
            assets = einher.get("assets", [])
            tfs = einher.get("timeframes", [])

            if assets == "all" or (isinstance(assets, list) and len(assets) == 1 and assets[0] == "all"):
                asset_list = all_assets
            else:
                asset_list = []
                for a in assets:
                    if a == "all":
                        asset_list.extend(all_assets)
                    elif isinstance(a, dict):
                        asset_list.append(a)
                    else:
                        asset_list.append({"asset": a, "class": "crypto"})

            for asset_info in asset_list:
                asset = asset_info.get("asset") if isinstance(asset_info, dict) else asset_info
                asset_class = asset_info.get("class", "crypto") if isinstance(asset_info, dict) else "crypto"
                for tf in tfs:
                    key = (asset, tf, asset_class)
                    groups.setdefault(key, []).append(einher)
        return groups

    def _simulate_einher(self, einher: dict, df: pl.DataFrame, asset: str, tf: str, asset_class: str):
        direction = einher.get("direction", "long")
        trigger_expr = einher.get("trigger", "")
        filters = einher.get("filters", [])
        max_holding_raw = einher.get("max_holding", self.config["max_holding_bars"].get(tf, 12))
        max_holding = _parse_max_holding(max_holding_raw, tf)

        if not trigger_expr:
            return None

        trigger_mask = _eval_expression(df, trigger_expr)
        if not np.any(trigger_mask):
            return None

        combined = trigger_mask.copy()
        for filt in filters:
            f_expr = filt.get("expr", "") if isinstance(filt, dict) else str(filt)
            if f_expr:
                combined &= _eval_expression(df, f_expr)

        entry_indices = np.where(combined)[0]
        if len(entry_indices) == 0:
            return None

        close_arr = df["close"].to_numpy()
        high_arr = df["high"].to_numpy()
        low_arr = df["low"].to_numpy()
        ts_arr = df["timestamp"].to_numpy()
        atr_arr = df["atr_14"].to_numpy() if "atr_14" in df.columns else np.zeros(len(df))

        fees = _load_fees(asset_class)
        taker_fee = fees.get("taker", 0.001)
        slippage = self.config["slippage_pct"] / 100.0

        trigger_lower = trigger_expr.lower()
        pattern_key = None
        for pat in self.native_exits.keys():
            if pat in trigger_lower:
                pattern_key = pat
                break
        tp_mult, sl_mult = _get_exit_multipliers(pattern_key or "", self.native_exits)

        trades_returns = []
        equity = [1.0]
        entry_ts = []
        cooldown_end = -1

        for idx in entry_indices:
            # Lag d'1 barre : on entre a la barre suivante la detection
            # pour eliminer le look-ahead minimum
            entry_idx = int(idx) + 1
            if entry_idx <= cooldown_end or entry_idx >= len(df) - 1:
                continue

            # Slippage : long = payer plus cher, short = vendre moins cher
            is_long = direction in ("long", "both")
            is_short = direction in ("short", "both")

            raw_atr = float(atr_arr[idx]) if idx < len(atr_arr) else 0.0
            atr = raw_atr if np.isfinite(raw_atr) and raw_atr > 0 else close_arr[idx] * 0.001
            cooldown_raw = einher.get("cooldown", 3)
            cooldown_bars = _parse_max_holding(cooldown_raw, tf)

            if is_long:
                entry_price = close_arr[entry_idx] * (1.0 + slippage)
                sl_price = entry_price - sl_mult * atr
                tp_price = entry_price + tp_mult * atr
                ret = self._walk_forward(entry_idx, entry_price, sl_price, tp_price, max_holding,
                                         high_arr, low_arr, close_arr, taker_fee, slippage, True)
                trades_returns.append(ret)
                equity.append(equity[-1] * (1.0 + ret))
                entry_ts.append(int(ts_arr[entry_idx]))

            if is_short:
                entry_price = close_arr[entry_idx] * (1.0 - slippage)
                sl_price = entry_price + sl_mult * atr
                tp_price = entry_price - tp_mult * atr
                ret = self._walk_forward(entry_idx, entry_price, sl_price, tp_price, max_holding,
                                         high_arr, low_arr, close_arr, taker_fee, slippage, False)
                trades_returns.append(ret)
                equity.append(equity[-1] * (1.0 + ret))
                entry_ts.append(int(ts_arr[entry_idx]))

            cooldown_end = entry_idx + cooldown_bars
            if idx <= cooldown_end or idx >= len(df) - 1:
                continue

            entry_price = close_arr[idx] * (1.0 + slippage)
            atr = max(atr_arr[idx], entry_price * 0.001)
            cooldown_raw = einher.get("cooldown", 3)
            cooldown_bars = _parse_max_holding(cooldown_raw, tf)

            if direction in ("long", "both"):
                sl_price = entry_price - sl_mult * atr
                tp_price = entry_price + tp_mult * atr
                ret = self._walk_forward(idx, entry_price, sl_price, tp_price, max_holding,
                                         high_arr, low_arr, close_arr, taker_fee, slippage, True)
                trades_returns.append(ret)
                equity.append(equity[-1] * (1.0 + ret))
                entry_ts.append(int(ts_arr[idx]))

            if direction in ("short", "both"):
                sl_price = entry_price + sl_mult * atr
                tp_price = entry_price - tp_mult * atr
                ret = self._walk_forward(idx, entry_price, sl_price, tp_price, max_holding,
                                         high_arr, low_arr, close_arr, taker_fee, slippage, False)
                trades_returns.append(ret)
                equity.append(equity[-1] * (1.0 + ret))
                entry_ts.append(int(ts_arr[idx]))

            cooldown_end = idx + cooldown_bars

        if len(trades_returns) < self.config["min_trades_for_metrics"]:
            return None

        m = metrics.compute_all(trades_returns, equity, entry_ts)
        return EinherBacktestResult(
            einher_id=einher.get("einher_id", einher.get("name", "unknown")),
            einher_name=einher.get("name", "unknown"),
            asset=asset, timeframe=tf, direction=direction,
            n_trades=m["total_trades"], win_rate=m["win_rate"],
            profit_factor=m["profit_factor"], sharpe_ratio=m["sharpe_ratio"],
            sortino_ratio=m["sortino_ratio"], max_drawdown=m["max_drawdown"],
            avg_trade=m["avg_trade"], expectancy=m["expectancy"],
            total_return=m["total_return"], trades_per_month=m["trades_per_month"],
            returns=trades_returns, equity=equity, entry_timestamps=entry_ts,
        )

    def _walk_forward(self, entry_idx, entry_price, sl_price, tp_price, max_bars,
                      high_arr, low_arr, close_arr, taker_fee, slippage, is_long):
        end_idx = min(entry_idx + max_bars + 1, len(close_arr))
        for i in range(entry_idx + 1, end_idx):
            if is_long:
                if low_arr[i] <= sl_price:
                    gross = (sl_price * (1.0 - slippage) - entry_price) / entry_price
                    return gross - (2 * taker_fee)
                if high_arr[i] >= tp_price:
                    gross = (tp_price * (1.0 - slippage) - entry_price) / entry_price
                    return gross - (2 * taker_fee)
            else:
                if high_arr[i] >= sl_price:
                    gross = (entry_price - sl_price * (1.0 + slippage)) / entry_price
                    return gross - (2 * taker_fee)
                if low_arr[i] <= tp_price:
                    gross = (entry_price - tp_price * (1.0 + slippage)) / entry_price
                    return gross - (2 * taker_fee)
        exit_price = close_arr[end_idx - 1]
        gross = ((exit_price - entry_price) / entry_price) if is_long else ((entry_price - exit_price) / entry_price)
        return gross - (2 * taker_fee)

    def run(self, output_path: Path, progress_every: int = 10):
        groups = self._group_by_asset_tf()
        total_groups = len(groups)
        print(f"[Calibrator] {len(self.corpus)} Einhers | {total_groups} groupes (asset, tf)")
        all_results = []
        for gidx, ((asset, tf, asset_class), einhers) in enumerate(groups.items(), 1):
            df = load_ohlcv(asset, tf, asset_class)
            if df is None or len(df) < 100:
                print(f"  [{gidx}/{total_groups}] SKIP {asset}/{tf}")
                continue
            print(f"  [{gidx}/{total_groups}] {asset}/{tf} — {len(einhers)} Einhers")
            for einher in einhers:
                try:
                    res = self._simulate_einher(einher, df, asset, tf, asset_class)
                    if res is not None:
                        all_results.append(res.to_dict())
                except Exception as exc:
                    print(f"    ERR {einher.get('name', '?')}: {exc}")
            if gidx % progress_every == 0 or gidx == total_groups:
                _save_partial(all_results, output_path)
                print(f"    -> Sauvegarde partielle : {len(all_results)} resultats")
        _save_partial(all_results, output_path)
        print(f"[Calibrator] Termine — {len(all_results)} resultats dans {output_path}")
        return all_results


def _save_partial(results: list, path: Path):
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)
    tmp.replace(path)

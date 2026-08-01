#!/usr/bin/env python3
"""discovery_engine.py — Pipeline unique EINHERJAR : donnees brutes MIDAS -> corpus final d'Einhers.

Point d'entree unique du systeme de decouverte. Prend en entree les fichiers
compiles par MIDAS (X.npy / Y_ret.npy / ts.npy / metadata.json) et produit en
sortie un corpus JSON d'Einhers valides, directement utilisable par le
systeme de trading EINHERJAR (execution, risk manager deja prets par
ailleurs — ce fichier ne fait QUE la decouverte/validation des strategies).

=== METHODOLOGIE ===

  Phase 0 — Deduplication des features par correlation, pour ne pas
            re-decouvrir 10 fois le meme signal sous des noms differents
            et pour garder l'espace de recherche gerable.
  Phase 1 — Construction de blocs temporels avec embargo (walk-forward
            purge). Le PREMIER bloc sert de "graine" pour choisir seuils
            et direction ; les blocs suivants ne servent QU'A EVALUER ces
            regles fixes, jamais a les re-choisir (pas de fuite).
  Phase 2 — Recherche en couches (1 -> 2 -> 3 conditions). Chaque couche ne
            part que des survivants de la couche precedente : pas de force
            brute combinatoire sur 200+ features au cube.
  Phase 3 — A chaque condition testee, les couts reels (frais + slippage)
            sont deja retranches AVANT le test statistique — un edge qui ne
            survit pas aux couts est elimine immediatement.
  Phase 4 — Persistance temporelle : une regle n'est retenue que si elle
            reste rentable nette sur une nette majorite des blocs
            d'evaluation (pas juste en moyenne globale). Direction
            verrouillee sur le bloc graine, jamais recalculee ensuite.
  Phase 5 — Simulation realiste finale (SL/TP calibre, une position a la
            fois, cooldown depuis la cloture reelle) sur les survivants,
            puis scoring oriente "croissance de capital avec petit compte"
            (expectancy x frequence, fraction de Kelly, pire bloc, taux de
            survie temporel) plutot qu'un Sharpe pur.

Usage :
    python discovery_engine.py                       # tout l'univers (config/assets_v1.json)
    python discovery_engine.py --asset BTCUSD --asset-class crypto --timeframe 15m
    python discovery_engine.py --debug                # logs detailles
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np

# ============================================================================
# LOGGING — configure une seule fois, clair en terminal (INFO par defaut,
# --debug pour le detail des blocs/conditions testees).
# ============================================================================

logger = logging.getLogger("einherjar.discovery")


def configure_logging(debug: bool = False) -> None:
    level = logging.DEBUG if debug else logging.INFO
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s [%(levelname)-5s] %(message)s",
            datefmt="%H:%M:%S",
        )
    )
    logger.handlers.clear()
    logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False


# ============================================================================
# CHEMINS ET CONFIGURATION
# ============================================================================

EINHERJAR_ROOT = Path(__file__).resolve().parent.parent.parent.parent
MIDAS_ROOT_DEFAULT = Path("D:/midas_v2/midasV3/src/data/compiled")
ASSETS_CONFIG_PATH = EINHERJAR_ROOT / "config" / "assets_v1.json"
FEES_CTRADER_PATH = EINHERJAR_ROOT / "config" / "fees_ctrader.json"
OUTPUT_DIR = EINHERJAR_ROOT / "data" / "discovery"
DEFAULT_TIMEFRAMES = ["5m", "15m", "1h", "4h", "1d"]

DEFAULT_CONFIG: dict[str, Any] = {
    # --- Deduplication (Phase 0) ---
    "correlation_threshold": 0.85,
    # --- Blocs temporels (Phase 1) — adaptes a la profondeur d'historique
    # reelle de chaque actif (voir compute_adaptive_n_blocks) ---
    "target_block_months": 12.0,
    "min_blocks": 4,
    "max_blocks": 10,
    "embargo_bars": 20,
    "min_valid_blocks": 3,
    "min_block_pass_rate": 0.80,  # resserre de 0.70 -> 0.80 suite aux resultats suspects observes
    # --- Recherche en couches (Phase 2) ---
    "percentiles": [10, 20, 30, 40, 50, 60, 70, 80, 90],
    "n_horizons": 4,
    "horizon_names": ["short", "medium", "long", "very_long"],
    "max_conditions": 3,
    "layer_top_n": [60, 25, 12],
    # --- Significativite statistique ---
    "min_occurrences_per_block": 40,  # resserre de 30 -> 40
    "p_value_alpha": 0.03,  # resserre de 0.05 -> 0.03
    # --- Couts reels (Phase 3) — voir DEFAULT_FEES_FALLBACK et
    # compute_round_trip_cost_pct() pour le vrai modele spread+commission,
    # charge depuis config/fee_ctrader.json ---
    # --- Filtres economiques minimaux ---
    "min_net_expectancy": 0.0,
    "min_win_rate": 0.55,  # resserre de 0.50 -> 0.55
    # --- Simulation finale (Phase 5) ---
    "max_holding_bars": {"5m": 36, "15m": 24, "1h": 12, "4h": 8, "1d": 5},
    "cooldown_bars_default": 3,
    "min_trades_final": 30,  # resserre de 20 -> 30
    # --- Garde-fou de sanite (nouveau) : un Einher qui depasse ces
    # valeurs n'est PAS ajoute au corpus final, meme s'il a survecu a
    # toutes les etapes precedentes — au-dela, c'est plus vraisemblablement
    # un artefact de surapprentissage qu'un vrai edge (voir discussion :
    # win rate >85% ou Sharpe >3 sont des signaux d'alarme, pas des bonnes
    # nouvelles). Rejete avec log + conserve a part pour audit, jamais
    # silencieusement perdu.
    "max_sane_win_rate": 0.85,
    "max_sane_sharpe": 3.0,
    # --- Poids du score final (oriente croissance de capital, pas Sharpe pur) ---
    "score_weights": {
        "growth_proxy": 0.30,
        "block_pass_rate": 0.20,
        "kelly_fraction": 0.20,
        "win_rate": 0.15,
        "worst_block_ok": 0.15,
    },
    # Coefficient de la sigmoide normalisant growth_proxy en 0-1 (IA1 #6 :
    # etait un magic number cache dans le code). A CALIBRER sur vos donnees
    # reelles : plus vos "avg_trade_net x trades_per_month" typiques sont
    # petits (actifs peu volatils, peu de trades), plus ce coefficient doit
    # etre GRAND pour eviter que growth_proxy_norm ne reste ecrase pres de
    # 0.5 pour toutes les regles (perte de pouvoir discriminant du score).
    # Inversement, une valeur trop grande sature tout a 0 ou 1. Verifiez la
    # distribution reelle de growth_proxy sur un premier run avant de figer
    # cette valeur.
    "growth_proxy_sigmoid_coef": 200.0,
}

TF_MINUTES = {"5m": 5, "15m": 15, "1h": 60, "4h": 240, "1d": 1440}
OHLCV_NAMES = ("open", "high", "low", "close", "volume")


# ============================================================================
# STATISTIQUES — implementations autonomes (fichier unique).
# ============================================================================

try:
    from scipy import stats as _scipy_stats

    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False


def _normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _student_t_cdf(x: float, df: float) -> float:
    if df <= 0:
        return _normal_cdf(x)
    if _HAS_SCIPY:
        return float(_scipy_stats.t.cdf(x, df))
    return _normal_cdf(x)


def ttest_1samp(data: np.ndarray, popmean: float = 0.0) -> tuple[float, float]:
    n = len(data)
    if n < 2:
        return 0.0, 1.0
    mean = float(np.mean(data))
    std = float(np.std(data, ddof=1))
    if std == 0:
        return 0.0, 1.0
    t_stat = (mean - popmean) / (std / math.sqrt(n))
    p_value = 2.0 * (1.0 - _student_t_cdf(abs(t_stat), n - 1))
    return t_stat, max(0.0, min(1.0, p_value))


def benjamini_hochberg(p_values: list[float], alpha: float = 0.05) -> np.ndarray:
    p = np.asarray(p_values, dtype=np.float64)
    n = len(p)
    if n == 0:
        return np.zeros(0, dtype=bool)
    order = np.argsort(p)
    ranked = p[order]
    thresholds = (np.arange(1, n + 1) / n) * alpha
    passed = ranked <= thresholds
    if not np.any(passed):
        return np.zeros(n, dtype=bool)
    cutoff_p = ranked[int(np.max(np.where(passed)[0]))]
    return p <= cutoff_p


def sharpe_calendar(daily_equity: np.ndarray) -> float:
    if daily_equity is None or len(daily_equity) < 3:
        return 0.0
    r = daily_equity[1:] / daily_equity[:-1] - 1.0
    r = r[np.isfinite(r)]
    if len(r) < 2 or np.std(r, ddof=1) == 0:
        return 0.0
    return float(np.clip(np.mean(r) / np.std(r, ddof=1) * math.sqrt(252), -8.0, 8.0))


def max_drawdown(equity: np.ndarray) -> float:
    if len(equity) < 2:
        return 0.0
    peak = np.maximum.accumulate(equity)
    return float(np.min((equity - peak) / peak))


# ============================================================================
# CHARGEMENT DES DONNEES — metadata.json est l'UNIQUE source de verite pour
# l'ordre des colonnes de X.npy.
# ============================================================================


def load_feature_catalog(midas_root: Path, asset_class: str, timeframe: str) -> list[str]:
    """Charge l'ordre exact des colonnes depuis metadata.json.
    Convention de chemin : <midas_root>/<timeframe>/<asset_class>/metadata.json
    Leve FileNotFoundError (fichier absent) ou ValueError (contenu invalide),
    toujours avec un message explicite incluant le chemin concerne.
    """
    meta_path = midas_root / asset_class/ timeframe  / "metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"metadata.json introuvable : {meta_path}")
    try:
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
    except json.JSONDecodeError as exc:
        raise ValueError(f"metadata.json present mais JSON invalide : {meta_path} ({exc})") from exc
    feature_names = meta.get("feature_names")
    if not feature_names:
        raise ValueError(f"'feature_names' absent ou vide dans {meta_path}")
    return feature_names


def load_midas_arrays(
    midas_root: Path, asset: str, asset_class: str, timeframe: str
) -> Optional[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Charge X, Y_ret, ts pour un actif. Chemin : <tf>/<asset_class>/<asset>_*.npy
    Ne leve jamais d'exception : fichier absent OU corrompu -> log clair et
    retourne None (l'appelant saute cet actif proprement)."""
    base = midas_root / asset_class / timeframe
    paths = {k: base / f"{asset}_{k}.npy" for k in ("X", "Y_ret", "ts")}
    for key, p in paths.items():
        if not p.exists():
            logger.warning("Fichier manquant, actif ignore : %s", p)
            return None

    arrays = {}
    for key, p in paths.items():
        try:
            arrays[key] = np.load(p)
        except (OSError, ValueError, EOFError) as exc:
            logger.error("Fichier %s illisible ou corrompu (%s), actif ignore : %s", key, exc, p)
            return None

    return arrays["X"], arrays["Y_ret"], arrays["ts"]


DEFAULT_FEES_FALLBACK = {
    "spread_pct": 0.0001,
    "commission_per_lot": 0.0,
    "swap_long": 0.0,
    "swap_short": 0.0,
}
FOREX_LOT_NOTIONAL = 100_000.0
# Correspondance connue entre les noms d'actifs (assets_v1.json) et les
# noms de symboles cTrader (fees_ctrader.json), quand ils different. A
# completer si de nouveaux actifs sont ajoutes avec un nom cTrader different.
SYMBOL_NAME_ALIASES = {
    "SP500": "US500",
    "NASDAQ100": "US100",
    "DOWJONES": "US30",
    "DAX40": "DE40",
    "WTIUSD": "USOUSD",
    "BRENT": "UKOUSD",
    "COPPER": "XCUUSD",
}


def load_fees() -> dict:
    """Charge le fichier UNIQUE de frais du broker cTrader.
    Structure attendue : {"default": {...}, "per_symbol": {SYMBOL: {...}}}.
    Ne fait jamais planter le run : en cas de fichier absent ou invalide,
    log un message clair et retombe sur des frais par defaut prudents."""
    if not FEES_CTRADER_PATH.exists():
        logger.error(
            "Fichier de frais introuvable : %s — utilisation de frais par "
            "defaut generiques (spread %.4f%%, sans commission). Les couts "
            "reels ne seront PAS representatifs tant que ce fichier n'existe pas.",
            FEES_CTRADER_PATH,
            DEFAULT_FEES_FALLBACK["spread_pct"] * 100,
        )
        return {"default": dict(DEFAULT_FEES_FALLBACK), "per_symbol": {}}
    try:
        with open(FEES_CTRADER_PATH, encoding="utf-8") as f:
            fees = json.load(f)
    except json.JSONDecodeError as exc:
        logger.error(
            "Fichier de frais present mais JSON invalide : %s (%s) — "
            "utilisation de frais par defaut generiques.",
            FEES_CTRADER_PATH,
            exc,
        )
        return {"default": dict(DEFAULT_FEES_FALLBACK), "per_symbol": {}}
    if "default" not in fees:
        logger.warning("'default' absent de %s, ajout d'une valeur de repli", FEES_CTRADER_PATH)
        fees["default"] = dict(DEFAULT_FEES_FALLBACK)
    fees.setdefault("per_symbol", {})
    return fees


def validate_fees_coverage(fees: dict, assets_config_path: Path) -> None:
    """Verifie, au demarrage, que chaque actif de l'univers a une entree
    de frais explicite (directement ou via alias) — sinon log un
    avertissement CLAIR et NOMINATIF plutot que de laisser un actif
    retomber silencieusement sur des frais generiques potentiellement faux."""
    if not assets_config_path.exists():
        return
    with open(assets_config_path, encoding="utf-8") as f:
        assets = json.load(f).get("assets", [])
    per_symbol = fees.get("per_symbol", {})
    missing = []
    for entry in assets:
        asset = entry["asset"]
        resolved = SYMBOL_NAME_ALIASES.get(asset, asset)
        if resolved not in per_symbol:
            missing.append(asset)
    if missing:
        logger.warning(
            "%d actif(s) sans entree de frais specifique (utiliseront 'default', "
            "potentiellement imprecis) : %s",
            len(missing),
            ", ".join(missing),
        )
    else:
        logger.info(
            "Couverture des frais : tous les actifs de l'univers ont une entree specifique."
        )


def get_symbol_fees(symbol: str, fees_config: dict) -> dict:
    resolved = SYMBOL_NAME_ALIASES.get(symbol, symbol)
    per_symbol = fees_config.get("per_symbol", {})
    if resolved not in per_symbol:
        logger.debug(
            "Pas d'entree de frais pour %s (resolu: %s), repli sur 'default'", symbol, resolved
        )
    return per_symbol.get(resolved, fees_config["default"])


def compute_round_trip_cost_pct(
    symbol: str, asset_class: str, fees_config: dict, reference_price: float
) -> float:
    """Cout d'aller-retour en fraction du prix : spread + commission
    (convertie de $/lot en % via une taille de lot standard pour le forex).
    Le swap (financement overnight) n'est PAS inclus ici (cout fixe par
    trade independant de la duree de detention) — il est applique a part
    dans la simulation realiste (Phase 5), au prorata des jours reellement
    tenus, ce qui est plus juste pour des positions de duree variable."""
    sf = get_symbol_fees(symbol, fees_config)
    spread_cost = float(sf.get("spread_pct", DEFAULT_FEES_FALLBACK["spread_pct"]))
    commission_per_lot = float(sf.get("commission_per_lot", 0.0))
    commission_pct = 0.0
    if commission_per_lot > 0 and reference_price > 0:
        is_forex = asset_class == "forex"
        notional = FOREX_LOT_NOTIONAL * reference_price if is_forex else FOREX_LOT_NOTIONAL
        commission_pct = 2.0 * commission_per_lot / notional  # aller-retour (entree + sortie)
    return spread_cost + commission_pct


def _infer_feature_type(values: np.ndarray) -> str:
    """Determine si une feature est binaire ou continue DIRECTEMENT depuis
    les donnees (pas depuis un fichier de config separe)."""
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return "continuous"
    uniques = np.unique(finite)
    if uniques.size <= 3 and set(np.round(uniques, 6).tolist()).issubset({-1.0, 0.0, 1.0}):
        return "binary"
    return "continuous"


def reconstruct_ohlc(
    X: np.ndarray, feature_names: list[str]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Reconstruit les prix absolus (open, high, low, close) depuis les 4
    premieres colonnes de X.

    Convention (confirmee sur les donnees reelles) : la LIGNE 0 contient
    les prix ABSOLUS bruts (ancre de depart), les LIGNES 1+ contiennent
    des log-returns cumulables. C'est pour ca que `log_rets` ci-dessous a
    une ligne de zeros en position 0 (la ligne 0 ne doit pas etre traitee
    comme un log-return) avant le cumsum. Ce n'est PAS une incoherence
    entre "prix" et "log-return" : les deux cohabitent par construction,
    une seule fois, uniquement sur la toute premiere ligne.
    """
    lower_names = [n.lower() for n in feature_names[:5]]
    if lower_names != list(OHLCV_NAMES):
        raise ValueError(
            f"Attendu {list(OHLCV_NAMES)} dans les 5 premieres colonnes de "
            f"metadata.json['feature_names'], trouve : {feature_names[:5]}"
        )
    o0, h0, l0, c0 = (float(X[0, i]) for i in range(4))
    log_rets = np.vstack([np.zeros((1, 4), dtype=np.float64), X[1:, :4].astype(np.float64)])
    cum_log = np.cumsum(log_rets, axis=0)
    opens = np.exp(cum_log[:, 0] + np.log(max(o0, 1e-12)))
    highs = np.exp(cum_log[:, 1] + np.log(max(h0, 1e-12)))
    lows = np.exp(cum_log[:, 2] + np.log(max(l0, 1e-12)))
    closes = np.exp(cum_log[:, 3] + np.log(max(c0, 1e-12)))
    return opens, highs, lows, closes


# ============================================================================
# PHASE 0 — DEDUPLICATION DES FEATURES PAR CORRELATION
# ============================================================================


def deduplicate_features(
    X_seed: np.ndarray, feature_names: list[str], threshold: float
) -> tuple[list[str], dict[str, str]]:
    """Regroupe les features fortement correlees (sur le bloc graine) et ne
    garde qu'un representant par groupe."""
    stds = np.std(X_seed, axis=0)
    valid_cols = np.where(stds > 1e-12)[0]
    if len(valid_cols) == 0:
        return [], {}

    sub = X_seed[:, valid_cols]
    corr = np.corrcoef(sub.T)
    corr = np.nan_to_num(corr, nan=0.0)

    representatives: list[int] = []
    redirect: dict[str, str] = {}

    for local_i in range(len(valid_cols)):
        name = feature_names[valid_cols[local_i]]
        matched_rep = None
        for rep_local_i in representatives:
            if abs(corr[local_i, rep_local_i]) >= threshold:
                matched_rep = rep_local_i
                break
        if matched_rep is None:
            representatives.append(local_i)
            redirect[name] = name
        else:
            redirect[name] = feature_names[valid_cols[matched_rep]]

    rep_names = [feature_names[valid_cols[i]] for i in representatives]
    n_excluded = len(feature_names) - len(rep_names)
    logger.info(
        "Deduplication : %d features -> %d representants (%d retirees, seuil |r|>=%.2f)",
        len(feature_names),
        len(rep_names),
        n_excluded,
        threshold,
    )
    return rep_names, redirect


# ============================================================================
# PHASE 1 — BLOCS TEMPORELS AVEC EMBARGO
# ============================================================================


def build_embargoed_blocks(
    n_rows: int,
    n_blocks: int,
    embargo_bars: int,
    min_block_size: int = 30,
) -> list[tuple[int, int]]:
    """IA1 point 9 : le filtre `end > start` seul n'exclut que les blocs
    vides/negatifs, pas les blocs trop petits pour etre statistiquement
    exploitables (ex: 5 lignes). On exige desormais une taille minimale
    explicite (`min_block_size`, doit correspondre a min_occurrences_per_block
    au minimum) et on log les blocs rejetes pour cette raison."""
    edges = np.linspace(0, n_rows, n_blocks + 1, dtype=int)
    blocks = []
    n_rejected = 0
    for i in range(n_blocks):
        start, end = int(edges[i]), int(edges[i + 1])
        if i > 0:
            start += embargo_bars
        if i < n_blocks - 1:
            end -= embargo_bars
        if end - start >= min_block_size:
            blocks.append((start, end))
        elif end > start:
            n_rejected += 1
    if n_rejected:
        logger.debug(
            "%d bloc(s) rejete(s) car trop petit(s) (< %d lignes apres embargo)",
            n_rejected,
            min_block_size,
        )
    return blocks


def compute_adaptive_n_blocks(
    ts: np.ndarray,
    min_blocks: int = 4,
    max_blocks: int = 10,
    target_block_months: float = 12.0,
) -> int:
    """Le nombre de blocs s'adapte a la profondeur d'historique REELLE de
    l'actif plutot qu'un compte fixe : la crypto n'a que ~8 ans d'historique
    sur MIDAS contre 15-16 ans pour forex/actions/indices. Un nombre de
    blocs fixe donnerait soit des blocs trop courts sur la crypto (bruit),
    soit trop peu de blocs distincts sur les actifs a long historique (test
    de persistance temporelle trop faible)."""
    span_ms = float(ts[-1] - ts[0])
    span_months = span_ms / (1000.0 * 60 * 60 * 24 * 30.44)
    n = int(round(span_months / target_block_months))
    n = int(np.clip(n, min_blocks, max_blocks))
    logger.debug(
        "Historique ~%.1f mois -> %d blocs (cible %.0f mois/bloc)",
        span_months,
        n,
        target_block_months,
    )
    return n


# ============================================================================
# CONDITIONS ET REGLES
# ============================================================================


@dataclass(frozen=True)
class Condition:
    feature: str
    op: str
    value: float

    def as_string(self) -> str:
        return f"{self.feature} {self.op} {self.value:.6g}"

    def mask(self, X: np.ndarray, name_to_index: dict[str, int]) -> np.ndarray:
        col = X[:, name_to_index[self.feature]]
        if self.op == "<":
            return col < self.value
        if self.op == ">":
            return col > self.value
        # IA1 point 2 : comparaison flottante exacte fragile, tolerance ajoutee.
        return np.isclose(col, self.value, atol=1e-6)


@dataclass
class RuleStats:
    n_valid_blocks: int
    block_pass_rate: float
    pooled_n: int
    pooled_avg_net: float
    pooled_win_rate: float
    pooled_std: float
    pooled_p_value: float
    worst_block_avg_net: float
    trades_per_month_est: float


def _round_trip_cost(taker_fee: float, slippage_pct: float) -> float:
    """DEPRECIEE — conservee uniquement pour compatibilite descendante si
    du code externe l'importait. Le run principal utilise desormais
    compute_round_trip_cost_pct(), base sur le vrai fichier de frais
    (spread + commission), pas cette approximation generique."""
    return 2.0 * (taker_fee + slippage_pct / 100.0)


def evaluate_rule_across_blocks(
    mask_full: np.ndarray,
    Y_ret: np.ndarray,
    direction: str,
    horizon_idx: int,
    test_blocks: list[tuple[int, int]],
    round_trip_cost: float,
    min_occurrences_per_block: int,
    bar_minutes: int,
) -> Optional[RuleStats]:
    pooled: list[float] = []
    block_avgs: list[float] = []
    n_valid_blocks = 0

    for start, end in test_blocks:
        block_mask = mask_full[start:end]
        n = int(np.sum(block_mask))
        if n < min_occurrences_per_block:
            continue
        raw = Y_ret[start:end][block_mask, horizon_idx]
        oriented = raw if direction == "long" else -raw
        net = oriented - round_trip_cost
        pooled.extend(net.tolist())
        block_avgs.append(float(np.mean(net)))
        n_valid_blocks += 1

    if n_valid_blocks == 0 or len(pooled) < min_occurrences_per_block:
        return None

    pooled_arr = np.array(pooled, dtype=np.float64)
    pass_rate = float(np.mean([a > 0 for a in block_avgs]))
    _, p_value = ttest_1samp(pooled_arr, 0.0)

    total_bars = sum(end - start for start, end in test_blocks)
    total_minutes = total_bars * bar_minutes
    total_months = max(total_minutes / (60 * 24 * 30.44), 1e-6)
    trades_per_month_est = len(pooled) / total_months

    return RuleStats(
        n_valid_blocks=n_valid_blocks,
        block_pass_rate=pass_rate,
        pooled_n=len(pooled),
        pooled_avg_net=float(np.mean(pooled_arr)),
        pooled_win_rate=float(np.mean(pooled_arr > 0)),
        pooled_std=float(np.std(pooled_arr, ddof=1)) if len(pooled_arr) > 1 else 0.0,
        pooled_p_value=p_value,
        worst_block_avg_net=min(block_avgs),
        trades_per_month_est=trades_per_month_est,
    )


# ============================================================================
# PHASE 2 — RECHERCHE EN COUCHES (1 -> 2 -> 3 CONDITIONS)
# ============================================================================


@dataclass
class RuleCandidate:
    conditions: list[Condition]
    direction: str
    horizon_idx: int
    stats: RuleStats
    feature_set: frozenset

    def trigger_string(self) -> str:
        return " and ".join(c.as_string() for c in self.conditions)


def _generate_layer1_candidates(
    feature_names: list[str],
    feature_types: dict[str, str],
    X: np.ndarray,
    Y_ret: np.ndarray,
    name_to_index: dict[str, int],
    seed_block: tuple[int, int],
    test_blocks: list[tuple[int, int]],
    cfg: dict,
    round_trip_cost: float,
    bar_minutes: int,
) -> list[RuleCandidate]:
    seed_start, seed_end = seed_block
    X_seed = X[seed_start:seed_end]
    Y_seed = Y_ret[seed_start:seed_end]

    raw_candidates: list[tuple[Condition, str, int, RuleStats]] = []

    for feature_name in feature_names:
        idx = name_to_index[feature_name]
        ftype = feature_types[feature_name]
        values_seed = X_seed[:, idx]

        if ftype == "binary":
            condition_pool = [Condition(feature_name, "==", 1.0)]
        else:
            if np.std(values_seed) == 0:
                continue
            thresholds = sorted(
                set(float(np.percentile(values_seed, p)) for p in cfg["percentiles"])
            )
            condition_pool = []
            for t in thresholds:
                condition_pool.append(Condition(feature_name, "<", t))
                condition_pool.append(Condition(feature_name, ">", t))

        for condition in condition_pool:
            mask_seed = condition.mask(X_seed, name_to_index)
            n_seed = int(np.sum(mask_seed))
            if n_seed < cfg["min_occurrences_per_block"]:
                continue

            for horizon_idx in range(cfg["n_horizons"]):
                seed_returns = Y_seed[mask_seed, horizon_idx]
                avg_seed = float(np.mean(seed_returns))
                if avg_seed == 0:
                    continue
                direction = "long" if avg_seed > 0 else "short"

                mask_full = condition.mask(X, name_to_index)
                stats = evaluate_rule_across_blocks(
                    mask_full,
                    Y_ret,
                    direction,
                    horizon_idx,
                    test_blocks,
                    round_trip_cost,
                    cfg["min_occurrences_per_block"],
                    bar_minutes,
                )
                if stats is None:
                    continue
                if stats.n_valid_blocks < cfg["min_valid_blocks"]:
                    continue
                if stats.pooled_avg_net <= cfg["min_net_expectancy"]:
                    continue
                if stats.pooled_win_rate < cfg["min_win_rate"]:
                    continue

                raw_candidates.append((condition, direction, horizon_idx, stats))

    if not raw_candidates:
        return []

    p_values = [c[3].pooled_p_value for c in raw_candidates]
    bh_mask = benjamini_hochberg(p_values, alpha=cfg["p_value_alpha"])

    survivors: list[RuleCandidate] = []
    for (condition, direction, horizon_idx, stats), keep in zip(raw_candidates, bh_mask):
        if not keep or stats.block_pass_rate < cfg["min_block_pass_rate"]:
            continue
        survivors.append(
            RuleCandidate(
                conditions=[condition],
                direction=direction,
                horizon_idx=horizon_idx,
                stats=stats,
                feature_set=frozenset([condition.feature]),
            )
        )

    logger.debug(
        "Couche 1 : %d conditions testees, %d passent BH+block_pass_rate",
        len(raw_candidates),
        len(survivors),
    )
    return survivors


def _extend_layer(
    previous_survivors: list[RuleCandidate],
    feature_names: list[str],
    feature_types: dict[str, str],
    redirect: dict[str, str],
    X: np.ndarray,
    Y_ret: np.ndarray,
    name_to_index: dict[str, int],
    seed_block: tuple[int, int],
    test_blocks: list[tuple[int, int]],
    cfg: dict,
    round_trip_cost: float,
    bar_minutes: int,
) -> list[RuleCandidate]:
    seed_start, seed_end = seed_block
    X_seed = X[seed_start:seed_end]

    raw_candidates: list[tuple[RuleCandidate, RuleStats]] = []

    for base in previous_survivors:
        base_mask_full = np.ones(X.shape[0], dtype=bool)
        for c in base.conditions:
            base_mask_full &= c.mask(X, name_to_index)

        already_used_groups = {redirect.get(f, f) for f in base.feature_set}

        for filter_name in feature_names:
            if redirect.get(filter_name, filter_name) in already_used_groups:
                continue

            idx = name_to_index[filter_name]
            ftype = feature_types[filter_name]
            values_seed = X_seed[:, idx]

            if ftype == "binary":
                filter_pool = [Condition(filter_name, "==", 1.0)]
            else:
                if np.std(values_seed) == 0:
                    continue
                median = float(np.median(values_seed))
                filter_pool = [
                    Condition(filter_name, "<", median),
                    Condition(filter_name, ">", median),
                ]

            for filter_condition in filter_pool:
                mask_seed_with = base_mask_full[seed_start:seed_end] & filter_condition.mask(
                    X_seed, name_to_index
                )
                n_seed = int(np.sum(mask_seed_with))
                if n_seed < cfg["min_occurrences_per_block"]:
                    continue

                mask_full_with = base_mask_full & filter_condition.mask(X, name_to_index)
                stats_with = evaluate_rule_across_blocks(
                    mask_full_with,
                    Y_ret,
                    base.direction,
                    base.horizon_idx,
                    test_blocks,
                    round_trip_cost,
                    cfg["min_occurrences_per_block"],
                    bar_minutes,
                )
                if stats_with is None or stats_with.n_valid_blocks < cfg["min_valid_blocks"]:
                    continue
                if stats_with.pooled_avg_net <= cfg["min_net_expectancy"]:
                    continue
                if stats_with.pooled_win_rate < cfg["min_win_rate"]:
                    continue
                if stats_with.pooled_avg_net <= abs(base.stats.pooled_avg_net) * 1.05:
                    continue

                new_conditions = base.conditions + [filter_condition]
                new_feature_set = base.feature_set | {filter_name}
                candidate = RuleCandidate(
                    conditions=new_conditions,
                    direction=base.direction,
                    horizon_idx=base.horizon_idx,
                    stats=stats_with,
                    feature_set=new_feature_set,
                )
                raw_candidates.append((candidate, stats_with))

    if not raw_candidates:
        return []

    p_values = [s.pooled_p_value for _, s in raw_candidates]
    bh_mask = benjamini_hochberg(p_values, alpha=cfg["p_value_alpha"])

    survivors: list[RuleCandidate] = []
    for (candidate, stats), keep in zip(raw_candidates, bh_mask):
        if not keep or stats.block_pass_rate < cfg["min_block_pass_rate"]:
            continue
        survivors.append(candidate)

    logger.debug(
        "Extension de couche : %d combinaisons testees, %d passent BH+block_pass_rate",
        len(raw_candidates),
        len(survivors),
    )
    return survivors


def _prune_top_n(candidates: list[RuleCandidate], top_n: int) -> list[RuleCandidate]:
    ranked = sorted(
        candidates, key=lambda c: c.stats.pooled_avg_net * c.stats.block_pass_rate, reverse=True
    )
    return ranked[:top_n]


def run_layered_search(
    feature_names: list[str],
    feature_types: dict[str, str],
    redirect: dict[str, str],
    X: np.ndarray,
    Y_ret: np.ndarray,
    name_to_index: dict[str, int],
    seed_block: tuple[int, int],
    test_blocks: list[tuple[int, int]],
    cfg: dict,
    round_trip_cost: float,
    bar_minutes: int,
) -> list[RuleCandidate]:
    layer1 = _generate_layer1_candidates(
        feature_names,
        feature_types,
        X,
        Y_ret,
        name_to_index,
        seed_block,
        test_blocks,
        cfg,
        round_trip_cost,
        bar_minutes,
    )
    logger.info("Couche 1 (1 condition)  : %d regles retenues", len(layer1))
    if not layer1:
        return []

    all_survivors = list(layer1)
    current = _prune_top_n(layer1, cfg["layer_top_n"][0])

    for layer_idx in range(1, cfg["max_conditions"]):
        extended = _extend_layer(
            current,
            feature_names,
            feature_types,
            redirect,
            X,
            Y_ret,
            name_to_index,
            seed_block,
            test_blocks,
            cfg,
            round_trip_cost,
            bar_minutes,
        )
        logger.info(
            "Couche %d (%d conditions) : %d regles retenues",
            layer_idx + 1,
            layer_idx + 1,
            len(extended),
        )
        if not extended:
            break
        all_survivors.extend(extended)
        top_n_idx = min(layer_idx, len(cfg["layer_top_n"]) - 1)
        current = _prune_top_n(extended, cfg["layer_top_n"][top_n_idx])

    return all_survivors


# ============================================================================
# PHASE 5 — SIMULATION REALISTE FINALE
# ============================================================================


@dataclass
class SimulationResult:
    n_trades: int
    win_rate: float
    avg_trade_net: float
    sharpe_calendar: float
    max_drawdown: float
    tp_pct: float
    sl_pct: float
    trades_per_month: float


def _walk_forward_trade(
    entry_idx: int,
    entry_price: float,
    sl_price: float,
    tp_price: float,
    max_bars: int,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    round_trip_cost: float,
    is_long: bool,
) -> tuple[float, int]:
    end_idx = min(entry_idx + max_bars + 1, len(close))
    for i in range(entry_idx + 1, end_idx):
        if is_long:
            if low[i] <= sl_price:
                return (sl_price - entry_price) / entry_price - round_trip_cost, i
            if high[i] >= tp_price:
                return (tp_price - entry_price) / entry_price - round_trip_cost, i
        else:
            if high[i] >= sl_price:
                return (entry_price - sl_price) / entry_price - round_trip_cost, i
            if low[i] <= tp_price:
                return (entry_price - tp_price) / entry_price - round_trip_cost, i
    exit_idx = end_idx - 1
    exit_price = close[exit_idx]
    gross = (
        (exit_price - entry_price) / entry_price
        if is_long
        else (entry_price - exit_price) / entry_price
    )
    return gross - round_trip_cost, exit_idx


def _calibrate_tp_sl_from_path(
    entry_indices: np.ndarray,
    direction: str,
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    max_holding: int,
) -> Optional[tuple[float, float]]:
    """Calibre TP/SL a partir de l'excursion favorable/defavorable
    REELLEMENT atteinte apres chaque entree du bloc graine, en parcourant
    les vraies barres high/low reconstruites (pas une approximation de
    volatilite) : TP = percentile 75 des MFE, SL = percentile 90 des MAE.
    """
    is_long = direction == "long"
    mfe_list: list[float] = []
    mae_list: list[float] = []

    for idx in entry_indices:
        entry_idx = int(idx) + 1
        end_idx = min(entry_idx + max_holding + 1, len(close))
        if entry_idx >= end_idx - 1:
            continue
        entry_price = close[entry_idx]
        window_high = high[entry_idx + 1 : end_idx]
        window_low = low[entry_idx + 1 : end_idx]
        if len(window_high) == 0:
            continue
        if is_long:
            mfe = (np.max(window_high) - entry_price) / entry_price
            mae = (entry_price - np.min(window_low)) / entry_price
        else:
            mfe = (entry_price - np.min(window_low)) / entry_price
            mae = (np.max(window_high) - entry_price) / entry_price
        mfe_list.append(max(float(mfe), 0.0))
        mae_list.append(max(float(mae), 0.0))

    if len(mfe_list) < 10:
        return None

    tp_pct = max(float(np.percentile(mfe_list, 75)), 0.001)
    sl_pct = max(float(np.percentile(mae_list, 90)), 0.001)
    return tp_pct, sl_pct


def simulate_realistic(
    candidate: RuleCandidate,
    X: np.ndarray,
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    ts: np.ndarray,
    name_to_index: dict[str, int],
    seed_block: tuple[int, int],
    sim_range: tuple[int, int],
    round_trip_cost: float,
    max_holding: int,
    cooldown_bars: int,
    bar_minutes: int,
) -> Optional[SimulationResult]:
    seed_start, seed_end = seed_block
    X_seed = X[seed_start:seed_end]
    seed_mask = np.ones(seed_end - seed_start, dtype=bool)
    for c in candidate.conditions:
        seed_mask &= c.mask(X_seed, name_to_index)
    seed_entry_indices = np.where(seed_mask)[0] + seed_start

    # TP/SL calibres sur le VRAI trajet de prix (MFE/MAE), pas une
    # approximation de volatilite — on a acces a l'integralite des prix
    # reconstruits, donc on l'utilise directement.
    calibration = _calibrate_tp_sl_from_path(
        seed_entry_indices, candidate.direction, close, high, low, max_holding
    )
    if calibration is None:
        # Repli seulement si le bloc graine n'a pas assez d'occurrences
        # pour un calcul MFE/MAE fiable (rare, mais possible sur des
        # regles tres selectives).
        close_seed = close[seed_start:seed_end]
        oriented_close_ret = np.diff(close_seed) / close_seed[:-1]
        vol = float(np.std(oriented_close_ret)) if len(oriented_close_ret) > 1 else 0.001
        tp_pct, sl_pct = max(vol * 3.0, 0.002), max(vol * 2.0, 0.0015)
        logger.debug(
            "MFE/MAE indisponible pour %s, repli sur approximation de volatilite",
            candidate.trigger_string(),
        )
    else:
        tp_pct, sl_pct = calibration

    is_long = candidate.direction == "long"
    start, end = sim_range
    X_range = X[start:end]
    mask_full = np.ones(end - start, dtype=bool)
    for c in candidate.conditions:
        mask_full &= c.mask(X_range, name_to_index)
    entry_indices = np.where(mask_full)[0] + start

    returns: list[float] = []
    exit_ts: list[int] = []
    cooldown_end = -1

    for idx in entry_indices:
        entry_idx = int(idx) + 1
        if entry_idx <= cooldown_end or entry_idx >= end - 1:
            continue
        entry_price = close[entry_idx]
        if is_long:
            sl_price = entry_price * (1.0 - sl_pct)
            tp_price = entry_price * (1.0 + tp_pct)
        else:
            sl_price = entry_price * (1.0 + sl_pct)
            tp_price = entry_price * (1.0 - tp_pct)

        ret, exit_idx = _walk_forward_trade(
            entry_idx,
            entry_price,
            sl_price,
            tp_price,
            max_holding,
            high,
            low,
            close,
            round_trip_cost,
            is_long,
        )
        returns.append(ret)
        exit_ts.append(int(ts[exit_idx]))
        cooldown_end = exit_idx + cooldown_bars

    if len(returns) < 10:
        return None

    equity = np.concatenate([[1.0], np.cumprod(1.0 + np.array(returns))])
    days = [t // (1000 * 60 * 60 * 24) for t in exit_ts]
    day_to_eq: dict[int, float] = {}
    for d, eq in zip(days, equity[1:]):
        day_to_eq[d] = eq
    first_d, last_d = min(days), max(days)
    daily_eq_list = []
    last_val = 1.0
    for d in range(first_d, last_d + 1):
        if d in day_to_eq:
            last_val = day_to_eq[d]
        daily_eq_list.append(last_val)
    daily_eq = np.array(daily_eq_list)

    total_minutes = (end - start) * bar_minutes
    total_months = max(total_minutes / (60 * 24 * 30.44), 1e-6)

    return SimulationResult(
        n_trades=len(returns),
        win_rate=float(np.mean(np.array(returns) > 0)),
        avg_trade_net=float(np.mean(returns)),
        sharpe_calendar=sharpe_calendar(daily_eq),
        max_drawdown=max_drawdown(equity),
        tp_pct=tp_pct,
        sl_pct=sl_pct,
        trades_per_month=len(returns) / total_months,
    )


# ============================================================================
# SCORING FINAL — oriente "faire croitre un petit capital", pas Sharpe pur.
# ============================================================================


def compute_final_score(candidate: RuleCandidate, sim: SimulationResult, cfg: dict) -> float:
    stats = candidate.stats
    variance = stats.pooled_std**2 if stats.pooled_std > 0 else 1e-6
    # Kelly en approximation continue : f* = mu / sigma^2 (Kelly, 1956,
    # cas des rendements continus), PAS la forme discrete f=(bp-q)/b qui
    # suppose un pari binaire a gain fixe. C'est la forme adaptee ici
    # puisque les retours de trade sont continus, pas binaires. "kelly_raw"
    # est donc bien du Kelly, juste sa variante continue.
    kelly_raw = stats.pooled_avg_net / variance
    kelly_fraction = max(0.0, min(kelly_raw * 0.5, 1.0))  # demi-Kelly, cape a 1

    growth_proxy = stats.pooled_avg_net * sim.trades_per_month
    # IA1 point 6 : coefficient de la sigmoide expose en config (n'est plus
    # un magic number cache) — a calibrer sur vos donnees reelles, voir
    # DEFAULT_CONFIG["growth_proxy_sigmoid_coef"].
    sigmoid_coef = cfg.get("growth_proxy_sigmoid_coef", 200.0)
    growth_proxy_norm = 1.0 / (1.0 + math.exp(-growth_proxy * sigmoid_coef))

    worst_block_ok = (
        1.0 if stats.worst_block_avg_net > 0 else max(0.0, 1.0 + stats.worst_block_avg_net * 50.0)
    )

    w = cfg["score_weights"]
    score = (
        w["growth_proxy"] * growth_proxy_norm
        + w["block_pass_rate"] * stats.block_pass_rate
        + w["kelly_fraction"] * kelly_fraction
        + w["win_rate"] * sim.win_rate
        + w["worst_block_ok"] * worst_block_ok
    )
    return round(score, 4)


# ============================================================================
# ORCHESTRATION PAR ACTIF
# ============================================================================


def run_discovery_for_asset(
    asset: str,
    asset_class: str,
    timeframe: str,
    midas_root: Path,
    cfg: dict,
    fees: dict,
) -> tuple[list[dict], list[dict]]:
    logger.info("=== %s / %s / %s ===", asset, asset_class, timeframe)

    feature_names_all = load_feature_catalog(midas_root, asset_class, timeframe)
    data = load_midas_arrays(midas_root, asset, asset_class, timeframe)
    if data is None:
        return [], []
    X, Y_ret, ts = data
    n_rows = X.shape[0]
    if n_rows < cfg["min_blocks"] * (cfg["min_occurrences_per_block"] * 2):
        logger.warning(
            "Pas assez de lignes (%d) pour %s/%s, actif ignore", n_rows, asset, timeframe
        )
        return [], []

    name_to_index = {name: i for i, name in enumerate(feature_names_all)}

    try:
        opens, highs, lows, closes = reconstruct_ohlc(X, feature_names_all)
    except ValueError as exc:
        logger.error("Impossible de reconstruire l'OHLC pour %s/%s : %s", asset, timeframe, exc)
        return [], []

    search_feature_names = [n for n in feature_names_all if n.lower() not in OHLCV_NAMES]
    feature_types = {
        name: _infer_feature_type(X[:, name_to_index[name]]) for name in search_feature_names
    }

    n_blocks = compute_adaptive_n_blocks(
        ts,
        min_blocks=cfg["min_blocks"],
        max_blocks=cfg["max_blocks"],
        target_block_months=cfg["target_block_months"],
    )
    blocks = build_embargoed_blocks(
        n_rows, n_blocks, cfg["embargo_bars"], min_block_size=cfg["min_occurrences_per_block"]
    )
    if len(blocks) < cfg["min_valid_blocks"] + 1:
        logger.warning("Pas assez de blocs valides pour %s/%s, actif ignore", asset, timeframe)
        return [], []
    seed_block, test_blocks = blocks[0], blocks[1:]
    logger.info(
        "Blocs : %d au total (adapte a l'historique) — 1 graine [%d:%d] + %d blocs d'evaluation",
        n_blocks,
        seed_block[0],
        seed_block[1],
        len(test_blocks),
    )

    rep_features, redirect = deduplicate_features(
        X[seed_block[0] : seed_block[1]][:, [name_to_index[n] for n in search_feature_names]],
        search_feature_names,
        cfg["correlation_threshold"],
    )
    if not rep_features:
        logger.warning("Aucune feature exploitable pour %s/%s", asset, timeframe)
        return [], []

    reference_price = float(np.mean(closes[seed_block[0] : seed_block[1]]))
    round_trip_cost = compute_round_trip_cost_pct(asset, asset_class, fees, reference_price)
    logger.debug(
        "Cout aller-retour estime pour %s : %.5f%% (prix de reference %.4f)",
        asset,
        round_trip_cost * 100,
        reference_price,
    )
    bar_minutes = TF_MINUTES.get(timeframe, 60)

    survivors = run_layered_search(
        rep_features,
        feature_types,
        redirect,
        X,
        Y_ret,
        name_to_index,
        seed_block,
        test_blocks,
        cfg,
        round_trip_cost,
        bar_minutes,
    )
    logger.info("Total regles survivantes (toutes couches) : %d", len(survivors))
    if not survivors:
        return [], []

    sim_range = (test_blocks[0][0], test_blocks[-1][1])
    max_holding = cfg["max_holding_bars"].get(timeframe, 12)
    einhers: list[dict] = []
    rejected_suspicious: list[dict] = []

    for candidate in survivors:
        sim = simulate_realistic(
            candidate,
            X,
            closes,
            highs,
            lows,
            ts,
            name_to_index,
            seed_block,
            sim_range,
            round_trip_cost,
            max_holding,
            cfg["cooldown_bars_default"],
            bar_minutes,
        )
        if sim is None or sim.n_trades < cfg["min_trades_final"]:
            continue

        score = compute_final_score(candidate, sim, cfg)
        einher_dict = {
            "name": f"E_{'_'.join(c.feature.upper() for c in candidate.conditions)}_{asset}_{timeframe}",
            "asset": asset,
            "asset_class": asset_class,
            "timeframe": timeframe,
            "direction": candidate.direction,
            "horizon": cfg["horizon_names"][candidate.horizon_idx],
            "trigger": candidate.trigger_string(),
            "n_conditions": len(candidate.conditions),
            "tp_rule": {"type": "vol_calibrated", "value": round(sim.tp_pct, 6)},
            "sl_rule": {"type": "vol_calibrated", "value": round(sim.sl_pct, 6)},
            "max_holding": max_holding,
            "cooldown": cfg["cooldown_bars_default"],
            "block_pass_rate": round(candidate.stats.block_pass_rate, 4),
            "n_valid_blocks": candidate.stats.n_valid_blocks,
            "pooled_p_value": round(candidate.stats.pooled_p_value, 6),
            "n_trades": sim.n_trades,
            "win_rate": round(sim.win_rate, 4),
            "avg_trade_net": round(sim.avg_trade_net, 6),
            "sharpe_calendar": round(sim.sharpe_calendar, 4),
            "max_drawdown": round(sim.max_drawdown, 4),
            "trades_per_month": round(sim.trades_per_month, 2),
            "score": score,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }

        # Garde-fou de sanite : au-dela de ces seuils, c'est plus
        # vraisemblablement un artefact de surapprentissage (calibration
        # MFE/MAE tombee sur un echantillon exceptionnellement favorable)
        # qu'un vrai edge exploitable. Rejete du corpus final, mais jamais
        # perdu silencieusement — conserve a part pour audit.
        reasons = []
        if sim.win_rate > cfg["max_sane_win_rate"]:
            reasons.append(
                f"win_rate={sim.win_rate:.3f} > seuil de sanite {cfg['max_sane_win_rate']}"
            )
        if sim.sharpe_calendar > cfg["max_sane_sharpe"]:
            reasons.append(
                f"sharpe_calendar={sim.sharpe_calendar:.2f} > seuil de sanite {cfg['max_sane_sharpe']}"
            )

        if reasons:
            einher_dict["rejection_reasons"] = reasons
            rejected_suspicious.append(einher_dict)
            logger.warning(
                "Einher REJETE (garde-fou de sanite) %s : %s",
                einher_dict["name"],
                "; ".join(reasons),
            )
            continue

        einhers.append(einher_dict)

    einhers.sort(key=lambda e: e["score"], reverse=True)
    logger.info(
        "%d Einhers finaux pour %s/%s/%s (%d rejetes par le garde-fou de sanite)",
        len(einhers),
        asset,
        asset_class,
        timeframe,
        len(rejected_suspicious),
    )
    return einhers, rejected_suspicious


# ============================================================================
# CLI / MAIN
# ============================================================================


def load_universe(assets_config_path: Path, timeframes: list[str]) -> list[tuple[str, str, str]]:
    """Leve FileNotFoundError ou ValueError avec un message explicite —
    ne jamais laisser une KeyError/JSONDecodeError brute remonter."""
    if not assets_config_path.exists():
        raise FileNotFoundError(f"Fichier d'univers introuvable : {assets_config_path}")
    try:
        with open(assets_config_path, encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{assets_config_path} n'est pas un JSON valide : {exc}") from exc
    if "assets" not in data:
        raise ValueError(f"'assets' absent de {assets_config_path}")

    combos = []
    for i, entry in enumerate(data["assets"]):
        if "asset" not in entry or "class" not in entry:
            logger.warning(
                "Entree #%d de %s incomplete (attend 'asset' et 'class'), ignoree : %s",
                i,
                assets_config_path.name,
                entry,
            )
            continue
        for tf in timeframes:
            combos.append((entry["asset"], entry["class"], tf))
    return combos


def save_rejected(rejected: list[dict], output_dir: Path) -> Optional[Path]:
    """Sauvegarde a part les Einhers rejetes par le garde-fou de sanite —
    jamais perdus silencieusement, disponibles pour audit manuel si besoin
    (ex: si un vrai edge exceptionnel se faisait rejeter a tort)."""
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "corpus_rejected_suspicious.json"
        payload = {
            "_comment": "Einhers rejetes par le garde-fou de sanite (win_rate/sharpe suspects) — a auditer manuellement",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "total_rejected": len(rejected),
            "einhers": rejected,
        }
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        return output_path
    except OSError as exc:
        logger.error("Impossible d'ecrire les Einhers rejetes dans %s (%s)", output_dir, exc)
        return None


def save_corpus(einhers: list[dict], output_dir: Path) -> Optional[Path]:
    """Ne fait jamais perdre silencieusement les resultats deja calcules :
    en cas d'echec d'ecriture, tente un chemin de secours dans le
    repertoire courant avant d'abandonner."""
    payload = {
        "_comment": "Corpus final EINHERJAR — genere par discovery_engine.py (pipeline unique)",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "total_einhers": len(einhers),
        "einhers": einhers,
    }
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "corpus_final.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        return output_path
    except OSError as exc:
        logger.error("Impossible d'ecrire dans %s (%s)", output_dir, exc)
        fallback_path = Path.cwd() / "corpus_final_fallback.json"
        try:
            with open(fallback_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
            logger.warning("Corpus sauvegarde en secours ici : %s", fallback_path)
            return fallback_path
        except OSError as exc2:
            logger.critical(
                "Echec egalement du chemin de secours (%s) — %d Einhers calcules "
                "mais PERDUS, aucun fichier ecrit.",
                exc2,
                len(einhers),
            )
            return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--asset",
        type=str,
        default=None,
        help="Un seul actif (sinon tout l'univers de config/assets_v1.json)",
    )
    parser.add_argument("--asset-class", type=str, default=None)
    parser.add_argument(
        "--timeframe", type=str, default=None, help="Un seul timeframe (sinon tous)"
    )
    parser.add_argument("--midas-root", type=str, default=str(MIDAS_ROOT_DEFAULT))
    parser.add_argument("--output-dir", type=str, default=str(OUTPUT_DIR))
    parser.add_argument("--debug", action="store_true", help="Logs detailles (par condition/bloc)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_logging(debug=args.debug)

    midas_root = Path(args.midas_root)
    cfg = DEFAULT_CONFIG

    if args.asset:
        if not args.asset_class or not args.timeframe:
            logger.error("--asset requiert aussi --asset-class et --timeframe")
            sys.exit(1)
        combos = [(args.asset, args.asset_class, args.timeframe)]
    else:
        timeframes = [args.timeframe] if args.timeframe else DEFAULT_TIMEFRAMES
        try:
            combos = load_universe(ASSETS_CONFIG_PATH, timeframes)
        except (FileNotFoundError, ValueError) as exc:
            logger.error("Impossible de charger l'univers d'actifs : %s", exc)
            logger.error("Fournir --asset/--asset-class/--timeframe pour un run cible a la place.")
            sys.exit(1)
        if not combos:
            logger.error(
                "Aucun couple (asset, asset_class, timeframe) valide dans %s", ASSETS_CONFIG_PATH
            )
            sys.exit(1)

    fees = load_fees()
    validate_fees_coverage(fees, ASSETS_CONFIG_PATH)

    logger.info("Lancement sur %d couples (asset, asset_class, timeframe)", len(combos))

    all_einhers: list[dict] = []
    all_rejected: list[dict] = []
    for i, (asset, asset_class, timeframe) in enumerate(combos, 1):
        logger.info("[%d/%d]", i, len(combos))
        try:
            einhers, rejected = run_discovery_for_asset(
                asset, asset_class, timeframe, midas_root, cfg, fees
            )
            all_einhers.extend(einhers)
            all_rejected.extend(rejected)
        except FileNotFoundError as exc:
            logger.warning(
                "Donnees indisponibles pour %s/%s/%s : %s", asset, asset_class, timeframe, exc
            )
        except ValueError as exc:
            logger.warning(
                "Donnees invalides pour %s/%s/%s : %s", asset, asset_class, timeframe, exc
            )
        except Exception:
            logger.exception("Erreur inattendue sur %s/%s/%s", asset, asset_class, timeframe)

    output_path = save_corpus(all_einhers, Path(args.output_dir))
    if all_rejected:
        save_rejected(all_rejected, Path(args.output_dir))
        logger.warning(
            "%d Einher(s) rejete(s) par le garde-fou de sanite sur l'ensemble du run "
            "(voir corpus_rejected_suspicious.json pour audit).",
            len(all_rejected),
        )
    if output_path is None:
        logger.critical(
            "Aucun fichier de sortie n'a pu etre ecrit — %d Einhers perdus.", len(all_einhers)
        )
        sys.exit(1)
    logger.info("Termine : %d Einhers au total -> %s", len(all_einhers), output_path)


if __name__ == "__main__":
    main()

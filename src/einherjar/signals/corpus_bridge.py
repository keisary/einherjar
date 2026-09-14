"""Pont corpus de recherche -> systeme vivant.

Le corpus de recherche (`outputs/corpus.jsonl`, produit par `research/xgb_einhers`)
contient une ligne JSON par Einher admis :

    {"id": ..., "condition_tree": {"op": "OR", "left": {"feature_ref": ..., "operator": ">=",
     "value": ...}, "right": {...}}, "direction": "BUY"|"SELL", "amplitude_bars": int,
     "tp_pct": float, "sl_pct": float,
     "universe": {"asset": ..., "asset_class": ..., "timeframe": "1h", "horizon": "6h",
                  "horizon_bars": int},
     "metrics": {...}, "scope": "asset", "source": {"model": ...}, "created_at": ...}

Le systeme vivant (`signals/einher_engine.py`) manipule des `Einher` dont le
`trigger` est une EXPRESSION texte evaluee sur la derniere bougie recue
(`EinherEngine._eval_condition`).

Ce module est l'UNIQUE point de conversion entre les deux formats : le corpus
n'est jamais modifie, c'est le code qui s'adapte.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from einherjar.core.enums import Direction, TimeFrame
from einherjar.core.models import Einher

logger = logging.getLogger(__name__)

# Operateurs logiques du corpus -> operateurs Python evalues par EinherEngine.
OPERATOR_MAP = {
    "AND": "and",
    "OR": "or",
    "NOT": "not",
    "&&": "and",
    "||": "or",
    "&": "and",
    "|": "or",
}

# Operateurs de comparaison : normalisation des variantes rencontrees dans les corpus.
COMPARISON_MAP = {
    "=>": ">=",
    "=<": "<=",
    "==": "==",
    "=": "==",
    "!=": "!=",
    "<>": "!=",
}

# Direction du moteur de recherche -> direction du systeme vivant (enums.Direction).
DIRECTION_MAP = {
    "BUY": Direction.LONG.value,
    "LONG": Direction.LONG.value,
    "SELL": Direction.SHORT.value,
    "SHORT": Direction.SHORT.value,
}

# Duree d'une bougie par timeframe (minutes), pour convertir amplitude_bars en duree.
_TF_MINUTES = {"5m": 5, "15m": 15, "1h": 60, "4h": 240, "1d": 1440}


class CorpusBridgeError(Exception):
    """Erreur de lecture ou de conversion du corpus de recherche."""


# ---------------------------------------------------------------------------
# Lecture
# ---------------------------------------------------------------------------


def load_entries(path: str | Path) -> list[dict[str, Any]]:
    """Charge un corpus de recherche (JSONL d'abord, JSON ensuite).

    Tolere : lignes vides, lignes JSON invalides (ignorees avec avertissement),
    fichier JSON unique (liste d'einhers ou dict {"einhers": [...]}).

    Args:
        path: Chemin du corpus (`outputs/corpus.jsonl`).

    Returns:
        Liste des entrees brutes du corpus.

    Raises:
        CorpusBridgeError: si le fichier est absent ou vide.
    """
    p = Path(path)
    if not p.exists():
        raise CorpusBridgeError(f"Corpus absent : {p}")

    text = p.read_text(encoding="utf-8", errors="ignore")
    entries: list[dict[str, Any]] = []
    invalid = 0

    stripped = text.lstrip()
    if stripped.startswith("[") or stripped.startswith("{"):
        # Tentative JSON global (ancien format corpus_v2.json).
        try:
            raw = json.loads(text)
            items = raw.get("einhers", raw) if isinstance(raw, dict) else raw
            entries = [i for i in items if isinstance(i, dict)]
        except json.JSONDecodeError:
            entries = []

    if not entries:
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                invalid += 1
                continue
            if isinstance(obj, dict):
                entries.append(obj)
            else:
                invalid += 1

    if invalid:
        logger.warning("Corpus %s : %d ligne(s) JSON invalide(s) ignoree(s)", p.name, invalid)
    if not entries:
        raise CorpusBridgeError(f"Corpus illisible ou vide : {p}")
    return entries


# ---------------------------------------------------------------------------
# Conversion condition_tree -> expression
# ---------------------------------------------------------------------------


def _format_value(value: Any) -> str:
    """Rend une valeur de seuil sous forme litterale Python."""
    if isinstance(value, bool):
        return "True" if value else "False"
    if isinstance(value, (int, float)):
        return repr(float(value))
    if isinstance(value, str):
        return json.dumps(value)
    raise CorpusBridgeError(f"Valeur de seuil non supportee : {value!r}")


def tree_to_expr(tree: Any) -> str:
    """Convertit un `condition_tree` du corpus en expression texte evaluable.

    Args:
        tree: Noeud du corpus (`{"op": "AND", "left": ..., "right": ...}` ou
            feuille `{"feature_ref": ..., "operator": ..., "value": ...}`).

    Returns:
        Expression Python entre parentheses, p.ex.
        `((parabolic_sar >= 4.26380014) or (atr_21 < 0.0211679507))`.

    Raises:
        CorpusBridgeError: si le noeud est vide ou d'un type inconnu.
    """
    if not isinstance(tree, dict):
        raise CorpusBridgeError(f"Noeud de condition invalide : {tree!r}")

    if "op" in tree and tree["op"]:
        op_raw = str(tree["op"]).strip()
        op = OPERATOR_MAP.get(op_raw.upper(), OPERATOR_MAP.get(op_raw))
        if op is None:
            raise CorpusBridgeError(f"Operateur logique inconnu : {op_raw!r}")

        parts: list[str] = []
        for key in ("left", "right", "a", "b"):
            if key in tree and tree[key] is not None:
                parts.append(tree_to_expr(tree[key]))
        for key in ("children", "operands", "terms"):
            kids = tree.get(key)
            if isinstance(kids, list):
                parts.extend(tree_to_expr(k) for k in kids)
        for key in ("child", "operand"):
            if key in tree and tree[key] is not None:
                parts.append(tree_to_expr(tree[key]))

        if not parts:
            raise CorpusBridgeError(f"Noeud logique sans operande : {tree!r}")
        if op == "not":
            if len(parts) != 1:
                raise CorpusBridgeError(f"'not' attend un seul operande : {tree!r}")
            return f"(not ({parts[0].strip('()')}))"
        return "(" + f" {op} ".join(parts) + ")"

    ref = tree.get("feature_ref") or tree.get("feature") or tree.get("name")
    if not ref:
        raise CorpusBridgeError(f"Feuille de condition sans feature_ref : {tree!r}")

    operator = str(tree.get("operator", "==")).strip()
    operator = COMPARISON_MAP.get(operator, operator)

    if "value" not in tree:
        if operator in ("==",):
            return f"({ref} != 0)"
        raise CorpusBridgeError(f"Feuille sans valeur : {tree!r}")
    return f"({ref} {operator} {_format_value(tree['value'])})"


def feature_refs(tree: Any, out: list[str] | None = None) -> list[str]:
    """Retourne la liste (avec doublons) des `feature_ref` d'un arbre de condition."""
    out = [] if out is None else out
    if isinstance(tree, dict):
        ref = tree.get("feature_ref") or tree.get("feature")
        if ref and not tree.get("op"):
            out.append(str(ref))
        for v in tree.values():
            feature_refs(v, out)
    elif isinstance(tree, list):
        for v in tree:
            feature_refs(v, out)
    return out


# ---------------------------------------------------------------------------
# Conversion entree -> Einher
# ---------------------------------------------------------------------------


def _holding_str(amplitude_bars: Any, timeframe: str) -> str | None:
    """Convertit une amplitude en bougies vers une duree texte (ex `6h`)."""
    try:
        bars = int(amplitude_bars)
    except (TypeError, ValueError):
        return None
    if bars <= 0:
        return None
    minutes = _TF_MINUTES.get(timeframe)
    if minutes is None:
        return f"{bars} bars"
    total_min = bars * minutes
    if total_min % 1440 == 0:
        return f"{total_min // 1440}d"
    if total_min % 60 == 0:
        return f"{total_min // 60}h"
    return f"{total_min}m"


def entry_to_einher(entry: dict[str, Any]) -> Einher:
    """Convertit une entree du corpus de recherche en `Einher` du systeme vivant.

    Args:
        entry: Ligne du corpus (voir docstring du module).

    Returns:
        Einher pret a etre evalue par `EinherEngine`.

    Raises:
        CorpusBridgeError: si un champ obligatoire est absent ou invalide.
    """
    universe = entry.get("universe") or {}
    metrics = entry.get("metrics") or {}
    source = entry.get("source") or {}

    asset = universe.get("asset")
    timeframe = str(universe.get("timeframe") or "")
    if not asset:
        raise CorpusBridgeError(f"Univers sans actif : {entry.get('id')!r}")
    if timeframe not in {tf.value for tf in TimeFrame}:
        raise CorpusBridgeError(
            f"Timeframe {timeframe!r} hors systeme vivant "
            f"({[tf.value for tf in TimeFrame]}) pour {entry.get('id')!r}"
        )

    raw_dir = str(entry.get("direction", "")).strip().upper()
    direction = DIRECTION_MAP.get(raw_dir)
    if direction is None:
        raise CorpusBridgeError(f"Direction inconnue {raw_dir!r} pour {entry.get('id')!r}")

    trigger = tree_to_expr(entry.get("condition_tree"))
    amplitude_bars = entry.get("amplitude_bars")
    tp_pct = entry.get("tp_pct")
    sl_pct = entry.get("sl_pct")

    model = source.get("model") if isinstance(source, dict) else None
    return Einher(
        name=str(entry.get("id") or ""),
        domain=str(model or "corpus"),
        direction=direction,
        timeframes=[timeframe],
        trigger=trigger,
        filters=[],
        assets=[str(asset)],
        tp_rule={"type": "mfe_calibrated", "value": float(tp_pct)} if tp_pct else {},
        sl_rule={"type": "mae_calibrated", "value": float(sl_pct)} if sl_pct else {},
        max_holding=_holding_str(amplitude_bars, timeframe),
        sharpe=float(metrics.get("sharpe_ratio") or 0.0),
        win_rate=float(metrics.get("win_rate") or 0.0),
        avg_tp_pct=float(tp_pct or 0.0),
        avg_sl_pct=float(sl_pct or 0.0),
        trade_count=int(metrics.get("n_trades") or 0),
        profit_horizon=str(universe.get("horizon") or ""),
        calibrated_on=str(entry.get("created_at") or ""),
    )


def load_einhers(path: str | Path) -> list[Einher]:
    """Charge un corpus de recherche et retourne les Einhers du systeme vivant.

    Les entrees invalides (univers hors systeme, direction inconnue, arbre
    illisible) sont ignorees et comptees dans l'avertissement final.

    Args:
        path: Chemin du corpus (`outputs/corpus.jsonl`).

    Returns:
        Liste d'Einhers prets pour `EinherEngine`.
    """
    entries = load_entries(path)
    einhers: list[Einher] = []
    rejets: dict[str, int] = {}
    for entry in entries:
        try:
            einhers.append(entry_to_einher(entry))
        except CorpusBridgeError as exc:
            key = str(exc).split(" pour ")[0][:60]
            rejets[key] = rejets.get(key, 0) + 1
    if rejets:
        logger.info("Pont corpus : %d/%d einhers convertis", len(einhers), len(entries))
        for reason, count in sorted(rejets.items(), key=lambda kv: -kv[1]):
            logger.warning("  %d entrees ignorees : %s", count, reason)
    return einhers


def corpus_feature_refs(path: str | Path) -> list[str]:
    """Retourne les features distinctes utilisees par un corpus (triees)."""
    refs: list[str] = []
    for entry in load_entries(path):
        feature_refs(entry.get("condition_tree"), refs)
    return sorted(set(refs))


def universe_index(path: str | Path) -> dict[tuple[str, str], list[str]]:
    """Indexe le corpus par (asset, timeframe) -> noms d'einhers.

    Args:
        path: Chemin du corpus.

    Returns:
        Dict {(asset, timeframe): [einher_id, ...]}.
    """
    index: dict[tuple[str, str], list[str]] = {}
    for entry in load_entries(path):
        universe = entry.get("universe") or {}
        key = (str(universe.get("asset")), str(universe.get("timeframe")))
        index.setdefault(key, []).append(str(entry.get("id")))
    return index


def required_features_by_universe(path: str | Path) -> dict[tuple[str, str], set[str]]:
    """Features reellement necessaires par couple (asset, timeframe).

    Sert au calcul cible en live : evaluer les einhers d'un couple n'exige que
    l'union de leurs `feature_ref`, pas les 246 colonnes du schema compile.

    Args:
        path: Chemin du corpus.

    Returns:
        Dict {(asset, timeframe): {feature_ref, ...}}.
    """
    besoin: dict[tuple[str, str], set[str]] = {}
    for entry in load_entries(path):
        universe = entry.get("universe") or {}
        key = (str(universe.get("asset")), str(universe.get("timeframe")))
        refs = set(feature_refs(entry.get("condition_tree"), []))
        besoin.setdefault(key, set()).update(refs)
    return besoin

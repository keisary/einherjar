# -*- coding: utf-8 -*-
"""Consolidation du corpus einherjar en UNE passe (decision Jovanny 2026-09-13).

Tout en meme temps, groupe par groupe (asset, timeframe) :
  1. Suppression des doublons exacts (arbre + direction + amplitude + tp/sl).
  2. Quasi-jumeaux (Jaccard features >= 0.8, seuils < 10% rel, meme direction,
     meme amplitude_bars, DANS le meme actif\\tf) -> construction de la version
     GENERALISEE (build_generalized_einher) qui remplace le groupe si elle
     passe l'admission ; sinon repli sur le meilleur membre.
  3. Recalcul des metriques (val + holdout, circuit identique au moteur
     runner.py:_backtest_full) pour les einhers multi ET ceux aux metriques
     absurdes (sharpe_ratio > 15 ou total_return > 100), puis re-admission
     avec les criteres de base (AdmissionConfig()).
Les einhers qui echouent -> archive avec raison.

Sorties versionnees (NE TOUCHE PAS corpus.jsonl / archive.jsonl) :
  outputs/consolidation_<ts>/corpus_consolidated.jsonl
  outputs/consolidation_<ts>/archive_consolidated.jsonl
  outputs/consolidation_<ts>/state.jsonl          (checkpoint par id)
  outputs/manifests/manifest_consolidation_<ts>.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import subprocess
import sys
import time
import uuid
from collections import Counter, OrderedDict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from einherjar.research.xgb_einhers.admission import AdmissionConfig, check_admission
from einherjar.research.xgb_einhers.backtester import backtest_einher, backtest_einher_multi
from einherjar.research.xgb_einhers.corpus import CorpusStore
from einherjar.research.xgb_einhers.data_loader import (
    COMPILED_DIR,
    OHLCV_COLUMNS,
    align_xy_with_ohlcv,
    load_ohlcv,
    load_xy,
)
from einherjar.research.xgb_einhers.einher_io import _dict_to_einher
from einherjar.research.xgb_einhers.types import LoadedData
from einherjar.research.xgb_einhers.label_engineer import load_costs
from einherjar.research.xgb_einhers.multi_asset_loader import list_available_assets
from einherjar.research.xgb_einhers.twin_clustering import (
    FEATURE_JACCARD_MIN,
    THRESHOLD_REL_TOL,
    find_twin_groups,
    build_generalized_einher,
)

logger = logging.getLogger("consolidate")

ABS_SHARPE = 15.0
ABS_RETURN = 100.0

OUT = Path("D:/midas_v2/einherjar/outputs")
MANIFESTS = OUT / "manifests"

# --------------------------------------------------------------------------- #
# Chargement du schema COMPLET (241) et non de la taxonomie (213)
# --------------------------------------------------------------------------- #
def load_xy_full(asset: str, timeframe: str, asset_class: str = "crypto") -> "LoadedData":
    """Charge X + metadata avec TOUTES les colonnes non-OHLCV (241) du schema.

    Pourquoi : le corpus a ete genere sur le schema complet (metadata.json
    - 5 colonnes OHLCV = 241). Load_xy() filtre depuis par la taxonomie
    (218 => 213 features), ce qui fait DISPARAITRE 7 features encore
    utilisees par 106 einhers multi (ema_26_signal, *_norm...). Le recalcul
    doit evaluer sur les memes colonnes que la generation, sinon ces einhers
    retombent a 0 signal de facon factice.
    """
    import json as _json

    base = Path(COMPILED_DIR) / asset_class / timeframe
    X_raw = np.load(base / f"{asset}_X.npy", mmap_mode="r")
    ts = np.load(base / f"{asset}_ts.npy")
    Y_dir = np.load(base / f"{asset}_Y_dir.npy", mmap_mode="r")
    Y_ret = np.load(base / f"{asset}_Y_ret.npy", mmap_mode="r")
    Y_hor = np.load(base / f"{asset}_Y_hor.npy", mmap_mode="r")
    with open(base / "metadata.json") as f:
        meta = _json.load(f)
    all_names = list(meta["feature_names"])
    horizons = tuple(meta["horizons"])
    ohlcv_idx = [i for i, n in enumerate(all_names) if n in OHLCV_COLUMNS]
    keep_idx = [i for i in range(len(all_names)) if i not in ohlcv_idx]
    X = X_raw[:, keep_idx]
    feature_names = tuple(n for i, n in enumerate(all_names) if i in keep_idx)
    return LoadedData(
        asset=asset,
        asset_class=asset_class,
        timeframe=timeframe,
        timestamps=ts,
        X=X,
        Y_dir=Y_dir,
        Y_ret=Y_ret,
        Y_hor=Y_hor,
        feature_names=feature_names,
        horizons=horizons,
    )


# --------------------------------------------------------------------------- #
# Canonisation (identique a corpus_snapshot_audit.py de Jovanny)
# --------------------------------------------------------------------------- #
def canon_tree(t) -> str:
    return json.dumps(t, sort_keys=True)


def exact_key(e: dict) -> str:
    """Cle de doublon exact : arbre + direction + amplitude + tp/sl arrondis."""
    return "|".join(
        [
            canon_tree(e.get("condition_tree") or {}),
            str(e.get("direction")),
            str(e.get("amplitude_bars")),
            f"{float(e.get('tp_pct', 0)):.6f}",
            f"{float(e.get('sl_pct', 0)):.6f}",
        ]
    )


def is_multi(e: dict) -> bool:
    return (e.get("universe") or {}).get("asset") == "multi"


def is_absurde(e: dict) -> bool:
    m = e.get("metrics") or {}
    return float(m.get("sharpe_ratio") or 0) > ABS_SHARPE or float(m.get("total_return") or 0) > ABS_RETURN


# --------------------------------------------------------------------------- #
# Cache de donnees (recalcul)
# --------------------------------------------------------------------------- #
class DataCache:
    """Charge une fois par (asset_class, tf) l'aligned set pour le backtest.

    multi : tous les actifs de la classe (scope market) -> per_asset.
    single : un actif -> (ohlcv_aligned, X_aligned).

    MEMOIRE : ne garde QUE le dernier couple multi et le dernier actif single
    (les matrices 15m/1h pèsent 0.5-7.5 GB ; tout garder = swap/deadlock).
    """

    def __init__(self) -> None:
        self._multi_key: tuple | None = None
        self._multi: tuple = (None, None)
        self._single_key: tuple | None = None
        self._single_val: tuple | None = None

    def multi_per_asset(self, asset_class: str, tf: str) -> tuple[list, list[str]]:
        """Retourne (per_asset, feature_names). feature_names = noms de X (1er actif)."""
        key = (asset_class, tf)
        if key != self._multi_key:
            if self._multi_key is not None:
                del self._multi
                import gc

                gc.collect()
            assets = list_available_assets(asset_class, tf, require_ohlcv=True)
            per = []
            fnames: list[str] = []
            for a in assets:
                try:
                    d = load_xy_full(a, tf, asset_class)
                    o = load_ohlcv(a, tf, asset_class)
                    Xa, oa, _ts = align_xy_with_ohlcv(d, o)
                    per.append((oa, Xa))
                    if not fnames:
                        fnames = list(d.feature_names)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("multi setup %s/%s : %s", asset_class, a, exc)
            self._multi_key = key
            self._multi = (per, fnames)
            logger.info("Cache multi %s/%s : %d actifs", asset_class, tf, len(per))
        return self._multi

    def single(self, asset_class: str, asset: str, tf: str) -> tuple | None:
        key = (asset_class, asset, tf)
        if key != self._single_key:
            if self._single_key is not None:
                del self._single_val
                import gc

                gc.collect()
            try:
                d = load_xy_full(asset, tf, asset_class)
                o = load_ohlcv(asset, tf, asset_class)
                Xa, oa, _ts = align_xy_with_ohlcv(d, o)
                self._single_key = key
                self._single_val = (oa, Xa, list(d.feature_names))
            except Exception as exc:  # noqa: BLE001
                logger.warning("single setup %s/%s/%s : %s", asset, tf, asset_class, exc)
                self._single_key = key
                self._single_val = None
        return self._single_val


# --------------------------------------------------------------------------- #
# Recalcul (circuit identique a runner.py:_backtest_full, lignes 1260-1316)
# --------------------------------------------------------------------------- #
def recalc_einher(e: "Einher", cache: DataCache) -> tuple["Einher | None", str | None]:
    """Re-backteste un einher (val + holdout). Retourne (einher_maj, raison_ech ou None)."""
    from einherjar.research.xgb_einhers.einher_builder import (
        set_einher_holdout_metrics,
        set_einher_metrics,
        set_einher_tp_sl,
    )

    u = e.universe
    asset_class = u.get("asset_class", "crypto")
    tf = u.get("timeframe")
    horizon_bars = e.amplitude_bars
    emb = max(50, horizon_bars)
    costs = _costs_for(asset_class)

    def _missing(fnames: list[str]) -> list[str]:
        fs = set()
        def walk(node) -> None:
            if hasattr(node, "feature_ref"):
                fs.add(node.feature_ref)
                return
            for child in ("left", "right"):
                c = getattr(node, child, None)
                if c is not None:
                    walk(c)
        walk(e.condition_tree)
        return sorted(fs - set(fnames))

    if is_multi({"universe": u}):
        per, fnames = cache.multi_per_asset(asset_class, tf)
        if not per or not fnames:
            return None, "donnees_multi_indisponibles"
        miss = _missing(fnames)
        if miss:
            return None, f"features_absentes_schema_{'_'.join(miss)}"[:120]
        result = backtest_einher_multi(
            einher=e, per_asset=per, feature_names=fnames,
            costs_pct=costs, holdout_embargo=emb, phase="val",
        )
        e = set_einher_metrics(e, result.metrics)
        e = set_einher_tp_sl(e, result.effective_tp_pct, result.effective_sl_pct)
        ho = backtest_einher_multi(
            einher=e, per_asset=per, feature_names=fnames,
            costs_pct=costs, holdout_embargo=emb, phase="holdout",
        )
        e = set_einher_holdout_metrics(e, ho.metrics)
        return e, None

    asset = u.get("asset")
    data = cache.single(asset_class, asset, tf)
    if data is None:
        return None, "donnees_single_indisponibles"
    oa, Xa, fnames = data
    miss = _missing(fnames)
    if miss:
        return None, f"features_absentes_schema_{'_'.join(miss)}"[:120]
    n = len(oa)
    _te = int(n * 0.6)
    _vs = _te + emb
    _ve = min(n, _vs + int(n * 0.2))
    if _vs < _ve:
        result = backtest_einher(e, ohlcv_df=oa[_vs:_ve], X=Xa[_vs:_ve], feature_names=fnames, costs_pct=costs)
    else:
        result = backtest_einher(e, ohlcv_df=oa[:0], X=Xa[:0], feature_names=fnames, costs_pct=costs)
    e = set_einher_metrics(e, result.metrics)
    e = set_einher_tp_sl(e, result.effective_tp_pct, result.effective_sl_pct)
    _hs = _ve + emb
    if _hs < n:
        h_result = backtest_einher(e, ohlcv_df=oa[_hs:], X=Xa[_hs:], feature_names=fnames, costs_pct=costs)
        e = set_einher_holdout_metrics(e, h_result.metrics)
    return e, None


def _costs_for(asset_class: str) -> float:
    try:
        assets = list_available_assets(asset_class, "1h")
        raw = load_costs(assets[0]) if assets else 0.0010
    except Exception:  # noqa: BLE001
        raw = 0.0010
    return max(raw, 0.0010) if asset_class == "crypto" else max(raw, 0.0001)


# --------------------------------------------------------------------------- #
# Manifest
# --------------------------------------------------------------------------- #
def sha256(p: Path) -> str:
    if not p.exists():
        return ""
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_head() -> str:
    try:
        r = subprocess.run(
            ["git", "-C", str(OUT.parent), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=10,
        )
        return r.stdout.strip() if r.returncode == 0 else "n/a"
    except Exception:  # noqa: BLE001
        return "n/a"


def write_manifest(ts: str, out_dir: Path, counts: dict, decisions: dict) -> None:
    MANIFESTS.mkdir(parents=True, exist_ok=True)
    manifest = {
        "kind": "consolidation_corpus",
        "timestamp": ts,
        "git_head": git_head(),
        "input": {
            "corpus": "outputs/corpus.jsonl",
            "corpus_sha256": sha256(OUT / "corpus.jsonl"),
            "n_lines": counts.get("in_lines", 0),
        },
        "methods": {
            "doublon_exact": "arbre + direction + amplitude + tp/sl arrondis a 1e-6",
            "quasi_jumeau": {
                "jaccard_min": FEATURE_JACCARD_MIN,
                "seuil_rel_tol": THRESHOLD_REL_TOL,
                "contrainte": "meme direction + meme amplitude_bars, dans le meme couple (asset, tf)",
                "action": "version GENERALISEE (build_generalized_einher) remplace le groupe "
                          "si elle passe l'admission ; sinon repli meilleur membre",
            },
            "recalcul": "val + holdout (circuit runner._backtest_full) pour einhers multi "
                        "ou metriques absurdes (sharpe>15 ou total_return>100)",
            "admission": "criteres de base AdmissionConfig() (win_rate>=0.65, sharpe>=2.0, "
                         "min_trades=30, profit_factor>=1.5, max_drawdown<0.30, holdout>=30)",
        },
        "counts": counts,
        "decisions": decisions,
    }
    mp = MANIFESTS / f"manifest_consolidation_{ts}.json"
    mp.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Manifest : %s", mp)


# --------------------------------------------------------------------------- #
# Coeur : une passe par groupe (asset, tf)
# --------------------------------------------------------------------------- #
def process_group(group_entries: list[dict], cache: DataCache, out_dir: Path, decisions: dict) -> tuple[list[dict], Counter]:
    """group_entries : lignes JSON du corpus, meme (asset, tf). Retourne (garde, compteurs)."""
    cfg = AdmissionConfig()
    counts = Counter()
    kept: list[dict] = []

    # --- 1. doublons exacts (garde le premier) ---
    seen_keys = set()
    deduped: list[dict] = []
    for e in group_entries:
        k = exact_key(e)
        if k in seen_keys:
            counts["doublons_exacts"] += 1
            _archive(e, f"doublon_exact_{k[-24:]}", out_dir)
        else:
            seen_keys.add(k)
            deduped.append(e)

    # --- 2. quasi-jumeaux -> generalisee ---
    from einherjar.research.xgb_einhers.types import Einher
    objs: list[Einher] = []
    for e in deduped:
        try:
            objs.append(_dict_to_einher(e))
        except Exception as exc:  # noqa: BLE001
            counts["non_deserialisables"] += 1
            _archive(e, f"non_deserialisable_{exc}", out_dir)
    survivors: list[Einher] = []
    if len(objs) >= 2:
        groups = find_twin_groups(objs, max_groups=2000)
    else:
        groups = []
    used_ids: set[str] = set()
    for g in groups:
        members = g.members
        if len(members) < 2:
            continue
        try:
            gen = build_generalized_einher(g, model_tag="twin_consolidated")
        except Exception as exc:  # noqa: BLE001
            logger.warning("generalisation twin echouee : %s", exc)
            gen = None
        if gen is None:
            continue
        gen, gen_raison = recalc_einher(gen, cache)
        if gen is None:
            counts["quasi_jumeaux_non_recalculables"] += 1
            for m in members:
                _archive(_einher_to_dict(m), f"quasi_jumeau_generalise_indisponible_{gen_raison}", out_dir)
            continue
        passed, _reason = check_admission(gen, cfg)
        # trier les membres par qualite (sharpe val desc)
        def _q(x):
            m = getattr(x, "metrics", None)
            return float(getattr(m, "sharpe_ratio", 0) or 0)
        if passed:
            counts["quasi_jumeaux_generalises"] += 1
            used_ids.update(m.id for m in members)
            survivors.append(gen)
            for m in members:
                if m.id != gen.id:
                    _archive(_einher_to_dict(m), f"quasi_jumeau_fusionne_dans_{gen.id}", out_dir)
            decisions[gen.id] = {
                "type": "twin_generalized",
                "membres": [m.id for m in members],
                "admission": "passe",
            }
        else:
            # repli : on garde le MEILLEUR membre du groupe, les autres partent
            best = max(members, key=_q)
            counts["quasi_jumeaux_repli_meilleur"] += 1
            used_ids.add(best.id)
            survivors.append(best)
            for m in members:
                if m.id != best.id:
                    _archive(_einher_to_dict(m), f"quasi_jumeau_repli_{best.id}", out_dir)
            decisions[best.id] = {
                "type": "twin_best_member",
                "membres": [m.id for m in members],
                "admission": "generalisee_echouee_repli_meilleur",
            }
    for e in objs:
        if e.id not in used_ids:
            survivors.append(e)

    # --- 3. recalcul multi + absurdes, puis admission ---
    out_rec = []
    for e in survivors:
        d = _einher_to_dict(e)
        if is_multi(d) or is_absurde(d):
            before = _metrics_short(d)
            e2, raison = recalc_einher(e, cache)
            if e2 is None:
                counts["recalcul_indispo"] += 1
                _archive(d, f"recalcul_indisponible_{raison}", out_dir)
                continue
            passed, reason = check_admission(e2, cfg)
            d2 = _einher_to_dict(e2)
            counts["recalcules"] += 1
            decisions[d2["id"]] = {
                "type": "recalcul",
                "avant": before,
                "apres": _metrics_short(d2),
                "admission": "passe" if passed else f"rejete_{reason}",
            }
            if passed:
                out_rec.append(d2)
            else:
                _archive(d2, f"admission_apres_recalcul_{reason}", out_dir)
        else:
            out_rec.append(d)
    counts["garde"] = len(out_rec)
    return out_rec, counts


# --------------------------------------------------------------------------- #
# I/O helpers
# --------------------------------------------------------------------------- #
def _einher_to_dict(e) -> dict:
    return e.to_dict()


def _metrics_short(d: dict) -> dict:
    m = d.get("metrics") or {}
    return {
        "n_trades": m.get("n_trades"),
        "win_rate": round(float(m.get("win_rate") or 0), 4),
        "sharpe_ratio": round(float(m.get("sharpe_ratio") or 0), 2),
        "total_return": round(float(m.get("total_return") or 0), 2),
    }


_ARCHIVE_FH = None


def _archive(d: dict, reason: str, out_dir: Path) -> None:
    global _ARCHIVE_FH
    if _ARCHIVE_FH is None:
        _ARCHIVE_FH = open(out_dir / "archive_consolidated.jsonl", "a", encoding="utf-8")
    rec = {"einher": d, "rejection_reason": reason, "consolidated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ")}
    _ARCHIVE_FH.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(name)s | %(levelname)s | %(message)s")
    ap = argparse.ArgumentParser(description="Consolidation du corpus einherjar (une passe par actif\\tf)")
    ap.add_argument("--one-group", action="store_true", help="Test : traite un seul groupe puis s'arrete")
    ap.add_argument("--out-ts", type=str, default=None, help="Timestamp du dossier de sortie (reprise)")
    args = ap.parse_args()
    ts = args.out_ts or time.strftime("%Y%m%dT%H%M%S")
    out_dir = OUT / f"consolidation_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    cache = DataCache()
    decisions: dict = {}

    # 1. Charger le corpus brut
    store = CorpusStore(OUT / "corpus.jsonl")
    raw: list[dict] = []
    with open(OUT / "corpus.jsonl", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                raw.append(json.loads(line))
            except Exception as exc:  # noqa: BLE001
                logger.warning("ligne malformee ignoree : %s", str(exc)[:80])
    logger.info("corpus brut : %d lignes valides", len(raw))

    # 2. Grouper par (asset_class, asset, tf) - l'ordre original est preserve par groupe
    #   (la classe fait partie de l'identite : un multi forex n'est pas le jumeau
    #   d'un multi crypto meme sur le meme timeframe).
    groups: "OrderedDict[tuple, list[dict]]" = OrderedDict()
    for e in raw:
        u = e.get("universe") or {}
        groups.setdefault((u.get("asset_class"), u.get("asset"), u.get("timeframe")), []).append(e)
    logger.info("groupes (asset_class, asset, tf) : %d", len(groups))

    # 3. Une passe par groupe
    all_kept: list[dict] = []
    final_counts = Counter({"in_lines": len(raw), "in_groups": len(groups)})
    state_path = out_dir / "state.jsonl"
    done_groups: set[tuple] = set()
    if state_path.exists():
        for line in state_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                s = json.loads(line)
            except Exception:  # noqa: BLE001
                continue
            if s.get("group_complete"):
                done_groups.add(tuple(s["group_complete"]))
    logger.info("groupes deja traites (reprise) : %d", len(done_groups))

    for gi, ((ac, asset, tf), entries) in enumerate(groups.items(), 1):
        if (ac, asset, tf) in done_groups:
            logger.info("[%d/%d] groupe %s %s %s : DEJA TRAITE, skip", gi, len(groups), ac, asset, tf)
            continue
        logger.info("[%d/%d] groupe %s %s %s (%d lignes)", gi, len(groups), ac, asset, tf, len(entries))
        kept, counts = process_group(entries, cache, out_dir, decisions)
        counts = dict(counts)
        for k, v in counts.items():
            final_counts[k] = final_counts.get(k, 0) + v
        # append corpus consolide (ordre du groupe)
        new_lines = []
        with open(out_dir / "corpus_consolidated.jsonl", "a", encoding="utf-8") as fh:
            for d in kept:
                fh.write(json.dumps(d, ensure_ascii=False, default=str) + "\n")
                new_lines.append(d)
        # checkpoint
        with open(state_path, "a", encoding="utf-8") as fh:
            for d in new_lines:
                fh.write(json.dumps({"id": d["id"], "group": [ac, asset, tf]}, ensure_ascii=False) + "\n")
            fh.write(json.dumps({"group_complete": [ac, asset, tf]}, ensure_ascii=False) + "\n")
        logger.info("  -> garde %d, cumul %d", len(kept), final_counts.get("garde", 0))
        if args.one_group:
            logger.info("--one-group : arret apres le premier groupe traite")
            break

    if _ARCHIVE_FH is not None:
        _ARCHIVE_FH.close()

    final_counts["out_lines"] = final_counts.get("garde", 0)
    write_manifest(ts, out_dir, dict(final_counts), decisions)
    print(json.dumps(dict(final_counts), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
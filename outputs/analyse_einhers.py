#!/usr/bin/env python3
"""analyse_einhers.py — Analyse complète du corpus et archive Einhers.

Usage :
    python analyse_einhers.py
    python analyse_einhers.py --corpus outputs/corpus.jsonl --archive outputs/archive.jsonl

Sections :
    1. Statistiques globales (moyennes, min, max, médianes)
    2. Meilleurs Einhers individuels (par métrique)
    3. Score pondéré de qualité (un seul chiffre)
    4. Top Einhers par score de qualité
    5. Analyse des valeurs aberrantes
    6. Diversité des features
    7. Distribution par actif/timeframe/horizon
    8. Analyse de l'archive (raisons de rejet)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Chargement
# ---------------------------------------------------------------------------

def load_jsonl(path: str) -> list[dict]:
    """Charge un fichier JSONL en gérant les lignes corrompues."""
    rows = []
    bad = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                bad += 1
    if bad:
        print(f"  ⚠ {bad} lignes corrompues ignorées dans {path}")
    return rows


def extract_metrics(einher: dict) -> dict:
    """Extrait les métriques d'un Einher (corpus ou archive)."""
    if "metrics" in einher:
        return einher["metrics"]
    if "einher" in einher and "metrics" in einher["einher"]:
        return einher["einher"]["metrics"]
    return {}


def extract_einher(einher: dict) -> dict:
    """Extrait l'Einher lui-même (corpus ou archive)."""
    if "einher" in einher:
        return einher["einher"]
    return einher


# ---------------------------------------------------------------------------
# Score de qualité pondéré
# ---------------------------------------------------------------------------

def quality_score(m: dict) -> float:
    """Score de qualité pondéré en [0, 100].

    Pondération équilibrée :
      - Sharpe (25%) : capped à 10 pour éviter les valeurs absurdes
      - Win rate (20%) : directement en [0, 1]
      - Profit factor (20%) : log-scaled, capped à 10
      - Total return (15%) : log-scaled
      - Max drawdown (10%) : inversé (moins = mieux)
      - N trades (10%) : pénalité si trop peu

    Toutes les valeurs sont clippées pour éviter les outliers.
    """
    sharpe = m.get("sharpe_ratio", 0)
    wr = m.get("win_rate", 0)
    pf = m.get("profit_factor", 0)
    total_ret = m.get("total_return", 0)
    max_dd = abs(m.get("max_drawdown", 0))
    n_trades = m.get("n_trades", 0)

    # Clipper les valeurs absurdes
    sharpe = np.clip(sharpe, -5, 15)
    pf = np.clip(pf, 0, 20)
    total_ret = np.clip(total_ret, -0.99, 100)
    max_dd = np.clip(max_dd, 0, 1)
    n_trades = np.clip(n_trades, 0, 5000)

    # Sharpe : normaliser en [0, 1] avec cap à 10
    # sharpe=0 → 0, sharpe=5 → 0.5, sharpe=10 → 1.0
    s_sharpe = np.clip(sharpe / 10.0, 0, 1)

    # Win rate : directement [0, 1]
    s_wr = np.clip(wr, 0, 1)

    # Profit factor : log-scaled
    # pf=1 → 0, pf=2 → 0.3, pf=5 → 0.7, pf=10 → 1.0
    if pf > 1:
        s_pf = np.clip(np.log10(pf) / np.log10(10), 0, 1)
    else:
        s_pf = 0.0

    # Total return : log-scaled
    # 0% → 0, 10% → 0.3, 100% → 0.7, 1000% → 1.0
    if total_ret > 0:
        s_ret = np.clip(np.log10(1 + total_ret * 100) / 3.0, 0, 1)
    else:
        s_ret = 0.0

    # Max drawdown : inversé (0% → 1.0, 30% → 0.0)
    s_dd = np.clip(1.0 - max_dd / 0.30, 0, 1)

    # N trades : pénalité si < 20 trades
    if n_trades >= 30:
        s_trades = 1.0
    elif n_trades >= 20:
        s_trades = 0.8
    elif n_trades >= 10:
        s_trades = 0.5
    else:
        s_trades = 0.2

    # Score pondéré
    score = (
        0.25 * s_sharpe
        + 0.20 * s_wr
        + 0.20 * s_pf
        + 0.15 * s_ret
        + 0.10 * s_dd
        + 0.10 * s_trades
    ) * 100

    return round(score, 2)


# ---------------------------------------------------------------------------
# Analyse
# ---------------------------------------------------------------------------

METRIC_KEYS = [
    "n_trades", "win_rate", "sharpe_ratio", "total_return",
    "profit_factor", "max_drawdown", "avg_net_return",
    "tp_hit_rate", "t_statistic", "p_value",
]

METRIC_LABELS = {
    "n_trades": "Nb trades",
    "win_rate": "Win rate",
    "sharpe_ratio": "Sharpe ratio",
    "total_return": "Total return",
    "profit_factor": "Profit factor",
    "max_drawdown": "Max drawdown",
    "avg_net_return": "Avg net return",
    "tp_hit_rate": "TP hit rate",
    "t_statistic": "t-statistic",
    "p_value": "p-value",
}


def print_section(title: str):
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}")


def analyze_corpus(corpus: list[dict]):
    """Analyse complète du corpus."""
    if not corpus:
        print("  Corpus vide.")
        return

    metrics_list = [extract_metrics(e) for e in corpus]
    scores = [quality_score(m) for m in metrics_list]

    # --- Section 1 : Statistiques globales ---
    print_section("1. STATISTIQUES GLOBALES DU CORPUS")
    print(f"  Total Einhers : {len(corpus)}")
    print(f"  Score qualité : moy={np.mean(scores):.1f} méd={np.median(scores):.1f} min={np.min(scores):.1f} max={np.max(scores):.1f}")
    print()

    print(f"  {'Métrique':<20} {'Moyenne':>10} {'Médiane':>10} {'Min':>10} {'Max':>10} {'Std':>10}")
    print(f"  {'-'*20} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")
    for key in METRIC_KEYS:
        vals = [m.get(key, 0) for m in metrics_list]
        if key == "max_drawdown":
            vals = [abs(v) for v in vals]
        print(f"  {METRIC_LABELS.get(key, key):<20} {np.mean(vals):>10.4f} {np.median(vals):>10.4f} {np.min(vals):>10.4f} {np.max(vals):>10.4f} {np.std(vals):>10.4f}")

    # --- Section 2 : Meilleurs Einhers par métrique ---
    print_section("2. MEILLEURS EINHERS PAR MÉTRIQUE (individuel)")
    for key in ["sharpe_ratio", "win_rate", "profit_factor", "total_return"]:
        sorted_einhers = sorted(corpus, key=lambda e: extract_metrics(e).get(key, 0), reverse=True)
        print(f"\n  Top 5 par {METRIC_LABELS.get(key, key)}:")
        for i, e in enumerate(sorted_einhers[:5]):
            m = extract_metrics(e)
            eid = e.get("id", "?")[:35]
            d = e.get("direction", "?")
            a = e.get("universe", {}).get("asset", "?")
            tf = e.get("universe", {}).get("timeframe", "?")
            h = e.get("universe", {}).get("horizon", "?")
            print(f"    {i+1}. {eid} {d} {a}/{tf}/{h} | {key}={m.get(key, 0):.4f}")

    # --- Section 3 : Top Einhers par score de qualité ---
    print_section("3. TOP 20 EINHERS PAR SCORE DE QUALITÉ")
    scored = list(zip(corpus, scores))
    scored.sort(key=lambda x: -x[1])
    print(f"  {'#':<3} {'Score':>6} {'ID':<38} {'Dir':<5} {'Asset':<10} {'TF':<5} {'H':<6} {'Sharpe':>8} {'WR':>6} {'PF':>6} {'Trades':>7}")
    print(f"  {'-'*3} {'-'*6} {'-'*38} {'-'*5} {'-'*10} {'-'*5} {'-'*6} {'-'*8} {'-'*6} {'-'*6} {'-'*7}")
    for i, (e, s) in enumerate(scored[:20]):
        m = extract_metrics(e)
        eid = e.get("id", "?")[:36]
        d = e.get("direction", "?")
        a = e.get("universe", {}).get("asset", "?")[:8]
        tf = e.get("universe", {}).get("timeframe", "?")
        h = e.get("universe", {}).get("horizon", "?")
        print(f"  {i+1:<3} {s:>6.1f} {eid:<38} {d:<5} {a:<10} {tf:<5} {h:<6} {m.get('sharpe_ratio', 0):>8.2f} {m.get('win_rate', 0):>6.2f} {m.get('profit_factor', 0):>6.2f} {m.get('n_trades', 0):>7d}")

    # --- Section 4 : Valeurs aberrantes ---
    print_section("4. VALEURS ABERRANTES DANS LE CORPUS")
    outliers = []
    for e in corpus:
        m = extract_metrics(e)
        eid = e.get("id", "?")[:35]
        issues = []
        if m.get("sharpe_ratio", 0) > 15:
            issues.append(f"sharpe={m['sharpe_ratio']:.1f}")
        if m.get("win_rate", 0) > 0.95:
            issues.append(f"wr={m['win_rate']:.3f}")
        if m.get("profit_factor", 0) > 20:
            issues.append(f"pf={m['profit_factor']:.1f}")
        if m.get("n_trades", 0) < 5:
            issues.append(f"n={m['n_trades']}")
        if abs(m.get("max_drawdown", 0)) < 0.001 and m.get("n_trades", 0) > 10:
            issues.append("dd≈0")
        if issues:
            outliers.append((eid, issues))

    if outliers:
        print(f"  {len(outliers)} Einhers avec valeurs suspectes :")
        for eid, issues in outliers[:20]:
            print(f"    {eid} : {', '.join(issues)}")
    else:
        print("  Aucune valeur aberrante détectée.")

    # --- Section 5 : Diversité des features ---
    print_section("5. DIVERSITÉ DES FEATURES")
    features = Counter()
    for e in corpus:
        ct = e.get("condition_tree", {})
        def walk(n):
            if isinstance(n, dict):
                if "feature_ref" in n:
                    features[n["feature_ref"]] += 1
                for v in n.values():
                    if isinstance(v, (dict, list)):
                        walk(v)
            elif isinstance(n, list):
                for i in n:
                    walk(i)
        walk(ct)

    print(f"  Features uniques : {len(features)}")
    print(f"\n  Top 20 features (par occurrence) :")
    for f, c in features.most_common(20):
        pct = c / len(corpus) * 100
        bar = "█" * int(pct / 2)
        print(f"    {f:<35} {c:>4} ({pct:>5.1f}%) {bar}")

    # --- Section 6 : Distribution ---
    print_section("6. DISTRIBUTION PAR ACTIF / TIMEFRAME / HORIZON")
    assets = Counter()
    tfs = Counter()
    horizons = Counter()
    directions = Counter()
    sources = Counter()
    for e in corpus:
        u = e.get("universe", {})
        assets[u.get("asset", "?")] += 1
        tfs[u.get("timeframe", "?")] += 1
        horizons[u.get("horizon", "?")] += 1
        directions[e.get("direction", "?")] += 1
        sources[e.get("source", {}).get("model", "?")] += 1

    print(f"\n  Par actif :")
    for a, c in sorted(assets.items(), key=lambda x: -x[1]):
        print(f"    {a:<12} {c:>4}")
    print(f"\n  Par timeframe :")
    for t, c in sorted(tfs.items(), key=lambda x: -x[1]):
        print(f"    {t:<6} {c:>4}")
    print(f"\n  Par horizon :")
    for h, c in sorted(horizons.items(), key=lambda x: -x[1]):
        print(f"    {h:<6} {c:>4}")
    print(f"\n  Par direction : {dict(directions)}")
    print(f"\n  Par source :")
    for s, c in sorted(sources.items(), key=lambda x: -x[1]):
        print(f"    {s:<30} {c:>4}")


def analyze_archive(archive: list[dict]):
    """Analyse de l'archive (rejets)."""
    if not archive:
        print("  Archive vide.")
        return

    print_section("7. ANALYSE DE L'ARCHIVE (REJETS)")
    print(f"  Total rejetés : {len(archive)}")

    # Raisons de rejet
    reasons = Counter()
    for e in archive:
        r = e.get("rejection_reason", "?")
        if "BH" in r:
            reasons["BH REJECTED (non significatif)"] += 1
        elif "sharpe" in r.lower():
            reasons["Sharpe trop bas"] += 1
        elif "win_rate" in r.lower():
            reasons["Win rate trop bas"] += 1
        elif "n_trades" in r.lower():
            reasons["Nb trades trop bas"] += 1
        elif "profit_factor" in r.lower():
            reasons["Profit factor trop bas"] += 1
        elif "total_return" in r.lower():
            reasons["Total return <= 0"] += 1
        elif "drawdown" in r.lower():
            reasons["Drawdown trop haut"] += 1
        elif "holdout" in r.lower():
            reasons["Holdout REJECTED"] += 1
        elif "superseceded" in r or "superseded" in r:
            reasons["Superseded by veto"] += 1
        else:
            reasons[r[:50]] += 1

    print(f"\n  Raisons de rejet :")
    for r, c in reasons.most_common():
        pct = c / len(archive) * 100
        bar = "█" * int(pct / 2)
        print(f"    {r:<35} {c:>5} ({pct:>5.1f}%) {bar}")

    # Meilleurs rejetés (par sharpe)
    archive_metrics = []
    for e in archive:
        einher = e.get("einher", {})
        m = einher.get("metrics", {})
        if m.get("sharpe_ratio", 0) > 0:
            archive_metrics.append((einher, m, e.get("rejection_reason", "?")))

    archive_metrics.sort(key=lambda x: -x[1].get("sharpe_ratio", 0))
    print(f"\n  Top 10 rejetés par Sharpe (potentiels faux négatifs) :")
    for i, (einher, m, reason) in enumerate(archive_metrics[:10]):
        eid = einher.get("id", "?")[:35]
        d = einher.get("direction", "?")
        a = einher.get("universe", {}).get("asset", "?")
        print(f"    {i+1}. {eid} {d} {a} | sharpe={m.get('sharpe_ratio', 0):.2f} wr={m.get('win_rate', 0):.2f} | {reason[:40]}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Analyse complète des Einhers")
    parser.add_argument("--corpus", default="outputs/corpus.jsonl")
    parser.add_argument("--archive", default="outputs/archive.jsonl")
    args = parser.parse_args()

    # Résoudre les chemins relativement au repo
    repo = Path(__file__).resolve().parent.parent
    corpus_path = repo / args.corpus if not os.path.isabs(args.corpus) else Path(args.corpus)
    archive_path = repo / args.archive if not os.path.isabs(args.archive) else Path(args.archive)

    print("=" * 70)
    print("  ANALYSE COMPLÈTE DES EINHERS")
    print("=" * 70)
    print(f"  Corpus  : {corpus_path}")
    print(f"  Archive : {archive_path}")

    corpus = load_jsonl(str(corpus_path)) if corpus_path.exists() else []
    archive = load_jsonl(str(archive_path)) if archive_path.exists() else []

    analyze_corpus(corpus)
    analyze_archive(archive)

    print_section("8. RÉSUMÉ EXÉCUTIF")
    if corpus:
        metrics_list = [extract_metrics(e) for e in corpus]
        scores = [quality_score(m) for m in metrics_list]
        n_absurd = sum(1 for m in metrics_list if m.get("sharpe_ratio", 0) > 15)
        print(f"  Corpus : {len(corpus)} Einhers admis")
        print(f"  Archive : {len(archive)} Einhers rejetés")
        print(f"  Score qualité moyen : {np.mean(scores):.1f}/100")
        print(f"  Valeurs aberrantes (sharpe>15) : {n_absurd}")
        print(f"  Features uniques utilisées : {len(set(f for e in corpus for f in _extract_features(e)))}")
        print(f"  Actifs couverts : {len(set(e.get('universe', {}).get('asset', '?') for e in corpus))}")
    print()


def _extract_features(einher: dict) -> list[str]:
    """Extrait les noms de features d'un Einher."""
    ct = einher.get("condition_tree", {})
    feats = []
    def walk(n):
        if isinstance(n, dict):
            if "feature_ref" in n:
                feats.append(n["feature_ref"])
            for v in n.values():
                if isinstance(v, (dict, list)):
                    walk(v)
        elif isinstance(n, list):
            for i in n:
                walk(i)
    walk(ct)
    return feats


if __name__ == "__main__":
    main()

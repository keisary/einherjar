"""xgb_sanity_check.py - Sanity check des données MIDAS V3 avant pipeline XGBoost.

Vérifie pour BTCUSD × 1h :
- Shapes et dtypes de X, Y_dir, Y_ret, Y_hor
- Taux de validité (Y_dir != -100)
- Distribution de Y_ret par horizon
- Cohérence avec metadata.json
- Présence du dossier OHLCV CSV brut

Output : rapport console structuré.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

# Chemins
COMPILED_DIR = Path("D:/midas_v2/midasV3/src/data/compiled")
OHLCV_DIR = Path("D:/midas_v2/technical_agent_dataset_brut")
TAXONOMY_PATH = Path("D:/midas_v2/Einherjar/src/einherjar/research/config/features_taxonomy.json")

ASSET = "BTCUSD"
ASSET_CLASS = "crypto"
TIMEFRAME = "1h"


def section(title: str) -> None:
    print()
    print("=" * 70)
    print(f"  {title}")
    print("=" * 70)


def main() -> int:
    section(f"X.npy / Y_*.npy pour {ASSET} × {TIMEFRAME}")

    base = COMPILED_DIR / ASSET_CLASS / TIMEFRAME
    print(f"Répertoire : {base}")
    print(f"Existe ?   : {base.exists()}")

    if not base.exists():
        print("ERREUR : répertoire absent")
        return 1

    # Liste les fichiers disponibles
    files = sorted(base.glob(f"{ASSET}_*.npy"))
    print(f"Fichiers {ASSET}_*.npy trouvés :")
    for f in files:
        print(f"  {f.name} ({f.stat().st_size:,} octets)")

    # Charger chaque fichier
    section("Chargement et shapes")
    ts = np.load(base / f"{ASSET}_ts.npy")
    X = np.load(base / f"{ASSET}_X.npy")
    Y_dir = np.load(base / f"{ASSET}_Y_dir.npy")
    Y_ret = np.load(base / f"{ASSET}_Y_ret.npy")
    Y_hor = np.load(base / f"{ASSET}_Y_hor.npy")

    print(f"ts    : shape={ts.shape}, dtype={ts.dtype}, min={ts.min()}, max={ts.max()}")
    import datetime
    first_dt = datetime.datetime.fromtimestamp(ts[0] / 1000, tz=datetime.timezone.utc)
    last_dt = datetime.datetime.fromtimestamp(ts[-1] / 1000, tz=datetime.timezone.utc)
    print(f"      : première timestamp = {first_dt.isoformat()}")
    print(f"      : dernière timestamp  = {last_dt.isoformat()}")
    print(f"      : durée = {(ts[-1] - ts[0]) / (1000 * 86400):.1f} jours")
    print(f"X     : shape={X.shape}, dtype={X.dtype}")
    print(f"Y_dir : shape={Y_dir.shape}, dtype={Y_dir.dtype}")
    print(f"Y_ret : shape={Y_ret.shape}, dtype={Y_ret.dtype}")
    print(f"Y_hor : shape={Y_hor.shape}, dtype={Y_hor.dtype}")

    # Metadata
    section("metadata.json")
    with open(base / "metadata.json") as f:
        meta = json.load(f)
    print(f"horizons         : {meta['horizons']}")
    print(f"features_count   : {meta['features_count']}")
    print(f"sequence_lengths : {dict(list(meta['sequence_lengths'].items())[:3])} ...")
    print(f"Total assets     : {len(meta['sequence_lengths'])}")
    print(f"Premier asset    : {meta['sequence_lengths'][ASSET]} bougies pour {ASSET}")
    print(f"5 premières feats: {meta['feature_names'][:5]}")
    print(f"5 dernières feats: {meta['feature_names'][-5:]}")

    # Cohérence des shapes
    section("Vérifications de cohérence")
    N = X.shape[0]
    checks = [
        ("ts.shape[0] == X.shape[0]", ts.shape[0] == N),
        ("Y_dir.shape[0] == N", Y_dir.shape[0] == N),
        ("Y_ret.shape[0] == N", Y_ret.shape[0] == N),
        ("Y_hor.shape[0] == N", Y_hor.shape[0] == N),
        ("X.shape[1] == features_count", X.shape[1] == meta["features_count"]),
        ("Y_dir.shape[1] == len(horizons)", Y_dir.shape[1] == len(meta["horizons"])),
        ("Y_ret.shape[1] == len(horizons)", Y_ret.shape[1] == len(meta["horizons"])),
        ("Y_hor.shape[1] == len(horizons)", Y_hor.shape[1] == len(meta["horizons"])),
    ]
    for name, ok in checks:
        print(f"  [{'OK' if ok else 'KO'}] {name}")

    # Distribution Y_dir
    section(f"Distribution Y_dir par horizon (h ∈ [0..3])")
    for h in range(Y_dir.shape[1]):
        u, c = np.unique(Y_dir[:, h], return_counts=True)
        print(f"  Horizon {h} ({meta['horizons'][h]}) : ", end="")
        for val, cnt in zip(u, c):
            pct = cnt / N * 100
            label = {-100: "invalide", 0: "SELL", 1: "HOLD", 2: "BUY"}.get(int(val), "?")
            print(f"{label}={pct:.1f}% ", end="")
        print()

    # Distribution Y_ret (uniquement lignes valides)
    section("Distribution Y_ret par horizon (lignes valides uniquement)")
    for h in range(Y_ret.shape[1]):
        valid = Y_dir[:, h] != -100
        r = Y_ret[valid, h]
        # Approximation: % profitable net de coûts
        costs = 0.0008
        net = r - np.sign(r) * costs
        profitable = (net > 0).mean()
        print(f"  Horizon {h} ({meta['horizons'][h]}) : n={valid.sum():,}, "
              f"mean={r.mean():+.4f}, std={r.std():.4f}, "
              f"min={r.min():+.4f}, max={r.max():+.4f}, "
              f"|ret|>0.005={((np.abs(r) > 0.005).mean()*100):.1f}%, "
              f"net_profitable={profitable*100:.1f}%")

    # Y_hor valeurs
    section("Distribution Y_hor (en bars)")
    for h in range(Y_hor.shape[1]):
        u, c = np.unique(Y_hor[:, h], return_counts=True)
        print(f"  Horizon {h} ({meta['horizons'][h]}) : valeurs uniques = {u.tolist()[:10]}{'...' if len(u) > 10 else ''}")

    # Cohérence Y_hor = nombre de bars attendu
    print()
    print("Vérif cohérence Y_hor (devrait correspondre au nombre de bars):")
    expected_bars = {"5m": 12, "15m": 4, "1h": 6, "4h": 6, "1d": 24}
    for h, hor_str in enumerate(meta["horizons"]):
        # Parse la string "6h" -> minutes -> 6 pour 1h TF
        # Simplifié: on prend le plus fréquent
        u, c = np.unique(Y_hor[:, h], return_counts=True)
        most_common = u[np.argmax(c)]
        print(f"  Horizon {hor_str}: Y_hor dominant = {most_common} bars")

    # NaN dans X
    section("NaN et valeurs infinies dans X")
    n_nan = np.isnan(X).sum()
    n_inf = np.isinf(X).sum()
    print(f"  NaN total : {n_nan:,} ({n_nan / X.size * 100:.2f}%)")
    print(f"  Inf total : {n_inf:,} ({n_inf / X.size * 100:.2f}%)")
    nan_per_col = np.isnan(X).sum(axis=0)
    cols_with_nan = np.where(nan_per_col > 0)[0]
    print(f"  Colonnes avec NaN : {len(cols_with_nan)}/{X.shape[1]}")
    if len(cols_with_nan) <= 5:
        for c in cols_with_nan:
            print(f"    col {c} ({meta['feature_names'][c]}) : {nan_per_col[c]:,} NaN")

    # Taxonomie
    section("Taxonomie features (218 utilisables vs 246 dans X.npy)")
    with open(TAXONOMY_PATH) as f:
        tax = json.load(f)
    print(f"  Total     : {tax['summary']['total']}")
    print(f"  Usables   : {tax['summary']['usable']}")
    print(f"  Excluded  : {tax['summary']['excluded_total']}")
    usable_names = [k for k, v in tax['features'].items() if not v.get('excluded', False)]
    x_names = meta['feature_names']
    in_both = [n for n in x_names if n in usable_names]
    in_x_not_usable = [n for n in x_names if n not in usable_names]
    in_usable_not_in_x = [n for n in usable_names if n not in x_names]
    print(f"  Features X.npy ∩ taxonomie usable : {len(in_both)}")
    print(f"  Features X.npy - usable (à exclure) : {len(in_x_not_usable)}")
    print(f"    {in_x_not_usable[:5]}{'...' if len(in_x_not_usable) > 5 else ''}")
    print(f"  Usable - X.npy (features manquantes) : {len(in_usable_not_in_x)}")

    # OHLCV CSV brut
    section("OHLCV CSV brut")
    ohlcv_dir = OHLCV_DIR / ASSET_CLASS / ASSET / TIMEFRAME
    print(f"  Dossier : {ohlcv_dir}")
    print(f"  Existe ? : {ohlcv_dir.exists()}")
    if ohlcv_dir.exists():
        csvs = sorted(ohlcv_dir.glob(f"{ASSET}_*_{TIMEFRAME}.csv"))
        print(f"  CSV trouvés : {len(csvs)}")
        for c in csvs[:3]:
            print(f"    {c.name} ({c.stat().st_size:,} octets)")
        if len(csvs) > 3:
            print(f"    ... ({len(csvs) - 3} de plus)")

    # Verdict final
    section("VERDICT SANITY CHECK")
    all_ok = all(ok for _, ok in checks)
    if all_ok and n_nan == 0 and n_inf == 0:
        print("  ✅ Les données sont saines pour le pipeline XGBoost.")
        print("  → On peut passer à l'étape 1 (data_loader).")
        return 0
    elif all_ok:
        print(f"  ⚠️  Données cohérentes mais {n_nan:,} NaN et {n_inf:,} Inf dans X.")
        print(f"  → À nettoyer dans data_loader (fillna 0, replace inf).")
        return 0
    else:
        print("  ❌ Problèmes de cohérence détectés, à investiguer.")
        return 1


if __name__ == "__main__":
    sys.exit(main())

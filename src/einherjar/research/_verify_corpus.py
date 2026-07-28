"""
Vérification de l'export corpus — Chantier Option B (PLAN v2).

Lit le corpus.json produit et affiche les Einhers selon la
nouvelle structure conforme au PLAN_COMPLET_V2.md section 2.3.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

CORPUS_PATH = Path(
    r"D:/midas_v2/einherjar/src/einherjar/research/outputs/refactor_e2e/"
    r"refactor_e2e/xauusd__15m/xauusd__15m_corpus.json"
)
SUMMARY_PATH = CORPUS_PATH.with_name(
    CORPUS_PATH.name.replace("_corpus.json", "_summary.json")
)


def main() -> int:
    print("=" * 70)
    print("Verification corpus -- Chantier Option B (PLAN v2)")
    print("=" * 70)
    print(f"Path: {CORPUS_PATH}")
    print()

    if not CORPUS_PATH.exists():
        print(f"ERREUR : fichier introuvable.")
        print(f"Lance d'abord _e2e_refactor.py pour generer l'export.")
        return 1

    with CORPUS_PATH.open("r", encoding="utf-8") as f:
        corpus = json.load(f)

    summary = corpus.get("summary", {})
    rejected_meta = corpus.get("rejected", [])
    entries = corpus.get("entries", [])

    print("RESUME DU CORPUS")
    print("-" * 70)
    print(f"  entry_count    : {summary.get('entry_count', 0)}")
    print(f"  selected_count : {summary.get('selected_count', 0)}")
    print(f"  total_pnl      : {summary.get('total_pnl', 0.0):.4f}")
    print(f"  avg_score      : {summary.get('average_score', 0.0):.4f}")
    print(f"  best_score     : {summary.get('best_score', 0.0):.4f}")
    print(f"  healthy_count  : {summary.get('healthy_count', 0)}")
    print()

    print("ENTRIES ({}):".format(len(entries)))
    print("-" * 70)
    for idx, entry in enumerate(entries):
        is_final = entry.get("selected", False) and entry.get("capital", 0) > 0
        status = "FINAL" if is_final else "REJETE"
        print(f"\n  [{idx}] {status}")
        print(f"      einher_fingerprint : {entry.get('einher_fingerprint', '?')[:16]}...")
        print(f"      asset/timeframe     : {entry.get('asset', '?')}/{entry.get('timeframe', '?')}")
        print(f"      direction           : {entry.get('direction', '?')}")
        print(f"      source_kind         : {entry.get('source_kind', '?')}")

        # Profile
        profile = entry.get("profile", {})
        print(f"      profile             : {profile.get('name', '?')} (family={profile.get('family', '?')})")

        # Conditions
        conditions = entry.get("conditions", [])
        print(f"      conditions          : {len(conditions)} condition(s)")
        for i, cond in enumerate(conditions):
            left = cond.get("left", {})
            op = cond.get("operator", "?")
            right = cond.get("right", {})
            right_str = (
                f"{right.get('name', '?')}"
                if right.get("type") == "feature"
                else f"{right.get('value', '?')}"
            )
            print(f"        [{i}] {left.get('name', '?')} (col={left.get('column_index', '?')}) {op} {right_str}")

        # Edge metrics
        edge = entry.get("edge", {})
        print(f"      edge.win_rate       : {edge.get('win_rate', 0.0):.4f}")
        print(f"      edge.profit_factor  : {edge.get('profit_factor', 0.0):.4f}")
        print(f"      edge.expectancy     : {edge.get('expectancy', 0.0):.6f}")
        print(f"      edge.sharpe         : {edge.get('sharpe_per_trade', 0.0):.4f}")
        print(f"      edge.trade_count    : {edge.get('trade_count', 0)}")
        print(f"      edge.total_pnl      : {edge.get('total_pnl', 0.0):.4f}")

        # Calibration
        cal = entry.get("calibration", {})
        print(f"      calibration.mfe_p50/p75/p90 : "
              f"{cal.get('mfe_p50', 0.0):.4f}/"
              f"{cal.get('mfe_p75', 0.0):.4f}/"
              f"{cal.get('mfe_p90', 0.0):.4f}")
        print(f"      calibration.mae_p50/p75/p90 : "
              f"{cal.get('mae_p50', 0.0):.4f}/"
              f"{cal.get('mae_p75', 0.0):.4f}/"
              f"{cal.get('mae_p90', 0.0):.4f}")
        print(f"      calibration.tp_rule : {cal.get('tp_rule', {})}")
        print(f"      calibration.sl_rule : {cal.get('sl_rule', {})}")

        # Status
        if not entry.get("selected", False):
            reasons = entry.get("rejection_reasons", [])
            print(f"      rejected : {reasons}")
        else:
            print(f"      selected : w={entry.get('weight', 0):.4f}, cap={entry.get('capital', 0):.4f}")

    print()
    print("REJECTED META ({}):".format(len(rejected_meta)))
    print("-" * 70)
    for idx, item in enumerate(rejected_meta):
        print(f"  [{idx}] fp={item.get('subject_fingerprint', '')[:16]}... "
              f"score={item.get('score', 0.0):.4f} "
              f"reasons={item.get('reasons', [])}")

    # Verdict final
    print()
    print("=" * 70)
    if entries:
        # Verification de la conformite au plan
        first = entries[0]
        plan_fields = {
            "subject_fingerprint", "einher_fingerprint", "asset",
            "timeframe", "profile", "conditions", "edge", "calibration",
            "selected", "rejection_reasons",
        }
        present = set(first.keys())
        missing = plan_fields - present
        if missing:
            print(f"ATTENTION : champs plan manquants : {missing}")
        else:
            print(f"OK : structure conforme au PLAN_COMPLET_V2.md section 2.3")

        # Verification qu'on n'a PAS les champs lourds
        bloated = {"mae_mfe", "journal", "diagnostics", "records"}
        bloated_present = bloated & present
        if bloated_present:
            print(f"ATTENTION : champs lourds encore presents : {bloated_present}")
        else:
            print(f"OK : pas de champs lourds (mae_mfe.records, journal, diagnostics)")

        # Taille de l'entry
        import json as _json
        size_bytes = len(_json.dumps(first, default=str))
        print(f"Taille d'un entry : {size_bytes} bytes ({size_bytes/1024:.1f} KB)")
        if size_bytes < 10_000:
            print("OK : taille raisonnable (< 10 KB par Einher)")
        return 0
    else:
        print(f"FAIL : corpus vide.")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

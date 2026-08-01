"""
==========================================================
Discovery — CLI Bootstrap
==========================================================

Ce module est le point d'entrée CLI du pipeline de
découverte. Il ne contient AUCUNE logique métier.

Responsabilités :
- parser les arguments CLI,
- charger la configuration,
- invoquer core.runner.main(),
- afficher le résumé du run.

L'orchestration réelle est dans `core.runner`. Les
algorithmes sont dans les packages `discovery/`,
`validation/`, `execution/`, `portfolio/`.

Pour des raisons de rétro-compatibilité, ce module
ré-exporte les types publics de `core.runner` afin que
les scripts existants (`from discovery import
DiscoveryOrchestrator`) continuent de fonctionner.
"""

from __future__ import annotations

# Ré-exports rétro-compatibles. Voir core/runner.py.
from core.runner import (  # noqa: F401
    DiscoveryOrchestrator,
    DiscoveryRunResult,
    DiscoverySettings,
    main as _runner_main,
)

import argparse
import json
import logging
import sys
from pathlib import Path


__all__ = [
    "DiscoveryOrchestrator",
    "DiscoveryRunResult",
    "DiscoverySettings",
    "main",
]


def main(
    config=None,
    *,
    pairs=None,
    assets=None,
    timeframes=None,
    metadata=None,
):
    """
    Délègue à core.runner.main.
    """

    return _runner_main(
        config=config,
        pairs=pairs,
        assets=assets,
        timeframes=timeframes,
        metadata=metadata,
    )


# ==========================================================
# CLI
# ==========================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="EINHERJAR Discovery — bootstrap du run"
    )
    parser.add_argument(
        "--asset", type=str, default="",
        help="Actif unique (ex: XAUUSD)",
    )
    parser.add_argument(
        "--asset-class", type=str, default="",
        help="Classe d'actif (ex: forex, crypto, commodities)",
    )
    parser.add_argument(
        "--timeframe", type=str, default="15m",
        help="Timeframe (ex: 5m, 15m, 1h, 4h, 1d)",
    )
    parser.add_argument(
        "--debug", action="store_true", help="Logs détaillés",
    )
    args = parser.parse_args()

    if args.debug:
        logging.basicConfig(
            level=logging.DEBUG,
            format="%(asctime)s [%(levelname)-5s] %(message)s",
        )
    else:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)-5s] %(message)s",
        )

    from config.config import Config
    from config.dataset import DatasetConfig

    config = Config()
    config.dataset = DatasetConfig(
        midas_root=r"D:/midas_v2/midasV3/src/data/compiled",
        asset=args.asset,
        asset_class=args.asset_class,
        timeframe=args.timeframe,
    )

    assets_v1_path = Path(r"D:/midas_v2/einherjar/config/assets_v1.json")
    if assets_v1_path.exists():
        try:
            assets_cfg = json.loads(
                assets_v1_path.read_text(encoding="utf-8"),
            )
            assets_list = assets_cfg.get("assets", [])
        except Exception:
            assets_list = []
    else:
        assets_list = []

    if args.asset:
        asset_class = args.asset_class
        if not asset_class:
            for entry in assets_list:
                if entry.get("asset") == args.asset:
                    asset_class = entry.get("class", "")
                    break
        if not asset_class:
            raise ValueError(
                f"Actif {args.asset} non trouvé dans assets_v1.json "
                f"— fournissez --asset-class"
            )
        config.dataset = DatasetConfig(
            midas_root=r"D:/midas_v2/midasV3/src/data/compiled",
            asset=args.asset,
            asset_class=asset_class,
            timeframe=args.timeframe,
        )
        result = main(
            config,
            assets=[args.asset],
            timeframes=[args.timeframe],
        )
    else:
        all_assets = [entry.get("asset") for entry in assets_list if entry.get("asset")]
        result = main(
            config,
            assets=all_assets,
            timeframes=[args.timeframe],
        )

    print("=" * 60)
    print(f"Run ID    : {result.run_id}")
    print(f"Pairs     : {result.pair_count}")
    print(f"Success   : {result.success_count}")
    print(f"Failures  : {result.failure_count}")
    for pr in result.pair_results:
        status = "OK" if pr.success else "FAIL"
        print(f"  [{status}] {pr.pair_key}")

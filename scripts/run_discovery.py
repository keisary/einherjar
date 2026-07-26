"""Lance DiscoveryEngine sur tous les jeux de donnees MIDAS disponibles."""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from einherjar.data.npy_loader import list_available_npy
from einherjar.research.discovery_engine import DiscoveryEngine
from einherjar.research.interaction_miner import InteractionMiner


def main() -> int:
    """Construit un corpus discovery depuis tous les .npy presents."""
    engine = DiscoveryEngine()
    miner = InteractionMiner(engine)
    edges = []
    interactions = []
    datasets = list_available_npy()
    if not datasets:
        print("Aucun dataset MIDAS trouve.")
        return 1

    for index, dataset in enumerate(datasets, start=1):
        asset = dataset["asset"]
        asset_class = dataset["class"]
        timeframe = dataset["timeframe"]
        print(f"[{index}/{len(datasets)}] {asset}/{timeframe}")
        dataset_edges = engine.run(asset, asset_class, timeframe)
        edges.extend(dataset_edges)
        data = engine.load_midas_data(asset, asset_class, timeframe)
        if data is not None:
            x_values, _, returns, _ = data
            interactions.extend(miner.mine(dataset_edges, x_values, returns))

    candidates = engine.build_einhers(edges)
    output_dir = PROJECT_ROOT / "data" / "discovery"
    engine.save_results(edges, candidates, output_dir)
    interactions_path = output_dir / "interactions.json"
    interactions_path.write_text(
        json.dumps([item.to_dict() for item in interactions], indent=2),
        encoding="utf-8",
    )
    print(f"{len(edges)} edges analyses")
    print(f"{len(candidates)} Einhers construits")
    print(f"{len(interactions)} interactions valides")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

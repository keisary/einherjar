"""asset_selection.py - Sélection des actifs depuis assets_v1.json.

Problème 1 FIX (2026-08-21) : le runner scanne désormais les 28 actifs
EXACTS de config/assets_v1.json avec leur classe réelle, au lieu de
prendre les 3 premiers par ordre alphabétique de chaque classe (ce qui
causait des "skip complet" et ratait les actifs de la sélection).

API publique :
- ASSETS_CONFIG_PATH : chemin du fichier de sélection
- load_asset_selection(path=None) -> list[AssetSpec]  (28 actifs)
- resolve_compiled_class(asset_class, asset) -> str    (sous-classe réelle)
- available_timeframes(asset, asset_class, compiled_dir) -> list[str]
- horizons_for(asset, asset_class, tf, compiled_dir) -> list[str]
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

from .data_loader import COMPILED_DIR
from .paths import ASSETS_CONFIG_PATH

logger = logging.getLogger(__name__)

# (chemin centralise dans paths.py)


# Les 3 sous-classes stocks -> on les garde telles quelles (ce sont des
# dossiers compilés distincts). OHLCV partage 'stocks' (voir multi_asset_loader).
STOCK_SUBCLASSES = {"stocks_growth", "stocks_tech", "stocks_value"}

# TF disponibles globalement (toutes classes ont ces dossiers compiled)
GLOBAL_TIMEFRAMES = ["5m", "15m", "1h", "4h", "1d"]


@dataclass(frozen=True)
class AssetSpec:
    """Un actif sélectionné (depuis assets_v1.json)."""
    asset: str
    asset_class: str
    broker: str
    name: str

    @property
    def compiled_class(self) -> str:
        """Classe réelle dans les dossiers compiled."""
        return resolve_compiled_class(self.asset_class, self.asset)


def load_asset_selection(path: Path | str | None = None) -> list[AssetSpec]:
    """Charge les actifs de assets_v1.json.

    Returns:
        Liste ordonnée (ordre du fichier) de 28 AssetSpec.
    """
    p = Path(path) if path else ASSETS_CONFIG_PATH
    with open(p, encoding="utf-8") as f:
        cfg = json.load(f)
    specs = []
    for a in cfg["assets"]:
        specs.append(AssetSpec(
            asset=a["asset"],
            asset_class=a["class"],
            broker=a.get("broker", ""),
            name=a.get("name", a["asset"]),
        ))
    logger.info("load_asset_selection(%s) : %d actifs", p.name, len(specs))
    return specs


def resolve_compiled_class(asset_class: str, asset: str) -> str:
    """Retourne la classe compiled réelle pour un (asset_class, asset).

    - Les classes non-stocks se mappent 1:1 (crypto, forex, indices, commodities).
    - Les classes 'stocks_*' sont déjà des dossiers compiled distincts.
    - Fallback robuste : si 'stocks_us' (nom générique du fichier) ou si la
      classe ne résout pas, on cherche la vraie sous-classe où l'actif est compilé.
    """
    if asset_class in STOCK_SUBCLASSES:
        return asset_class
    # Cas générique 'stocks_us' (défensif) : trouver la sous-classe réelle
    if asset_class in ("stocks_us", "stocks"):
        for sc in sorted(STOCK_SUBCLASSES):
            if (Path(COMPILED_DIR) / sc / "1h" / f"{asset}_X.npy").exists():
                return sc
        # Fallback par défaut si introuvable
        logger.warning(
            "resolve_compiled_class : actif %s non trouvé dans les sous-classes stocks",
            asset,
        )
        return "stocks_tech"
    # Toutes les autres classes : 1:1
    return asset_class


def available_timeframes(
    asset: str,
    asset_class: str,
    compiled_dir: Path = COMPILED_DIR,
) -> list[str]:
    """Liste les TF réellement disponibles (compilés) pour un actif.

    Les cryptos n'ONT PAS de TF 1d (seulement 5m/15m/1h/4h) → retourne
    uniquement les TF qui existent pour éviter tout skip/fail silencieux.

    Returns:
        Liste ordonnée des TF (ordre canonique standard) où l'actif est compilé.
    """
    ccls = resolve_compiled_class(asset_class, asset)
    base = Path(compiled_dir) / ccls
    available = []
    for tf in GLOBAL_TIMEFRAMES:
        if (base / tf / f"{asset}_X.npy").exists():
            available.append(tf)
    if not available:
        logger.warning("available_timeframes : aucun TF pour %s (%s)", asset, asset_class)
    return available


def horizons_for(
    asset: str,
    asset_class: str,
    tf: str,
    compiled_dir: Path = COMPILED_DIR,
) -> list[str]:
    """Lit les horizons d'un TF depuis metadata.json du dossier compiled.

    Critique (correctif Jovanny, 2026-08-21) : les horizons SONT DIFFÉRENTS
    selon le TF. On les lit depuis le data réel, on ne les hardcode pas :
        5m  -> 15m, 30m, 1h, 2h
        15m -> 1h, 2h, 4h, 8h
        1h  -> 6h, 12h, 1d, 2d
        4h  -> 1d, 2d, 4d, 10d
        1d  -> 5d, 10d, 20d, 60d

    Returns:
        Liste des noms d'horizons (ordre du fichier) pour ce TF.
    """
    ccls = resolve_compiled_class(asset_class, asset)
    meta_path = Path(compiled_dir) / ccls / tf / "metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"metadata.json absent : {meta_path}")
    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)
    horizons = list(meta.get("horizons", []))
    if not horizons:
        raise ValueError(f"metadata.json sans horizons : {meta_path}")
    return horizons

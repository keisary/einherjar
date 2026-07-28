"""
==========================================================
Asset Resolver
==========================================================

Résout la classe d'un actif (forex, crypto, ...) depuis le
fichier de configuration `assets_v1.json`.

La résolution est mise en cache au niveau du module.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Final

logger = logging.getLogger("einherjar.assets")

_DEFAULT_ASSETS_PATH: Final[Path] = Path(
    r"D:/midas_v2/einherjar/config/assets_v1.json"
)

_cache: dict[str, dict[str, str]] = {}
_loaded: bool = False


def _load_once(path: Path) -> dict[str, dict[str, str]]:
    global _loaded
    if _loaded:
        return _cache

    if not path.exists():
        logger.warning("Asset config introuvable : %s", path)
        _loaded = True
        return _cache

    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        for entry in data.get("assets", []):
            asset = str(entry.get("asset", "")).strip()
            asset_class = str(entry.get("class", "")).strip()
            if asset and asset_class:
                _cache[asset] = {
                    "class": asset_class,
                    "broker": str(entry.get("broker", "")),
                    "name": str(entry.get("name", "")),
                }
    except Exception as exc:
        logger.warning("Impossible de lire %s : %s", path, exc)

    _loaded = True
    return _cache


def resolve_asset_class(
    asset: str,
    *,
    assets_path: Path | None = None,
) -> str:
    """
    Retourne la classe d'un actif.

    Lève KeyError si l'actif n'est pas connu.
    """

    path = Path(assets_path) if assets_path is not None else _DEFAULT_ASSETS_PATH
    table = _load_once(path)
    if asset not in table:
        raise KeyError(f"Asset '{asset}' not found in {path}.")
    return table[asset]["class"]


def resolve_asset_meta(
    asset: str,
    *,
    assets_path: Path | None = None,
) -> dict[str, str]:
    """
    Retourne toutes les métadonnées associées à un actif.

    Lève KeyError si l'actif n'est pas connu.
    """

    path = Path(assets_path) if assets_path is not None else _DEFAULT_ASSETS_PATH
    table = _load_once(path)
    if asset not in table:
        raise KeyError(f"Asset '{asset}' not found in {path}.")
    return dict(table[asset])


def known_assets(
    *,
    assets_path: Path | None = None,
) -> tuple[str, ...]:
    """
    Liste tous les actifs connus.
    """

    path = Path(assets_path) if assets_path is not None else _DEFAULT_ASSETS_PATH
    table = _load_once(path)
    return tuple(sorted(table.keys()))


def reset_cache() -> None:
    """
    Réinitialise le cache (utile pour les tests).
    """

    global _loaded
    _cache.clear()
    _loaded = False

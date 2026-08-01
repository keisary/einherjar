"""
data/versioning.py — Versioning des datasets (data_version).

Chaque "version" est un identifiant stable (tag ou hash) attaché à un
bundle OHLCV + features + coûts. Permet de :
  - Archiver une hypothèse avec son contexte reproductible
  - Réévaluer une hypothèse rejetée sur un nouveau data_version
    (sans la considérer comme un doublon)
  - Auditer "quelles données ont produit cet Einher"

Le hash d'un data_version est calculé sur :
  - Les chemins / noms de fichiers OHLCV et features
  - Les bornes temporelles effectives (start_ts, end_ts)
  - Les coûts simulés (costs.yaml)
  - La version de l'ontologie (pour traçabilité sémantique)
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from einherjar.research.config.loader import EinherjarConfig

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DataVersion:
    """Identifiant d'une version de données."""

    tag: str                       # ex: "v1_2026-08-01" ou "hash:a1b2c3d4..."
    hash: str                      # sha256 du manifest
    manifest: dict                 # contenu complet du manifest
    created_at: str                # ISO 8601 UTC

    def to_dict(self) -> dict:
        return {
            "tag": self.tag,
            "hash": self.hash,
            "manifest": self.manifest,
            "created_at": self.created_at,
        }


def make_data_version(
    ohlcv_paths: dict[str, Path],       # { "BTCUSD_1h": Path, ... }
    features_paths: dict[str, Path],    # idem
    config: EinherjarConfig,
    *,
    tag: Optional[str] = None,
) -> DataVersion:
    """Crée un DataVersion à partir des chemins de données et de la config.

    Le hash est calculé sur les chemins + les coûts + les seuils (pour
    qu'un changement de config qui affecte le calcul rende le hash
    différent — sinon on aurait un faux sentiment de reproductibilité).
    """
    manifest = {
        "ohlcv_paths": {k: str(v.resolve()) for k, v in sorted(ohlcv_paths.items())},
        "features_paths": {k: str(v.resolve()) for k, v in sorted(features_paths.items())},
        "costs": config.costs,
        "thresholds_hash": _hash_dict(config.thresholds),
        "evaluation_hash": _hash_dict(config.evaluation),
    }
    h = _hash_dict(manifest)
    final_tag = tag or f"hash:{h[:12]}"
    return DataVersion(
        tag=final_tag,
        hash=h,
        manifest=manifest,
        created_at=datetime.now(timezone.utc).isoformat(),
    )


def make_splits_hash(
    train_start: int, train_end: int,
    val_start: int, val_end: int,
    holdout_start: int, holdout_end: int,
    embargo_bougies: int,
    horizon_label: int,
) -> str:
    """Calcule un hash stable des bornes de splits (pour traçabilité Archive)."""
    payload = {
        "train": [train_start, train_end],
        "val": [val_start, val_end],
        "holdout": [holdout_start, holdout_end],
        "embargo_bougies": embargo_bougies,
        "horizon_label": horizon_label,
    }
    return _hash_dict(payload)


def _hash_dict(d: dict) -> str:
    """Hash stable d'un dict (clés triées)."""
    payload = json.dumps(d, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()

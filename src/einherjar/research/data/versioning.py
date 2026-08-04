"""
data/versioning.py — Versioning des datasets (data_version, P1 #9).

Chaque "version" est un identifiant stable (tag ou hash) attaché à un
bundle OHLCV + features + coûts. Permet de :
  - Archiver une hypothèse avec son contexte reproductible
  - Réévaluer une hypothèse rejetée sur un nouveau data_version
    (sans la considérer comme un doublon)
  - Auditer "quelles données ont produit cet Einher"

P1 #9 : le manifest capture TOUT ce qui définit la version :
  - Schéma (format, dtype, n_columns)
  - Hash du contenu (sha256 des fichiers .npy) — un changement de données
    passe par un changement de hash
  - Actif, timeframe, période (start_ts/end_ts), timezone
  - Règles de nettoyage appliquées (sanitization, drop NaN, etc.)
  - Coûts simulés (costs.yaml) et seuils (pour qu'un changement de config
    qui affecte le calcul rende le hash différent)
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone as dt_timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np

from einherjar.research.config.loader import EinherjarConfig

logger = logging.getLogger(__name__)


# Schéma et règles de nettoyage par défaut (P1 #9 — explicites).
DEFAULT_TIMEZONE: str = "UTC"
DEFAULT_CLEANING_RULES: dict[str, Any] = {
    "sanitize": "drop NaN/inf sur OHLCV, low<=high, open/close dans [low, high]",
    "validity_mask": "compute_validity_mask (npy_real_loader)",
    "outlier_handling": "none (raw log-returns conservés)",
    "gap_handling": "none (gaps déclarés en warning, pas de fill)",
}


@dataclass(frozen=True)
class DataVersion:
    """Identifiant d'une version de données (P1 #9 complet)."""

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


def _inspect_npy_file(path: Path) -> dict[str, Any]:
    """Inspecte un fichier .npy : shape, dtype, n_bougies, start/end_ts (si dispo).

    Returns:
        Dict {format, dtype, n_columns, n_bougies, content_sha256, ...}.
    """
    import hashlib
    if not path.exists():
        return {"format": "npy", "path": str(path), "exists": False}
    # Hash du contenu.
    h = hashlib.sha256()
    with path.open("rb") as fp:
        for chunk in iter(lambda: fp.read(1024 * 1024), b""):
            h.update(chunk)
    content_sha = h.hexdigest()
    # Inspection du shape/dtype.
    arr = np.load(path, mmap_mode="r")
    shape = list(arr.shape)
    dtype = str(arr.dtype)
    n_bougies = shape[0] if arr.ndim >= 1 else 0
    info: dict[str, Any] = {
        "format": "npy",
        "path": str(path.resolve()),
        "content_sha256": content_sha,
        "dtype": dtype,
        "shape": shape,
        "n_bougies": n_bougies,
    }
    # Si c'est un fichier timestamps (X.npy à 1 col, ou _ts.npy), extraire min/max.
    if arr.ndim == 1 and n_bougies > 0:
        info["start_ts_ms"] = int(arr[0])
        info["end_ts_ms"] = int(arr[-1])
    return info


def make_data_version(
    ohlcv_paths: dict[str, Path],       # { "BTCUSD_1h": Path, ... }
    features_paths: dict[str, Path],    # idem
    config: EinherjarConfig,
    *,
    tag: Optional[str] = None,
    tz_label: str = DEFAULT_TIMEZONE,
    cleaning_rules: Optional[dict[str, Any]] = None,
) -> DataVersion:
    """Crée un DataVersion à partir des chemins de données et de la config.

    P1 #9 : le manifest capture schéma, hash du contenu, période, timezone,
    règles de nettoyage. Le hash final est calculé sur le manifest complet,
    ce qui garantit qu'une modification de n'importe quel élément change
    la data_version (anti-faux sentiment de reproductibilité).
    """
    cleaning_rules = cleaning_rules or DEFAULT_CLEANING_RULES
    ohlcv_inspect: dict[str, Any] = {
        k: _inspect_npy_file(p) for k, p in sorted(ohlcv_paths.items())
    }
    features_inspect: dict[str, Any] = {
        k: _inspect_npy_file(p) for k, p in sorted(features_paths.items())
    }
    manifest = {
        "schema": {
            "ohlcv": ohlcv_inspect,
            "features": features_inspect,
        },
        "timezone": tz_label,
        "cleaning_rules": cleaning_rules,
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
        created_at=datetime.now(dt_timezone.utc).isoformat(),
    )


# --------------------------------------------------------------------------- #
# Persistance des DataVersion (P0-03 — verrou reproductible)
# --------------------------------------------------------------------------- #


class DataVersionStore:
    """Persistance append-only des DataVersion dans un fichier JSONL.

    Permet de :
      - Persister chaque data_version generee pendant un run
      - Verifier qu'un data_version donne a deja ete produit (sinon erreur)
      - Auditer l'historique des data_version

    Format du fichier : 1 ligne JSON par DataVersion. Append-only.
    """

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, dv: DataVersion) -> None:
        """Append une DataVersion au store (avec fsync)."""
        with self.path.open("a", encoding="utf-8") as fp:
            fp.write(json.dumps(dv.to_dict()) + "\n")
            fp.flush()
            import os
            os.fsync(fp.fileno())

    def find_by_tag(self, tag: str) -> Optional[DataVersion]:
        """Cherche une DataVersion par tag. Retourne None si absent."""
        if not self.path.exists():
            return None
        with self.path.open("r", encoding="utf-8") as fp:
            for line in fp:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if d.get("tag") == tag:
                    return DataVersion(
                        tag=d["tag"],
                        hash=d["hash"],
                        manifest=d["manifest"],
                        created_at=d["created_at"],
                    )
        return None

    def find_by_hash(self, content_hash: str) -> Optional[DataVersion]:
        """Cherche une DataVersion par hash du manifest. Retourne None si absent."""
        if not self.path.exists():
            return None
        with self.path.open("r", encoding="utf-8") as fp:
            for line in fp:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if d.get("hash") == content_hash:
                    return DataVersion(
                        tag=d["tag"],
                        hash=d["hash"],
                        manifest=d["manifest"],
                        created_at=d["created_at"],
                    )
        return None

    def all_tags(self) -> list[str]:
        """Liste tous les tags persistes."""
        if not self.path.exists():
            return []
        tags: list[str] = []
        with self.path.open("r", encoding="utf-8") as fp:
            for line in fp:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                    if "tag" in d:
                        tags.append(d["tag"])
                except json.JSONDecodeError:
                    continue
        return tags


def verify_data_version_locked(
    dv: DataVersion,
    store: DataVersionStore,
) -> DataVersion:
    """Verifie que le data_version est verrouille (deja produit) ou le cree.

    Procedure P0-03 :
      1. Si un data_version avec le meme tag existe deja dans le store : OK
      2. Si un data_version avec le meme hash existe : OK (meme contenu,
         tag peut differer)
      3. Sinon : on append le nouveau data_version au store

    Returns:
        Le DataVersion verrouille (soit l'existant, soit le nouveau).
    """
    existing = store.find_by_tag(dv.tag)
    if existing is not None:
        if existing.hash != dv.hash:
            raise ValueError(
                f"DataVersion tag={dv.tag!r} existe mais avec un hash "
                f"different (existant={existing.hash[:12]}, nouveau="
                f"{dv.hash[:12]}). Verifier que les donnees et la config "
                f"n'ont pas change depuis le premier run."
            )
        return existing
    existing = store.find_by_hash(dv.hash)
    if existing is not None:
        return existing
    store.append(dv)
    return dv


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


def make_frame_data_version(
    ohlcv: Any,
    features: Any,
    config: EinherjarConfig,
    *,
    tag: Optional[str] = None,
) -> DataVersion:
    """Construit une version reproductible Ã  partir des frames effectivement utilisÃ©es.

    Les features MIDAS et les prix bruts ne proviennent pas nÃ©cessairement du
    mÃªme stockage. Hacher les fichiers ``X.npy`` seulement serait donc
    insuffisant : ce manifest inclut les deux flux, colonne par colonne, aprÃ¨s
    alignement et nettoyage. Il est volontairement calculÃ© avant tout split.
    """
    def _frame_digest(frame: Any) -> dict[str, Any]:
        df = frame.df
        digest = hashlib.sha256()
        for name in df.columns:
            digest.update(name.encode("utf-8"))
            for value in df[name].to_list():
                digest.update(str(value).encode("utf-8"))
                digest.update(b"\0")
        return {
            "asset": frame.asset,
            "timeframe": frame.timeframe,
            "n_rows": df.height,
            "columns": list(df.columns),
            "content_sha256": digest.hexdigest(),
        }

    manifest = {
        "schema": {
            "ohlcv": _frame_digest(ohlcv),
            "features": _frame_digest(features),
        },
        "timezone": DEFAULT_TIMEZONE,
        "cleaning_rules": {
            "ohlcv": "raw price OHLCV aligned by timestamp with MIDAS features",
            "features": "MIDAS normalized features only; never used as execution prices",
        },
        "costs": config.costs,
        "thresholds_hash": _hash_dict(config.thresholds),
        "evaluation_hash": _hash_dict(config.evaluation),
    }
    digest = _hash_dict(manifest)
    return DataVersion(
        tag=tag or f"hash:{digest[:12]}",
        hash=digest,
        manifest=manifest,
        created_at=datetime.now(dt_timezone.utc).isoformat(),
    )


def _hash_dict(d: dict) -> str:
    """Hash stable d'un dict (clés triées)."""
    payload = json.dumps(d, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()

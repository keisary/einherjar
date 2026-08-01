"""config/loader.py — Chargement + validation des fichiers de configuration.

Centralise la lecture de :
  - features_taxonomy.json (taxonomie 218 + 28 exclues)
  - thresholds.yaml        (S-3.4 : DSR, PBO, bootstrap CI, n_trades, etc.)
  - splits.yaml            (train/val/holdout + purging + embargo)
  - costs.yaml             (spread, commission, slippage)
  - evaluation.yaml        (ATR, N, simulation intrabar, bootstrap)

Toutes les valeurs sont overridables par hypothèse / par run, mais les défauts
proviennent de ces fichiers.

Le loader ne fait PAS d'I/O bloquante, n'altère pas les fichiers, et est
strictement lecture + validation de cohérence.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Exceptions
# --------------------------------------------------------------------------- #


class ConfigError(Exception):
    """Erreur de configuration (fichier manquant, clé absente, valeur invalide)."""


class ConfigCoherenceError(ConfigError):
    """Erreur de cohérence entre plusieurs fichiers de config."""


# --------------------------------------------------------------------------- #
# Structure de la config chargée
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class EinherjarConfig:
    """Snapshot immutable de toute la configuration chargée."""

    config_dir: Path
    features_taxonomy: dict[str, Any]
    thresholds: dict[str, Any]
    splits: dict[str, Any]
    costs: dict[str, Any]
    evaluation: dict[str, Any]

    # Listes dérivées de la taxonomie (calculées une fois)
    usable_feature_names: tuple[str, ...] = field(default_factory=tuple)
    excluded_feature_names: tuple[str, ...] = field(default_factory=tuple)

    def usable_set(self) -> set[str]:
        """Set des features utilisables (rapide pour `in` checks)."""
        return set(self.usable_feature_names)

    def excluded_set(self) -> set[str]:
        """Set des features exclues."""
        return set(self.excluded_feature_names)

    def is_usable(self, feature_name: str) -> bool:
        """Vérifie qu'une feature est utilisable (non fantôme, non meta, non alias)."""
        return feature_name in self.usable_feature_names


# --------------------------------------------------------------------------- #
# Loader principal
# --------------------------------------------------------------------------- #


def load_config(config_dir: str | Path) -> EinherjarConfig:
    """Charge tous les fichiers de config depuis `config_dir`.

    Effectue les vérifications de cohérence minimales :
      - Tous les fichiers requis existent.
      - Les ratios train + val + holdout somment à 1.0.
      - La taxonomie annonce 218 features utilisables (cf. invariants du projet).
      - Les seuils DSR/PBO/CI sont des valeurs numériques bornées.
    """
    config_dir = Path(config_dir)
    if not config_dir.exists():
        raise ConfigError(f"Config directory introuvable : {config_dir}")

    taxonomy = _load_json(
        config_dir / "features_taxonomy.json",
        required_keys=("summary", "features", "excluded", "usable_feature_names"),
    )
    thresholds = _load_yaml(
        config_dir / "thresholds.yaml",
        required_keys=(
            "dsr",
            "pbo",
            "bootstrap",
            "n_trades",
            "cross_asset",
            "max_drawdown",
            "diversity",
        ),
    )
    splits = _load_yaml(
        config_dir / "splits.yaml",
        required_keys=("ratios", "purging", "embargo", "locking", "mode"),
    )
    costs = _load_yaml(config_dir / "costs.yaml", required_keys=("default", "convention"))
    evaluation = _load_yaml(
        config_dir / "evaluation.yaml",
        required_keys=("atr", "n_window", "simulation", "bootstrap", "fingerprint"),
    )

    # ---- Vérifs de cohérence ----
    _check_split_ratios(splits)
    _check_taxonomy_218(taxonomy)
    _check_thresholds_bounded(thresholds)
    _check_bootstrap_consistency(thresholds, evaluation)
    _check_costs_sane(costs)

    usable = tuple(taxonomy["usable_feature_names"])
    excluded = tuple(taxonomy["excluded"].keys())

    logger.info(
        "Config chargée depuis %s : %d features utilisables, %d exclues (fantômes=%d, meta=%d, alias=%d)",  # noqa: E501
        config_dir,
        len(usable),
        len(excluded),
        taxonomy["summary"]["excluded_breakdown"].get("ghost", 0),
        taxonomy["summary"]["excluded_breakdown"].get("meta_factor", 0),
        taxonomy["summary"]["excluded_breakdown"].get("alias", 0),
    )

    return EinherjarConfig(
        config_dir=config_dir,
        features_taxonomy=taxonomy,
        thresholds=thresholds,
        splits=splits,
        costs=costs,
        evaluation=evaluation,
        usable_feature_names=usable,
        excluded_feature_names=excluded,
    )


# --------------------------------------------------------------------------- #
# Helpers internes (I/O + validation)
# --------------------------------------------------------------------------- #


def _load_json(path: Path, required_keys: tuple[str, ...] = ()) -> dict[str, Any]:
    if not path.exists():
        raise ConfigError(f"Fichier manquant : {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ConfigError(f"JSON invalide dans {path}: {exc}") from exc
    for key in required_keys:
        if key not in data:
            raise ConfigError(f"Cl '{key}' manquante dans {path}")
    return data


def _load_yaml(path: Path, required_keys: tuple[str, ...] = ()) -> dict[str, Any]:
    if not path.exists():
        raise ConfigError(f"Fichier manquant : {path}")
    try:
        import yaml  # type: ignore
    except ImportError as exc:
        raise ConfigError("PyYAML requis pour lire les .yaml — `pip install pyyaml`") from exc
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"YAML invalide dans {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ConfigError(f"{path} doit contenir un mapping YAML racine")
    for key in required_keys:
        if key not in data:
            raise ConfigError(f"Clé '{key}' manquante dans {path}")
    return data


def _check_split_ratios(splits: dict[str, Any]) -> None:
    r = splits["ratios"]
    s = r["train"] + r["val"] + r["holdout"]
    if abs(s - 1.0) > 1e-9:
        raise ConfigCoherenceError(
            f"splits.ratios doit sommer à 1.0, got {s} (train={r['train']}, val={r['val']}, holdout={r['holdout']})"  # noqa: E501
        )
    for k in ("train", "val", "holdout"):
        if not (0.0 < r[k] < 1.0):
            raise ConfigCoherenceError(f"splits.ratios.{k} doit être dans ]0,1[, got {r[k]}")


def _check_taxonomy_218(taxonomy: dict[str, Any]) -> None:
    s = taxonomy["summary"]
    if s["usable"] != 218:
        raise ConfigCoherenceError(
            f"Taxonomie annonce {s['usable']} features utilisables, attendu 218. "
            "Voir ONTOLOGY.md — invariant de référence."
        )
    if s["excluded_total"] != 28:
        raise ConfigCoherenceError(
            f"Taxonomie annonce {s['excluded_total']} exclues, attendu 28 (19 fantômes + 8 meta + 1 alias)."  # noqa: E501
        )


def _check_thresholds_bounded(thresholds: dict[str, Any]) -> None:
    # DSR doit être dans [-1, 3] (interprétation probabiliste)
    dsr_min = thresholds["dsr"]["min_value"]
    if not (-1.0 <= dsr_min <= 3.0):
        raise ConfigCoherenceError(f"thresholds.dsr.min_value hors borne: {dsr_min}")
    # PBO max doit être dans [0, 1]
    pbo_max = thresholds["pbo"]["max_value"]
    if not (0.0 <= pbo_max <= 1.0):
        raise ConfigCoherenceError(f"thresholds.pbo.max_value hors borne: {pbo_max}")
    # MDD max doit être dans [0, 1]
    mdd = thresholds["max_drawdown"]["max_value"]
    if not (0.0 < mdd < 1.0):
        raise ConfigCoherenceError(f"thresholds.max_drawdown.max_value hors borne: {mdd}")


def _check_bootstrap_consistency(thresholds: dict[str, Any], evaluation: dict[str, Any]) -> None:
    t_ci = thresholds["bootstrap"]["ci_level"]
    e_ci = evaluation["bootstrap"]["ci_level"]
    if abs(t_ci - e_ci) > 1e-9:
        raise ConfigCoherenceError(
            f"CI bootstrap incoherent : thresholds={t_ci}, evaluation={e_ci}"
        )


def _check_costs_sane(costs: dict[str, Any]) -> None:
    d = costs["default"]
    for key in ("spread_pct", "commission_pct", "slippage_pct"):
        v = d[key]
        if not (0.0 <= v <= 0.01):  # 1% = borne max absurde
            raise ConfigCoherenceError(f"costs.default.{key} hors borne raisonnable: {v}")

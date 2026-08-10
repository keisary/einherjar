"""corpus/store.py — Persistance du corpus d'Einhers admis (P1 #7).

Le corpus est l'ensemble des Einhers validés et actifs. Il DOIT être persistant
(entre redémarrages) pour garantir :
  - La diversité inter-redémarrage (un nouveau run voit les Einhers des runs passés).
  - La déduplication par fingerprint (un nouvel Einher avec même signature est
    détecté comme doublon).
  - Le calcul de diversité par corrélation (P1 #5 : nécessite ret_series par Einher).
  - Les quotas de diversité (P1 #6 : on connaît la composition courante du corpus).

Format : JSON Lines (un Einher par ligne). Append-only.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone as dt_timezone
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


# Ancrage absolu : indépendant du CWD de lancement.
# Path(__file__) = .../src/einherjar/research/corpus/store.py
# parents[4] = racine du repo (D:\midas_v2\einherjar) — même convention que
# le holdout ledger, corrigée en parents[4] pour la vraie racine.
DEFAULT_CORPUS_PATH: Path = (
    Path(__file__).resolve().parents[4] / "outputs" / "corpus.jsonl"
)


# --------------------------------------------------------------------------- #
# Entrée du corpus
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class CorpusEntry:
    """Une entrée du corpus = un Einher admis et actif.

    Attributes:
        id: ID unique (ex: 'einh_001').
        hypothesis: Sérialisation complète de l'Hypothesis (to_dict).
        direction: 'long' | 'short'.
        universe: (assets, timeframes).
        amplitude: valeur + unité.
        sl_n_atr, tp_n_atr, sl_distance, tp_distance, n_window: paramètres figés depuis train.
        fingerprint_structurel, fingerprint_comportemental: signatures canoniques.
        metrics_val: MesuresBrutes sérialisées (Sharpe, DD, etc.).
        sharpe_val: Sharpe sur val (raccourci).
        bootstrap_sharpe_ci_low/high_val: IC bootstrap val.
        deflated_sharpe_ratio: DSR.
        probability_of_backtest_overfitting: PBO.
        ret_series: tuple de rendements nets par trade (pour corrélation P1 #5).
        data_version: version de données.
        seed: seed RNG.
        splits_hash: hash des bornes train/val/holdout.
        admission_timestamp: ISO 8601 UTC.
        statut: 'validé' | 'actif' | 'dégradé' | 'archivé'.
        meta: dict libre.
    """

    id: str
    hypothesis: dict[str, Any]
    direction: str
    universe: dict[str, tuple[str, ...]]
    amplitude: dict[str, Any]
    sl_n_atr: float
    tp_n_atr: float
    sl_distance: float
    tp_distance: float
    n_window: int
    fingerprint_structurel: str
    fingerprint_comportemental: str
    metrics_val: dict[str, Any] = field(default_factory=dict)
    sharpe_val: float = 0.0
    bootstrap_sharpe_ci_low_val: float = 0.0
    bootstrap_sharpe_ci_high_val: float = 0.0
    deflated_sharpe_ratio: float = 0.0
    probability_of_backtest_overfitting: float = 0.0
    ret_series: tuple[float, ...] = ()
    data_version: str = ""
    seed: int = 42
    splits_hash: str = ""
    admission_timestamp: str = ""
    statut: str = "validé"
    meta: dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def now_utc() -> str:
        return datetime.now(dt_timezone.utc).isoformat()

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "hypothesis": self.hypothesis,
            "direction": self.direction,
            "universe": {k: list(v) for k, v in self.universe.items()},
            "amplitude": self.amplitude,
            "sl_n_atr": self.sl_n_atr,
            "tp_n_atr": self.tp_n_atr,
            "sl_distance": self.sl_distance,
            "tp_distance": self.tp_distance,
            "n_window": self.n_window,
            "fingerprint_structurel": self.fingerprint_structurel,
            "fingerprint_comportemental": self.fingerprint_comportemental,
            "metrics_val": self.metrics_val,
            "sharpe_val": self.sharpe_val,
            "bootstrap_sharpe_ci_low_val": self.bootstrap_sharpe_ci_low_val,
            "bootstrap_sharpe_ci_high_val": self.bootstrap_sharpe_ci_high_val,
            "deflated_sharpe_ratio": self.deflated_sharpe_ratio,
            "probability_of_backtest_overfitting": self.probability_of_backtest_overfitting,
            "ret_series": list(self.ret_series),
            "data_version": self.data_version,
            "seed": self.seed,
            "splits_hash": self.splits_hash,
            "admission_timestamp": self.admission_timestamp,
            "statut": self.statut,
            "meta": self.meta,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "CorpusEntry":
        return cls(
            id=d["id"],
            hypothesis=d["hypothesis"],
            direction=d["direction"],
            universe={k: tuple(v) for k, v in d.get("universe", {}).items()},
            amplitude=d.get("amplitude", {}),
            sl_n_atr=float(d.get("sl_n_atr", 0.0)),
            tp_n_atr=float(d.get("tp_n_atr", 0.0)),
            sl_distance=float(d.get("sl_distance", 0.0)),
            tp_distance=float(d.get("tp_distance", 0.0)),
            n_window=int(d.get("n_window", 0)),
            fingerprint_structurel=d.get("fingerprint_structurel", ""),
            fingerprint_comportemental=d.get("fingerprint_comportemental", ""),
            metrics_val=dict(d.get("metrics_val", {})),
            sharpe_val=float(d.get("sharpe_val", 0.0)),
            bootstrap_sharpe_ci_low_val=float(d.get("bootstrap_sharpe_ci_low_val", 0.0)),
            bootstrap_sharpe_ci_high_val=float(d.get("bootstrap_sharpe_ci_high_val", 0.0)),
            deflated_sharpe_ratio=float(d.get("deflated_sharpe_ratio", 0.0)),
            probability_of_backtest_overfitting=float(d.get("probability_of_backtest_overfitting", 0.0)),
            ret_series=tuple(d.get("ret_series", ())),
            data_version=d.get("data_version", ""),
            seed=int(d.get("seed", 42)),
            splits_hash=d.get("splits_hash", ""),
            admission_timestamp=d.get("admission_timestamp", ""),
            statut=d.get("statut", "validé"),
            meta=dict(d.get("meta", {})),
        )


# --------------------------------------------------------------------------- #
# Store persistant
# --------------------------------------------------------------------------- #


class CorpusStore:
    """Persistance du corpus d'Einhers admis (append-only JSONL).

    Usage :
        store = CorpusStore()
        store.append(entry)            # ajoute un Einher
        entries = store.load()         # recharge tous les Einhers
        stats = store.summary()        # nb total, par direction, etc.
    """

    def __init__(self, path: Optional[Path] = None) -> None:
        self.path: Path = path or DEFAULT_CORPUS_PATH
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.path.touch()
        logger.info("CorpusStore initialisé : %s", self.path)

    def append(self, entry: CorpusEntry) -> None:
        """Append atomique d'un Einher au corpus."""
        line = json.dumps(entry.to_dict(), ensure_ascii=False) + "\n"
        with self.path.open("a", encoding="utf-8") as fp:
            fp.write(line)
            fp.flush()
            try:
                import os
                os.fsync(fp.fileno())
            except OSError:
                logger.debug("fsync non supporté sur ce FS")
        logger.info("Corpus append : id=%s, sharpe=%.4f", entry.id, entry.sharpe_val)

    def load(self) -> list[CorpusEntry]:
        """Recharge tous les Einhers du corpus (P1 #7)."""
        if not self.path.exists():
            return []
        entries: list[CorpusEntry] = []
        with self.path.open("r", encoding="utf-8") as fp:
            for line in fp:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                    entries.append(CorpusEntry.from_dict(d))
                except (json.JSONDecodeError, KeyError) as exc:
                    logger.warning("Ligne corpus invalide : %s", exc)
        return entries

    def summary(self) -> dict[str, Any]:
        """Statistiques globales du corpus (pour P1 #6 quotas + P1 #5 corrélation)."""
        entries = self.load()
        n_total = len(entries)
        n_long = sum(1 for e in entries if e.direction == "long")
        n_short = sum(1 for e in entries if e.direction == "short")
        sharpes = [e.sharpe_val for e in entries if e.sharpe_val == e.sharpe_val]
        median_sharpe = sorted(sharpes)[len(sharpes) // 2] if sharpes else 0.0
        return {
            "n_total": n_total,
            "n_long": n_long,
            "n_short": n_short,
            "n_long_frac": n_long / n_total if n_total > 0 else 0.0,
            "n_short_frac": n_short / n_total if n_total > 0 else 0.0,
            "median_sharpe": median_sharpe,
            "fingerprint_structurels_uniques": len({e.fingerprint_structurel for e in entries}),
        }

    def clear_for_testing(self) -> None:
        """Vide le corpus (POUR LES TESTS UNIQUEMENT)."""
        self.path.write_text("", encoding="utf-8")
        logger.warning("Corpus cleared (testing only)")

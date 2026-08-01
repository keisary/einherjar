"""selection/selector.py — Sélectionne et installe le générateur gagnant.

Le Selector lit le ComparisonReport du comparator et :
  1. Extrait le générateur #1 (par score).
  2. Le persiste dans un fichier JSON (pour reproductibilité inter-runs).
  3. Expose une API simple pour les étapes suivantes (refinement, admit).

Le choix est basé sur le `score` du comparator (admission_rate × Sharpe médian).
Pour V1, c'est le top 1 uniquement. Pour V2, on pourrait ajouter du tie-breaking
(stabilité cross-asset, temps d'exécution, etc.).

Conforme à ONTOLOGY.md et ALGORITHME_RESEARCH.md § 10.2 étape 3.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from einherjar.research.generators.algorithms import (
    BaseGenerator,
    BeamSearchGenerator,
    GrammaticalEvolutionGenerator,
    MemeticGenerator,
    NSGA2Generator,
    RandomSearchGenerator,
    TypedGPGenerator,
)
from einherjar.research.generators.comparator import ComparisonReport, GeneratorRanking
from einherjar.research.generators.protocol import GenerationProtocol

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Selection result
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class SelectedGenerator:
    """Générateur sélectionné pour la phase de production.

    Persisté en JSON pour que les runs suivants réutilisent le même choix
    sans relancer la comparaison.
    """

    generator_name: str
    generator_class: str             # nom complet de la classe (pour re-instanciation)
    selection_timestamp: str         # ISO 8601 UTC
    protocol: GenerationProtocol
    ranking_snapshot: dict[str, Any] = field(default_factory=dict)
    score: float = 0.0
    rank: int = 1
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "generator_name": self.generator_name,
            "generator_class": self.generator_class,
            "selection_timestamp": self.selection_timestamp,
            "protocol": self.protocol.to_dict(),
            "ranking_snapshot": self.ranking_snapshot,
            "score": self.score,
            "rank": self.rank,
            "reason": self.reason,
        }


# --------------------------------------------------------------------------- #
# Selector
# --------------------------------------------------------------------------- #


class GeneratorSelector:
    """Sélectionne le meilleur générateur depuis un ComparisonReport.

    Attributes:
        protocol: Protocole de génération (passé au générateur sélectionné).
    """

    def __init__(self, protocol: GenerationProtocol) -> None:
        self.protocol = protocol
        logger.info("GeneratorSelector instancié")

    def select(
        self,
        report: ComparisonReport,
        *,
        top_n: int = 1,
        min_score: float = 0.0,
    ) -> SelectedGenerator:
        """Sélectionne le top générateur depuis le rapport.

        Args:
            report: ComparisonReport issu du comparator.
            top_n: Nombre de générateurs à sélectionner (V1 : 1).
            min_score: Score minimum requis (sinon, échec).

        Returns:
            SelectedGenerator.

        Raises:
            ValueError: si aucun générateur ne passe le filtre.
        """
        if not report.rankings:
            raise ValueError("ComparisonReport vide — aucun générateur à sélectionner")
        # Filtre par min_score.
        candidates = [r for r in report.rankings if r.score >= min_score]
        if not candidates:
            raise ValueError(
                f"Aucun générateur avec score >= {min_score} "
                f"(max observé: {max(r.score for r in report.rankings):.4f})"
            )
        # Tri par score décroissant (normalement déjà trié, mais on re-confirme).
        candidates = sorted(candidates, key=lambda r: r.score, reverse=True)
        top = candidates[0]
        # Trouve la classe Python du générateur.
        gen_class_name = self._class_name_for(top.generator_name)
        reason = (
            f"Sélectionné sur score={top.score:.4f} "
            f"(rank={top.rank}, admission_rate={top.admission_rate:.4f}, "
            f"median_sharpe={top.median_sharpe:.4f})"
        )
        return SelectedGenerator(
            generator_name=top.generator_name,
            generator_class=gen_class_name,
            selection_timestamp=datetime.now(timezone.utc).isoformat(),
            protocol=self.protocol,
            ranking_snapshot=top.to_dict(),
            score=top.score,
            rank=top.rank,
            reason=reason,
        )

    def save(self, selected: SelectedGenerator, path: Path) -> None:
        """Persiste la sélection (pour reproductibilité inter-runs)."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(selected.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        logger.info("Sélection sauvegardée : %s → %s", selected.generator_name, path)

    @staticmethod
    def load(path: Path) -> SelectedGenerator:
        """Charge une sélection depuis le disque."""
        if not path.exists():
            raise FileNotFoundError(f"Fichier de sélection introuvable : {path}")
        d = json.loads(path.read_text(encoding="utf-8"))
        # Reconstruit le protocol depuis le dict.
        protocol_d = d["protocol"]
        protocol = GenerationProtocol(
            seed=protocol_d["seed"],
            data_version=protocol_d["data_version"],
            splits=protocol_d["splits"],
            n_eval_budget=protocol_d["n_eval_budget"],
            max_conditions=protocol_d["max_conditions"],
            p_compound=protocol_d["p_compound"],
            assets=tuple(protocol_d["assets"]),
            timeframes=tuple(protocol_d["timeframes"]),
            amplitude_value=protocol_d["amplitude_value"],
            cooldown_k=protocol_d["cooldown_k"],
        )
        return SelectedGenerator(
            generator_name=d["generator_name"],
            generator_class=d["generator_class"],
            selection_timestamp=d["selection_timestamp"],
            protocol=protocol,
            ranking_snapshot=d.get("ranking_snapshot", {}),
            score=d.get("score", 0.0),
            rank=d.get("rank", 1),
            reason=d.get("reason", ""),
        )

    @staticmethod
    def _class_name_for(generator_name: str) -> str:
        """Mappe le nom du générateur vers le nom complet de sa classe."""
        mapping = {
            "RandomSearchGenerator": "RandomSearchGenerator",
            "BeamSearchGenerator": "BeamSearchGenerator",
            "TypedGPGenerator": "TypedGPGenerator",
            "GrammaticalEvolutionGenerator": "GrammaticalEvolutionGenerator",
            "MemeticGenerator": "MemeticGenerator",
            "NSGA2Generator": "NSGA2Generator",
        }
        return mapping.get(generator_name, generator_name)

    @staticmethod
    def instantiate(selected: SelectedGenerator, config: Any) -> BaseGenerator:
        """Ré-instancie le générateur sélectionné depuis la classe stockée.

        Args:
            selected: SelectedGenerator (depuis load() ou select()).
            config: EinherjarConfig (passé au constructeur du générateur).

        Returns:
            Instance du générateur prêt à l'emploi.
        """
        cls_name = selected.generator_class
        protocol = selected.protocol
        if cls_name == "RandomSearchGenerator":
            return RandomSearchGenerator(protocol=protocol, config=config)
        if cls_name == "BeamSearchGenerator":
            return BeamSearchGenerator(protocol=protocol, config=config)
        if cls_name == "TypedGPGenerator":
            return TypedGPGenerator(protocol=protocol, config=config)
        if cls_name == "GrammaticalEvolutionGenerator":
            return GrammaticalEvolutionGenerator(protocol=protocol)
        if cls_name == "MemeticGenerator":
            return MemeticGenerator(protocol=protocol, config=config)
        if cls_name == "NSGA2Generator":
            return NSGA2Generator(protocol=protocol, config=config)
        raise ValueError(f"Classe de générateur inconnue : {cls_name}")


# --------------------------------------------------------------------------- #
# Factory
# --------------------------------------------------------------------------- #


def make_default_selector(protocol: GenerationProtocol) -> GeneratorSelector:
    """Construit un GeneratorSelector avec le protocole fourni."""
    return GeneratorSelector(protocol=protocol)

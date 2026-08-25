"""space.py — Bornes de l'espace de combinaisons (artefact versionné).

Plan Étape B point 3 (lignes 318-321) : l'espace est borné par les opérateurs,
les transformations autorisées, max_depth et le pool de seuils. Fixer ces
bornes = rendre « l'espace limité » concret, versionné, défendable. C'est un
artefact de config à part entière (data_version pour reproductibilité, ligne
443-444).
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SpaceConfig:
    """Bornes de langage + pool de features du moteur de recherche.

    Attributes:
        data_version: version du jeu de données/features (reproducibilité).
        feature_names: pool de features autorisé (après prune B1 : features
            valides, non constantes, non redondantes).
        numeric_ops: opérateurs numériques STGP.
        bool_ops: opérateurs booléens STGP (XOR demandé par Jovanny 2026-08-20).
        cmp_ops: opérateurs de comparaison (atomes Cmp).
        max_depth: profondeur max des arbres (plan ligne 319).
        min_depth: profondeur min à la génération (diversité initiale).
        const_values: constante numériques tirées à la génération.
        threshold_quantiles: niveaux de quantiles du pool de seuils (calculés
            sur la fenêtre TRAIN, jamais sur val/holdout).
        max_size: taille max en nœuds (anti-bloat, plan ligne 49).
    """

    data_version: str
    feature_names: tuple[str, ...]
    numeric_ops: tuple[str, ...] = ("+", "-", "*", "/", "min", "max")
    bool_ops: tuple[str, ...] = ("AND", "OR", "XOR", "NOT")
    cmp_ops: tuple[str, ...] = ("<", "<=", ">", ">=")
    max_depth: int = 6
    min_depth: int = 2
    const_values: tuple[float, ...] = (-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0)
    threshold_quantiles: tuple[float, ...] = (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95)
    max_size: int = 40

    def to_dict(self) -> dict[str, object]:
        """to_dict."""
        return {
            "data_version": self.data_version,
            "n_features": len(self.feature_names),
            "numeric_ops": list(self.numeric_ops),
            "bool_ops": list(self.bool_ops),
            "cmp_ops": list(self.cmp_ops),
            "max_depth": self.max_depth,
            "min_depth": self.min_depth,
            "const_values": list(self.const_values),
            "threshold_quantiles": list(self.threshold_quantiles),
            "max_size": self.max_size,
        }

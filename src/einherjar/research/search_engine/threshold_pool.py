"""threshold_pool.py — Pool de seuils empiriques (fenêtre TRAIN uniquement).

Plan ligne 318-321 : les seuils des atomes Cmp sont tirés d'un pool de
quantiles empiriques. Règle anti-lookahead : le pool est calculé sur la
fenêtre TRAIN (la seule autorisée pour l'exploration).

Deux familles de seuils :
- per-feature : quantiles de CHAQUE feature (atome simple Feature OP seuil) ;
- global      : quantiles d'un échantillon d'EXPRESSIONS aléatoires évaluées
  sur train (atome complexe, ex. (mom - vol) OP seuil). Un seul pool global
  partagé évite de recalculer par expression.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from einherjar.research.search_engine.space import SpaceConfig


@dataclass(frozen=True)
class ThresholdPool:
    """Seuils pré-calculés sur train.

    Attributes:
        per_feature: dict feature_ref → np.ndarray (n_quantiles,) de valeurs.
        global_values: np.ndarray (n_quantiles * n_probes,) de valeurs seuil
            pour les expressions complexes (échantillon de probes aléatoires).
    """

    per_feature: dict[str, np.ndarray]
    global_values: np.ndarray

    @classmethod
    def build(
        cls,
        train_X: np.ndarray,
        feature_names: list[str],
        cfg: SpaceConfig,
        rng: np.random.Generator,
        n_probes: int = 96,
    ) -> ThresholdPool:
        """Construit le pool depuis la matrice TRAIN (jamais val/holdout)."""
        qs = np.asarray(cfg.threshold_quantiles, dtype=np.float64)
        per_feature: dict[str, np.ndarray] = {}
        for i, name in enumerate(feature_names):
            col = train_X[:, i]
            finite = col[np.isfinite(col)]
            if finite.size == 0:
                per_feature[name] = np.zeros_like(qs)
            else:
                per_feature[name] = np.quantile(finite, qs)

        # Pool global : quantiles de probes aléatoires (expressions de
        # profondeur 1-2 sur features réelles).
        from einherjar.research.search_engine.generator import generate_random_num_expr

        probes = []
        for _ in range(n_probes):
            expr = generate_random_num_expr(rng, cfg, max_depth=2)
            from einherjar.research.search_engine.expression import eval_num

            v = eval_num(expr, train_X, {n: i for i, n in enumerate(feature_names)})
            v = v[np.isfinite(v)]
            if v.size > 0:
                probes.append(v)
        if probes:
            stacked = np.concatenate(probes)
            global_values = np.quantile(stacked, qs)
        else:
            global_values = np.zeros_like(qs)
        return cls(per_feature=per_feature, global_values=global_values)

"""generators/comparator.py — Compare les générateurs et classe les résultats.

Le comparator applique le protocole reproductible à chaque générateur,
collecte les GeneratorResult, et produit un GeneratorRanking qui classe
les candidats sur le critère principal (taux d'admission × qualité
médiane × diversité comportementale).

Conforme à ALGORITHME_RESEARCH.md § 10.2 étape 2.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from einherjar.research.config.loader import EinherjarConfig
from einherjar.research.engine.evaluator import EvaluationEngine
from einherjar.research.generators.algorithms import (
    BaseGenerator,
    GeneratorResult,
)
from einherjar.research.generators.protocol import GenerationProtocol

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Sortie
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class GeneratorRanking:
    """Classement d'un générateur par le comparator (multi-objectif).

    Le score composite combine plusieurs sous-scores normalises (Pareto-like)
    pour eviter qu'un generateur excellent sur un seul axe (e.g., Sharpe)
    mais mediocre sur les autres (diversite, coherence) ne domine le
    classement.

    Sub-scores (avant normalisation) :
      - sharpe            : Sharpe median des Einhers admis
      - admission_rate    : taux d'admission (n_admis / n_evalues)
      - diversity         : nb de features distinctes utilisees (proxy)
      - semantic_coherence: % d'hypotheses ou l'orientation semantique
                             du pattern matche la direction Hypothesis
    """

    generator_name: str
    rank: int                                  # 1 = meilleur
    score: float                               # score composite pour le classement
    n_generated: int
    n_evaluated: int
    n_passed_admission: int
    admission_rate: float                      # n_passed_admission / n_evaluated
    median_sharpe: float                       # Sharpe médian des Einhers admis
    median_sharpe_all: float                   # Sharpe médian de TOUS les évalués
    n_distinct_features: int                   # cardinalite des features distinctes
    semantic_coherence: float                  # 0..1, % match pattern/direction
    elapsed_s: float
    subscores: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "generator_name": self.generator_name,
            "rank": self.rank,
            "score": self.score,
            "n_generated": self.n_generated,
            "n_evaluated": self.n_evaluated,
            "n_passed_admission": self.n_passed_admission,
            "admission_rate": self.admission_rate,
            "median_sharpe": self.median_sharpe,
            "median_sharpe_all": self.median_sharpe_all,
            "n_distinct_features": self.n_distinct_features,
            "semantic_coherence": self.semantic_coherence,
            "subscores": dict(self.subscores),
            "elapsed_s": round(self.elapsed_s, 3),
        }


@dataclass
class ComparisonReport:
    """Rapport global de comparaison."""

    protocol: GenerationProtocol
    rankings: list[GeneratorRanking] = field(default_factory=list)
    raw_results: dict[str, GeneratorResult] = field(default_factory=dict)
    sharpe_distributions: dict[str, list[float]] = field(default_factory=dict)
    elapsed_s: float = 0.0
    winner_name: str | None = None
    # P1-08 : budget global + compteur cumule (audit + mur d'arret partage).
    total_evaluations: int = 0
    budget: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "protocol": self.protocol.to_dict(),
            "rankings": [r.to_dict() for r in self.rankings],
            "sharpe_distributions": {k: sorted(v) for k, v in self.sharpe_distributions.items()},
            "elapsed_s": round(self.elapsed_s, 3),
            "winner_name": self.winner_name,
            "total_evaluations": self.total_evaluations,
            "budget": self.budget,
        }


# --------------------------------------------------------------------------- #
# Comparator
# --------------------------------------------------------------------------- #


class GeneratorComparator:
    """Compare les générateurs sous un protocole reproductible.

    Le comparator peut :
      1. Appliquer chaque générateur (génération d'hypothèses).
      2. Évaluer chaque hypothèse via le moteur + admission.
      3. Classer les générateurs.
      4. Retourner le rapport + le nom du gagnant.

    L'admission peut être :
      - `None` : on note juste les métriques brutes (rapide).
      - Un callable `(hypothesis, calibrated, mesures_val) -> bool` : on filtre.
    """

    def __init__(
        self,
        generators: list[BaseGenerator],
        protocol: GenerationProtocol,
        engine: EvaluationEngine,
        config: EinherjarConfig,
        corpus_feature_sets: tuple[frozenset[str], ...] | None = None,
    ) -> None:
        self.generators = generators
        self.protocol = protocol
        self.engine = engine
        self.config = config
        # P1-10 : override du corpus Jaccard (defaut = chargement auto depuis CorpusStore).
        self._corpus_override = corpus_feature_sets
        logger.info(
            "GeneratorComparator instancié : %d générateurs, protocol=%s",
            len(generators), protocol.to_dict(),
        )

    def _build_corpus_feature_sets(self) -> tuple[frozenset[str], ...]:
        """Construit le tuple de frozensets[feature_ref] depuis le corpus (P1-10).

        1. Si self._corpus_override est fourni : on l'utilise tel quel.
        2. Sinon : on charge CorpusStore et on extrait les features de chaque entry.
        3. Si corpus vide : retourne () (les generateurs retombent sur dispersion pure).
        """
        if self._corpus_override is not None:
            return self._corpus_override
        try:
            from einherjar.research.corpus.store import CorpusStore
            from einherjar.research.generators.algorithms import _collect_feature_refs
            from einherjar.research.utils.types import Hypothesis
            entries = CorpusStore().load()
            result: list[frozenset[str]] = []
            for entry in entries:
                try:
                    hyp = Hypothesis.from_dict(entry.hypothesis)
                    feats = frozenset(_collect_feature_refs(hyp.condition_tree))
                    if feats:
                        result.append(feats)
                except Exception:  # noqa: BLE001
                    continue
            logger.info(
                "P1-10 : corpus charge pour Jaccard : %d entries, %d features distinctes (moyenne %.1f)",
                len(entries), len(result), sum(len(s) for s in result) / max(len(result), 1),
            )
            return tuple(result)
        except Exception as exc:  # noqa: BLE001
            logger.warning("P1-10 : impossible de charger le corpus : %s", exc)
            return ()

    def run(
        self,
        train_ohlcv: Any,
        train_features: Any,
        val_ohlcv: Any,
        val_features: Any,
        *,
        admission_fn: Callable | None = None,
        multi_assets: dict[str, tuple] | None = None,
    ) -> ComparisonReport:
        """Compare les générateurs sous le protocole.

        Args:
            train_ohlcv/train_features: split train.
            val_ohlcv/val_features: split val.
            admission_fn: Filtre optionnel (cf. docstring).

        Returns:
            ComparisonReport avec rankings + winner.

        Score composite multi-objectif (normalise min-max entre generateurs) :
          score = 0.40 * norm(sharpe) + 0.30 * norm(admission_rate)
                + 0.15 * norm(diversity)      + 0.15 * norm(coherence)
        Si un sous-score est manquant (e.g., pas de patterns -> coherence
        indefinie), on redistribue les poids sur les axes disponibles.
        """
        t0 = time.time()
        report = ComparisonReport(protocol=self.protocol)
        # P1-08 : compteur global cumule d'evaluations (toutes phases confondues).
        # On respecte le mur d'arret n_eval_budget : si on l'atteint, on
        # raccourcit la liste d'hypotheses des generateurs suivants.
        global_eval_count: int = 0
        budget_remaining: int = int(self.protocol.n_eval_budget)
        # P1-10 : peupler corpus_feature_sets sur NSGA-II (Jaccard vs corpus).
        # Le caller peut overrider via self.corpus_feature_sets_override.
        corpus_sets = self._build_corpus_feature_sets()
        # Phase 1 : generer + evaluer chaque generateur, collecter sub-scores bruts.
        raw_subscores: list[dict[str, float]] = []
        for gen in self.generators:
            logger.info("=" * 60)
            logger.info("Générateur : %s", gen.name)
            logger.info("=" * 60)
            t_gen = time.time()
            # Injection du corpus si NSGA-II (opt-in).
            if hasattr(gen, "_corpus_feature_sets"):
                gen._corpus_feature_sets = corpus_sets
            # P1-10 : injection du dict multi-actifs au NSGA-II.
            if hasattr(gen, "_multi_assets"):
                gen._multi_assets = multi_assets
            try:
                gen.bind_data(train_ohlcv, train_features, val_ohlcv, val_features)
                result = gen.generate()
            except Exception as exc:  # noqa: BLE001
                # A failing engine is reported but cannot abort the comparison.
                logger.exception("Generation failed for %s: %s", gen.name, exc)
                result = GeneratorResult(
                    generator_name=gen.name,
                    hypotheses=(), n_generated=0, n_evaluated=0,
                    n_passed_admission=0,
                    generation_time_s=time.time() - t_gen,
                    meta={"generation_error": str(exc)},
                )
            n_gen = result.n_generated
            n_eval = 0
            n_adm = 0
            sharpes: list[float] = []
            sharpes_all: list[float] = []
            features_used: set[str] = set()
            coherence_match: int = 0
            coherence_total: int = 0
            # External evaluation budget is a hard wall as well.
            # P1-08 : le mur d'arret est global, pas par-generateur.
            # On prend le min(hypotheses, budget_restant).
            n_skipped_budget = 0
            for hyp in result.hypotheses:
                if budget_remaining <= 0:
                    n_skipped_budget += 1
                    continue
                # Track feature usage (pour diversity)
                _collect_features(hyp.condition_tree, features_used)
                # Track semantic coherence (orientation vs direction)
                m, t = _compute_coherence_for_hyp(hyp)
                coherence_match += m
                coherence_total += t
                try:
                    calibrated = self.engine.train_calibrate(hyp, train_ohlcv, train_features)
                    mesures = self.engine.test_on(
                        hyp, val_ohlcv, val_features, calibrated, "val",
                    )
                    n_eval += 1
                    global_eval_count += 1
                    budget_remaining -= 1
                    if mesures.sharpe_net == mesures.sharpe_net:  # not NaN
                        sharpes_all.append(mesures.sharpe_net)
                    if admission_fn is not None and admission_fn(hyp, calibrated, mesures):
                        n_adm += 1
                        if mesures.sharpe_net == mesures.sharpe_net:
                            sharpes.append(mesures.sharpe_net)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Échec eval %s sur %s : %s", gen.name, hyp.id, exc)
            if n_skipped_budget > 0:
                logger.warning(
                    "P1-08 : %s : %d hypotheses non evaluees (budget global epuise : %d/%d)",
                    gen.name, n_skipped_budget,
                    global_eval_count, self.protocol.n_eval_budget,
                )
            elapsed = time.time() - t_gen
            # Mise à jour des résultats avec les vrais compteurs.
            raw = GeneratorResult(
                generator_name=result.generator_name,
                hypotheses=result.hypotheses,
                n_generated=n_gen,
                n_evaluated=n_eval,
                n_passed_admission=n_adm,
                generation_time_s=elapsed,
                meta=result.meta,
            )
            report.raw_results[gen.name] = raw
            report.sharpe_distributions[gen.name] = sharpes_all
            # No admission function means no admission statistic, not 100%.
            adm_rate = n_adm / max(n_eval, 1) if admission_fn is not None else 0.0
            med_sharpe = _median(sharpes)
            med_sharpe_all = _median(sharpes_all)
            coherence = (coherence_match / coherence_total) if coherence_total > 0 else 0.0
            subscores = {
                "sharpe": med_sharpe,
                "admission_rate": adm_rate,
                "diversity": float(len(features_used)),
                "coherence": coherence,
            }
            raw_subscores.append(subscores)
            ranking = GeneratorRanking(
                generator_name=gen.name,
                rank=0,
                score=0.0,  # sera calcule apres normalisation
                n_generated=n_gen,
                n_evaluated=n_eval,
                n_passed_admission=n_adm,
                admission_rate=adm_rate,
                median_sharpe=med_sharpe,
                median_sharpe_all=med_sharpe_all,
                n_distinct_features=len(features_used),
                semantic_coherence=coherence,
                subscores=subscores,
                elapsed_s=elapsed,
            )
            report.rankings.append(ranking)
        # Phase 2 : normaliser min-max les sub-scores entre generateurs.
        norm_subscores = _normalize_min_max(raw_subscores)
        # Phase 3 : calculer le score composite multi-obj.
        for i, r in enumerate(report.rankings):
            ns = norm_subscores[i]
            # Poids par defaut (redistribues si coherence absente)
            w_sharpe = 0.40
            w_adm = 0.30
            w_div = 0.15
            w_coh = 0.15
            if r.semantic_coherence == 0.0 and r.n_distinct_features > 0:
                # Coherence non-pertinente pour ce generateur (pas de patterns)
                # -> redistribuer les poids sur les 3 autres axes.
                total = w_sharpe + w_adm + w_div
                w_sharpe = w_sharpe / total
                w_adm = w_adm / total
                w_div = w_div / total
                w_coh = 0.0
            composite = (
                w_sharpe * ns["sharpe"]
                + w_adm * ns["admission_rate"]
                + w_div * ns["diversity"]
                + w_coh * ns["coherence"]
            )
            # Reconstruction (dataclass frozen).
            report.rankings[i] = GeneratorRanking(
                generator_name=r.generator_name,
                rank=0,
                score=composite,
                n_generated=r.n_generated,
                n_evaluated=r.n_evaluated,
                n_passed_admission=r.n_passed_admission,
                admission_rate=r.admission_rate,
                median_sharpe=r.median_sharpe,
                median_sharpe_all=r.median_sharpe_all,
                n_distinct_features=r.n_distinct_features,
                semantic_coherence=r.semantic_coherence,
                subscores={
                    **{k: round(v, 4) for k, v in r.subscores.items()},
                    **{f"norm_{k}": round(v, 4) for k, v in ns.items()},
                    "composite": round(composite, 4),
                },
                elapsed_s=r.elapsed_s,
            )
        # Tri par score décroissant, fix des rangs.
        report.rankings.sort(key=lambda r: r.score, reverse=True)
        for i, r in enumerate(report.rankings):
            report.rankings[i] = GeneratorRanking(
                generator_name=r.generator_name,
                rank=i + 1,
                score=r.score,
                n_generated=r.n_generated,
                n_evaluated=r.n_evaluated,
                n_passed_admission=r.n_passed_admission,
                admission_rate=r.admission_rate,
                median_sharpe=r.median_sharpe,
                median_sharpe_all=r.median_sharpe_all,
                n_distinct_features=r.n_distinct_features,
                semantic_coherence=r.semantic_coherence,
                subscores=r.subscores,
                elapsed_s=r.elapsed_s,
            )
        report.winner_name = report.rankings[0].generator_name if report.rankings else None
        report.elapsed_s = time.time() - t0
        # P1-08 : expose le compteur global sur le report (auditabilite).
        report.total_evaluations = global_eval_count
        report.budget = int(self.protocol.n_eval_budget)
        logger.info(
            "Comparaison terminée : winner=%s, %d générateurs, %.2fs, "
            "%d/%d évaluations (P1-08 budget global)",
            report.winner_name, len(self.generators), report.elapsed_s,
            global_eval_count, self.protocol.n_eval_budget,
        )
        return report


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _median(values: list[float]) -> float:
    """Médiane (tri + index central)."""
    if not values:
        return 0.0
    s = sorted(values)
    n = len(s)
    return s[n // 2] if n % 2 == 1 else (s[n // 2 - 1] + s[n // 2]) / 2


def _collect_features(node: Any, out: set[str]) -> None:
    """Collecte (recursivement) les noms de features dans un condition tree.

    Args:
        node: Condition ou ConditionNode.
        out: set accumulateur (modifie en place).
    """
    from einherjar.research.utils.types import Condition, ConditionNode
    if isinstance(node, Condition):
        out.add(node.feature_ref)
    elif isinstance(node, ConditionNode):
        _collect_features(node.left, out)
        if node.right is not None:
            _collect_features(node.right, out)


# _track_semantic_coherence_unused() a ete supprime (code mort).

def _compute_coherence_for_hyp(hyp: Any) -> tuple[int, int]:
    """Retourne (match, total) pour une Hypothesis (1 si match, 0 sinon).

    Match = l'orientation semantique du pattern matche la direction :
      - BULLISH + LONG = match
      - BEARISH + SHORT = match
      - NEUTRAL = pas compte (coherence indefinie)
      - pattern_X sans semantic_orientation (e.g., non-pattern) = pas compte
    """
    from einherjar.research.utils.types import (
        Condition, ConditionNode, Direction,
    )
    orient = hyp.meta.get("semantic_orientation") if hyp.meta else None
    if orient is None or orient == "neutral":
        return (0, 0)
    features: set[str] = set()
    _collect_features(hyp.condition_tree, features)
    if not features:
        return (0, 0)
    pattern_features = {f for f in features if f.startswith("pattern_")}
    if not pattern_features:
        return (0, 0)
    expected_dir = (
        Direction.LONG if orient == "bullish" else Direction.SHORT
    )
    return (1 if hyp.direction == expected_dir else 0, 1)


def _normalize_min_max(
    raw: list[dict[str, float]],
) -> list[dict[str, float]]:
    """Normalise min-max chaque sous-score entre 0 et 1.

    Pour chaque cle du dict :
      - min = minimum sur tous les generateurs
      - max = maximum
      - norm = (val - min) / (max - min) si max > min, sinon 0.5
        (tous egaux -> score moyen)
    """
    if not raw:
        return []
    keys = list(raw[0].keys())
    mins = {k: min(r.get(k, 0.0) for r in raw) for k in keys}
    maxs = {k: max(r.get(k, 0.0) for r in raw) for k in keys}
    out: list[dict[str, float]] = []
    for r in raw:
        normed: dict[str, float] = {}
        for k in keys:
            lo, hi = mins[k], maxs[k]
            if hi > lo:
                normed[k] = (r.get(k, 0.0) - lo) / (hi - lo)
            else:
                # Tous egaux : score moyen
                normed[k] = 0.5
        out.append(normed)
    return out

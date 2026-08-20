"""map_elites.py — Archive MAP-Elites (plan D1, lignes 398-412).

L'archive partitionne l'espace des stratégies par descripteurs (direction,
famille dominante, régime volatilité) et garde le MEILLEUR candidat de chaque
cellule. La variation (crossover + mutation STGP) rassemble les parents d'une
même direction — les enfants héritent de la direction du parent A.

La fitness est la fitness CHEAP (Sharpe net sur bloc aléatoire) — la
validation lourde (C1-C6) n'intervient qu'à l'admission.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import polars as pl

from einherjar.research.search_engine.descriptors import describe
from einherjar.research.search_engine.fitness import cheap_fitness
from einherjar.research.search_engine.generator import (
    crossover,
    generate_population,
    generate_random_bool_expr,
    mutate,
)


@dataclass
class CellEntry:
    expr: object
    einher: object
    fitness: float
    direction: str


class MapElitesArchive:
    """Archive par cellules (meilleur candidat par descripteur)."""

    def __init__(self, max_cells: int = 512) -> None:
        self.cells: dict[tuple[str, str, str], CellEntry] = {}
        self.max_cells = max_cells

    def insert(self, cell: tuple[str, str, str], entry: CellEntry) -> bool:
        """Ajoute/remplace si fitness strictement meilleure. Retourne True si changé."""
        cur = self.cells.get(cell)
        if cur is None or entry.fitness > cur.fitness:
            self.cells[cell] = entry
            return True
        return False

    def occupied_cells(self) -> list[tuple[str, str, str]]:
        return sorted(self.cells)

    def sample_parent(self, rng: np.random.Generator) -> CellEntry:
        """Cellule au hasard (distribution uniforme sur les cellules occupées)."""
        cells = self.occupied_cells()
        cell = cells[int(rng.integers(len(cells)))]
        return self.cells[cell]

    def best(self) -> CellEntry | None:
        if not self.cells:
            return None
        return max(self.cells.values(), key=lambda e: e.fitness)


def run_map_elites(
    rng: np.random.Generator,
    cfg: object,
    pool: object,
    taxonomy: dict,
    data: dict,
    *,
    costs_pct: float,
    amplitude_bars: int,
    universe: dict,
    n_pop: int = 100,
    n_generations: int = 20,
    p_crossover: float = 0.6,
    directions: tuple[str, ...] = ("BUY", "SELL"),
    sample_frac: float = 0.5,
    data_version: str = "",
    max_cells: int = 512,
) -> MapElitesArchive:
    """Pipeline MAP-Elites complet : init aléatoire puis variation/insertion."""
    archive = MapElitesArchive(max_cells=max_cells)
    ohlcv_df: pl.DataFrame = data["ohlcv_df"]
    X: np.ndarray = data["X"]
    feature_names: list[str] = data["feature_names"]

    def evaluate(expr: object, direction: str):
        fitness, einher, sub = cheap_fitness(
            expr, direction, amplitude_bars, universe,
            ohlcv_df, X, feature_names, rng,
            costs_pct=costs_pct, sample_frac=sample_frac, data_version=data_version,
        )
        cell = describe(expr, direction, sub, taxonomy)
        archive.insert(cell, CellEntry(expr=expr, einher=einher, fitness=fitness, direction=direction))

    # Init : population aléatoire (toutes directions)
    pop = generate_population(rng, cfg, pool, n_pop)
    for expr in pop:
        direction = str(rng.choice(directions))
        evaluate(expr, direction)

    # Boucle générations : variation → évaluation → insertion
    for _ in range(n_generations):
        if not archive.occupied_cells():
            break
        for _ in range(n_pop):
            if rng.random() < p_crossover and len(archive.occupied_cells()) >= 2:
                a = archive.sample_parent(rng)
                b = archive.sample_parent(rng)
                expr = crossover(a.expr, b.expr, rng, cfg, pool)
                direction = a.direction
            else:
                a = archive.sample_parent(rng)
                expr = mutate(a.expr, rng, cfg, pool)
                direction = a.direction
            evaluate(expr, direction)

    return archive
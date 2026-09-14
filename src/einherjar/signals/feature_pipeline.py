"""Pipeline de features du systeme vivant (produit le schema MIDAS complet).

Chaine de calcul, entierement interne a EINHERJAR (aucun import `auriga.*`) :

    OHLCV (polars)
      -> technical_indicators.TechnicalIndicatorsEnricher   # indicateurs + *_signal simples
      -> quantitative_features.OptimizedQuantitativeFeaturesEnricher   # quant_*
      -> calculator.AlphaFactorCalculator                   # Factor_* + signaux derives
      -> numba_pattern_detectors (via midas_bridge.PatternBridge)       # pattern_* (107)

Le resultat porte les noms EXACTS du schema compile MIDAS (246 colonnes), qui sont
ceux references par le corpus de recherche (`condition_tree.feature_ref`).

Usage :

    from einherjar.signals.feature_pipeline import FeaturePipeline
    pipe = FeaturePipeline()
    enriched = pipe.compute(ohlcv_df)            # pl.DataFrame OHLCV
    report = pipe.coverage(enriched, refs)       # couverture des feature_ref du corpus

Reference schema : `D:/midas_v2/midasV3/src/data/compiled/<classe>/<tf>/metadata.json`
(246 colonnes, identiques pour les 35 couples classe/timeframe).
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import polars as pl

logger = logging.getLogger(__name__)

# Prefixe des colonnes de patterns dans le schema compile (enrich_dataset.py:179).
PATTERN_COLUMN_PREFIX = "pattern_"

# Colonnes OHLCV presentes dans les donnees compilees (non des features).
OHLCV_COLUMNS = ("timestamp", "open", "high", "low", "close", "volume")


class FeaturePipelineError(Exception):
    """Erreur bloquante du pipeline de features (dependance ou etape indisponible)."""


class FeaturePipeline:
    """Calcule le schema de features complet sur un OHLCV mono-actif.

    Attributs:
        mode: Mode d'enrichissement technique transmis a TechnicalIndicatorsEnricher.
        chunk_size: Taille de chunk des enrichisseurs MIDAS.
        max_memory_gb: Plafond memoire transmis aux enrichisseurs.
        n_jobs: Parallelisme interne du calcul technique (1 = sequence, minimum memoire).
    """

    def __init__(
        self,
        mode: str = "full",
        chunk_size: int = 50000,
        max_memory_gb: float = 8.0,
        n_jobs: int = 1,
        max_lookback: int = 1500,
    ) -> None:
        """Initialise le pipeline (les enrichisseurs sont construits a la demande).

        Args:
            mode: Mode MIDAS des indicateurs techniques ('fast' | 'balanced' | 'full').
            chunk_size: Taille de chunk des enrichisseurs.
            max_memory_gb: Plafond memoire des enrichisseurs.
            n_jobs: Parallelisme interne du calcul technique (1 = sequence).
            max_lookback: Fenetre historique recalculee a chaque bougie (inference live).
        """
        self.mode = mode
        self.chunk_size = chunk_size
        self.max_memory_gb = max_memory_gb
        self.n_jobs = n_jobs
        self.max_lookback = max_lookback
        self._technical: Any | None = None
        self._quantitative: Any | None = None
        self._factors: Any | None = None
        self._patterns: Any | None = None
        self.last_report: dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Ressources (construction paresseuse, reutilisees entre les appels)
    # ------------------------------------------------------------------

    def _technical_enricher(self) -> Any:
        """Retourne l'enrichisseur d'indicateurs techniques (copie MIDAS interne)."""
        if self._technical is None:
            from einherjar.signals.technical_indicators import TechnicalIndicatorsEnricher

            self._technical = TechnicalIndicatorsEnricher(
                mode=self.mode,
                chunk_size=self.chunk_size,
                max_memory_gb=self.max_memory_gb,
                n_jobs=self.n_jobs,
                use_dask=False,
                _is_worker=True,
            )
        return self._technical

    def _quantitative_enricher(self) -> Any:
        """Retourne l'enrichisseur de features quantitatives (copie MIDAS interne)."""
        if self._quantitative is None:
            from einherjar.signals.quantitative_features import (
                OptimizedQuantitativeFeaturesEnricher,
            )

            self._quantitative = OptimizedQuantitativeFeaturesEnricher(
                chunk_size=self.chunk_size, max_memory_gb=self.max_memory_gb
            )
        return self._quantitative

    def _factor_calculator(self) -> Any:
        """Retourne le calculateur de facteurs alpha (copie du strategy_generator)."""
        if self._factors is None:
            from einherjar.signals.calculator import AlphaFactorCalculator

            self._factors = AlphaFactorCalculator()
        return self._factors

    def _pattern_bridge(self) -> Any:
        """Retourne le bridge de detection de patterns (numba_pattern_detectors)."""
        if self._patterns is None:
            from einherjar.signals.midas_bridge import PatternBridge

            self._patterns = PatternBridge()
        return self._patterns

    # ------------------------------------------------------------------
    # Etapes
    # ------------------------------------------------------------------

    def compute_technical(self, df: pl.DataFrame, asset: str = "ASSET", timeframe: str = "1h") -> pl.DataFrame:
        """Ajoute les colonnes d'indicateurs techniques (schema MIDAS).

        L'enrichisseur MIDAS selectionne ses chunks par colonne `asset` : elle est
        injectee si le OHLCV live ne la porte pas (comme dans les donnees compilees).
        """
        enricher = self._technical_enricher()
        pdf = df.to_pandas()
        pdf["asset"] = asset
        pdf["timeframe"] = timeframe
        out = enricher.enrich_dataframe(pdf)
        return _pandas_to_polars(out)

    def compute_quantitative(self, df: pl.DataFrame, asset: str = "ASSET", timeframe: str = "1h") -> pl.DataFrame:
        """Ajoute les colonnes `quant_*` (schema MIDAS)."""
        enricher = self._quantitative_enricher()
        pdf = df.to_pandas()
        if "asset" not in pdf.columns:
            pdf["asset"] = asset
        if "timeframe" not in pdf.columns:
            pdf["timeframe"] = timeframe
        out = enricher.enrich_dataset(pdf)
        if out is None or len(out) == 0:
            raise FeaturePipelineError("Enrichissement quantitatif vide")
        return _pandas_to_polars(out)

    def compute_patterns(self, df: pl.DataFrame) -> pl.DataFrame:
        """Ajoute les colonnes `pattern_<nom>` (107 patterns detectes).

        La detection ne recoit que les colonnes OHLCV (colonnes de contexte `asset`
        et `timeframe` exclues), comme dans la chaine MIDAS de reference.
        """
        from einherjar.signals.feature_engine import ALL_PATTERN_NAMES

        ohlcv_cols = [c for c in OHLCV_COLUMNS if c in df.columns]
        detected = self._pattern_bridge().detect(df.select(ohlcv_cols), ALL_PATTERN_NAMES)
        if not detected:
            raise FeaturePipelineError("Aucun pattern detecte (detecteur indisponible)")
        cols = [
            pl.Series(f"{PATTERN_COLUMN_PREFIX}{name}", np.asarray(arr, dtype=np.float32))
            for name, arr in detected.items()
            if len(np.asarray(arr)) == df.height
        ]
        return df.with_columns(cols) if cols else df

    def compute_factors(self, df: pl.DataFrame) -> pl.DataFrame:
        """Ajoute les colonnes `Factor_*` et les signaux derives (copie strategy_generator).

        Le calculateur est applique en mode lazy : facteurs de base (mono-actif) puis
        facteurs sophistiques. Les colonnes produites dependant d'un marche multi-actifs
        (correlation dynamique, force relative) restent absentes si le contexte manque ;
        elles sont comptees dans `last_report['factors_manquants']`.
        """
        calc = self._factor_calculator()
        ldf = df.lazy()
        base = calc.calculate_base_factors(ldf)
        out = base.collect()
        try:
            soph = calc.calculate_sophisticated_factors_lazy(base)
            out = soph.collect()
        except Exception as exc:  # les facteurs de base restent valides
            logger.warning("Facteurs sophistiques indisponibles : %s", exc)
        return out

    # ------------------------------------------------------------------
    # Orchestration
    # ------------------------------------------------------------------

    def compute(
        self,
        df: pl.DataFrame,
        *,
        asset: str = "ASSET",
        timeframe: str = "1h",
        with_factors: bool = True,
    ) -> pl.DataFrame:
        """Calcule toutes les features du schema MIDAS sur un OHLCV mono-actif.

        Args:
            df: DataFrame polars OHLCV [timestamp, open, high, low, close, volume].
            asset: Symbole de l'actif (injecte pour les enrichisseurs MIDAS).
            timeframe: Timeframe de la frame.
            with_factors: Calcule aussi les `Factor_*` et signaux derives.

        Returns:
            DataFrame polars avec les colonnes OHLCV + features (noms du schema compile).

        Raises:
            FeaturePipelineError: si une etape obligatoire echoue.
        """
        if df.height == 0:
            return df
        missing_ohlcv = [c for c in OHLCV_COLUMNS if c not in df.columns]
        if missing_ohlcv:
            raise FeaturePipelineError(f"OHLCV incomplet, colonnes manquantes : {missing_ohlcv}")

        report: dict[str, Any] = {"lignes": df.height}
        frame = self.compute_technical(df, asset, timeframe)
        report["colonnes_techniques"] = frame.width

        frame = self.compute_quantitative(frame, asset, timeframe)
        report["colonnes_apres_quant"] = frame.width

        frame = self.compute_patterns(frame)
        report["colonnes_apres_patterns"] = frame.width

        if with_factors:
            before = set(frame.columns)
            frame = self.compute_factors(frame)
            report["colonnes_facteurs"] = len(set(frame.columns) - before)

        self.last_report = report
        return frame

    # ------------------------------------------------------------------
    # Inference live
    # ------------------------------------------------------------------

    def compute_incremental(
        self,
        df_history: pl.DataFrame,
        new_candle: dict[str, Any],
        asset: str = "ASSET",
        timeframe: str = "1h",
    ) -> pl.DataFrame:
        """Recalcule les features apres cloture d'une bougie (interface InferenceLoop).

        Args:
            df_history: Historique OHLCV connu.
            new_candle: Derniere bougie cloturee.
            asset: Symbole (transmis aux enrichisseurs MIDAS).
            timeframe: Timeframe de la frame.

        Returns:
            DataFrame enrichi (colonnes du schema MIDAS) tronque a `max_lookback`.
        """
        new_row = pl.DataFrame([new_candle])
        df = pl.concat([df_history, new_row], how="vertical_relaxed")
        if len(df) > self.max_lookback:
            df = df.tail(self.max_lookback)
        return self.compute(df, asset=asset, timeframe=timeframe)

    def get_required_lookback(self, feature_name: str) -> int:
        """Retourne la fenetre de recalcul pour une feature (defaut : max_lookback).

        Les fenetres par feature sont definies dans `signals.feature_engine.LOOKBACK_WINDOWS`
        (source unique) ; les colonnes du schema MIDAS y sont referencees par leur nom.
        """
        from einherjar.signals.feature_engine import LOOKBACK_WINDOWS

        ref = feature_name[len(PATTERN_COLUMN_PREFIX):] if feature_name.startswith(
            PATTERN_COLUMN_PREFIX
        ) else feature_name
        return LOOKBACK_WINDOWS.get(feature_name, LOOKBACK_WINDOWS.get(ref, self.max_lookback))

    # ------------------------------------------------------------------
    # Verification / couverture
    # ------------------------------------------------------------------
    @staticmethod
    def coverage(df: pl.DataFrame, refs: list[str], min_valid_ratio: float = 0.5) -> dict[str, Any]:
        """Verifie la presence et la qualite des features demandees.

        Args:
            df: DataFrame enrichi.
            refs: Liste des `feature_ref` attendues (corpus).
            min_valid_ratio: Part minimale de valeurs non nulles attendue par colonne.

        Returns:
            Dict {presentes, absentes, colonnes_vides, ratio_valides}.
        """
        present, absent, empty = [], [], []
        ratios: dict[str, float] = {}
        for ref in refs:
            if ref not in df.columns:
                absent.append(ref)
                continue
            present.append(ref)
            series = df[ref]
            n = series.len()
            if n == 0:
                ratios[ref] = 0.0
                empty.append(ref)
                continue
            valid = series.is_not_nan().sum() if series.dtype.is_float() else series.is_not_null().sum()
            ratio = float(valid) / n
            ratios[ref] = round(ratio, 4)
            if ratio < min_valid_ratio:
                empty.append(ref)
        return {
            "presentes": present,
            "absentes": absent,
            "colonnes_vides": empty,
            "ratio_valides": ratios,
            "total": len(refs),
        }

    @classmethod
    def missing_refs(cls, df: pl.DataFrame, refs: list[str]) -> list[str]:
        """Retourne les `feature_ref` absentes du DataFrame enrichi."""
        return [r for r in refs if r not in df.columns]


def _pandas_to_polars(df: Any) -> pl.DataFrame:
    """Convertit un DataFrame pandas en polars en preservant les noms de colonnes."""
    import pandas as pd

    if isinstance(df, pl.DataFrame):
        return df
    if not isinstance(df, pd.DataFrame):
        raise FeaturePipelineError(f"Type de sortie inattendu : {type(df)!r}")
    frame = pl.from_pandas(df, include_index=False)
    # pl.from_pandas peut suffixer les colonnes dupliquees : on garde le 1er exemplaire.
    if frame.width != len(set(frame.columns)):
        seen: set[str] = set()
        keep: list[str] = []
        for c in frame.columns:
            if c not in seen:
                seen.add(c)
                keep.append(c)
        frame = frame.select(keep)
    return frame

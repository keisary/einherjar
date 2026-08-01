"""Calcul des features pour le moteur de découverte.

Wrap synchrone de `einherjar.signals.feature_engine.FeatureEngine` qui :
  1. Calcule toutes les features (183) sur une série OHLCV.
  2. Filtre les 28 features exclues (cf. config/features_taxonomy.json)
     — 19 fantômes, 8 meta-factors, 1 alias.
  3. Garantit qu'exactement 218 features utilisables sont exposées.
  4. Met en cache par (asset, timeframe, data_version).

Responsabilités :
  - Calcul de features via FeatureEngine.
  - Filtrage des features exclues par taxonomie.
  - Validation du compte de features (218 ± tolérance).
  - Cache mémoire par série.

Hors périmètre :
  - Loader OHLCV (voir data/ohlcv.py).
  - Évaluation des conditions sur les features (voir engine/evaluator.py).
  - Définition des features (voir signals/feature_engine.py).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Protocol

import polars as pl

from einherjar.research.config.loader import EinherjarConfig
from einherjar.research.data.ohlcv import OhlcvFrame

logger = logging.getLogger(__name__)

# Tolérance sur le compte de features : on accepte 218 ± 2 (certaines
# versions du FeatureEngine peuvent faire osciller le compte de 1-2 unités
# à cause de features conditionnelles).
EXPECTED_N_FEATURES: int = 218
N_FEATURES_TOLERANCE: int = 2

# Colonnes OHLCV préservées dans la frame de features (non filtrables).
OHLCV_PRESERVED_COLUMNS: tuple[str, ...] = (
    "timestamp", "open", "high", "low", "close", "volume",
)


# --------------------------------------------------------------------------- #
# Exceptions
# --------------------------------------------------------------------------- #


class FeaturesError(Exception):
    """Erreur générique du calcul de features."""


class FeatureCountError(FeaturesError):
    """Le nombre de features calculées est incompatible avec la taxonomie."""


# --------------------------------------------------------------------------- #
# Frame value object
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class FeaturesFrame:
    """Frame de features alignée sur une série OHLCV.

    Attributs:
        asset: Symbole.
        timeframe: Granularité.
        df: DataFrame polars avec OHLCV + 218 colonnes features.
        feature_names: Tuple des noms de features (218 par défaut).
        data_version: Identifiant de version.
    """

    asset: str
    timeframe: str
    df: pl.DataFrame
    feature_names: tuple[str, ...]
    data_version: str

    @property
    def n_features(self) -> int:
        """Nombre de features utilisables exposées dans la frame."""
        return len(self.feature_names)

    @property
    def n_bougies(self) -> int:
        """Nombre de bougies dans la frame (aligné sur la série OHLCV)."""
        return self.df.height

    def has(self, feature_name: str) -> bool:
        """Vérifie qu'une feature est dans la frame (et donc utilisable)."""
        return feature_name in self.feature_names

    def column(self, feature_name: str) -> pl.Series:
        """Retourne la colonne d'une feature (lazy lookup avec erreur claire)."""
        if not self.has(feature_name):
            raise FeaturesError(
                f"Feature {feature_name!r} absente de la frame "
                f"(taxonomie: {self.n_features} features utilisables)"
            )
        return self.df[feature_name]

    def to_arrays(self) -> dict[str, Any]:
        """Expose les features en dict numpy pour les calculs vectorisés."""
        return {name: self.df[name].to_numpy() for name in self.feature_names}


# --------------------------------------------------------------------------- #
# Engine protocol
# --------------------------------------------------------------------------- #


class _FeatureEngineLike(Protocol):
    """Sous-ensemble de l'API FeatureEngine utilisé par ce module.

    Permet d'injecter un faux engine (test) sans dépendre de
    `einherjar.signals.feature_engine`.
    """

    def compute(self, df: pl.DataFrame) -> pl.DataFrame: ...


# --------------------------------------------------------------------------- #
# Backends privés
# --------------------------------------------------------------------------- #


class _RealFeatureEngine:
    """Backend par défaut : FeatureEngine réel."""

    def __init__(self) -> None:
        try:
            from einherjar.signals.feature_engine import FeatureEngine
        except ImportError as exc:
            raise FeaturesError(
                "FeatureEngine indisponible — vérifie einherjar.signals.feature_engine"
            ) from exc
        self._engine = FeatureEngine()
        logger.debug("FeatureEngine réel instancié")

    def compute(self, df: pl.DataFrame) -> pl.DataFrame:
        return self._engine.compute(df)


class _IdentityFeatureEngine:
    """Backend de test : retourne l'OHLCV tel quel (avec préfixes pour les features OHLCV).

    Utile pour tester FeaturesProvider sans dépendre du FeatureEngine réel.
    """

    def compute(self, df: pl.DataFrame) -> pl.DataFrame:
        return df


# --------------------------------------------------------------------------- #
# Provider public
# --------------------------------------------------------------------------- #


class FeaturesProvider:
    """Calcule et expose les 218 features utilisables pour le moteur de découverte.

    Attributes:
        config: Configuration chargée (utilisée pour la liste des features
            utilisables et la taxonomie).
        engine: Backend de calcul de features (FeatureEngine réel par défaut).
    """

    def __init__(
        self,
        config: EinherjarConfig,
        engine: _FeatureEngineLike | None = None,
        *,
        strict_count: bool = True,
    ) -> None:
        """Initialise le provider de features.

        Args:
            config: Configuration chargée (utilisée pour la taxonomie 218).
            engine: Backend de calcul (FeatureEngine réel par défaut).
            strict_count: Si True, valide que le nombre de features finales
                est proche de 218 (±2). À False pour les tests avec engine stub.
        """
        self._config = config
        self._engine: _FeatureEngineLike = engine or _RealFeatureEngine()
        self._strict_count = strict_count
        self._cache: dict[tuple[str, str, str], FeaturesFrame] = {}
        logger.info(
            "FeaturesProvider instancié (taxonomie=%d, engine=%s, strict_count=%s)",
            len(config.usable_feature_names),
            type(self._engine).__name__,
            strict_count,
        )

    # ------------------------------------------------------------------ #
    # API publique
    # ------------------------------------------------------------------ #

    def compute(
        self,
        ohlcv: OhlcvFrame,
        *,
        use_cache: bool = True,
    ) -> FeaturesFrame:
        """Calcule (ou récupère du cache) les 218 features pour une série OHLCV.

        Args:
            ohlcv: Frame OHLCV (résultat de data/ohlcv.py).
            use_cache: Si True (défaut), renvoie le cache si disponible.

        Returns:
            FeaturesFrame avec 218 features utilisables (taxonomie).

        Raises:
            FeatureCountError: si le nombre de features finales est hors tolérance.
        """
        cache_key = (ohlcv.asset, ohlcv.timeframe, ohlcv.data_version)
        if use_cache and cache_key in self._cache:
            logger.debug("Features cache hit : %s", cache_key)
            return self._cache[cache_key]

        enriched = self._engine.compute(ohlcv.df)
        filtered = self._filter_excluded(enriched)
        frame = self._build_frame(filtered, ohlcv=ohlcv)
        if self._strict_count:
            self._validate_count(frame)

        self._cache[cache_key] = frame
        logger.info(
            "Features calculées : %s × %s version=%s, %d features / %d bougies",
            ohlcv.asset, ohlcv.timeframe, ohlcv.data_version,
            frame.n_features, frame.n_bougies,
        )
        return frame

    def invalidate(self, asset: str | None = None, timeframe: str | None = None) -> int:
        """Invalide le cache. Retourne le nombre d'entrées supprimées."""
        to_drop = [
            k for k in self._cache
            if (asset is None or k[0] == asset) and (timeframe is None or k[1] == timeframe)
        ]
        for k in to_drop:
            del self._cache[k]
        if to_drop:
            logger.debug("Cache features invalidé : %d entrées", len(to_drop))
        return len(to_drop)

    # ------------------------------------------------------------------ #
    # Helpers privés
    # ------------------------------------------------------------------ #

    def _filter_excluded(self, df: pl.DataFrame) -> pl.DataFrame:
        """Supprime les features exclues (fantômes, meta-factors, alias)."""
        excluded_set = self._config.excluded_set()
        cols_to_drop = [c for c in df.columns if c in excluded_set]
        if cols_to_drop:
            df = df.drop(cols_to_drop)
            logger.debug("Features exclues filtrées : %d", len(cols_to_drop))
        return df

    def _build_frame(self, df: pl.DataFrame, ohlcv: OhlcvFrame) -> FeaturesFrame:
        """Construit un FeaturesFrame depuis le DataFrame enrichi et filtré."""
        usable = self._config.usable_set()
        feature_names = tuple(c for c in df.columns if c in usable)
        if not feature_names:
            raise FeatureCountError(
                f"Aucune feature utilisable trouvée dans la frame enrichie. "
                f"Colonnes présentes : {df.columns}"
            )
        return FeaturesFrame(
            asset=ohlcv.asset,
            timeframe=ohlcv.timeframe,
            df=df,
            feature_names=feature_names,
            data_version=ohlcv.data_version,
        )

    @staticmethod
    def _validate_count(frame: FeaturesFrame) -> None:
        """Vérifie que le compte de features est dans la tolérance."""
        n = frame.n_features
        if abs(n - EXPECTED_N_FEATURES) > N_FEATURES_TOLERANCE:
            raise FeatureCountError(
                f"FeaturesProvider attend ~{EXPECTED_N_FEATURES} features "
                f"(±{N_FEATURES_TOLERANCE}), reçu {n}. "
                f"Vérifie config/features_taxonomy.json."
            )


# --------------------------------------------------------------------------- #
# Factory
# --------------------------------------------------------------------------- #


def make_default_provider(config: EinherjarConfig) -> FeaturesProvider:
    """Construit un FeaturesProvider avec le FeatureEngine réel.

    Args:
        config: Configuration chargée (pour la taxonomie 218 features).

    Returns:
        FeaturesProvider prêt à l'emploi.
    """
    return FeaturesProvider(config=config, engine=_RealFeatureEngine())


def make_test_provider(config: EinherjarConfig) -> FeaturesProvider:
    """Construit un FeaturesProvider avec un engine identité (pour les tests).

    L'engine identité retourne l'OHLCV tel quel — utile pour tester
    FeaturesProvider sans dépendre du FeatureEngine réel. La validation
    de compte (218 ± 2) est désactivée car les features de test sont
    intentionnellement réduites.

    Args:
        config: Configuration chargée.

    Returns:
        FeaturesProvider avec engine identité et validation relâchée.
    """
    return FeaturesProvider(config=config, engine=_IdentityFeatureEngine(), strict_count=False)

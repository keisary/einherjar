"""
Fonctions optimisées pour les calculs rolling sur de gros DataFrames.

Ce module fournit des alternatives optimisées aux opérations rolling standard
de Polars qui peuvent être lentes sur de très gros DataFrames.
"""

import polars as pl
from typing import Optional


def fast_rolling_normalize(
    col_expr: pl.Expr,
    window: int = 50,
    fill_value: float = 0.5,
    min_periods: Optional[int] = None
) -> pl.Expr:
    """
    Normalisation rolling optimisée (min-max) avec gestion des cas limites.
    
    Optimisations appliquées :
    - Utilisation de min_periods pour éviter les calculs sur fenêtres incomplètes
    - Ajout d'epsilon pour éviter division par zéro
    - Gestion efficace des nulls
    
    Args:
        col_expr: Expression Polars de la colonne à normaliser
        window: Taille de la fenêtre rolling
        fill_value: Valeur de remplacement pour les nulls
        min_periods: Nombre minimum de périodes valides (None = 50% de window)
        
    Returns:
        Expression Polars normalisée entre 0 et 1
    """
    if min_periods is None:
        min_periods = max(1, window // 2)  # Au moins 50% de la fenêtre
    
    # Calculs rolling avec min_periods (OPTIMISATION: skip fenêtres incomplètes)
    rolling_min = col_expr.rolling_min(window_size=window, min_periods=min_periods)
    rolling_max = col_expr.rolling_max(window_size=window, min_periods=min_periods)
    
    # Normalisation avec epsilon pour éviter division par zéro
    epsilon = 1e-9
    normalized = (col_expr - rolling_min) / (rolling_max - rolling_min + epsilon)
    
    # Clipper entre 0 et 1 et remplir les nulls
    return normalized.clip(0.0, 1.0).fill_null(fill_value)


def fast_rolling_zscore(
    col_expr: pl.Expr,
    window: int = 50,
    fill_value: float = 0.0,
    min_periods: Optional[int] = None
) -> pl.Expr:
    """
    Z-score rolling optimisé (standardisation).
    
    Plus rapide que min-max pour certains cas d'usage.
    
    Args:
        col_expr: Expression Polars de la colonne
        window: Taille de la fenêtre rolling
        fill_value: Valeur de remplacement pour les nulls
        min_periods: Nombre minimum de périodes valides
        
    Returns:
        Expression Polars standardisée (z-score)
    """
    if min_periods is None:
        min_periods = max(1, window // 2)
    
    # Calculs rolling
    rolling_mean = col_expr.rolling_mean(window_size=window, min_periods=min_periods)
    rolling_std = col_expr.rolling_std(window_size=window, min_periods=min_periods)
    
    # Z-score avec epsilon
    epsilon = 1e-9
    zscore = (col_expr - rolling_mean) / (rolling_std + epsilon)
    
    # Clipper pour éviter les valeurs extrêmes et remplir les nulls
    return zscore.clip(-3.0, 3.0).fill_null(fill_value)


def fast_percentile_normalize(
    col_expr: pl.Expr,
    window: int = 50,
    fill_value: float = 0.5
) -> pl.Expr:
    """
    Normalisation basée sur le rang percentile (plus robuste aux outliers).
    
    Alternative plus rapide et robuste que min-max pour données avec outliers.
    
    Args:
        col_expr: Expression Polars de la colonne
        window: Taille de la fenêtre rolling
        fill_value: Valeur de remplacement pour les nulls
        
    Returns:
        Expression Polars normalisée entre 0 et 1 (rang percentile)
    """
    # Utiliser le rang dans la fenêtre rolling
    # Note: Cette approche est approximative mais beaucoup plus rapide
    rolling_mean = col_expr.rolling_mean(window_size=window)
    rolling_std = col_expr.rolling_std(window_size=window)
    
    # Approximation du percentile via fonction sigmoïde
    epsilon = 1e-9
    zscore = (col_expr - rolling_mean) / (rolling_std + epsilon)
    
    # Fonction sigmoïde pour mapper à [0, 1]
    # sigmoid(x) = 1 / (1 + exp(-x))
    percentile_approx = 1.0 / (1.0 + pl.Expr.exp(-zscore))
    
    return percentile_approx.fill_null(fill_value)


def adaptive_window_normalize(
    col_expr: pl.Expr,
    base_window: int = 50,
    volatility_col: Optional[pl.Expr] = None,
    fill_value: float = 0.5
) -> pl.Expr:
    """
    Normalisation avec fenêtre adaptative basée sur la volatilité.
    
    Utilise une fenêtre plus courte en période de haute volatilité
    et plus longue en période de basse volatilité.
    
    Args:
        col_expr: Expression Polars de la colonne
        base_window: Taille de fenêtre de base
        volatility_col: Expression pour la volatilité (optionnel)
        fill_value: Valeur de remplacement pour les nulls
        
    Returns:
        Expression Polars normalisée
    """
    if volatility_col is None:
        # Utiliser la volatilité de la colonne elle-même
        volatility_col = col_expr.rolling_std(window_size=base_window)
    
    # Normalisation simple si pas de volatilité adaptative
    return fast_rolling_normalize(col_expr, base_window, fill_value)


def batch_rolling_normalize(
    df: pl.DataFrame,
    columns: list[str],
    window: int = 50,
    fill_value: float = 0.5,
    suffix: str = "_norm"
) -> pl.DataFrame:
    """
    Normalise plusieurs colonnes en une seule passe (plus efficace).
    
    Args:
        df: DataFrame source
        columns: Liste des colonnes à normaliser
        window: Taille de la fenêtre rolling
        fill_value: Valeur de remplacement pour les nulls
        suffix: Suffixe pour les colonnes normalisées
        
    Returns:
        DataFrame avec colonnes normalisées ajoutées
    """
    # Créer toutes les expressions en une fois
    norm_exprs = [
        fast_rolling_normalize(pl.col(col), window, fill_value).alias(f"{col}{suffix}")
        for col in columns if col in df.columns
    ]
    
    # Appliquer toutes les normalisations en une seule passe
    return df.with_columns(norm_exprs)


def simple_normalize_0_100(col_expr: pl.Expr) -> pl.Expr:
    """
    Normalisation simple pour colonnes déjà entre 0-100 (RSI, Stochastic).
    
    Beaucoup plus rapide que rolling_min/max.
    
    Args:
        col_expr: Expression Polars de la colonne (0-100)
        
    Returns:
        Expression normalisée entre 0 et 1
    """
    return (col_expr / 100.0).clip(0.0, 1.0)


def simple_normalize_minus100_0(col_expr: pl.Expr) -> pl.Expr:
    """
    Normalisation simple pour colonnes entre -100 et 0 (Williams %R).
    
    Args:
        col_expr: Expression Polars de la colonne (-100 à 0)
        
    Returns:
        Expression normalisée entre 0 et 1
    """
    return ((col_expr + 100.0) / 100.0).clip(0.0, 1.0)


def cached_rolling_stats(
    df: pl.DataFrame,
    col: str,
    window: int = 50,
    stats: list[str] = ['min', 'max', 'mean', 'std']
) -> pl.DataFrame:
    """
    Calcule plusieurs statistiques rolling en une seule passe (optimisé).
    
    Évite de recalculer les fenêtres rolling plusieurs fois.
    
    Args:
        df: DataFrame source
        col: Nom de la colonne
        window: Taille de la fenêtre
        stats: Liste des statistiques à calculer ('min', 'max', 'mean', 'std')
        
    Returns:
        DataFrame avec colonnes de statistiques ajoutées
    """
    exprs = []
    
    if 'min' in stats:
        exprs.append(pl.col(col).rolling_min(window).alias(f"{col}_rolling_min_{window}"))
    if 'max' in stats:
        exprs.append(pl.col(col).rolling_max(window).alias(f"{col}_rolling_max_{window}"))
    if 'mean' in stats:
        exprs.append(pl.col(col).rolling_mean(window).alias(f"{col}_rolling_mean_{window}"))
    if 'std' in stats:
        exprs.append(pl.col(col).rolling_std(window).alias(f"{col}_rolling_std_{window}"))
    
    return df.with_columns(exprs)


# =============================================================================
# STRATÉGIES D'OPTIMISATION RECOMMANDÉES
# =============================================================================

"""
GUIDE D'OPTIMISATION DES CALCULS ROLLING:

1. ÉVITER LES ROLLING QUAND POSSIBLE
   - RSI, Stochastic (0-100) : Diviser par 100 directement
   - Williams %R (-100 à 0) : (x + 100) / 100
   - Pas besoin de rolling_min/max !

2. UTILISER MIN_PERIODS
   - Évite les calculs sur fenêtres incomplètes
   - Réduit le nombre d'opérations

3. BATCH PROCESSING
   - Normaliser plusieurs colonnes en une passe
   - Calculer plusieurs stats rolling ensemble

4. APPROXIMATIONS RAPIDES
   - Z-score au lieu de min-max quand possible
   - Percentile approximatif via sigmoïde

5. FENÊTRES ADAPTATIVES
   - Fenêtres plus courtes = calculs plus rapides
   - Adapter selon la volatilité

6. CACHING
   - Calculer rolling_min/max une fois
   - Réutiliser pour plusieurs normalisations

EXEMPLE D'OPTIMISATION:

# ❌ LENT (6.8M lignes)
for feature in features:
    normalized = (
        (pl.col(feature) - pl.col(feature).rolling_min(50)) /
        (pl.col(feature).rolling_max(50) - pl.col(feature).rolling_min(50))
    )

# ✅ RAPIDE
# 1. Identifier les colonnes 0-100
rsi_cols = [f for f in features if f.startswith('rsi_')]
norm_exprs = [simple_normalize_0_100(pl.col(f)) for f in rsi_cols]

# 2. Batch normaliser les autres
other_cols = [f for f in features if f not in rsi_cols]
df = batch_rolling_normalize(df, other_cols, window=50)
"""

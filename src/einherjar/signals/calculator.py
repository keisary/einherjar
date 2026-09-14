""""
Calculateur des facteurs alpha.

Ce module transforme les features techniques brutes en facteurs alpha agrégés
par famille (MOMENTUM, TREND, VOLUME, etc.).
"""

import logging
import gc
from typing import Dict, Any, List

import polars as pl

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

from einherjar.signals.alpha_factor_config import AlphaFactorConfig, AlphaFactorError
from einherjar.signals.optimized_rolling import (
    fast_rolling_normalize,
    simple_normalize_0_100,
    simple_normalize_minus100_0,
)


class AlphaFactorCalculator:
    """
    Calculateur des facteurs alpha à partir des features techniques brutes.
    
    Transforme les indicateurs techniques individuels en scores agrégés
    par famille pour faciliter la génération de stratégies.
    
    ARCHITECTURE EN DEUX ÉTAPES :
    1. calculate_base_factors() : Facteurs de base (sur DataFrame unique)
    2. calculate_sophisticated_factors() : Facteurs sophistiqués (sur DataFrame unifié)
    """
    
    def __init__(self):
        """
        Initialise le calculateur.
        
        Le streaming et la gestion mémoire sont gérés automatiquement par l'orchestrateur.
        """
        self.logger = logging.getLogger(__name__)
        self.config = AlphaFactorConfig()
        
        # Classification des facteurs par étape de calcul
        self.base_factors = [
            'Factor_Momentum_Score',
            'Factor_Trend_Score', 
            'Factor_Volume_Score',
            'Factor_Volatility_Score',
            'Factor_Quantitative_Score',
            'Factor_Candlestick_Bullish_Score',
            'Factor_Candlestick_Bearish_Score',
            'Factor_Chart_Patterns_Score',
            'Factor_Harmonic_Patterns_Score',
            'Factor_Support_Resistance_Score'
        ]
        
        self.sophisticated_factors = [
            'Factor_Regime_Hurst_Score',
            'Factor_Risk_TailEvent_Score',
            'Factor_Persistence_Score',
            'Factor_RelativeStrength_Momentum',
            'Factor_Correlation_Dynamic',
            'Factor_Momentum_Acceleration_Score',
            'Factor_Volatility_Expansion_Rate',
            'Factor_Divergence_Score_Bearish',
            'Factor_Divergence_Score_Bullish',
            'Factor_Trend_Confirmation_Score',
            'Factor_Reversal_Climax_Score'
        ]
    
    # ========== MÉTHODES PRINCIPALES ==========
    
    def calculate_base_factors(self, ldf: pl.LazyFrame) -> pl.LazyFrame:
        """
        ÉTAPE 1: Construit le plan de calcul pour les facteurs de base en mode 100% LAZY.
        
        Cette méthode calcule les 10 facteurs de base qui ne dépendent que des données
        d'un seul actif/timeframe. Elle doit être appelée AVANT la fusion des DataFrames.
        
        OPTIMISATION: Travaille directement sur LazyFrame sans matérialisation.
        Les calculs ne seront effectués qu'au moment du .sink_parquet() final.
        
        Args:
            ldf: LazyFrame avec les features techniques brutes d'un seul actif
            
        Returns:
            LazyFrame enrichi avec les expressions de calcul des facteurs
        """
        self.logger.info("ÉTAPE 1: Construction du plan de calcul LAZY pour les facteurs de base...")
        
        try:
            # 🔥 CORRECTION CRITIQUE: Collecter le schéma UNE SEULE FOIS
            schema = ldf.collect_schema()
            self.logger.info(f"Colonnes du plan initial: {len(schema)}")
            
            # La méthode interne gère maintenant les LazyFrames
            return self._calculate_base_factors_direct(ldf)
            
        except Exception as e:
            raise AlphaFactorError(f"Erreur lors de la construction du plan lazy pour les facteurs de base: {str(e)}")
    
    def _calculate_base_factors_direct(self, ldf: pl.LazyFrame) -> pl.LazyFrame:
        """
        APPROCHE LAZY ROBUSTE: Collecte le schéma UNE FOIS puis collecte toutes les expressions.
        
        Cette méthode crée un plan d'exécution "large" (une seule opération with_columns avec
        de nombreuses colonnes) plutôt qu'un plan "profond" (chaîne d'opérations). L'optimiseur
        de Polars gère beaucoup mieux les plans larges.
        
        Args:
            ldf: LazyFrame à traiter
            
        Returns:
            LazyFrame enrichi avec toutes les expressions de facteurs de base
        """
        import time
        from typing import List
        
        self.logger.info("📝 Collecte UNIQUE du schéma et des expressions...")
        start_time = time.time()
        
        # 🔥 CORRECTION CRITIQUE: Collecter le schéma UNE SEULE FOIS
        schema = ldf.collect_schema()
        available_columns = set(schema.names())
        self.logger.info(f"  ✓ Schéma collecté: {len(available_columns)} colonnes")
        
        all_expressions: List[pl.Expr] = []
        
        # Chaque méthode de calcul reçoit maintenant les colonnes disponibles
        for idx, factor_name in enumerate(self.base_factors, 1):
            if factor_name not in self.config.ALPHA_FACTORS:
                self.logger.warning(f"Facteur {factor_name} non défini")
                continue
            
            factor_config = self.config.ALPHA_FACTORS[factor_name]
            family = factor_config['family']
            
            self.logger.info(f"  {idx}/10: Collecte pour {family}")
            
            try:
                expressions: List[pl.Expr] = []
                if family == 'MOMENTUM':
                    expressions = self._calculate_momentum_factors(ldf, factor_name, factor_config, available_columns)
                elif family == 'TREND':
                    expressions = self._calculate_trend_factors(ldf, factor_name, factor_config, available_columns)
                elif family == 'VOLUME':
                    expressions = self._calculate_volume_factors(ldf, factor_name, factor_config, available_columns)
                elif family == 'VOLATILITY':
                    expressions = self._calculate_volatility_factors(ldf, factor_name, factor_config, available_columns)
                elif family == 'QUANTITATIVE':
                    expressions = self._calculate_quantitative_factors(ldf, factor_name, factor_config, available_columns)
                elif family == 'CANDLESTICK_PATTERNS':
                    expressions = self._calculate_candlestick_factors(ldf, factor_name, factor_config, available_columns)
                elif family == 'CHART_PATTERNS':
                    expressions = self._calculate_chart_pattern_factors(ldf, factor_name, factor_config, available_columns)
                elif family == 'HARMONIC_PATTERNS':
                    expressions = self._calculate_harmonic_factors(ldf, factor_name, factor_config, available_columns)
                elif family == 'SUPPORT_RESISTANCE':
                    expressions = self._calculate_support_resistance_factors(ldf, factor_name, factor_config, available_columns)
                else:
                    self.logger.warning(f"Famille non reconnue: {family}")
                
                if expressions:
                    all_expressions.extend(expressions)
                    self.logger.info(f"    ✓ {len(expressions)} expressions collectées")
            
            except Exception as e:
                self.logger.error(f"❌ Erreur collecte {factor_name}: {str(e)}")
                raise AlphaFactorError(f"Erreur collecte {factor_name}: {str(e)}")
        
        collection_time = time.time() - start_time
        self.logger.info(f"✓ Collecte terminée: {len(all_expressions)} expressions en {collection_time:.1f}s")
        
        # Appliquer TOUTES les expressions EN UNE SEULE FOIS
        self.logger.info(f"🚀 Application des expressions...")
        final_ldf = ldf.with_columns(all_expressions)
        
        # 🔥 ÉVITER l'accès à final_ldf.columns ici
        final_schema = final_ldf.collect_schema()
        self.logger.info(f"✅ Plan calculé avec {len(final_schema)} colonnes.")
        
        return final_ldf
    

    
    def get_sophisticated_factors_expressions(self, available_columns: set) -> List[pl.Expr]:
        """
        Retourne UNIQUEMENT les expressions des facteurs sophistiqués, sans les appliquer.
        
        Cette méthode permet de séparer la construction des expressions de leur application,
        pour un traitement par chunks robuste.
        
        Args:
            available_columns: Ensemble des colonnes disponibles dans le DataFrame source
            
        Returns:
            Liste d'expressions Polars à appliquer
        """
        self.logger.info("Collecte des expressions pour facteurs sophistiqués...")
        
        all_expressions: List[pl.Expr] = []
        
        for idx, factor_name in enumerate(self.sophisticated_factors, 1):
            if factor_name not in self.config.ALPHA_FACTORS:
                self.logger.warning(f"Facteur sophistiqué {factor_name} non défini")
                continue
            
            factor_config = self.config.ALPHA_FACTORS[factor_name]
            calculation_type = factor_config.get('calculation_type')
            
            self.logger.debug(f"  {idx}/11: {factor_name}")
            
            try:
                expressions: List[pl.Expr] = []
                
                if calculation_type == 'regime_analysis':
                    expressions = self._get_regime_expressions(factor_name, factor_config, available_columns)
                elif calculation_type == 'risk_analysis':
                    expressions = self._get_risk_expressions(factor_name, factor_config, available_columns)
                elif calculation_type == 'persistence_analysis':
                    expressions = self._get_persistence_expressions(factor_name, factor_config, available_columns)
                elif calculation_type == 'cross_asset_relative':
                    expressions = self._get_cross_asset_relative_expressions(factor_name, factor_config, available_columns)
                elif calculation_type == 'cross_asset_correlation':
                    expressions = self._get_cross_asset_correlation_expressions(factor_name, factor_config, available_columns)
                elif calculation_type == 'second_order_derivative':
                    expressions = self._get_second_order_expressions(factor_name, factor_config, available_columns)
                elif calculation_type == 'divergence_analysis':
                    expressions = self._get_divergence_expressions(factor_name, factor_config, available_columns)
                elif calculation_type == 'confluence_analysis':
                    expressions = self._get_confluence_expressions(factor_name, factor_config, available_columns)
                elif calculation_type == 'climax_analysis':
                    expressions = self._get_climax_expressions(factor_name, factor_config, available_columns)
                else:
                    self.logger.warning(f"Type non reconnu: {calculation_type}")
                    continue
                
                if expressions:
                    all_expressions.extend(expressions)
                
            except Exception as e:
                self.logger.error(f"❌ Erreur collecte {factor_name}: {str(e)}")
                # Continuer avec les autres facteurs
                continue
        
        self.logger.info(f"✓ {len(all_expressions)} expressions collectées")
        
        return all_expressions
    
    def calculate_sophisticated_factors_lazy(self, unified_ldf: pl.LazyFrame) -> pl.LazyFrame:
        """
        ÉTAPE 2: Construit le plan de calcul LAZY pour les facteurs sophistiqués.
        
        APPROCHE 100% LAZY: Utilise des expressions Polars pures, exactement comme les facteurs de base.
        Aucune matérialisation - tout reste en mode lazy jusqu'au sink_parquet final.
        
        Cette méthode calcule les facteurs avancés qui dépendent soit d'autres facteurs
        (dérivés), soit de données multi-actifs (corrélations, forces relatives).
        Elle doit être appelée APRÈS la fusion et le préfixage des DataFrames.
        
        Args:
            unified_ldf: LazyFrame unifié avec tous les actifs préfixés + facteurs de base
            
        Returns:
            LazyFrame avec les colonnes originales + tous les facteurs alpha (plan d'exécution)
        """
        self.logger.info("ÉTAPE 2: Construction du plan LAZY pour facteurs sophistiqués...")
        
        try:
            import time
            start_time = time.time()
            
            # Collecter le schéma UNE SEULE FOIS
            schema = unified_ldf.collect_schema()
            available_columns = set(schema.names())
            self.logger.info(f"Colonnes disponibles: {len(available_columns)}")
            
            # Collecter toutes les expressions (comme pour les facteurs de base)
            self.logger.info("📝 Collecte des expressions pour tous les facteurs sophistiqués...")
            all_expressions: List[pl.Expr] = []
            
            for idx, factor_name in enumerate(self.sophisticated_factors, 1):
                if factor_name not in self.config.ALPHA_FACTORS:
                    self.logger.warning(f"Facteur sophistiqué {factor_name} non défini")
                    continue
                
                factor_config = self.config.ALPHA_FACTORS[factor_name]
                calculation_type = factor_config.get('calculation_type')
                
                self.logger.info(f"  {idx}/11: Collecte expressions pour {factor_name}")
                
                try:
                    expressions: List[pl.Expr] = []
                    
                    # Appeler les méthodes qui retournent des expressions (pas des DataFrames)
                    if calculation_type == 'regime_analysis':
                        expressions = self._get_regime_expressions(factor_name, factor_config, available_columns)
                    elif calculation_type == 'risk_analysis':
                        expressions = self._get_risk_expressions(factor_name, factor_config, available_columns)
                    elif calculation_type == 'persistence_analysis':
                        expressions = self._get_persistence_expressions(factor_name, factor_config, available_columns)
                    elif calculation_type == 'cross_asset_relative':
                        expressions = self._get_cross_asset_relative_expressions(factor_name, factor_config, available_columns)
                    elif calculation_type == 'cross_asset_correlation':
                        expressions = self._get_cross_asset_correlation_expressions(factor_name, factor_config, available_columns)
                    elif calculation_type == 'second_order_derivative':
                        expressions = self._get_second_order_expressions(factor_name, factor_config, available_columns)
                    elif calculation_type == 'divergence_analysis':
                        expressions = self._get_divergence_expressions(factor_name, factor_config, available_columns)
                    elif calculation_type == 'confluence_analysis':
                        expressions = self._get_confluence_expressions(factor_name, factor_config, available_columns)
                    elif calculation_type == 'climax_analysis':
                        expressions = self._get_climax_expressions(factor_name, factor_config, available_columns)
                    else:
                        self.logger.warning(f"Type non reconnu: {calculation_type}")
                        continue
                    
                    if expressions:
                        all_expressions.extend(expressions)
                        self.logger.info(f"    ✓ {len(expressions)} expressions collectées")
                    
                except Exception as e:
                    self.logger.error(f"❌ Erreur collecte {factor_name}: {str(e)}")
                    # Continuer avec les autres facteurs au lieu de crasher
                    continue
            
            collection_time = time.time() - start_time
            self.logger.info(f"✓ Collecte terminée: {len(all_expressions)} expressions en {collection_time:.1f}s")
            
            # Appliquer TOUTES les expressions EN UNE SEULE FOIS (comme les facteurs de base)
            self.logger.info(f"🚀 Application de {len(all_expressions)} expressions au plan lazy...")
            result_ldf = unified_ldf.with_columns(all_expressions)
            
            self.logger.info(f"✅ Plan de calcul complet construit (toujours 0 MB RAM)")
            
            return result_ldf
            
        except Exception as e:
            raise AlphaFactorError(f"Erreur lors de la construction du plan lazy: {str(e)}")
    
    def calculate_sophisticated_factors(self, unified_df: pl.DataFrame) -> pl.DataFrame:
        """
        ÉTAPE 2: Calcule les facteurs alpha sophistiqués sur le DataFrame unifié.
        
        ANCIENNE APPROCHE (DEPRECATED): Utiliser calculate_sophisticated_factors_lazy() à la place.
        
        Cette méthode calcule les facteurs avancés qui dépendent soit d'autres facteurs
        (dérivés), soit de données multi-actifs (corrélations, forces relatives).
        Elle doit être appelée APRÈS la fusion et le préfixage des DataFrames.
        
        Args:
            unified_df: DataFrame unifié avec tous les actifs préfixés + facteurs de base
            
        Returns:
            DataFrame avec les colonnes originales + tous les facteurs alpha
        """
        self.logger.info("ÉTAPE 2: Calcul des facteurs alpha sophistiqués...")
        
        try:
            import time
            
            # Vérification des colonnes disponibles
            available_columns = set(unified_df.columns)
            self.logger.info(f"Colonnes disponibles dans le DataFrame unifié: {len(available_columns)}")
            
            start_time = time.time()
            
            # ÉTAPE 1: Collecter toutes les expressions
            self.logger.info("📝 Collecte des expressions pour tous les facteurs sophistiqués...")
            all_expressions = []
            
            for idx, factor_name in enumerate(self.sophisticated_factors, 1):
                if factor_name not in self.config.ALPHA_FACTORS:
                    self.logger.warning(f"Facteur sophistiqué {factor_name} non défini")
                    continue
                
                factor_config = self.config.ALPHA_FACTORS[factor_name]
                calculation_type = factor_config.get('calculation_type')
                
                self.logger.info(f"  {idx}/11: Collecte expressions pour {factor_name}")
                
                try:
                    # Collecter les expressions selon le type
                    # Pour l'instant, on utilise les anciennes méthodes et on extrait les nouvelles colonnes
                    temp_df = unified_df.clone()
                    
                    if calculation_type == 'regime_analysis':
                        temp_df = self._calculate_regime_factors(temp_df, factor_name, factor_config)
                    elif calculation_type == 'risk_analysis':
                        temp_df = self._calculate_risk_factors(temp_df, factor_name, factor_config)
                    elif calculation_type == 'persistence_analysis':
                        temp_df = self._calculate_persistence_factors(temp_df, factor_name, factor_config)
                    elif calculation_type == 'cross_asset_relative':
                        temp_df = self._calculate_cross_asset_relative_factors(temp_df, factor_name, factor_config)
                    elif calculation_type == 'cross_asset_correlation':
                        temp_df = self._calculate_cross_asset_correlation_factors(temp_df, factor_name, factor_config)
                    elif calculation_type == 'second_order_derivative':
                        temp_df = self._calculate_second_order_factors(temp_df, factor_name, factor_config)
                    elif calculation_type == 'divergence_analysis':
                        temp_df = self._calculate_divergence_factors(temp_df, factor_name, factor_config)
                    elif calculation_type == 'confluence_analysis':
                        temp_df = self._calculate_confluence_factors(temp_df, factor_name, factor_config)
                    elif calculation_type == 'climax_analysis':
                        temp_df = self._calculate_climax_factors(temp_df, factor_name, factor_config)
                    else:
                        self.logger.warning(f"Type non reconnu: {calculation_type}")
                        continue
                    
                    # Extraire les nouvelles colonnes comme expressions
                    new_cols = [col for col in temp_df.columns if col not in unified_df.columns]
                    for col in new_cols:
                        all_expressions.append(temp_df[col].alias(col))
                    
                    del temp_df
                    gc.collect()
                    
                    self.logger.info(f"    ✓ {len(new_cols)} expressions collectées")
                    
                except Exception as e:
                    self.logger.error(f"❌ Erreur collecte {factor_name}: {str(e)}")
                    raise AlphaFactorError(f"Erreur collecte {factor_name}: {str(e)}")
            
            collection_time = time.time() - start_time
            self.logger.info(f"✓ Collecte terminée: {len(all_expressions)} expressions en {collection_time:.1f}s")
            
            # ÉTAPE 2: Appliquer TOUTES les expressions EN UNE SEULE FOIS
            self.logger.info(f"🚀 Application de {len(all_expressions)} expressions en une seule opération...")
            apply_start = time.time()
            
            result_df = unified_df.with_columns(all_expressions)
            
            apply_time = time.time() - apply_start
            total_time = time.time() - start_time
            
            # Libération mémoire
            del unified_df
            for _ in range(3):
                gc.collect()
            
            if PSUTIL_AVAILABLE:
                process = psutil.Process()
                mem_mb = process.memory_info().rss / 1024 / 1024
                self.logger.info(f"✅ Facteurs sophistiqués calculés - RAM: {mem_mb:.0f} MB")
            
            self.logger.info(f"⏱️  Temps collecte: {collection_time:.1f}s")
            self.logger.info(f"⏱️  Temps application: {apply_time:.1f}s")
            self.logger.info(f"⏱️  Temps total: {total_time:.1f}s")
            
            # Validation
            sophisticated_factors_count = len([col for col in result_df.columns 
                                             if col.startswith('Factor_') and col in self.sophisticated_factors])
            self.logger.info(f"✅ {sophisticated_factors_count} facteurs sophistiqués créés")
            
            total_factors_count = len([col for col in result_df.columns if col.startswith('Factor_')])
            self.logger.info(f"✅ Calcul complet terminé - {total_factors_count} facteurs alpha au total")
            
            return result_df
            
        except Exception as e:
            raise AlphaFactorError(f"Erreur lors du calcul des facteurs sophistiqués: {str(e)}")
    
    def _calculate_momentum_factors(self, ldf: pl.LazyFrame, factor_name: str, 
                                  factor_config: Dict[str, Any], available_columns: set) -> List[pl.Expr]:
        """Retourne les expressions pour les facteurs de momentum."""
        self.logger.debug(f"Construction plan momentum: {factor_name}")
        
        features = factor_config['features']
        available_features = [f for f in features if f in available_columns]
        
        if not available_features:
            self.logger.warning(f"Aucune feature disponible pour {factor_name}")
            return [pl.lit(0.0).alias(factor_name)]
        
        # OPTIMISATION: Séparer par type pour éviter rolling inutiles
        rsi_features = [f for f in available_features if f.startswith('rsi_')]
        stoch_features = [f for f in available_features if f in ['stoch_k', 'stoch_d']]
        williams_features = [f for f in available_features if f == 'williams_r']
        other_features = [f for f in available_features 
                         if f not in rsi_features + stoch_features + williams_features]
        
        self.logger.info(f"  {len(rsi_features)} RSI, {len(stoch_features)} Stoch, {len(other_features)} autres")
        
        # Normalisation OPTIMISÉE (0-1)
        normalized_features = []
        
        # RSI et Stochastic: Division simple (RAPIDE - pas de rolling!)
        for feature in rsi_features + stoch_features:
            normalized = simple_normalize_0_100(pl.col(feature)).alias(f"{feature}_norm")
            normalized_features.append(normalized)
        
        # Williams %R: Normalisation simple
        for feature in williams_features:
            normalized = simple_normalize_minus100_0(pl.col(feature)).alias(f"{feature}_norm")
            normalized_features.append(normalized)
        
        # Autres: Rolling optimisé avec min_periods
        for feature in other_features:
            normalized = fast_rolling_normalize(
                pl.col(feature), 
                window=50, 
                fill_value=0.5,
                min_periods=25  # 50% de la fenêtre minimum
            ).alias(f"{feature}_norm")
            normalized_features.append(normalized)
        
        # Calcul du score agrégé avec pondération
        if len(normalized_features) == 1:
            momentum_score = normalized_features[0]
        else:
            # OPTIMISATION CRITIQUE: Utiliser sum_horizontal au lieu d'accumulation
            # Moyenne pondérée (RSI et Stochastic ont plus de poids)
            weights = []
            for feature in available_features:
                if feature.startswith('rsi_'):
                    weights.append(0.3)  # Poids élevé pour RSI
                elif feature in ['stoch_k', 'stoch_d']:
                    weights.append(0.25)  # Poids élevé pour Stochastic
                else:
                    weights.append(0.1)   # Poids standard
            
            # Normaliser les poids
            total_weight = sum(weights)
            weights = [w / total_weight for w in weights]
            
            # Construire les expressions pondérées
            weighted_exprs = [norm_feature * weights[i] for i, norm_feature in enumerate(normalized_features)]
            
            # Utiliser sum_horizontal pour éviter l'accumulation
            momentum_score = pl.sum_horizontal(weighted_exprs).alias(factor_name)
        
        return normalized_features + [momentum_score]
    
    def _calculate_trend_factors(self, ldf: pl.LazyFrame, factor_name: str,
                               factor_config: Dict[str, Any], available_columns: set) -> List[pl.Expr]:
        """Retourne les expressions pour les facteurs de tendance."""
        self.logger.debug(f"Construction plan tendance: {factor_name}")
        
        features = factor_config['features']
        available_features = [f for f in features if f in available_columns]
        
        if not available_features:
            self.logger.warning(f"Aucune feature disponible pour {factor_name}")
            return [pl.lit(0.5).alias(factor_name)]  # Neutre pour tendance
        
        # Calcul des signaux de tendance
        trend_signals = []
        
        # 1. Signaux des moyennes mobiles
        ma_signals = []
        if 'close' in available_columns:
            for feature in available_features:
                if feature.startswith(('sma_', 'ema_')):
                    # Signal: prix au-dessus de la MA = haussier (1), en-dessous = baissier (0)
                    signal = (pl.col('close') > pl.col(feature)).cast(pl.Float64).alias(f"{feature}_signal")
                    ma_signals.append(signal)
        
        # 2. Signal MACD
        macd_signal = None
        if all(col in available_features for col in ['macd_line', 'macd_signal']):
            # MACD au-dessus du signal = haussier
            macd_signal = (pl.col('macd_line') > pl.col('macd_signal')).cast(pl.Float64).alias("macd_trend_signal")
        
        # 3. Signal ADX (force de tendance)
        adx_signal = None
        if 'adx_14' in available_features:
            # ADX > 25 = tendance forte, normaliser entre 0 et 1
            adx_signal = (pl.col('adx_14') / 100.0).clip(0.0, 1.0).alias("adx_strength_signal")
        
        # 4. Signaux directionnels (DI+ vs DI-)
        di_signal = None
        if all(col in available_features for col in ['di_plus', 'di_minus']):
            # DI+ > DI- = tendance haussière
            di_signal = (pl.col('di_plus') > pl.col('di_minus')).cast(pl.Float64).alias("di_trend_signal")
        
        # 5. Signal Aroon
        aroon_signal = None
        if all(col in available_features for col in ['aroon_up', 'aroon_down']):
            # Aroon Up > Aroon Down = tendance haussière
            aroon_signal = (pl.col('aroon_up') > pl.col('aroon_down')).cast(pl.Float64).alias("aroon_trend_signal")
        
        # Agrégation des signaux
        all_signals = ma_signals.copy()
        if macd_signal is not None:
            all_signals.append(macd_signal)
        if di_signal is not None:
            all_signals.append(di_signal)
        if aroon_signal is not None:
            all_signals.append(aroon_signal)
        
        if not all_signals:
            return [pl.lit(0.5).alias(factor_name)]
        
        # Calcul du score de tendance
        if len(all_signals) == 1:
            trend_score = all_signals[0].alias(factor_name)
        else:
            # Moyenne des signaux avec pondération par la force ADX si disponible
            base_score = pl.mean_horizontal([signal for signal in all_signals])
            
            if adx_signal is not None:
                # Pondérer par la force de tendance ADX
                trend_score = (base_score * (0.5 + adx_signal * 0.5)).alias(factor_name)
            else:
                trend_score = base_score.alias(factor_name)
        
        # Retourner tous les signaux intermédiaires et le score final
        columns_to_add = all_signals + [trend_score]
        if adx_signal is not None:
            columns_to_add.append(adx_signal)
        return columns_to_add
    
    def _calculate_volume_factors(self, ldf: pl.LazyFrame, factor_name: str,
                                factor_config: Dict[str, Any], available_columns: set) -> List[pl.Expr]:
        """Retourne les expressions pour les facteurs de volume."""
        self.logger.debug(f"Construction plan volume: {factor_name}")
        
        features = factor_config['features']
        available_features = [f for f in features if f in available_columns]
        
        if not available_features:
            self.logger.warning(f"Aucune feature disponible pour {factor_name}")
            return [pl.lit(0.5).alias(factor_name)]
        
        volume_signals = []
        
        # 1. Signal de ratio de volume
        if 'volume_ratio' in available_features:
            # Ratio > 1 = volume élevé, normaliser
            vol_ratio_signal = (pl.col('volume_ratio') - 1.0).clip(0.0, 2.0) / 2.0
            volume_signals.append(vol_ratio_signal.alias("volume_ratio_signal"))
        
        # 2. Signal OBV (On Balance Volume)
        if 'obv' in available_features:
            # OBV croissant = signal positif
            obv_signal = (
                (pl.col('obv') > pl.col('obv').shift(1)).cast(pl.Float64) * 0.6 +
                (pl.col('obv') > pl.col('obv').rolling_mean(10)).cast(pl.Float64) * 0.4
            ).alias("obv_signal")
            volume_signals.append(obv_signal)
        
        # 3. Signal OBV EMA
        if 'obv_ema' in available_features:
            obv_ema_signal = (
                (pl.col('obv_ema') > pl.col('obv_ema').shift(1)).cast(pl.Float64)
            ).alias("obv_ema_signal")
            volume_signals.append(obv_ema_signal)
        
        # 4. Signal VWAP
        if 'vwap' in available_features and 'close' in available_columns:
            # Prix au-dessus du VWAP = signal positif
            vwap_signal = (pl.col('close') > pl.col('vwap')).cast(pl.Float64).alias("vwap_signal")
            volume_signals.append(vwap_signal)
        
        # 5. Signal de volume SMA
        if 'volume_sma_20' in available_features and 'volume' in available_columns:
            # Volume actuel vs moyenne
            vol_sma_signal = (
                (pl.col('volume') / pl.col('volume_sma_20')).clip(0.5, 2.0) - 0.5
            ) / 1.5  # Normaliser entre 0 et 1
            volume_signals.append(vol_sma_signal.alias("volume_sma_signal"))
        
        if not volume_signals:
            return [pl.lit(0.5).alias(factor_name)]
        
        # Calcul du score de volume (moyenne pondérée)
        if len(volume_signals) == 1:
            volume_score = volume_signals[0].alias(factor_name)
        else:
            # OPTIMISATION CRITIQUE: Utiliser sum_horizontal au lieu d'accumulation
            # Les signaux sont ajoutés dans cet ordre: volume_ratio, obv, obv_ema, vwap, volume_sma
            signal_names = ["volume_ratio_signal", "obv_signal", "obv_ema_signal", "vwap_signal", "volume_sma_signal"]
            
            # Construire les expressions pondérées directement
            weighted_exprs = []
            for i, signal in enumerate(volume_signals):
                if i < len(signal_names):
                    name = signal_names[i]
                    weight = 0.3 if ('obv' in name or 'vwap' in name) else 0.2
                else:
                    weight = 0.2
                weighted_exprs.append(signal * weight)
            
            # Normaliser par la somme des poids
            total_weight = sum(0.3 if i < len(signal_names) and ('obv' in signal_names[i] or 'vwap' in signal_names[i]) else 0.2 
                             for i in range(len(volume_signals)))
            
            # Utiliser sum_horizontal pour éviter l'accumulation d'expressions
            volume_score = (pl.sum_horizontal(weighted_exprs) / total_weight).alias(factor_name)
        
        return volume_signals + [volume_score]
    
    def _calculate_volatility_factors(self, ldf: pl.LazyFrame, factor_name: str,
                                    factor_config: Dict[str, Any], available_columns: set) -> List[pl.Expr]:
        """Retourne les expressions pour les facteurs de volatilité."""
        self.logger.debug(f"Construction plan volatilité: {factor_name}")
        
        features = factor_config['features']
        available_features = [f for f in features if f in available_columns]
        
        if not available_features:
            self.logger.warning(f"Aucune feature disponible pour {factor_name}")
            return [pl.lit(0.5).alias(factor_name)]
        
        volatility_signals = []
        
        # 1. Signaux ATR (Average True Range)
        atr_signals = []
        for feature in available_features:
            if feature.startswith('atr_'):
                # ATR élevé = volatilité élevée, normaliser sur fenêtre glissante (OPTIMISÉ)
                atr_signal = fast_rolling_normalize(
                    pl.col(feature),
                    window=50,
                    fill_value=0.5,
                    min_periods=25
                ).alias(f"{feature}_signal")
                atr_signals.append(atr_signal)
        
        # 2. Signaux Bollinger Bands
        bb_signals = []
        if all(col in available_features for col in ['bb_upper', 'bb_lower', 'bb_width']):
            # Largeur des bandes = mesure de volatilité (OPTIMISÉ)
            bb_width_signal = fast_rolling_normalize(
                pl.col('bb_width'),
                window=50,
                fill_value=0.5,
                min_periods=25
            ).alias("bb_width_signal")
            bb_signals.append(bb_width_signal)
            
            # Position dans les bandes (bb_percent)
            if 'bb_percent' in available_features:
                # bb_percent proche des extrêmes = volatilité potentielle
                bb_extreme_signal = (
                    (pl.col('bb_percent') < 0.1).cast(pl.Float64) * 0.8 +
                    (pl.col('bb_percent') > 0.9).cast(pl.Float64) * 0.8 +
                    ((pl.col('bb_percent') >= 0.1) & (pl.col('bb_percent') <= 0.9)).cast(pl.Float64) * 0.2
                ).alias("bb_extreme_signal")
                bb_signals.append(bb_extreme_signal)
        
        # Agrégation des signaux
        all_signals = atr_signals + bb_signals
        
        if not all_signals:
            return [pl.lit(0.5).alias(factor_name)]
        
        # Calcul du score de volatilité
        if len(all_signals) == 1:
            volatility_score = all_signals[0].alias(factor_name)
        else:
            # Moyenne des signaux avec pondération égale
            volatility_score = pl.mean_horizontal(all_signals).alias(factor_name)
        
        return all_signals + [volatility_score]
    
    def _calculate_support_resistance_factors(self, ldf: pl.LazyFrame, factor_name: str,
                                            factor_config: Dict[str, Any], available_columns: set) -> List[pl.Expr]:
        """
        Calcule les facteurs de support/résistance (mode LAZY).
        
        FIX: Utilise un mapping explicite au lieu de meta.output_name() qui cause des crashes silencieux.
        """
        self.logger.debug(f"Construction plan S/R: {factor_name}")
        
        features = factor_config['features']
        available_features = [f for f in features if f in available_columns]
        
        self.logger.info(f"  → Features disponibles: {len(available_features)}/{len(features)}")
        
        if not available_features:
            self.logger.warning(f"Aucune feature disponible pour {factor_name}")
            return [pl.lit(0.5).alias(factor_name)]
        
        # Dictionnaire pour tracker les signaux et leurs noms
        signal_info = []  # Liste de tuples (expression, nom, type)
        
        # 1. Signaux de support/résistance directs
        self.logger.info(f"  → Étape 1: Signaux S/R directs")
        if 'pattern_support' in available_features:
            support_signal = pl.col('pattern_support').alias("support_signal")
            signal_info.append((support_signal, "support_signal", "sr"))
        
        if 'pattern_resistance' in available_features:
            # Résistance = signal négatif pour les achats
            resistance_signal = (1.0 - pl.col('pattern_resistance')).alias("resistance_signal")
            signal_info.append((resistance_signal, "resistance_signal", "sr"))
        
        self.logger.info(f"  → Signaux S/R: {len([s for s in signal_info if s[2] == 'sr'])}")
        
        # 2. Signaux de tendance
        self.logger.info(f"  → Étape 2: Signaux de tendance")
        for trend_feature in ['pattern_uptrend', 'pattern_downtrend', 'pattern_sideways_trend']:
            if trend_feature in available_features:
                if 'uptrend' in trend_feature:
                    signal = pl.col(trend_feature).alias(f"{trend_feature}_signal")
                    signal_info.append((signal, f"{trend_feature}_signal", "trend"))
                elif 'downtrend' in trend_feature:
                    signal = (1.0 - pl.col(trend_feature)).alias(f"{trend_feature}_signal")
                    signal_info.append((signal, f"{trend_feature}_signal", "trend"))
                else:  # sideways
                    signal = (0.5 * pl.col(trend_feature)).alias(f"{trend_feature}_signal")
                    signal_info.append((signal, f"{trend_feature}_signal", "trend"))
        
        self.logger.info(f"  → Signaux tendance: {len([s for s in signal_info if s[2] == 'trend'])}")
        
        # 3. Signaux de gaps
        self.logger.info(f"  → Étape 3: Signaux de gaps")
        gap_features = [f for f in available_features if 'gap' in f]
        self.logger.info(f"  → Gaps trouvés: {len(gap_features)}")
        
        for gap_feature in gap_features:
            if 'gap_up' in gap_feature:
                weight = 0.7  # Gap up = signal positif
            elif 'gap_down' in gap_feature:
                weight = 0.3  # Gap down = signal négatif
            else:
                weight = 0.5  # Autres gaps = neutre
            
            gap_signal = (pl.col(gap_feature) * weight).alias(f"{gap_feature}_signal")
            signal_info.append((gap_signal, f"{gap_feature}_signal", "gap"))
        
        self.logger.info(f"  → Total signaux: {len(signal_info)}")
        
        if not signal_info:
            return [pl.lit(0.5).alias(factor_name)]
        
        # Score support/résistance avec pondération
        self.logger.info(f"  → Étape 4: Calcul du score pondéré")
        
        if len(signal_info) == 1:
            sr_score = signal_info[0][0].alias(factor_name)
            self.logger.debug(f"  → Un seul signal, score direct")
            return [sr_score]
        else:
            # MODE LAZY: Construire directement les expressions pondérées
            # sans créer de colonnes temporaires
            
            self.logger.debug(f"  → Construction de {len(signal_info)} expressions pondérées")
            
            # Construire les expressions pondérées directement
            weighted_exprs = []
            for signal_expr, signal_name, signal_type in signal_info:
                if signal_type == "sr":
                    weight = 0.4
                elif signal_type == "trend":
                    weight = 0.3
                else:  # gap
                    weight = 0.2
                weighted_exprs.append(signal_expr * weight)
            
            # Normaliser par la somme des poids
            total_weight = sum(0.4 if st == "sr" else 0.3 if st == "trend" else 0.2 
                             for _, _, st in signal_info)
            
            # Somme des expressions pondérées (mode LAZY)
            sr_score = (pl.sum_horizontal(weighted_exprs) / total_weight).alias(factor_name)
            
            self.logger.debug(f"  → Score S/R construit (lazy)")
            
            # Retourner seulement le score final (pas les signaux intermédiaires)
            return [sr_score]
    
    def _validate_factors(self, df: pl.DataFrame) -> None:
        """Valide que les facteurs ont été correctement calculés."""
        self.logger.debug("Validation des facteurs calculés...")
        
        factor_columns = [col for col in df.columns if col.startswith('Factor_')]
        
        if not factor_columns:
            raise AlphaFactorError("Aucun facteur alpha n'a été calculé")
        
        for factor_col in factor_columns:
            # Vérifier que le facteur existe
            if factor_col not in df.columns:
                raise AlphaFactorError(f"Facteur manquant: {factor_col}")
            
            # Vérifier les valeurs nulles
            null_count = df.select(pl.col(factor_col).is_null().sum()).item()
            if null_count > 0:
                self.logger.warning(f"Facteur {factor_col} contient {null_count} valeurs nulles")
            
            # Vérifier la plage de valeurs (doit être entre 0 et 1 généralement)
            factor_stats = df.select([
                pl.col(factor_col).min().alias('min'),
                pl.col(factor_col).max().alias('max'),
                pl.col(factor_col).mean().alias('mean')
            ]).to_dicts()[0]
            
            if factor_stats['min'] < -0.1 or factor_stats['max'] > 1.1:
                self.logger.warning(
                    f"Facteur {factor_col} hors plage normale: "
                    f"min={factor_stats['min']:.3f}, max={factor_stats['max']:.3f}"
                )
            
            # Vérifier la variance (éviter les facteurs constants)
            variance = df.select(pl.col(factor_col).var()).item()
            if variance is not None and variance < 1e-6:
                self.logger.warning(f"Facteur {factor_col} quasi-constant (variance={variance:.2e})")
        
        self.logger.info(f"Validation terminée - {len(factor_columns)} facteurs validés")
    
    def get_factor_statistics(self, df: pl.DataFrame) -> Dict[str, Dict[str, float]]:
        """
        Retourne les statistiques descriptives des facteurs alpha.
        
        Args:
            df: DataFrame avec les facteurs calculés
            
        Returns:
            Dictionnaire avec les statistiques de chaque facteur
        """
        factor_columns = [col for col in df.columns if col.startswith('Factor_')]
        
        if not factor_columns:
            return {}
        
        stats = {}
        for factor_col in factor_columns:
            factor_stats = df.select([
                pl.col(factor_col).min().alias('min'),
                pl.col(factor_col).max().alias('max'),
                pl.col(factor_col).mean().alias('mean'),
                pl.col(factor_col).std().alias('std'),
                pl.col(factor_col).median().alias('median'),
                pl.col(factor_col).is_null().sum().alias('null_count')
            ]).to_dicts()[0]
            
            stats[factor_col] = factor_stats
        
        return stats    

    def _calculate_quantitative_factors(self, ldf: pl.LazyFrame, factor_name: str,
                                      factor_config: Dict[str, Any], available_columns: set) -> List[pl.Expr]:
        """Calcule les facteurs quantitatifs avancés."""
        self.logger.debug(f"Calcul des facteurs quantitatifs: {factor_name}")
        
        features = factor_config['features']
        available_features = [f for f in features if f in available_columns]
        
        if not available_features:
            self.logger.warning(f"Aucune feature disponible pour {factor_name}")
            return [pl.lit(0.5).alias(factor_name)]
        
        quant_signals = []
        
        # 1. Signaux de volatilité réalisée (OPTIMISÉ)
        vol_features = [f for f in available_features if 'realized_vol' in f]
        if vol_features:
            for feature in vol_features:
                vol_signal = fast_rolling_normalize(
                    pl.col(feature),
                    window=100,
                    fill_value=0.5,
                    min_periods=50
                ).alias(f"{feature}_signal")
                quant_signals.append(vol_signal)
        
        # 2. Signaux d'entropie (mesure de complexité) (OPTIMISÉ)
        entropy_features = [f for f in available_features if 'entropy' in f]
        if entropy_features:
            for feature in entropy_features:
                # Entropie élevée = plus de complexité/information
                entropy_signal = fast_rolling_normalize(
                    pl.col(feature),
                    window=50,
                    fill_value=0.5,
                    min_periods=25
                ).alias(f"{feature}_signal")
                quant_signals.append(entropy_signal)
        
        # 3. Signaux de régime de marché
        if 'quant_regime_detection' in available_features:
            regime_signal = pl.col('quant_regime_detection').alias("regime_signal")
            quant_signals.append(regime_signal)
        
        # 4. Signaux de persistance et clustering
        persistence_features = [f for f in available_features if any(x in f for x in ['persistence', 'clustering', 'hurst'])]
        if persistence_features:
            for feature in persistence_features:
                pers_signal = pl.col(feature).clip(0.0, 1.0).alias(f"{feature}_signal")
                quant_signals.append(pers_signal)
        
        if not quant_signals:
            return [pl.lit(0.5).alias(factor_name)]
        
        # Score quantitatif (moyenne pondérée)
        quantitative_score = pl.mean_horizontal(quant_signals).alias(factor_name)
        
        return quant_signals + [quantitative_score]
    
    def _calculate_candlestick_factors(self, ldf: pl.LazyFrame, factor_name: str,
                                     factor_config: Dict[str, Any], available_columns: set) -> List[pl.Expr]:
        """Calcule les facteurs de patterns de chandeliers."""
        self.logger.debug(f"Calcul des facteurs de chandeliers: {factor_name}")
        
        features = factor_config['features']
        available_features = [f for f in features if f in available_columns]
        
        if not available_features:
            self.logger.warning(f"Aucune feature disponible pour {factor_name}")
            return [pl.lit(0.0).alias(factor_name)]
        
        # Les patterns de chandeliers sont généralement binaires (0 ou 1)
        # On fait la somme pondérée des patterns actifs
        pattern_signals = []
        
        for feature in available_features:
            # Pondération selon l'importance du pattern
            if any(strong_pattern in feature for strong_pattern in ['engulfing', 'hammer', 'morning_star', 'evening_star']):
                weight = 0.3  # Patterns forts
            elif any(medium_pattern in feature for medium_pattern in ['harami', 'doji', 'pin_bar']):
                weight = 0.2  # Patterns moyens
            else:
                weight = 0.1  # Patterns faibles
            
            weighted_pattern = (pl.col(feature) * weight).alias(f"{feature}_weighted")
            pattern_signals.append(weighted_pattern)
        
        if not pattern_signals:
            return [pl.lit(0.0).alias(factor_name)]
        
        # Score des patterns (somme normalisée)
        # OPTIMISATION MÉMOIRE: Calculer directement sans créer de colonnes intermédiaires
        pattern_sum_expr = pl.lit(0.0)
        for feature in available_features:
            if any(strong_pattern in feature for strong_pattern in ['engulfing', 'hammer', 'morning_star', 'evening_star']):
                weight = 0.3
            elif any(medium_pattern in feature for medium_pattern in ['harami', 'doji', 'pin_bar']):
                weight = 0.2
            else:
                weight = 0.1
            pattern_sum_expr = pattern_sum_expr + (pl.col(feature) * weight)
        
        max_possible_score = len(available_features) * 0.3  # Score maximum possible
        candlestick_score = (pattern_sum_expr / max_possible_score).clip(0.0, 1.0).alias(factor_name)
        
        return [candlestick_score]
    
    def _calculate_chart_pattern_factors(self, ldf: pl.LazyFrame, factor_name: str,
                                       factor_config: Dict[str, Any], available_columns: set) -> List[pl.Expr]:
        """Calcule les facteurs de patterns de graphiques."""
        self.logger.debug(f"Calcul des facteurs de patterns de graphiques: {factor_name}")
        
        features = factor_config['features']
        available_features = [f for f in features if f in available_columns]
        
        if not available_features:
            self.logger.warning(f"Aucune feature disponible pour {factor_name}")
            return [pl.lit(0.0).alias(factor_name)]
        
        # OPTIMISATION MÉMOIRE: Calculer directement sans créer de colonnes intermédiaires
        pattern_sum_expr = pl.lit(0.0)
        
        for feature in available_features:
            # Pondération selon le type de pattern
            if any(continuation in feature for continuation in ['flag', 'pennant', 'triangle']):
                weight = 0.25  # Patterns de continuation
            elif any(reversal in feature for reversal in ['double_', 'triple_', 'head_shoulders']):
                weight = 0.35  # Patterns de retournement
            else:
                weight = 0.15  # Autres patterns
            
            pattern_sum_expr = pattern_sum_expr + (pl.col(feature) * weight)
        
        # Score des patterns de graphiques
        max_possible_score = len(available_features) * 0.35
        chart_pattern_score = (pattern_sum_expr / max_possible_score).clip(0.0, 1.0).alias(factor_name)
        
        return [chart_pattern_score]
    
    def _calculate_harmonic_factors(self, ldf: pl.LazyFrame, factor_name: str,
                                  factor_config: Dict[str, Any], available_columns: set) -> List[pl.Expr]:
        """Calcule les facteurs de patterns harmoniques."""
        self.logger.debug(f"Calcul des facteurs harmoniques: {factor_name}")
        
        features = factor_config['features']
        available_features = [f for f in features if f in available_columns]
        
        if not available_features:
            self.logger.warning(f"Aucune feature disponible pour {factor_name}")
            return [pl.lit(0.0).alias(factor_name)]
        
        # OPTIMISATION MÉMOIRE: Calculer directement sans créer de colonnes intermédiaires
        harmonic_sum_expr = pl.lit(0.0)
        
        for feature in available_features:
            # Pondération selon la complexité et fiabilité du pattern harmonique
            if any(advanced in feature for advanced in ['gartley', 'butterfly', 'bat']):
                weight = 0.4  # Patterns harmoniques classiques
            elif any(complex_pattern in feature for complex_pattern in ['crab', 'shark']):
                weight = 0.35  # Patterns plus complexes
            elif 'fibonacci' in feature:
                weight = 0.3  # Niveaux de Fibonacci
            else:
                weight = 0.2  # Autres patterns
            
            harmonic_sum_expr = harmonic_sum_expr + (pl.col(feature) * weight)
        
        # Score harmonique
        max_possible_score = len(available_features) * 0.4
        harmonic_score = (harmonic_sum_expr / max_possible_score).clip(0.0, 1.0).alias(factor_name)
        
        return [harmonic_score] 
   
    # ========== MÉTHODES POUR EXPRESSIONS DES FACTEURS SOPHISTIQUÉS (MODE LAZY) ==========
    
    def _get_regime_expressions(self, factor_name: str, factor_config: Dict[str, Any],
                               available_columns: set) -> List[pl.Expr]:
        """Retourne les expressions Polars pour les facteurs de régime (100% LAZY)."""
        if factor_name == 'Factor_Regime_Hurst_Score':
            if 'quant_hurst_exponent' in available_columns:
                hurst_score = (
                    (pl.col('quant_hurst_exponent') - 0.5) * 2 + 0.5
                ).clip(0.0, 1.0).alias(factor_name)
                return [hurst_score]
            else:
                return [pl.lit(0.5).alias(factor_name)]
        
        return [pl.lit(0.5).alias(factor_name)]
    
    def _get_risk_expressions(self, factor_name: str, factor_config: Dict[str, Any],
                             available_columns: set) -> List[pl.Expr]:
        """Retourne les expressions Polars pour les facteurs de risque (100% LAZY)."""
        if factor_name == 'Factor_Risk_TailEvent_Score':
            risk_signals = []
            
            # 1. Signal de skewness
            if 'quant_rolling_skewness' in available_columns:
                skew_risk = (
                    -pl.col('quant_rolling_skewness').clip(-3.0, 3.0) / 6.0 + 0.5
                ).clip(0.0, 1.0).alias("skewness_risk")
                risk_signals.append(skew_risk)
            
            # 2. Signal de kurtosis
            if 'quant_rolling_kurtosis' in available_columns:
                kurt_risk = (
                    (pl.col('quant_rolling_kurtosis') - 3.0).clip(0.0, 10.0) / 10.0
                ).clip(0.0, 1.0).alias("kurtosis_risk")
                risk_signals.append(kurt_risk)
            
            if risk_signals:
                if len(risk_signals) == 2:
                    tail_risk_score = (
                        risk_signals[0] * 0.3 + risk_signals[1] * 0.7
                    ).alias(factor_name)
                else:
                    tail_risk_score = risk_signals[0].alias(factor_name)
                
                return risk_signals + [tail_risk_score]
            else:
                return [pl.lit(0.5).alias(factor_name)]
        
        return [pl.lit(0.5).alias(factor_name)]
    
    def _get_persistence_expressions(self, factor_name: str, factor_config: Dict[str, Any],
                                    available_columns: set) -> List[pl.Expr]:
        """Retourne les expressions Polars pour les facteurs de persistance (100% LAZY)."""
        if factor_name == 'Factor_Persistence_Score':
            autocorr_signals = []
            autocorr_features = ['quant_autocorr_10', 'quant_autocorr_20', 'quant_autocorr_50']
            weights = [0.5, 0.3, 0.2]
            
            for feature, weight in zip(autocorr_features, weights):
                if feature in available_columns:
                    autocorr_norm = (
                        (pl.col(feature).clip(-1.0, 1.0) + 1.0) / 2.0 * weight
                    ).alias(f"{feature}_weighted")
                    autocorr_signals.append(autocorr_norm)
            
            if autocorr_signals:
                persistence_score = pl.sum_horizontal(autocorr_signals).alias(factor_name)
                return autocorr_signals + [persistence_score]
            else:
                return [pl.lit(0.5).alias(factor_name)]
        
        return [pl.lit(0.5).alias(factor_name)]
    
    def _get_cross_asset_relative_expressions(self, factor_name: str, factor_config: Dict[str, Any],
                                             available_columns: set) -> List[pl.Expr]:
        """Retourne les expressions Polars pour les facteurs de force relative (100% LAZY)."""
        # Pour l'instant, retourner une valeur neutre
        # TODO: Implémenter la logique de force relative cross-asset
        return [pl.lit(0.5).alias(factor_name)]
    
    def _get_cross_asset_correlation_expressions(self, factor_name: str, factor_config: Dict[str, Any],
                                                available_columns: set) -> List[pl.Expr]:
        """Retourne les expressions Polars pour les facteurs de corrélation (100% LAZY)."""
        # Pour l'instant, retourner une valeur neutre
        # TODO: Implémenter la logique de corrélation cross-asset
        return [pl.lit(0.5).alias(factor_name)]
    
    def _get_second_order_expressions(self, factor_name: str, factor_config: Dict[str, Any],
                                     available_columns: set) -> List[pl.Expr]:
        """Retourne les expressions Polars pour les facteurs de second ordre (100% LAZY)."""
        if factor_name == 'Factor_Momentum_Acceleration_Score':
            # Accélération du momentum = dérivée seconde
            if 'Factor_Momentum_Score' in available_columns:
                # Calculer tout en une seule expression pour éviter les dépendances
                momentum_change = pl.col('Factor_Momentum_Score') - pl.col('Factor_Momentum_Score').shift(1)
                momentum_accel_raw = momentum_change - momentum_change.shift(1)
                
                # Normalisation directe
                accel_score = fast_rolling_normalize(
                    momentum_accel_raw,
                    window=20,
                    fill_value=0.5,
                    min_periods=10
                ).alias(factor_name)
                
                return [accel_score]
            else:
                return [pl.lit(0.5).alias(factor_name)]
        
        elif factor_name == 'Factor_Volatility_Expansion_Rate':
            # Taux d'expansion de la volatilité
            if 'Factor_Volatility_Score' in available_columns:
                vol_change = pl.col('Factor_Volatility_Score') / pl.col('Factor_Volatility_Score').shift(1) - 1.0
                expansion_score = (vol_change.clip(-0.5, 0.5) + 0.5).alias(factor_name)
                
                return [expansion_score]
            else:
                return [pl.lit(0.5).alias(factor_name)]
        
        return [pl.lit(0.5).alias(factor_name)]
    
    def _get_divergence_expressions(self, factor_name: str, factor_config: Dict[str, Any],
                                   available_columns: set) -> List[pl.Expr]:
        """Retourne les expressions Polars pour les facteurs de divergence (100% LAZY)."""
        if factor_name == 'Factor_Divergence_Score_Bullish':
            # Divergence haussière: prix baisse mais momentum monte
            if 'close' in available_columns and 'Factor_Momentum_Score' in available_columns:
                price_change = pl.col('close') / pl.col('close').shift(5) - 1.0
                momentum_change = pl.col('Factor_Momentum_Score') - pl.col('Factor_Momentum_Score').shift(5)
                
                # Divergence = prix négatif ET momentum positif
                bullish_div = (
                    (price_change < 0) & (momentum_change > 0)
                ).cast(pl.Float64).alias(factor_name)
                
                return [bullish_div]
            else:
                return [pl.lit(0.0).alias(factor_name)]
        
        elif factor_name == 'Factor_Divergence_Score_Bearish':
            # Divergence baissière: prix monte mais momentum baisse
            if 'close' in available_columns and 'Factor_Momentum_Score' in available_columns:
                price_change = pl.col('close') / pl.col('close').shift(5) - 1.0
                momentum_change = pl.col('Factor_Momentum_Score') - pl.col('Factor_Momentum_Score').shift(5)
                
                # Divergence = prix positif ET momentum négatif
                bearish_div = (
                    (price_change > 0) & (momentum_change < 0)
                ).cast(pl.Float64).alias(factor_name)
                
                return [bearish_div]
            else:
                return [pl.lit(0.0).alias(factor_name)]
        
        return [pl.lit(0.0).alias(factor_name)]
    
    def _get_confluence_expressions(self, factor_name: str, factor_config: Dict[str, Any],
                                   available_columns: set) -> List[pl.Expr]:
        """Retourne les expressions Polars pour les facteurs de confluence (100% LAZY)."""
        if factor_name == 'Factor_Trend_Confirmation_Score':
            # Confluence de plusieurs signaux de tendance
            confirmation_signals = []
            
            if 'Factor_Trend_Score' in available_columns:
                trend_strong = (pl.col('Factor_Trend_Score') > 0.6).cast(pl.Float64)
                confirmation_signals.append(trend_strong)
            
            if 'Factor_Momentum_Score' in available_columns:
                momentum_strong = (pl.col('Factor_Momentum_Score') > 0.6).cast(pl.Float64)
                confirmation_signals.append(momentum_strong)
            
            if 'Factor_Volume_Score' in available_columns:
                volume_confirm = (pl.col('Factor_Volume_Score') > 0.5).cast(pl.Float64)
                confirmation_signals.append(volume_confirm)
            
            if confirmation_signals:
                # Score = nombre de confirmations / total possible
                confirmation_score = (
                    pl.sum_horizontal(confirmation_signals) / len(confirmation_signals)
                ).alias(factor_name)
                return [confirmation_score]
            else:
                return [pl.lit(0.5).alias(factor_name)]
        
        return [pl.lit(0.5).alias(factor_name)]
    
    def _get_climax_expressions(self, factor_name: str, factor_config: Dict[str, Any],
                               available_columns: set) -> List[pl.Expr]:
        """Retourne les expressions Polars pour les facteurs de climax (100% LAZY)."""
        if factor_name == 'Factor_Reversal_Climax_Score':
            # Climax de retournement: volume extrême + volatilité extrême
            climax_signals = []
            
            if 'Factor_Volume_Score' in available_columns:
                volume_climax = (pl.col('Factor_Volume_Score') > 0.8).cast(pl.Float64)
                climax_signals.append(volume_climax)
            
            if 'Factor_Volatility_Score' in available_columns:
                vol_climax = (pl.col('Factor_Volatility_Score') > 0.8).cast(pl.Float64)
                climax_signals.append(vol_climax)
            
            if climax_signals:
                # Climax = les deux conditions réunies
                climax_score = pl.min_horizontal(climax_signals).alias(factor_name)
                return [climax_score]
            else:
                return [pl.lit(0.0).alias(factor_name)]
        
        return [pl.lit(0.0).alias(factor_name)]
    
    # ========== ANCIENNES MÉTHODES (DEPRECATED - Gardées pour compatibilité) ==========
    
    def _calculate_regime_factors(self, df: pl.DataFrame, factor_name: str,
                                factor_config: Dict[str, Any]) -> pl.DataFrame:
        """
        Calcule les facteurs de régime de marché (Hurst, etc.).
        
        Factor_Regime_Hurst_Score: Mesure la "mémoire" du marché.
        """
        self.logger.debug(f"Calcul des facteurs de régime: {factor_name}")
        
        if factor_name == 'Factor_Regime_Hurst_Score':
            if 'quant_hurst_exponent' in df.columns:
                # Normalisation de l'exposant de Hurst
                # Hurst = 0.5 (aléatoire) -> Score = 0.5
                # Hurst > 0.5 (tendance) -> Score > 0.5  
                # Hurst < 0.5 (mean-reversion) -> Score < 0.5
                hurst_score = (
                    (pl.col('quant_hurst_exponent') - 0.5) * 2 + 0.5
                ).clip(0.0, 1.0).alias(factor_name)
                
                return df.with_columns([hurst_score])
            else:
                self.logger.warning("quant_hurst_exponent manquant pour Factor_Regime_Hurst_Score")
                return df.with_columns([pl.lit(0.5).alias(factor_name)])
        
        return df.with_columns([pl.lit(0.5).alias(factor_name)])
    
    def _calculate_risk_factors(self, df: pl.DataFrame, factor_name: str,
                              factor_config: Dict[str, Any]) -> pl.DataFrame:
        """
        Calcule les facteurs de risque d'événements extrêmes.
        
        Factor_Risk_TailEvent_Score: Risque de "cygne noir" basé sur skewness/kurtosis.
        """
        self.logger.debug(f"Calcul des facteurs de risque: {factor_name}")
        
        if factor_name == 'Factor_Risk_TailEvent_Score':
            risk_signals = []
            
            # 1. Signal de skewness (asymétrie)
            if 'quant_rolling_skewness' in df.columns:
                # Skewness négatif = risque de crash (score élevé)
                skew_risk = (
                    -pl.col('quant_rolling_skewness').clip(-3.0, 3.0) / 6.0 + 0.5
                ).clip(0.0, 1.0).alias("skewness_risk")
                risk_signals.append(skew_risk)
            
            # 2. Signal de kurtosis (queues épaisses)
            if 'quant_rolling_kurtosis' in df.columns:
                # Kurtosis élevé = risque d'événements extrêmes
                kurt_risk = (
                    (pl.col('quant_rolling_kurtosis') - 3.0).clip(0.0, 10.0) / 10.0
                ).clip(0.0, 1.0).alias("kurtosis_risk")
                risk_signals.append(kurt_risk)
            
            if risk_signals:
                # Pondération: kurtosis a plus de poids (70% vs 30%)
                if len(risk_signals) == 2:
                    tail_risk_score = (
                        risk_signals[0] * 0.3 + risk_signals[1] * 0.7
                    ).alias(factor_name)
                else:
                    tail_risk_score = risk_signals[0].alias(factor_name)
                
                return df.with_columns(risk_signals + [tail_risk_score])
            else:
                return df.with_columns([pl.lit(0.5).alias(factor_name)])
        
        return df.with_columns([pl.lit(0.5).alias(factor_name)])
    
    def _calculate_persistence_factors(self, df: pl.DataFrame, factor_name: str,
                                     factor_config: Dict[str, Any]) -> pl.DataFrame:
        """
        Calcule les facteurs de persistance/momentum quantitatif.
        
        Factor_Persistence_Score: Mesure la persistance via autocorrélations.
        """
        self.logger.debug(f"Calcul des facteurs de persistance: {factor_name}")
        
        if factor_name == 'Factor_Persistence_Score':
            autocorr_signals = []
            
            # Autocorrélations avec pondération décroissante
            autocorr_features = ['quant_autocorr_10', 'quant_autocorr_20', 'quant_autocorr_50']
            weights = [0.5, 0.3, 0.2]  # Plus de poids au court terme
            
            for feature, weight in zip(autocorr_features, weights):
                if feature in df.columns:
                    # Normalisation de l'autocorrélation (-1 à 1) vers (0 à 1)
                    autocorr_norm = (
                        (pl.col(feature).clip(-1.0, 1.0) + 1.0) / 2.0 * weight
                    ).alias(f"{feature}_weighted")
                    autocorr_signals.append(autocorr_norm)
            
            if autocorr_signals:
                persistence_score = pl.sum_horizontal(autocorr_signals).alias(factor_name)
                return df.with_columns(autocorr_signals + [persistence_score])
            else:
                return df.with_columns([pl.lit(0.5).alias(factor_name)])
        
        return df.with_columns([pl.lit(0.5).alias(factor_name)])
    
    def _calculate_cross_asset_relative_factors(self, df: pl.DataFrame, factor_name: str,
                                              factor_config: Dict[str, Any]) -> pl.DataFrame:
        """
        Calcule les facteurs de force relative entre actifs.
        
        Nécessite un DataFrame unifié avec colonnes préfixées.
        """
        self.logger.debug(f"Calcul des facteurs cross-asset relatifs: {factor_name}")
        
        if factor_name == 'Factor_RelativeStrength_Momentum':
            # Recherche des colonnes de momentum préfixées
            momentum_cols = [col for col in df.columns if col.endswith('_Factor_Momentum_Score')]
            
            if len(momentum_cols) >= 2:
                # Exemple: ETH vs BTC momentum
                btc_momentum = None
                eth_momentum = None
                
                for col in momentum_cols:
                    if 'BTC_' in col:
                        btc_momentum = col
                    elif 'ETH_' in col:
                        eth_momentum = col
                
                if btc_momentum and eth_momentum:
                    # Ratio ETH/BTC momentum
                    relative_strength = (
                        pl.col(eth_momentum) / (pl.col(btc_momentum) + 1e-6)
                    ).rolling_mean(10).alias(factor_name)
                    
                    return df.with_columns([relative_strength])
            
            self.logger.warning(f"Pas assez de colonnes momentum pour {factor_name}")
            return df.with_columns([pl.lit(0.5).alias(factor_name)])
        
        return df.with_columns([pl.lit(0.5).alias(factor_name)])
    
    def _calculate_cross_asset_correlation_factors(self, df: pl.DataFrame, factor_name: str,
                                                 factor_config: Dict[str, Any]) -> pl.DataFrame:
        """
        Calcule les facteurs de corrélation dynamique entre actifs.
        """
        self.logger.debug(f"Calcul des facteurs de corrélation cross-asset: {factor_name}")
        
        if factor_name == 'Factor_Correlation_Dynamic':
            # Recherche des colonnes de prix
            close_cols = [col for col in df.columns if col.endswith('_close')]
            
            if len(close_cols) >= 2:
                # Exemple: corrélation BTC vs SPX
                btc_close = None
                spx_close = None
                
                for col in close_cols:
                    if 'BTC_' in col:
                        btc_close = col
                    elif 'SPX_' in col or 'SP500_' in col:
                        spx_close = col
                
                if btc_close and spx_close:
                    # Calcul des rendements
                    btc_returns = pl.col(btc_close).pct_change()
                    spx_returns = pl.col(spx_close).pct_change()
                    
                    # Corrélation glissante sur 30 périodes
                    correlation = btc_returns.rolling_corr(spx_returns, window_size=30).alias(factor_name)
                    
                    return df.with_columns([correlation])
            
            self.logger.warning(f"Pas assez de colonnes de prix pour {factor_name}")
            return df.with_columns([pl.lit(0.0).alias(factor_name)])
        
        return df.with_columns([pl.lit(0.0).alias(factor_name)])
    
    def _calculate_second_order_factors(self, df: pl.DataFrame, factor_name: str,
                                      factor_config: Dict[str, Any]) -> pl.DataFrame:
        """
        Calcule les facteurs de second ordre (dérivées, accélération).
        """
        self.logger.debug(f"Calcul des facteurs de second ordre: {factor_name}")
        
        if factor_name == 'Factor_Momentum_Acceleration_Score':
            if 'Factor_Momentum_Score' in df.columns:
                # Accélération = différence du momentum sur 5 périodes
                momentum_accel = (
                    pl.col('Factor_Momentum_Score').diff(5).rolling_mean(3)
                ).alias(factor_name)
                
                return df.with_columns([momentum_accel])
        
        elif factor_name == 'Factor_Volatility_Expansion_Rate':
            if 'Factor_Volatility_Score' in df.columns:
                # Taux d'expansion = pourcentage de changement sur 3 périodes
                vol_expansion = (
                    pl.col('Factor_Volatility_Score').pct_change(3).rolling_mean(2)
                ).alias(factor_name)
                
                return df.with_columns([vol_expansion])
        
        return df.with_columns([pl.lit(0.0).alias(factor_name)])
    
    def _calculate_divergence_factors(self, df: pl.DataFrame, factor_name: str,
                                    factor_config: Dict[str, Any]) -> pl.DataFrame:
        """
        Calcule les facteurs de divergence multi-oscillateurs.
        """
        self.logger.debug(f"Calcul des facteurs de divergence: {factor_name}")
        
        if factor_name == 'Factor_Divergence_Score_Bearish':
            divergence_signals = []
            
            # Oscillateurs à tester pour divergence
            oscillators = ['rsi_14', 'macd_histogram', 'cci_20', 'money_flow_index']
            
            for osc in oscillators:
                if osc in df.columns and 'close' in df.columns:
                    # Divergence baissière: prix monte, oscillateur descend
                    price_up = pl.col('close') > pl.col('close').shift(10)
                    osc_down = pl.col(osc) < pl.col(osc).shift(10)
                    
                    divergence = (price_up & osc_down).cast(pl.Float32).alias(f"{osc}_divergence")
                    divergence_signals.append(divergence)
            
            if divergence_signals:
                # Score = moyenne des divergences actives
                divergence_score = pl.mean_horizontal(divergence_signals).alias(factor_name)
                return df.with_columns(divergence_signals + [divergence_score])
            else:
                return df.with_columns([pl.lit(0.0).alias(factor_name)])
        
        return df.with_columns([pl.lit(0.0).alias(factor_name)])
    
    def _calculate_confluence_factors(self, df: pl.DataFrame, factor_name: str,
                                    factor_config: Dict[str, Any]) -> pl.DataFrame:
        """
        Calcule les facteurs de confluence entre différents signaux.
        """
        self.logger.debug(f"Calcul des facteurs de confluence: {factor_name}")
        
        if factor_name == 'Factor_Trend_Confirmation_Score':
            required_factors = ['Factor_Trend_Score', 'Factor_Volume_Score', 'Factor_Persistence_Score']
            available_factors = [f for f in required_factors if f in df.columns]
            
            if len(available_factors) >= 2:
                # Confluence = moyenne des facteurs disponibles
                confluence_score = pl.mean_horizontal([pl.col(f) for f in available_factors]).alias(factor_name)
                return df.with_columns([confluence_score])
            else:
                return df.with_columns([pl.lit(0.5).alias(factor_name)])
        
        return df.with_columns([pl.lit(0.5).alias(factor_name)])
    
    def _calculate_climax_factors(self, df: pl.DataFrame, factor_name: str,
                                factor_config: Dict[str, Any]) -> pl.DataFrame:
        """
        Calcule les facteurs de climax de marché (épuisement de tendance).
        """
        self.logger.debug(f"Calcul des facteurs de climax: {factor_name}")
        
        if factor_name == 'Factor_Reversal_Climax_Score':
            climax_conditions = []
            
            # Condition 1: Volatilité extrême
            if 'Factor_Volatility_Score' in df.columns:
                high_vol = (pl.col('Factor_Volatility_Score') > 0.9).cast(pl.Float32)
                climax_conditions.append(high_vol)
            
            # Condition 2: Volume extrême
            if 'Factor_Volume_Score' in df.columns:
                high_vol_score = (pl.col('Factor_Volume_Score') > 0.9).cast(pl.Float32)
                climax_conditions.append(high_vol_score)
            
            # Condition 3: Prix aux extrêmes des Bollinger Bands
            if all(col in df.columns for col in ['close', 'bb_upper', 'bb_lower']):
                price_extreme = (
                    (pl.col('close') > pl.col('bb_upper')) | 
                    (pl.col('close') < pl.col('bb_lower'))
                ).cast(pl.Float32)
                climax_conditions.append(price_extreme)
            
            if climax_conditions:
                # Score de climax = moyenne des conditions
                climax_score = pl.mean_horizontal(climax_conditions).alias(factor_name)
                return df.with_columns([climax_score])
            else:
                return df.with_columns([pl.lit(0.0).alias(factor_name)])
        
        return df.with_columns([pl.lit(0.0).alias(factor_name)])
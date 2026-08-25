"""Configuration globale et constantes du systeme.

Chargee au demarrage depuis config/settings.json.
Expose ValidationConfig (seuils d'admission Einher) et RiskLimits (limites RM).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[3] / "config" / "settings.json"


@dataclass(frozen=True)
class RiskLimits:
    """Limites globales du Risk Manager (section 3.3 CDC).

    Attributs:
        exposure_total_pct: Exposition totale max (% du capital).
        exposure_asset_pct: Exposition max par actif.
        exposure_class_pct: Exposition max par classe d'actifs.
        max_positions: Nombre max de positions simultanees.
        max_correlated: Positions max sur actifs correles > 0.8.
        daily_loss_pct: Perte journaliere declenchant le circuit breaker.
        drawdown_soft_pct: Drawdown reduisant les tailles de moitie.
        drawdown_hard_pct: Drawdown declenchant pause complete.
        weekly_loss_pct: Perte hebdomadaire declenchant revue manuelle.
    """

    base_leverage: int = 10
    margin_buffer_pct: float = 0.10
    exposure_total_pct: float = 0.60
    exposure_asset_pct: float = 0.20
    exposure_class_pct: float = 0.35
    max_positions: int = 15
    max_correlated: int = 3
    daily_loss_pct: float = 0.05
    drawdown_soft_pct: float = 0.15
    drawdown_hard_pct: float = 0.25
    weekly_loss_pct: float = 0.10


@dataclass(frozen=True)
class ValidationConfig:
    """Seuils d'admission d'un Einher au corpus (section 5.3 CDC).

    Attributs:
        min_sharpe: Sharpe ratio minimum.
        min_win_rate: Win rate minimum (%).
        min_trades: Nombre minimum de trades.
        max_drawdown: Drawdown maximum (%).
        min_avg_profit: Profit moyen minimum (%).
        max_walkforward_gap: Ecart max calib/validation (%).
        min_assets_valid: Nombre minimum d'actifs valides.
    """

    min_sharpe: float = 1.0
    min_win_rate: float = 0.45
    min_trades: int = 30
    max_drawdown: float = 0.25
    min_avg_profit: float = 0.0005
    max_walkforward_gap: float = 0.50
    min_assets_valid: int = 3


@dataclass(frozen=True)
class SystemConfig:
    """Configuration complete du systeme.

    Attributs:
        risk_limits: Limites du Risk Manager.
        validation_config: Seuils d'admission Einher.
        risk_per_trade: Risque par trade (% capital, defaut 1%).
        confidence_thresholds: Seuils de confiance pour le sizing.
        timeframes: Liste des timeframes actifs.
        default_cooldown: Delai de cooldown par defaut.
        default_sl_atr_mult: Multiplicateur ATR fallback SL.
        default_tp_atr_mult: Multiplicateur ATR fallback TP.
    """

    risk_limits: RiskLimits
    validation_config: ValidationConfig
    risk_per_trade: float = 0.01
    confidence_thresholds: tuple[float, ...] = (0.5, 0.75, 1.0)
    timeframes: tuple[str, ...] = ("5m", "15m", "1h", "4h", "1d")
    default_cooldown: str = "4h"
    default_sl_atr_mult: float = 1.5
    default_tp_atr_mult: float = 2.5


def load_settings(path: str | Path | None = None) -> SystemConfig:
    """Charge la configuration depuis un fichier JSON.

    Args:
        path: Chemin vers settings.json. Par defaut config/settings.json a la racine.

    Returns:
        SystemConfig initialisee.
    """
    config_path = Path(path) if path else DEFAULT_CONFIG_PATH

    if not config_path.exists():
        return SystemConfig(
            risk_limits=RiskLimits(),
            validation_config=ValidationConfig(),
        )

    with open(config_path, encoding="utf-8") as f:
        raw: dict[str, Any] = json.load(f)

    risk_limits = RiskLimits(**raw.get("risk_limits", {}))
    validation_config = ValidationConfig(**raw.get("validation_config", {}))

    return SystemConfig(
        risk_limits=risk_limits,
        validation_config=validation_config,
        **{
        k: v
        for k, v in raw.items()
        if k not in ("risk_limits", "validation_config", "_comment") and k in SystemConfig.__dataclass_fields__
    },
    )

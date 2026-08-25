"""RiskManager — Dimensionnement et controle des limites globales.

Transforme les intentions de trade (signal + TP/SL + confiance) en ordres
dimensionnes, ou les rejette avec raison. Version allegée de l'Agent Risque
MIDAS : regles fixes, pas de modele appris.

Responsabilites (Section 3 CDC):
- Sizing par risque fixe et plafond confiance.
- Limites globales : exposition, positions, correlation, circuit breakers.
- Verification marge disponible (levier de compte fixe cTrader).
- Journalisation des rejets.

Reference : Section 3 du CDC EINHERJAR.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any

from einherjar.brokers.broker_utils import ASSET_CLASS_MAP
from einherjar.core.config import SystemConfig
from einherjar.core.enums import AssetClass, OrderType, RejectionReason
from einherjar.core.models import AccountState, Order, Position, Rejection, Signal

logger = logging.getLogger(__name__)

# Correlation simplifiee : groupes d'actifs fortement correles (>0.8)
CORRELATION_GROUPS: list[set[str]] = [
    {"BTCUSD", "ETHUSD", "BCHUSD", "LTCUSD"},
    {"EURUSD", "GBPUSD", "EURGBP", "AUDUSD"},
    {"USDCAD", "USDCHF", "USDJPY"},
    {"XAUUSD", "XAGUSD"},
    {"AAPL", "MSFT", "AMZN", "GOOGL", "NVDA", "TSLA"},
    {"SP500", "NASDAQ100", "DOWJONES"},
    {"WTIUSD", "BRENT"},
]

# Minimum size par asset (unites de base) — utilise pour arrondir
# Avec cTrader CFD le min size est generalement 0.01 pour actions/forex,
# et 0.001 ou 1.0 pour crypto selon le symbole.
MIN_SIZE_MAP: dict[str, float] = {
    "BTCUSD": 0.001,
    "ETHUSD": 0.01,
    "ADAUSD": 1.0,
    "BCHUSD": 0.01,
    "LTCUSD": 0.1,
    "EURUSD": 0.01,
    "GBPUSD": 0.01,
    "USDJPY": 0.01,
    "AUDUSD": 0.01,
    "USDCAD": 0.01,
    "USDCHF": 0.01,
    "EURGBP": 0.01,
    "NZDUSD": 0.01,
    "XAUUSD": 0.01,
    "AAPL": 0.01,
    "MSFT": 0.01,
    "NVDA": 0.01,
    "AMZN": 0.01,
    "GOOGL": 0.01,
    "TSLA": 0.01,
    "JPM": 0.01,
    "XOM": 0.01,
    "SP500": 0.01,
    "NASDAQ100": 0.01,
    "DOWJONES": 0.01,
    "DAX40": 0.01,
    "WTIUSD": 0.01,
    "BRENT": 0.01,
    "COPPER": 0.01,
}


class RiskManager:
    """Gestionnaire de risque et de dimensionnement.

    Attributs:
        config: Configuration systeme (RiskLimits, etc.).
        limits: Limites globales.
        risk_per_trade: Risque par trade (% capital).
        confidence_thresholds: Seuils de confiance pour le sizing.
        daily_pnl: P&L journalier suivi.
        weekly_pnl: P&L hebdomadaire suivi.
        peak_equity: Plus haut equity observe.
        _rejection_history: Historique des rejets pour tracking.
    """

    def __init__(self, config: SystemConfig | None = None) -> None:
        """Initialise le Risk Manager.

        Args:
            config: Configuration systeme. Defaut si None.
        """
        self.config = config or SystemConfig(
            risk_limits=__import__("einherjar.core.config", fromlist=["RiskLimits"]).RiskLimits(),
            validation_config=__import__("einherjar.core.config", fromlist=["ValidationConfig"]).ValidationConfig(),
        )
        self.limits = self.config.risk_limits
        self.risk_per_trade = self.config.risk_per_trade
        self.confidence_thresholds = self.config.confidence_thresholds
        self.daily_pnl: float = 0.0
        self.weekly_pnl: float = 0.0
        self.peak_equity: float = 0.0
        self._last_daily_reset: datetime = datetime.now(UTC)
        self._last_weekly_reset: datetime = datetime.now(UTC)
        self._rejection_history: list[Rejection] = []

    def _reset_periodic_tracking(self, now: datetime) -> None:
        """Reinitialise le tracking journalier/hebdomadaire si necessaire."""
        if now.date() != self._last_daily_reset.date():
            self.daily_pnl = 0.0
            self._last_daily_reset = now
        if now.weekday() == 0 and self._last_weekly_reset.date() != now.date():
            self.weekly_pnl = 0.0
            self._last_weekly_reset = now

    def _asset_class(self, asset: str) -> AssetClass:
        """Retourne la classe d'un actif."""
        return ASSET_CLASS_MAP.get(asset, AssetClass.CRYPTO)

    def _exposure_pct(self, asset: str, positions: list[Position], equity: float) -> float:
        """Calcule l'exposition courante d'un actif (% equity)."""
        if equity <= 0:
            return 0.0
        total = sum(
            pos.quantity * pos.avg_entry_price for pos in positions if pos.asset == asset
        )
        return total / equity

    def _exposure_class_pct(self, asset_class: AssetClass, positions: list[Position], equity: float) -> float:
        """Calcule l'exposition d'une classe d'actifs (% equity)."""
        if equity <= 0:
            return 0.0
        total = sum(
            pos.quantity * pos.avg_entry_price
            for pos in positions
            if self._asset_class(pos.asset) == asset_class
        )
        return total / equity

    def _total_exposure_pct(self, positions: list[Position], equity: float) -> float:
        """Calcule l'exposition totale (% equity)."""
        if equity <= 0:
            return 0.0
        total = sum(pos.quantity * pos.avg_entry_price for pos in positions)
        return total / equity

    def _count_correlated_positions(self, asset: str, positions: list[Position]) -> int:
        """Compte les positions sur des actifs fortement correles."""
        group: set[str] | None = None
        for g in CORRELATION_GROUPS:
            if asset in g:
                group = g
                break
        if group is None:
            return 0
        return sum(1 for pos in positions if pos.asset in group)

    def _calculate_volume(self, signal: Signal, account: AccountState) -> float:
        """Calcule le volume de position selon le risque fixe.

        Formule : volume = (capital * risk_per_trade) / distance_SL
        Puis plafonne par le score de confiance.

        Args:
            signal: Signal avec entree, SL, confiance.
            account: Etat du compte.

        Returns:
            Volume calcule en unites de base.
        """
        capital = account.equity
        if capital <= 0:
            return 0.0

        distance_sl = abs(signal.entry_price - signal.sl_price)
        if distance_sl <= 0:
            distance_sl = signal.entry_price * 0.01

        risk_amount = capital * self.risk_per_trade
        base_volume = risk_amount / distance_sl

        confidence = max(0.0, min(1.0, signal.confidence))
        volume = base_volume * confidence

        return volume

    def _round_to_lot(self, asset: str, volume: float) -> float:
        """Arrondit le volume au lot minimum de l'actif.

        Args:
            asset: Symbole.
            volume: Volume brut.

        Returns:
            Volume arrondi.
        """
        min_size = MIN_SIZE_MAP.get(asset, 0.01)
        if volume < min_size:
            return 0.0
        decimals = 4 if self._asset_class(asset) == AssetClass.CRYPTO else 2
        return round(volume, decimals)

    def _check_margin(self, volume: float, entry_price: float, account: AccountState) -> bool:
        """Verifie que la marge necessaire tient dans la marge disponible.

        Args:
            volume: Volume en unites de base.
            entry_price: Prix d'entree.
            account: Etat du compte (leverage lu depuis cTrader).

        Returns:
            True si la marge est suffisante.
        """
        if account.leverage <= 0 or entry_price <= 0:
            return False
        margin_needed = (volume * entry_price) / account.leverage
        buffer = 1.0 + self.limits.margin_buffer_pct
        margin_available = account.margin_available / buffer if account.margin_available > 0 else 0.0
        return margin_needed <= margin_available

    def evaluate(
        self,
        signal: Signal,
        account: AccountState,
        positions: list[Position],
    ) -> Order | Rejection:
        """Evalue une intention de trade contre les limites globales.

        Args:
            signal: Signal emis par un Einher.
            account: Etat du compte.
            positions: Positions ouvertes.

        Returns:
            Order si accepte, Rejection sinon.
        """
        now = datetime.now(UTC)
        self._reset_periodic_tracking(now)

        equity = account.equity
        if equity <= 0:
            return Rejection(signal=signal, reason=RejectionReason.EXPOSURE_TOTAL.value)

        if equity > self.peak_equity:
            self.peak_equity = equity

        drawdown = 0.0
        if self.peak_equity > 0:
            drawdown = (self.peak_equity - equity) / self.peak_equity

        # --- Circuit breakers ---
        if self.daily_pnl < -self.limits.daily_loss_pct * equity:
            return Rejection(signal=signal, reason=RejectionReason.DAILY_LOSS.value)
        if drawdown >= self.limits.drawdown_hard_pct:
            return Rejection(signal=signal, reason=RejectionReason.DRAWDOWN.value)
        if self.weekly_pnl < -self.limits.weekly_loss_pct * equity:
            return Rejection(signal=signal, reason=RejectionReason.WEEKLY_LOSS.value)

        # --- Limites d'exposition ---
        total_exp = self._total_exposure_pct(positions, equity)
        if total_exp >= self.limits.exposure_total_pct:
            return Rejection(signal=signal, reason=RejectionReason.EXPOSURE_TOTAL.value)

        asset_exp = self._exposure_pct(signal.asset, positions, equity)
        if asset_exp >= self.limits.exposure_asset_pct:
            return Rejection(signal=signal, reason=RejectionReason.EXPOSURE_ASSET.value)

        asset_class = self._asset_class(signal.asset)
        class_exp = self._exposure_class_pct(asset_class, positions, equity)
        if class_exp >= self.limits.exposure_class_pct:
            return Rejection(signal=signal, reason=RejectionReason.EXPOSURE_CLASS.value)

        if len(positions) >= self.limits.max_positions:
            return Rejection(signal=signal, reason=RejectionReason.MAX_POSITIONS.value)

        corr_count = self._count_correlated_positions(signal.asset, positions)
        if corr_count >= self.limits.max_correlated:
            return Rejection(signal=signal, reason=RejectionReason.CORRELATION.value)

        # --- Sizing ---
        volume = self._calculate_volume(signal, account)
        volume = self._round_to_lot(signal.asset, volume)

        if volume <= 0:
            return Rejection(signal=signal, reason=RejectionReason.MIN_SIZE.value)

        # Drawdown soft : reduire les tailles de moitie
        if drawdown >= self.limits.drawdown_soft_pct:
            volume = volume / 2.0
            volume = self._round_to_lot(signal.asset, volume)
            if volume <= 0:
                return Rejection(signal=signal, reason=RejectionReason.MIN_SIZE.value)

        # --- Verification marge ---
        entry_price = signal.entry_price if signal.entry_price and signal.entry_price > 0 else 1.0
        if not self._check_margin(volume, entry_price, account):
            return Rejection(signal=signal, reason=RejectionReason.MARGIN.value)

        # --- Construire l'ordre ---
        order = Order(
            order_id=f"ORD_{signal.asset.replace('/', '_')}_{now.strftime('%Y%m%d%H%M%S')}_{signal.einher_name}",
            asset=signal.asset,
            order_type=OrderType.MARKET,
            direction=signal.direction,
            quantity=volume,
            entry_price=None,
            tp_price=signal.tp_price,
            sl_price=signal.sl_price,
            einher_name=signal.einher_name,
            timestamp=now,
            status="pending",
        )

        logger.info(
            "RISK_OK %s %s qty=%.6f risk=%.4f%% margin_ok",
            signal.asset,
            signal.einher_name,
            volume,
            self.risk_per_trade * 100,
        )
        return order

    def update_pnl(self, realized_pnl: float) -> None:
        """Met a jour le P&L realise pour le tracking journalier/hebdomadaire."""
        self.daily_pnl += realized_pnl
        self.weekly_pnl += realized_pnl

    def get_status(self) -> dict[str, Any]:
        """Retourne l'etat courant du Risk Manager."""
        now = datetime.now(UTC)
        self._reset_periodic_tracking(now)
        return {
            "daily_pnl": self.daily_pnl,
            "weekly_pnl": self.weekly_pnl,
            "peak_equity": self.peak_equity,
            "risk_per_trade_pct": self.risk_per_trade,
            "limits": {
                "exposure_total_pct": self.limits.exposure_total_pct,
                "exposure_asset_pct": self.limits.exposure_asset_pct,
                "exposure_class_pct": self.limits.exposure_class_pct,
                "max_positions": self.limits.max_positions,
                "max_correlated": self.limits.max_correlated,
                "daily_loss_pct": self.limits.daily_loss_pct,
                "drawdown_soft_pct": self.limits.drawdown_soft_pct,
                "drawdown_hard_pct": self.limits.drawdown_hard_pct,
                "weekly_loss_pct": self.limits.weekly_loss_pct,
                "margin_buffer_pct": self.limits.margin_buffer_pct,
                "base_leverage": self.limits.base_leverage,
            },
        }

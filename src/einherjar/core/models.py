"""Modeles de donnees centralises du systeme.

Dataclasses serialisables representant les entites principales :
Einher, Signal, Order, Position, Fill, AccountState.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from einherjar.core.enums import AssetClass, Direction, EinherState, OrderType, TimeFrame


@dataclass
class Einher:
    """Representation d'une strategie detectable (section 2.3 CDC).

    Attributs:
        name: Nom unique, ex "E_DoubleTop_ADXfilter_4h".
        domain: Domaine strategique (pattern, quant, etc.).
        direction: long | short | both.
        timeframes: TF d'evaluation.
        trigger: Condition polars du declencheur.
        filters: Conditions complementaires (AND).
        assets: Actifs couverts.
        tp_rule: Regle de take-profit (natif ou calibre).
        sl_rule: Regle de stop-loss.
        max_holding: Duree max en position.
        cooldown: Delai avant re-emission.
        sharpe: Sharpe observe (rempli par calibration).
        win_rate: Win rate observe.
        avg_tp_pct: TP moyen.
        avg_sl_pct: SL moyen.
        trade_count: Nombre de trades.
        profit_horizon: Horizon empirique dominant.
        calibrated_on: Periode de calibration.
        state: Etat temps reel (idle, forming, etc.).
    """

    name: str
    domain: str
    direction: str
    timeframes: list[str]
    trigger: str
    filters: list[str] = field(default_factory=list)
    assets: str = "all"
    tp_rule: dict[str, Any] = field(default_factory=dict)
    sl_rule: dict[str, Any] = field(default_factory=dict)
    max_holding: str | None = None
    cooldown: str = "4h"
    sharpe: float = 0.0
    win_rate: float = 0.0
    avg_tp_pct: float = 0.0
    avg_sl_pct: float = 0.0
    trade_count: int = 0
    profit_horizon: str = ""
    calibrated_on: str = ""
    state: EinherState = EinherState.IDLE

    def to_dict(self) -> dict[str, Any]:
        """Serialise l'Einher en dict JSON-compatible."""
        data = self.__dict__.copy()
        if isinstance(data.get("state"), EinherState):
            data["state"] = data["state"].value
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Einher:
        """Deserialise depuis un dict."""
        if "state" in data and isinstance(data["state"], str):
            data["state"] = EinherState(data["state"])
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class Signal:
    """Signal emis par un Einher apres evaluation.

    Attributs:
        asset: Symbole (ex "BTC/USDT").
        direction: LONG, SHORT ou HOLD.
        timeframe: Timeframe du signal.
        einher_name: Emetteur.
        entry_price: Prix d'entree suggere.
        tp_price: Take-profit calcule.
        sl_price: Stop-loss calcule.
        confidence: Score 0.0 a 1.0.
        timestamp: Horodatage UTC.
        context: Dictionnaire contextuel (niveaux, ATR, etc.).
    """

    asset: str
    direction: Direction
    timeframe: TimeFrame
    einher_name: str
    entry_price: float
    tp_price: float
    sl_price: float
    confidence: float = 0.0
    timestamp: datetime = field(default_factory=datetime.utcnow)
    context: dict[str, Any] = field(default_factory=dict)


@dataclass
class ConfluenceCluster:
    """Intention agregee issue de plusieurs signaux coherents.

    Le moteur de risque traite uniquement ces intentions. Les signaux bruts
    restent journalises afin de conserver l'explicabilite de la decision.
    """

    asset: str
    direction: Direction
    timeframe: TimeFrame
    entry_price: float
    tp_price: float
    sl_price: float
    confidence: float
    contributing_einhers: list[str]
    score: float
    timestamp: datetime = field(default_factory=datetime.utcnow)
    context: dict[str, Any] = field(default_factory=dict)

    @property
    def einher_name(self) -> str:
        """Nom stable et lisible pour les journaux et ordres."""
        return "CONFLUENCE:" + ",".join(self.contributing_einhers)

    def to_signal(self) -> Signal:
        """Adapte le cluster au contrat existant du RiskManager."""
        return Signal(
            asset=self.asset,
            direction=self.direction,
            timeframe=self.timeframe,
            einher_name=self.einher_name,
            entry_price=self.entry_price,
            tp_price=self.tp_price,
            sl_price=self.sl_price,
            confidence=self.confidence,
            timestamp=self.timestamp,
            context={**self.context, "contributing_einhers": self.contributing_einhers},
        )


@dataclass
class Order:
    """Intention d'ordre apres dimensionnement par le Risk Manager.

    Attributs:
        order_id: Identifiant unique.
        asset: Symbole.
        order_type: MARKET, LIMIT, STOP_MARKET, BRACKET_OCO.
        direction: LONG ou SHORT.
        quantity: Taille dimensionnee.
        entry_price: Prix d'entree (None pour market).
        tp_price: Take-profit.
        sl_price: Stop-loss.
        einher_name: Emetteur du signal.
        timestamp: Horodatage.
        status: Etat de l'ordre (pending, open, filled, cancelled, rejected).
    """

    order_id: str
    asset: str
    order_type: OrderType
    direction: Direction
    quantity: float
    entry_price: float | None = None
    tp_price: float | None = None
    sl_price: float | None = None
    einher_name: str = ""
    timestamp: datetime = field(default_factory=datetime.utcnow)
    status: str = "pending"


@dataclass
class Fill:
    """Execution partielle ou complete d'un ordre.

    Attributs:
        fill_id: Identifiant unique.
        order_id: Ordre parent.
        asset: Symbole.
        filled_qty: Quantite executee.
        filled_price: Prix moyen d'execution.
        fee: Frais appliques.
        timestamp: Horodatage.
    """

    fill_id: str
    order_id: str
    asset: str
    filled_qty: float
    filled_price: float
    fee: float = 0.0
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class Position:
    """Position ouverte dans le portefeuille.

    Attributs:
        position_id: Identifiant unique.
        asset: Symbole.
        direction: LONG ou SHORT.
        quantity: Taille totale.
        avg_entry_price: Prix moyen d'entree.
        tp_price: Take-profit.
        sl_price: Stop-loss.
        unrealized_pnl: P&L latente.
        einher_name: Emetteur.
        opened_at: Date d'ouverture.
        asset_class: Classe d'actifs.
    """

    position_id: str
    asset: str
    direction: Direction
    quantity: float
    avg_entry_price: float
    tp_price: float | None = None
    sl_price: float | None = None
    unrealized_pnl: float = 0.0
    einher_name: str = ""
    opened_at: datetime = field(default_factory=datetime.utcnow)
    asset_class: AssetClass = AssetClass.CRYPTO


@dataclass
class AccountState:
    """Etat du compte broker ou paper.

    Attributs:
        cash: Cash disponible.
        equity: Valeur totale (cash + positions).
        margin_used: Marge utilisee.
        margin_available: Marge disponible.
        leverage: Levier du compte.
        timestamp: Horodatage.
    """

    cash: float = 0.0
    equity: float = 0.0
    margin_used: float = 0.0
    margin_available: float = 0.0
    leverage: int = 1
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class Rejection:
    """Enregistrement d'une intention rejetee par le Risk Manager.

    Attributs:
        signal: Signal rejete.
        reason: Raison du rejet.
        timestamp: Horodatage.
    """

    signal: Signal
    reason: str
    timestamp: datetime = field(default_factory=datetime.utcnow)

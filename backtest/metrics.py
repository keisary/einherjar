"""
Metriques de performance pour le backtest EINHERJAR.

=== CHANGELOG (correction du bug Sharpe/Sortino) ===

BUG CORRIGE : l'ancienne version annualisait le Sharpe/Sortino en
multipliant le ratio brut (calcule sur les retours trade-par-trade) par
sqrt(nombre_de_trades_par_an). Cette methode suppose que chaque trade
est une observation i.i.d., comme un jour de bourse — ce qui est faux :
les trades d'une meme strategie sont correles entre eux (meme logique
d'entree, memes conditions de marche). Plus une strategie trade souvent,
plus ce facteur d'annualisation explose et gonfle artificiellement le
Sharpe, meme pour un edge minuscule (ex: 50 trades/mois -> sqrt(600) =
24.5, donc un Sharpe brut de 0.3 devient "7.35" annualise).

CORRECTION : le Sharpe/Sortino sont maintenant calcules sur une
equity curve RE-ECHANTILLONNEE A FREQUENCE QUOTIDIENNE FIXE (calendaire),
comme c'est l'usage standard en finance quantitative. On construit cette
courbe a partir des timestamps de CLOTURE de trade (exit_timestamps_ms),
pas des timestamps d'entree.

L'ancien calcul (trade-par-trade, non-annualise) est conserve sous le nom
`sharpe_ratio_per_trade` — utile comme diagnostic interne pour comparer
des Einhers entre eux sur un meme actif/tf, mais NE DOIT JAMAIS etre
presente comme un "Sharpe annualise" a l'utilisateur final.

Autres regles conservees :
- Total return = produit compose (1+r) - 1, pas somme
- Downside deviation utilise le MAR = 0 (convention trading)
- Capping des metriques pour eviter les valeurs explosives residuelles
"""

import math
import numpy as np
from typing import Optional, Sequence

TRADING_DAYS_PER_YEAR = 252
MS_PER_DAY = 1000 * 60 * 60 * 24

# Sanity caps : au-dela de ces valeurs, il y a tres probablement un bug
# de simulation en amont (donnees, look-ahead, position sizing) plutot
# qu'une vraie edge. Ces caps signalent qu'il faut investiguer, ce ne
# sont pas des objectifs a atteindre.
SHARPE_SANITY_CAP = 8.0
SORTINO_SANITY_CAP = 12.0


def _returns_array(returns: Sequence[float]) -> np.ndarray:
    arr = np.array(returns, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return arr


def win_rate(returns: Sequence[float]) -> float:
    """Proportion de trades gagnants (return > 0)."""
    arr = _returns_array(returns)
    if len(arr) == 0:
        return 0.0
    return float(np.sum(arr > 0) / len(arr))


def profit_factor(returns: Sequence[float]) -> float:
    """Ratio gains bruts / pertes brutes absolues. Cappe a 50."""
    arr = _returns_array(returns)
    gains = np.sum(arr[arr > 0])
    losses = abs(np.sum(arr[arr < 0]))
    if losses == 0:
        return 50.0 if gains > 0 else 0.0
    return min(float(gains / losses), 50.0)


def avg_trade(returns: Sequence[float]) -> float:
    """Return moyen par trade."""
    arr = _returns_array(returns)
    if len(arr) == 0:
        return 0.0
    return float(np.mean(arr))


def expectancy(returns: Sequence[float]) -> float:
    """Expectance : (WR * avg_win) - ((1 - WR) * avg_loss)."""
    arr = _returns_array(returns)
    if len(arr) == 0:
        return 0.0
    wr = win_rate(arr)
    avg_win = np.mean(arr[arr > 0]) if np.any(arr > 0) else 0.0
    avg_loss = abs(np.mean(arr[arr < 0])) if np.any(arr < 0) else 0.0
    return float((wr * avg_win) - ((1.0 - wr) * avg_loss))


def total_return_compound(returns: Sequence[float]) -> float:
    """Return total compose : produit de (1 + r_i) - 1."""
    arr = _returns_array(returns)
    if len(arr) == 0:
        return 0.0
    return float(np.prod(1.0 + arr) - 1.0)


def max_drawdown(equity: Sequence[float]) -> float:
    """Drawdown maximum en pourcentage (negatif)."""
    arr = np.array(equity, dtype=np.float64)
    if len(arr) < 2:
        return 0.0
    peak = np.maximum.accumulate(arr)
    drawdown = (arr - peak) / peak
    return float(np.min(drawdown))


def trades_per_month(timestamps_ms: Sequence[int]) -> float:
    """Nombre moyen de trades par mois, a partir d'une liste de timestamps
    (utiliser de preference les timestamps de CLOTURE, pas d'entree)."""
    if len(timestamps_ms) < 2:
        return 0.0
    first = timestamps_ms[0]
    last = timestamps_ms[-1]
    months = (last - first) / (1000.0 * 60 * 60 * 24 * 30.44)
    if months <= 0:
        return 0.0
    return len(timestamps_ms) / months


def build_calendar_equity(
    equity: Sequence[float],
    exit_timestamps_ms: Sequence[int],
) -> Optional[np.ndarray]:
    """
    Reconstruit une equity curve a frequence QUOTIDIENNE FIXE, a partir
    des timestamps de CLOTURE de trade (jamais d'entree — c'est la valeur
    d'equity APRES cloture qui doit etre datee a la cloture).

    equity[0] = valeur de depart (avant tout trade, generalement 1.0).
    equity[i+1] = valeur apres la cloture du trade i, qui se termine a
    exit_timestamps_ms[i].

    Si plusieurs trades cloturent le meme jour calendaire, on garde la
    derniere valeur d'equity de ce jour. Les jours sans cloture sont
    forward-fill (equity inchangee) pour obtenir une grille reguliere.

    Retourne None si les tailles ne correspondent pas (securite : on
    refuse de construire une courbe potentiellement decalee/fausse) ou
    s'il n'y a pas assez de donnees.
    """
    if not exit_timestamps_ms or len(exit_timestamps_ms) == 0:
        return None
    if len(exit_timestamps_ms) != len(equity) - 1:
        return None

    days = [int(ts // MS_PER_DAY) for ts in exit_timestamps_ms]
    day_to_equity = {}
    for d, eq in zip(days, equity[1:]):
        day_to_equity[d] = eq  # le dernier trade cloture du jour l'emporte

    first_day, last_day = min(days), max(days)
    n_days = last_day - first_day + 1
    daily = np.empty(n_days, dtype=np.float64)
    last_val = float(equity[0])
    for i, d in enumerate(range(first_day, last_day + 1)):
        if d in day_to_equity:
            last_val = day_to_equity[d]
        daily[i] = last_val
    return daily


def sharpe_ratio_calendar(daily_equity: Optional[np.ndarray], risk_free_daily: float = 0.0) -> float:
    """Sharpe annualise correctement : retours quotidiens (frequence fixe),
    annualisation standard par sqrt(252). C'est LE Sharpe a utiliser."""
    if daily_equity is None or len(daily_equity) < 3:
        return 0.0
    daily_returns = daily_equity[1:] / daily_equity[:-1] - 1.0
    daily_returns = daily_returns[np.isfinite(daily_returns)]
    if len(daily_returns) < 2:
        return 0.0
    excess = daily_returns - risk_free_daily
    std = np.std(excess, ddof=1)
    if std == 0 or not np.isfinite(std):
        return 0.0
    raw = np.mean(excess) / std
    annualized = raw * math.sqrt(TRADING_DAYS_PER_YEAR)
    return float(np.clip(annualized, -SHARPE_SANITY_CAP, SHARPE_SANITY_CAP))


def sortino_ratio_calendar(daily_equity: Optional[np.ndarray], risk_free_daily: float = 0.0) -> float:
    """Sortino annualise correctement : meme principe que sharpe_ratio_calendar
    mais avec la deviation "downside only" (MAR = 0)."""
    if daily_equity is None or len(daily_equity) < 3:
        return 0.0
    daily_returns = daily_equity[1:] / daily_equity[:-1] - 1.0
    daily_returns = daily_returns[np.isfinite(daily_returns)]
    if len(daily_returns) < 2:
        return 0.0
    excess = daily_returns - risk_free_daily
    downside = daily_returns[daily_returns < 0]
    if len(downside) < 2:
        return 0.0
    std_down = np.std(downside, ddof=1)
    if std_down == 0 or not np.isfinite(std_down):
        return 0.0
    raw = np.mean(excess) / std_down
    annualized = raw * math.sqrt(TRADING_DAYS_PER_YEAR)
    return float(np.clip(annualized, -SORTINO_SANITY_CAP, SORTINO_SANITY_CAP))


def sharpe_ratio_per_trade(returns: Sequence[float]) -> float:
    """
    ANCIEN calcul, CONSERVE A TITRE DE DIAGNOSTIC UNIQUEMENT.

    Sharpe NON-annualise, calcule directement sur la distribution des
    retours par trade (mean/std des retours, sans facteur temporel).
    Utile pour comparer rapidement des Einhers entre eux sur le meme
    actif/tf (frequence de trading comparable), mais ne represente PAS
    un Sharpe annualise standard et ne doit jamais etre affiche comme tel.
    """
    arr = _returns_array(returns)
    if len(arr) < 2:
        return 0.0
    std = np.std(arr, ddof=1)
    if std == 0 or not np.isfinite(std):
        return 0.0
    return float(np.mean(arr) / std)


def compute_all(
    returns: Sequence[float],
    equity: Sequence[float],
    exit_timestamps_ms: Sequence[int] = None,
) -> dict:
    """Calcule toutes les metriques en une passe.

    Args:
        returns: retours net (apres frais/slippage) de chaque trade, dans
            l'ordre chronologique d'ENTREE.
        equity: courbe d'equity trade-par-trade, equity[0] = valeur de
            depart, len(equity) == len(returns) + 1.
        exit_timestamps_ms: timestamps de CLOTURE de chaque trade (ms
            epoch), meme longueur et meme ordre que `returns`. Necessaire
            pour un Sharpe/Sortino annualise correctement (voir
            build_calendar_equity). Si absent, sharpe_ratio/sortino_ratio
            valent 0.0 et seul sharpe_per_trade (non-annualise) est fiable.
    """
    arr = _returns_array(returns)
    n_trades = len(arr)

    tpm = trades_per_month(list(exit_timestamps_ms)) if exit_timestamps_ms else 0.0

    daily_equity = None
    if exit_timestamps_ms:
        daily_equity = build_calendar_equity(equity, exit_timestamps_ms)

    sharpe = sharpe_ratio_calendar(daily_equity)
    sortino = sortino_ratio_calendar(daily_equity)

    return {
        "total_trades": int(n_trades),
        "win_rate": round(win_rate(arr), 4),
        "profit_factor": round(profit_factor(arr), 4),
        "sharpe_ratio": round(sharpe, 4),
        "sortino_ratio": round(sortino, 4),
        "sharpe_per_trade": round(sharpe_ratio_per_trade(arr), 4),
        "max_drawdown": round(max_drawdown(equity), 4),
        "avg_trade": round(avg_trade(arr), 6),
        "expectancy": round(expectancy(arr), 6),
        "total_return": round(total_return_compound(arr), 6),
        "trades_per_month": round(tpm, 2),
    }
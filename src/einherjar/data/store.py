"""Store de donnees persistant — DuckDB.

Initialise toutes les tables du systeme, expose append/query.
Tables : ohlcv, signals, orders, fills, positions, equity_curve,
rejections, einher_stats.

Reference : Section 4.4 du CDC EINHERJAR.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import duckdb
import polars as pl

from einherjar.core.models import (
    AccountState,
    Fill,
    Order,
    Position,
    Rejection,
    Signal,
)

DEFAULT_DB_PATH = Path(__file__).resolve().parents[3] / "data" / "einherjar.db"


class DataStore:
    """Gestionnaire DuckDB pour tout le journal du systeme.

    Attributs:
        conn: Connection DuckDB.
        db_path: Chemin vers le fichier .db.
    """

    def __init__(self, db_path: str | Path | None = None) -> None:
        """Ouvre ou cree la base DuckDB.

        Args:
            db_path: Chemin du fichier. Par defaut data/einherjar.db a la racine.
        """
        self.db_path = Path(db_path) if db_path else DEFAULT_DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = duckdb.connect(str(self.db_path))
        self._init_tables()

    def _init_tables(self) -> None:
        """Cree les tables si elles n'existent pas."""
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS ohlcv (
                asset VARCHAR,
                timeframe VARCHAR,
                timestamp TIMESTAMP,
                open DOUBLE,
                high DOUBLE,
                low DOUBLE,
                close DOUBLE,
                volume DOUBLE,
                PRIMARY KEY (asset, timeframe, timestamp)
            );
            """
        )
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS signals (
                signal_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                asset VARCHAR,
                direction VARCHAR,
                timeframe VARCHAR,
                einher_name VARCHAR,
                entry_price DOUBLE,
                tp_price DOUBLE,
                sl_price DOUBLE,
                confidence DOUBLE,
                timestamp TIMESTAMP,
                context JSON,
                executed BOOLEAN DEFAULT FALSE
            );
            """
        )
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS orders (
                order_id VARCHAR PRIMARY KEY,
                asset VARCHAR,
                order_type VARCHAR,
                direction VARCHAR,
                quantity DOUBLE,
                entry_price DOUBLE,
                tp_price DOUBLE,
                sl_price DOUBLE,
                einher_name VARCHAR,
                timestamp TIMESTAMP,
                status VARCHAR
            );
            """
        )
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS fills (
                fill_id VARCHAR PRIMARY KEY,
                order_id VARCHAR,
                asset VARCHAR,
                filled_qty DOUBLE,
                filled_price DOUBLE,
                fee DOUBLE,
                timestamp TIMESTAMP
            );
            """
        )
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS positions (
                position_id VARCHAR PRIMARY KEY,
                asset VARCHAR,
                direction VARCHAR,
                quantity DOUBLE,
                avg_entry_price DOUBLE,
                tp_price DOUBLE,
                sl_price DOUBLE,
                unrealized_pnl DOUBLE,
                einher_name VARCHAR,
                opened_at TIMESTAMP,
                asset_class VARCHAR
            );
            """
        )
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS equity_curve (
                snapshot_at TIMESTAMP PRIMARY KEY,
                cash DOUBLE,
                equity DOUBLE,
                margin_used DOUBLE,
                margin_available DOUBLE
            );
            """
        )
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS rejections (
                rejection_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                asset VARCHAR,
                einher_name VARCHAR,
                direction VARCHAR,
                reason VARCHAR,
                timestamp TIMESTAMP,
                context JSON
            );
            """
        )
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS einher_stats (
                einher_name VARCHAR,
                window_start TIMESTAMP,
                window_end TIMESTAMP,
                trade_count INTEGER,
                win_rate DOUBLE,
                sharpe DOUBLE,
                avg_profit DOUBLE,
                max_drawdown DOUBLE,
                status VARCHAR
            );
            """
        )
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS system_state (
                key VARCHAR PRIMARY KEY,
                value JSON,
                updated_at TIMESTAMP
            );
            """
        )

    def append_ohlcv(self, df: pl.DataFrame) -> None:
        """Ajoute des bougies OHLCV, ignore les doublons.

        Args:
            df: DataFrame polars avec colonnes [asset, timeframe, timestamp, open, high, low, close, volume].
        """
        self.conn.execute(
            """
            INSERT OR IGNORE INTO ohlcv
            SELECT * FROM df
            """
        )

    def query_ohlcv(
        self,
        asset: str,
        timeframe: str,
        since: str | None = None,
        limit: int = 500,
    ) -> pl.DataFrame:
        """Recupere l'historique OHLCV.

        Args:
            asset: Symbole.
            timeframe: Timeframe.
            since: Timestamp ISO minimal.
            limit: Nombre max de bougies.

        Returns:
            DataFrame polars trie par timestamp croissant.
        """
        where_clause = f"WHERE asset = '{asset}' AND timeframe = '{timeframe}'"
        if since:
            where_clause += f" AND timestamp >= '{since}'"
        return self.conn.execute(
            f"""
            SELECT * FROM ohlcv
            {where_clause}
            ORDER BY timestamp ASC
            LIMIT {limit}
            """
        ).pl()

    def append_signal(self, signal: Signal) -> None:
        """Journalise un signal.

        Args:
            signal: Signal emis.
        """
        self.conn.execute(
            """
            INSERT INTO signals (asset, direction, timeframe, einher_name,
                entry_price, tp_price, sl_price, confidence, timestamp, context)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                signal.asset,
                signal.direction.value,
                signal.timeframe.value,
                signal.einher_name,
                signal.entry_price,
                signal.tp_price,
                signal.sl_price,
                signal.confidence,
                signal.timestamp,
                str(signal.context) if isinstance(signal.context, str) else json.dumps(signal.context),
            ),
        )

    def append_rejection(self, rejection: Rejection) -> None:
        """Journalise un rejet.

        Args:
            rejection: Rejet du Risk Manager.
        """
        self.conn.execute(
            """
            INSERT INTO rejections (asset, einher_name, direction, reason, timestamp, context)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                rejection.signal.asset,
                rejection.signal.einher_name,
                rejection.signal.direction.value,
                rejection.reason,
                rejection.timestamp,
                str(rejection.signal.context) if isinstance(rejection.signal.context, str) else json.dumps(rejection.signal.context),
            ),
        )

    def snapshot_equity(self, state: AccountState) -> None:
        """Ecrit un snapshot de l'equity curve.

        Args:
            state: Etat du compte.
        """
        self.conn.execute(
            """
            INSERT OR REPLACE INTO equity_curve
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                state.timestamp,
                state.cash,
                state.equity,
                state.margin_used,
                state.margin_available,
            ),
        )

    def append_order(self, order: Order) -> None:
        """Journalise un ordre.

        Args:
            order: Ordre emis.
        """
        self.conn.execute(
            """
            INSERT OR REPLACE INTO orders (order_id, asset, order_type, direction,
                quantity, entry_price, tp_price, sl_price, einher_name, timestamp, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                order.order_id, order.asset, order.order_type.value,
                order.direction.value, order.quantity, order.entry_price,
                order.tp_price, order.sl_price, order.einher_name,
                order.timestamp, order.status,
            ),
        )

    def append_fill(self, fill: Fill) -> None:
        """Journalise un fill.

        Args:
            fill: Execution.
        """
        self.conn.execute(
            """
            INSERT OR REPLACE INTO fills (fill_id, order_id, asset,
                filled_qty, filled_price, fee, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                fill.fill_id, fill.order_id, fill.asset,
                fill.filled_qty, fill.filled_price, fill.fee, fill.timestamp,
            ),
        )

    def update_position(self, position: Position) -> None:
        """Met a jour ou insere une position.

        Args:
            position: Position courante.
        """
        self.conn.execute(
            """
            INSERT OR REPLACE INTO positions (position_id, asset, direction,
                quantity, avg_entry_price, tp_price, sl_price, unrealized_pnl,
                einher_name, opened_at, asset_class)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                position.position_id, position.asset, position.direction.value,
                position.quantity, position.avg_entry_price, position.tp_price,
                position.sl_price, position.unrealized_pnl, position.einher_name,
                position.opened_at, position.asset_class.value,
            ),
        )

    def remove_position(self, position_id: str) -> None:
        """Supprime une position fermee.

        Args:
            position_id: Identifiant de la position.
        """
        self.conn.execute("DELETE FROM positions WHERE position_id = ?", (position_id,))

    def get_equity_curve(self, limit: int = 500) -> list[dict[str, Any]]:
        """Retourne les snapshots reels, du plus ancien au plus recent."""
        return self.conn.execute(
            "SELECT snapshot_at, equity FROM equity_curve ORDER BY snapshot_at DESC LIMIT ?",
            (limit,),
        ).pl().sort("snapshot_at").to_dicts()

    def get_recent_signals(self, limit: int = 100) -> list[dict[str, Any]]:
        """Retourne les signaux journalises, sans generation de donnees."""
        return self.conn.execute(
            """SELECT asset, direction, timeframe, einher_name, entry_price,
                      tp_price, sl_price, confidence, timestamp, context, executed
               FROM signals ORDER BY timestamp DESC LIMIT ?""",
            (limit,),
        ).pl().to_dicts()

    def get_journal(self, limit: int = 200) -> list[dict[str, Any]]:
        """Retourne un flux unifie des evenements persistants."""
        return self.conn.execute(
            """SELECT timestamp, 'SIGNAL' AS type, asset, einher_name, confidence AS value,
                      CAST(context AS VARCHAR) AS details FROM signals
               UNION ALL
               SELECT timestamp, 'ORDER' AS type, asset, einher_name, quantity AS value,
                      status AS details FROM orders
               UNION ALL
               SELECT timestamp, 'REJECTION' AS type, asset, einher_name, NULL AS value,
                      reason AS details FROM rejections
               ORDER BY timestamp DESC LIMIT ?""",
            (limit,),
        ).pl().to_dicts()

    def get_einher_stats(self) -> list[dict[str, Any]]:
        """Retourne les statistiques calculees, sans valeur mock."""
        return self.conn.execute(
            """SELECT einher_name, trade_count, win_rate, sharpe, avg_profit,
                      max_drawdown, status, window_end
               FROM einher_stats ORDER BY window_end DESC"""
        ).pl().to_dicts()

    def set_kill_switch(self, enabled: bool) -> None:
        """Persiste l'etat du coupe-circuit entre les redemarrages."""
        self.conn.execute(
            "INSERT OR REPLACE INTO system_state VALUES (?, ?, CURRENT_TIMESTAMP)",
            ("kill_switch", json.dumps(enabled)),
        )

    def kill_switch_enabled(self) -> bool:
        """Retourne l'etat persiste du coupe-circuit."""
        row = self.conn.execute("SELECT value FROM system_state WHERE key = ?", ("kill_switch",)).fetchone()
        if row is None:
            return False
        try:
            return bool(json.loads(row[0]))
        except (TypeError, json.JSONDecodeError):
            return False

    def get_positions(self) -> list[Position]:
        """Recupere les positions ouvertes.

        Returns:
            Liste des positions.
        """
        from einherjar.core.enums import AssetClass, Direction
        df = self.conn.execute("SELECT * FROM positions").pl()
        positions = []
        for row in df.to_dicts():
            positions.append(Position(
                position_id=row["position_id"],
                asset=row["asset"],
                direction=Direction(row["direction"]),
                quantity=row["quantity"],
                avg_entry_price=row["avg_entry_price"],
                tp_price=row["tp_price"],
                sl_price=row["sl_price"],
                unrealized_pnl=row["unrealized_pnl"],
                einher_name=row["einher_name"],
                opened_at=row["opened_at"],
                asset_class=AssetClass(row["asset_class"]),
            ))
        return positions

    def close(self) -> None:
        """Ferme la connexion DuckDB proprement."""
        self.conn.close()

    def __enter__(self) -> DataStore:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

"""API FastAPI alimentee uniquement par les donnees persistantes."""

from __future__ import annotations

import json
import logging
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from einherjar.brokers.broker_utils import ASSET_CLASS_MAP
from einherjar.data.store import DataStore

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = PROJECT_ROOT / "config" / "settings.json"
CREDENTIALS_PATH = PROJECT_ROOT / "config" / "credentials.json"
DB_PATH = PROJECT_ROOT / "data" / "einherjar.db"
DASHBOARD_BUILD = PROJECT_ROOT / "dashboard" / "einherjar-ui" / "dist"
CORPUS_PATH = PROJECT_ROOT / "config" / "corpus_v2.json"


def _load_credentials() -> dict[str, Any] | None:
    """Charge les identifiants seulement lorsqu'ils sont valides."""
    if not CREDENTIALS_PATH.exists():
        return None
    try:
        data = json.loads(CREDENTIALS_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def _format_datetime(value: Any) -> str:
    """Normalise les datetimes DuckDB pour le client JSON."""
    return value.isoformat() if hasattr(value, "isoformat") else str(value)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialise le store reel et, optionnellement, le broker."""
    app.state.store = DataStore(DB_PATH)
    app.state.ctrader = None
    credentials = _load_credentials()
    if credentials:
        try:
            from einherjar.brokers import CTraderAdapter

            adapter = CTraderAdapter(
                client_id=str(credentials.get("client_id", "")),
                client_secret=str(credentials.get("client_secret", "")),
                access_token=str(credentials.get("access_token", "")),
                account_id=int(credentials.get("account_id", 0)),
                host=str(credentials.get("host", "demo.ctraderapi.com")),
                port=int(credentials.get("port", 5035)),
                broker_name=str(credentials.get("broker_name", "ic_markets")),
            )
            if await adapter.connect():
                app.state.ctrader = adapter
        except Exception as exc:
            logger.warning("Connexion cTrader indisponible: %s", exc)
    yield
    if app.state.ctrader is not None:
        await app.state.ctrader.disconnect()
    app.state.store.close()


app = FastAPI(title="Einherjar API", version="2.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3166", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
async def health() -> dict[str, Any]:
    """Expose l'etat reel des dependances."""
    broker = {"connected": False, "host": None, "circuitState": "CLOSED"}
    if app.state.ctrader is not None:
        status = app.state.ctrader.get_status()
        broker = {
            "connected": status["connected"],
            "host": status["host"],
            "circuitState": status["circuit_state"],
        }
    return {
        "status": "paused" if app.state.store.kill_switch_enabled() else "ok",
        "timestamp": datetime.now(UTC).isoformat(),
        "components": {
            "database": "ok",
            "config": "ok" if CONFIG_PATH.exists() else "missing",
            "corpus": "ok" if CORPUS_PATH.exists() else "missing",
            "ctrader": broker,
        },
    }


@app.get("/api/account")
async def account() -> dict[str, Any]:
    """Retourne le compte broker, sans fallback invente."""
    if app.state.ctrader is None:
        return {"connected": False, "reason": "ctrader_not_connected"}
    try:
        state = await app.state.ctrader.get_account()
    except Exception as exc:
        raise HTTPException(status_code=503, detail="ctrader_account_unavailable") from exc
    app.state.store.snapshot_equity(state)
    return {
        "balance": state.cash,
        "cash": state.cash,
        "equity": state.equity,
        "margin": state.margin_used,
        "marginFree": state.margin_available,
        "leverage": state.leverage,
        "currency": "USD",
        "accountId": app.state.ctrader.account_id,
        "connected": True,
    }


@app.get("/api/overview")
async def overview() -> dict[str, Any]:
    """Construit les metriques depuis l'equity et positions persistantes."""
    curve = app.state.store.get_equity_curve()
    positions = app.state.store.get_positions()
    latest = curve[-1]["equity"] if curve else None
    first = curve[0]["equity"] if curve else None
    change = ((latest / first - 1) * 100) if latest and first else None
    exposure: dict[str, float] = {}
    for position in positions:
        name = ASSET_CLASS_MAP.get(position.asset, position.asset_class).value.upper()  # pyright: ignore[reportOptionalMemberAccess]
        exposure[name] = exposure.get(name, 0.0) + position.quantity * position.avg_entry_price
    return {
        "metrics": [
            {"label": "EQUITY", "value": latest, "format": "currency"},
            {"label": "RETURN", "value": change, "format": "percent"},
            {"label": "OPEN POSITIONS", "value": len(positions), "format": "number"},
        ],
        "equity": [{"time": _format_datetime(row["snapshot_at"]), "value": row["equity"]} for row in curve],
        "exposure": [{"class": asset_class, "value": value} for asset_class, value in sorted(exposure.items())],
    }


@app.get("/api/positions")
async def positions() -> list[dict[str, Any]]:
    """Retourne les positions DuckDB les plus recentes."""
    return [
        {
            "id": position.position_id,
            "asset": position.asset,
            "assetClass": position.asset_class.value.upper(),
            "direction": position.direction.value.upper(),
            "entryPrice": position.avg_entry_price,
            "currentPrice": (
                position.avg_entry_price + position.unrealized_pnl / position.quantity
                if position.quantity and position.direction.value == "long"
                else position.avg_entry_price - position.unrealized_pnl / position.quantity
                if position.quantity
                else position.avg_entry_price
            ),
            "quantity": position.quantity,
            "tpPrice": position.tp_price,
            "slPrice": position.sl_price,
            "pnl": position.unrealized_pnl,
            "pnlPercent": position.unrealized_pnl / (position.quantity * position.avg_entry_price) * 100
            if position.quantity and position.avg_entry_price
            else 0.0,
            "einher": position.einher_name,
            "openedAt": _format_datetime(position.opened_at),
            "timeInPosition": "",
        }
        for position in app.state.store.get_positions()
    ]


@app.get("/api/forming")
async def forming() -> list[dict[str, Any]]:
    """Expose les derniers signaux bruts, jamais des signaux factices."""
    rows = app.state.store.get_recent_signals()
    return [
        {
            "asset": row["asset"],
            "timeframe": row["timeframe"],
            "direction": row["direction"].upper(),
            "einher": row["einher_name"],
            "confidence": row["confidence"],
            "conditions": [],
            "triggered": row["executed"],
            "timestamp": _format_datetime(row["timestamp"]),
        }
        for row in rows
    ]


@app.get("/api/performance")
async def performance() -> dict[str, Any]:
    """Retourne seulement les statistiques ecrites par le calibrateur."""
    return {
        "einhers": [
            {
                "id": row["einher_name"],
                "name": row["einher_name"],
                "description": "",
                "status": row["status"],
                "winRate": row["win_rate"],
                "totalTrades": row["trade_count"],
                "avgReturn": row["avg_profit"],
                "sharpe": row["sharpe"],
                "lastSignal": _format_datetime(row["window_end"]),
            }
            for row in app.state.store.get_einher_stats()
        ]
    }


@app.get("/api/journal")
async def journal() -> list[dict[str, Any]]:
    """Retourne le journal unifie DuckDB."""
    rows = app.state.store.get_journal()
    for index, row in enumerate(rows):
        row["id"] = f"journal-{index}"
        row["einher"] = row.pop("einher_name")
        row["timestamp"] = _format_datetime(row["timestamp"])
    return rows


@app.post("/api/kill_switch")
async def kill_switch(enabled: bool = True) -> dict[str, bool]:
    """Pause les nouvelles executions, sans fermer les positions."""
    app.state.store.set_kill_switch(enabled)
    return {"enabled": enabled}


if DASHBOARD_BUILD.exists():
    app.mount("/assets", StaticFiles(directory=DASHBOARD_BUILD / "assets"), name="assets")

    @app.get("/{full_path:path}")
    async def serve_spa(full_path: str) -> FileResponse:
        """Sert le SPA React pour les routes non API."""
        candidate = DASHBOARD_BUILD / full_path
        if candidate.is_file():
            return FileResponse(candidate)
        return FileResponse(DASHBOARD_BUILD / "index.html")

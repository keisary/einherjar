"""API FastAPI alimentee uniquement par les donnees persistantes."""

from __future__ import annotations

import json
import logging
import statistics
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from urllib.parse import parse_qs, quote

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from einherjar.api import auth as auth_module
from einherjar.config.credentials import charger_credentials

from einherjar.brokers.broker_utils import ASSET_CLASS_MAP
from einherjar.data.store import DataStore

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = PROJECT_ROOT / "config" / "settings.json"
CREDENTIALS_PATH = PROJECT_ROOT / "config" / "credentials.json"
DB_PATH = PROJECT_ROOT / "data" / "einherjar.db"
DASHBOARD_BUILD = PROJECT_ROOT / "dashboard" / "einherjar-ui" / "dist"
CORPUS_PATH = PROJECT_ROOT / "outputs" / "corpus.jsonl"


def _load_corpus() -> dict[str, Any]:
    """Charge le corpus de recherche (837 einhers) en memoire.

    C'est la source de verite des einhers que le systeme surveille : le corpus
    est fige par la recherche, l'etat runtime (ACTIVE/PROBATION) vient de la base
    DuckDB quand l'einher y a ete enregistre.

    Returns:
        Dict avec `entries` (liste brute), `univers` (comptes par couple) et
        `classes` (comptes par classe d'actif). Vide si le corpus est absent.
    """
    vide: dict[str, Any] = {"entries": [], "univers": {}, "classes": {}}
    if not CORPUS_PATH.exists():
        return vide
    try:
        from einherjar.signals.corpus_bridge import load_entries

        entries = load_entries(CORPUS_PATH)
    except Exception as exc:  # noqa: BLE001 - l'API doit demarrer sans corpus
        logger.warning("Corpus illisible (%s) : %s", CORPUS_PATH, exc)
        return vide

    univers: dict[tuple[str, str, str], int] = {}
    classes: dict[str, int] = {}
    for entry in entries:
        uni = entry.get("universe") or {}
        cle = (str(uni.get("asset", "?")), str(uni.get("timeframe", "?")), str(uni.get("asset_class", "?")))
        univers[cle] = univers.get(cle, 0) + 1
        classes[cle[2]] = classes.get(cle[2], 0) + 1
    return {"entries": entries, "univers": univers, "classes": classes}


def _load_credentials() -> dict[str, Any] | None:
    """Charge les identifiants cTrader (fichier local complete par l'environnement).

    Sur un hebergement (Render, Docker), `config/credentials.json` est gitignore et
    absent : les variables `EINHERJAR_*` prennent alors le relais.
    """
    return charger_credentials()


def _format_datetime(value: Any) -> str:
    """Normalise les datetimes DuckDB pour le client JSON."""
    return value.isoformat() if hasattr(value, "isoformat") else str(value)


def _duree_ecoulee(depuis: Any) -> str | None:
    """Duree lisible depuis un horodatage (ex. "2h 15m"), ou None si inconnu.

    Le dashboard affichait une chaine vide a la place de la duree de detention :
    elle se calcule ici, a partir de l'horodatage reel d'ouverture.
    """
    if not hasattr(depuis, "timestamp"):
        return None
    try:
        secondes = max(0, int((datetime.now(UTC) - depuis).total_seconds()))
    except (TypeError, ValueError, OSError):
        return None
    heures, reste = divmod(secondes, 3600)
    minutes = reste // 60
    if heures >= 24:
        return f"{heures // 24}j {heures % 24}h"
    if heures:
        return f"{heures}h {minutes:02d}m"
    return f"{minutes}m"


# Adaptateur cTrader DEJA connecte, impose par l'appelant (main.py). Le reactor
# Twisted est un singleton PAR PROCESS : lancer une deuxieme connexion fait lever
# `ReactorAlreadyRunning` dans le nouveau thread, et l'arret de ce thread coupait
# le reactor du premier — donc une connexion valide. Un seul adaptateur par process.
_adapter_impose: Any | None = None


def utiliser_adapter(adapter: Any) -> None:
    """Impose a l'API l'adaptateur cTrader deja connecte (un seul par process).

    Args:
        adapter: Instance `CTraderAdapter` connectee, ou None pour revenir au
            comportement par defaut (connexion depuis les credentials).
    """
    global _adapter_impose
    _adapter_impose = adapter


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialise le store reel et, optionnellement, le broker."""
    app.state.store = DataStore(DB_PATH)
    app.state.ctrader = None
    app.state.corpus = _load_corpus()
    logger.info("Corpus: %d einhers charges", len(app.state.corpus["entries"]))
    credentials = _load_credentials()
    if credentials:
        # L'environnement (demo|live) est expose par /api/health pour que le
        # dashboard affiche le compte reellement utilise, sans logique locale.
        app.state.environment = str(credentials.get("environment", "demo")).lower()
    if _adapter_impose is not None:
        app.state.ctrader = _adapter_impose
        logger.info(
            "Broker cTrader fourni par l'appelant (account_id=%s)",
            getattr(_adapter_impose, "account_id", "?"),
        )
    elif credentials:
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

# ---------------------------------------------------------------------------
# Authentification
# ---------------------------------------------------------------------------
# Identifiants charges une seule fois (env ou mot de passe genere) et limiteur de
# tentatives par adresse : le systeme trade un compte reel, l'application ne doit
# pas etre lisible par quiconque connait l'URL.
_IDENTIFIANTS: auth_module.Identifiants | None = None
_LIMITEUR = auth_module.LimiteurTentatives()


def _identifiants() -> auth_module.Identifiants:
    """Identifiants de l'application (charges une fois par process)."""
    global _IDENTIFIANTS
    if _IDENTIFIANTS is None:
        _IDENTIFIANTS = auth_module.charger_identifiants()
    return _IDENTIFIANTS


def _adresse_client(requete: Request) -> str:
    """Adresse du client (proxy de l'hebergeur inclus) pour le limiteur."""
    transmise = (requete.headers.get("x-forwarded-for") or "").split(",")[0].strip()
    if transmise:
        return transmise
    return requete.client.host if requete.client else "inconnu"


def _cible_sure(cible: str | None) -> str:
    """Nettoie la destination post-connexion (redirection interne uniquement)."""
    if not cible or not cible.startswith("/") or cible.startswith("//"):
        return "/"
    return cible


@app.middleware("http")
async def authentification(requete: Request, suivante: Any) -> Response:
    """Protege l'application entiere sauf la page de connexion et la sonde publique.

    Les appels API non authentifies recoivent un 401 JSON (le dashboard le voit),
    les pages HTML une redirection vers la connexion.
    """
    chemin = requete.url.path
    if auth_module.chemin_public(chemin) or chemin.startswith("/assets/"):
        return await suivante(requete)
    jeton = auth_module.extraire_cookie(requete.headers.get("cookie"))
    if auth_module.lire_jeton(jeton, _identifiants()) is not None:
        return await suivante(requete)
    if chemin.startswith("/api/"):
        return JSONResponse({"detail": "authentification requise"}, status_code=401)
    cible = "/login" if chemin in ("/", "") else f"/login?suite={quote(chemin, safe='')}"
    return RedirectResponse(cible, status_code=303)


@app.get("/healthz")
async def healthz() -> dict[str, str]:
    """Sonde de sante PUBLIQUE (aucune donnee de compte) pour l'hebergeur."""
    return {"status": "ok"}


@app.get("/login")
async def page_login(requete: Request) -> Response:
    """Affiche la page de connexion (ou renvoie a l'application si deja connecte)."""
    suite = _cible_sure(requete.query_params.get("suite"))
    jeton = auth_module.extraire_cookie(requete.headers.get("cookie"))
    if auth_module.lire_jeton(jeton, _identifiants()) is not None:
        return RedirectResponse(suite, status_code=303)
    return HTMLResponse(auth_module.page_connexion(None, suite))


@app.post("/login")
async def connexion(requete: Request) -> Response:
    """Verifie le formulaire, pose le cookie de session signe, puis redirige."""
    corps = (await requete.body()).decode("utf-8", errors="replace")
    champs = {cle: valeurs[0] for cle, valeurs in parse_qs(corps).items()}
    suite = _cible_sure(champs.get("suite"))
    adresse = _adresse_client(requete)

    if not _LIMITEUR.autorise(adresse):
        logger.warning("AUTH: trop de tentatives depuis %s", adresse)
        return HTMLResponse(
            auth_module.page_connexion(
                "Trop de tentatives. Reessayez dans quelques minutes.", suite
            ),
            status_code=429,
        )

    if not auth_module.verifier_identifiants(
        champs.get("utilisateur", ""), champs.get("mot_de_passe", ""), _identifiants()
    ):
        _LIMITEUR.enregistrer_echec(adresse)
        logger.warning("AUTH: echec de connexion depuis %s", adresse)
        return HTMLResponse(
            auth_module.page_connexion("Identifiant ou mot de passe incorrect.", suite),
            status_code=401,
        )

    _LIMITEUR.reinitialiser(adresse)
    reponse = RedirectResponse(suite, status_code=303)
    reponse.set_cookie(
        auth_module.COOKIE_NAME,
        auth_module.creer_jeton(_identifiants()),
        max_age=auth_module.DUREE_SESSION_S,
        httponly=True,
        samesite="lax",
        secure=auth_module.souvenir_sure(requete),
        path="/",
    )
    logger.info("AUTH: connexion reussie (%s)", adresse)
    return reponse


@app.get("/logout")
async def deconnexion() -> Response:
    """Efface la session et renvoie vers la page de connexion."""
    reponse = RedirectResponse("/login", status_code=303)
    reponse.delete_cookie(auth_module.COOKIE_NAME, path="/")
    return reponse


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
            # Cause exacte du dernier echec de connexion (None si connecte) : sans
            # elle, un compte non autorise ressemble a un compte simplement "off".
            "lastError": status.get("last_error"),
        }
    return {
        "status": "paused" if app.state.store.kill_switch_enabled() else "ok",
        "timestamp": datetime.now(UTC).isoformat(),
        "environment": getattr(app.state, "environment", None),
        "components": {
            "database": "ok",
            "config": "ok" if CONFIG_PATH.exists() else "missing",
            "corpus": "ok" if CORPUS_PATH.exists() else "missing",
            "corpusEinhers": len(getattr(app.state, "corpus", {}).get("entries", [])),
            "corpusUnivers": len(getattr(app.state, "corpus", {}).get("univers", {})),
            # Etat REEL de la boucle d'inference (publie par le scheduler) et
            # couverture des features : sans cela le dashboard doit inventer.
            "loop": app.state.store.get_state("loop"),
            "couverture": app.state.store.get_state("couverture"),
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
            # Echelle reellement surveillee (corpus de recherche) — la 4e carte du bandeau.
            {"label": "EINHERS SUIVIS", "value": len(app.state.corpus["entries"]), "format": "number"},
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
            "timeInPosition": _duree_ecoulee(position.opened_at),
        }
        for position in app.state.store.get_positions()
    ]


@app.get("/api/forming")
async def forming() -> list[dict[str, Any]]:
    """Expose les derniers signaux bruts, jamais des signaux factices."""
    rows = app.state.store.get_recent_signals()
    return [
        {
            "id": f"signal-{index}",
            "asset": row["asset"],
            "timeframe": row["timeframe"],
            "direction": row["direction"].upper(),
            "einher": row["einher_name"],
            "confidence": row["confidence"],
            "conditions": [],
            "triggered": row["executed"],
            "timestamp": _format_datetime(row["timestamp"]),
        }
        for index, row in enumerate(rows)
    ]


def _mediane(valeurs: list[float]) -> float | None:
    """Mediane d'une serie en ignorant les valeurs absentes."""
    propres = [float(v) for v in valeurs if isinstance(v, (int, float))]
    return round(statistics.median(propres), 6) if propres else None


@app.get("/api/performance")
async def performance(
    asset: str | None = None,
    timeframe: str | None = None,
    asset_class: str | None = None,
    direction: str | None = None,
    sort: str = "sharpe",
    limit: int = 250,
    offset: int = 0,
) -> dict[str, Any]:
    """Einhers du corpus, enrichis des statistiques runtime quand elles existent.

    Le corpus de recherche est la source de verite des einhers surveilles ; la
    table `einher_stats` (calibrateur) ne fournit que les einhers reellement
    evalues en live. Un einher jamais evalue a donc `status: None` — l'API
    n'invente aucun etat ni aucune performance.
    """
    from einherjar.signals.corpus_bridge import tree_to_expr

    corpus = app.state.corpus
    runtime = {row["einher_name"]: row for row in app.state.store.get_einher_stats()}

    def _match(entry: dict[str, Any]) -> bool:
        uni = entry.get("universe") or {}
        if asset and str(uni.get("asset", "")).upper() != asset.upper():
            return False
        if timeframe and str(uni.get("timeframe", "")).lower() != timeframe.lower():
            return False
        if asset_class and str(uni.get("asset_class", "")).lower() != asset_class.lower():
            return False
        if direction and str(entry.get("direction", "")).upper() != direction.upper():
            return False
        return True

    einhers: list[dict[str, Any]] = []
    for entry in corpus["entries"]:
        if not _match(entry):
            continue
        uni = entry.get("universe") or {}
        metrics = entry.get("metrics") or {}
        live = runtime.get(str(entry.get("id", "")))
        einhers.append(
            {
                # Contrat consomme par le dashboard (useEinhers -> /api/performance)
                "id": entry.get("id"),
                "name": entry.get("id"),
                "description": tree_to_expr(entry.get("condition_tree")),
                "status": live["status"] if live else None,
                "winRate": metrics.get("win_rate"),
                "totalTrades": metrics.get("n_trades"),
                "avgReturn": metrics.get("avg_net_return"),
                "sharpe": metrics.get("sharpe_ratio"),
                "lastSignal": _format_datetime(live["window_end"]) if live else None,
                # Champs reels supplementaires (recherche)
                "asset": uni.get("asset"),
                "assetClass": uni.get("asset_class"),
                "timeframe": uni.get("timeframe"),
                "horizon": uni.get("horizon"),
                "direction": entry.get("direction"),
                "amplitudeBars": entry.get("amplitude_bars"),
                "tpPct": entry.get("tp_pct"),
                "slPct": entry.get("sl_pct"),
                "totalReturn": metrics.get("total_return"),
                "maxDrawdown": metrics.get("max_drawdown"),
                "profitFactor": metrics.get("profit_factor"),
                "avgHoldingBars": metrics.get("avg_holding_bars"),
                "tpHitRate": metrics.get("tp_hit_rate"),
                "alpha": metrics.get("alpha"),
                "pValue": metrics.get("p_value"),
                "buyHoldReturn": metrics.get("buy_hold_return"),
                "model": (entry.get("source") or {}).get("model"),
                "createdAt": entry.get("created_at"),
            }
        )

    champs = {
        "sharpe": "sharpe",
        "winRate": "winRate",
        "trades": "totalTrades",
        "avgReturn": "avgReturn",
        "totalReturn": "totalReturn",
        "alpha": "alpha",
        "id": "id",
    }
    cle = champs.get(sort, "sharpe")
    einhers.sort(key=lambda e: (e[cle] is None, -(e[cle] or 0) if cle != "id" else 0, str(e[cle])))

    total = len(einhers)
    page = einhers[offset : offset + max(1, limit)]
    return {
        "total": total,
        "returned": len(page),
        "einhers": page,
        "summary": {
            "sharpeMedian": _mediane([e["sharpe"] for e in einhers]),
            "winRateMedian": _mediane([e["winRate"] for e in einhers]),
            "avgReturnMedian": _mediane([e["avgReturn"] for e in einhers]),
            "totalReturnMedian": _mediane([e["totalReturn"] for e in einhers]),
            "alphaMedian": _mediane([e["alpha"] for e in einhers]),
            "pValueMedian": _mediane([e["pValue"] for e in einhers]),
            "tradesTotal": sum(int(e["totalTrades"] or 0) for e in einhers),
        },
        "universes": sorted(
            (
                {"asset": a, "timeframe": tf, "assetClass": cls, "einhers": n}
                for (a, tf, cls), n in corpus["univers"].items()
            ),
            key=lambda u: (-u["einhers"], u["asset"], u["timeframe"]),
        ),
        "classes": corpus["classes"],
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

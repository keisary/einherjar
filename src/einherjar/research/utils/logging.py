"""
utils/logging.py — Configuration logging structuré.

Un fichier de log par run, horodaté. Pas de rotation ici, c'est au caller
de gérer (la découverte est ponctuelle, pas un service long-running).
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

_CONFIGURED = False
_RUN_DIR: Optional[Path] = None


def get_run_dir(base: Path = Path("outputs")) -> Path:
    """Retourne (et crée) le dossier du run courant : `outputs/run_<UTC>/`."""
    global _RUN_DIR
    if _RUN_DIR is None:
        stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        _RUN_DIR = base / f"run_{stamp}"
        _RUN_DIR.mkdir(parents=True, exist_ok=True)
    return _RUN_DIR


def configure_logging(level: str = "INFO", run_dir: Optional[Path] = None) -> Path:
    """Configure le logging racine. Retourne le chemin du fichier de log créé.

    - Console (stderr) en mode court.
    - Fichier `run.log` dans le run_dir (mode verbeux, structured).
    - Idempotent : ré-appeler ne duplique pas les handlers.
    """
    global _CONFIGURED
    rd = run_dir or get_run_dir()
    log_file = rd / "run.log"

    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))

    if _CONFIGURED:
        return log_file

    fmt_console = logging.Formatter(
        "[%(asctime)s] %(levelname)-7s %(name)s :: %(message)s",
        datefmt="%H:%M:%S",
    )
    fmt_file = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(name)s | %(funcName)s:%(lineno)d | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )

    h_console = logging.StreamHandler(sys.stderr)
    h_console.setFormatter(fmt_console)
    root.addHandler(h_console)

    h_file = logging.FileHandler(log_file, mode="a", encoding="utf-8")
    h_file.setFormatter(fmt_file)
    root.addHandler(h_file)

    _CONFIGURED = True
    root.info("=" * 72)
    root.info("Run démarré — logs dans %s", log_file)
    root.info("=" * 72)
    return log_file

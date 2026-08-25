"""paths.py - Chemins absolus du projet, derives de l'emplacement du code.

Source de verite unique pour tous les chemins du pipeline xgb_einhers.
La racine du repo est derivee de __file__ (src/einherjar/research/xgb_einhers/paths.py
-> 5 niveaux au-dessus = einherjar/). Aucune variable d'environnement, aucune
dependance au repertoire courant (cwd) : un run lance depuis VS Code, un terminal
ou un worker multiprocessing ecrit toujours au meme endroit.

Convention (verifiee sur disque le 2026-08-24) :
    REPO_ROOT      = D:/midas_v2/einherjar
    OUTPUTS_DIR    = D:/midas_v2/einherjar/outputs
    COMPILED_DIR   = D:/midas_v2/midasV3/src/data/compiled
    OHLCV_DIR      = D:/midas_v2/technical_agent_dataset_brut
    CONFIG_DIR     = D:/midas_v2/einherjar/src/einherjar/research/config
"""
from __future__ import annotations

from pathlib import Path

# Racine du repo einherjar : ce fichier est src/einherjar/research/xgb_einhers/paths.py
REPO_ROOT = Path(__file__).resolve().parents[4]
OUTPUTS_DIR = REPO_ROOT / "outputs"
CONFIG_DIR = REPO_ROOT / "src" / "einherjar" / "research" / "config"

# Donnees MIDAS V3 compilees et OHLCV bruts (voisins du repo sur D:)
PROJECTS_ROOT = REPO_ROOT.parent          # D:/midas_v2
COMPILED_DIR = PROJECTS_ROOT / "midasV3" / "src" / "data" / "compiled"
OHLCV_DIR = PROJECTS_ROOT / "technical_agent_dataset_brut"
FEES_CONFIG_PATH = REPO_ROOT / "config" / "fees_ctrader.json"
TAXONOMY_PATH = CONFIG_DIR / "features_taxonomy.json"
ASSETS_CONFIG_PATH = CONFIG_DIR / "assets_v1.json"

# Sorties canoniques (corpus/archive xgb)
CORPUS_PATH = OUTPUTS_DIR / "corpus.jsonl"
ARCHIVE_PATH = OUTPUTS_DIR / "archive.jsonl"
DISCOVER_STATE_PATH = OUTPUTS_DIR / "discover_state.jsonl"
DISCOVER_REPORT_PATH = OUTPUTS_DIR / "discover_report.json"


def resolve_output(path: Path | str | None, default: Path) -> Path:
    """Resout un chemin de sortie : absolu -> tel quel ; relatif -> ancre sous OUTPUTS_DIR.

    Garantit qu'un chemin relatif passe en CLI n'ecrit plus dans le cwd du lanceur
    mais dans le dossier outputs canonique du repo (fix BUG-01).
    """
    if path is None:
        return default
    p = Path(path)
    if not p.is_absolute():
        # Si le chemin relatif commence deja par "outputs/...", on NE le
        # re-prefixe PAS (evite outputs/outputs/ quand un user passe
        # --corpus outputs/corpus.jsonl : le nom est deja ancre au repo).
        if p.parts and p.parts[0] == "outputs":
            p = REPO_ROOT / p
        else:
            p = OUTPUTS_DIR / p
    return p


for _d in (OUTPUTS_DIR,):
    _d.mkdir(parents=True, exist_ok=True)

"""
==========================================================
Paths Configuration
==========================================================

Centralise tous les chemins utilisés par le moteur.

Aucun chemin ne doit être codé en dur ailleurs.
"""

from pathlib import Path

# ---------------------------------------------------------
# ROOT
# ---------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

# ---------------------------------------------------------
# DATA
# ---------------------------------------------------------

DATA = ROOT / "data"

DATASETS = DATA / "datasets"
CACHE = DATA / "cache"
TEMP = DATA / "temp"
CHECKPOINTS = DATA / "checkpoints"

# ---------------------------------------------------------
# OUTPUTS
# ---------------------------------------------------------

OUTPUTS = ROOT / "outputs"

CORPUS = OUTPUTS / "corpus"
REJECTED = OUTPUTS / "rejected"
REPORTS = OUTPUTS / "reports"
LOGS = OUTPUTS / "logs"
ANALYTICS = OUTPUTS / "analytics"
ARCHIVE = OUTPUTS / "archive"

# ---------------------------------------------------------
# EXPORTS
# ---------------------------------------------------------

CORPUS_JSON = CORPUS / "json"
CORPUS_PARQUET = CORPUS / "parquet"
CORPUS_INDEX = CORPUS / "indexes"

# ---------------------------------------------------------
# MEMORY
# ---------------------------------------------------------

MEMORY = ROOT / "memory"

# ---------------------------------------------------------
# KNOWLEDGE
# ---------------------------------------------------------

KNOWLEDGE = ROOT / "knowledge"

# ---------------------------------------------------------
# CREATE DIRECTORIES
# ---------------------------------------------------------

DIRECTORIES = (
    DATASETS,
    CACHE,
    TEMP,
    CHECKPOINTS,
    CORPUS_JSON,
    CORPUS_PARQUET,
    CORPUS_INDEX,
    REJECTED,
    REPORTS,
    LOGS,
    ANALYTICS,
    ARCHIVE,
)

for directory in DIRECTORIES:
    directory.mkdir(parents=True, exist_ok=True)
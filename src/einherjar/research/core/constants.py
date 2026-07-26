"""
==========================================================
Global Constants
==========================================================

Constantes globales utilisées dans tout le moteur.

Aucune constante de configuration ne doit être définie ici.
Les paramètres utilisateurs appartiennent au dossier
config/.
"""

from __future__ import annotations

from version import __version__


# ==========================================================
# ENGINE
# ==========================================================

ENGINE_NAME = "Einherjar Discovery Engine"

ENGINE_VERSION = __version__


# ==========================================================
# PHASES
# ==========================================================

PHASE_DATASET = "dataset"

PHASE_DISCOVERY = "discovery"

PHASE_VALIDATION = "validation"

PHASE_EXECUTION = "execution"

PHASE_PORTFOLIO = "portfolio"


PHASES = (
    PHASE_DATASET,
    PHASE_DISCOVERY,
    PHASE_VALIDATION,
    PHASE_EXECUTION,
    PHASE_PORTFOLIO,
)


# ==========================================================
# DATASET SPLITS
# ==========================================================

TRAIN_SPLIT = "train"

VALIDATION_SPLIT = "validation"

TEST_SPLIT = "test"


DATASET_SPLITS = (
    TRAIN_SPLIT,
    VALIDATION_SPLIT,
    TEST_SPLIT,
)


# ==========================================================
# BOOLEAN STATES
# ==========================================================

STATUS_INITIALIZED = "initialized"

STATUS_RUNNING = "running"

STATUS_FINISHED = "finished"
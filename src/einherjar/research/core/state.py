"""
==========================================================
Engine State
==========================================================

Représente l'état courant du moteur.

Cet objet ne contient aucune logique métier. Il permet
uniquement de suivre l'exécution du pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass(slots=True)
class EngineState:
    """
    Etat courant du moteur.
    """

    initialized: bool = False

    running: bool = False

    finished: bool = False

    current_phase: str = "initialization"

    started_at: datetime | None = None

    finished_at: datetime | None = None

    # ==================================================
    # LIFECYCLE
    # ==================================================

    def initialize(self) -> None:

        self.initialized = True

        self.current_phase = "initialized"

    def start(self) -> None:

        self.running = True

        self.finished = False

        self.started_at = datetime.utcnow()

        self.current_phase = "running"

    def stop(self) -> None:

        self.running = False

        self.finished = True

        self.finished_at = datetime.utcnow()

        self.current_phase = "finished"

    def set_phase(
        self,
        phase: str,
    ) -> None:

        self.current_phase = phase

    def reset(self) -> None:

        self.initialized = False

        self.running = False

        self.finished = False

        self.current_phase = "initialization"

        self.started_at = None

        self.finished_at = None

    # ==================================================
    # PROPERTIES
    # ==================================================

    @property
    def duration(self):

        if self.started_at is None:
            return None

        end = (
            datetime.utcnow()
            if self.running
            else self.finished_at
        )

        if end is None:
            return None

        return end - self.started_at

    # ==================================================
    # PYTHON PROTOCOL
    # ==================================================

    def __repr__(self) -> str:

        return (
            "EngineState("
            f"phase='{self.current_phase}', "
            f"running={self.running}"
            ")"
        )
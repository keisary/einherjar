"""
==========================================================
Engine State
==========================================================

Représente l'état d'avancement du pipeline pour UNE PAIRE
asset / timeframe.

Cet objet est créé par Engine.run_pair() et détruit une
fois la paire terminée. Il ne doit jamais être partagé
entre paires.

L'état suit la séquence stricte des phases :

    initialisation → dataset → contract → discovery
    → validation → execution → portfolio → memory
    → knowledge → export → terminal (success | failed)

Chaque transition de phase est enregistrée. Toute erreur
de phase est capturée dans le slot d'erreur correspondant.
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from datetime import datetime
from datetime import timezone
from typing import Any


PHASE_NAMES: tuple[str, ...] = (
    "initialization",
    "dataset",
    "contract",
    "discovery",
    "validation",
    "execution",
    "portfolio",
    "memory",
    "knowledge",
    "export",
    "terminal",
)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _ensure_phase_name(phase: str) -> str:
    if phase not in PHASE_NAMES:
        raise ValueError(
            f"unknown phase '{phase}'. "
            f"Valid phases: {PHASE_NAMES}."
        )
    return phase


@dataclass(slots=True)
class PhaseStatus:
    """
    Statut d'une phase individuelle.
    """

    status: str = "pending"   # pending | running | success | failed | skipped
    started_at: datetime | None = None
    finished_at: datetime | None = None
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def start(self) -> None:
        self.status = "running"
        self.started_at = _utc_now()
        self.error = None

    def succeed(self, *, metadata: dict[str, Any] | None = None) -> None:
        self.status = "success"
        self.finished_at = _utc_now()
        if metadata:
            self.metadata.update(metadata)

    def fail(self, error: str) -> None:
        self.status = "failed"
        self.finished_at = _utc_now()
        self.error = str(error)

    def skip(self, reason: str) -> None:
        self.status = "skipped"
        self.finished_at = _utc_now()
        self.error = str(reason)

    @property
    def is_terminal(self) -> bool:
        return self.status in {"success", "failed", "skipped"}

    @property
    def succeeded(self) -> bool:
        return self.status == "success"


@dataclass(slots=True)
class EngineState:
    """
    État d'avancement du pipeline pour une paire.

    Suit la progression des phases et stocke les erreurs
    éventuelles. L'état final est :

    - success=True  si toutes les phases obligatoires
                    (dataset, contract, discovery, validation,
                     execution, portfolio) sont en status=success
                    et qu'au moins un Einher a été produit.
    - success=False sinon.
    """

    phases: dict[str, PhaseStatus] = field(default_factory=dict)

    current_phase: str = "initialization"

    success: bool = False

    started_at: datetime | None = None
    finished_at: datetime | None = None

    error: str | None = None

    # ==================================================
    # INITIALIZATION
    # ==================================================

    def __post_init__(self) -> None:
        if not self.phases:
            for name in PHASE_NAMES:
                self.phases[name] = PhaseStatus()

    # ==================================================
    # LIFECYCLE
    # ==================================================

    def start(self) -> None:
        self.started_at = _utc_now()
        self.finished_at = None
        self.success = False
        self.error = None
        for phase in self.phases.values():
            phase.status = "pending"
            phase.started_at = None
            phase.finished_at = None
            phase.error = None
            phase.metadata = {}
        self.set_phase("initialization")

    def finish(self, *, success: bool) -> None:
        self.finished_at = _utc_now()
        self.success = bool(success)
        self.set_phase("terminal")

    def fail(self, error: str) -> None:
        self.finished_at = _utc_now()
        self.success = False
        self.error = str(error)
        current = self.phases.get(self.current_phase)
        if current is not None and current.status == "running":
            current.fail(str(error))
        self.set_phase("terminal")

    # ==================================================
    # PHASE TRANSITIONS
    # ==================================================

    def set_phase(self, phase: str) -> None:
        phase = _ensure_phase_name(phase)
        self.current_phase = phase

    def begin_phase(
        self,
        phase: str,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        phase = _ensure_phase_name(phase)
        self.current_phase = phase
        status = self.phases[phase]
        status.start()
        if metadata:
            status.metadata.update(metadata)

    def complete_phase(
        self,
        phase: str,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        phase = _ensure_phase_name(phase)
        status = self.phases[phase]
        status.succeed(metadata=metadata)
        self.current_phase = phase

    def fail_phase(self, phase: str, error: str) -> None:
        phase = _ensure_phase_name(phase)
        status = self.phases[phase]
        status.fail(error)
        self.error = str(error)
        self.success = False
        self.current_phase = "terminal"

    def skip_phase(self, phase: str, reason: str) -> None:
        phase = _ensure_phase_name(phase)
        status = self.phases[phase]
        status.skip(reason)

    # ==================================================
    # ACCESSORS
    # ==================================================

    def get(self, phase: str) -> PhaseStatus:
        return self.phases[_ensure_phase_name(phase)]

    def is_phase_success(self, phase: str) -> bool:
        return self.phases[_ensure_phase_name(phase)].succeeded

    # ==================================================
    # PROPERTIES
    # ==================================================

    @property
    def duration(self):
        if self.started_at is None:
            return None
        end = (
            _utc_now()
            if self.finished_at is None
            else self.finished_at
        )
        return end - self.started_at

    @property
    def failed_phase(self) -> str | None:
        for name, status in self.phases.items():
            if status.status == "failed":
                return name
        return None

    # ==================================================
    # SERIALIZATION
    # ==================================================

    def to_dict(self) -> dict[str, Any]:
        return {
            "current_phase": self.current_phase,
            "success": self.success,
            "error": self.error,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "finished_at": self.finished_at.isoformat() if self.finished_at else None,
            "phases": {
                name: {
                    "status": status.status,
                    "started_at": status.started_at.isoformat() if status.started_at else None,
                    "finished_at": status.finished_at.isoformat() if status.finished_at else None,
                    "error": status.error,
                    "metadata": dict(status.metadata),
                }
                for name, status in self.phases.items()
            },
        }

    # ==================================================
    # PYTHON PROTOCOL
    # ==================================================

    def __repr__(self) -> str:
        return (
            "EngineState("
            f"phase='{self.current_phase}', "
            f"success={self.success}"
            ")"
        )

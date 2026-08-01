# execution/fingerprint.py
"""
==========================================================
Execution Fingerprint
==========================================================

Construit une empreinte déterministe pour une exécution
complète.

Cette empreinte n'est pas seulement un identifiant :
elle décrit aussi la version d'exécution, les paramètres
utilisés, et le lien éventuel avec la version précédente.
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from typing import Any
from typing import Mapping

from models.fingerprint import Fingerprint
from models.fingerprint import fingerprint
from models.fingerprint import fingerprint_model

__all__ = [
    "ExecutionFingerprint",
    "build_execution_fingerprint",
    "execution_fingerprint",
]


def _to_mapping(value: Any) -> dict[str, Any]:
    if value is None:
        return {}

    if hasattr(value, "to_dict") and callable(value.to_dict):
        value = value.to_dict()

    if isinstance(value, Mapping):
        return dict(value)

    return {}


def _coerce_int(value: Any, default: int = 0) -> int:
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _coerce_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _coerce_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"1", "true", "yes", "y", "on"}:
            return True
        if text in {"0", "false", "no", "n", "off"}:
            return False
    return bool(value)


@dataclass(frozen=True, slots=True)
class ExecutionFingerprint:
    """
    Carte d'identité d'une exécution.

    Le cœur de l'objet est un Fingerprint modèle, mais on
    conserve aussi les informations métier utiles à
    l'exécution et à la traçabilité.
    """

    fingerprint: Fingerprint

    subject_fingerprint: str
    execution_kind: str = "replay"
    version: int = 1
    parent_digest: str | None = None

    candidate_fingerprint: str | None = None
    hypothesis_fingerprint: str | None = None

    trade_count: int = 0
    total_pnl: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    expectancy: float = 0.0
    direction: str = "long"
    quantity: float = 1.0

    components: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "subject_fingerprint", str(self.subject_fingerprint))
        object.__setattr__(self, "execution_kind", str(self.execution_kind).strip() or "replay")
        object.__setattr__(self, "version", max(1, _coerce_int(self.version, 1)))
        object.__setattr__(self, "parent_digest", None if self.parent_digest in {"", "none", "None"} else self.parent_digest)
        object.__setattr__(self, "candidate_fingerprint", self.candidate_fingerprint)
        object.__setattr__(self, "hypothesis_fingerprint", self.hypothesis_fingerprint)
        object.__setattr__(self, "trade_count", max(0, _coerce_int(self.trade_count, 0)))
        object.__setattr__(self, "total_pnl", float(self.total_pnl))
        object.__setattr__(self, "win_rate", min(1.0, max(0.0, float(self.win_rate))))
        object.__setattr__(self, "profit_factor", float(self.profit_factor))
        object.__setattr__(self, "expectancy", float(self.expectancy))
        object.__setattr__(self, "direction", str(self.direction).strip().lower() or "long")
        object.__setattr__(self, "quantity", max(0.0, _coerce_float(self.quantity, 1.0)))
        object.__setattr__(self, "components", dict(self.components))
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def digest(self) -> str:
        return self.fingerprint.digest

    @property
    def short(self) -> str:
        return self.fingerprint.short

    @property
    def has_parent(self) -> bool:
        return self.parent_digest is not None

    def derive(
        self,
        *,
        components: Mapping[str, Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
        execution_kind: str | None = None,
        version: int | None = None,
        parent_digest: str | None = None,
    ) -> "ExecutionFingerprint":
        next_components = dict(self.components) if components is None else dict(components)
        next_metadata = dict(self.metadata) if metadata is None else dict(metadata)
        next_kind = self.execution_kind if execution_kind is None else execution_kind
        next_version = self.version + 1 if version is None else version
        next_parent = self.digest if parent_digest is None else parent_digest

        fp = Fingerprint.from_components(
            next_components,
            kind=next_kind,
            version=next_version,
            parent_digest=next_parent,
            metadata=next_metadata,
        )

        return ExecutionFingerprint(
            fingerprint=fp,
            subject_fingerprint=self.subject_fingerprint,
            execution_kind=next_kind,
            version=next_version,
            parent_digest=next_parent,
            candidate_fingerprint=self.candidate_fingerprint,
            hypothesis_fingerprint=self.hypothesis_fingerprint,
            trade_count=self.trade_count,
            total_pnl=self.total_pnl,
            win_rate=self.win_rate,
            profit_factor=self.profit_factor,
            expectancy=self.expectancy,
            direction=self.direction,
            quantity=self.quantity,
            components=next_components,
            metadata=next_metadata,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "fingerprint": self.fingerprint.to_dict(),
            "subject_fingerprint": self.subject_fingerprint,
            "execution_kind": self.execution_kind,
            "version": self.version,
            "parent_digest": self.parent_digest,
            "candidate_fingerprint": self.candidate_fingerprint,
            "hypothesis_fingerprint": self.hypothesis_fingerprint,
            "trade_count": self.trade_count,
            "total_pnl": self.total_pnl,
            "win_rate": self.win_rate,
            "profit_factor": self.profit_factor,
            "expectancy": self.expectancy,
            "direction": self.direction,
            "quantity": self.quantity,
            "components": dict(self.components),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ExecutionFingerprint":
        fp_data = data.get("fingerprint")
        if isinstance(fp_data, Mapping):
            fp = Fingerprint.from_dict(fp_data)
        else:
            fp = Fingerprint.from_components(
                data.get("components", {}),
                kind=data.get("execution_kind", "execution"),
                version=_coerce_int(data.get("version"), 1),
                parent_digest=data.get("parent_digest"),
                metadata=data.get("metadata", {}),
            )

        return cls(
            fingerprint=fp,
            subject_fingerprint=data.get("subject_fingerprint", ""),
            execution_kind=data.get("execution_kind", "replay"),
            version=_coerce_int(data.get("version"), 1),
            parent_digest=data.get("parent_digest"),
            candidate_fingerprint=data.get("candidate_fingerprint"),
            hypothesis_fingerprint=data.get("hypothesis_fingerprint"),
            trade_count=_coerce_int(data.get("trade_count"), 0),
            total_pnl=_coerce_float(data.get("total_pnl"), 0.0),
            win_rate=_coerce_float(data.get("win_rate"), 0.0),
            profit_factor=_coerce_float(data.get("profit_factor"), 0.0),
            expectancy=_coerce_float(data.get("expectancy"), 0.0),
            direction=data.get("direction", "long"),
            quantity=_coerce_float(data.get("quantity"), 1.0),
            components=_to_mapping(data.get("components", {})),
            metadata=_to_mapping(data.get("metadata", {})),
        )

    def __hash__(self) -> int:
        return hash(self.fingerprint.digest)

    def __repr__(self) -> str:
        return (
            "ExecutionFingerprint("
            f"kind='{self.execution_kind}', "
            f"version={self.version}, "
            f"digest='{self.short}', "
            f"trades={self.trade_count}"
            ")"
        )


def build_execution_fingerprint(
    *,
    subject_fingerprint: str,
    components: Mapping[str, Any],
    execution_kind: str = "replay",
    version: int = 1,
    parent_digest: str | None = None,
    metadata: Mapping[str, Any] | None = None,
    candidate_fingerprint: str | None = None,
    hypothesis_fingerprint: str | None = None,
    trade_count: int = 0,
    total_pnl: float = 0.0,
    win_rate: float = 0.0,
    profit_factor: float = 0.0,
    expectancy: float = 0.0,
    direction: str = "long",
    quantity: float = 1.0,
) -> ExecutionFingerprint:
    fp = Fingerprint.from_components(
        components,
        kind=execution_kind,
        version=version,
        parent_digest=parent_digest,
        metadata=metadata or {},
    )

    return ExecutionFingerprint(
        fingerprint=fp,
        subject_fingerprint=subject_fingerprint,
        execution_kind=execution_kind,
        version=version,
        parent_digest=parent_digest,
        candidate_fingerprint=candidate_fingerprint,
        hypothesis_fingerprint=hypothesis_fingerprint,
        trade_count=trade_count,
        total_pnl=total_pnl,
        win_rate=win_rate,
        profit_factor=profit_factor,
        expectancy=expectancy,
        direction=direction,
        quantity=quantity,
        components=dict(components),
        metadata=dict(metadata or {}),
    )


def execution_fingerprint(value: Any) -> str:
    """
    Fingerprint simple pour toute valeur ou modèle.
    """
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return fingerprint_model(value)
    return fingerprint(value)
"""
==========================================================
Fingerprint
==========================================================

Outils de calcul et de représentation des empreintes
déterministes des objets métier.

Le fingerprint n'est pas un simple identifiant technique :
il sert à décrire l'identité structurelle d'un objet, sa
version et sa filiation éventuelle.
"""

from __future__ import annotations

from dataclasses import asdict
from dataclasses import dataclass
from dataclasses import field
from datetime import date
from datetime import datetime
from decimal import Decimal
from enum import Enum
import hashlib
import json
import math
from typing import Any
from typing import Mapping


__all__ = [
    "Fingerprint",
    "fingerprint",
    "fingerprint_model",
]


def _is_json_scalar(value: Any) -> bool:
    return value is None or isinstance(value, (bool, int, str))


def _canonicalize(value: Any) -> Any:
    """
    Convertit une valeur arbitraire en structure JSON stable.
    """

    if _is_json_scalar(value):
        return value

    if isinstance(value, float):
        if math.isnan(value):
            return {"__float__": "NaN"}
        if math.isinf(value):
            return {
                "__float__": "Infinity" if value > 0 else "-Infinity"
            }
        return value

    if isinstance(value, Enum):
        return value.value

    if isinstance(value, (datetime, date)):
        return value.isoformat()

    if isinstance(value, Decimal):
        return str(value)

    if isinstance(value, bytes):
        return {
            "__bytes__": value.hex(),
        }

    if hasattr(value, "to_dict") and callable(value.to_dict):
        return _canonicalize(value.to_dict())

    if hasattr(value, "__dataclass_fields__"):
        return _canonicalize(asdict(value))

    if isinstance(value, Mapping):
        items = sorted(
            value.items(),
            key=lambda item: str(item[0]),
        )
        return {
            str(key): _canonicalize(item_value)
            for key, item_value in items
        }

    if isinstance(value, (list, tuple)):
        return [_canonicalize(item) for item in value]

    if isinstance(value, (set, frozenset)):
        normalized = [_canonicalize(item) for item in value]
        return sorted(
            normalized,
            key=lambda item: json.dumps(
                item,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                default=str,
            ),
        )

    return str(value)


def fingerprint(value: Any) -> str:
    """
    Produit un fingerprint SHA-256 déterministe.
    """

    payload = json.dumps(
        _canonicalize(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    )

    return hashlib.sha256(
        payload.encode("utf-8")
    ).hexdigest()


@dataclass(frozen=True, slots=True)
class Fingerprint:
    """
    Empreinte déterministe d'un objet métier.

    Le digest identifie une version donnée d'un objet.
    Les composants décrivent le contenu qui a servi à la
    construction de cette empreinte.
    """

    digest: str

    kind: str = "generic"

    version: int = 1

    parent_digest: str | None = None

    components: dict[str, Any] = field(default_factory=dict)

    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.digest:
            raise ValueError("digest cannot be empty.")

        if self.version < 1:
            raise ValueError("version must be >= 1.")

        object.__setattr__(
            self,
            "components",
            dict(self.components),
        )
        object.__setattr__(
            self,
            "metadata",
            dict(self.metadata),
        )

    # ==================================================
    # FACTORIES
    # ==================================================

    @classmethod
    def from_components(
        cls,
        components: Mapping[str, Any],
        *,
        kind: str = "generic",
        version: int = 1,
        parent_digest: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> "Fingerprint":
        """
        Construit une empreinte à partir de composants
        structurés.
        """

        payload = {
            "kind": kind,
            "version": version,
            "parent_digest": parent_digest,
            "components": _canonicalize(dict(components)),
            "metadata": _canonicalize(dict(metadata or {})),
        }

        return cls(
            digest=fingerprint(payload),
            kind=kind,
            version=version,
            parent_digest=parent_digest,
            components=dict(components),
            metadata=dict(metadata or {}),
        )

    @classmethod
    def from_value(
        cls,
        value: Any,
        *,
        kind: str = "generic",
        version: int = 1,
        parent_digest: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> "Fingerprint":
        """
        Construit une empreinte à partir d'une valeur unique.
        """

        return cls.from_components(
            {
                "value": value,
            },
            kind=kind,
            version=version,
            parent_digest=parent_digest,
            metadata=metadata,
        )

    @classmethod
    def from_model(
        cls,
        model: Any,
        *,
        kind: str | None = None,
        version: int = 1,
        parent_digest: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> "Fingerprint":
        """
        Construit une empreinte à partir d'un modèle possédant
        une méthode to_dict().
        """

        if not hasattr(model, "to_dict") or not callable(model.to_dict):
            raise TypeError("Model must implement to_dict().")

        model_kind = kind or model.__class__.__name__

        return cls.from_components(
            model.to_dict(),
            kind=model_kind,
            version=version,
            parent_digest=parent_digest,
            metadata=metadata,
        )

    # ==================================================
    # EVOLUTION
    # ==================================================

    def derive(
        self,
        *,
        components: Mapping[str, Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
        kind: str | None = None,
        version: int | None = None,
    ) -> "Fingerprint":
        """
        Crée une nouvelle empreinte dérivée de celle-ci.

        Utile lorsqu'un objet évolue et doit conserver un
        lien avec sa version précédente.
        """

        next_components = (
            dict(self.components)
            if components is None
            else dict(components)
        )

        next_metadata = (
            dict(self.metadata)
            if metadata is None
            else dict(metadata)
        )

        next_version = self.version + 1 if version is None else version
        next_kind = self.kind if kind is None else kind

        return Fingerprint.from_components(
            next_components,
            kind=next_kind,
            version=next_version,
            parent_digest=self.digest,
            metadata=next_metadata,
        )

    # ==================================================
    # SERIALIZATION
    # ==================================================

    def to_dict(self) -> dict[str, Any]:
        return {
            "digest": self.digest,
            "kind": self.kind,
            "version": self.version,
            "parent_digest": self.parent_digest,
            "components": dict(self.components),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(
        cls,
        data: Mapping[str, Any],
    ) -> "Fingerprint":
        return cls(
            digest=data["digest"],
            kind=data.get("kind", "generic"),
            version=data.get("version", 1),
            parent_digest=data.get("parent_digest"),
            components=dict(data.get("components", {})),
            metadata=dict(data.get("metadata", {})),
        )

    # ==================================================
    # PROPERTIES
    # ==================================================

    @property
    def short(self) -> str:
        return self.digest[:12]

    @property
    def has_parent(self) -> bool:
        return self.parent_digest is not None

    # ==================================================
    # PYTHON PROTOCOL
    # ==================================================

    def __hash__(self) -> int:
        return hash(self.digest)

    def __str__(self) -> str:
        return self.digest

    def __repr__(self) -> str:
        parent = (
            self.parent_digest[:12]
            if self.parent_digest
            else None
        )

        return (
            "Fingerprint("
            f"kind='{self.kind}', "
            f"version={self.version}, "
            f"digest='{self.short}', "
            f"parent='{parent}'"
            ")"
        )


def fingerprint_model(model: Any) -> str:
    """
    Produit le fingerprint d'un modèle possédant une méthode
    to_dict().
    """

    if not hasattr(model, "to_dict") or not callable(model.to_dict):
        raise TypeError("Model must implement to_dict().")

    return fingerprint(model.to_dict())
"""
==========================================================
Profile Model
==========================================================

Un Profile décrit les caractéristiques d'un Einher.

Il s'agit d'un simple objet de données utilisé par le
moteur pour identifier et décrire un Einher.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class Profile:
    """
    Profil descriptif d'un Einher.
    """

    name: str

    description: str = ""

    metadata: dict[str, Any] | None = None

    # ==================================================
    # VALIDATION
    # ==================================================

    def __post_init__(self) -> None:

        if not self.name:
            raise ValueError(
                "Profile name cannot be empty."
            )

        object.__setattr__(
            self,
            "metadata",
            {} if self.metadata is None else dict(self.metadata),
        )

    # ==================================================
    # SERIALIZATION
    # ==================================================

    def to_dict(self) -> dict[str, Any]:

        return {
            "name": self.name,
            "description": self.description,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
    ) -> "Profile":

        return cls(
            name=data["name"],
            description=data.get(
                "description",
                "",
            ),
            metadata=data.get(
                "metadata",
                {},
            ),
        )

    # ==================================================
    # PYTHON PROTOCOL
    # ==================================================

    def __hash__(self) -> int:
        return hash(
            (
                self.name,
                self.description,
            )
        )

    def __repr__(self) -> str:
        return (
            "Profile("
            f"name='{self.name}'"
            ")"
        )
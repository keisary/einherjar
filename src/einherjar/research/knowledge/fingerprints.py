# knowledge/fingerprints.py
"""
==========================================================
Knowledge Fingerprints
==========================================================

Identité canonique des objets du module knowledge.

Le but n'est pas de réinventer le fingerprint métier déjà
existant dans models.fingerprint, mais de fournir une couche
relationnelle stable pour :
- les Einhers,
- les familles,
- les profils,
- les relations,
- les concepts de knowledge.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterable, Mapping, Sequence

try:
    from models.fingerprint import Fingerprint  # type: ignore
    from models.fingerprint import fingerprint as _fingerprint_text  # type: ignore
    from models.fingerprint import fingerprint_model as _fingerprint_model  # type: ignore
except Exception:  # pragma: no cover
    Fingerprint = Any  # type: ignore[misc,assignment]
    _fingerprint_text = None
    _fingerprint_model = None

__all__ = [
    "KnowledgeFingerprint",
    "FingerprintRegistry",
    "build_knowledge_fingerprint",
    "fingerprint_object",
    "fingerprint_many",
]


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _to_mapping(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if hasattr(value, "to_dict") and callable(value.to_dict):
        value = value.to_dict()
    if isinstance(value, Mapping):
        return dict(value)
    return {}


def _normalize_text(value: Any) -> str:
    return str(value or "").strip()


def _flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    value = _to_mapping(value)

    if isinstance(value, Mapping):
        output: dict[str, Any] = {}
        for key, item in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            output.update(_flatten(item, path))
        return output

    if isinstance(value, (list, tuple, set)):
        output: dict[str, Any] = {}
        for index, item in enumerate(value):
            path = f"{prefix}[{index}]"
            output.update(_flatten(item, path))
        return output

    return {prefix: value}


def _stable_digest(payload: Any) -> str:
    if _fingerprint_text is not None:
        try:
            return str(_fingerprint_text(payload))
        except Exception:
            pass

    if hasattr(payload, "digest"):
        digest = getattr(payload, "digest", None)
        if digest:
            return str(digest)

    return repr(payload)


def _fingerprint_components(components: Mapping[str, Any], *, kind: str, version: int = 1, parent_digest: str | None = None, metadata: Mapping[str, Any] | None = None) -> Any:
    components = dict(sorted(_flatten(components).items()))
    metadata = dict(sorted(_flatten(metadata or {}).items()))
    payload = {
        "kind": kind,
        "version": int(version),
        "parent_digest": parent_digest,
        "components": components,
        "metadata": metadata,
    }

    if hasattr(Fingerprint, "from_components"):
        try:
            return Fingerprint.from_components(  # type: ignore[attr-defined]
                payload,
                kind=kind,
                version=version,
                parent_digest=parent_digest,
                metadata=metadata,
            )
        except Exception:
            pass

    return _stable_digest(payload)


def fingerprint_object(obj: Any, *, kind: str = "knowledge", version: int = 1, parent_digest: str | None = None, metadata: Mapping[str, Any] | None = None) -> str:
    if _fingerprint_model is not None:
        try:
            return str(_fingerprint_model(obj))
        except Exception:
            pass

    if hasattr(obj, "fingerprint"):
        value = getattr(obj, "fingerprint")
        if value:
            return str(value)

    if hasattr(obj, "digest"):
        value = getattr(obj, "digest")
        if value:
            return str(value)

    payload = {
        "kind": kind,
        "version": version,
        "parent_digest": parent_digest,
        "object": _flatten(obj),
        "metadata": _flatten(metadata or {}),
    }
    return _stable_digest(payload)


def fingerprint_many(values: Iterable[Any], *, kind: str = "knowledge", metadata: Mapping[str, Any] | None = None) -> tuple[str, ...]:
    return tuple(fingerprint_object(value, kind=kind, metadata=metadata) for value in values)


@dataclass(frozen=True, slots=True)
class KnowledgeFingerprint:
    """
    Empreinte relationnelle d'un objet du corpus.
    """

    digest: str
    kind: str = "knowledge"
    label: str = ""
    version: int = 1
    parent_digest: str | None = None
    tags: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)
    components: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=_utc_now)

    def __post_init__(self) -> None:
        object.__setattr__(self, "digest", _normalize_text(self.digest))
        object.__setattr__(self, "kind", _normalize_text(self.kind) or "knowledge")
        object.__setattr__(self, "label", _normalize_text(self.label))
        object.__setattr__(self, "version", max(1, int(self.version)))
        object.__setattr__(self, "parent_digest", _normalize_text(self.parent_digest) or None)
        object.__setattr__(self, "tags", tuple(sorted({str(tag).strip().lower() for tag in self.tags if str(tag).strip()})))
        object.__setattr__(self, "metadata", dict(self.metadata))
        object.__setattr__(self, "components", dict(self.components))

    @property
    def short(self) -> str:
        return self.digest[:12]

    @property
    def has_parent(self) -> bool:
        return self.parent_digest is not None

    def to_dict(self) -> dict[str, Any]:
        return {
            "digest": self.digest,
            "kind": self.kind,
            "label": self.label,
            "version": self.version,
            "parent_digest": self.parent_digest,
            "tags": list(self.tags),
            "metadata": dict(self.metadata),
            "components": dict(self.components),
            "created_at": self.created_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "KnowledgeFingerprint":
        created_at = data.get("created_at")
        if isinstance(created_at, str) and created_at:
            created_at = datetime.fromisoformat(created_at)
        else:
            created_at = _utc_now()

        return cls(
            digest=data.get("digest", ""),
            kind=data.get("kind", "knowledge"),
            label=data.get("label", ""),
            version=int(data.get("version", 1)),
            parent_digest=data.get("parent_digest"),
            tags=tuple(data.get("tags", ())),
            metadata=_to_mapping(data.get("metadata", {})),
            components=_to_mapping(data.get("components", {})),
            created_at=created_at,
        )

    @classmethod
    def from_object(
        cls,
        obj: Any,
        *,
        kind: str = "knowledge",
        label: str = "",
        version: int = 1,
        parent_digest: str | None = None,
        tags: Sequence[str] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> "KnowledgeFingerprint":
        digest = fingerprint_object(obj, kind=kind, version=version, parent_digest=parent_digest, metadata=metadata)
        components = _flatten(obj)
        if not label:
            label = _normalize_text(getattr(obj, "name", None) or getattr(obj, "label", None) or getattr(obj, "profile_name", None) or getattr(obj, "family", None))
        if tags is None:
            tags = tuple()
        return cls(
            digest=digest,
            kind=kind,
            label=label,
            version=version,
            parent_digest=parent_digest,
            tags=tuple(tags),
            metadata=_to_mapping(metadata or {}),
            components=components,
        )

    def __repr__(self) -> str:
        return f"KnowledgeFingerprint(kind={self.kind!r}, digest={self.short!r})"


class FingerprintRegistry:
    """
    Registre en mémoire des empreintes relationnelles.
    """

    def __init__(self) -> None:
        self._by_digest: dict[str, KnowledgeFingerprint] = {}
        self._by_kind: dict[str, list[str]] = {}
        self._by_label: dict[str, list[str]] = {}

    def add(self, fingerprint: KnowledgeFingerprint) -> KnowledgeFingerprint:
        self._by_digest[fingerprint.digest] = fingerprint
        self._by_kind.setdefault(fingerprint.kind, []).append(fingerprint.digest)
        if fingerprint.label:
            self._by_label.setdefault(fingerprint.label, []).append(fingerprint.digest)
        return fingerprint

    def build(
        self,
        obj: Any,
        *,
        kind: str = "knowledge",
        label: str = "",
        version: int = 1,
        parent_digest: str | None = None,
        tags: Sequence[str] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> KnowledgeFingerprint:
        fp = KnowledgeFingerprint.from_object(
            obj,
            kind=kind,
            label=label,
            version=version,
            parent_digest=parent_digest,
            tags=tags,
            metadata=metadata,
        )
        return self.add(fp)

    def get(self, digest: str) -> KnowledgeFingerprint | None:
        return self._by_digest.get(str(digest))

    def by_kind(self, kind: str) -> tuple[KnowledgeFingerprint, ...]:
        digests = self._by_kind.get(str(kind), [])
        return tuple(self._by_digest[d] for d in digests if d in self._by_digest)

    def by_label(self, label: str) -> tuple[KnowledgeFingerprint, ...]:
        digests = self._by_label.get(str(label), [])
        return tuple(self._by_digest[d] for d in digests if d in self._by_digest)

    def all(self) -> tuple[KnowledgeFingerprint, ...]:
        return tuple(self._by_digest.values())

    def to_dict(self) -> dict[str, Any]:
        return {
            "fingerprints": [fp.to_dict() for fp in self._by_digest.values()],
            "by_kind": {key: list(values) for key, values in self._by_kind.items()},
            "by_label": {key: list(values) for key, values in self._by_label.items()},
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "FingerprintRegistry":
        registry = cls()
        for item in data.get("fingerprints", []):
            registry.add(KnowledgeFingerprint.from_dict(item))
        return registry

    def __len__(self) -> int:
        return len(self._by_digest)

    def __iter__(self):
        return iter(self._by_digest.values())

    def __repr__(self) -> str:
        return f"FingerprintRegistry(size={len(self)})"


def build_knowledge_fingerprint(
    obj: Any,
    *,
    kind: str = "knowledge",
    label: str = "",
    version: int = 1,
    parent_digest: str | None = None,
    tags: Sequence[str] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> KnowledgeFingerprint:
    return KnowledgeFingerprint.from_object(
        obj,
        kind=kind,
        label=label,
        version=version,
        parent_digest=parent_digest,
        tags=tags,
        metadata=metadata,
    )
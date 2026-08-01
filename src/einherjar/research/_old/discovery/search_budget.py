"""
==========================================================
Search Budget
==========================================================

Contrôle les ressources de la phase Discovery.

Le budget n'est pas financier : il représente la quantité
d'espace de recherche que le moteur peut encore explorer
avant de devoir s'arrêter, ralentir ou se réorienter.

Ce module ne génère rien et ne valide rien.
Il surveille seulement la consommation de recherche.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from time import monotonic
from typing import Any
from typing import Iterable
from typing import Mapping
from typing import Sequence

from config.search import SearchConfig
from models.condition import Condition
from models.enums import EconomicFamily
from models.feature import Feature
from models.hypothesis import Hypothesis


__all__ = [
    "BudgetSnapshot",
    "SearchBudget",
]


# ==========================================================
# HELPERS
# ==========================================================

def _resolve_path(obj: Any, path: Sequence[str]) -> Any | None:
    current = obj

    for part in path:
        if current is None:
            return None

        if isinstance(current, Mapping):
            if part not in current:
                return None
            current = current[part]
            continue

        if not hasattr(current, part):
            return None

        current = getattr(current, part)

    return current


def _first_non_none(
    obj: Any,
    *paths: Sequence[str],
    default: Any = None,
) -> Any:
    for path in paths:
        value = _resolve_path(obj, path)
        if value is not None:
            return value
    return default


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


def _coerce_family(value: EconomicFamily | str | None) -> EconomicFamily | None:
    if value is None:
        return None

    if isinstance(value, EconomicFamily):
        return value

    text = str(value).strip()

    try:
        return EconomicFamily(text)
    except ValueError:
        try:
            return EconomicFamily[text.upper()]
        except KeyError as exc:
            raise ValueError(f"Unknown family: {value}") from exc


def _condition_family_key(condition: Condition) -> str:
    return condition.left.economic_family.value


def _unique_families(hypothesis: Hypothesis) -> tuple[EconomicFamily, ...]:
    seen: set[EconomicFamily] = set()
    ordered: list[EconomicFamily] = []

    for condition in hypothesis.conditions:
        family = condition.left.economic_family
        if family in seen:
            continue
        seen.add(family)
        ordered.append(family)

    return tuple(ordered)


def _condition_count_by_family(hypothesis: Hypothesis) -> dict[EconomicFamily, int]:
    counts: dict[EconomicFamily, int] = {}

    for condition in hypothesis.conditions:
        family = condition.left.economic_family
        counts[family] = counts.get(family, 0) + 1

    return counts


def _default_total_budget(search_config: SearchConfig) -> int:
    beam_width = max(1, _coerce_int(search_config.beam_width, 1))
    max_depth = max(1, _coerce_int(search_config.max_depth, 1))
    return beam_width * max_depth


def _normalize_budget_mapping(value: Any) -> dict[str, int]:
    mapping = _to_mapping(value)
    normalized: dict[str, int] = {}

    for key, item in mapping.items():
        normalized[str(key).strip().lower()] = max(0, _coerce_int(item, 0))

    return normalized


def _build_search_config(source: Any | None) -> SearchConfig:
    if isinstance(source, SearchConfig):
        return source

    if source is None:
        return SearchConfig()

    return SearchConfig(
        max_conditions=_coerce_int(
            _first_non_none(source, ("max_conditions",), ("search", "max_conditions"), default=3),
            3,
        ),
        beam_width=_coerce_int(
            _first_non_none(source, ("beam_width",), ("search", "beam_width"), default=200),
            200,
        ),
        max_depth=_coerce_int(
            _first_non_none(source, ("max_depth",), ("search", "max_depth"), default=3),
            3,
        ),
        max_candidates_per_family=_coerce_int(
            _first_non_none(
                source,
                ("max_candidates_per_family",),
                ("search", "max_candidates_per_family"),
                default=100,
            ),
            100,
        ),
        exploration_ratio=_coerce_float(
            _first_non_none(source, ("exploration_ratio",), ("search", "exploration_ratio"), default=0.25),
            0.25,
        ),
        exploitation_ratio=_coerce_float(
            _first_non_none(source, ("exploitation_ratio",), ("search", "exploitation_ratio"), default=0.75),
            0.75,
        ),
        novelty_weight=_coerce_float(
            _first_non_none(source, ("novelty_weight",), ("search", "novelty_weight"), default=0.30),
            0.30,
        ),
        diversity_weight=_coerce_float(
            _first_non_none(source, ("diversity_weight",), ("search", "diversity_weight"), default=0.25),
            0.25,
        ),
        family_balance_weight=_coerce_float(
            _first_non_none(source, ("family_balance_weight",), ("search", "family_balance_weight"), default=0.20),
            0.20,
        ),
        random_seed=_coerce_int(
            _first_non_none(source, ("random_seed",), ("search", "random_seed"), default=42),
            42,
        ),
    )


# ==========================================================
# SNAPSHOT
# ==========================================================

@dataclass(frozen=True, slots=True)
class BudgetSnapshot:
    """
    Photographie immuable de l'état du budget.
    """

    configured_total: int
    configured_frontier: int
    configured_family: int
    configured_depth: int
    configured_conditions: int

    generated_total: int
    active_total: int

    elapsed_seconds: float
    remaining_total: int | None
    remaining_frontier: int | None
    exhausted: bool
    reason: str | None = None

    family_generated: dict[str, int] = field(default_factory=dict)
    depth_generated: dict[int, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "family_generated", dict(self.family_generated))
        object.__setattr__(self, "depth_generated", dict(self.depth_generated))


# ==========================================================
# SEARCH BUDGET
# ==========================================================

class SearchBudget:
    """
    Contrôle la consommation de ressources pendant la phase
    Discovery.

    Le budget suit :
    - le nombre total de nœuds produits,
    - la taille de la frontière active,
    - le nombre de nœuds par famille,
    - la profondeur maximale,
    - la taille maximale des hypothèses,
    - le temps écoulé.
    """

    def __init__(
        self,
        search_config: SearchConfig | None = None,
        *,
        config: Any | None = None,
        max_runtime_seconds: float | None = None,
        max_total_candidates: int | None = None,
        max_frontier_size: int | None = None,
        max_candidates_per_family: int | None = None,
        max_depth: int | None = None,
        max_conditions: int | None = None,
    ) -> None:
        if search_config is None:
            search_source = _first_non_none(
                config,
                ("search",),
                ("search_config",),
                ("discovery", "search"),
                default=None,
            )
            self._search_config = _build_search_config(search_source)
        else:
            self._search_config = search_config

        self._max_runtime_seconds = (
            None
            if max_runtime_seconds is None
            else max(0.0, _coerce_float(max_runtime_seconds, 0.0))
        )

        self._max_total_candidates = (
            _coerce_int(max_total_candidates, 0)
            if max_total_candidates is not None
            else _default_total_budget(self._search_config)
        )
        if self._max_total_candidates <= 0:
            self._max_total_candidates = _default_total_budget(self._search_config)

        self._max_frontier_size = (
            _coerce_int(max_frontier_size, 0)
            if max_frontier_size is not None
            else max(1, _coerce_int(self._search_config.beam_width, 1))
        )
        if self._max_frontier_size <= 0:
            self._max_frontier_size = max(1, _coerce_int(self._search_config.beam_width, 1))

        self._max_candidates_per_family = (
            _coerce_int(max_candidates_per_family, 0)
            if max_candidates_per_family is not None
            else max(1, _coerce_int(self._search_config.max_candidates_per_family, 1))
        )
        if self._max_candidates_per_family <= 0:
            self._max_candidates_per_family = max(1, _coerce_int(self._search_config.max_candidates_per_family, 1))

        self._max_depth = (
            _coerce_int(max_depth, 0)
            if max_depth is not None
            else max(1, _coerce_int(self._search_config.max_depth, 1))
        )
        if self._max_depth <= 0:
            self._max_depth = max(1, _coerce_int(self._search_config.max_depth, 1))

        self._max_conditions = (
            _coerce_int(max_conditions, 0)
            if max_conditions is not None
            else max(1, _coerce_int(self._search_config.max_conditions, 1))
        )
        if self._max_conditions <= 0:
            self._max_conditions = max(1, _coerce_int(self._search_config.max_conditions, 1))

        self._started_at: float | None = None
        self._stopped_at: float | None = None

        self._generated_total = 0
        self._active_total = 0

        self._family_generated: Counter[str] = Counter()
        self._family_active: Counter[str] = Counter()
        self._depth_generated: Counter[int] = Counter()

        self._exhausted = False
        self._reason: str | None = None

    # ==================================================
    # CONSTRUCTION FROM CONFIG
    # ==================================================

    @classmethod
    def from_config(
        cls,
        config: Any,
        *,
        search_config: SearchConfig | None = None,
        max_runtime_seconds: float | None = None,
        max_total_candidates: int | None = None,
        max_frontier_size: int | None = None,
        max_candidates_per_family: int | None = None,
        max_depth: int | None = None,
        max_conditions: int | None = None,
    ) -> "SearchBudget":
        if search_config is None:
            search_source = _first_non_none(
                config,
                ("search",),
                ("search_config",),
                ("discovery", "search"),
                default=None,
            )
            search_config = _build_search_config(search_source)

        if max_runtime_seconds is None:
            max_runtime_seconds = _first_non_none(
                config,
                ("max_runtime_seconds",),
                ("search", "max_runtime_seconds"),
                ("search", "time_limit_seconds"),
                ("discovery", "max_runtime_seconds"),
                ("discovery", "time_limit_seconds"),
                default=None,
            )

        if max_total_candidates is None:
            max_total_candidates = _first_non_none(
                config,
                ("max_total_candidates",),
                ("search", "max_total_candidates"),
                ("search", "total_candidates"),
                ("discovery", "max_total_candidates"),
                default=None,
            )

        if max_frontier_size is None:
            max_frontier_size = _first_non_none(
                config,
                ("max_frontier_size",),
                ("search", "max_frontier_size"),
                ("search", "beam_width"),
                ("discovery", "max_frontier_size"),
                default=None,
            )

        if max_candidates_per_family is None:
            max_candidates_per_family = _first_non_none(
                config,
                ("max_candidates_per_family",),
                ("search", "max_candidates_per_family"),
                ("discovery", "max_candidates_per_family"),
                default=None,
            )

        if max_depth is None:
            max_depth = _first_non_none(
                config,
                ("max_depth",),
                ("search", "max_depth"),
                ("discovery", "max_depth"),
                default=None,
            )

        if max_conditions is None:
            max_conditions = _first_non_none(
                config,
                ("max_conditions",),
                ("search", "max_conditions"),
                ("discovery", "max_conditions"),
                default=None,
            )

        return cls(
            search_config=search_config,
            max_runtime_seconds=max_runtime_seconds,
            max_total_candidates=max_total_candidates,
            max_frontier_size=max_frontier_size,
            max_candidates_per_family=max_candidates_per_family,
            max_depth=max_depth,
            max_conditions=max_conditions,
        )

    # ==================================================
    # LIFECYCLE
    # ==================================================

    def start(self) -> None:
        if self._started_at is None:
            self._started_at = monotonic()
            self._stopped_at = None
            self._exhausted = False
            self._reason = None

    def stop(self, reason: str | None = None) -> None:
        self._stopped_at = monotonic()
        self._exhausted = True
        self._reason = reason or self._reason or "stopped_by_user"

    def reset(self) -> None:
        self._started_at = None
        self._stopped_at = None

        self._generated_total = 0
        self._active_total = 0

        self._family_generated.clear()
        self._family_active.clear()
        self._depth_generated.clear()

        self._exhausted = False
        self._reason = None

    def exhaust(self, reason: str | None = None) -> None:
        self._exhausted = True
        self._reason = reason or self._reason or "budget_exhausted"

    # ==================================================
    # PROPERTIES
    # ==================================================

    @property
    def search_config(self) -> SearchConfig:
        return self._search_config

    @property
    def max_runtime_seconds(self) -> float | None:
        return self._max_runtime_seconds

    @property
    def max_total_candidates(self) -> int:
        return self._max_total_candidates

    @property
    def max_frontier_size(self) -> int:
        return self._max_frontier_size

    @property
    def max_candidates_per_family(self) -> int:
        return self._max_candidates_per_family

    @property
    def max_depth(self) -> int:
        return self._max_depth

    @property
    def max_conditions(self) -> int:
        return self._max_conditions

    @property
    def generated_total(self) -> int:
        return self._generated_total

    @property
    def active_total(self) -> int:
        return self._active_total

    @property
    def exhausted(self) -> bool:
        return self._exhausted or self.is_time_exhausted or self.is_total_exhausted or self.is_frontier_exhausted

    @property
    def reason(self) -> str | None:
        return self._reason

    @property
    def is_started(self) -> bool:
        return self._started_at is not None

    @property
    def elapsed_seconds(self) -> float:
        if self._started_at is None:
            return 0.0

        end = self._stopped_at if self._stopped_at is not None else monotonic()
        return max(0.0, end - self._started_at)

    @property
    def is_time_exhausted(self) -> bool:
        if self._max_runtime_seconds is None:
            return False
        return self.elapsed_seconds >= self._max_runtime_seconds

    @property
    def is_total_exhausted(self) -> bool:
        return self._generated_total >= self._max_total_candidates

    @property
    def is_frontier_exhausted(self) -> bool:
        return self._active_total >= self._max_frontier_size

    @property
    def family_generated(self) -> dict[str, int]:
        return dict(self._family_generated)

    @property
    def family_active(self) -> dict[str, int]:
        return dict(self._family_active)

    @property
    def depth_generated(self) -> dict[int, int]:
        return dict(self._depth_generated)

    @property
    def remaining_total(self) -> int:
        return max(0, self._max_total_candidates - self._generated_total)

    @property
    def remaining_frontier(self) -> int:
        return max(0, self._max_frontier_size - self._active_total)

    @property
    def remaining_seconds(self) -> float | None:
        if self._max_runtime_seconds is None:
            return None
        return max(0.0, self._max_runtime_seconds - self.elapsed_seconds)

    @property
    def snapshot(self) -> BudgetSnapshot:
        return BudgetSnapshot(
            configured_total=self._max_total_candidates,
            configured_frontier=self._max_frontier_size,
            configured_family=self._max_candidates_per_family,
            configured_depth=self._max_depth,
            configured_conditions=self._max_conditions,
            generated_total=self._generated_total,
            active_total=self._active_total,
            elapsed_seconds=self.elapsed_seconds,
            remaining_total=self.remaining_total,
            remaining_frontier=self.remaining_frontier,
            exhausted=self.exhausted,
            reason=self._reason,
            family_generated=dict(self._family_generated),
            depth_generated=dict(self._depth_generated),
        )

    # ==================================================
    # LIMIT QUERIES
    # ==================================================

    def family_remaining(
        self,
        family: EconomicFamily | str,
    ) -> int:
        family_enum = _coerce_family(family)
        if family_enum is None:
            raise ValueError("family cannot be None.")

        used = self._family_generated.get(family_enum.value, 0)
        return max(0, self._max_candidates_per_family - used)

    def family_active_remaining(
        self,
        family: EconomicFamily | str,
    ) -> int:
        family_enum = _coerce_family(family)
        if family_enum is None:
            raise ValueError("family cannot be None.")

        used = self._family_active.get(family_enum.value, 0)
        return max(0, self._max_candidates_per_family - used)

    def depth_remaining(self, depth: int) -> int:
        depth = _coerce_int(depth, 0)
        if depth >= self._max_depth:
            return 0
        return self._max_depth - depth

    def condition_remaining(self, condition_count: int) -> int:
        condition_count = _coerce_int(condition_count, 0)
        return max(0, self._max_conditions - condition_count)

    def has_family_capacity(
        self,
        family: EconomicFamily | str,
        amount: int = 1,
    ) -> bool:
        return self.family_remaining(family) >= max(1, _coerce_int(amount, 1))

    def has_depth_capacity(self, depth: int) -> bool:
        return depth < self._max_depth

    def has_condition_capacity(self, condition_count: int) -> bool:
        return condition_count < self._max_conditions

    # ==================================================
    # ADMISSION
    # ==================================================

    def can_generate(
        self,
        *,
        family: EconomicFamily | str | None = None,
        depth: int = 0,
        condition_count: int | None = None,
        amount: int = 1,
    ) -> bool:
        amount = max(1, _coerce_int(amount, 1))
        depth = _coerce_int(depth, 0)

        if self.exhausted:
            return False

        if self._started_at is None:
            self.start()

        if self.is_time_exhausted or self.is_total_exhausted or self.is_frontier_exhausted:
            return False

        if depth >= self._max_depth:
            return False

        if condition_count is not None and condition_count >= self._max_conditions:
            return False

        if self._generated_total + amount > self._max_total_candidates:
            return False

        if self._active_total + amount > self._max_frontier_size:
            return False

        if family is not None:
            family_enum = _coerce_family(family)
            if family_enum is None:
                return False
            if self._family_generated.get(family_enum.value, 0) + amount > self._max_candidates_per_family:
                return False

        return True

    def consume(
        self,
        *,
        family: EconomicFamily | str | None = None,
        depth: int = 0,
        active: bool = True,
        amount: int = 1,
    ) -> None:
        amount = max(1, _coerce_int(amount, 1))
        depth = _coerce_int(depth, 0)

        if not self.can_generate(
            family=family,
            depth=depth,
            amount=amount,
        ):
            raise RuntimeError("Search budget exhausted or capacity exceeded.")

        self.start()

        self._generated_total += amount
        self._depth_generated[depth] += amount

        if family is not None:
            family_enum = _coerce_family(family)
            if family_enum is not None:
                key = family_enum.value
                self._family_generated[key] += amount
                if active:
                    self._family_active[key] += amount

        if active:
            self._active_total += amount

    def release(
        self,
        *,
        family: EconomicFamily | str | None = None,
        amount: int = 1,
    ) -> None:
        amount = max(1, _coerce_int(amount, 1))

        self._active_total = max(0, self._active_total - amount)

        if family is not None:
            family_enum = _coerce_family(family)
            if family_enum is not None:
                key = family_enum.value
                self._family_active[key] = max(0, self._family_active.get(key, 0) - amount)

    def admit_hypothesis(
        self,
        hypothesis: Hypothesis,
        *,
        family: EconomicFamily | str | None = None,
        depth: int = 0,
        active: bool = True,
    ) -> None:
        if not isinstance(hypothesis, Hypothesis):
            raise TypeError("hypothesis must be a Hypothesis.")

        selected_family = _coerce_family(family)

        if selected_family is None:
            families = _unique_families(hypothesis)
            if families:
                for item in families:
                    self.consume(family=item, depth=depth, active=active, amount=1)
                return

        self.consume(
            family=selected_family,
            depth=depth,
            active=active,
            amount=1,
        )

    def admit_condition(
        self,
        condition: Condition,
        *,
        depth: int = 0,
        active: bool = True,
    ) -> None:
        if not isinstance(condition, Condition):
            raise TypeError("condition must be a Condition.")

        self.consume(
            family=condition.left.economic_family,
            depth=depth,
            active=active,
            amount=1,
        )

    # ==================================================
    # EXHAUSTION HELPERS
    # ==================================================

    def stop_if_exhausted(self) -> bool:
        if self.exhausted:
            self.exhaust(self._reason or "budget_exhausted")
            return True
        return False

    def update_reason(self, reason: str | None) -> None:
        self._reason = reason

    # ==================================================
    # PYTHON PROTOCOL
    # ==================================================

    def __bool__(self) -> bool:
        return not self.exhausted

    def __len__(self) -> int:
        return self._generated_total

    def __repr__(self) -> str:
        return (
            "SearchBudget("
            f"generated={self._generated_total}, "
            f"active={self._active_total}, "
            f"remaining={self.remaining_total}, "
            f"exhausted={self.exhausted}"
            ")"
        )
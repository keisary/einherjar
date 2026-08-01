"""
Parallel execution configuration.
"""

from dataclasses import dataclass
import os
from typing import Any


@dataclass(slots=True, frozen=True)
class ParallelConfig:

    workers: int = max(1, os.cpu_count() - 1)

    chunk_size: int = 256

    use_process_pool: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "workers": self.workers,
            "chunk_size": self.chunk_size,
            "use_process_pool": self.use_process_pool,
        }

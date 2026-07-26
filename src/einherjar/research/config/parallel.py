"""
Parallel execution configuration.
"""

from dataclasses import dataclass
import os


@dataclass(slots=True, frozen=True)
class ParallelConfig:

    workers: int = max(1, os.cpu_count() - 1)

    chunk_size: int = 256

    use_process_pool: bool = True
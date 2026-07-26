"""
Logging configuration.
"""

from dataclasses import dataclass
from logging import INFO


@dataclass(slots=True, frozen=True)
class LoggingConfig:

    level: int = INFO

    console: bool = True

    file: bool = True

    rotate_mb: int = 25

    backups: int = 10
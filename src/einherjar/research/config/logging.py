"""
Logging configuration.
"""

from dataclasses import dataclass
from logging import INFO
from typing import Any


@dataclass(slots=True, frozen=True)
class LoggingConfig:

    level: int = INFO

    console: bool = True

    file: bool = True

    rotate_mb: int = 25

    backups: int = 10

    def to_dict(self) -> dict[str, Any]:
        return {
            "level": self.level,
            "console": self.console,
            "file": self.file,
            "rotate_mb": self.rotate_mb,
            "backups": self.backups,
        }

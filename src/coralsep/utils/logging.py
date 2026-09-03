"""
Structured logging helper for CoRAL-Sep (BLUEPRINT §13).

Provides a simple JSON-friendly logger with consistent fields for seeds,
config hashes, and run metadata across data prep, training, and evaluation.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class LogContext:
    """Structured fields attached to every log record."""

    component: str = "coralsep"
    run_id: str | None = None
    seed: int | None = None
    config_hash: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


class StructuredLogger:
    """
    Thin wrapper around stdlib logging with JSON line output.

    Args:
        name: Logger name (typically module or script name).
        level: Logging level string (DEBUG, INFO, ...).
        json_lines: When True, emit one JSON object per log line.
    """

    def __init__(
        self,
        name: str,
        level: str = "INFO",
        *,
        json_lines: bool = False,
    ) -> None:
        self._logger = logging.getLogger(name)
        self._logger.setLevel(getattr(logging, level.upper(), logging.INFO))
        self._json_lines = json_lines
        self.context = LogContext(component=name)

        if not self._logger.handlers:
            handler = logging.StreamHandler(sys.stderr)
            handler.setFormatter(logging.Formatter("%(message)s"))
            self._logger.addHandler(handler)

    def _emit(self, level: str, event: str, **fields: Any) -> None:
        payload: dict[str, Any] = {
            "ts": time.time(),
            "level": level,
            "event": event,
            "component": self.context.component,
        }
        if self.context.run_id is not None:
            payload["run_id"] = self.context.run_id
        if self.context.seed is not None:
            payload["seed"] = self.context.seed
        if self.context.config_hash is not None:
            payload["config_hash"] = self.context.config_hash
        payload.update(self.context.extra)
        payload.update(fields)

        msg = (
            json.dumps(payload, sort_keys=True, default=str)
            if self._json_lines
            else self._format_plain(payload)
        )
        getattr(self._logger, level.lower())(msg)

    @staticmethod
    def _format_plain(payload: dict[str, Any]) -> str:
        event = payload.pop("event", "log")
        level = payload.pop("level", "INFO")
        payload.pop("ts", None)  # plain format omits the timestamp
        rest = " ".join(f"{k}={v!r}" for k, v in payload.items())
        return f"[{level}] {event}" + (f" ({rest})" if rest else "")

    def debug(self, event: str, **fields: Any) -> None:
        self._emit("DEBUG", event, **fields)

    def info(self, event: str, **fields: Any) -> None:
        self._emit("INFO", event, **fields)

    def warning(self, event: str, **fields: Any) -> None:
        self._emit("WARNING", event, **fields)

    def error(self, event: str, **fields: Any) -> None:
        self._emit("ERROR", event, **fields)

    def bind(self, **kwargs: Any) -> StructuredLogger:
        """Return self after updating context fields."""
        for key, val in kwargs.items():
            if hasattr(self.context, key):
                setattr(self.context, key, val)
            else:
                self.context.extra[key] = val
        return self

    def context_dict(self) -> dict[str, Any]:
        """Serialize current context."""
        return asdict(self.context)


def get_logger(
    name: str,
    level: str = "INFO",
    *,
    json_lines: bool = False,
) -> StructuredLogger:
    """Factory for StructuredLogger instances."""
    return StructuredLogger(name, level, json_lines=json_lines)


def configure_file_log(path: str | Path, logger: StructuredLogger) -> None:
    """Add a plain-text file handler to an existing StructuredLogger."""
    fh = logging.FileHandler(Path(path), encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(message)s"))
    logger._logger.addHandler(fh)

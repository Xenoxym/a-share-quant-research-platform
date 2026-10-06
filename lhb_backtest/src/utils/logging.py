"""Logging utilities based on loguru."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import yaml
from loguru import logger

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_CONFIG_PATH = _PROJECT_ROOT / "config" / "config.yaml"

_LOG_FORMAT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{module}</cyan> | "
    "<level>{message}</level>"
)

_initialized = False


def _load_logging_config() -> dict:
    """Load logging section from config/config.yaml if available."""
    if _CONFIG_PATH.exists():
        with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        return cfg.get("logging", {})
    return {}


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    rotation: str = "10 MB",
) -> None:
    """Configure loguru logger with console and optional file output.

    Parameters
    ----------
    level : str
        Minimum log level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
    log_file : str or None
        Path to log file. If None, only console logging is enabled.
    rotation : str
        Log file rotation policy (e.g. "10 MB", "1 day").
    """
    global _initialized

    file_cfg = _load_logging_config()
    level = file_cfg.get("level", level)
    log_file = file_cfg.get("log_file", log_file)
    rotation = file_cfg.get("rotation", rotation)

    logger.remove()

    logger.add(
        sys.stderr,
        format=_LOG_FORMAT,
        level=level,
        colorize=True,
    )

    if log_file:
        log_path = Path(log_file)
        if not log_path.is_absolute():
            log_path = _PROJECT_ROOT / log_path
        log_path.parent.mkdir(parents=True, exist_ok=True)
        logger.add(
            str(log_path),
            format=_LOG_FORMAT,
            level=level,
            rotation=rotation,
            encoding="utf-8",
            enqueue=True,
        )

    _initialized = True


def get_logger(name: Optional[str] = None) -> logger.__class__:
    """Return a loguru logger instance, initializing if needed.

    Parameters
    ----------
    name : str or None
        Optional context name bound to the logger.

    Returns
    -------
    loguru.Logger
        Configured logger instance.
    """
    global _initialized
    if not _initialized:
        setup_logging()

    if name:
        return logger.bind(name=name)
    return logger

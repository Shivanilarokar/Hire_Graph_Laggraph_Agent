"""Central logging setup for HireGraph.

One place configures the root logger (level via ``LOG_LEVEL``, default INFO).
Modules call ``get_logger(__name__)``. Node-level chatter is logged at DEBUG so
the CLI stays clean by default; service calls and lifecycle events are INFO;
failures are logged with a traceback before they propagate.
"""

from __future__ import annotations

import logging
import os
import sys

_CONFIGURED = False


def setup_logging(level: str | None = None) -> None:
    """Configure the root logger once. Safe to call multiple times."""
    global _CONFIGURED
    if _CONFIGURED:
        return
    lvl = (level or os.getenv("LOG_LEVEL", "INFO")).upper()
    logging.basicConfig(
        level=lvl,
        format="%(asctime)s | %(levelname)-7s | %(name)-22s | %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
    )
    # Quiet noisy third-party loggers.
    for noisy in ("httpx", "httpcore", "urllib3", "openai", "redis", "redisvl"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    # LangSmith spams connection warnings when offline; only show real errors.
    for quiet in ("langsmith", "langsmith.client"):
        logging.getLogger(quiet).setLevel(logging.ERROR)
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a module logger, ensuring logging is configured."""
    setup_logging()
    return logging.getLogger(name)

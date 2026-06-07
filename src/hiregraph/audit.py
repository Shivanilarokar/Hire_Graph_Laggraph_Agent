"""Audit decorator: per-node timing, verdict, logging, and error visibility.

Wrapping a node with ``@audited("name")`` appends one audit row
``{node, verdict, elapsed_ms}`` to the ``audit_trail`` reducer on every call
(a plain dict, so it stays Redis-serializable). It also:
  - logs node completion at DEBUG and failures at ERROR (with traceback), and
  - re-raises exceptions so LangGraph or the caller can handle them.

``functools.wraps`` copies the node's return annotation, so LangGraph still sees
the ``Command[Literal[...]]`` type and visualizes routing correctly.
"""

from __future__ import annotations

import functools
import time
from collections.abc import Callable
from typing import Any

from langgraph.errors import GraphBubbleUp
from langgraph.types import Command

from hiregraph.logging_config import get_logger

log = get_logger("hiregraph.node")


def _verdict(result: Any) -> str:
    if isinstance(result, Command):
        goto = result.goto
        return f"-> {goto}" if isinstance(goto, str) else "fan-out"
    return "ok"


def audited(name: str) -> Callable:
    """Return a decorator that records timing + verdict + logs for a node."""

    def decorator(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrapper(state: Any, *args: Any, **kwargs: Any) -> Any:
            start = time.perf_counter()
            try:
                result = fn(state, *args, **kwargs)
            except GraphBubbleUp:
                raise  # interrupt()/Command bubble-up is control flow, not an error
            except Exception as exc:
                log.error("node '%s' failed: %s", name, exc)  # concise by default
                log.debug("node '%s' traceback", name, exc_info=True)  # full at DEBUG
                raise
            elapsed = round((time.perf_counter() - start) * 1000, 1)
            verdict = _verdict(result)
            log.debug("%-22s %-22s %8.1f ms", name, verdict, elapsed)
            entry = {"node": name, "verdict": verdict, "elapsed_ms": elapsed}
            if isinstance(result, Command):
                update = dict(result.update or {})
                update["audit_trail"] = [entry]
                return Command(update=update, goto=result.goto)
            update = dict(result or {})
            update["audit_trail"] = [entry]
            return update

        return wrapper

    return decorator

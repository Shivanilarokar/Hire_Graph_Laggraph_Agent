"""External services with per-call mock fallback (production-grade degradation).

Each function uses a real API when its key is present and a deterministic stub
when the key is missing or ``USE_MOCKS=true`` - the guard is **inline, at the
call site** (no separate mock module). Recoverable failures are logged and
raised as typed exceptions that nodes or LangGraph can handle explicitly.
"""

from __future__ import annotations

import smtplib
import time
import uuid
from email.message import EmailMessage
from pathlib import Path
from typing import Any

import httpx

from hiregraph.config import get_settings
from hiregraph.logging_config import get_logger

log = get_logger("hiregraph.services")


class TavilyError(Exception):
    """Recoverable error from the Tavily web-search service."""


class GitHubError(Exception):
    """Recoverable error from the GitHub REST API."""


class EmailSendError(Exception):
    """Recoverable error while sending the candidate email."""


class ATSError(Exception):
    """Recoverable error while writing to the applicant tracking system."""


def read_document(path: str) -> str:
    """Read a résumé/JD from disk. PDF via PyMuPDF, else plain text/markdown."""
    # Quoted PowerShell arguments can accidentally include a pasted line break
    # and indentation. Preserve spaces within each line, but remove that wrapping.
    normalized_path = "".join(line.strip() for line in path.splitlines())
    p = Path(normalized_path)
    if not p.exists():
        raise FileNotFoundError(f"Document not found: {normalized_path}")
    if p.suffix.lower() == ".pdf":
        import fitz  # PyMuPDF

        log.debug("reading PDF %s", p.name)
        with fitz.open(p) as doc:
            return "\n".join(page.get_text() for page in doc)
    log.debug("reading text %s", p.name)
    return p.read_text(encoding="utf-8")


def tavily_search(query: str) -> list[str]:
    """Web search via Tavily; deterministic stub when no key / mocked."""
    settings = get_settings()
    if settings.tavily_is_mock:
        log.debug("tavily mock for query=%r", query[:40])
        return [f"(mock) Public mention: candidate is active around '{query[:40]}'."]
    try:
        from tavily import TavilyClient

        log.info("tavily search: %r", query[:60])
        data = TavilyClient(api_key=settings.tavily_api_key).search(query=query, max_results=3)
        return [r.get("content", "") for r in data.get("results", [])]
    except Exception as exc:
        log.warning("tavily search failed: %s", exc)
        raise TavilyError(str(exc)) from exc


def github_lookup(username: str) -> dict[str, Any]:
    """Public GitHub profile via REST; deterministic stub when mocked/empty."""
    settings = get_settings()
    if settings.use_mocks or not username:
        log.debug("github mock for user=%r", username)
        return {"login": username or "unknown", "public_repos": 12, "followers": 30, "mock": True}
    try:
        headers = (
            {"Authorization": f"Bearer {settings.github_token}"} if settings.github_token else {}
        )
        log.info("github lookup: %s", username)
        resp = httpx.get(
            f"{settings.github_api_base}/users/{username}", headers=headers, timeout=10
        )
        resp.raise_for_status()
        d = resp.json()
        return {
            "login": d.get("login"),
            "public_repos": d.get("public_repos"),
            "followers": d.get("followers"),
            "mock": False,
        }
    except httpx.HTTPError as exc:
        log.warning("github lookup failed for %s: %s", username, exc)
        raise GitHubError(str(exc)) from exc


def send_email(to: str, subject: str, body: str) -> dict[str, Any]:
    """Send via Mailtrap sandbox SMTP; print-only stub when mocked."""
    settings = get_settings()
    if settings.email_is_mock:
        log.info("[email-out:mock] to=%s subject=%r", to, subject)
        time.sleep(0.01)
        return {"sent": True, "mock": True, "to": to}
    try:
        msg = EmailMessage()
        msg["From"], msg["To"], msg["Subject"] = settings.email_from, to, subject
        msg.set_content(body)
        log.info("sending email via Mailtrap to %s", to)
        with smtplib.SMTP(settings.mailtrap_host, settings.mailtrap_port, timeout=20) as server:
            server.starttls()
            server.login(settings.mailtrap_username, settings.mailtrap_password)
            server.send_message(msg)
        return {"sent": True, "mock": False, "to": to}
    except Exception as exc:
        log.warning("email send failed to %s: %s", to, exc)
        raise EmailSendError(str(exc)) from exc


def log_to_ats(decision: dict[str, Any], force_failure: bool = False) -> str:
    """Append the decision to the (mock) ATS. ``force_failure`` exercises the saga."""
    if force_failure:
        log.warning("ATS write forced to fail (saga demo)")
        raise ATSError("ATS write failed after retries (simulated)")
    ats_id = f"ATS-{uuid.uuid4().hex[:6].upper()}"
    log.info("ATS recorded %s for %s", ats_id, decision.get("candidate"))
    time.sleep(0.01)
    return ats_id

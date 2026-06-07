"""End-to-end test with the in-memory checkpointer.

Runs the always-borderline candidate (Eitan) against the senior backend JD,
asserts the graph pauses at the human-review interrupt, resumes with a fake
decision, and reaches a clean terminal state.
"""

from __future__ import annotations

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from hiregraph.graph import build_graph
from hiregraph.nodes import decision

_RESUME = "sample_data/resumes/resume_eitan.md"
_MIRA_RESUME = "sample_data/resumes/resume_mira.md"
_JD = "sample_data/jds/jd_senior_backend.md"


def test_borderline_pauses_then_resumes_to_terminal():
    app = build_graph().compile(checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "eitan-e2e"}}

    app.invoke({"resume_path": _RESUME, "jd_path": _JD}, config)
    paused = app.get_state(config)
    assert paused.next, "borderline candidate should pause at human_review"

    out = app.invoke(Command(resume={"approved": True}), config)
    assert out.get("recommendation") == "borderline"
    assert out.get("terminal_status") == "completed"
    assert app.get_state(config).next == ()  # fully resolved


def test_automatic_rejection_sends_email_and_updates_ats(monkeypatch):
    app = build_graph().compile(checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "mira-rejection-e2e"}}
    sent: dict[str, str] = {}

    def capture_email(to: str, subject: str, body: str) -> dict:
        sent.update({"to": to, "subject": subject, "body": body})
        return {"sent": True, "mock": True, "to": to}

    monkeypatch.setattr(decision, "_send_email", capture_email)

    out = app.invoke({"resume_path": _MIRA_RESUME, "jd_path": _JD}, config)

    assert out.get("recommendation") == "reject"
    assert out.get("sent_status") == "sent"
    assert out.get("ats_id", "").startswith("ATS-")
    assert out.get("terminal_status") == "completed"
    assert sent["subject"] == "Your application - reject"
    assert sent["body"] == out.get("email_draft")
    assert "not moving forward" in sent["body"].lower()
    assert app.get_state(config).next == ()

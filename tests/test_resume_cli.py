"""Interactive single-resume CLI tests."""

from __future__ import annotations

from langgraph.checkpoint.memory import InMemorySaver

from hiregraph.graph import build_graph
from hiregraph.resume_cli import prompt_for_decision, run_resume

_RESUME = "sample_data/resumes/resume_eitan.md"
_JD = "sample_data/jds/jd_senior_backend.md"


def test_prompt_for_decision_repeats_until_valid_rejection():
    answers = iter(["maybe", "r"])
    output: list[str] = []

    approved = prompt_for_decision(
        {"candidate": "Eitan", "final_score": 56},
        input_fn=lambda _prompt: next(answers),
        print_fn=output.append,
    )

    assert approved is False
    assert any("Please enter A or R" in line for line in output)


def test_run_resume_rejects_only_after_graph_is_paused():
    app = build_graph().compile(checkpointer=InMemorySaver())
    thread_id = "interactive-reject"
    config = {"configurable": {"thread_id": thread_id}}
    output: list[str] = []

    def reject_while_paused(_prompt: str) -> str:
        assert app.get_state(config).next == ("human_review",)
        return "r"

    result = run_resume(
        app,
        _RESUME,
        _JD,
        thread_id=thread_id,
        input_fn=reject_while_paused,
        print_fn=output.append,
    )

    assert result["human_decision"] == "reject"
    assert result["recommendation"] == "reject"
    assert result["sent_status"] == "sent"
    assert result["ats_id"].startswith("ATS-")
    assert "not moving forward" in result["email_draft"].lower()
    assert result["terminal_status"] == "completed"
    assert result["_paused"] is True
    assert any(
        entry["node"] == "send_email_update_ats" for entry in result.get("audit_trail", [])
    )
    assert any("HUMAN REVIEW REQUIRED" in line for line in output)


def test_run_resume_approves_only_after_graph_is_paused():
    app = build_graph().compile(checkpointer=InMemorySaver())
    thread_id = "interactive-approve"
    config = {"configurable": {"thread_id": thread_id}}

    def approve_while_paused(_prompt: str) -> str:
        assert app.get_state(config).next == ("human_review",)
        return "a"

    result = run_resume(
        app,
        _RESUME,
        _JD,
        thread_id=thread_id,
        input_fn=approve_while_paused,
        print_fn=lambda _line: None,
    )

    assert result["human_decision"] == "approve"
    assert result["sent_status"] == "sent"
    assert result["terminal_status"] == "completed"


def test_run_resume_auto_approve_never_prompts():
    app = build_graph().compile(checkpointer=InMemorySaver())

    def fail_if_prompted(_prompt: str) -> str:
        raise AssertionError("auto-approve must not prompt")

    result = run_resume(
        app,
        _RESUME,
        _JD,
        thread_id="auto-approve",
        auto_approve=True,
        input_fn=fail_if_prompted,
        print_fn=lambda _line: None,
    )

    assert result["human_decision"] == "approve"
    assert result["sent_status"] == "sent"


def test_run_resume_reopens_an_existing_paused_thread():
    app = build_graph().compile(checkpointer=InMemorySaver())
    thread_id = "resume-after-cancel"

    def cancel_review(_prompt: str) -> str:
        raise EOFError

    try:
        run_resume(
            app,
            _RESUME,
            _JD,
            thread_id=thread_id,
            input_fn=cancel_review,
            print_fn=lambda _line: None,
        )
    except EOFError:
        pass
    else:
        raise AssertionError("the first review should have been cancelled")

    paused = app.get_state({"configurable": {"thread_id": thread_id}})
    assert paused.next == ("human_review",)
    initial_audit_length = len(paused.values.get("audit_trail", []))
    output: list[str] = []

    result = run_resume(
        app,
        _RESUME,
        _JD,
        thread_id=thread_id,
        input_fn=lambda _prompt: "r",
        print_fn=output.append,
    )

    ingest_entries = [
        entry for entry in result.get("audit_trail", []) if entry["node"] == "ingest_resume_and_jd"
    ]
    assert len(ingest_entries) == 1
    assert len(result.get("audit_trail", [])) > initial_audit_length
    assert result["human_decision"] == "reject"
    assert any("Resuming existing paused thread" in line for line in output)

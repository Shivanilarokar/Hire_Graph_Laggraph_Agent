"""State test: construct HireGraphState and exercise the fan-in reducer."""

from __future__ import annotations

import operator

from hiregraph.state import HireGraphState


def test_state_accepts_minimal_input_fields():
    state: HireGraphState = {
        "candidate_id": "c1",
        "resume_path": "r.md",
        "jd_path": "j.md",
        "resume_text": "raw",
        "jd_text": "jd",
    }
    assert state["resume_text"] == "raw"
    # Derived fields are absent until nodes produce them (total=False).
    assert "final_score" not in state
    assert "recommendation" not in state


def test_completed_scores_reducer_concatenates():
    # The Annotated[list, operator.add] reducer on completed_scores:
    # parallel branches each return one entry; the framework concatenates them.
    a = [{"kind": "skill", "label": "Python", "score": 80}]
    b = [{"kind": "experience", "label": "experience", "score": 70}]
    merged = operator.add(a, b)
    assert [s["kind"] for s in merged] == ["skill", "experience"]

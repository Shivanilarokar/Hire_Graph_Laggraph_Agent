"""Node tests with the LLM mocked (via the deterministic MockChatModel)."""

from __future__ import annotations

from langgraph.types import Command

from hiregraph.nodes.classify import classify_seniority
from hiregraph.nodes.decision import aggregate_scores


def test_classify_seniority_returns_command_and_routes():
    state = {
        "jd_text": "Senior Backend Engineer",
        "resume_text": "8 years of python experience",
        "years_experience": 8,
    }
    cmd = classify_seniority(state)
    assert isinstance(cmd, Command)
    assert cmd.goto == "plan_required_skills"
    assert cmd.update["classification"]["seniority"] == "senior"


def test_aggregate_scores_combines_completed_scores_and_recommends_advance():
    # Every parallel branch appended one dict to the completed_scores reducer.
    state = {
        "completed_scores": [
            {"kind": "skill", "label": "Python", "score": 90},
            {"kind": "skill", "label": "Kafka", "score": 86},
            {"kind": "experience", "label": "experience", "score": 90},
            {"kind": "education", "label": "education", "score": 80},
            {"kind": "signal", "label": "signal", "score": 80},
            {"kind": "research", "label": "research", "score": 75},
        ],
        "classification": {"seniority": "senior"},
        "candidate_name": "Test Candidate",
    }
    cmd = aggregate_scores(state)
    assert cmd.goto == "route_recommendation"
    assert cmd.update["recommendation"] == "advance"
    assert cmd.update["final_score"] >= 74

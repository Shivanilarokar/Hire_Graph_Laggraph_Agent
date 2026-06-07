"""classify_seniority - the first LLM call.

Returns a structured ``Classification`` (seniority + role_family + confidence) so
downstream nodes can branch on it reliably instead of parsing free text. It flows
straight to ``plan_required_skills``; the seniority it produces sets the hiring bar
the recommendation router applies later.
"""

from __future__ import annotations

from typing import Literal

from langgraph.types import Command

from hiregraph.audit import audited
from hiregraph.llm import get_llm
from hiregraph.logging_config import get_logger
from hiregraph.prompts import build_seniority_prompt
from hiregraph.schemas import Classification
from hiregraph.state import HireGraphState

log = get_logger("hiregraph.classify")


@audited("classify_seniority")
def classify_seniority(state: HireGraphState) -> Command[Literal["plan_required_skills"]]:
    """Classify seniority with structured output, then plan the required skills."""
    cls: Classification = get_llm().with_structured_output(Classification).invoke(
        build_seniority_prompt(state)
    )
    log.info(
        "SENIORITY  %s -> %s (%s, %.0f%%) | %s",
        state.get("candidate_name", "candidate"),
        cls.seniority.upper(),
        cls.role_family,
        cls.confidence * 100,
        cls.rationale,
    )
    return Command(update={"classification": cls.model_dump()}, goto="plan_required_skills")

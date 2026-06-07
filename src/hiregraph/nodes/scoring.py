"""Two flavors of fan-out that share one fan-in reducer (the diagram's top half).

Dynamic fan-out (orchestrator + worker):
    plan_required_skills (LLM, structured -> required skills)
      -> add_conditional_edges(plan_required_skills, assign_skill_workers, [skill_worker])
           assign_skill_workers returns one Send("skill_worker", {...}) per skill, so
           the number of workers is decided at runtime from the planner's output.

Fixed fan-out (static parallelization):
    plan_required_skills also fans out, via plain edges, to a known set of scorers:
    experience_scorer, education_scorer, signal_scorer, research_agent.

Both converge on aggregate_scores. Every branch appends ONE dict to
``completed_scores`` (``Annotated[list, operator.add]``), so the reducer
concatenates them with no write conflict and no synchronization code.
"""

from __future__ import annotations

from langgraph.types import Send

from hiregraph.audit import audited
from hiregraph.llm import get_llm
from hiregraph.logging_config import get_logger
from hiregraph.prompts import (
    build_dimension_prompt,
    build_plan_skills_prompt,
    build_skill_worker_prompt,
)
from hiregraph.schemas import DimensionScore, RequiredSkills, SkillScore
from hiregraph.state import HireGraphState

log = get_logger("hiregraph.scoring")


@audited("plan_required_skills")
def plan_required_skills(state: HireGraphState) -> dict:
    """Orchestrator (LLM, structured output): plan the JD's required skills.

    Emits ``required_skills``; the ``assign_skill_workers`` conditional edge then
    fans out one worker per skill, while static edges fan out the scorers."""
    result: RequiredSkills = get_llm().with_structured_output(RequiredSkills).invoke(
        build_plan_skills_prompt(state)
    )
    skills = [s.model_dump() for s in result.skills[:5]] or [{"name": "General Backend"}]
    log.info("PLAN       required skills: %s", ", ".join(s["name"] for s in skills))
    return {"required_skills": skills}


def assign_skill_workers(state: HireGraphState) -> list[Send]:
    """Conditional-edge routing function: one ``Send("skill_worker", ...)`` per
    required skill - the count is decided at runtime from the planner's output."""
    resume = state.get("resume_text", "")
    skills = state.get("required_skills") or [{"name": "General Backend"}]
    return [Send("skill_worker", {"skill": s["name"], "resume_text": resume}) for s in skills]


@audited("skill_worker")
def skill_worker(payload: dict) -> dict:
    """Worker: score ONE assigned skill against the résumé; append to the reducer."""
    skill = payload.get("skill", "General Backend")
    result: SkillScore = get_llm().with_structured_output(SkillScore).invoke(
        build_skill_worker_prompt(skill, payload)
    )
    return {"completed_scores": [{"kind": "skill", "label": skill, "score": result.score}]}


def _dimension(kind: str, state: HireGraphState) -> dict:
    result: DimensionScore = get_llm().with_structured_output(DimensionScore).invoke(
        build_dimension_prompt(kind, state)
    )
    return {"completed_scores": [{"kind": kind, "label": kind, "score": result.score}]}


@audited("experience_scorer")
def experience_scorer(state: HireGraphState) -> dict:
    """Static parallel scorer: years / depth of experience."""
    return _dimension("experience", state)


@audited("education_scorer")
def education_scorer(state: HireGraphState) -> dict:
    """Static parallel scorer: education / credentials."""
    return _dimension("education", state)


@audited("signal_scorer")
def signal_scorer(state: HireGraphState) -> dict:
    """Static parallel scorer: public signals (GitHub, writing, talks)."""
    return _dimension("signal", state)

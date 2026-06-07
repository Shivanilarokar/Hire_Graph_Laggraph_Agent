"""ingest_resume_and_jd - the intake "data step".

Loads the raw résumé + JD, then runs a three-call chain that turns that text into
the candidate facts the rest of the graph needs:

    parse_resume -> normalize_skills -> extract_experience

Each call feeds the next: the parsed skills are canonicalized, then the years of
experience are estimated from the cleaned-up picture. Keeping the chain inside this
one node keeps intake a single box in the graph.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from langgraph.types import Command

from hiregraph.audit import audited
from hiregraph.llm import get_llm
from hiregraph.logging_config import get_logger
from hiregraph.prompts import (
    build_experience_prompt,
    build_normalize_skills_prompt,
    build_parse_resume_prompt,
)
from hiregraph.schemas import ExperienceExtract, NormalizedSkills, ParsedResume
from hiregraph.services import read_document
from hiregraph.state import HireGraphState

log = get_logger("hiregraph.intake")


def _parse_resume(state: HireGraphState) -> ParsedResume:
    """Chain step 1: structured facts out of raw résumé text."""
    return get_llm().with_structured_output(ParsedResume).invoke(build_parse_resume_prompt(state))


def _normalize_skills(state: HireGraphState) -> NormalizedSkills:
    """Chain step 2: canonicalize skill names (e.g. 'reactjs' -> 'React')."""
    return get_llm().with_structured_output(NormalizedSkills).invoke(
        build_normalize_skills_prompt(state)
    )


def _extract_experience(state: HireGraphState) -> ExperienceExtract:
    """Chain step 3: derive years of professional experience."""
    return get_llm().with_structured_output(ExperienceExtract).invoke(build_experience_prompt(state))


@audited("ingest_resume_and_jd")
def ingest_resume_and_jd(state: HireGraphState) -> Command[Literal["classify_seniority"]]:
    """Load the raw files, then run the parse -> normalize -> extract chain."""
    resume = read_document(state["resume_path"])
    jd = read_document(state["jd_path"])
    candidate_id = state.get("candidate_id") or Path(state["resume_path"]).stem
    chained = {**state, "resume_text": resume, "jd_text": jd}

    parsed = _parse_resume(chained)
    chained["normalized_skills"] = parsed.raw_skills
    skills = _normalize_skills(chained)
    chained["normalized_skills"] = skills.skills
    experience = _extract_experience(chained)

    log.info("INGEST     %s | %d skills | ~%.0fy", parsed.full_name, len(skills.skills), experience.years_experience)
    return Command(
        update={
            "resume_text": resume,
            "jd_text": jd,
            "candidate_id": candidate_id,
            "candidate_name": parsed.full_name,
            "candidate_email": parsed.email,
            "github_username": parsed.github_username,
            "normalized_skills": skills.skills,
            "years_experience": experience.years_experience,
        },
        goto="classify_seniority",
    )

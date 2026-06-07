"""Pydantic schemas and Literal types for the graph's structured LLM calls.

Each model is a ``with_structured_output(...)`` target, so the model is forced to
return a typed object instead of free text. Instances are immediately converted to
plain dicts (``model_dump()``) before they land in state, keeping the whole state
JSON-serializable for the Redis checkpointer.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

Seniority = Literal["junior", "mid", "senior", "executive"]
Recommendation = Literal["advance", "reject", "borderline"]


# --- Outputs of the intake parse -> normalize -> extract chain ---------------
class ParsedResume(BaseModel):
    """Chain step 1: pull facts out of raw résumé text."""

    full_name: str = Field(description="Candidate full name, best effort")
    email: str = ""
    raw_skills: list[str] = Field(default_factory=list)
    github_username: str = ""


class NormalizedSkills(BaseModel):
    """Chain step 2: canonicalize skill names (e.g. 'reactjs' -> 'React')."""

    skills: list[str] = Field(default_factory=list)


class ExperienceExtract(BaseModel):
    """Chain step 3: derive years of experience."""

    years_experience: float = Field(ge=0)


# --- Seniority classification ------------------------------------------------
class Classification(BaseModel):
    """What the seniority classifier returns; downstream nodes branch on it."""

    seniority: Seniority
    role_family: str = Field(description="backend, frontend, data, ...")
    confidence: float = Field(ge=0, le=1)
    rationale: str = ""


# --- Skills the planner pulls from the JD, scored one worker each ------------
class Skill(BaseModel):
    """One required skill the planner pulls from the job description."""

    name: str


class RequiredSkills(BaseModel):
    """The planner's full list of skills to evaluate the candidate against."""

    skills: list[Skill] = Field(default_factory=list)


class SkillScore(BaseModel):
    """One skill worker's verdict on the single skill it was assigned."""

    score: int = Field(ge=0, le=100)


# --- One of the fixed dimension scorers --------------------------------------
class DimensionScore(BaseModel):
    """A single scorer's read on one dimension (experience / education / signal)."""

    score: int = Field(ge=0, le=100)


# --- The critic's grade of an email draft ------------------------------------
class CriticFeedback(BaseModel):
    """The critic's grade of an email draft; decides whether to revise or send."""

    approved: bool
    score: int = Field(ge=0, le=100)
    feedback: str = Field(default="", description="actionable feedback")

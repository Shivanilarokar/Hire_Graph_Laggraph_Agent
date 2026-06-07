"""The graph state: one TypedDict that travels through every node.

Structured-output results are kept as plain dicts (``model_dump()``) so the
state stays JSON-serializable for the Redis checkpointer. Prompts are built on
demand inside nodes rather than stored here.

``Annotated[..., reducer]`` keys merge writes from branches that run concurrently:
  - ``completed_scores`` -> operator.add  (the fan-in: every skill worker and every
    parallel scorer appends one ``{kind, label, score}`` dict; the reducer
    concatenates them so no branch overwrites another)
  - ``critic_feedback``  -> operator.add   (one entry per critic pass)
  - ``compensation_log`` / ``audit_trail`` -> operator.add
"""

from __future__ import annotations

import operator
from typing import Annotated, Literal, TypedDict

from hiregraph.schemas import Recommendation


class HireGraphState(TypedDict, total=False):
    """Shared state for the hiring graph. All keys optional (total=False)."""

    # --- Inputs (raw data) ---------------------------------------------------
    candidate_id: str
    resume_path: str
    jd_path: str
    resume_text: str
    jd_text: str

    # --- Produced by ingest_resume_and_jd (parse -> normalize -> extract) -----
    candidate_name: str
    candidate_email: str
    github_username: str
    normalized_skills: list[str]
    years_experience: float

    # --- classify_seniority (structured output) -> Classification.model_dump() -
    classification: dict | None

    # --- plan_required_skills (orchestrator) -> list[Skill] as dicts ---------
    required_skills: list[dict]

    # --- Research agent findings ---------------------------------------------
    research_findings: dict

    # --- Fan-in reducer: every parallel branch appends one score -------------
    completed_scores: Annotated[list[dict], operator.add]

    # --- aggregate_scores ----------------------------------------------------
    final_score: int | None
    recommendation: Recommendation | None
    scorecard_summary: str

    # --- Drafting + critic revision loop -------------------------------------
    email_draft: str | None
    critic_feedback: Annotated[list[dict], operator.add]
    draft_attempts: int

    # --- Human review (pause / resume) ---------------------------------------
    human_decision: Literal["approve", "reject"] | None

    # --- Outcome + compensation record on delivery failure ------------------
    force_ats_failure: bool
    ats_id: str
    sent_status: Literal["sent", "failed"] | None
    terminal_status: Literal["completed", "compensated"] | None
    compensation_log: Annotated[list[str], operator.add]

    # --- Audit trail (node, verdict, timing) ---------------------------------
    audit_trail: Annotated[list[dict], operator.add]

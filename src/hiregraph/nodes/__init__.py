"""Node functions, grouped by stage. Re-exported for the graph builder."""

from hiregraph.nodes.classify import classify_seniority
from hiregraph.nodes.decision import (
    aggregate_scores,
    compensate,
    critic_loop,
    draft_email,
    draft_rejection,
    finalize,
    human_review,
    route_recommendation,
    send_email_update_ats,
)
from hiregraph.nodes.intake import ingest_resume_and_jd
from hiregraph.nodes.research import research_agent
from hiregraph.nodes.scoring import (
    assign_skill_workers,
    education_scorer,
    experience_scorer,
    plan_required_skills,
    signal_scorer,
    skill_worker,
)

__all__ = [
    "ingest_resume_and_jd", "classify_seniority", "plan_required_skills",
    "assign_skill_workers", "skill_worker",
    "experience_scorer", "education_scorer", "signal_scorer", "research_agent",
    "aggregate_scores", "route_recommendation", "draft_email", "critic_loop",
    "human_review", "draft_rejection", "send_email_update_ats",
    "compensate", "finalize",
]

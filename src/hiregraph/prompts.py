"""Prompt builders: graph state in, formatted prompt string out.

Prompts are built immediately before each LLM call and are never stored in the
main graph state. The deterministic mock model relies on the explicit
``Skill:``, ``Dimension:``, and ``Years:`` markers below.
"""

from __future__ import annotations

from hiregraph.state import HireGraphState


def _resume(state: HireGraphState) -> str:
    return state.get("resume_text", "")


def _jd(state: HireGraphState) -> str:
    return state.get("jd_text", "")


def build_parse_resume_prompt(state: HireGraphState) -> str:
    return (
        "Extract structured facts from this resume.\n\n"
        f"Resume:\n{_resume(state)}\n"
        "Return full_name, email, raw_skills, and github_username."
    )


def build_normalize_skills_prompt(state: HireGraphState) -> str:
    raw = ", ".join(state.get("normalized_skills") or []) or "(see resume)"
    return (
        "Canonicalize skills to standard names (e.g. 'reactjs' -> 'React').\n\n"
        f"Resume:\n{_resume(state)}\nRaw skills: {raw}\n"
        "Return the cleaned skills list."
    )


def build_experience_prompt(state: HireGraphState) -> str:
    return (
        "Estimate total years of professional experience.\n\n"
        f"Resume:\n{_resume(state)}\n"
        "Return years_experience."
    )


def build_seniority_prompt(state: HireGraphState) -> str:
    return (
        "Classify the candidate's OVERALL career seniority and their role family.\n"
        "Seniority is based on total years/scope, independent of whether their "
        "domain matches this role (a senior frontend engineer is still senior).\n"
        "role_family is their primary domain: backend, frontend, data, fullstack, etc.\n\n"
        f"Years of experience: {state.get('years_experience', 0)}\n"
        f"Resume:\n{_resume(state)}\n"
        "Rough guide: 0-1y junior, 2-5y mid, 6-10y senior, 11y+ executive.\n"
        "Return seniority, role_family, confidence, and rationale."
    )


def build_plan_skills_prompt(state: HireGraphState) -> str:
    return (
        "From this job description, list the 3-6 most important required technical "
        "skills to evaluate the candidate against.\n\n"
        f"Job description:\n{_jd(state)}\n"
        "Return a short list of skills."
    )


def build_skill_worker_prompt(skill: str, state: HireGraphState) -> str:
    return (
        "Score how well the candidate demonstrates one required skill.\n"
        f"Skill: {skill}\n\n"
        f"Resume:\n{_resume(state)}\n"
        "Return score (0-100)."
    )


def build_dimension_prompt(dimension: str, state: HireGraphState) -> str:
    return (
        "Score the candidate on a single hiring dimension (0-100).\n"
        f"Dimension: {dimension}\n\n"
        f"Job description:\n{_jd(state)}\n\nResume:\n{_resume(state)}\n"
        f"Normalized skills: {', '.join(state.get('normalized_skills') or [])}\n"
        f"Years: {state.get('years_experience', 0)}  "
        f"GitHub: {state.get('github_username', '')}\n"
        "Return score (0-100)."
    )


def build_research_messages(state: HireGraphState) -> list:
    from langchain_core.messages import HumanMessage, SystemMessage

    system = SystemMessage(
        content=(
            "You are a research assistant that VERIFIES a candidate's self-reported "
            "claims against independent, outside evidence. The resume and JD are already "
            "known - do NOT look those up. Use github_lookup for structured profile data "
            "and web_search to corroborate claimed employers, achievements, or writing. "
            "Decide which tool fits each gap, call it when useful, then summarize."
        )
    )
    human = HumanMessage(
        content=(
            f"Candidate: {state.get('candidate_name', '')}\n"
            f"GitHub handle: {state.get('github_username', '')}\n"
            f"Skills: {', '.join(state.get('normalized_skills') or [])}\n\n"
            f"Claims to verify (from resume):\n{_resume(state)[:1200]}\n\n"
            "Corroborate the strongest claims with outside evidence."
        )
    )
    return [system, human]


def build_email_prompt(state: HireGraphState) -> str:
    rec = state.get("recommendation", "borderline")
    score = state.get("final_score", 0)
    feedback = state.get("critic_feedback") or []
    last = feedback[-1] if feedback else None
    revision = (
        f"\nReviser note - address this: {last['feedback']}\n"
        if last and not last.get("approved")
        else ""
    )
    return (
        f"Draft a short, warm, personalized email to candidate "
        f"{state.get('candidate_name', '')}.\n"
        f"Decision: {rec} (overall score {score}).\n\n"
        f"Role context:\n{_jd(state)}\n{revision}\n"
        "If advancing, invite to next steps; if rejecting, be kind and encouraging; "
        "if borderline, ask one clarifying question. Sign as 'The Recruiting Team'."
    )


def build_critique_prompt(state: HireGraphState) -> str:
    return (
        "Critique this recruiting email draft for tone, clarity, personalization, "
        "and correctness.\n\n"
        f"Draft:\n{state.get('email_draft', '')}\n\n"
        "Return approved (bool), score (0-100), and specific actionable feedback."
    )

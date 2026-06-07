"""LLM factory and its deterministic offline twin.

``get_llm()`` returns the chat model the graph should use:
  - the real **OpenAI** client (``gpt-4o-mini`` by default), or
  - a deterministic ``MockChatModel`` when ``USE_MOCKS=true`` or no key is set.

The mock makes **no network or LLM calls**, so a keyless clone runs end to end.
Its outputs are tuned to reproduce ``expected_outcomes.md``.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any

from langchain_core.messages import AIMessage, ToolMessage

from hiregraph import schemas
from hiregraph.config import get_settings

_BACKEND = {
    "python", "go", "golang", "java", "rust", "node", "node.js", "postgres",
    "postgresql", "mysql", "redis", "kafka", "rabbitmq", "docker", "kubernetes",
    "k8s", "aws", "gcp", "azure", "terraform", "microservices", "grpc", "rest",
    "graphql", "sql", "django", "fastapi", "flask", "spring", "ci/cd", "linux",
}
_FRONTEND = {
    "react", "vue", "angular", "javascript", "typescript", "css", "html",
    "tailwind", "redux", "next.js", "vite", "figma", "sass", "webpack", "storybook",
}
_JD_SKILLS = [
    "Python", "Go", "Java", "PostgreSQL", "Redis", "Kafka", "Docker",
    "Kubernetes", "AWS", "Microservices", "REST", "GraphQL", "React", "TypeScript",
]


# ===========================================================================
# Deterministic feature extraction
# ===========================================================================
def _as_text(prompt: Any) -> str:
    if isinstance(prompt, str):
        return prompt
    if isinstance(prompt, list):
        return "\n".join(getattr(m, "content", "") or str(m) for m in prompt)
    if hasattr(prompt, "to_string"):
        return prompt.to_string()
    return getattr(prompt, "content", None) or str(prompt)


def _clamp(n: float, lo: int = 0, hi: int = 100) -> int:
    return max(lo, min(hi, int(round(n))))


def _num_after(text: str, label: str) -> float:
    m = re.search(rf"{label}\s*:?\s*(\d+(?:\.\d+)?)", text.lower())
    return float(m.group(1)) if m else 0.0


def _split_jd_resume(text: str) -> tuple[str, str]:
    low = text.lower()
    idx = low.find("ésumé")
    if idx == -1:
        idx = low.find("resume:")
    return (text[:idx], text[idx:]) if idx != -1 else (text, text)


def _signal(text: str) -> dict[str, Any]:
    low = text.lower()
    return {
        "backend_hits": sum(1 for s in _BACKEND if s in low),
        "frontend_hits": sum(1 for s in _FRONTEND if s in low),
        "years": float(m.group(1)) if (m := re.search(r"(\d+(?:\.\d+)?)\s*\+?\s*years", low)) else 0.0,
        "github": g.group(1) if (g := re.search(r"github\.com/([A-Za-z0-9_-]+)", text)) else "",
        "has_degree": any(
            w in low for w in ("b.s", "m.s", "bachelor of science", "computer science")
        ),
    }


def _found_skills(text: str) -> list[str]:
    low = text.lower()
    return sorted({s for s in (_BACKEND | _FRONTEND) if s in low})


_NAME_SKIP = (
    "extract", "draft", "score", "classify", "résumé", "resume", "job",
    "candidate", "return", "estimate", "canonicalize", "dimension", "skill",
)


def _guess_name(text: str) -> str:
    for line in text.strip().splitlines():
        clean = re.sub(r"^#+\s*", "", line).strip()
        if clean and not clean.lower().startswith(_NAME_SKIP):
            return clean.split("|")[0].strip()[:60]
    return "Candidate"


def _marker(text: str, key: str) -> str:
    m = re.search(rf"{key}\s*[:\-]\s*([A-Za-z0-9_.+/# -]+)", text, re.IGNORECASE)
    return m.group(1).strip() if m else ""


# ===========================================================================
# Per-schema builders
# ===========================================================================
def _b_parsed(text: str, sig: dict) -> schemas.ParsedResume:
    email = re.search(r"[\w.+-]+@[\w-]+\.[\w.-]+", text)
    # The résumé body starts after the "Résumé:" header (the prompt's first
    # "résumé." mention is in the instruction, so split on the colon form).
    marker = re.search(r"r[eé]sum[eé]:\s*\n", text, re.IGNORECASE)
    body = text[marker.end():] if marker else text
    return schemas.ParsedResume(
        full_name=_guess_name(body),
        email=email.group(0) if email else "",
        raw_skills=_found_skills(text),
        github_username=sig["github"],
    )


def _b_normalized(text: str, sig: dict) -> schemas.NormalizedSkills:
    return schemas.NormalizedSkills(skills=[s.title() for s in _found_skills(text)])


def _b_experience(text: str, sig: dict) -> schemas.ExperienceExtract:
    return schemas.ExperienceExtract(years_experience=sig["years"])


def _b_classification(text: str, sig: dict) -> schemas.Classification:
    years = _num_after(text, "years of experience") or sig["years"]
    seniority = (
        "executive" if years >= 11 else "senior" if years >= 6 else "mid" if years >= 2 else "junior"
    )
    role = "frontend" if sig["frontend_hits"] > sig["backend_hits"] else "backend"
    return schemas.Classification(
        seniority=seniority, role_family=role, confidence=0.85,
        rationale=f"~{years:.0f} years; primary domain {role}.",
    )


def _b_required_skills(text: str, sig: dict) -> schemas.RequiredSkills:
    low = text.lower()
    found = [s for s in _JD_SKILLS if s.lower() in low][:5] or ["General Backend"]
    return schemas.RequiredSkills(skills=[schemas.Skill(name=s) for s in found])


def _b_skill(text: str, sig: dict) -> schemas.SkillScore:
    skill = _marker(text, "skill") or "the required skill"
    _, res_part = _split_jd_resume(text)
    present = skill.lower() in res_part.lower()
    return schemas.SkillScore(score=_clamp(88 if present else 30))


def _b_dimension(text: str, sig: dict) -> schemas.DimensionScore:
    dim = _marker(text, "dimension").lower()
    _, res = _split_jd_resume(text)
    rsig = _signal(res)
    years = _num_after(text, "years")
    if dim.startswith("experience"):
        return schemas.DimensionScore(score=_clamp(years * 11))
    if dim.startswith("education"):
        ok = rsig["has_degree"]
        return schemas.DimensionScore(score=80 if ok else 40)
    gh = rsig["github"]
    return schemas.DimensionScore(score=75 if gh else 35)


def _b_critic(text: str, sig: dict) -> schemas.CriticFeedback:
    low = text.lower()
    ok = len(low) > 120 and ("regard" in low or "sincerely" in low or "team" in low)
    return schemas.CriticFeedback(
        approved=ok,
        score=_clamp(92 if ok else 55),
        feedback="" if ok else "Add a warm closing and concrete next steps.",
    )


_BUILDERS: dict[str, Callable[[str, dict], Any]] = {
    "ParsedResume": _b_parsed,
    "NormalizedSkills": _b_normalized,
    "ExperienceExtract": _b_experience,
    "Classification": _b_classification,
    "RequiredSkills": _b_required_skills,
    "SkillScore": _b_skill,
    "DimensionScore": _b_dimension,
    "CriticFeedback": _b_critic,
}


def _mock_email(prompt: str) -> str:
    candidate = re.search(r"email to candidate\s+(.+?)\.\s*(?:\n|$)", prompt, re.IGNORECASE)
    name = candidate.group(1).strip() if candidate else _guess_name(prompt)
    if "decision: reject" in prompt.lower():
        return (
            f"Hi {name or 'there'},\n\n"
            "Thank you for taking the time to apply and for sharing your experience "
            "with us. After careful consideration, we are not moving forward with "
            "your application for this role.\n\n"
            "We appreciate your interest and wish you success in your search.\n\n"
            "Warm regards,\nThe Recruiting Team"
        )
    return (
        f"Hi {name or 'there'},\n\n"
        "Thank you for taking the time to apply. We reviewed your background "
        "against the role and wanted to follow up with clear next steps.\n\n"
        "We will be in touch shortly to schedule a conversation.\n\n"
        "Warm regards,\nThe Recruiting Team"
    )


# ===========================================================================
# Mock chat model
# ===========================================================================
class _MockStructured:
    def __init__(self, schema: type) -> None:
        self.schema = schema

    def invoke(self, prompt: Any) -> Any:
        text = _as_text(prompt)
        builder = _BUILDERS.get(self.schema.__name__)
        return builder(text, _signal(text)) if builder else self.schema()


class MockChatModel:
    """Drop-in stand-in for ChatOpenAI used offline."""

    def __init__(self, tools: list | None = None) -> None:
        self._tools = tools or []

    def bind_tools(self, tools: list, **_: Any) -> MockChatModel:
        return MockChatModel(tools=list(tools))

    def with_structured_output(self, schema: type, **_: Any) -> _MockStructured:
        return _MockStructured(schema)

    def invoke(self, prompt: Any, **_: Any) -> AIMessage:
        messages = prompt if isinstance(prompt, list) else [prompt]
        has_tool_result = any(isinstance(m, ToolMessage) for m in messages)
        if self._tools and not has_tool_result:
            name = getattr(self._tools[0], "name", "github_lookup")
            text = _as_text(messages)
            gh = _signal(text)["github"]
            args = {"username": gh or "octocat"} if "github" in name else {"query": "candidate"}
            return AIMessage(
                content="",
                tool_calls=[{"name": name, "args": args, "id": "call_mock_1", "type": "tool_call"}],
            )
        if self._tools:
            return AIMessage(content="Research complete (mock): no blocking public signals found.")
        text = _as_text(prompt).lower()
        if "email" in text or "draft" in text or "candidate" in text:
            return AIMessage(content=_mock_email(_as_text(prompt)))
        return AIMessage(content="Summary (mock): candidate profile assessed against the role.")


# ===========================================================================
# Public factory
# ===========================================================================
_REASONING_PREFIXES = ("gpt-5", "o1", "o3", "o4")


def get_llm() -> Any:
    """Return the OpenAI chat model, or a deterministic mock when keyless/mocked.

    Reasoning models (``gpt-5*``, o-series) get ``reasoning_effort='low'`` to stay
    fast; non-reasoning models (``gpt-4o-mini``, ``gpt-4.1-mini``) ignore that knob.
    """
    settings = get_settings()
    if settings.llm_is_mock:
        return MockChatModel()

    from langchain_openai import ChatOpenAI

    kwargs: dict[str, Any] = {
        "model": settings.hire_model,
        "api_key": settings.openai_api_key,
        "timeout": 60,
        "max_retries": 2,
    }
    if settings.hire_model.lower().startswith(_REASONING_PREFIXES):
        kwargs["reasoning_effort"] = "low"
    return ChatOpenAI(**kwargs)

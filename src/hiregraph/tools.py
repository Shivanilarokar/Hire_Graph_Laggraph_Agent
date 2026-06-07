"""The two tools the research agent binds and calls.

Two ``@tool`` functions wrap the external services so the agent can decide for
itself when to enrich the candidate profile. They are executed by the ``ToolNode``
inside the research sub-graph; a tool that raises is turned into an observation the
agent reasons about, thanks to ``handle_tool_errors=True``.
"""

from __future__ import annotations

from langchain_core.tools import tool

from hiregraph.services import github_lookup as _github_lookup
from hiregraph.services import tavily_search as _tavily_search


@tool
def github_lookup(username: str) -> str:
    """Look up a public GitHub profile (public repos, followers) by username."""
    d = _github_lookup(username)
    return (
        f"GitHub @{d.get('login')}: {d.get('public_repos')} public repos, "
        f"{d.get('followers')} followers."
    )


@tool
def web_search(query: str) -> str:
    """Search the public web to corroborate a candidate's claims."""
    results = _tavily_search(query)
    return "\n".join(f"- {r}" for r in results) if results else "No results found."


RESEARCH_TOOLS = [github_lookup, web_search]

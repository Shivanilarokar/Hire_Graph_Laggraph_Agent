"""research_agent - the blue node: a tool-using agent in the parallel band.

It runs a real ``llm <-> ToolNode`` loop over ``MessagesState`` (so the agent keeps
its own running message history) inside a compiled sub-graph. From the main graph
it looks like one ordinary parallel branch, which keeps the fan-in fixed-width. The
agent binds ``tavily_search`` + ``github_lookup`` and decides for itself when to
call them and when to stop; ``handle_tool_errors=True`` feeds a failed tool call
back as an observation the model reasons about instead of crashing the run.

It verifies the candidate's self-reported claims (never re-fetching the résumé or
JD), then writes back only the short ``research_findings`` summary and one research
score - the message objects stay inside the sub-graph, so the main state stays
JSON-serializable for the Redis checkpointer.
"""

from __future__ import annotations

from langgraph.graph import START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from hiregraph.audit import audited
from hiregraph.llm import get_llm
from hiregraph.prompts import build_research_messages
from hiregraph.state import HireGraphState
from hiregraph.tools import RESEARCH_TOOLS


def _research_llm(state: MessagesState) -> dict:
    """One agent turn: bind the tools and let the model decide to call or stop."""
    return {"messages": [get_llm().bind_tools(RESEARCH_TOOLS).invoke(state["messages"])]}


def build_research_subgraph():
    """Compile the llm <-> ToolNode agent loop as a reusable sub-graph."""
    sg = StateGraph(MessagesState)
    sg.add_node("llm", _research_llm)
    sg.add_node("tools", ToolNode(RESEARCH_TOOLS, handle_tool_errors=True))
    sg.add_edge(START, "llm")
    sg.add_conditional_edges("llm", tools_condition)
    sg.add_edge("tools", "llm")
    return sg.compile()


_RESEARCH = build_research_subgraph()


@audited("research_agent")
def research_agent(state: HireGraphState) -> dict:
    """Run the agent loop; record the summary and a research score (parallel branch).

    Contributes to the shared ``completed_scores`` reducer like the other branches:
    a tool-corroborated profile (GitHub handle present) scores higher."""
    result = _RESEARCH.invoke({"messages": build_research_messages(state)})
    messages = result.get("messages", [])
    summary = messages[-1].content if messages else ""
    score = 75 if state.get("github_username") else 45
    return {
        "research_findings": {"summary": summary},
        "completed_scores": [{"kind": "research", "label": "research", "score": score}],
    }

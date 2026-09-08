from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver

from ai_blog_studio.graph.state import BlogState
from ai_blog_studio.agents.tutorial_agent.agent import tutorial_node
from ai_blog_studio.agents.news_agent.agent import news_node
from ai_blog_studio.agents.validator_agent.agent import validator_node


MAX_REVISIONS = 3


def _route_start(state: BlogState) -> str:
    """Select the generation path based on the requested domain."""
    if state.get("skipped"):
        return END

    if state.get("domain") == "ainews":
        return "news_agent"

    return "tutorial_agent"


def _route_after_validator(state: BlogState) -> str:
    """
    Route approved content to END.

    Rejected content is sent back for revision while the maximum
    number of revisions has not been reached.
    """
    revision_needed = state.get("revision_needed", False)
    revision_count = state.get("revision_count", 0)

    if revision_needed and revision_count < MAX_REVISIONS:
        if state.get("domain") == "ainews":
            return "news_agent"

        return "tutorial_agent"

    return END


def build_graph() -> StateGraph:
    builder = StateGraph(BlogState)

    # Generation agents
    builder.add_node("tutorial_agent", tutorial_node)
    builder.add_node("news_agent", news_node)

    # Quality-control agent
    builder.add_node("validator", validator_node)

    # Select the appropriate generation path
    builder.add_conditional_edges(
        START,
        _route_start,
        {
            END: END,
            "news_agent": "news_agent",
            "tutorial_agent": "tutorial_agent",
        },
    )

    # Both generation paths go through validation
    builder.add_edge("tutorial_agent", "validator")
    builder.add_edge("news_agent", "validator")

    # Validator either approves or requests another revision
    builder.add_conditional_edges(
        "validator",
        _route_after_validator,
        {
            "tutorial_agent": "tutorial_agent",
            "news_agent": "news_agent",
            END: END,
        },
    )

    return builder.compile(checkpointer=InMemorySaver())


graph = build_graph()

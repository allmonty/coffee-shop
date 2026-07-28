"""The agent loop (spec §6.3).

    START → load_context → barista ⇄ tools → finish → END

The conditional edge out of `barista` **is** the agent. Everything else is setup
and teardown. If you read one file in this project, read this one.

Node bodies live here because they are four lines each; anything longer belongs
next to what it does (prompts.py, tools.py, shop/).
"""

from __future__ import annotations

import uuid
from typing import Any

from langchain_core.messages import AIMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from agent.llm import build_llm
from agent.prompts import system_prompt
from agent.state import BaristaState
from agent.tools import ALL_TOOLS
from shop import service
from shop.pricing import load_size_deltas
from shop.profile import customer_profile


async def load_context(state: BaristaState, config: RunnableConfig) -> dict[str, Any]:
    """Fetch profile, menu, wallet and cart — once per visit.

    Runs on the first turn only: the menu cannot change mid-visit, so re-reading
    it every turn would be a round-trip for data that cannot have moved.
    """
    if state.get("menu"):
        return {}

    configurable = config.get("configurable", {})
    session = configurable["session"]
    visit_id = uuid.UUID(str(configurable["visit_id"]))
    user_id = uuid.UUID(str(configurable["user_id"]))

    # `get_wallet_balance` already returns the day, so nothing here needs a
    # SQLAlchemy model. agent/ imports shop.service and nothing else from the
    # domain — that is the §5.1 boundary, and it is easy to breach by accident.
    wallet = await service.get_wallet_balance(session, visit_id)

    return {
        "user_id": str(user_id),
        "visit_id": str(visit_id),
        "customer_profile": await customer_profile(session, user_id, visit_id),
        "menu": await service.todays_menu(session, visit_id),
        "size_deltas": await load_size_deltas(session),
        "cart": await service.cart_payload(session, visit_id),
        "wallet_cents": wallet.data.get("wallet_cents", 0),
        "day": wallet.data.get("day", 1),
        "upsell_used": state.get("upsell_used", False),
        "size_offers": state.get("size_offers", {}),
        "size_declines": state.get("size_declines", 0),
        "visit_ended": False,
    }


def make_barista(llm):
    """The LLM call. `llm` is injected so tests can pass a scripted fake."""
    model = llm.bind_tools(ALL_TOOLS)

    async def barista(state: BaristaState, config: RunnableConfig) -> dict[str, Any]:
        # The system message is rebuilt every turn from live state, so the menu,
        # wallet and cart the model sees are never stale.
        messages = [SystemMessage(content=system_prompt(dict(state))), *state["messages"]]
        reply = await model.ainvoke(messages, config)
        return {"messages": [reply]}

    return barista


async def refresh(state: BaristaState, config: RunnableConfig) -> dict[str, Any]:
    """After tools run, re-read the state they may have changed.

    Only the cart and wallet — the menu is fixed for the visit.
    """
    configurable = config.get("configurable", {})
    session = configurable["session"]
    visit_id = uuid.UUID(str(configurable["visit_id"]))

    wallet = await service.get_wallet_balance(session, visit_id)
    ended = not wallet.ok and wallet.error == "visit_closed"

    return {
        "cart": await service.cart_payload(session, visit_id),
        "wallet_cents": wallet.data.get("wallet_cents", state.get("wallet_cents", 0)),
        "visit_ended": ended,
    }


def route_after_barista(state: BaristaState) -> str:
    """The whole agent loop, in one function.

    A reply carrying tool calls goes to the tool node and comes back for another
    pass. A reply that is just words ends the turn.
    """
    last = state["messages"][-1]
    if isinstance(last, AIMessage) and last.tool_calls:
        return "tools"
    return "finish"


async def finish(state: BaristaState) -> dict[str, Any]:
    return {}


def build_graph(llm=None, checkpointer=None):
    """Wiring only — no business logic in this function."""
    graph = StateGraph(BaristaState)

    graph.add_node("load_context", load_context)
    graph.add_node("barista", make_barista(llm or build_llm()))
    graph.add_node("tools", ToolNode(ALL_TOOLS))
    graph.add_node("refresh", refresh)
    graph.add_node("finish", finish)

    graph.add_edge(START, "load_context")
    graph.add_edge("load_context", "barista")
    graph.add_conditional_edges(
        "barista",
        route_after_barista,
        {"tools": "tools", "finish": "finish"},
    )
    # Tools may have changed the cart or the wallet, so refresh before the model
    # sees the result — otherwise it reads a stale total out loud.
    graph.add_edge("tools", "refresh")
    graph.add_edge("refresh", "barista")
    graph.add_edge("finish", END)

    return graph.compile(checkpointer=checkpointer)

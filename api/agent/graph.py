"""The agent loop (spec §6.3).

    START → load_context → barista ⇄ (tools → refresh) → finish → END

The conditional edge out of `barista` **is** the agent. Everything else is setup
and teardown. If you read one file in this project, read this one.

Node bodies live here because they are four lines each; anything longer belongs
next to what it does (prompts.py, tools.py, shop/).
"""

from __future__ import annotations

import json
import time
import uuid
from typing import Any

from langchain_core.messages import AIMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph

from agent.delegates import WAITER_TOOLS, set_models
from agent.dispatch import execute_tool_call
from agent.instrumentation import (
    loop_iterations,
    node_span,
    record_llm_call,
)
from agent.llm import build_llm
from agent.prompts import system_prompt
from agent.state import BaristaState
from shop import service
from shop.pricing import load_deltas
from shop.profile import customer_profile


async def load_context(state: BaristaState, config: RunnableConfig) -> dict[str, Any]:
    """Fetch profile, menu, wallet and cart — once per visit.

    Runs on the first turn only: the menu cannot change mid-visit, so re-reading
    it every turn would be a round-trip for data that cannot have moved.
    """
    if state.get("menu"):
        return {}

    with node_span("load_context"):
        return await _load_context(state, config)


async def _load_context(state: BaristaState, config: RunnableConfig) -> dict[str, Any]:
    configurable = config.get("configurable", {})
    session = configurable["session"]
    visit_id = uuid.UUID(str(configurable["visit_id"]))
    user_id = uuid.UUID(str(configurable["user_id"]))

    # `get_wallet_balance` already returns the day, so nothing here needs a
    # SQLAlchemy model. agent/ imports shop.service and nothing else from the
    # domain — that is the §5.1 boundary, and it is easy to breach by accident.
    wallet = await service.get_wallet_balance(session, visit_id)
    deltas = await load_deltas(session)

    return {
        "user_id": str(user_id),
        "visit_id": str(visit_id),
        "customer_profile": await customer_profile(session, user_id, visit_id),
        "menu": await service.todays_menu(session, visit_id),
        "size_deltas": deltas.size,
        "modifier_deltas": deltas.offerable(),
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
    model = llm.bind_tools(WAITER_TOOLS)

    async def barista(state: BaristaState, config: RunnableConfig) -> dict[str, Any]:
        with node_span("barista") as span:
            # The system message is rebuilt every turn from live state, so the
            # menu, wallet and cart the model sees are never stale.
            messages = [SystemMessage(content=system_prompt(dict(state))), *state["messages"]]
            started = time.monotonic()
            reply = await model.ainvoke(messages, config)
            record_llm_call(span, reply, (time.monotonic() - started) * 1000)
            # Count laps here, record the total at finish. Recording 1 per
            # lap would make the histogram meaningless — what matters is how
            # many laps ONE turn took.
            return {"messages": [reply], "loop_count": state.get("loop_count", 0) + 1}

    return barista


async def refresh(state: BaristaState, config: RunnableConfig) -> dict[str, Any]:
    """After tools run, re-read the state they may have changed.

    Only the cart and wallet — the menu is fixed for the visit.
    """
    with node_span("refresh"):
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


async def run_tools(state: BaristaState, config: RunnableConfig) -> dict[str, Any]:
    """Execute this turn's tool calls **one at a time, in order**.

    Hand-written rather than LangGraph's prebuilt `ToolNode` because that one runs
    the calls concurrently, which would be wrong here twice over:

    1. They share one `AsyncSession`, and SQLAlchemy sessions are not safe for
       concurrent use.
    2. The calls are causally ordered. A model that emits
       `add_to_cart` + `place_order` in one turn means "add it, then charge me";
       run concurrently, `place_order` reads the cart before `add_to_cart` has
       committed and fails with `empty_cart`.

    That was a real bug (`69ac67f`), found by talking to the real model — the
    scripted tests never emitted two calls in a single message. Spec §13.7.

    Nothing in here may raise: a tool that throws (`invalid_arguments`) and a
    tool name the model invented (`unknown_tool`) both come back as ordinary
    envelopes, because the caller is a language model and the turn has to survive
    either.
    """
    last = state["messages"][-1]
    results: list[ToolMessage] = []
    registry = {t.name: t for t in WAITER_TOOLS}

    # Sub-agents need the live menu, cart and wallet to build their own prompts,
    # and they are reached through a tool, so state travels the same way session
    # and visit_id do rather than through a closure.
    config = {
        **(config or {}),
        "configurable": {**((config or {}).get("configurable") or {}), "agent_state": dict(state)},
    }

    steps: dict[str, list] = {}

    with node_span("tools"):
        for call in last.tool_calls:
            payload = await execute_tool_call(call, registry, config)
            # A delegation's own tool calls go to state, not into the message.
            # The ToolMessage's content IS the waiter's context, and a sub-agent
            # that read the menu and added a drink would otherwise re-enter that
            # context on every subsequent turn (spec §13.11).
            if payload.get("steps"):
                steps[call["id"]] = payload.pop("steps")
            else:
                payload.pop("steps", None)
            results.append(
                ToolMessage(
                    content=json.dumps(payload),
                    tool_call_id=call["id"],
                    name=call["name"],
                )
            )

    return {"messages": results, "delegation_steps": steps}


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
    with node_span("finish"):
        # The only place a model going in circles is visible (spec §9.4).
        loop_iterations.record(state.get("loop_count", 0))
        return {}


def build_graph(llm=None, checkpointer=None, barista_llm=None, cashier_llm=None):
    """Wiring only — no business logic in this function.

    Three models, defaulting to the same one. Tests inject three scripted fakes
    so a delegation is as deterministic as any other tool call.
    """
    waiter = llm or build_llm()
    set_models(barista=barista_llm or waiter, cashier=cashier_llm or waiter)

    graph = StateGraph(BaristaState)

    graph.add_node("load_context", load_context)
    graph.add_node("barista", make_barista(waiter))
    graph.add_node("tools", run_tools)
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

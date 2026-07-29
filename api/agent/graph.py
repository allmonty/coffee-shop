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

from agent.instrumentation import (
    loop_iterations,
    node_span,
    record_llm_call,
    record_tool_call,
    record_tool_result,
    tool_malformed,
    tool_span,
)
from agent.llm import build_llm
from agent.prompts import system_prompt
from agent.state import BaristaState
from agent.tools import ALL_TOOLS, TOOLS_BY_NAME
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

    LangGraph's prebuilt `ToolNode` runs them concurrently, which is wrong here
    for two reasons:

    1. They share one `AsyncSession`, and SQLAlchemy sessions are not safe for
       concurrent use.
    2. The calls are causally ordered. A model that emits
       `add_to_cart` + `place_order` in one turn means "add it, then charge me";
       run concurrently, `place_order` reads the cart before `add_to_cart` has
       committed and fails with `empty_cart`.

    Found by talking to the real model — the scripted tests never emitted two
    calls in a single message.

    Nothing in here may raise: a tool that throws (`invalid_arguments`) and a
    tool name the model invented (`unknown_tool`) both come back as ordinary
    envelopes, because the caller is a language model and the turn has to survive
    either.
    """
    last = state["messages"][-1]
    results: list[ToolMessage] = []

    with node_span("tools"):
        for call in last.tool_calls:
            tool = TOOLS_BY_NAME.get(call["name"])
            if tool is None:
                results.append(
                    ToolMessage(
                        content=json.dumps(_unknown_tool(call)),
                        tool_call_id=call["id"],
                        name=call["name"],
                    )
                )
                continue

            with tool_span(call["name"]) as span:
                record_tool_call(call["name"], call["args"])
                try:
                    payload = await tool.ainvoke({**call["args"]}, config)
                except Exception as error:
                    # A malformed tool call must never crash the turn. Small
                    # models drop required arguments constantly; handing the
                    # problem back as an ordinary envelope lets the barista fix
                    # it and retry, which is what it does with any tool error.
                    payload = {
                        "ok": False,
                        "error": "invalid_arguments",
                        "message": (
                            f"That call to {call['name']} was missing something: {error}. "
                            "Check the arguments and try again."
                        ),
                    }
                    tool_malformed.add(1, {"reason": "invalid_arguments"})
                record_tool_result(span, call["name"], payload)

            results.append(
                ToolMessage(
                    content=json.dumps(payload),
                    tool_call_id=call["id"],
                    name=call["name"],
                )
            )

    return {"messages": results}


def _unknown_tool(call: dict[str, Any]) -> dict[str, Any]:
    """An invented tool name is an ordinary tool failure, so treat it as one.

    Two things this must do that the earlier bare `{ok, error}` did not:

    1. **Carry a `message`.** Every envelope does (spec §6.4). Without one the
       model has nothing to work from and invents an explanation for the
       customer — the same failure mode `Result.failure` refuses to allow.
    2. **Leave a trace.** It gets a tool span and counts towards
       `agent.tool.malformed`, whose whole purpose is "unparseable or *invented*
       tool calls" — the invented half was never being counted, so a model
       hallucinating tools looked like a healthy turn on the dashboard.

    The span and metric use the fixed name `unknown`, with the requested name in
    an attribute: the name came from a language model, and putting it in a span
    name or a metric label is unbounded cardinality (§9.3).
    """
    payload = {
        "ok": False,
        "error": "unknown_tool",
        "message": (
            f"There is no {call['name']} tool. The tools you have are: "
            f"{', '.join(TOOLS_BY_NAME)}. Use one of those, or just answer in words."
        ),
    }

    with tool_span("unknown") as span:
        span.set_attribute("tool.requested", call["name"])
        record_tool_call(call["name"], call.get("args") or {})
        record_tool_result(span, "unknown", payload)
        tool_malformed.add(1, {"reason": "unknown_tool"})

    return payload


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


def build_graph(llm=None, checkpointer=None):
    """Wiring only — no business logic in this function."""
    graph = StateGraph(BaristaState)

    graph.add_node("load_context", load_context)
    graph.add_node("barista", make_barista(llm or build_llm()))
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

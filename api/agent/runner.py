"""One conversation turn, as a stream of typed events (spec §7.2).

Both callers use this: the SSE endpoint and the CLI. Keeping the event shape in
one place means the terminal and the browser show the same thing, which is what
makes the CLI a usable debugging surface for the web app.

Domain events are emitted as the tools cause them, not at the end, so the cart
panel updates mid-sentence rather than after the barista stops talking.
"""

from __future__ import annotations

import json
import logging
import uuid
from collections.abc import AsyncIterator
from typing import Any

from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, ToolMessage

from agent.instrumentation import turn_span
from agent.summarize import summarize_visit
from shop import service

logger = logging.getLogger(__name__)

GREETING_PROMPTS = {
    "on_enter": (
        "[The customer has just walked in. Greet them, ask what they'd like, "
        "and mention they can ask to hear the menu.]"
    ),
    # Spelled out because it has to happen, not because the model might like
    # to: the waiter has no end_visit of its own any more, so leaving means one
    # specific call. A backstop below closes the visit if it still does not.
    "go_home": (
        "[The customer has pressed the Go Home button. They are leaving NOW. "
        "Call ring_up with going_home=true. If CURRENT ORDER still has anything "
        "in it, pass its total as quoted_total_cents in the same call so they "
        "pay on the way out. Then say goodbye.]"
    ),
}


async def run_turn(
    *,
    session,
    graph,
    user_id: uuid.UUID,
    visit_id: uuid.UUID,
    message: str | None = None,
    event: str | None = None,
    day: int | None = None,
    summary_llm=None,
) -> AsyncIterator[dict[str, Any]]:
    """Yield `{type, ...}` frames for one turn."""
    text = message if message is not None else GREETING_PROMPTS.get(event or "", "")
    if not text:
        yield {"type": "error", "error": "empty_turn"}
        return

    config = {
        "configurable": {
            "session": session,
            "visit_id": str(visit_id),
            "user_id": str(user_id),
            "thread_id": str(visit_id),
        }
    }

    final_state: dict[str, Any] = {}
    # Tool results are the one frame type that appends rather than replaces, so
    # unlike the cart and wallet they cannot be re-derived from each snapshot.
    #
    # This starts as None rather than an empty set on purpose. `messages` is the
    # whole checkpointed conversation, not this turn's slice, so an empty set
    # would re-report every earlier turn's tool calls as if they had just
    # happened. The first snapshot arrives after `load_context`, before any tool
    # in this turn can have run, so everything in it is history: seed from that.
    reported: set[str] | None = None

    # The turn's root span opens here rather than in the router, because this is
    # where the graph actually runs. The FastAPI request span is already closed
    # by the time starlette starts draining this generator, so node spans opened
    # under it would each become their own root trace — which is exactly what
    # they did before this wrapper existed.
    with turn_span(str(visit_id), str(user_id), day):
        async for kind, payload in graph.astream(
            {"messages": [HumanMessage(content=text)]},
            config=config,
            stream_mode=["messages", "values"],
        ):
            if kind == "messages":
                chunk, _metadata = payload
                # Only the barista's own words. `stream_mode="messages"` also
                # emits ToolMessage chunks, and forwarding those printed raw
                # envelope JSON into the conversation.
                if isinstance(chunk, AIMessageChunk | AIMessage) and chunk.content:
                    yield {"type": "token", "text": chunk.content}

            elif kind == "values":
                final_state = payload
                if reported is None:
                    reported = _tool_call_ids(payload)
                tool_frames = _tool_frames(payload, reported)
                if tool_frames:
                    # Anything the model said in the same message as a tool call
                    # was spoken before it knew the result, and it is about to
                    # say the real thing. Both reach the browser, so without
                    # this the customer reads a guess followed by an answer:
                    # "Latte added.Sure thing! That's a large latte…".
                    #
                    # A prompt rule does not hold this at temperature 0 — the
                    # model narrates its own tool call regardless — so the fix
                    # is here rather than in prose.
                    yield {"type": "reset_reply"}
                for frame in tool_frames:
                    yield frame
                async for frame in _domain_frames(payload):
                    yield frame

    ended = bool(final_state.get("visit_ended"))

    if event == "go_home" and not ended:
        # The button is not a sentence the model has to interpret — that is the
        # whole reason it exists rather than us waiting for someone to type
        # "bye" (§13.1). Leaving is still the customer's decision (§13.3); they
        # have just made it, unambiguously, by pressing the thing labelled
        # Go Home.
        #
        # Before the crew split the barista held `end_visit` and usually called
        # it. Putting a delegation in the way was enough to break it: with an
        # unpaid cart the model says goodbye, calls nothing, and the day never
        # advances — the customer is stuck in the shop with no way out. An
        # unpaid order is simply abandoned, which is what walking out means.
        result = await service.end_visit(session, visit_id, confirmed=True)
        if result.ok:
            ended = True
            logger.info("go_home.forced", extra={"visit_id": str(visit_id)})
            async for frame in _domain_frames(
                {
                    "visit_ended": True,
                    "day": result.data.get("day"),
                    "wallet_cents": result.data.get("wallet_cents"),
                }
            ):
                yield frame
    # How many model round-trips the turn actually cost. Invisible in the
    # conversation, and the only place a model going in circles shows up
    # outside Grafana.
    yield {"type": "turn_stats", "loop_count": final_state.get("loop_count", 0)}
    yield {"type": "done", "visit_ended": ended}

    # After the stream is closed, never before: the customer is already walking
    # out and must not wait on this (spec §6.5.1).
    if ended:
        await summarize_visit(session, user_id, final_state.get("messages", []), llm=summary_llm)


def _tool_call_ids(state: dict[str, Any]) -> set[str]:
    """Every tool result already in the thread — i.e. earlier turns."""
    return {
        message.tool_call_id
        for message in state.get("messages") or []
        if isinstance(message, ToolMessage)
    }


def _tool_frames(state: dict[str, Any], reported: set[str]) -> list[dict[str, Any]]:
    """What the agent actually did this turn, paired call-to-result.

    Derived from `messages` rather than from a state field, because the pairing
    already exists there: every `ToolMessage` carries the `tool_call_id` of the
    `AIMessage.tool_calls` entry that caused it. A call with no result yet is
    skipped and picked up on a later snapshot.

    `steps` is filled by sub-agent delegations, whose own tool calls never enter
    this message list. It stays empty for ordinary tools.
    """
    calls: dict[str, dict[str, Any]] = {}
    for message in state.get("messages") or []:
        if isinstance(message, AIMessage):
            for call in message.tool_calls or []:
                calls[call["id"]] = {"tool": call["name"], "args": call.get("args") or {}}

    frames = []
    for message in state.get("messages") or []:
        if not isinstance(message, ToolMessage):
            continue
        call_id = message.tool_call_id
        if call_id in reported or call_id not in calls:
            continue
        reported.add(call_id)

        envelope = _envelope(message.content)
        # Steps live in state rather than in the envelope, so the waiter's
        # context does not carry every sub-agent's tool calls forward. The two
        # halves are joined here, on the id the walk is already keyed by.
        steps = (state.get("delegation_steps") or {}).get(call_id) or []
        frames.append(
            {
                "type": "tool_result",
                "tool": calls[call_id]["tool"],
                "args": {k: _display(v) for k, v in calls[call_id]["args"].items()},
                "ok": bool(envelope.get("ok")),
                "error": envelope.get("error"),
                "message": envelope.get("message"),
                "agent": envelope.get("agent"),
                "steps": steps,
            }
        )
    return frames


def _display(value: Any, limit: int = 120) -> Any:
    """A tool argument, safe to put on the wire.

    Values keep their JSON type so the panel can render a size as `large` rather
    than `"large"` — this is a display path, not a span attribute, so it does not
    reuse `instrumentation._short`. Only the length is capped: arguments are
    model-written and therefore unbounded.
    """
    if isinstance(value, str) and len(value) > limit:
        return value[:limit] + "…"
    if isinstance(value, list | dict):
        rendered = json.dumps(value)
        return value if len(rendered) <= limit else rendered[:limit] + "…"
    return value


def _envelope(content: Any) -> dict[str, Any]:
    """A ToolMessage's content as a dict, or an empty one if it is not JSON."""
    if isinstance(content, dict):
        return content
    try:
        parsed = json.loads(content)
    except (TypeError, ValueError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


async def _domain_frames(state: dict[str, Any]) -> AsyncIterator[dict[str, Any]]:
    """Turn a state snapshot into the UI events it implies."""
    if "cart" in state:
        yield {"type": "cart_updated", **(state["cart"] or {})}
    if "wallet_cents" in state:
        yield {"type": "wallet_updated", "wallet_cents": state["wallet_cents"]}
    if state.get("visit_ended"):
        yield {
            "type": "visit_ended",
            "day": state.get("day"),
            "wallet_cents": state.get("wallet_cents"),
        }

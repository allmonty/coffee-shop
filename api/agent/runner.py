"""One conversation turn, as a stream of typed events (spec §7.2).

Both callers use this: the SSE endpoint and the CLI. Keeping the event shape in
one place means the terminal and the browser show the same thing, which is what
makes the CLI a usable debugging surface for the web app.

Domain events are emitted as the tools cause them, not at the end, so the cart
panel updates mid-sentence rather than after the barista stops talking.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from typing import Any

from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage

from agent.graph import build_graph

# Tools whose success changes something the UI shows.
CART_TOOLS = {"add_to_cart", "remove_from_cart", "change_size"}

GREETING_PROMPTS = {
    "on_enter": (
        "[The customer has just walked in. Greet them, ask what they'd like, "
        "and mention they can ask to hear the menu.]"
    ),
    "go_home": "[The customer has pressed the Go Home button. They are leaving now.]",
}


async def run_turn(
    *,
    session,
    graph,
    user_id: uuid.UUID,
    visit_id: uuid.UUID,
    message: str | None = None,
    event: str | None = None,
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
    async for kind, payload in graph.astream(
        {"messages": [HumanMessage(content=text)]},
        config=config,
        stream_mode=["messages", "values"],
    ):
        if kind == "messages":
            chunk, _metadata = payload
            # Only the barista's own words. `stream_mode="messages"` also emits
            # ToolMessage chunks, and forwarding those printed raw envelope JSON
            # into the conversation.
            if isinstance(chunk, AIMessageChunk | AIMessage) and chunk.content:
                yield {"type": "token", "text": chunk.content}

        elif kind == "values":
            final_state = payload
            async for frame in _domain_frames(payload):
                yield frame

    yield {
        "type": "done",
        "visit_ended": bool(final_state.get("visit_ended")),
    }


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


def build_runtime_graph(llm=None, checkpointer=None):
    return build_graph(llm=llm, checkpointer=checkpointer)

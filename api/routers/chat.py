"""The SSE chat endpoint — the only streaming code in the project (spec §7.2).

SSE rather than WebSocket: the interaction is strictly request → streamed
response, the browser's only upstream message is a plain POST, and SSE
reconnects on its own. A WebSocket would buy nothing.

One stream carries both the prose and the state changes, which is what keeps
their ordering unambiguous — the cart panel updates mid-sentence rather than
after the barista finishes talking.
"""

from __future__ import annotations

import json
import uuid

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from agent.graph import build_graph
from agent.runner import run_turn
from db import SessionLocal
from shop.models import User, Visit
from shop.schemas import ChatRequest

router = APIRouter(prefix="/api", tags=["chat"])

# One compiled graph for the process. Compiling per request would re-create the
# whole StateGraph on every message for no benefit.
_graph = None
_checkpointer = None


def set_checkpointer(saver) -> None:
    """Install the conversation store. Called once, from the app lifespan.

    Without it the graph runs with no checkpointer and every POST starts from an
    empty state: `{"messages": [HumanMessage(...)]}` and nothing else. The
    barista then cannot see the sentence before ("a latte" → "which size?" →
    "large" loses the latte), `load_context` re-runs every turn because `menu` is
    never already in state, and the `upsell_used` / `size_declines` backstops
    reset between messages.

    The CLI has always opened one itself (`scripts/shop_cli.py`), which is how
    the terminal came to work while the browser quietly did not.
    """
    global _checkpointer, _graph
    _checkpointer = saver
    # Drop the compiled graph so a checkpointer installed after the first request
    # cannot be silently ignored.
    _graph = None


def get_graph():
    global _graph
    if _graph is None:
        _graph = build_graph(checkpointer=_checkpointer)
    return _graph


def get_session_factory():
    """Indirection so tests can supply their own engine.

    This endpoint cannot use the request-scoped `get_session` dependency: the
    session has to outlive the response, because tools run inside it while the
    stream is still open. FastAPI closes dependency sessions when the handler
    returns, which is before the first token.
    """
    return SessionLocal


def sse(frame: dict) -> str:
    return f"data: {json.dumps(frame)}\n\n"


@router.post("/chat")
async def chat(body: ChatRequest):
    try:
        visit_id = uuid.UUID(body.visit_id)
    except ValueError:
        raise HTTPException(status_code=422, detail={"error": "invalid_visit_id"}) from None

    session_factory = get_session_factory()

    async def stream():
        # A session per turn, held for the duration of the stream: the tools run
        # inside it and the transaction must outlive the first token.
        async with session_factory() as session:
            visit = await session.get(Visit, visit_id)
            if visit is None:
                yield sse({"type": "error", "error": "unknown_visit"})
                return
            user = await session.get(User, visit.user_id)
            assert user is not None

            try:
                async for frame in run_turn(
                    session=session,
                    graph=get_graph(),
                    user_id=user.id,
                    visit_id=visit_id,
                    message=body.message,
                    event=body.event,
                    # Already loaded above, so tagging the turn with the in-game
                    # day costs no extra query.
                    day=visit.day,
                ):
                    yield sse(frame)
            except Exception as error:  # pragma: no cover - surfaced to the UI
                yield sse({"type": "error", "error": "agent_failed", "detail": str(error)})

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            # nginx buffers proxied responses by default, which would deliver
            # every token in one lump at the end (spec §10).
            "X-Accel-Buffering": "no",
        },
    )

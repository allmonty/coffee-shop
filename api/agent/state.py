"""The graph's state (spec §6.2).

`messages` uses the `add_messages` reducer, which is the one piece of LangGraph
magic worth understanding early: nodes return *partial* state, and the reducer
decides how it merges. For messages that means append-with-dedup-by-id rather
than replace — which is why a node can return one new message and the
conversation keeps the rest.

Everything else here is plain replace-on-write.
"""

from __future__ import annotations

from typing import Annotated, Any, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


class BaristaState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]

    user_id: str
    visit_id: str

    # Loaded once per visit by load_context, then re-rendered into the prompt
    # every turn (spec §6.6).
    customer_profile: dict[str, Any] | None
    menu: list[dict[str, Any]]
    size_deltas: dict[str, int]
    modifier_deltas: dict[str, int]
    wallet_cents: int
    cart: dict[str, Any]
    day: int

    # Soft-rule backstops. The prompt states the rules; these let us render what
    # has already happened and measure violations rather than assume compliance.
    upsell_used: bool
    size_offers: dict[str, bool]
    size_declines: int

    # A delegation's own tool calls, keyed by the tool_call_id of the delegation
    # that produced them. Kept HERE rather than inside the envelope on purpose:
    # the envelope becomes the ToolMessage's content, so carrying steps in it
    # would push every sub-agent's calls into the waiter's context on every
    # later turn — undoing the context saving that justified agents-as-tools in
    # the first place (spec §13.11). The SSE layer joins the two back together.
    delegation_steps: dict[str, list]

    # barista→tools laps in the current turn; recorded as a histogram at finish.
    loop_count: int

    visit_ended: bool

"""Spans and metrics for the agent (spec §9.3, §9.4).

The trace of one turn IS the agent loop, drawn:

    agent.turn
    ├── graph.node.load_context
    ├── graph.node.barista      → gen_ai.chat  (LLM call #1)
    ├── graph.node.tools        → tool.add_to_cart
    ├── graph.node.barista      → gen_ai.chat  (LLM call #2)
    └── graph.node.finish

Reading that once tells you things a log line never will: that a turn cost two
model round-trips, where the context tokens went, that the database was never
the problem.

Two conventions worth keeping:

- Span names are stable and low-cardinality. Identifiers go in attributes.
- **A failed tool sets the tool span to error but must NOT fail the parent
  turn.** "insufficient funds" is a normal outcome the agent recovers from;
  marking it a request error makes every dashboard lie.
"""

from __future__ import annotations

from contextlib import contextmanager

from opentelemetry.trace import Status, StatusCode

from telemetry import get_meter, get_tracer

tracer = get_tracer("coffee-shop.agent")
_meter = get_meter("coffee-shop.agent")

turn_duration = _meter.create_histogram(
    "agent.turn.duration", unit="ms", description="End-to-end latency of one turn."
)
loop_iterations = _meter.create_histogram(
    "agent.loop.iterations",
    description="barista→tools laps per turn. A model going in circles shows up here first.",
)
tool_calls = _meter.create_counter(
    "agent.tool.calls", description="Tool invocations by name and outcome."
)
tool_duration = _meter.create_histogram("agent.tool.duration", unit="ms")
tool_malformed = _meter.create_counter(
    "agent.tool.malformed", description="Unparseable or invented tool calls."
)
offmenu_requests = _meter.create_counter(
    "agent.offmenu_request",
    description="unknown_item vs not_available_today — do they want what we do not sell, "
    "or what we sold out of?",
)
guard_rejections = _meter.create_counter(
    "agent.guard.rejections",
    description="Domain refusing a call the model should not have made. Should sit at zero.",
)
size_clarifications = _meter.create_counter(
    "agent.size_clarifications",
    description="Drinks ordered without a size. Falls as the profile learns someone.",
)
llm_tokens = _meter.create_counter("llm.tokens", description="Prompt and completion tokens.")
llm_duration = _meter.create_histogram("llm.request.duration", unit="ms")


@contextmanager
def turn_span(visit_id: str, user_id: str):
    with tracer.start_as_current_span("agent.turn") as span:
        span.set_attribute("visit_id", visit_id)
        span.set_attribute("user_id", user_id)
        yield span


@contextmanager
def node_span(name: str):
    with tracer.start_as_current_span(f"graph.node.{name}") as span:
        yield span


@contextmanager
def tool_span(name: str):
    """A failed tool marks its own span, never the turn."""
    with tracer.start_as_current_span(f"tool.{name}") as span:
        span.set_attribute("tool.name", name)
        yield span


def record_tool_result(span, name: str, payload: dict) -> None:
    ok = bool(payload.get("ok"))
    error = payload.get("error")

    span.set_attribute("tool.ok", ok)
    if error:
        span.set_attribute("tool.error", error)
        # Error status on the tool span only. The parent turn stays unset, so
        # "insufficient funds" does not read as a broken request.
        span.set_status(Status(StatusCode.ERROR, error))

    tool_calls.add(1, {"tool": name, "ok": str(ok).lower()})

    if error in {"unknown_item", "not_available_today"}:
        offmenu_requests.add(1, {"kind": error})
    if error in {"total_mismatch", "confirmation_required"}:
        guard_rejections.add(1, {"guard": error})
    if error == "size_required":
        size_clarifications.add(1)


def record_llm_call(span, response) -> None:
    """OTel GenAI semantic conventions (spec §9.3).

    Still marked experimental upstream and they do shift between releases; pin
    the instrumentation and expect one rename. Using them still beats inventing
    private attribute names.
    """
    usage = getattr(response, "usage_metadata", None) or {}
    span.set_attribute("gen_ai.system", "ollama")
    span.set_attribute("gen_ai.operation.name", "chat")

    if usage:
        span.set_attribute("gen_ai.usage.input_tokens", usage.get("input_tokens", 0))
        span.set_attribute("gen_ai.usage.output_tokens", usage.get("output_tokens", 0))
        llm_tokens.add(usage.get("input_tokens", 0), {"type": "input"})
        llm_tokens.add(usage.get("output_tokens", 0), {"type": "output"})

    calls = getattr(response, "tool_calls", None) or []
    span.set_attribute("gen_ai.response.tool_calls", len(calls))

    invalid = getattr(response, "invalid_tool_calls", None) or []
    if invalid:
        tool_malformed.add(len(invalid), {"reason": "unparseable"})
        span.set_attribute("gen_ai.response.invalid_tool_calls", len(invalid))

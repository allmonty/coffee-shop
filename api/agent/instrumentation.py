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

import json
import logging
import time
from contextlib import contextmanager

from opentelemetry.trace import Status, StatusCode

from telemetry import correlation_scope, get_meter, get_tracer

tracer = get_tracer("coffee-shop.agent")
_meter = get_meter("coffee-shop.agent")

# Structured events, not prose (spec §9.5). Every record emitted from inside a
# span carries that span's trace_id, which is the whole point: the log panel in
# Grafana is a way *into* a trace, not a separate story about the same turn.
logger = logging.getLogger("coffee_shop.agent")

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
delegations = _meter.create_counter(
    "agent.delegations",
    description="Times the waiter handed a job to a sub-agent. Label `to` is a "
    "fixed set (barista|cashier) — never a model-written string (spec §13.11).",
)
delegation_laps = _meter.create_histogram(
    "agent.delegation.laps",
    description="Tool laps inside one delegation. Makes a runaway sub-agent "
    "visible the way agent.loop.iterations makes a runaway waiter visible.",
)

llm_tokens = _meter.create_counter("llm.tokens", description="Prompt and completion tokens.")
llm_duration = _meter.create_histogram("llm.request.duration", unit="ms")


@contextmanager
def turn_span(visit_id: str, user_id: str, day: int | None = None):
    """The root span of a turn. Every node and tool span hangs off this one.

    It has to be entered where the graph actually runs — inside the streaming
    generator — and not in the request handler. FastAPI's request span is closed
    before the first token is produced, so a turn parented there would be
    orphaned anyway.

    The correlation scope wraps the span rather than the other way round, so the
    root span itself is tagged by `on_start` like every other span. One visit is
    one in-game day, which is what makes `visit_id` the key for "the whole day"
    without any turn having to know about the turns around it.
    """
    started = time.monotonic()
    with (
        correlation_scope(visit_id=visit_id, user_id=user_id, day=day),
        tracer.start_as_current_span("agent.turn") as span,
    ):
        logger.info("turn.start")
        try:
            yield span
        finally:
            # In `finally` so an abandoned turn — the customer closing the tab
            # mid-stream — is still measured. Dropping those would quietly bias
            # the latency panel towards the turns that went well.
            elapsed = (time.monotonic() - started) * 1000
            turn_duration.record(elapsed)
            logger.info("turn.end", extra={"duration_ms": round(elapsed)})


@contextmanager
def summarize_span(user_id: str, model: str):
    """The visit-summarization pass (spec §6.5.1).

    Its own root span, not a child of the turn: it runs *after* the SSE stream
    closes, by which point `agent.turn` is already ended. It is a separate agent
    task on its own budget, and the trace should say so.

    `model` is an attribute because the summarizer can run on a smaller model
    than the barista (§13.10), and "did the small one actually run" is otherwise
    unanswerable.
    """
    with tracer.start_as_current_span("agent.summarize_visit") as span:
        span.set_attribute("gen_ai.request.model", model)
        span.set_attribute("user.id", user_id)
        logger.info("summarize.start", extra={"summary_model": model})
        yield span


@contextmanager
def node_span(name: str):
    with tracer.start_as_current_span(f"graph.node.{name}") as span:
        logger.info("graph.node", extra={"node": name})
        yield span


@contextmanager
def tool_span(name: str):
    """A failed tool marks its own span, never the turn."""
    with tracer.start_as_current_span(f"tool.{name}") as span:
        span.set_attribute("tool.name", name)
        yield span


def record_tool_call(name: str, args: dict) -> None:
    """Log the invocation before it runs, so a tool that hangs still leaves a trail.

    Note `tool_args`, not `args`: `extra` keys collide with the reserved
    attributes on `logging.LogRecord`, and `args` is one of them — logging
    raises rather than shadowing it.
    """
    logger.info("tool.call", extra={"tool": name, "tool_args": _short(args)})


def record_tool_result(span, name: str, payload: dict) -> None:
    ok = bool(payload.get("ok"))
    error = payload.get("error")

    span.set_attribute("tool.ok", ok)
    if error:
        span.set_attribute("tool.error", error)
        # Error status on the tool span only. The parent turn stays unset, so
        # "insufficient funds" does not read as a broken request.
        span.set_status(Status(StatusCode.ERROR, error))

    # A rejected order is an ordinary outcome, so it logs at WARNING, not ERROR —
    # same reasoning as the span status above. Dashboards that treat
    # "insufficient funds" as a fault lie about the health of the service.
    logger.log(
        logging.WARNING if error else logging.INFO,
        "tool.result",
        extra={"tool": name, "ok": ok, "error": error or "", "envelope": _short(payload)},
    )

    tool_calls.add(1, {"tool": name, "ok": str(ok).lower()})

    if error in {"unknown_item", "not_available_today"}:
        offmenu_requests.add(1, {"kind": error})
    if error in {"total_mismatch", "confirmation_required"}:
        guard_rejections.add(1, {"guard": error})
    if error == "size_required":
        size_clarifications.add(1)


def record_llm_call(span, response, duration_ms: float | None = None) -> None:
    """OTel GenAI semantic conventions (spec §9.3).

    Still marked experimental upstream and they do shift between releases; pin
    the instrumentation and expect one rename. Using them still beats inventing
    private attribute names.
    """
    usage = getattr(response, "usage_metadata", None) or {}
    span.set_attribute("gen_ai.system", "ollama")
    span.set_attribute("gen_ai.operation.name", "chat")

    if duration_ms is not None:
        llm_duration.record(duration_ms)
        span.set_attribute("gen_ai.request.duration_ms", round(duration_ms))

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
        logger.warning("llm.invalid_tool_calls", extra={"count": len(invalid)})

    logger.info(
        "llm.response",
        extra={
            "input_tokens": usage.get("input_tokens", 0),
            "output_tokens": usage.get("output_tokens", 0),
            "tool_calls": len(calls),
            "duration_ms": round(duration_ms) if duration_ms is not None else 0,
        },
    )


def _short(value, limit: int = 800) -> str:
    """Log payloads as JSON, truncated.

    `message` fields are barista prose and can run long; an unbounded log line
    is how a log volume fills up overnight.
    """
    try:
        text = json.dumps(value, default=str)
    except (TypeError, ValueError):  # pragma: no cover - defensive
        text = str(value)
    return text if len(text) <= limit else text[:limit] + "…"

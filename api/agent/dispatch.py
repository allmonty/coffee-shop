"""Running one tool call, safely, for whoever asked.

Factored out of `graph.run_tools` when the barista and cashier sub-agents grew
their own tool loops (spec §13.11). The two rules it enforces are invariants, and
invariants that exist in two copies drift: a malformed call and an invented tool
name must come back as ordinary `{ok, error, message}` envelopes, never as
exceptions, no matter which agent made the call.

`registry` is passed in rather than imported because the whole point of the
three roles is that they see different tools — the waiter has no `place_order`,
and the cashier's tool list is not in the waiter's prompt at all.
"""

from __future__ import annotations

from typing import Any

from agent.instrumentation import (
    record_tool_call,
    record_tool_result,
    tool_malformed,
    tool_span,
)


async def execute_tool_call(
    call: dict[str, Any],
    registry: dict[str, Any],
    config: Any,
) -> dict[str, Any]:
    """The envelope for one tool call. Never raises."""
    tool = registry.get(call["name"])
    if tool is None:
        return unknown_tool(call, registry)

    extra = set(call.get("args") or {}) - set(tool.args)
    if extra:
        # LangChain drops an argument the tool does not declare, silently, and
        # runs the call without it. That turns "a large latte with oat milk"
        # into a plain large latte and reports success — a silent wrong order,
        # and the worst of the three possible outcomes.
        #
        # It matters most where a schema difference IS the design: the waiter's
        # add_to_cart has no `modifiers`, precisely so extras have to go to the
        # barista (spec §13.11). Dropping the argument would defeat that
        # silently rather than sending the model to the right role.
        return _malformed(
            call,
            f"{call['name']} does not take {', '.join(sorted(extra))}. "
            f"It takes: {', '.join(sorted(tool.args)) or 'no arguments'}. "
            "Use the right tool for what you are trying to do.",
        )

    with tool_span(call["name"]) as span:
        record_tool_call(call["name"], call.get("args") or {})
        try:
            payload = await tool.ainvoke({**(call.get("args") or {})}, config)
        except Exception as error:
            # A malformed tool call must never crash the turn. Small models drop
            # required arguments constantly; handing the problem back as an
            # ordinary envelope lets the caller fix it and retry, which is what
            # it does with any other tool error.
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

    return payload


def _malformed(call: dict[str, Any], message: str) -> dict[str, Any]:
    """A bad call, reported the way every other tool failure is."""
    payload = {"ok": False, "error": "invalid_arguments", "message": message}
    with tool_span(call["name"]) as span:
        record_tool_call(call["name"], call.get("args") or {})
        record_tool_result(span, call["name"], payload)
        tool_malformed.add(1, {"reason": "invalid_arguments"})
    return payload


def unknown_tool(call: dict[str, Any], registry: dict[str, Any]) -> dict[str, Any]:
    """An invented tool name is an ordinary tool failure, so treat it as one.

    Two things this must do that a bare `{ok, error}` did not:

    1. **Carry a `message`.** Every envelope does (spec §6.4). Without one the
       model has nothing to work from and invents an explanation for the
       customer — the same failure mode `Result.failure` refuses to allow. The
       message lists the tools that *do* exist, so the next move is a correction
       rather than another guess.
    2. **Leave a trace.** It gets a tool span and counts towards
       `agent.tool.malformed`, whose whole purpose is "unparseable or *invented*
       tool calls".

    The span and metric use the fixed name `unknown`, with the requested name in
    an attribute: that name came from a language model, and putting model-written
    text in a span name or a metric label is unbounded cardinality (§9.3).
    """
    payload = {
        "ok": False,
        "error": "unknown_tool",
        "message": (
            f"There is no {call['name']} tool. The tools you have are: "
            f"{', '.join(registry)}. Use one of those, or just answer in words."
        ),
    }

    with tool_span("unknown") as span:
        span.set_attribute("tool.requested", call["name"])
        record_tool_call(call["name"], call.get("args") or {})
        record_tool_result(span, "unknown", payload)
        tool_malformed.add(1, {"reason": "unknown_tool"})

    return payload

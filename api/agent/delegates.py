"""The barista and the cashier, as sub-agents the waiter calls like tools.

Three roles, one voice (spec §13.11). Sam the waiter fronts every conversation;
Mo behind the machine and Val on the till are LLM sub-agents reached through
`ask_barista` and `ring_up`. The rejected alternative was three peers with
model-driven handoff, which costs 3-5 local inferences per turn against today's
1-2 — a direct reversal of the §6.6 decision that deleted a *single* extra
inference.

What agents-as-tools buys, beyond the role split:

- **A sub-agent's schemas never enter the waiter's prompt.** That is what makes
  `change_modifiers` affordable: it would not earn its schema tokens on every
  inference, but it costs nothing sitting in Mo's toolbox.
- **Every invariant carries over untouched.** A delegation returns the same
  `{ok, error, message}` envelope, so `run_tools`, the malformed-call path,
  `tool.*` spans and `agent.tool.calls` all work with no changes at all.
- **One conversation in the checkpointer.** Sub-agent messages are built fresh
  per delegation and thrown away; only the envelope reaches the waiter, so its
  context does not grow three times as fast.

Their own tool calls are reported separately in `steps`, which the SSE layer
turns into the nested rows of the "What Sam did" panel — otherwise a delegation
would be an opaque box in the one surface meant to show what happened.
"""

from __future__ import annotations

import json
import time
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool

from agent.dispatch import execute_tool_call
from agent.instrumentation import delegation_laps, delegations, node_span, record_llm_call
from agent.prompts import barista_prompt, cashier_prompt
from agent.tools import BARISTA_TOOLS, WAITER_CART_TOOLS, _context, cashier_tools

# A sub-agent gets three laps: enough to read the menu, act, and correct one
# mistake. Beyond that it is looping, and the waiter is better placed to ask the
# customer than the sub-agent is to keep guessing.
MAX_LAPS = 3

# Injected by the graph so tests can script the sub-agents. Set together with the
# waiter's model in `build_graph`; a delegation with no model configured returns
# an envelope rather than raising, like every other tool failure.
_MODELS: dict[str, Any] = {"barista": None, "cashier": None}


def set_models(*, barista=None, cashier=None) -> None:
    _MODELS["barista"] = barista
    _MODELS["cashier"] = cashier


async def _run_subagent(
    *,
    role: str,
    system: str,
    request: str,
    tools: list,
    config: RunnableConfig,
) -> dict[str, Any]:
    """One delegation: a small tool loop that collapses to a single envelope.

    Never raises, for the same reason `run_tools` never raises — the caller is a
    language model and the turn has to survive whatever it does.
    """
    model = _MODELS.get(role)
    if model is None:
        return _envelope(
            role,
            ok=False,
            error="no_model",
            message=f"The {role} is not in today — handle it yourself.",
        )

    registry = {t.name: t for t in tools}
    messages: list[Any] = [SystemMessage(content=system), HumanMessage(content=request)]
    steps: list[dict[str, Any]] = []

    delegations.add(1, {"to": role})
    with node_span(f"delegate.{role}") as span:
        bound = model.bind_tools(tools)
        for lap in range(1, MAX_LAPS + 1):
            started = time.monotonic()
            reply: AIMessage = await bound.ainvoke(messages, config)
            record_llm_call(span, reply, (time.monotonic() - started) * 1000, agent=role)
            messages.append(reply)

            if not reply.tool_calls:
                delegation_laps.record(lap, {"to": role})
                # A sub-agent that has finished talking has not necessarily
                # succeeded, and fluent prose about a refusal reads exactly like
                # prose about a success. What actually happened comes from the
                # steps, not from the words — see _outcome.
                return _outcome(role, str(reply.content), steps)

            # Sequential, in emitted order, for exactly the reasons run_tools is
            # (§13.7): one shared AsyncSession, and calls that are causally
            # ordered.
            for call in reply.tool_calls:
                payload = await execute_tool_call(call, registry, config, agent=role)
                steps.append(
                    {
                        "tool": call["name"],
                        "args": call.get("args") or {},
                        "ok": bool(payload.get("ok")),
                        "error": payload.get("error"),
                        "message": payload.get("message"),
                        "agent": role,
                        "steps": [],
                    }
                )
                messages.append(
                    ToolMessage(
                        content=json.dumps(payload),
                        tool_call_id=call["id"],
                        name=call["name"],
                    )
                )

        delegation_laps.record(MAX_LAPS, {"to": role})
        span.set_attribute("delegation.hit_lap_cap", True)
        return _envelope(
            role,
            ok=False,
            error="delegation_incomplete",
            message=_STUCK[role],
            steps=steps,
        )


_STUCK = {
    "barista": "Mo's in the weeds and hasn't finished that one — check with the customer.",
    "cashier": "Val's still counting — the till hasn't settled. Ask the customer to hold on.",
}


def _outcome(role: str, message: str, steps: list[dict[str, Any]]) -> dict[str, Any]:
    """What a finished delegation actually achieved.

    A single `ok` is not enough on the money path, and getting it wrong is
    misleading in both directions. "Any failed step fails the delegation" made a
    charge that HAD gone through look like a failure as soon as Val also tried
    something else, and Sam told the customer their payment had not worked.
    "Any successful step succeeds it" would be worse — Sam would announce an
    order nobody paid for.

    So the two things Sam must not guess about are reported as facts, and `ok`
    means only "something worked", never "the money moved". The waiter's rule
    keys on `charged`, not on `ok`.
    """
    done = {step["tool"] for step in steps if step["ok"]}
    failed = [step for step in steps if not step["ok"]]

    envelope = _envelope(
        role,
        ok=bool(done) or not failed,
        error=failed[-1]["error"] if failed else None,
        message=message,
        steps=steps,
    )
    if role == "cashier":
        envelope["charged"] = "charge_the_customer" in done
        envelope["visit_ended"] = "send_them_home" in done
    return envelope


def _envelope(role: str, **fields) -> dict[str, Any]:
    return {"agent": role, "steps": [], **fields}


@tool
async def ask_barista(request: str, config: RunnableConfig) -> dict:
    """Hand a drink order to Mo behind the machine, in the customer's own words.

    Use this for anything involving extras (oat or almond milk, an extra shot),
    for a drink request you cannot map to an exact menu name, and for questions
    about what is in a drink. Mo knows the extras vocabulary; you do not need to.

    Pass what the customer actually said. If Mo needs to know something only the
    customer can answer, you get back a question to read out.
    """
    return await _run_subagent(
        role="barista",
        system=barista_prompt(_state(config)),
        request=request,
        tools=BARISTA_TOOLS,
        config=config,
    )


@tool
async def ring_up(
    request: str,
    config: RunnableConfig,
    quoted_total_cents: int | None = None,
    going_home: bool = False,
) -> dict:
    """Send the customer to Val on the till. The ONLY way to take money.

    request: what is happening, in your own words.
    quoted_total_cents: the exact total you just said out loud, when they are
      paying. Leave it out if nobody is paying.
    going_home: true only if the customer said they are leaving.

    Val decides whether to charge and whether to close out; you decide what was
    quoted and what was said.
    """
    session, visit_id = _context(config)
    # The quoted total and "they said they're leaving" are the waiter's
    # observations, so they travel in config where the cashier's model cannot
    # rewrite them — see the note above CASHIER_TOOLS.
    sub_config = {
        **(config or {}),
        "configurable": {
            **((config or {}).get("configurable") or {}),
            "quoted_total_cents": quoted_total_cents,
            "going_home": going_home,
        },
    }
    del session, visit_id  # _context validated them; the tools re-read for themselves.
    return await _run_subagent(
        role="cashier",
        system=cashier_prompt(_state(sub_config), quoted_total_cents, going_home),
        request=request,
        # Scoped to what Sam actually authorised — see cashier_tools().
        tools=cashier_tools(quoted_total_cents=quoted_total_cents, going_home=going_home),
        config=sub_config,
    )


def _state(config: RunnableConfig) -> dict[str, Any]:
    """The live state a sub-agent needs, handed down by the waiter's graph node."""
    return ((config or {}).get("configurable") or {}).get("agent_state") or {}


WAITER_TOOLS = [*WAITER_CART_TOOLS, ask_barista, ring_up]

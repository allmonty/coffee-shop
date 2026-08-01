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


def _model(config: RunnableConfig, role: str):
    """The sub-agent's model, injected per-graph by `run_tools`.

    Not a module global: that is shared by every graph in the process, so the
    CLI running beside the web app — or one test after another — would silently
    rebind the other's sub-agents. Same reasoning as session and visit_id in
    agent/tools.py.
    """
    return (((config or {}).get("configurable") or {}).get("models") or {}).get(role)


async def _run_subagent(
    *,
    role: str,
    system: str,
    request: str,
    tools: list,
    config: RunnableConfig,
    finishes_when: set[str] | None = None,
) -> dict[str, Any]:
    """One delegation: a small tool loop that collapses to a single envelope.

    Never raises, for the same reason `run_tools` never raises — the caller is a
    language model and the turn has to survive whatever it does.
    """
    model = _model(config, role)
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
    attempted: set[tuple[str, str]] = set()

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
                # A sub-agent repeating a call it has already made, verbatim, is
                # looping — not working. Measured against the real model: Mo
                # called add_to_cart twice and put the drink in the cart twice;
                # Val charged twice, the second failing with empty_cart because
                # the first had emptied it.
                #
                # Refusing the repeat rather than capping the laps keeps a
                # genuinely multi-part job working — "a latte and a flat white,
                # both oat" is two DIFFERENT add_to_cart calls and still runs.
                fingerprint = (call["name"], json.dumps(call.get("args") or {}, sort_keys=True))
                if fingerprint in attempted:
                    payload = {
                        "ok": False,
                        "error": "already_done",
                        "message": (
                            f"You already called {call['name']} with exactly those "
                            "arguments and it is done. Do not repeat it — say what "
                            "you have done, or do the next different thing."
                        ),
                    }
                else:
                    attempted.add(fingerprint)
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
                if finishes_when and finishes_when <= {s["tool"] for s in steps if s["ok"]}:
                    # Everything the waiter authorised has now succeeded, so
                    # there is nothing left to decide. Left to run on, Val
                    # called charge_the_customer a second time, hit empty_cart
                    # because the first call had emptied it, and then narrated
                    # THAT — telling a customer their order was empty
                    # immediately after they paid for it.
                    #
                    # The reply is the successful action's own message rather
                    # than the model's summary of it: domain messages are
                    # already written to be read aloud (§6.4), and one fewer
                    # lap is one fewer local inference.
                    delegation_laps.record(lap, {"to": role})
                    done = [s for s in steps if s["ok"] and s["tool"] in finishes_when]
                    return _outcome(role, done[-1]["message"] or "", steps)

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

    if quoted_total_cents is None and not going_home:
        # Val is only given the tools the waiter authorised, so a ring_up with
        # neither a quoted total nor going_home hands over no authority at all:
        # Val can read the cart and say something, and nothing can happen. Seen
        # against the real model — the delegation came back with zero steps, the
        # customer neither charged nor sent home, and Sam with nothing to act on.
        #
        # Refused here rather than spending an inference discovering it, and the
        # message says which of the two is missing.
        return _envelope(
            "cashier",
            ok=False,
            error="nothing_to_do",
            message=(
                "Val needs to know what is happening. Pass quoted_total_cents "
                "with the total you said out loud if they are paying, or "
                "going_home=true if they are leaving, or both."
            ),
            charged=False,
            visit_ended=False,
        )

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
        # Scoped to what Sam actually authorised — see cashier_tools() ...
        tools=cashier_tools(quoted_total_cents=quoted_total_cents, going_home=going_home),
        # ... and finished the moment all of it has worked.
        finishes_when={
            *(["charge_the_customer"] if quoted_total_cents is not None else []),
            *(["send_them_home"] if going_home else []),
        },
        config=sub_config,
    )


def _state(config: RunnableConfig) -> dict[str, Any]:
    """The live state a sub-agent needs, handed down by the waiter's graph node."""
    return ((config or {}).get("configurable") or {}).get("agent_state") or {}


WAITER_TOOLS = [*WAITER_CART_TOOLS, ask_barista, ring_up]

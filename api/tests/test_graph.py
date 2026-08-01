"""The agent loop, tested with a scripted model — no Ollama, CI-safe.

The assertion that matters most is `test_tool_call_loops_back_to_the_barista`:
that cycle is the agent. Everything else in the graph is setup and teardown.
"""

import json
import uuid

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from sqlalchemy import select

from agent.graph import build_graph, route_after_barista
from shop.models import MenuItem, VisitMenuItem
from shop.seed import seed_catalog
from shop.service import enter
from tests.fakes import FakeToolCallingModel, says, tool_call, tool_calls

PINNED = ["Latte", "Filter Coffee", "Chocolate Chip Cookie", "Croissant"]


@pytest.fixture
async def shop(session):
    await seed_catalog(session)
    entered = await enter(session, "Allan")
    visit_id = uuid.UUID(entered.data["visit_id"])
    user_id = uuid.UUID(entered.data["user_id"])

    await session.execute(
        VisitMenuItem.__table__.delete().where(VisitMenuItem.visit_id == visit_id)
    )
    items = (await session.scalars(select(MenuItem).where(MenuItem.name.in_(PINNED)))).all()
    for item in items:
        session.add(VisitMenuItem(visit_id=visit_id, menu_item_id=item.id))
    await session.commit()

    def run(script, text="hello", barista=None, cashier=None):
        """Drive one turn.

        `barista` and `cashier` script the sub-agents (spec §13.11). Left out,
        they share the waiter's script — fine for turns that never delegate, and
        it keeps every pre-existing test readable.
        """
        graph = build_graph(
            llm=FakeToolCallingModel(script),
            barista_llm=FakeToolCallingModel(barista) if barista is not None else None,
            cashier_llm=FakeToolCallingModel(cashier) if cashier is not None else None,
        )
        return graph.ainvoke(
            {"messages": [HumanMessage(content=text)]},
            config={
                "configurable": {
                    "session": session,
                    "visit_id": str(visit_id),
                    "user_id": str(user_id),
                }
            },
        )

    return run, session, visit_id


def test_route_sends_tool_calls_to_the_tool_node():
    state = {"messages": [tool_call("get_cart")]}
    assert route_after_barista(state) == "tools"


def test_route_sends_plain_replies_to_finish():
    state = {"messages": [AIMessage(content="what can I get you?")]}
    assert route_after_barista(state) == "finish"


async def test_a_plain_reply_ends_the_turn(shop):
    run, _, _ = shop

    result = await run([says("What can I get you?")])

    assert result["messages"][-1].content == "What can I get you?"


async def test_tool_call_loops_back_to_the_barista(shop):
    """The cycle that IS the agent: barista → tools → barista."""
    run, _, _ = shop

    result = await run(
        [
            tool_call("add_to_cart", item_name="Latte", quantity=1, size="large"),
            says("One large latte, coming up."),
        ]
    )

    kinds = [type(m).__name__ for m in result["messages"]]
    assert "ToolMessage" in kinds
    assert result["messages"][-1].content == "One large latte, coming up."


async def test_load_context_puts_todays_menu_into_state(shop):
    run, _, _ = shop

    result = await run([says("hi")])

    names = {item["name"] for item in result["menu"]}
    assert names == set(PINNED)
    assert result["wallet_cents"] == 2000
    assert result["customer_profile"]["name"] == "Allan"


async def test_tool_result_reaches_the_domain(shop):
    """Not a mock: the tool really wrote to Postgres."""
    run, session, visit_id = shop

    result = await run(
        [
            tool_call("add_to_cart", item_name="Latte", quantity=2, size="small"),
            says("Two small lattes."),
        ]
    )

    assert result["cart"]["total_cents"] == 800
    assert result["cart"]["lines"][0]["quantity"] == 2


async def test_cart_is_refreshed_before_the_model_speaks_again(shop):
    """Otherwise the barista reads a stale total out loud."""
    run, _, _ = shop

    result = await run(
        [
            tool_call("add_to_cart", item_name="Latte", quantity=1, size="large"),
            says("That's $5.20."),
        ]
    )

    assert result["cart"]["total_cents"] == 520


async def test_domain_errors_come_back_as_readable_tool_messages(shop):
    """`size_required` must arrive as a question the barista can just say."""
    run, _, _ = shop

    result = await run(
        [
            tool_call("add_to_cart", item_name="Latte", quantity=1),
            says("Which size would you like?"),
        ]
    )

    tool_messages = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    assert "size_required" in tool_messages[0].content
    assert "small, medium, or large" in tool_messages[0].content


async def test_unavailable_item_is_distinct_from_unknown(shop):
    run, _, _ = shop

    result = await run(
        [
            tool_call("add_to_cart", item_name="Mocha", quantity=1, size="large"),
            says("No mochas today, sorry."),
        ]
    )

    tool_messages = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    assert "not_available_today" in tool_messages[0].content


async def test_several_tool_calls_in_one_turn(shop):
    """Three laps of the loop, then a reply."""
    run, _, _ = shop

    result = await run(
        [
            tool_call("add_to_cart", item_name="Latte", quantity=1, size="small"),
            tool_call("add_to_cart", item_name="Chocolate Chip Cookie", quantity=1),
            tool_call("get_cart"),
            says("A latte and a cookie, $6.00."),
        ]
    )

    assert result["cart"]["total_cents"] == 600
    assert len([m for m in result["messages"] if isinstance(m, ToolMessage)]) == 3


async def test_wrong_quoted_total_is_refused_through_the_graph(shop):
    """The domain gate, reached the way the model would reach it.

    Now with a second model in the way: the waiter quotes, Val charges, and the
    domain still refuses the mismatch.
    """
    run, _, _ = shop

    result = await run(
        [
            tool_call("add_to_cart", item_name="Latte", quantity=1, size="large"),
            tool_call("ring_up", request="they said yes", quoted_total_cents=400),
            says("Sorry, my mistake — it's $5.20."),
        ],
        cashier=[tool_call("charge_the_customer"), says("That's not the right total.")],
    )

    tool_messages = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    assert "total_mismatch" in tool_messages[-1].content
    assert result["wallet_cents"] == 2000


async def test_correct_total_completes_the_order(shop):
    run, _, _ = shop

    result = await run(
        [
            tool_call("add_to_cart", item_name="Latte", quantity=1, size="large"),
            tool_call("ring_up", request="they said yes", quoted_total_cents=520),
            says("Thanks, enjoy."),
        ],
        cashier=[tool_call("charge_the_customer"), says("Paid.")],
    )

    assert result["wallet_cents"] == 1480
    assert result["cart"]["lines"] == []


async def test_the_cashier_cannot_close_out_uninvited(shop):
    """Authority the waiter did not hand over is not on the table.

    Asked only to take payment, Val reached for `send_them_home` as well — the
    domain refused it, but the delegation then reported failure for a charge
    that HAD gone through, and Sam told the customer their payment had not
    worked. Now the tool is not bound at all unless Sam passed going_home.
    (The domain's own `confirmed` gate still exists; see test_orders.py.)
    """
    run, _, _ = shop

    result = await run(
        [
            tool_call("ring_up", request="are they going?", going_home=False),
            says("Off already?"),
        ],
        cashier=[tool_call("send_them_home"), says("Not yet, then.")],
    )

    last = [m for m in result["messages"] if isinstance(m, ToolMessage)][-1]
    steps = result["delegation_steps"][last.tool_call_id]
    assert steps[0]["error"] == "unknown_tool"
    assert result["visit_ended"] is False


async def test_a_charge_that_worked_is_not_reported_as_a_failure(shop):
    """The bug scoping the cashier's tools fixed, kept honest.

    A successful charge must come back ok: true even though Val, left to its own
    devices, would have tried to close out too.
    """
    run, _, _ = shop

    result = await run(
        [
            tool_call("add_to_cart", item_name="Latte", quantity=1, size="large"),
            tool_call("ring_up", request="paying", quoted_total_cents=520),
            says("Lovely."),
        ],
        cashier=[tool_call("charge_the_customer"), says("Paid.")],
    )

    last = [m for m in result["messages"] if isinstance(m, ToolMessage)][-1]
    assert json.loads(last.content)["ok"] is True
    assert result["wallet_cents"] == 1480
    assert result["visit_ended"] is False


async def test_end_visit_closes_the_visit(shop):
    run, _, _ = shop

    result = await run(
        [
            tool_call("ring_up", request="they're leaving", going_home=True),
            says("See you tomorrow."),
        ],
        cashier=[tool_call("send_them_home"), says("Night.")],
    )

    assert result["visit_ended"] is True


async def test_two_tool_calls_in_one_message_run_in_order(shop):
    """Regression: "add it and charge me" in a single assistant turn.

    LangGraph's prebuilt ToolNode runs a turn's calls concurrently. Sharing one
    AsyncSession that is unsafe, and causally wrong — place_order read the cart
    before add_to_cart had committed and failed with empty_cart. Found by
    talking to the real model; no scripted test emitted two calls at once.
    """
    run, _, _ = shop

    result = await run(
        [
            tool_calls(
                ("add_to_cart", {"item_name": "Latte", "quantity": 1, "size": "small"}),
                ("ring_up", {"request": "and charge me", "quoted_total_cents": 400}),
            ),
            says("Enjoy."),
        ],
        cashier=[tool_call("charge_the_customer"), says("Paid.")],
    )

    tool_messages = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    assert len(tool_messages) == 2
    assert "empty_cart" not in tool_messages[1].content
    assert result["wallet_cents"] == 1600


async def test_an_unknown_tool_name_does_not_crash_the_turn(shop):
    run, _, _ = shop

    result = await run([tool_calls(("teleport", {})), says("Sorry, what was that?")])

    tool_messages = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    assert "unknown_tool" in tool_messages[0].content


async def test_an_unknown_tool_comes_back_with_a_message(shop):
    """An invented tool name is an ordinary tool failure, envelope and all.

    A bare `{ok, error}` leaves the model to invent an explanation for the
    customer, which is the one thing `Result.failure` exists to prevent. The
    message also has to name the real tools, or the model's only repair strategy
    is another guess.
    """
    run, _, _ = shop

    result = await run([tool_calls(("teleport", {})), says("Sorry, what was that?")])

    envelope = json.loads([m for m in result["messages"] if isinstance(m, ToolMessage)][0].content)

    assert envelope == {"ok": False, "error": "unknown_tool", "message": envelope["message"]}
    assert "teleport" in envelope["message"]
    assert "add_to_cart" in envelope["message"]


async def test_tools_require_injected_config(shop):
    """A tool with no session must fail loudly, not operate on nothing."""
    from agent.tools import get_cart

    with pytest.raises(RuntimeError, match="configurable"):
        await get_cart.ainvoke({"args": {}, "id": "1", "name": "get_cart", "type": "tool_call"})


async def test_running_past_the_script_fails_rather_than_hanging(shop):
    """Guards the graph against an infinite barista ⇄ tools loop."""
    run, _, _ = shop

    with pytest.raises(AssertionError, match="ran out of scripted replies"):
        await run([tool_call("get_cart")])  # never returns a plain reply


async def test_a_modifier_reaches_the_domain_through_the_tool(shop):
    """The whole lap: model names codes, domain prices them.

    Extras are Mo's, so this goes through the barista — the waiter's
    add_to_cart has no `modifiers` argument (spec §13.11).
    """
    run, session, visit_id = shop

    result = await run(
        [
            tool_call("ask_barista", request="a large latte with oat milk"),
            says("One large oat latte."),
        ],
        barista=[
            tool_call(
                "add_to_cart", item_name="Latte", quantity=1, size="large", modifiers=["oat_milk"]
            ),
            says("In."),
        ],
    )

    assert result["cart"]["total_cents"] == 580
    assert result["cart"]["lines"][0]["modifiers"] == ["oat_milk"]


async def test_a_modifier_on_food_comes_back_as_a_readable_message(shop):
    """A rejected modifier is an envelope the barista can act on, not a crash."""
    run, _, _ = shop

    result = await run(
        [
            tool_call("ask_barista", request="a croissant with oat milk"),
            says("No extras on a croissant."),
        ],
        barista=[
            tool_call("add_to_cart", item_name="Croissant", quantity=1, modifiers=["oat_milk"]),
            says("Croissants don't take extras."),
        ],
    )

    last = [m for m in result["messages"] if isinstance(m, ToolMessage)][-1]
    step = result["delegation_steps"][last.tool_call_id][0]
    assert step["error"] == "modifier_not_applicable"
    assert json.loads(last.content)["ok"] is False


async def test_the_extras_line_reaches_the_prompt(shop):
    """load_context has to put modifier prices into state, or the model cannot quote them."""
    run, _, _ = shop

    result = await run([says("hi")])

    assert result["modifier_deltas"] == {"oat_milk": 60, "almond_milk": 60, "extra_shot": 100}


# --- the crew (spec §13.11) ------------------------------------------------


async def test_the_waiter_cannot_charge_without_the_cashier(shop):
    """The hard gate: it is not a prompt rule, the tool is simply not bound.

    A waiter that tries to charge directly gets the unknown-tool envelope, the
    same as any invented name.
    """
    run, _, _ = shop

    result = await run(
        [
            tool_call("add_to_cart", item_name="Latte", quantity=1, size="large"),
            tool_call("place_order", confirmed_total_cents=520),
            says("Let me get Val."),
        ]
    )

    tool_messages = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    assert "unknown_tool" in tool_messages[-1].content
    assert "ring_up" in tool_messages[-1].content
    assert result["wallet_cents"] == 2000


async def test_the_cashier_cannot_choose_the_total_it_charges(shop):
    """The most important test in the crew change.

    `confirmed_total_cents` only proves anything because it is the number the
    model that SPOKE to the customer said out loud. The cashier can read the
    cart, so if it supplied the figure it would always quote a matching one and
    the guard would quietly become a no-op. It has no way to pass one.
    """
    run, _, _ = shop

    result = await run(
        [
            tool_call("add_to_cart", item_name="Latte", quantity=1, size="large"),
            tool_call("ring_up", request="pay", quoted_total_cents=400),
            says("My mistake."),
        ],
        # Val reads the cart, sees $5.20, and tries to charge that instead.
        cashier=[
            tool_call("get_cart"),
            tool_call("charge_the_customer", confirmed_total_cents=520),
            says("Charged."),
        ],
    )

    # Val's attempt to name a figure is not even a well-formed call: the tool
    # has no such argument, so it is refused before the domain is reached.
    assert result["wallet_cents"] == 2000
    last = [m for m in result["messages"] if isinstance(m, ToolMessage)][-1]
    steps = result["delegation_steps"][last.tool_call_id]
    charge = [step for step in steps if step["tool"] == "charge_the_customer"][0]
    assert charge["error"] == "invalid_arguments"


async def test_the_domain_still_refuses_a_total_the_waiter_got_wrong(shop):
    """The other half of the guard, one layer down.

    Val charges correctly — using the injected figure — and the domain refuses
    it, because Sam quoted $4.00 for a $5.20 cart. The model can still only fail
    a charge, never lower one.
    """
    run, _, _ = shop

    result = await run(
        [
            tool_call("add_to_cart", item_name="Latte", quantity=1, size="large"),
            tool_call("ring_up", request="pay", quoted_total_cents=400),
            says("My mistake — it's $5.20."),
        ],
        cashier=[tool_call("charge_the_customer"), says("That's not the total.")],
    )

    assert result["wallet_cents"] == 2000
    last = [m for m in result["messages"] if isinstance(m, ToolMessage)][-1]
    assert json.loads(last.content)["error"] == "total_mismatch"


async def test_a_delegation_reports_its_own_tool_calls(shop):
    """`steps` is what stops a delegation being an opaque box in the UI."""
    run, _, _ = shop

    result = await run(
        [
            tool_call("ask_barista", request="a large latte with oat milk"),
            says("Coming up."),
        ],
        barista=[
            tool_call(
                "add_to_cart", item_name="Latte", quantity=1, size="large", modifiers=["oat_milk"]
            ),
            says("Large oat latte in."),
        ],
    )

    last = [m for m in result["messages"] if isinstance(m, ToolMessage)][-1]
    envelope = json.loads(last.content)
    assert envelope["agent"] == "barista"
    # Steps reach state, not the message — see test_a_sub_agents_steps... below.
    assert "steps" not in envelope
    steps = result["delegation_steps"][last.tool_call_id]
    assert [step["tool"] for step in steps] == ["add_to_cart"]
    assert result["cart"]["lines"][0]["modifiers"] == ["oat_milk"]


async def test_the_barista_can_take_several_laps_inside_one_delegation(shop):
    """A sub-agent is a small loop, not a one-shot call."""
    run, _, _ = shop

    result = await run(
        [tool_call("ask_barista", request="a large latte, actually make it oat"), says("Done.")],
        barista=[
            tool_call("add_to_cart", item_name="Latte", quantity=1, size="large"),
            tool_call("change_modifiers", item_name="Latte", to_modifiers=["oat_milk"]),
            says("Large oat latte."),
        ],
    )

    last = [m for m in result["messages"] if isinstance(m, ToolMessage)][-1]
    steps = result["delegation_steps"][last.tool_call_id]
    assert [step["tool"] for step in steps] == ["add_to_cart", "change_modifiers"]
    assert result["cart"]["total_cents"] == 580


async def test_a_barista_clarification_reaches_the_customer(shop):
    """`size_required`'s shape, lifted to the agent layer.

    Mo cannot tell almond from hazelnut, so it answers with a question and calls
    no tool. Sam reads it out; the cart is untouched.
    """
    run, _, _ = shop

    result = await run(
        [tool_call("ask_barista", request="a latte with the nutty milk"), says("Almond or oat?")],
        barista=[says("Almond or oat?")],
    )

    envelope = json.loads([m for m in result["messages"] if isinstance(m, ToolMessage)][-1].content)
    assert envelope["ok"] is True
    assert envelope["message"] == "Almond or oat?"
    assert result["cart"]["lines"] == []


async def test_a_runaway_sub_agent_returns_an_envelope_at_the_lap_cap(shop):
    """Never raises, and comes back as something the waiter can say out loud."""
    run, _, _ = shop

    result = await run(
        [tool_call("ask_barista", request="a latte"), says("One moment.")],
        barista=[tool_call("get_menu")] * 6,
    )

    envelope = json.loads([m for m in result["messages"] if isinstance(m, ToolMessage)][-1].content)
    assert envelope["ok"] is False
    assert envelope["error"] == "delegation_incomplete"
    assert envelope["message"]


async def test_a_delegation_and_a_ring_up_in_one_message_run_in_order(shop):
    """The 69ac67f shape, now at the agent layer.

    "A large oat latte and then ring me up" emits both in one message. Run
    concurrently the cashier reads the cart before the barista's add_to_cart has
    committed, and the charge fails with empty_cart.
    """
    run, _, _ = shop

    result = await run(
        [
            tool_calls(
                ("ask_barista", {"request": "large latte with oat"}),
                ("ring_up", {"request": "and charge me", "quoted_total_cents": 580}),
            ),
            says("Enjoy."),
        ],
        barista=[
            tool_call(
                "add_to_cart", item_name="Latte", quantity=1, size="large", modifiers=["oat_milk"]
            ),
            says("In."),
        ],
        cashier=[tool_call("charge_the_customer"), says("Paid.")],
    )

    assert result["wallet_cents"] == 1420
    assert result["cart"]["lines"] == []


async def test_the_cashier_reads_the_cart_to_explain_a_refusal(shop):
    """Why the cashier is a model and not a passthrough."""
    run, session, visit_id = shop

    result = await run(
        [
            tool_call("add_to_cart", item_name="Latte", quantity=1, size="large"),
            tool_call("ring_up", request="pay", quoted_total_cents=520),
            says("Val says that's fine."),
        ],
        cashier=[
            tool_call("get_cart"),
            tool_call("charge_the_customer"),
            says("$5.20, done."),
        ],
    )

    last = [m for m in result["messages"] if isinstance(m, ToolMessage)][-1]
    steps = result["delegation_steps"][last.tool_call_id]
    assert [step["tool"] for step in steps] == ["get_cart", "charge_the_customer"]
    assert result["wallet_cents"] == 1480


async def test_a_sub_agents_steps_stay_out_of_the_waiters_context(shop):
    """The saving that justifies agents-as-tools in the first place.

    The ToolMessage's content IS the waiter's context on every later turn. If a
    delegation's internal calls rode along inside the envelope, Sam's prompt
    would grow by everything Mo and Val ever did.
    """
    run, _, _ = shop

    result = await run(
        [tool_call("ask_barista", request="a large latte with oat"), says("Coming up.")],
        barista=[
            tool_call("get_menu"),
            tool_call(
                "add_to_cart", item_name="Latte", quantity=1, size="large", modifiers=["oat_milk"]
            ),
            says("In."),
        ],
    )

    last = [m for m in result["messages"] if isinstance(m, ToolMessage)][-1]
    assert "get_menu" not in last.content
    assert "steps" not in last.content
    # But the record still exists, for the panel.
    assert len(result["delegation_steps"][last.tool_call_id]) == 2


async def test_the_waiter_cannot_add_extras_itself(shop):
    """The barista gate, and it has to be structural for the same reason the
    cashier gate is.

    Given `modifiers` in its own schema the waiter simply never delegates —
    against the real model it took "a large espresso with oat milk" alone and
    Mo never ran. The argument is not in the waiter's schema at all, so a call
    carrying it is a malformed call.
    """
    run, _, _ = shop

    result = await run(
        [
            tool_call(
                "add_to_cart", item_name="Latte", quantity=1, size="large", modifiers=["oat_milk"]
            ),
            says("Let me ask Mo."),
        ]
    )

    envelope = json.loads([m for m in result["messages"] if isinstance(m, ToolMessage)][0].content)
    assert envelope["ok"] is False
    assert envelope["error"] == "invalid_arguments"
    assert result["cart"]["lines"] == []


async def test_a_failed_charge_does_not_hide_a_successful_close_out(shop):
    """Both outcomes are reported as facts, because one `ok` cannot carry them.

    Val charges an already-paid (so empty) cart and then closes out. The charge
    failing must not make the visit look still-open, and the close-out
    succeeding must not make the payment look like it went through.
    """
    run, _, _ = shop

    result = await run(
        [
            tool_call("ring_up", request="pay and go", quoted_total_cents=520, going_home=True),
            says("Night!"),
        ],
        cashier=[
            tool_call("charge_the_customer"),
            tool_call("send_them_home"),
            says("Nothing to pay — off you go."),
        ],
    )

    envelope = json.loads([m for m in result["messages"] if isinstance(m, ToolMessage)][-1].content)
    assert envelope["charged"] is False
    assert envelope["visit_ended"] is True
    assert envelope["error"] == "empty_cart"
    assert result["visit_ended"] is True


async def test_a_successful_charge_reports_charged(shop):
    """The field the waiter's "never say it is paid for" rule keys on."""
    run, _, _ = shop

    result = await run(
        [
            tool_call("add_to_cart", item_name="Latte", quantity=1, size="large"),
            tool_call("ring_up", request="paying", quoted_total_cents=520),
            says("Lovely."),
        ],
        cashier=[tool_call("charge_the_customer"), says("Paid.")],
    )

    envelope = json.loads([m for m in result["messages"] if isinstance(m, ToolMessage)][-1].content)
    assert envelope["charged"] is True
    assert envelope["visit_ended"] is False


async def test_a_tool_that_raises_comes_back_as_an_envelope(shop):
    """The invariant behind TOUR bug 2, which had no regression test at all.

    A missing REQUIRED argument makes the tool itself raise, which is a
    different path from the unknown-argument check above — that one never
    reaches the tool. Small models drop required arguments constantly, and the
    turn has to survive it.
    """
    run, _, _ = shop

    result = await run(
        [tool_call("add_to_cart", quantity=1, size="large"), says("Sorry — which drink?")]
    )

    envelope = json.loads([m for m in result["messages"] if isinstance(m, ToolMessage)][0].content)
    assert envelope["ok"] is False
    assert envelope["error"] == "invalid_arguments"
    assert envelope["message"], "an envelope without a message leaves the model to invent one"
    assert result["messages"][-1].content == "Sorry — which drink?"


async def test_the_cashier_cannot_charge_when_nothing_was_quoted(shop):
    """Val is normally not even given the tool, so this is the belt to that
    braces — it must still refuse rather than charge some default."""
    run, _, _ = shop
    from agent.tools import charge_the_customer

    result = await run(
        [tool_call("ring_up", request="just leaving"), says("Right you are.")],
        cashier=[says("Nothing to charge.")],
    )
    assert result["wallet_cents"] == 2000

    session, visit_id = shop[1], shop[2]
    payload = await charge_the_customer.ainvoke(
        {}, {"configurable": {"session": session, "visit_id": str(visit_id)}}
    )
    assert payload["error"] == "nothing_quoted"
    assert payload["message"]


async def test_a_sub_agent_stops_once_the_job_is_done(shop):
    """Val charged twice in one delegation: the first worked, the second hit
    empty_cart because the first had emptied the cart — and Val then narrated
    THAT, telling the customer their order was empty right after paying.

    Once everything the waiter authorised has succeeded there is nothing left
    to decide, so the delegation returns instead of taking another lap.
    """
    run, _, _ = shop

    result = await run(
        [
            tool_call("add_to_cart", item_name="Latte", quantity=1, size="large"),
            tool_call("ring_up", request="pay", quoted_total_cents=520),
            says("Lovely."),
        ],
        cashier=[
            tool_call("charge_the_customer"),
            tool_call("charge_the_customer"),
            says("Order was empty!"),
        ],
    )

    last = [m for m in result["messages"] if isinstance(m, ToolMessage)][-1]
    envelope = json.loads(last.content)
    steps = result["delegation_steps"][last.tool_call_id]

    assert len(steps) == 1, "the second charge should never have been attempted"
    assert envelope["charged"] is True
    assert "empty" not in (envelope["message"] or "").lower()
    assert result["wallet_cents"] == 1480


async def test_a_sub_agent_cannot_repeat_the_same_call(shop):
    """Measured against the real model: Mo called add_to_cart twice and put the
    drink in twice; Val charged twice and the second failed with empty_cart.

    The guard refuses the repeat rather than capping laps, so a genuinely
    multi-part job still works — see the next test.
    """
    run, _, _ = shop

    result = await run(
        [tool_call("ask_barista", request="a large latte with oat"), says("Done.")],
        barista=[
            tool_call(
                "add_to_cart", item_name="Latte", quantity=1, size="large", modifiers=["oat_milk"]
            ),
            tool_call(
                "add_to_cart", item_name="Latte", quantity=1, size="large", modifiers=["oat_milk"]
            ),
            says("In."),
        ],
    )

    assert result["cart"]["lines"][0]["quantity"] == 1, "the drink went in twice"
    last = [m for m in result["messages"] if isinstance(m, ToolMessage)][-1]
    steps = result["delegation_steps"][last.tool_call_id]
    assert [s["error"] for s in steps] == [None, "already_done"]


async def test_a_sub_agent_can_still_make_several_different_calls(shop):
    """The guard keys on the arguments, not the tool name, so two genuinely
    different drinks are two calls and both run."""
    run, _, _ = shop

    result = await run(
        [
            tool_call("ask_barista", request="a large latte and a small one, both oat"),
            says("Done."),
        ],
        barista=[
            tool_call(
                "add_to_cart", item_name="Latte", quantity=1, size="large", modifiers=["oat_milk"]
            ),
            tool_call(
                "add_to_cart", item_name="Latte", quantity=1, size="small", modifiers=["oat_milk"]
            ),
            says("Both in."),
        ],
    )

    assert len(result["cart"]["lines"]) == 2
    assert result["cart"]["total_cents"] == 580 + 460

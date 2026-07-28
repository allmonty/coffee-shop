"""The agent loop, tested with a scripted model — no Ollama, CI-safe.

The assertion that matters most is `test_tool_call_loops_back_to_the_barista`:
that cycle is the agent. Everything else in the graph is setup and teardown.
"""

import uuid

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from sqlalchemy import select

from agent.graph import build_graph, route_after_barista
from shop.models import MenuItem, VisitMenuItem
from shop.seed import seed_catalog
from shop.service import enter
from tests.fakes import FakeToolCallingModel, says, tool_call

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

    def run(script, text="hello"):
        graph = build_graph(llm=FakeToolCallingModel(script))
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
    """The domain gate, reached the way the model would reach it."""
    run, _, _ = shop

    result = await run(
        [
            tool_call("add_to_cart", item_name="Latte", quantity=1, size="large"),
            tool_call("place_order", confirmed_total_cents=400),
            says("Sorry, my mistake — it's $5.20."),
        ]
    )

    tool_messages = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    assert "total_mismatch" in tool_messages[-1].content
    assert result["wallet_cents"] == 2000


async def test_correct_total_completes_the_order(shop):
    run, _, _ = shop

    result = await run(
        [
            tool_call("add_to_cart", item_name="Latte", quantity=1, size="large"),
            tool_call("place_order", confirmed_total_cents=520),
            says("Thanks, enjoy."),
        ]
    )

    assert result["wallet_cents"] == 1480
    assert result["cart"]["lines"] == []


async def test_end_visit_without_confirmation_is_refused(shop):
    run, _, _ = shop

    result = await run(
        [
            tool_call("end_visit", confirmed=False),
            says("Off already?"),
        ]
    )

    tool_messages = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    assert "confirmation_required" in tool_messages[0].content
    assert result["visit_ended"] is False


async def test_end_visit_closes_the_visit(shop):
    run, _, _ = shop

    result = await run(
        [
            tool_call("end_visit", confirmed=True),
            says("See you tomorrow."),
        ]
    )

    assert result["visit_ended"] is True


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

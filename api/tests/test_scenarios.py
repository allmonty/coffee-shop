"""Scripted conversations against the REAL model (spec §12).

    uv run pytest tests/test_scenarios.py -m scenario -v

Deselected by default and kept out of CI. These are flaky by nature — that is
not a defect, it is the honest state of a 7B model driving a transaction — so
they assert on **final domain state**, never on wording. "Did the wallet end up
right?" is a fact; "did it say the word large?" is a coin toss.

When one fails, look at the trace before touching the code: most failures here
are prompt or model problems, and the graph is rarely at fault.
"""

from __future__ import annotations

import uuid

import pytest
from sqlalchemy import select

from agent.graph import build_graph
from agent.runner import run_turn
from shop.models import MenuItem, Order, User, VisitMenuItem
from shop.seed import seed_catalog
from shop.service import enter, get_cart

# The 30s default exists to catch deadlocks in the fast suite. These drive a
# real model through several turns; a 14B on consumer hardware needs minutes,
# and a timeout here would look like a model failure when it is only slowness.
pytestmark = [pytest.mark.scenario, pytest.mark.timeout(600)]

PINNED = ["Latte", "Filter Coffee", "Chocolate Chip Cookie", "Croissant"]


@pytest.fixture
async def shop(session):
    """A real graph on the real model, with today's menu pinned so the
    assertions do not depend on the daily draw."""
    await seed_catalog(session)
    entered = await enter(session, "Scenario")
    visit_id = uuid.UUID(entered.data["visit_id"])
    user_id = uuid.UUID(entered.data["user_id"])

    await session.execute(
        VisitMenuItem.__table__.delete().where(VisitMenuItem.visit_id == visit_id)
    )
    items = (await session.scalars(select(MenuItem).where(MenuItem.name.in_(PINNED)))).all()
    for item in items:
        session.add(VisitMenuItem(visit_id=visit_id, menu_item_id=item.id))
    await session.commit()

    graph = build_graph()

    async def say(text: str) -> str:
        spoken = []
        async for frame in run_turn(
            session=session,
            graph=graph,
            user_id=user_id,
            visit_id=visit_id,
            message=text,
        ):
            if frame["type"] == "token":
                spoken.append(frame["text"])
        return "".join(spoken)

    return say, session, visit_id, user_id


async def test_orders_a_drink_with_a_size(shop):
    say, session, visit_id, _ = shop

    await say("a large latte please")

    cart = (await get_cart(session, visit_id)).data
    assert cart["lines"], "nothing was added to the cart"
    assert cart["lines"][0]["item"] == "Latte"
    assert cart["lines"][0]["size"] == "large"


async def test_asks_which_size_rather_than_guessing(shop):
    """A drink with no size is incomplete, not wrong — the barista must ask."""
    say, session, visit_id, _ = shop

    reply = await say("a latte please")

    cart = (await get_cart(session, visit_id)).data
    if cart["lines"]:
        # If it did add something it must have asked first, not invented a size.
        pytest.fail(f"picked a size without asking: {cart['lines']}")
    assert "size" in reply.lower() or "small" in reply.lower()


async def test_does_not_ask_what_size_a_pastry_is(shop):
    """The most obvious way for the barista to sound like a machine."""
    say, session, visit_id, _ = shop

    reply = await say("a croissant thanks")

    assert "what size" not in reply.lower()
    cart = (await get_cart(session, visit_id)).data
    assert cart["lines"][0]["size"] is None


async def test_refuses_something_not_on_todays_menu(shop):
    say, session, visit_id, _ = shop

    await say("one mocha please")

    cart = (await get_cart(session, visit_id)).data
    assert not cart["lines"], "sold an item that is not on today's menu"


async def test_cannot_overspend(shop):
    """The wallet is a domain invariant, not a prompt rule."""
    say, session, visit_id, user_id = shop
    user = await session.get(User, user_id)
    user.wallet_cents = 300
    await session.commit()

    await say("a large latte, and please charge me for it now")

    user = await session.get(User, user_id)
    assert user.wallet_cents == 300, "charged past the wallet"


async def test_a_completed_order_debits_exactly_once(shop):
    say, session, visit_id, user_id = shop

    await say("one small latte please")
    await say("yes, that's everything — charge me")

    user = await session.get(User, user_id)
    assert user.wallet_cents in (2000, 1600), (
        f"wallet ended at {user.wallet_cents}; expected untouched or exactly one latte"
    )


# --- the crew (spec §13.11) -------------------------------------------------
#
# Every significant bug in the three-role split was found by talking to the real
# model, not by a scripted test: Sam never delegating when it had a `modifiers`
# argument of its own, LangChain silently dropping that argument, Mo adding a
# drink twice, Val charging twice and narrating the failure. Scripted tests
# cannot find those, because they only ever emit the shapes we imagined.
#
# So these assert the two things that must hold however the models behave: the
# domain ends up right, and the roles stayed inside their authority.


async def test_a_drink_with_an_extra_is_priced_with_the_surcharge(shop):
    """Whoever routes it, the customer is charged for the oat milk."""
    say, session, visit_id, _ = shop

    await say("a large latte with oat milk please")

    cart = (await get_cart(session, visit_id)).data
    assert len(cart["lines"]) == 1
    line = cart["lines"][0]
    assert (line["item"], line["size"]) == ("Latte", "large")
    assert line["modifiers"] == ["oat_milk"]
    assert cart["total_cents"] == 580  # 400 base + 120 large + 60 oat


async def test_plain_milk_is_not_an_extra(shop):
    """ "with milk" is the drink as listed, so it must not become a modifier and
    must not cost anything more (§3.6)."""
    say, session, visit_id, _ = shop

    await say("a large latte with milk please")

    cart = (await get_cart(session, visit_id)).data
    assert cart["lines"][0]["modifiers"] == []
    assert cart["total_cents"] == 520


async def test_a_drink_only_ever_goes_in_once(shop):
    """Mo called add_to_cart twice for one request and put the drink in the cart
    twice. The delegation now refuses a verbatim repeat."""
    say, session, visit_id, _ = shop

    await say("one large latte with an extra shot, please")

    cart = (await get_cart(session, visit_id)).data
    assert sum(line["quantity"] for line in cart["lines"]) == 1


async def test_paying_debits_exactly_once(shop):
    """Val charged twice in one delegation, the second failing with empty_cart —
    and then narrated that, telling the customer their order was empty right
    after they paid for it."""
    say, session, visit_id, user_id = shop

    await say("a large latte with oat milk")
    await say("that's everything, I'll pay now")
    await say("yes please, go ahead")

    user = await session.get(User, user_id)
    orders = (
        (
            await session.execute(
                select(Order.total_cents)
                .where(Order.visit_id == visit_id)
                .order_by(Order.placed_at)
            )
        )
        .scalars()
        .all()
    )

    # Asserted on `orders`, not on the cart. "yes please, go ahead" is ambiguous
    # enough that the model sometimes reads it as a fresh order and adds a drink
    # after paying — annoying, but not wrong, and not what this test is about.
    # What must never happen is a partial or a double debit.
    assert user.wallet_cents == 2000 - sum(orders), (user.wallet_cents, orders)
    assert len(orders) <= 1, f"charged more than once: {orders}"
    if orders:
        assert orders[0] == 580, orders


async def test_the_waiter_never_spends_money_itself(shop):
    """The hard gate, end to end: Sam has no place_order and no end_visit, so
    every debit in the whole conversation went through Val."""
    say, session, visit_id, user_id = shop

    await say("a large latte and a croissant")
    await say("that's all, charge me")

    user = await session.get(User, user_id)
    # 520 + 350 = 870. Any other non-2000 figure means something charged a total
    # nobody quoted.
    assert user.wallet_cents in (2000, 1130), user.wallet_cents


async def test_going_home_always_advances_the_day(shop):
    """The Go Home button is a deterministic control, not a sentence to
    interpret — it must work with an unpaid cart, which is where it broke."""
    say, session, visit_id, user_id = shop
    graph = build_graph()

    await say("a large latte please")

    async for _ in run_turn(
        session=session, graph=graph, user_id=user_id, visit_id=visit_id, event="go_home"
    ):
        pass

    user = await session.get(User, user_id)
    assert user.current_day == 2
    assert (await get_cart(session, visit_id)).error == "visit_closed"

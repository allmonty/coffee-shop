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
from shop.models import MenuItem, User, VisitMenuItem
from shop.seed import seed_catalog
from shop.service import enter, get_cart

pytestmark = pytest.mark.scenario

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

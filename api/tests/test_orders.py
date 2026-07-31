"""Paying and going home — the two operations guarded in the domain (spec §6.4)."""

import uuid

import pytest
from sqlalchemy import func, select

from shop.models import (
    DrinkModifier,
    MenuItem,
    Order,
    OrderLine,
    SizeModifier,
    User,
    Visit,
    VisitMenuItem,
)
from shop.seed import seed_catalog
from shop.service import (
    add_to_cart,
    end_visit,
    enter,
    get_cart,
    get_wallet_balance,
    order_history,
    place_order,
)


@pytest.fixture
async def shop(session):
    await seed_catalog(session)
    result = await enter(session, "Allan")
    visit_id = uuid.UUID(result.data["visit_id"])
    user_id = uuid.UUID(result.data["user_id"])

    await session.execute(
        VisitMenuItem.__table__.delete().where(VisitMenuItem.visit_id == visit_id)
    )
    wanted = ["Latte", "Filter Coffee", "Chocolate Chip Cookie", "Affogato"]
    items = (await session.scalars(select(MenuItem).where(MenuItem.name.in_(wanted)))).all()
    for item in items:
        session.add(VisitMenuItem(visit_id=visit_id, menu_item_id=item.id))
    await session.commit()

    return session, visit_id, user_id


async def test_place_order_charges_the_wallet(shop):
    session, visit_id, user_id = shop
    await add_to_cart(session, visit_id, "Latte", 1, "large")  # 520

    result = await place_order(session, visit_id, confirmed_total_cents=520)

    assert result.ok is True
    assert result.data["wallet_cents"] == 2000 - 520

    user = await session.get(User, user_id)
    assert user.wallet_cents == 1480


async def test_place_order_empties_the_cart(shop):
    session, visit_id, _ = shop
    await add_to_cart(session, visit_id, "Latte", 1, "small")

    await place_order(session, visit_id, confirmed_total_cents=400)

    cart = await get_cart(session, visit_id)
    assert cart.data["lines"] == []


async def test_wrong_quoted_total_is_refused(shop):
    """The gate that replaces "the prompt says confirm first".

    Charging without confirming now requires guessing the exact cart total.
    """
    session, visit_id, user_id = shop
    await add_to_cart(session, visit_id, "Latte", 1, "large")  # 520

    result = await place_order(session, visit_id, confirmed_total_cents=400)

    assert result.error == "total_mismatch"
    assert "5.20" in result.message
    user = await session.get(User, user_id)
    assert user.wallet_cents == 2000  # nothing charged


async def test_cannot_undercharge_by_quoting_less(shop):
    session, visit_id, user_id = shop
    await add_to_cart(session, visit_id, "Affogato", 1, "large")  # 650 + 120

    result = await place_order(session, visit_id, confirmed_total_cents=1)

    assert result.error == "total_mismatch"
    user = await session.get(User, user_id)
    assert user.wallet_cents == 2000


async def test_empty_cart_is_refused(shop):
    session, visit_id, _ = shop

    result = await place_order(session, visit_id, confirmed_total_cents=0)

    assert result.error == "empty_cart"


async def test_insufficient_funds_names_both_figures(shop):
    session, visit_id, user_id = shop
    user = await session.get(User, user_id)
    user.wallet_cents = 300
    await session.commit()
    await add_to_cart(session, visit_id, "Latte", 1, "large")  # 520

    result = await place_order(session, visit_id, confirmed_total_cents=520)

    assert result.error == "insufficient_funds"
    assert "5.20" in result.message
    assert "3.00" in result.message


async def test_duplicate_place_order_charges_once(shop):
    """An LLM emitting the same tool call twice must not double-charge."""
    session, visit_id, user_id = shop
    await add_to_cart(session, visit_id, "Latte", 1, "small")

    first = await place_order(session, visit_id, confirmed_total_cents=400)
    assert first.ok is True

    # Same cart version, replayed.
    second = await place_order(session, visit_id, confirmed_total_cents=400)

    user = await session.get(User, user_id)
    assert user.wallet_cents == 1600
    # The cart is empty after the first order, so the replay reports that
    # rather than charging again.
    assert second.error == "empty_cart"

    orders = await session.scalar(select(func.count()).select_from(Order))
    assert orders == 1


async def test_order_lines_snapshot_the_price_actually_charged(shop):
    """Editing size_modifiers later must not rewrite history."""
    session, visit_id, _ = shop
    await add_to_cart(session, visit_id, "Latte", 2, "large")  # 520 each

    await place_order(session, visit_id, confirmed_total_cents=1040)

    large = await session.get(SizeModifier, "large")
    large.delta_cents = 999
    await session.commit()

    line = await session.scalar(select(OrderLine))
    assert line.unit_price_cents == 520
    assert line.size == "large"
    assert line.quantity == 2


async def test_order_lines_snapshot_the_modifier_surcharge(shop):
    """Same guarantee, same mechanism, one axis over: editing drink_modifiers
    later must not rewrite what a customer was charged."""
    session, visit_id, _ = shop
    await add_to_cart(session, visit_id, "Latte", 1, "large", ["oat_milk"])  # 580

    await place_order(session, visit_id, confirmed_total_cents=580)

    oat = await session.get(DrinkModifier, "oat_milk")
    oat.delta_cents = 999
    await session.commit()

    line = await session.scalar(select(OrderLine))
    assert line.unit_price_cents == 580
    assert line.modifiers == "oat_milk"


async def test_multiple_orders_in_one_visit(shop):
    session, visit_id, user_id = shop

    await add_to_cart(session, visit_id, "Latte", 1, "small")
    await place_order(session, visit_id, confirmed_total_cents=400)
    await add_to_cart(session, visit_id, "Chocolate Chip Cookie", 1)
    second = await place_order(session, visit_id, confirmed_total_cents=200)

    assert second.ok is True
    user = await session.get(User, user_id)
    assert user.wallet_cents == 2000 - 400 - 200


async def test_end_visit_requires_confirmation(shop):
    session, visit_id, _ = shop

    result = await end_visit(session, visit_id, confirmed=False)

    assert result.error == "confirmation_required"
    visit = await session.get(Visit, visit_id)
    assert visit.ended_at is None


async def test_end_visit_advances_the_day_and_refills_the_wallet(shop):
    session, visit_id, user_id = shop
    await add_to_cart(session, visit_id, "Latte", 1, "small")
    await place_order(session, visit_id, confirmed_total_cents=400)

    result = await end_visit(session, visit_id, confirmed=True)

    assert result.ok is True
    assert result.data == {"day": 2, "wallet_cents": 2000}
    assert result.message == "See you tomorrow."

    user = await session.get(User, user_id)
    assert (user.current_day, user.wallet_cents) == (2, 2000)


async def test_unspent_money_does_not_carry_over(shop):
    session, visit_id, user_id = shop
    await end_visit(session, visit_id, confirmed=True)

    user = await session.get(User, user_id)
    assert user.wallet_cents == 2000  # not 2000 + yesterday's leftovers


async def test_next_day_gets_a_fresh_visit_and_menu(shop):
    session, visit_id, _ = shop
    await end_visit(session, visit_id, confirmed=True)

    result = await enter(session, "Allan")

    assert result.data["visit_id"] != str(visit_id)
    assert result.data["day"] == 2
    assert result.data["weekday"] == "Tuesday"


async def test_operations_on_a_finished_visit_are_refused(shop):
    session, visit_id, _ = shop
    await end_visit(session, visit_id, confirmed=True)

    assert (await add_to_cart(session, visit_id, "Latte", 1, "small")).error == "visit_closed"
    assert (await place_order(session, visit_id, 400)).error == "visit_closed"
    assert (await end_visit(session, visit_id, True)).error == "visit_closed"
    assert (await get_wallet_balance(session, visit_id)).error == "visit_closed"


async def test_wallet_balance_reports_remaining_money(shop):
    session, visit_id, _ = shop
    await add_to_cart(session, visit_id, "Latte", 1, "large")
    await place_order(session, visit_id, confirmed_total_cents=520)

    result = await get_wallet_balance(session, visit_id)

    assert result.data["wallet_cents"] == 1480
    assert result.data["day"] == 1


async def test_order_history_records_sizes(shop):
    session, visit_id, user_id = shop
    await add_to_cart(session, visit_id, "Latte", 1, "large")
    await place_order(session, visit_id, confirmed_total_cents=520)

    history = await order_history(session, user_id)

    assert len(history) == 1
    assert history[0]["lines"][0] == {
        "item": "Latte",
        "size": "large",
        "modifiers": [],
        "quantity": 1,
        "unit_price_cents": 520,
    }

"""The profile is aggregated, never stored (spec §6.5)."""

import uuid

import pytest
from sqlalchemy import select

from shop.models import MenuItem, VisitMenuItem
from shop.profile import customer_profile
from shop.seed import seed_catalog
from shop.service import add_to_cart, end_visit, enter, place_order


async def _pin_menu(session, visit_id, names):
    await session.execute(
        VisitMenuItem.__table__.delete().where(VisitMenuItem.visit_id == visit_id)
    )
    items = (await session.scalars(select(MenuItem).where(MenuItem.name.in_(names)))).all()
    for item in items:
        session.add(VisitMenuItem(visit_id=visit_id, menu_item_id=item.id))
    await session.commit()


@pytest.fixture
async def regular(session):
    """Allan, who has bought a large latte and a cookie three days running."""
    await seed_catalog(session)
    user_id = None

    for _ in range(3):
        entered = await enter(session, "Allan")
        visit_id = uuid.UUID(entered.data["visit_id"])
        user_id = uuid.UUID(entered.data["user_id"])
        await _pin_menu(session, visit_id, ["Latte", "Chocolate Chip Cookie", "Croissant"])

        await add_to_cart(session, visit_id, "Latte", 1, "large")
        await add_to_cart(session, visit_id, "Chocolate Chip Cookie", 1)
        await place_order(session, visit_id, confirmed_total_cents=520 + 200)
        await end_visit(session, visit_id, confirmed=True)

    return session, user_id


async def test_favourites_come_from_order_history(regular):
    session, user_id = regular

    profile = await customer_profile(session, user_id)

    assert profile["favorite_drink"] == "Latte"
    assert profile["favorite_food"] == "Chocolate Chip Cookie"
    assert profile["visit_count"] == 3
    assert profile["last_visit_day"] == 3


async def test_usual_order_carries_the_size(regular):
    """Grouping by (item, size) is what lets the barista say "large, like
    always?" instead of asking a regular the same question daily."""
    session, user_id = regular

    profile = await customer_profile(session, user_id)

    assert profile["usual_order"] == [
        {"item": "Latte", "size": "large", "qty": 1},
        {"item": "Chocolate Chip Cookie", "size": None, "qty": 1},
    ]


async def test_usual_is_flagged_against_todays_menu(regular):
    """Offering an unavailable usual is worse than not offering at all."""
    session, user_id = regular

    entered = await enter(session, "Allan")
    visit_id = uuid.UUID(entered.data["visit_id"])
    await _pin_menu(session, visit_id, ["Latte", "Croissant"])  # no cookie today

    profile = await customer_profile(session, user_id, visit_id)

    by_item = {line["item"]: line for line in profile["usual_order"]}
    assert by_item["Latte"]["available_today"] is True
    assert by_item["Chocolate Chip Cookie"]["available_today"] is False


async def test_favourite_size_follows_the_majority(session):
    await seed_catalog(session)
    entered = await enter(session, "Mary")
    visit_id = uuid.UUID(entered.data["visit_id"])
    user_id = uuid.UUID(entered.data["user_id"])
    await _pin_menu(session, visit_id, ["Latte", "Croissant"])

    await add_to_cart(session, visit_id, "Latte", 3, "small")
    await add_to_cart(session, visit_id, "Latte", 1, "large")
    await place_order(session, visit_id, confirmed_total_cents=400 * 3 + 520)

    profile = await customer_profile(session, user_id)

    assert profile["usual_order"][0]["size"] == "small"


async def test_new_customer_has_no_usual(session):
    await seed_catalog(session)
    entered = await enter(session, "Newbie")
    user_id = uuid.UUID(entered.data["user_id"])

    profile = await customer_profile(session, user_id)

    assert profile["usual_order"] == []
    assert profile["favorite_drink"] is None
    assert profile["last_visit_day"] is None

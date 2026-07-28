"""Cart behaviour, especially the four distinct add_to_cart failures.

Each error is a different truth the barista has to tell the customer, so each
gets its own test — a single generic "rejected" would let them collapse.
"""

import uuid

import pytest
from sqlalchemy import select

from shop.models import MenuItem, VisitMenuItem
from shop.seed import seed_catalog
from shop.service import add_to_cart, change_size, enter, get_cart, remove_from_cart


@pytest.fixture
async def shop(session):
    """A seeded shop with Allan inside, and today's menu forced to a known set.

    Pinning the menu matters: the daily draw is random, so a test that ordered
    "Latte" would fail on the days Latte was not drawn.
    """
    await seed_catalog(session)
    result = await enter(session, "Allan")
    visit_id = uuid.UUID(result.data["visit_id"])

    await session.execute(
        VisitMenuItem.__table__.delete().where(VisitMenuItem.visit_id == visit_id)
    )
    wanted = ["Latte", "Filter Coffee", "Chocolate Chip Cookie", "Croissant"]
    items = (await session.scalars(select(MenuItem).where(MenuItem.name.in_(wanted)))).all()
    for item in items:
        session.add(VisitMenuItem(visit_id=visit_id, menu_item_id=item.id))
    await session.commit()

    return session, visit_id


async def test_add_a_sized_drink(shop):
    session, visit_id = shop

    result = await add_to_cart(session, visit_id, "Latte", 1, "large")

    assert result.ok is True
    assert result.data["total_cents"] == 520  # 400 base + 120 large
    assert result.data["lines"][0]["size"] == "large"


async def test_add_food_without_a_size(shop):
    session, visit_id = shop

    result = await add_to_cart(session, visit_id, "Chocolate Chip Cookie", 2)

    assert result.ok is True
    assert result.data["total_cents"] == 400


async def test_unknown_item_is_not_the_same_as_unavailable(shop):
    session, visit_id = shop

    result = await add_to_cart(session, visit_id, "Bubble Tea", 1, "large")

    assert result.error == "unknown_item"
    assert "don't do" in result.message


async def test_real_item_not_on_todays_menu(shop):
    """ "We're not doing mochas today" is true; "we don't sell mochas" is not."""
    session, visit_id = shop

    result = await add_to_cart(session, visit_id, "Mocha", 1, "large")

    assert result.error == "not_available_today"
    assert "today" in result.message.lower()


async def test_drink_without_a_size_asks_rather_than_guessing(shop):
    session, visit_id = shop

    result = await add_to_cart(session, visit_id, "Latte", 1)

    assert result.error == "size_required"
    # The message is a question the barista can read aloud verbatim.
    assert result.message.endswith("?")


async def test_food_with_a_size_is_rejected(shop):
    session, visit_id = shop

    result = await add_to_cart(session, visit_id, "Croissant", 1, "large")

    assert result.error == "size_not_applicable"


async def test_unknown_size_is_rejected(shop):
    session, visit_id = shop

    result = await add_to_cart(session, visit_id, "Latte", 1, "venti")

    assert result.error == "unknown_size"


async def test_zero_quantity_is_rejected(shop):
    session, visit_id = shop

    result = await add_to_cart(session, visit_id, "Latte", 0, "small")

    assert result.error == "invalid_quantity"


async def test_item_lookup_is_case_insensitive(shop):
    session, visit_id = shop

    result = await add_to_cart(session, visit_id, "  latte ", 1, "small")

    assert result.ok is True
    assert result.data["lines"][0]["item"] == "Latte"


async def test_same_drink_twice_stacks_onto_one_line(shop):
    session, visit_id = shop

    await add_to_cart(session, visit_id, "Latte", 1, "small")
    result = await add_to_cart(session, visit_id, "Latte", 2, "small")

    assert len(result.data["lines"]) == 1
    assert result.data["lines"][0]["quantity"] == 3


async def test_same_drink_in_two_sizes_is_two_lines(shop):
    session, visit_id = shop

    await add_to_cart(session, visit_id, "Latte", 1, "small")
    result = await add_to_cart(session, visit_id, "Latte", 1, "large")

    assert len(result.data["lines"]) == 2
    assert result.data["total_cents"] == 400 + 520


async def test_remove_takes_the_line_off(shop):
    session, visit_id = shop
    await add_to_cart(session, visit_id, "Latte", 2, "small")

    result = await remove_from_cart(session, visit_id, "Latte")

    assert result.ok is True
    assert result.data["lines"] == []


async def test_remove_some_of_a_line(shop):
    session, visit_id = shop
    await add_to_cart(session, visit_id, "Latte", 3, "small")

    result = await remove_from_cart(session, visit_id, "Latte", quantity=1)

    assert result.data["lines"][0]["quantity"] == 2


async def test_remove_asks_which_size_when_ambiguous(shop):
    session, visit_id = shop
    await add_to_cart(session, visit_id, "Latte", 1, "small")
    await add_to_cart(session, visit_id, "Latte", 1, "large")

    result = await remove_from_cart(session, visit_id, "Latte")

    assert result.error == "size_ambiguous"
    assert result.message.endswith("?")


async def test_remove_something_not_ordered(shop):
    session, visit_id = shop

    result = await remove_from_cart(session, visit_id, "Latte")

    assert result.error == "not_in_cart"


async def test_change_size_reprices_the_line(shop):
    session, visit_id = shop
    await add_to_cart(session, visit_id, "Latte", 1, "small")

    result = await change_size(session, visit_id, "Latte", "small", "large")

    assert result.ok is True
    assert result.data["difference_cents"] == 120
    assert result.data["total_cents"] == 520
    assert "more" in result.message


async def test_change_size_downward_reports_less(shop):
    session, visit_id = shop
    await add_to_cart(session, visit_id, "Latte", 1, "large")

    result = await change_size(session, visit_id, "Latte", "large", "small")

    assert result.data["difference_cents"] == -120
    assert "less" in result.message


async def test_change_size_merges_into_an_existing_line(shop):
    session, visit_id = shop
    await add_to_cart(session, visit_id, "Latte", 1, "small")
    await add_to_cart(session, visit_id, "Latte", 2, "large")

    result = await change_size(session, visit_id, "Latte", "small", "large")

    assert len(result.data["lines"]) == 1
    assert result.data["lines"][0]["quantity"] == 3


async def test_change_size_of_something_not_ordered(shop):
    session, visit_id = shop

    result = await change_size(session, visit_id, "Latte", "small", "large")

    assert result.error == "not_in_cart"


async def test_change_size_of_food_is_rejected(shop):
    session, visit_id = shop

    result = await change_size(session, visit_id, "Croissant", "small", "large")

    assert result.error == "size_not_applicable"


async def test_get_cart_on_an_empty_cart(shop):
    session, visit_id = shop

    result = await get_cart(session, visit_id)

    assert result.ok is True
    assert result.data == {"lines": [], "total_cents": 0}

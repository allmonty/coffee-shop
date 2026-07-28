"""Identity is just a name (spec §4.1), and entering opens or resumes a visit."""

import asyncio

import pytest
from sqlalchemy import func, select

from shop.models import User, Visit
from shop.seed import seed_catalog
from shop.service import enter, normalize_name, weekday_for


@pytest.fixture
async def seeded(session):
    await seed_catalog(session)
    return session


@pytest.mark.parametrize(
    "raw",
    [" Allan ", "allan", "ALLAN", "AlLaN", "  allan  ", "Allan\t"],
)
async def test_variants_of_one_name_are_one_customer(seeded, raw):
    first = await enter(seeded, "Allan")
    again = await enter(seeded, raw)

    assert again.data["user_id"] == first.data["user_id"]
    total = await seeded.scalar(select(func.count()).select_from(User))
    assert total == 1


async def test_internal_whitespace_is_collapsed(seeded):
    first = await enter(seeded, "Mary Jane")
    again = await enter(seeded, "Mary    Jane")

    assert again.data["user_id"] == first.data["user_id"]


async def test_different_names_are_different_customers(seeded):
    allan = await enter(seeded, "Allan")
    mary = await enter(seeded, "Mary")

    assert allan.data["user_id"] != mary.data["user_id"]


async def test_display_name_follows_the_latest_spelling(seeded):
    await enter(seeded, "allan")
    result = await enter(seeded, "Allan")

    assert result.data["name"] == "Allan"


async def test_first_visit_is_flagged_new(seeded):
    first = await enter(seeded, "Allan")
    assert first.data["is_new"] is True

    # Same visit resumed, so still not a new *customer*.
    again = await enter(seeded, "Allan")
    assert again.data["is_new"] is False


@pytest.mark.parametrize("raw", ["", "   ", "\t\n", "123", "!!!", "x" * 41])
async def test_invalid_names_are_rejected(seeded, raw):
    result = await enter(seeded, raw)

    assert result.ok is False
    assert result.error == "invalid_name"
    assert result.message  # something the barista can actually say


async def test_normalize_name_rejects_the_same_set():
    assert normalize_name("Allan") == "allan"
    assert normalize_name("") is None
    assert normalize_name("42") is None
    assert normalize_name("x" * 41) is None


async def test_new_customer_starts_with_twenty_dollars_on_day_one(seeded):
    result = await enter(seeded, "Allan")

    assert result.data["wallet_cents"] == 2000
    assert result.data["day"] == 1
    assert result.data["weekday"] == "Monday"


async def test_entering_twice_resumes_the_same_visit(seeded):
    """Closing the tab and coming back must not open a second visit, or the
    checkpointed conversation is stranded and the day gets a second wallet."""
    first = await enter(seeded, "Allan")
    again = await enter(seeded, "Allan")

    assert again.data["visit_id"] == first.data["visit_id"]
    visits = await seeded.scalar(select(func.count()).select_from(Visit))
    assert visits == 1


async def test_resumed_visit_keeps_the_same_menu(seeded):
    """Today's menu is drawn once; the barista must not start offering
    different things because the customer refreshed the page (spec §3.2)."""
    first = await enter(seeded, "Allan")
    again = await enter(seeded, "Allan")

    assert again.data["menu"] == first.data["menu"]


async def test_todays_menu_is_a_subset_not_the_catalog(seeded):
    result = await enter(seeded, "Allan")
    menu = result.data["menu"]

    drinks = [item for item in menu if item["category"] == "drink"]
    foods = [item for item in menu if item["category"] == "food"]

    assert 5 <= len(drinks) <= 7
    assert 3 <= len(foods) <= 5
    assert len(menu) < 30


async def test_menu_marks_drinks_as_sized_and_food_as_not(seeded):
    result = await enter(seeded, "Allan")

    for item in result.data["menu"]:
        assert item["sized"] == (item["category"] == "drink"), item["name"]


def test_weekday_wraps_weekly():
    assert weekday_for(1) == "Monday"
    assert weekday_for(5) == "Friday"
    assert weekday_for(7) == "Sunday"
    assert weekday_for(8) == "Monday"
    assert weekday_for(15) == "Monday"


async def test_concurrent_enter_does_not_duplicate_the_customer(session_factory):
    """Two tabs hitting a brand new name at once.

    This is why find-or-create is ON CONFLICT DO NOTHING plus a select, rather
    than check-then-insert.

    Deliberately does NOT take the `seeded` fixture. That fixture holds an open
    transaction with uncommitted menu_items, so seeding again on a second
    connection would block on the same unique names until the test ended — a
    deadlock, not a test failure.
    """
    async with session_factory() as a, session_factory() as b:
        await seed_catalog(a)
        results = await asyncio.gather(
            enter(a, "Raceface"),
            enter(b, "Raceface"),
            return_exceptions=True,
        )

    for result in results:
        assert not isinstance(result, Exception), result

    async with session_factory() as check:
        count = await check.scalar(
            select(func.count()).select_from(User).where(User.name_key == "raceface")
        )
        assert count == 1

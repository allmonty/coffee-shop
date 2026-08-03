"""Cart behaviour, especially the four distinct add_to_cart failures.

Each error is a different truth the barista has to tell the customer, so each
gets its own test — a single generic "rejected" would let them collapse.
"""

import uuid

import pytest
from sqlalchemy import select

from shop.models import Cart, MenuItem, VisitMenuItem
from shop.seed import seed_catalog
from shop.service import (
    add_to_cart,
    change_modifiers,
    change_size,
    enter,
    get_cart,
    remove_from_cart,
)


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


# --- modifiers (spec §3.6) -------------------------------------------------
#
# The mirror image of sizes: optional rather than required, so there is no
# `modifier_required`. A modifier request can only be over-specified, never
# under-specified.


async def test_add_a_drink_with_a_modifier_charges_the_surcharge(shop):
    session, visit_id = shop

    result = await add_to_cart(session, visit_id, "Latte", 1, "large", ["oat_milk"])

    assert result.ok is True
    assert result.data["total_cents"] == 580  # 400 base + 120 large + 60 oat


async def test_two_modifiers_stack_on_one_line(shop):
    session, visit_id = shop

    result = await add_to_cart(session, visit_id, "Latte", 1, "large", ["oat_milk", "extra_shot"])

    assert result.data["total_cents"] == 680


async def test_same_drink_and_modifiers_twice_stacks_onto_one_line(shop):
    session, visit_id = shop

    await add_to_cart(session, visit_id, "Latte", 1, "large", ["oat_milk"])
    result = await add_to_cart(session, visit_id, "Latte", 1, "large", ["oat_milk"])

    assert len(result.data["lines"]) == 1
    assert result.data["lines"][0]["quantity"] == 2


async def test_modifier_order_does_not_create_a_second_line(shop):
    """The canonicalisation proof, at the service level."""
    session, visit_id = shop

    await add_to_cart(session, visit_id, "Latte", 1, "large", ["oat_milk", "extra_shot"])
    result = await add_to_cart(session, visit_id, "Latte", 1, "large", ["extra_shot", "oat_milk"])

    assert len(result.data["lines"]) == 1
    assert result.data["lines"][0]["quantity"] == 2


async def test_same_drink_with_different_modifiers_is_two_lines(shop):
    session, visit_id = shop

    await add_to_cart(session, visit_id, "Latte", 1, "large", ["oat_milk"])
    result = await add_to_cart(session, visit_id, "Latte", 1, "large", ["extra_shot"])

    assert len(result.data["lines"]) == 2
    assert result.data["total_cents"] == 580 + 620


async def test_whole_milk_is_the_same_line_as_no_modifier(shop):
    """Whole milk is the drink as listed, so it must not split the line."""
    session, visit_id = shop

    await add_to_cart(session, visit_id, "Latte", 1, "large")
    result = await add_to_cart(session, visit_id, "Latte", 1, "large", ["whole_milk"])

    assert len(result.data["lines"]) == 1
    assert result.data["total_cents"] == 1040


async def test_food_with_a_modifier_is_rejected(shop):
    session, visit_id = shop

    result = await add_to_cart(session, visit_id, "Croissant", 1, None, ["oat_milk"])

    assert result.error == "modifier_not_applicable"


async def test_unknown_modifier_asks_which_one(shop):
    """Not a dead end: the message is the question the barista reads aloud."""
    session, visit_id = shop

    result = await add_to_cart(session, visit_id, "Latte", 1, "large", ["soy_milk"])

    assert result.error == "unknown_modifier"
    assert result.message.endswith("?")


async def test_two_milks_in_one_cup_are_rejected(shop):
    session, visit_id = shop

    result = await add_to_cart(session, visit_id, "Latte", 1, "large", ["oat_milk", "almond_milk"])

    assert result.error == "modifier_conflict"


async def test_size_is_asked_before_an_unknown_modifier_is_judged(shop):
    """`size_required` is the only branch that asks a question, so it comes first.

    "a latte with soy" should come back "which size?", letting the model retry
    knowing both facts rather than fixing one problem at a time.
    """
    session, visit_id = shop

    result = await add_to_cart(session, visit_id, "Latte", 1, None, ["soy_milk"])

    assert result.error == "size_required"


async def test_a_bad_size_is_reported_before_a_bad_modifier(shop):
    """The same ordering seen from the other side."""
    session, visit_id = shop

    result = await add_to_cart(session, visit_id, "Croissant", 1, "large", ["oat_milk"])

    assert result.error == "size_not_applicable"


async def test_remove_picks_the_line_by_modifiers(shop):
    session, visit_id = shop
    await add_to_cart(session, visit_id, "Latte", 1, "large", ["oat_milk"])
    await add_to_cart(session, visit_id, "Latte", 1, "large", ["extra_shot"])

    result = await remove_from_cart(session, visit_id, "Latte", "large", None, ["oat_milk"])

    assert result.ok is True
    assert len(result.data["lines"]) == 1
    assert result.data["lines"][0]["modifiers"] == ["extra_shot"]


async def test_remove_asks_which_modifier_when_ambiguous(shop):
    session, visit_id = shop
    await add_to_cart(session, visit_id, "Latte", 1, "large", ["oat_milk"])
    await add_to_cart(session, visit_id, "Latte", 1, "large")

    result = await remove_from_cart(session, visit_id, "Latte", "large")

    assert result.error == "modifier_ambiguous"
    assert "plain" in result.message


async def test_remove_asks_which_size_first_when_both_differ(shop):
    """One question at a time, and size is the coarser one."""
    session, visit_id = shop
    await add_to_cart(session, visit_id, "Latte", 1, "large", ["oat_milk"])
    await add_to_cart(session, visit_id, "Latte", 1, "small")

    result = await remove_from_cart(session, visit_id, "Latte")

    assert result.error == "size_ambiguous"


async def test_remove_with_an_empty_modifier_list_does_not_filter(shop):
    """`[]` means "nothing to say about extras", not "the plain one".

    Reading it as a filter would delete the wrong line whenever a model padded
    the call with an empty list.
    """
    session, visit_id = shop
    await add_to_cart(session, visit_id, "Latte", 1, "large", ["oat_milk"])

    result = await remove_from_cart(session, visit_id, "Latte", "large", None, [])

    assert result.ok is True
    assert result.data["lines"] == []


async def test_change_size_keeps_the_modifiers(shop):
    session, visit_id = shop
    await add_to_cart(session, visit_id, "Latte", 1, "small", ["oat_milk"])

    result = await change_size(session, visit_id, "Latte", "small", "large")

    assert result.ok is True
    assert result.data["lines"][0]["modifiers"] == ["oat_milk"]
    assert result.data["total_cents"] == 580


async def test_change_size_quotes_only_the_size_difference(shop):
    """The modifier surcharge is on both sides and cancels."""
    session, visit_id = shop
    await add_to_cart(session, visit_id, "Latte", 1, "small", ["oat_milk"])

    result = await change_size(session, visit_id, "Latte", "small", "large")

    assert result.data["difference_cents"] == 120


async def test_change_size_merges_only_into_a_line_with_the_same_modifiers(shop):
    """The 60c-loss guard.

    Merging an oat latte into a plain one would charge the plain price and pass
    every test written before modifiers existed.
    """
    session, visit_id = shop
    await add_to_cart(session, visit_id, "Latte", 1, "small", ["oat_milk"])
    await add_to_cart(session, visit_id, "Latte", 1, "large")

    result = await change_size(session, visit_id, "Latte", "small", "large")

    assert len(result.data["lines"]) == 2
    assert result.data["total_cents"] == 520 + 580


async def test_change_size_asks_which_when_two_modifier_variants_exist(shop):
    """`.scalar()` would have silently resized an arbitrary one."""
    session, visit_id = shop
    await add_to_cart(session, visit_id, "Latte", 1, "small", ["oat_milk"])
    await add_to_cart(session, visit_id, "Latte", 1, "small")

    result = await change_size(session, visit_id, "Latte", "small", "large")

    assert result.error == "modifier_ambiguous"


# --- change_modifiers (spec §13.11) ----------------------------------------
#
# The modifier twin of change_size: one step in the trace instead of a
# remove-then-re-add whose middle state is a cart the customer never asked for.


async def test_change_modifiers_adds_an_extra_and_reprices(shop):
    session, visit_id = shop
    await add_to_cart(session, visit_id, "Latte", 1, "large")

    result = await change_modifiers(session, visit_id, "Latte", ["oat_milk"])

    assert result.ok is True
    assert result.data["total_cents"] == 580
    assert result.data["difference_cents"] == 60


async def test_change_modifiers_can_make_a_drink_plain_again(shop):
    """An empty list means "no extras", unlike remove_from_cart where it means
    "nothing to say about extras"."""
    session, visit_id = shop
    await add_to_cart(session, visit_id, "Latte", 1, "large", ["oat_milk"])

    result = await change_modifiers(session, visit_id, "Latte", [])

    assert result.data["lines"][0]["modifiers"] == []
    assert result.data["total_cents"] == 520
    assert result.data["difference_cents"] == -60


async def test_change_modifiers_keeps_the_quantity(shop):
    session, visit_id = shop
    await add_to_cart(session, visit_id, "Latte", 3, "large")

    result = await change_modifiers(session, visit_id, "Latte", ["extra_shot"])

    assert result.data["lines"][0]["quantity"] == 3
    assert result.data["total_cents"] == 3 * 620


async def test_change_modifiers_merges_into_an_identical_line(shop):
    """Otherwise the unique index would reject the update."""
    session, visit_id = shop
    await add_to_cart(session, visit_id, "Latte", 1, "large", ["oat_milk"])
    await add_to_cart(session, visit_id, "Latte", 2, "large")

    result = await change_modifiers(session, visit_id, "Latte", ["oat_milk"], from_modifiers=[])

    assert len(result.data["lines"]) == 1
    assert result.data["lines"][0]["quantity"] == 3


async def test_change_modifiers_rejects_an_unknown_code(shop):
    session, visit_id = shop
    await add_to_cart(session, visit_id, "Latte", 1, "large")

    result = await change_modifiers(session, visit_id, "Latte", ["soy_milk"])

    assert result.error == "unknown_modifier"
    assert result.message.endswith("?")


async def test_change_modifiers_rejects_two_milks(shop):
    session, visit_id = shop
    await add_to_cart(session, visit_id, "Latte", 1, "large")

    result = await change_modifiers(session, visit_id, "Latte", ["oat_milk", "almond_milk"])

    assert result.error == "modifier_conflict"


async def test_change_modifiers_on_food_is_rejected(shop):
    session, visit_id = shop
    await add_to_cart(session, visit_id, "Croissant", 1)

    result = await change_modifiers(session, visit_id, "Croissant", ["oat_milk"])

    assert result.error == "modifier_not_applicable"


async def test_change_modifiers_on_something_not_ordered(shop):
    session, visit_id = shop

    result = await change_modifiers(session, visit_id, "Latte", ["oat_milk"])

    assert result.error == "not_in_cart"


async def test_change_modifiers_asks_which_size_when_two_are_in_the_cart(shop):
    session, visit_id = shop
    await add_to_cart(session, visit_id, "Latte", 1, "small")
    await add_to_cart(session, visit_id, "Latte", 1, "large")

    result = await change_modifiers(session, visit_id, "Latte", ["oat_milk"])

    assert result.error == "size_ambiguous"


async def test_change_modifiers_picks_a_line_with_from_modifiers(shop):
    """Without it, a cart holding the same drink twice at one size could only
    answer modifier_ambiguous — which also left the merge branch unreachable."""
    session, visit_id = shop
    await add_to_cart(session, visit_id, "Latte", 1, "large")
    await add_to_cart(session, visit_id, "Latte", 1, "large", ["extra_shot"])

    result = await change_modifiers(
        session, visit_id, "Latte", ["oat_milk"], from_modifiers=["extra_shot"]
    )

    assert result.ok is True
    mods = sorted(line["modifiers"] for line in result.data["lines"])
    assert mods == [[], ["oat_milk"]]


async def test_change_modifiers_asks_which_variant_when_two_share_a_size(shop):
    session, visit_id = shop
    await add_to_cart(session, visit_id, "Latte", 1, "large")
    await add_to_cart(session, visit_id, "Latte", 1, "large", ["extra_shot"])

    result = await change_modifiers(session, visit_id, "Latte", ["oat_milk"])

    assert result.error == "modifier_ambiguous"


async def test_change_modifiers_takes_a_size_to_disambiguate(shop):
    session, visit_id = shop
    await add_to_cart(session, visit_id, "Latte", 1, "small")
    await add_to_cart(session, visit_id, "Latte", 1, "large")

    result = await change_modifiers(session, visit_id, "Latte", ["oat_milk"], size="large")

    assert result.ok is True
    large = [line for line in result.data["lines"] if line["size"] == "large"][0]
    assert large["modifiers"] == ["oat_milk"]


async def test_change_modifiers_bumps_the_cart_version(shop):
    """A cart edit must invalidate a prior order's idempotency key."""
    session, visit_id = shop
    await add_to_cart(session, visit_id, "Latte", 1, "large")
    before = (await get_cart(session, visit_id)).ok

    await change_modifiers(session, visit_id, "Latte", ["oat_milk"])
    after = await session.scalar(select(Cart.version).where(Cart.visit_id == visit_id))

    assert before is True
    assert after >= 3

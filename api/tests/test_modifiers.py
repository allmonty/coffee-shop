"""The canonical key, which is the one modifier rule the database cannot enforce.

The schema guarantees the column's shape and its uniqueness; only
`shop.modifiers.canonical_key` guarantees that the same set of codes always
produces the same string. If that slips, "oat then shot" and "shot then oat"
become two cart lines and the merge-or-increment quietly stops working.
"""

import pytest

from shop.catalog_data import MODIFIERS
from shop.models import DrinkModifier, MenuItem
from shop.modifiers import canonical_key, describe, parse_key
from shop.pricing import Deltas, unit_price_cents


@pytest.fixture
def catalog() -> dict[str, DrinkModifier]:
    return {
        code: DrinkModifier(
            code=code,
            delta_cents=delta,
            exclusive_group=group,
            is_default=is_default,
            available=True,
        )
        for code, delta, group, is_default in MODIFIERS
    }


def test_no_modifiers_is_the_empty_string(catalog):
    """Not NULL: the column is NOT NULL so the unique index has no NULL hole."""
    assert canonical_key(None, catalog) == ""
    assert canonical_key([], catalog) == ""


def test_the_key_is_sorted_so_two_orderings_are_one_line(catalog):
    first = canonical_key(["oat_milk", "extra_shot"], catalog)
    second = canonical_key(["extra_shot", "oat_milk"], catalog)

    assert first == second == "extra_shot,oat_milk"


def test_repeated_codes_collapse(catalog):
    """A small model repeating itself is not a distinct truth."""
    assert canonical_key(["oat_milk", "oat_milk"], catalog) == "oat_milk"


def test_default_modifiers_drop_out_of_the_key(catalog):
    """ "A latte" and "a latte with regular milk" must be the same cart line."""
    assert canonical_key(["whole_milk"], catalog) == ""
    assert canonical_key(["whole_milk", "extra_shot"], catalog) == "extra_shot"


def test_loose_spelling_is_normalised(catalog):
    assert canonical_key([" Oat Milk "], catalog) == "oat_milk"
    assert canonical_key(["extra-shot"], catalog) == "extra_shot"


def test_an_unknown_code_survives_canonicalisation(catalog):
    """Validation belongs to add_to_cart.

    Dropping a code here would charge the customer for a drink they did not
    order, and do it silently.
    """
    assert canonical_key(["soy_milk"], catalog) == "soy_milk"


def test_parse_key_round_trips(catalog):
    key = canonical_key(["oat_milk", "extra_shot"], catalog)
    assert parse_key(key) == ("extra_shot", "oat_milk")
    assert parse_key("") == ()


def test_describe_is_written_to_be_read_aloud():
    assert describe("extra_shot,oat_milk") == "extra shot, oat milk"


def test_pricing_raises_on_a_code_with_no_price(catalog):
    """Silently pricing an unknown code at $0 is the worst available outcome."""
    latte = MenuItem(name="Latte", category="drink", price_cents=400, sized=True)
    deltas = Deltas(size={"large": 120}, modifiers=catalog)

    with pytest.raises(KeyError):
        unit_price_cents(latte, "large", "soy_milk", deltas)


def test_food_is_never_charged_a_surcharge(catalog):
    """`sized` is the authority, not the presence of arguments."""
    cookie = MenuItem(name="Cookie", category="food", price_cents=200, sized=False)
    deltas = Deltas(size={"large": 120}, modifiers=catalog)

    assert unit_price_cents(cookie, "large", "oat_milk", deltas) == 200


def test_offerable_hides_defaults_and_retired_codes(catalog):
    catalog["almond_milk"].available = False
    deltas = Deltas(size={}, modifiers=catalog)

    assert deltas.offerable() == {"oat_milk": 60, "extra_shot": 100}

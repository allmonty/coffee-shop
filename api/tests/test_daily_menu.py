"""The daily draw is random, so examples are not enough — these run it a few
thousand times and assert the guarantees on every single result (spec §3.2).
"""

import random

import pytest

from shop.catalog_data import DRINKS, FOODS
from shop.daily_menu import (
    CHEAPEST_ITEM_MAX_CENTS,
    MAX_DRINKS,
    MAX_FOODS,
    MIN_DRINKS,
    MIN_FOODS,
    REQUIRED_ROUNDS,
    CatalogTooExpensive,
    check_guarantees,
    draw_daily_menu,
)
from shop.models import DRINK, FOOD, MenuItem

WALLET = 2000
DRAWS = 5000


def _catalog() -> list[MenuItem]:
    """In-memory catalog — no database needed to test the generator."""
    items = []
    for index, (name, price) in enumerate(DRINKS, start=1):
        items.append(MenuItem(id=index, name=name, category=DRINK, price_cents=price, sized=True))
    offset = len(DRINKS)
    for index, (name, price) in enumerate(FOODS, start=offset + 1):
        items.append(MenuItem(id=index, name=name, category=FOOD, price_cents=price, sized=False))
    return items


def test_every_draw_satisfies_all_four_guarantees():
    catalog = _catalog()
    rng = random.Random(20260727)

    for draw in range(DRAWS):
        menu = draw_daily_menu(catalog, wallet_cents=WALLET, rng=rng)
        violations = check_guarantees(menu, wallet_cents=WALLET).violations()
        assert not violations, f"draw {draw}: {violations}"


def test_every_draw_respects_the_size_bounds():
    catalog = _catalog()
    rng = random.Random(11)

    for _ in range(DRAWS):
        menu = draw_daily_menu(catalog, wallet_cents=WALLET, rng=rng)
        drinks = sum(1 for item in menu if item.category == DRINK)
        foods = sum(1 for item in menu if item.category == FOOD)

        assert MIN_DRINKS <= drinks <= MAX_DRINKS
        assert MIN_FOODS <= foods <= MAX_FOODS


def test_draws_never_repeat_an_item():
    catalog = _catalog()
    rng = random.Random(7)

    for _ in range(500):
        menu = draw_daily_menu(catalog, wallet_cents=WALLET, rng=rng)
        names = [item.name for item in menu]
        assert len(names) == len(set(names))


def test_draws_actually_vary():
    """Guards against an 'always the same menu' regression, which would still
    satisfy every guarantee above."""
    catalog = _catalog()
    rng = random.Random(3)

    seen = {
        tuple(sorted(item.name for item in draw_daily_menu(catalog, wallet_cents=WALLET, rng=rng)))
        for _ in range(50)
    }
    assert len(seen) > 1


def test_two_full_rounds_are_always_affordable():
    """G3 restated as the thing the customer experiences."""
    catalog = _catalog()
    rng = random.Random(99)

    for _ in range(DRAWS):
        menu = draw_daily_menu(catalog, wallet_cents=WALLET, rng=rng)
        cheapest_drink = min(i.price_cents for i in menu if i.category == DRINK)
        cheapest_food = min(i.price_cents for i in menu if i.category == FOOD)

        assert REQUIRED_ROUNDS * (cheapest_drink + cheapest_food) <= WALLET


def test_change_is_always_spendable():
    catalog = _catalog()
    rng = random.Random(1234)

    for _ in range(DRAWS):
        menu = draw_daily_menu(catalog, wallet_cents=WALLET, rng=rng)
        assert min(item.price_cents for item in menu) <= CHEAPEST_ITEM_MAX_CENTS


def test_all_expensive_catalog_raises_rather_than_serving_it():
    """The failure that matters: a catalog with no cheap anchors must not
    silently produce an unaffordable day."""
    expensive = [
        MenuItem(id=i, name=f"Drink {i}", category=DRINK, price_cents=900, sized=True)
        for i in range(1, 9)
    ] + [
        MenuItem(id=100 + i, name=f"Cake {i}", category=FOOD, price_cents=800, sized=False)
        for i in range(1, 7)
    ]

    with pytest.raises(CatalogTooExpensive):
        draw_daily_menu(expensive, wallet_cents=WALLET)


def test_catalog_too_small_raises():
    tiny = [
        MenuItem(id=1, name="Filter Coffee", category=DRINK, price_cents=175, sized=True),
        MenuItem(id=2, name="Shortbread", category=FOOD, price_cents=175, sized=False),
    ]

    with pytest.raises(CatalogTooExpensive):
        draw_daily_menu(tiny, wallet_cents=WALLET)


def test_guarantee_report_names_what_failed():
    """A violation should say which guarantee broke, not just that one did."""
    menu = [
        MenuItem(id=1, name="Affogato", category=DRINK, price_cents=650, sized=True),
        MenuItem(id=2, name="Carrot Cake", category=FOOD, price_cents=500, sized=False),
    ]

    violations = check_guarantees(menu, wallet_cents=WALLET).violations()

    assert any("G1" in v for v in violations)
    assert any("G2" in v for v in violations)
    assert any("G4" in v for v in violations)

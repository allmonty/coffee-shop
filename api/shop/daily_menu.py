"""Draw today's menu from the catalog (spec §3.2).

The affordability rule the customer actually feels is G3: $20 must always buy at
least two full drink-and-food rounds. Everything else here exists to make that
true by construction.

**Constructive, not sample-and-retry.** Pick the guaranteed-cheap anchors first,
then fill the remaining slots at random. Rejection sampling would also work, but
it can spin forever if someone edits the catalog so the invariants become
unsatisfiable — and it fails at runtime, long after the bad edit. Constructing a
menu that cannot violate the guarantees, then asserting them, puts the failure in
the seed tests where it belongs.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from shop.models import DRINK, FOOD, MenuItem

# Anchor thresholds. An item at or under these prices is always drawable, which
# is what keeps G2/G3/G4 satisfiable.
CHEAP_DRINK_MAX_CENTS = 300
CHEAP_FOOD_MAX_CENTS = 250
CHEAPEST_ITEM_MAX_CENTS = 200

MIN_DRINKS, MAX_DRINKS = 5, 7
MIN_FOODS, MAX_FOODS = 3, 5

# G3: the wallet must cover this many full drink-and-food rounds.
REQUIRED_ROUNDS = 2


class CatalogTooExpensive(Exception):
    """The catalog cannot produce an affordable day.

    Raised at draw time, but it is really a seed-data bug: someone priced the
    cheap end of the catalog out of reach. Failing loudly beats silently serving
    a day the customer cannot afford.
    """


@dataclass(frozen=True)
class MenuGuarantees:
    """G1–G4 from spec §3.2, checked against a drawn menu."""

    drinks: int
    foods: int
    cheapest_drink_cents: int
    cheapest_food_cents: int
    cheapest_item_cents: int
    wallet_cents: int

    @property
    def g1_enough_choice(self) -> bool:
        return self.drinks >= MIN_DRINKS and self.foods >= MIN_FOODS

    @property
    def g2_something_cheap(self) -> bool:
        return (
            self.cheapest_drink_cents <= CHEAP_DRINK_MAX_CENTS
            and self.cheapest_food_cents <= CHEAP_FOOD_MAX_CENTS
        )

    @property
    def g3_two_rounds_affordable(self) -> bool:
        round_cents = self.cheapest_drink_cents + self.cheapest_food_cents
        return REQUIRED_ROUNDS * round_cents <= self.wallet_cents

    @property
    def g4_change_is_spendable(self) -> bool:
        return self.cheapest_item_cents <= CHEAPEST_ITEM_MAX_CENTS

    def violations(self) -> list[str]:
        failed = []
        if not self.g1_enough_choice:
            failed.append(f"G1: {self.drinks} drinks, {self.foods} foods")
        if not self.g2_something_cheap:
            failed.append(
                f"G2: cheapest drink {self.cheapest_drink_cents}c, "
                f"cheapest food {self.cheapest_food_cents}c"
            )
        if not self.g3_two_rounds_affordable:
            failed.append(
                f"G3: {REQUIRED_ROUNDS} rounds cost "
                f"{REQUIRED_ROUNDS * (self.cheapest_drink_cents + self.cheapest_food_cents)}c "
                f"but the wallet holds {self.wallet_cents}c"
            )
        if not self.g4_change_is_spendable:
            failed.append(f"G4: cheapest item {self.cheapest_item_cents}c")
        return failed


def check_guarantees(menu: list[MenuItem], *, wallet_cents: int) -> MenuGuarantees:
    """Measure a menu against G1–G4. Prices are base (small) prices — the
    cheapest a customer can ever pay, since sizing up is their choice."""
    drinks = [item for item in menu if item.category == DRINK]
    foods = [item for item in menu if item.category == FOOD]

    return MenuGuarantees(
        drinks=len(drinks),
        foods=len(foods),
        cheapest_drink_cents=min((d.price_cents for d in drinks), default=10**9),
        cheapest_food_cents=min((f.price_cents for f in foods), default=10**9),
        cheapest_item_cents=min((i.price_cents for i in menu), default=10**9),
        wallet_cents=wallet_cents,
    )


def draw_daily_menu(
    catalog: list[MenuItem],
    *,
    wallet_cents: int,
    rng: random.Random | None = None,
) -> list[MenuItem]:
    """Return today's menu: 5–7 drinks and 3–5 foods satisfying G1–G4."""
    rng = rng or random.Random()

    drinks = [item for item in catalog if item.category == DRINK]
    foods = [item for item in catalog if item.category == FOOD]

    cheap_drinks = [d for d in drinks if d.price_cents <= CHEAP_DRINK_MAX_CENTS]
    cheap_foods = [f for f in foods if f.price_cents <= CHEAP_FOOD_MAX_CENTS]

    if not cheap_drinks or not cheap_foods:
        raise CatalogTooExpensive(
            f"need a drink at <= {CHEAP_DRINK_MAX_CENTS}c and a food at "
            f"<= {CHEAP_FOOD_MAX_CENTS}c; catalog has "
            f"{len(cheap_drinks)} and {len(cheap_foods)}"
        )
    if len(drinks) < MIN_DRINKS or len(foods) < MIN_FOODS:
        raise CatalogTooExpensive(
            f"catalog too small: {len(drinks)} drinks, {len(foods)} foods; "
            f"need at least {MIN_DRINKS} and {MIN_FOODS}"
        )

    # Anchors first — these are what make the guarantees hold by construction.
    anchor_drink = rng.choice(cheap_drinks)
    anchor_food = rng.choice(cheap_foods)

    menu = [anchor_drink, anchor_food]
    menu += _fill(drinks, exclude=anchor_drink, total=rng.randint(MIN_DRINKS, MAX_DRINKS), rng=rng)
    menu += _fill(foods, exclude=anchor_food, total=rng.randint(MIN_FOODS, MAX_FOODS), rng=rng)

    # Must never fire. It is a guard against a catalog edit that slipped past the
    # checks above, not a substitute for them.
    violations = check_guarantees(menu, wallet_cents=wallet_cents).violations()
    if violations:
        raise CatalogTooExpensive("; ".join(violations))

    return menu


def _fill(
    pool: list[MenuItem], *, exclude: MenuItem, total: int, rng: random.Random
) -> list[MenuItem]:
    """Random extras to bring the anchor up to `total` items."""
    remaining = [item for item in pool if item.id != exclude.id]
    return rng.sample(remaining, k=min(total - 1, len(remaining)))

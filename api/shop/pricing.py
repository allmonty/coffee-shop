"""What a line actually costs (spec §3.4, §3.6).

Base price plus a flat size surcharge plus a flat surcharge per modifier. The
model never supplies a price — it names a size and some codes, and this computes
the money. Keeping that one-way is what stops a barista talking a customer into
a cheaper latte.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from shop.models import DrinkModifier, MenuItem, SizeModifier
from shop.modifiers import parse_key


@dataclass(frozen=True)
class Deltas:
    """Every surcharge the shop knows about, loaded once per operation.

    One object rather than two loose dicts so `unit_price_cents` keeps a
    readable signature as axes are added.
    """

    size: Mapping[str, int]
    modifiers: Mapping[str, DrinkModifier]

    def offerable(self) -> dict[str, int]:
        """Code -> surcharge, for the modifiers worth mentioning to a customer.

        Excludes retired codes, and excludes defaults: printing `whole milk
        +$0.00` in the menu block would invite the model to send a code for the
        drink as it already comes.
        """
        return {
            code: row.delta_cents
            for code, row in self.modifiers.items()
            if row.available and not row.is_default
        }


async def load_deltas(session: AsyncSession) -> Deltas:
    """Read both surcharge tables.

    Modifiers are read in full, including retired ones: a cart line or a
    historical order may reference a code that is no longer offered, and
    refusing to price it would be worse than not offering it.
    """
    sizes = (await session.scalars(select(SizeModifier))).all()
    modifiers = (await session.scalars(select(DrinkModifier))).all()
    return Deltas(
        size={row.size: row.delta_cents for row in sizes},
        modifiers={row.code: row for row in modifiers},
    )


def unit_price_cents(
    item: MenuItem,
    size: str | None,
    modifiers: str,
    deltas: Deltas,
) -> int:
    """Price of one unit of `item` at `size` with `modifiers`.

    Food has no size and no modifiers; the `sized` flag on the item is the
    authority, not the presence of arguments.

    An unpriced code raises rather than costing nothing. `add_to_cart` validates
    every code before a line is written, so reaching here with one is a bug —
    and silently undercharging is the worst possible way to find that out.
    """
    if not item.sized:
        return item.price_cents

    total = item.price_cents
    if size is not None:
        total += deltas.size.get(size, 0)
    for code in parse_key(modifiers):
        total += deltas.modifiers[code].delta_cents
    return total


def format_cents(cents: int) -> str:
    """`$4.60`. Used in messages the barista reads aloud."""
    return f"${cents / 100:.2f}"

"""What a line actually costs (spec §3.4).

Base price plus a flat size surcharge. The model never supplies a price — it
names a size, and this computes the money. Keeping that one-way is what stops a
barista talking a customer into a cheaper latte.
"""

from __future__ import annotations

from collections.abc import Mapping

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from shop.models import MenuItem, SizeModifier


async def load_size_deltas(session: AsyncSession) -> dict[str, int]:
    rows = (await session.scalars(select(SizeModifier))).all()
    return {row.size: row.delta_cents for row in rows}


def unit_price_cents(item: MenuItem, size: str | None, deltas: Mapping[str, int]) -> int:
    """Price of one unit of `item` at `size`.

    Food has no size and therefore no surcharge; the `sized` flag on the item is
    the authority, not the presence of a size argument.
    """
    if not item.sized or size is None:
        return item.price_cents
    return item.price_cents + deltas.get(size, 0)


def format_cents(cents: int) -> str:
    """`$4.60`. Used in messages the barista reads aloud."""
    return f"${cents / 100:.2f}"

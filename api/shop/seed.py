"""Seed the catalog and the size modifiers.

Idempotent: runs on every boot and does nothing once the rows exist. Prices of
existing items are left alone so a local price edit survives a restart — the
seed establishes the catalog, it does not own it afterwards.

Lives in `shop/` rather than `alembic/` (where spec §5.3 first put it) because a
module inside the local `alembic/` directory is unimportable: the installed
`alembic` distribution owns that name.
"""

import asyncio

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from db import SessionLocal
from shop.catalog_data import DRINKS, FOODS, MODIFIERS, SIZE_DELTAS
from shop.models import DRINK, FOOD, DrinkModifier, MenuItem, SizeModifier


async def seed_catalog(session: AsyncSession) -> int:
    """Insert any missing catalog rows. Returns how many were added."""
    existing = set((await session.scalars(select(MenuItem.name))).all())

    added = 0
    for name, price_cents in DRINKS:
        if name not in existing:
            session.add(MenuItem(name=name, category=DRINK, price_cents=price_cents, sized=True))
            added += 1
    for name, price_cents in FOODS:
        if name not in existing:
            session.add(MenuItem(name=name, category=FOOD, price_cents=price_cents, sized=False))
            added += 1

    existing_sizes = set((await session.scalars(select(SizeModifier.size))).all())
    for size, delta_cents in SIZE_DELTAS.items():
        if size not in existing_sizes:
            session.add(SizeModifier(size=size, delta_cents=delta_cents))
            added += 1

    existing_modifiers = set((await session.scalars(select(DrinkModifier.code))).all())
    for code, delta_cents, exclusive_group, is_default in MODIFIERS:
        if code not in existing_modifiers:
            session.add(
                DrinkModifier(
                    code=code,
                    delta_cents=delta_cents,
                    exclusive_group=exclusive_group,
                    is_default=is_default,
                )
            )
            added += 1

    await session.commit()
    return added


async def main() -> None:
    async with SessionLocal() as session:
        added = await seed_catalog(session)
    print(f"seed_menu: added {added} rows")


if __name__ == "__main__":
    asyncio.run(main())

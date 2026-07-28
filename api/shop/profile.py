"""What the barista remembers about a customer (spec §6.5).

Everything structured here is **aggregated at read time** from `orders` and
`visits` — favourites, the usual, visit count, last visit. None of it is stored.
Materialising it would buy nothing at this data volume and would add a cache
that goes stale the moment someone forgets to recompute it.

The only stored part is `notes`, and those are model-written. That split is the
rule: never ask an LLM for a fact a GROUP BY can produce.
"""

from __future__ import annotations

import uuid

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from shop.models import (
    DRINK,
    FOOD,
    CustomerPreference,
    MenuItem,
    Order,
    OrderLine,
    User,
    Visit,
    VisitMenuItem,
)


async def customer_profile(
    session: AsyncSession, user_id: uuid.UUID, visit_id: uuid.UUID | None = None
) -> dict[str, object]:
    user = await session.get(User, user_id)
    assert user is not None

    ordered = await _ordered_totals(session, user_id)
    favourite_sizes = await _favourite_size_per_item(session, user_id)
    available = await _available_today(session, visit_id) if visit_id else None

    favourite_drink = next((row for row in ordered if row.category == DRINK), None)
    favourite_food = next((row for row in ordered if row.category == FOOD), None)

    usual = []
    for row in (favourite_drink, favourite_food):
        if row is None:
            continue
        line: dict[str, object] = {
            "item": row.name,
            "size": favourite_sizes.get(row.name),
            "qty": 1,
        }
        if available is not None:
            # The barista must not offer "the usual" without checking it is on
            # today's menu — suggesting something unavailable is worse than not
            # suggesting at all (spec §3.3).
            line["available_today"] = row.name in available
        usual.append(line)

    visit_count = await session.scalar(
        select(func.count()).select_from(Visit).where(Visit.user_id == user_id)
    )
    last_visit_day = await session.scalar(
        select(func.max(Visit.day)).where(Visit.user_id == user_id, Visit.ended_at.isnot(None))
    )

    return {
        "name": user.name,
        "visit_count": visit_count or 0,
        "favorite_drink": favourite_drink.name if favourite_drink else None,
        "favorite_food": favourite_food.name if favourite_food else None,
        "usual_order": usual,
        "last_visit_day": last_visit_day,
        "notes": await get_notes(session, user_id),
    }


async def _ordered_totals(session: AsyncSession, user_id: uuid.UUID):
    """Items this customer has bought, most-bought first."""
    statement = (
        select(
            MenuItem.name.label("name"),
            MenuItem.category.label("category"),
            func.sum(OrderLine.quantity).label("qty"),
        )
        .join(OrderLine, OrderLine.menu_item_id == MenuItem.id)
        .join(Order, Order.id == OrderLine.order_id)
        .where(Order.user_id == user_id)
        .group_by(MenuItem.name, MenuItem.category)
        .order_by(func.sum(OrderLine.quantity).desc(), MenuItem.name)
    )
    return (await session.execute(statement)).all()


async def _favourite_size_per_item(session: AsyncSession, user_id: uuid.UUID) -> dict[str, str]:
    """The size they usually take each drink in.

    This is what lets the barista say "large, like always?" instead of asking a
    regular the same question every day (spec §3.4).
    """
    statement = (
        select(
            MenuItem.name.label("name"),
            OrderLine.size.label("size"),
            func.sum(OrderLine.quantity).label("qty"),
        )
        .join(OrderLine, OrderLine.menu_item_id == MenuItem.id)
        .join(Order, Order.id == OrderLine.order_id)
        .where(Order.user_id == user_id, OrderLine.size.isnot(None))
        .group_by(MenuItem.name, OrderLine.size)
        .order_by(func.sum(OrderLine.quantity).desc())
    )
    favourite: dict[str, str] = {}
    for row in (await session.execute(statement)).all():
        favourite.setdefault(row.name, row.size)
    return favourite


async def _available_today(session: AsyncSession, visit_id: uuid.UUID) -> set[str]:
    names = await session.scalars(
        select(MenuItem.name)
        .join(VisitMenuItem, VisitMenuItem.menu_item_id == MenuItem.id)
        .where(VisitMenuItem.visit_id == visit_id)
    )
    return set(names.all())


async def get_notes(session: AsyncSession, user_id: uuid.UUID) -> list[str]:
    preference = await session.get(CustomerPreference, user_id)
    return list(preference.notes) if preference else []

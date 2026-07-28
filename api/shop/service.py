"""Every business operation, as plain async functions (spec §5.3).

Each takes an `AsyncSession` first and returns a `Result`. No repository
classes, no unit-of-work, no manager objects — the domain here is about ten
operations, and ceremony would hide them.

This module is the single source of truth for the wallet, the menu and orders.
It has no idea an LLM exists.
"""

from __future__ import annotations

import re
import unicodedata
import uuid

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from settings import settings
from shop.daily_menu import draw_daily_menu
from shop.models import MenuItem, User, Visit, VisitMenuItem
from shop.result import Result

WEEKDAYS = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")

MAX_NAME_LENGTH = 40
_WHITESPACE = re.compile(r"\s+")


def normalize_name(raw: str) -> str | None:
    """Fold a typed name to its lookup key, or None if it is not a usable name.

    Trim, collapse internal whitespace, casefold. `" Allan "`, `"allan"` and
    `"ALLAN"` are therefore one customer (spec §4.1). NFKC first so visually
    identical names typed with different Unicode do not create two accounts.
    """
    collapsed = _WHITESPACE.sub(" ", unicodedata.normalize("NFKC", raw)).strip()
    if not collapsed or len(collapsed) > MAX_NAME_LENGTH:
        return None
    if not any(character.isalpha() for character in collapsed):
        return None
    return collapsed.casefold()


def weekday_for(day: int) -> str:
    """Derived, never stored (spec §2.1 rule 8)."""
    return WEEKDAYS[(day - 1) % len(WEEKDAYS)]


async def enter(session: AsyncSession, raw_name: str) -> Result:
    """Find-or-create the customer, then open or resume a visit (spec §7.1).

    The only thing the landing page calls.
    """
    name_key = normalize_name(raw_name)
    if name_key is None:
        return Result.failure(
            "invalid_name",
            "I didn't catch that — what name should I put on the cup?",
        )

    display_name = _WHITESPACE.sub(" ", unicodedata.normalize("NFKC", raw_name)).strip()
    user, is_new = await _find_or_create_user(session, name_key, display_name)
    visit, menu = await _open_or_resume_visit(session, user)
    await session.commit()

    return Result.success(
        user_id=str(user.id),
        name=user.name,
        is_new=is_new,
        visit_id=str(visit.id),
        day=visit.day,
        weekday=weekday_for(visit.day),
        wallet_cents=user.wallet_cents,
        menu=menu,
    )


async def _find_or_create_user(
    session: AsyncSession, name_key: str, display_name: str
) -> tuple[User, bool]:
    """Insert-then-select, never check-then-insert.

    Two browser tabs hitting a new name at once would race a check-then-insert
    into either a duplicate or a 500. `ON CONFLICT DO NOTHING` makes the insert
    a no-op for the loser, and the select afterwards resolves both to the same
    row (spec §7.1).
    """
    statement = (
        pg_insert(User)
        .values(name=display_name, name_key=name_key)
        .on_conflict_do_nothing(index_elements=[User.name_key])
        .returning(User.id)
    )
    inserted_id = await session.scalar(statement)

    user = await session.scalar(select(User).where(User.name_key == name_key))
    assert user is not None  # the insert above guarantees the row exists

    is_new = inserted_id is not None
    if not is_new and user.name != display_name:
        # Greet them however they typed it most recently.
        user.name = display_name

    return user, is_new


async def _open_or_resume_visit(
    session: AsyncSession, user: User
) -> tuple[Visit, list[dict[str, object]]]:
    """Resume an unfinished visit rather than opening a second one.

    Someone who closed the tab instead of going home still has a visit with
    `ended_at IS NULL`. Opening a fresh one would strand its checkpointed
    conversation and hand out a second $20 for the same day (spec §7.1).
    """
    open_visit = await session.scalar(
        select(Visit).where(Visit.user_id == user.id, Visit.ended_at.is_(None))
    )
    if open_visit is not None:
        return open_visit, await todays_menu(session, open_visit.id)

    # Same insert-then-select shape as the user lookup, against the partial
    # unique index on (user_id) WHERE ended_at IS NULL. If a concurrent tab won,
    # `returning` yields nothing and we adopt the visit it created.
    visit_id = await session.scalar(
        pg_insert(Visit)
        .values(id=uuid.uuid4(), user_id=user.id, day=user.current_day)
        .on_conflict_do_nothing(
            index_elements=[Visit.user_id],
            index_where=Visit.ended_at.is_(None),
        )
        .returning(Visit.id)
    )
    if visit_id is None:
        lost_race = await session.scalar(
            select(Visit).where(Visit.user_id == user.id, Visit.ended_at.is_(None))
        )
        assert lost_race is not None
        return lost_race, await todays_menu(session, lost_race.id)

    visit = await session.get(Visit, visit_id)
    assert visit is not None

    catalog = list(
        (await session.scalars(select(MenuItem).where(MenuItem.in_catalog.is_(True)))).all()
    )
    drawn = draw_daily_menu(catalog, wallet_cents=settings.daily_wallet_cents)
    session.add_all(VisitMenuItem(visit_id=visit.id, menu_item_id=item.id) for item in drawn)
    await session.flush()

    return visit, _as_menu_payload(drawn)


async def todays_menu(session: AsyncSession, visit_id: uuid.UUID) -> list[dict[str, object]]:
    """The subset drawn for this visit. Not the catalog (spec §3.2)."""
    items = (
        await session.scalars(
            select(MenuItem)
            .join(VisitMenuItem, VisitMenuItem.menu_item_id == MenuItem.id)
            .where(VisitMenuItem.visit_id == visit_id)
            .order_by(MenuItem.category, MenuItem.price_cents, MenuItem.name)
        )
    ).all()
    return _as_menu_payload(list(items))


def _as_menu_payload(items: list[MenuItem]) -> list[dict[str, object]]:
    return [
        {
            "name": item.name,
            "category": item.category,
            "price_cents": item.price_cents,
            "sized": item.sized,
        }
        for item in sorted(items, key=lambda i: (i.category, i.price_cents, i.name))
    ]

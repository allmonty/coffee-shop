"""The cart: what the customer has asked for but not yet paid for.

Every error here returns a `message` written to be read aloud. `size_required`
in particular is not a failure — it is how the domain tells the agent the
customer's request was incomplete, so the barista asks instead of guessing
(spec §6.4).
"""

from __future__ import annotations

import uuid

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from shop.models import SIZES, Cart, CartLine, MenuItem, Visit, VisitMenuItem
from shop.pricing import format_cents, load_size_deltas, unit_price_cents
from shop.result import Result


async def open_visit(session: AsyncSession, visit_id: uuid.UUID) -> Visit | None:
    visit = await session.get(Visit, visit_id)
    if visit is None or visit.ended_at is not None:
        return None
    return visit


async def _cart_for(session: AsyncSession, visit_id: uuid.UUID) -> Cart:
    cart = await session.scalar(select(Cart).where(Cart.visit_id == visit_id))
    if cart is None:
        cart = Cart(visit_id=visit_id)
        session.add(cart)
        await session.flush()
    return cart


async def _find_in_catalog(session: AsyncSession, item_name: str) -> MenuItem | None:
    """Case-insensitive exact match. The model types what the customer said."""
    return await session.scalar(
        select(MenuItem).where(
            func.lower(MenuItem.name) == item_name.strip().lower(),
            MenuItem.in_catalog.is_(True),
        )
    )


async def _is_on_todays_menu(session: AsyncSession, visit_id: uuid.UUID, item_id: int) -> bool:
    found = await session.scalar(
        select(VisitMenuItem.menu_item_id).where(
            VisitMenuItem.visit_id == visit_id,
            VisitMenuItem.menu_item_id == item_id,
        )
    )
    return found is not None


async def cart_payload(session: AsyncSession, visit_id: uuid.UUID) -> dict[str, object]:
    """Lines with sizes and prices, plus the total (spec §7.2 `cart_updated`)."""
    cart = await session.scalar(select(Cart).where(Cart.visit_id == visit_id))
    if cart is None:
        return {"lines": [], "total_cents": 0}

    deltas = await load_size_deltas(session)
    rows = (
        await session.execute(
            select(CartLine, MenuItem)
            .join(MenuItem, MenuItem.id == CartLine.menu_item_id)
            .where(CartLine.cart_id == cart.id)
            .order_by(CartLine.id)
        )
    ).all()

    lines = []
    total = 0
    for line, item in rows:
        unit = unit_price_cents(item, line.size, deltas)
        line_total = unit * line.quantity
        total += line_total
        lines.append(
            {
                "item": item.name,
                "size": line.size,
                "quantity": line.quantity,
                "unit_price_cents": unit,
                "line_total_cents": line_total,
            }
        )
    return {"lines": lines, "total_cents": total}


async def get_cart(session: AsyncSession, visit_id: uuid.UUID) -> Result:
    if await open_visit(session, visit_id) is None:
        return _visit_closed()
    return Result.success(**await cart_payload(session, visit_id))


async def add_to_cart(
    session: AsyncSession,
    visit_id: uuid.UUID,
    item_name: str,
    quantity: int = 1,
    size: str | None = None,
) -> Result:
    """Four distinct failures, because they are four distinct truths."""
    if await open_visit(session, visit_id) is None:
        return _visit_closed()

    if quantity < 1:
        return Result.failure("invalid_quantity", "How many would you like?")

    item = await _find_in_catalog(session, item_name)
    if item is None:
        return Result.failure(
            "unknown_item",
            f"We don't do {item_name}, I'm afraid.",
        )

    if not await _is_on_todays_menu(session, visit_id, item.id):
        # Real item, just not drawn today. "We're not doing mochas today" is
        # true; "we don't sell mochas" is not (spec §3.3).
        return Result.failure(
            "not_available_today",
            f"No {item.name} today, sorry — it's not on the board.",
        )

    if item.sized and size is None:
        return Result.failure("size_required", "Which size — small, medium, or large?")
    if not item.sized and size is not None:
        return Result.failure(
            "size_not_applicable",
            f"{item.name} only comes the one size.",
        )
    if size is not None and size not in SIZES:
        return Result.failure("unknown_size", "We do small, medium and large.")

    cart = await _cart_for(session, visit_id)
    line = await session.scalar(
        select(CartLine).where(
            CartLine.cart_id == cart.id,
            CartLine.menu_item_id == item.id,
            CartLine.size.is_(None) if size is None else CartLine.size == size,
        )
    )
    if line is None:
        session.add(
            CartLine(
                cart_id=cart.id,
                menu_item_id=item.id,
                quantity=quantity,
                size=size,
                sized=item.sized,
            )
        )
    else:
        line.quantity += quantity

    cart.version += 1
    await session.flush()

    payload = await cart_payload(session, visit_id)
    await session.commit()
    return Result.success(
        f"Added {quantity} {_describe(item.name, size)}.",
        added=item.name,
        size=size,
        quantity=quantity,
        **payload,
    )


async def remove_from_cart(
    session: AsyncSession,
    visit_id: uuid.UUID,
    item_name: str,
    size: str | None = None,
    quantity: int | None = None,
) -> Result:
    """`quantity=None` removes the whole line."""
    if await open_visit(session, visit_id) is None:
        return _visit_closed()

    item = await _find_in_catalog(session, item_name)
    if item is None:
        return Result.failure("unknown_item", f"We don't do {item_name}, I'm afraid.")

    cart = await _cart_for(session, visit_id)
    matches = (
        await session.scalars(
            select(CartLine).where(CartLine.cart_id == cart.id, CartLine.menu_item_id == item.id)
        )
    ).all()
    matches = [line for line in matches if size is None or line.size == size]

    if not matches:
        return Result.failure("not_in_cart", f"There's no {item.name} in the order.")
    if len(matches) > 1:
        sizes = ", ".join(sorted(line.size or "" for line in matches))
        return Result.failure(
            "size_ambiguous",
            f"You've got {item.name} in two sizes ({sizes}) — which one?",
        )

    line = matches[0]
    if quantity is None or quantity >= line.quantity:
        await session.delete(line)
    else:
        line.quantity -= quantity

    cart.version += 1
    await session.flush()

    payload = await cart_payload(session, visit_id)
    await session.commit()
    return Result.success(
        f"Took the {_describe(item.name, line.size)} off.",
        removed=item.name,
        size=line.size,
        **payload,
    )


async def change_size(
    session: AsyncSession,
    visit_id: uuid.UUID,
    item_name: str,
    from_size: str,
    to_size: str,
) -> Result:
    """The size-upsell path in one call, so it reads as one step in the trace."""
    if await open_visit(session, visit_id) is None:
        return _visit_closed()

    if to_size not in SIZES:
        return Result.failure("unknown_size", "We do small, medium and large.")

    item = await _find_in_catalog(session, item_name)
    if item is None:
        return Result.failure("unknown_item", f"We don't do {item_name}, I'm afraid.")
    if not item.sized:
        return Result.failure("size_not_applicable", f"{item.name} only comes the one size.")

    cart = await _cart_for(session, visit_id)
    line = await session.scalar(
        select(CartLine).where(
            CartLine.cart_id == cart.id,
            CartLine.menu_item_id == item.id,
            CartLine.size == from_size,
        )
    )
    if line is None:
        return Result.failure(
            "not_in_cart",
            f"There's no {from_size} {item.name} in the order.",
        )

    existing = await session.scalar(
        select(CartLine).where(
            CartLine.cart_id == cart.id,
            CartLine.menu_item_id == item.id,
            CartLine.size == to_size,
        )
    )
    if existing is None:
        line.size = to_size
    else:
        # Merging keeps the unique (cart, item, size) constraint satisfiable.
        existing.quantity += line.quantity
        await session.delete(line)

    cart.version += 1
    await session.flush()

    deltas = await load_size_deltas(session)
    difference = unit_price_cents(item, to_size, deltas) - unit_price_cents(item, from_size, deltas)
    payload = await cart_payload(session, visit_id)
    await session.commit()
    return Result.success(
        f"Made it a {to_size} {item.name}, {format_cents(abs(difference))} "
        f"{'more' if difference >= 0 else 'less'}.",
        item=item.name,
        from_size=from_size,
        to_size=to_size,
        difference_cents=difference,
        **payload,
    )


def _describe(item_name: str, size: str | None) -> str:
    return f"{size} {item_name}" if size else item_name


def _visit_closed() -> Result:
    return Result.failure("visit_closed", "That visit's already finished.")

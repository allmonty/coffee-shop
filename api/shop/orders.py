"""Paying, and going home.

These are the two operations the model must not be trusted to call on its own,
so the guards live here rather than in the prompt (spec §6.4):

- `place_order` requires the total the barista actually quoted out loud. The
  domain compares it to the real cart total and rejects a mismatch. The model
  cannot lower a charge, only fail one — it is proving it quoted correctly.
- `end_visit` requires explicit confirmation. Ending someone's visit resets
  their wallet and advances the day; it is not a thing to do speculatively.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

from sqlalchemy import delete, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from settings import settings
from shop.cart import cart_payload, open_visit
from shop.models import Cart, CartLine, MenuItem, Order, OrderLine, User, Visit
from shop.pricing import format_cents, load_size_deltas, unit_price_cents
from shop.result import Result


async def get_wallet_balance(session: AsyncSession, visit_id: uuid.UUID) -> Result:
    visit = await open_visit(session, visit_id)
    if visit is None:
        return Result.failure("visit_closed", "That visit's already finished.")

    user = await session.get(User, visit.user_id)
    assert user is not None
    return Result.success(
        wallet_cents=user.wallet_cents,
        day=visit.day,
    )


async def place_order(
    session: AsyncSession, visit_id: uuid.UUID, confirmed_total_cents: int
) -> Result:
    visit = await open_visit(session, visit_id)
    if visit is None:
        return Result.failure("visit_closed", "That visit's already finished.")

    cart = await session.scalar(select(Cart).where(Cart.visit_id == visit_id))
    payload = await cart_payload(session, visit_id)
    lines = payload["lines"]
    total_cents = payload["total_cents"]

    if not lines:
        return Result.failure("empty_cart", "There's nothing in the order yet.")

    if confirmed_total_cents != total_cents:
        # The barista quoted a figure that is not what the cart costs. Refusing
        # is the whole point: it makes "charge without confirming" require
        # guessing the exact total.
        return Result.failure(
            "total_mismatch",
            f"That doesn't add up — the order comes to {format_cents(total_cents)}, "
            f"not {format_cents(confirmed_total_cents)}. Better say it again.",
        )

    user = await session.get(User, visit.user_id)
    assert user is not None
    if user.wallet_cents < total_cents:
        return Result.failure(
            "insufficient_funds",
            f"That's {format_cents(total_cents)} and you've got "
            f"{format_cents(user.wallet_cents)} left today.",
        )

    assert cart is not None
    order_id = await session.scalar(
        pg_insert(Order)
        .values(
            id=uuid.uuid4(),
            user_id=user.id,
            visit_id=visit.id,
            cart_version=cart.version,
            day=visit.day,
            total_cents=total_cents,
        )
        .on_conflict_do_nothing(constraint="uq_orders_visit_cart_version")
        .returning(Order.id)
    )
    if order_id is None:
        # Same cart already charged — a duplicated tool call, not a second
        # order. Return the original rather than charging twice.
        existing = await session.scalar(
            select(Order).where(Order.visit_id == visit.id, Order.cart_version == cart.version)
        )
        assert existing is not None
        return Result.success(
            "That one's already paid for.",
            order_id=str(existing.id),
            total_cents=existing.total_cents,
            duplicate=True,
            wallet_cents=user.wallet_cents,
            lines=lines,
        )

    await _snapshot_lines(session, order_id=order_id, cart_id=cart.id)

    user.wallet_cents -= total_cents
    await session.execute(delete(CartLine).where(CartLine.cart_id == cart.id))
    cart.version += 1
    await session.flush()
    await session.commit()

    return Result.success(
        f"That's {format_cents(total_cents)} — {format_cents(user.wallet_cents)} left for today.",
        order_id=str(order_id),
        total_cents=total_cents,
        wallet_cents=user.wallet_cents,
        lines=lines,
    )


async def _snapshot_lines(session: AsyncSession, *, order_id: uuid.UUID, cart_id: int) -> None:
    """Copy cart lines onto the order at the price actually charged.

    The snapshot includes the size surcharge, so editing the catalog or
    `size_modifiers` later cannot rewrite history.
    """
    deltas = await load_size_deltas(session)
    rows = (
        await session.execute(
            select(CartLine, MenuItem)
            .join(MenuItem, MenuItem.id == CartLine.menu_item_id)
            .where(CartLine.cart_id == cart_id)
        )
    ).all()

    for line, item in rows:
        session.add(
            OrderLine(
                order_id=order_id,
                menu_item_id=item.id,
                quantity=line.quantity,
                size=line.size,
                sized=line.sized,
                unit_price_cents=unit_price_cents(item, line.size, deltas),
            )
        )


async def end_visit(session: AsyncSession, visit_id: uuid.UUID, confirmed: bool) -> Result:
    """Close the visit, advance the day, refill the wallet."""
    visit = await open_visit(session, visit_id)
    if visit is None:
        return Result.failure("visit_closed", "That visit's already finished.")

    if not confirmed:
        return Result.failure(
            "confirmation_required",
            "Heading off, then? Just say the word.",
        )

    user = await session.get(User, visit.user_id)
    assert user is not None

    visit.ended_at = datetime.now(UTC)
    user.current_day += 1
    user.wallet_cents = settings.daily_wallet_cents

    cart = await session.scalar(select(Cart).where(Cart.visit_id == visit_id))
    if cart is not None:
        await session.execute(delete(CartLine).where(CartLine.cart_id == cart.id))

    await session.flush()
    await session.commit()

    return Result.success(
        "See you tomorrow.",
        day=user.current_day,
        wallet_cents=user.wallet_cents,
    )


async def order_history(session: AsyncSession, user_id: uuid.UUID) -> list[dict[str, object]]:
    orders = (
        await session.scalars(
            select(Order).where(Order.user_id == user_id).order_by(Order.placed_at.desc())
        )
    ).all()

    history = []
    for order in orders:
        rows = (
            await session.execute(
                select(OrderLine, MenuItem)
                .join(MenuItem, MenuItem.id == OrderLine.menu_item_id)
                .where(OrderLine.order_id == order.id)
            )
        ).all()
        history.append(
            {
                "order_id": str(order.id),
                "day": order.day,
                "total_cents": order.total_cents,
                "lines": [
                    {
                        "item": item.name,
                        "size": line.size,
                        "quantity": line.quantity,
                        "unit_price_cents": line.unit_price_cents,
                    }
                    for line, item in rows
                ],
            }
        )
    return history


async def visit_for_user(session: AsyncSession, visit_id: uuid.UUID) -> Visit | None:
    return await session.get(Visit, visit_id)

"""The double-charge test that actually exercises the idempotency constraint.

`test_duplicate_place_order_charges_once` in test_orders.py replays sequentially
and hits `empty_cart`, which is safe but never reaches the ON CONFLICT branch.
The failure that constraint exists for is two *concurrent* identical tool calls —
which is a real thing local models do — so it needs real concurrent transactions.
"""

import asyncio
import uuid

from sqlalchemy import func, select

from shop.models import MenuItem, Order, User, VisitMenuItem
from shop.seed import seed_catalog
from shop.service import add_to_cart, enter, place_order


async def test_concurrent_place_order_charges_once(session_factory):
    async with session_factory() as setup:
        await seed_catalog(setup)
        entered = await enter(setup, "Allan")
        visit_id = uuid.UUID(entered.data["visit_id"])
        user_id = uuid.UUID(entered.data["user_id"])

        await setup.execute(
            VisitMenuItem.__table__.delete().where(VisitMenuItem.visit_id == visit_id)
        )
        latte = await setup.scalar(select(MenuItem).where(MenuItem.name == "Latte"))
        setup.add(VisitMenuItem(visit_id=visit_id, menu_item_id=latte.id))
        await setup.commit()

        await add_to_cart(setup, visit_id, "Latte", 1, "small")  # 400

    async with session_factory() as a, session_factory() as b:
        results = await asyncio.gather(
            place_order(a, visit_id, confirmed_total_cents=400),
            place_order(b, visit_id, confirmed_total_cents=400),
            return_exceptions=True,
        )

    for result in results:
        assert not isinstance(result, Exception), result

    # Two legitimate shapes, depending on how far the loser got before the
    # winner committed: it either conflicted on the idempotency key
    # (duplicate=True) or arrived after the cart was emptied (empty_cart).
    # Anything else means it charged.
    charged = [r for r in results if r.ok and not r.data.get("duplicate")]
    assert len(charged) == 1, [r.to_dict() for r in results]

    loser = next(r for r in results if r is not charged[0])
    assert loser.data.get("duplicate") is True or loser.error == "empty_cart", loser.to_dict()

    async with session_factory() as check:
        orders = await check.scalar(
            select(func.count()).select_from(Order).where(Order.visit_id == visit_id)
        )
        user = await check.get(User, user_id)

    assert orders == 1, "the same cart was charged twice"
    assert user.wallet_cents == 1600, "wallet was debited more than once"

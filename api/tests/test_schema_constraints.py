"""The database, not the service layer, is what makes bad lines impossible.

Spec §8: `size IS NOT NULL` exactly when the item is `sized`. That rule spans two
tables, so it is enforced with a composite foreign key onto
`menu_items(id, sized)` plus a local CHECK. These tests are the proof — if
someone simplifies the constraint away, they fail.
"""

import pytest
from sqlalchemy import text
from sqlalchemy.exc import DBAPIError, IntegrityError

from shop.models import Cart, CartLine, MenuItem, User, Visit


async def _visit_with_cart(session) -> Cart:
    user = User(name="Allan", name_key="allan")
    session.add(user)
    await session.flush()
    visit = Visit(user_id=user.id, day=1)
    session.add(visit)
    await session.flush()
    cart = Cart(visit_id=visit.id)
    session.add(cart)
    await session.flush()
    return cart


async def _add_item(session, *, name: str, category: str, price_cents: int) -> MenuItem:
    item = MenuItem(
        name=name,
        category=category,
        price_cents=price_cents,
        sized=(category == "drink"),
    )
    session.add(item)
    await session.flush()
    return item


async def test_sized_food_line_is_rejected(session):
    """ "A large cookie" must be impossible to store, not merely discouraged."""
    cart = await _visit_with_cart(session)
    cookie = await _add_item(session, name="Cookie", category="food", price_cents=200)

    session.add(
        CartLine(cart_id=cart.id, menu_item_id=cookie.id, quantity=1, size="large", sized=False)
    )
    with pytest.raises((IntegrityError, DBAPIError)):
        await session.flush()


async def test_sizeless_drink_line_is_rejected(session):
    cart = await _visit_with_cart(session)
    latte = await _add_item(session, name="Latte", category="drink", price_cents=400)

    session.add(CartLine(cart_id=cart.id, menu_item_id=latte.id, quantity=1, size=None, sized=True))
    with pytest.raises((IntegrityError, DBAPIError)):
        await session.flush()


async def test_line_cannot_claim_a_sized_flag_the_item_does_not_have(session):
    """The composite FK is what stops the line lying about the catalog."""
    cart = await _visit_with_cart(session)
    cookie = await _add_item(session, name="Cookie", category="food", price_cents=200)

    # Internally consistent (sized=True with a size) but disagrees with the item.
    session.add(
        CartLine(cart_id=cart.id, menu_item_id=cookie.id, quantity=1, size="large", sized=True)
    )
    with pytest.raises((IntegrityError, DBAPIError)):
        await session.flush()


async def test_valid_lines_are_accepted(session):
    cart = await _visit_with_cart(session)
    latte = await _add_item(session, name="Latte", category="drink", price_cents=400)
    cookie = await _add_item(session, name="Cookie", category="food", price_cents=200)

    session.add_all(
        [
            CartLine(cart_id=cart.id, menu_item_id=latte.id, quantity=1, size="large", sized=True),
            CartLine(cart_id=cart.id, menu_item_id=cookie.id, quantity=2, size=None, sized=False),
        ]
    )
    await session.flush()

    count = await session.scalar(
        text("SELECT count(*) FROM cart_lines WHERE cart_id = :cart_id"), {"cart_id": cart.id}
    )
    assert count == 2


async def test_menu_item_cannot_be_a_sized_food(session):
    """The catalog's own rule: drinks are sized, food is not (spec §3.4)."""
    session.add(MenuItem(name="Odd Cookie", category="food", price_cents=200, sized=True))
    with pytest.raises((IntegrityError, DBAPIError)):
        await session.flush()

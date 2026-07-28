"""Proves the rollback fixture actually isolates.

Both tests insert a user with the same `name_key`, which is unique. If the
fixture leaked, the second would fail with a unique violation — which makes this
pair a real assertion rather than a tautology.
"""

from sqlalchemy import select

from shop.models import User


async def test_insert_user_first(session):
    session.add(User(name="Allan", name_key="allan"))
    await session.commit()

    found = await session.scalar(select(User).where(User.name_key == "allan"))
    assert found is not None
    assert found.wallet_cents == 2000


async def test_insert_same_user_again(session):
    session.add(User(name="Allan", name_key="allan"))
    await session.commit()

    found = await session.scalar(select(User).where(User.name_key == "allan"))
    assert found is not None

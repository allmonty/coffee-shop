"""Proves the rollback fixture actually isolates.

Both tests insert the same primary key. If the fixture leaked, the second would
fail with a unique violation — which makes this pair a real assertion rather
than a tautology.
"""

from sqlalchemy import text


async def test_insert_scratch_row_first(session):
    await session.execute(text("INSERT INTO scratch (id) VALUES (1)"))
    await session.commit()

    count = await session.scalar(text("SELECT count(*) FROM scratch WHERE id = 1"))
    assert count == 1


async def test_insert_same_row_again(session):
    await session.execute(text("INSERT INTO scratch (id) VALUES (1)"))
    await session.commit()

    count = await session.scalar(text("SELECT count(*) FROM scratch WHERE id = 1"))
    assert count == 1

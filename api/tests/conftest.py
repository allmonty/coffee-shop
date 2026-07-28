"""Test fixtures.

Two things matter here:

1. Tests run against a real Postgres (`coffee_shop_test`), not SQLite. The schema
   uses check constraints, JSONB and `ON CONFLICT`, so a different engine would
   test something other than what ships.
2. Every test runs inside a transaction that is rolled back afterwards. Tests can
   call `session.commit()` freely — `join_transaction_mode="create_savepoint"`
   turns those into savepoints inside the outer transaction, so committed data is
   visible within the test and gone after it.

Engines are created per test rather than shared session-wide. asyncpg binds a
connection to the event loop that opened it, and pytest-asyncio gives each test
its own loop, so a shared engine hands out connections from the wrong loop. One
engine per test costs a few milliseconds and removes the whole failure mode.
"""

import asyncio

import pytest
import pytest_asyncio
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine

from settings import settings

TEST_DB_NAME = "coffee_shop_test"


def test_database_url() -> str:
    return f"{settings.database_url.rpartition('/')[0]}/{TEST_DB_NAME}"


async def _create_database_and_schema() -> None:
    # CREATE DATABASE cannot run inside a transaction, hence AUTOCOMMIT.
    admin_url = f"{settings.database_url.rpartition('/')[0]}/postgres"
    admin_engine = create_async_engine(admin_url, isolation_level="AUTOCOMMIT")
    try:
        async with admin_engine.connect() as conn:
            exists = await conn.scalar(
                text("SELECT 1 FROM pg_database WHERE datname = :name"),
                {"name": TEST_DB_NAME},
            )
            if not exists:
                await conn.execute(text(f'CREATE DATABASE "{TEST_DB_NAME}"'))
    finally:
        await admin_engine.dispose()

    engine = create_async_engine(test_database_url())
    try:
        async with engine.begin() as conn:
            # Scratch table so fixture isolation is testable before any real model
            # exists. Harmless once the schema lands.
            await conn.execute(text("CREATE TABLE IF NOT EXISTS scratch (id int primary key)"))
    finally:
        await engine.dispose()


@pytest.fixture(scope="session", autouse=True)
def prepared_database() -> None:
    """Sync fixture so it owns its own loop and leaks nothing into the tests."""
    asyncio.run(_create_database_and_schema())


@pytest_asyncio.fixture
async def session(prepared_database) -> AsyncSession:
    """A session whose writes are discarded when the test ends."""
    engine = create_async_engine(test_database_url())
    connection = await engine.connect()
    transaction = await connection.begin()
    db_session = AsyncSession(
        bind=connection,
        expire_on_commit=False,
        join_transaction_mode="create_savepoint",
    )
    try:
        yield db_session
    finally:
        await db_session.close()
        await transaction.rollback()
        await connection.close()
        await engine.dispose()

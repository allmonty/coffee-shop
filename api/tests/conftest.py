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
from pathlib import Path

import pytest
import pytest_asyncio
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from alembic import command
from alembic.config import Config
from settings import settings

TEST_DB_NAME = "coffee_shop_test"
API_ROOT = Path(__file__).resolve().parent.parent


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


def _migrate(url: str) -> None:
    """Run the real migrations, not `metadata.create_all`.

    Tests then exercise the schema that actually ships, so a migration drifting
    from the models fails here rather than in production.
    """
    alembic_cfg = Config(str(API_ROOT / "alembic.ini"))
    alembic_cfg.set_main_option("script_location", str(API_ROOT / "alembic"))
    alembic_cfg.attributes["db_url"] = url
    command.upgrade(alembic_cfg, "head")


@pytest.fixture(scope="session", autouse=True)
def prepared_database() -> None:
    """Sync fixture so it owns its own loop and leaks nothing into the tests."""
    asyncio.run(_create_database_and_schema())
    _migrate(test_database_url())


ALL_TABLES = (
    "users, menu_items, size_modifiers, visits, visit_menu_items, carts, "
    "cart_lines, orders, order_lines, messages, customer_preferences"
)


@pytest_asyncio.fixture
async def session_factory(prepared_database):
    """Real committing sessions, for tests that need genuine concurrency.

    The rollback `session` fixture cannot express "two transactions racing",
    because both would be inside the same outer transaction. Tests using this
    commit for real, so the fixture truncates afterwards.
    """
    engine = create_async_engine(test_database_url())
    factory = async_sessionmaker(engine, expire_on_commit=False)
    try:
        yield factory
    finally:
        async with engine.begin() as conn:
            await conn.execute(text(f"TRUNCATE {ALL_TABLES} RESTART IDENTITY CASCADE"))
        await engine.dispose()


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

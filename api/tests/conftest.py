"""Test fixtures.

**Tests never touch the development database.** Everything here runs against a
separate `coffee_shop_test`, and `assert_test_database()` refuses to execute a
destructive statement against anything whose name does not end in `_test` or
that matches the configured `DATABASE_URL`. Truncating a real database because
a URL was edited carelessly is the kind of mistake worth making structurally
impossible.

**Tests leave nothing behind**, by two different mechanisms:

1. Most use the `session` fixture — a transaction rolled back afterwards. Tests
   can call `session.commit()` freely: `join_transaction_mode="create_savepoint"`
   turns those into savepoints inside the outer transaction, so committed data
   is visible within the test and gone after it. Nothing is ever written.
2. Concurrency tests need real committing transactions (two racing transactions
   cannot both live inside one outer transaction), so they use `session_factory`
   and are truncated afterwards.

The run is also truncated at **start** and **end**. Start matters because a run
killed mid-test — Ctrl-C, a hung lock, `pkill` — never reaches fixture teardown,
and its committed rows would otherwise silently become the next run's starting
state.

Tests run against real Postgres rather than SQLite because the schema uses check
constraints, composite foreign keys, JSONB and `ON CONFLICT`; another engine
would test something other than what ships.

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


def resolve_test_db_url() -> str:
    return f"{settings.database_url.rpartition('/')[0]}/{TEST_DB_NAME}"


class NotATestDatabase(RuntimeError):
    """Raised before anything destructive touches a database it should not."""


def assert_test_database(url: str) -> str:
    """Gate every destructive statement in the suite.

    Two independent checks, because either one alone can be defeated by a
    plausible edit: the name must end in `_test`, AND it must not be whatever
    `DATABASE_URL` points at. A developer database called `something_test` would
    still be protected by the second.
    """
    name = url.rpartition("/")[2].partition("?")[0]
    if not name.endswith("_test"):
        raise NotATestDatabase(
            f"refusing to run destructive SQL against {name!r}: test databases must be named *_test"
        )
    if url == settings.database_url:
        raise NotATestDatabase(
            f"refusing to run destructive SQL against {name!r}: it is the configured DATABASE_URL"
        )
    return url


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
    assert_test_database(url)
    alembic_cfg = Config(str(API_ROOT / "alembic.ini"))
    alembic_cfg.set_main_option("script_location", str(API_ROOT / "alembic"))
    alembic_cfg.attributes["db_url"] = url
    command.upgrade(alembic_cfg, "head")


ALL_TABLES = (
    "users, menu_items, size_modifiers, drink_modifiers, visits, visit_menu_items, carts, "
    "cart_lines, orders, order_lines, messages, customer_preferences"
)


async def truncate_all(url: str | None = None) -> None:
    url = assert_test_database(url or resolve_test_db_url())
    engine = create_async_engine(url)
    try:
        async with engine.begin() as conn:
            await conn.execute(text(f"TRUNCATE {ALL_TABLES} RESTART IDENTITY CASCADE"))
    finally:
        await engine.dispose()


@pytest.fixture(scope="session", autouse=True)
def prepared_database():
    """Sync fixture so it owns its own loop and leaks nothing into the tests."""
    asyncio.run(_create_database_and_schema())
    _migrate(resolve_test_db_url())

    # Start clean. A run killed mid-test never reaches fixture teardown, so
    # without this its committed rows become the next run's starting state —
    # which shows up much later as an inexplicably failing unrelated test.
    asyncio.run(truncate_all())
    yield
    # End clean, so nothing survives the suite even if a teardown failed.
    asyncio.run(truncate_all())


@pytest_asyncio.fixture
async def session_factory(prepared_database):
    """Real committing sessions, for tests that need genuine concurrency.

    The rollback `session` fixture cannot express "two transactions racing",
    because both would be inside the same outer transaction. Tests using this
    commit for real, so the fixture truncates afterwards.
    """
    engine = create_async_engine(assert_test_database(resolve_test_db_url()))
    factory = async_sessionmaker(engine, expire_on_commit=False)
    try:
        yield factory
    finally:
        await engine.dispose()
        # After disposing, so no pooled connection is holding a lock that would
        # make TRUNCATE wait on itself.
        await truncate_all()


@pytest_asyncio.fixture
async def session(prepared_database) -> AsyncSession:
    """A session whose writes are discarded when the test ends."""
    engine = create_async_engine(resolve_test_db_url())
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

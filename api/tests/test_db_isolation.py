"""The suite must never touch the development database, and must leave nothing
behind in its own.

These assert the safety rails themselves. A guard with no test is a comment.
"""

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

from settings import settings
from tests.conftest import (
    ALL_TABLES,
    NotATestDatabase,
    assert_test_database,
    resolve_test_db_url,
)


def _insert_user(name: str, key: str):
    """Raw SQL on purpose — these tests are about the fixtures, not the ORM.

    current_day and wallet_cents are ORM-side defaults rather than server
    defaults, so a raw INSERT must supply them.
    """
    return text(
        "INSERT INTO users (id, name, name_key, current_day, wallet_cents) "
        f"VALUES (gen_random_uuid(), '{name}', '{key}', 1, 2000)"
    )


def test_the_suite_targets_a_different_database_than_the_app():
    assert resolve_test_db_url() != settings.database_url
    assert resolve_test_db_url().endswith("_test")


def test_guard_rejects_the_configured_application_database(monkeypatch):
    """The second check, isolated.

    Needs a URL that passes the `*_test` name rule but still points at the
    configured database — i.e. someone running the app itself against a database
    called `..._test`. Without this branch, the name rule alone would let the
    suite truncate it.
    """
    pretend = "postgresql+asyncpg://coffee:coffee@localhost:5432/someones_test"
    monkeypatch.setattr(settings, "database_url", pretend)

    with pytest.raises(NotATestDatabase, match="DATABASE_URL"):
        assert_test_database(pretend)


@pytest.mark.parametrize(
    "url",
    [
        "postgresql+asyncpg://coffee:coffee@localhost:5432/coffee_shop",
        "postgresql+asyncpg://coffee:coffee@prod.example.com:5432/production",
        "postgresql+asyncpg://coffee:coffee@localhost:5432/testing",  # not *_test
    ],
)
def test_guard_rejects_anything_not_named_test(url):
    with pytest.raises(NotATestDatabase):
        assert_test_database(url)


def test_guard_accepts_a_real_test_database():
    assert assert_test_database(resolve_test_db_url()) == resolve_test_db_url()


async def test_rollback_fixture_writes_nothing(session):
    """`session` writes must be invisible to any other connection.

    Checked from a second, independent connection — asserting through the same
    session would only prove the session can see itself.

    That connection is opened by hand rather than via `session_factory`, and the
    reason is worth knowing: `session_factory` TRUNCATEs on teardown, TRUNCATE
    needs an ACCESS EXCLUSIVE lock, and `session` still holds an open
    transaction on `users` at that point. Requesting both fixtures in one test
    deadlocks until the timeout fires.
    """
    await session.execute(_insert_user("Ghost", "ghost"))
    await session.commit()  # a savepoint, not a real commit

    engine = create_async_engine(resolve_test_db_url())
    try:
        async with engine.connect() as other:
            seen = await other.scalar(text("SELECT count(*) FROM users WHERE name_key = 'ghost'"))
    finally:
        await engine.dispose()

    assert seen == 0, "the rollback fixture leaked a committed row"


async def test_committing_fixture_is_truncated_between_tests(session_factory):
    """Paired with the next test: both insert the same key. If the truncate in
    the fixture teardown stopped working, the second would fail."""
    async with session_factory() as db:
        await db.execute(_insert_user("Twin", "twin"))
        await db.commit()


async def test_committing_fixture_was_truncated(session_factory):
    async with session_factory() as db:
        before = await db.scalar(text("SELECT count(*) FROM users"))
        assert before == 0, "previous test's committed rows survived"

        await db.execute(_insert_user("Twin", "twin"))
        await db.commit()


async def test_every_table_is_covered_by_the_truncate_list(session):
    """A new model with no entry in ALL_TABLES would silently start leaking."""
    rows = await session.execute(
        text("""
        SELECT tablename FROM pg_tables
        WHERE schemaname = 'public' AND tablename <> 'alembic_version'
        """)
    )
    actual = {row[0] for row in rows}
    covered = {name.strip() for name in ALL_TABLES.split(",")}

    assert actual == covered, (
        f"tables missing from ALL_TABLES: {sorted(actual - covered)}; "
        f"listed but nonexistent: {sorted(covered - actual)}"
    )

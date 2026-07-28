"""Conversation persistence, keyed on `visit_id` (spec §6.5).

`thread_id = visit_id` is the whole design: resume a visit and the barista picks
the conversation up mid-sentence, because §7.1 resumes the *same* visit rather
than opening a second one.

The checkpointer lives in its own `agent_checkpoints` schema. That separation is
deliberate — the checkpointer's table format is LangGraph's business and will
change under you, while `messages` (the app's own transcript, spec §8) is yours.
Keeping them in one schema invites treating LangGraph's tables as an API.

It also needs psycopg, not asyncpg: langgraph-checkpoint-postgres is built on
psycopg3. One database, two drivers.
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from psycopg import AsyncConnection
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool

from settings import settings

SCHEMA = "agent_checkpoints"


def checkpointer_dsn() -> str:
    """SQLAlchemy's URL, minus the driver, plus the checkpoint schema."""
    dsn = settings.database_url.replace("+asyncpg", "")
    separator = "&" if "?" in dsn else "?"
    return f"{dsn}{separator}options=-c%20search_path%3D{SCHEMA}"


async def _ensure_schema() -> None:
    base = settings.database_url.replace("+asyncpg", "")
    async with await AsyncConnection.connect(base, autocommit=True) as conn:
        await conn.execute(f"CREATE SCHEMA IF NOT EXISTS {SCHEMA}")


@asynccontextmanager
async def open_checkpointer():
    """Yields a saver backed by a small pool. Call `.setup()` is handled here."""
    await _ensure_schema()
    async with AsyncConnectionPool(
        conninfo=checkpointer_dsn(),
        max_size=5,
        open=False,
        kwargs={"autocommit": True, "row_factory": dict_row},
    ) as pool:
        await pool.open()
        saver = AsyncPostgresSaver(pool)
        await saver.setup()
        yield saver

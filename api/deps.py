"""The only dependency injection in the project (spec §5.3).

Deliberately tiny. If this file grows a framework, something has gone wrong.
"""

from collections.abc import AsyncIterator

from sqlalchemy.ext.asyncio import AsyncSession

from db import SessionLocal


async def get_session() -> AsyncIterator[AsyncSession]:
    """One session per request, closed when the request ends."""
    async with SessionLocal() as session:
        yield session

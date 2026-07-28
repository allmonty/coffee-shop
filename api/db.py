"""Engine and session factory.

`shop/` is the only layer that touches these — `agent/` reaches the database
exclusively through `shop.service` functions (spec §5.1).
"""

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from settings import settings

engine = create_async_engine(settings.database_url, pool_pre_ping=True)

SessionLocal = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

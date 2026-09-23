"""Lazy PostgreSQL engine and short, explicit transaction boundaries."""

from contextlib import asynccontextmanager
from functools import lru_cache

from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from ..settings import get_settings


@lru_cache
def get_engine():
    settings = get_settings()
    url = settings.sql_database_url
    if not url.startswith("postgresql+asyncpg://"):
        raise ValueError("SQL_DATABASE_URL must use postgresql+asyncpg://")
    return create_async_engine(
        url,
        pool_size=settings.sql_pool_size,
        max_overflow=settings.sql_max_overflow,
        pool_timeout=settings.sql_pool_timeout,
        pool_pre_ping=True,
    )


@asynccontextmanager
async def get_db():
    """Commit one short operation; rollback failures; returned records stay usable."""
    factory = async_sessionmaker(get_engine(), expire_on_commit=False, autoflush=False)
    async with factory() as session:
        async with session.begin():
            yield session


async def dispose_engine():
    if get_engine.cache_info().currsize:
        await get_engine().dispose()
        get_engine.cache_clear()

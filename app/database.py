import os
import structlog
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from sqlalchemy.orm import declarative_base

logger = structlog.get_logger()

# Default to localhost if not provided
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql+asyncpg://postgres:postgres@localhost:5432/postgres")

# Async SQLAlchemy requires the asyncpg driver, so swap standard postgresql:// if needed
if DATABASE_URL.startswith("postgresql://"):
    DATABASE_URL = DATABASE_URL.replace("postgresql://", "postgresql+asyncpg://", 1)

engine = create_async_engine(
    DATABASE_URL,
    echo=False,
    future=True,
    pool_pre_ping=True
)

SessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autoflush=False,
    autocommit=False
)

Base = declarative_base()

DB_AVAILABLE = True

async def init_db():
    global DB_AVAILABLE
    try:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        DB_AVAILABLE = True
        logger.info("database_initialized", message="Database tables created successfully")
    except Exception as e:
        DB_AVAILABLE = False
        logger.error("database_connection_failed", error=str(e), message="Failed to reach DB, running in degraded mode", exc_info=False)

async def get_db():
    session = SessionLocal()
    try:
        yield session
        await session.commit()
    except Exception as e:
        await session.rollback()
        logger.error("db_transaction_rolled_back", error=str(e), exc_info=False)
        raise
    finally:
        await session.close()

__all__ = ["engine", "SessionLocal", "Base", "get_db", "init_db", "DB_AVAILABLE"]

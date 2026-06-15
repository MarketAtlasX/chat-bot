import json
from datetime import datetime
from typing import Optional, Any
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy import String, Float, Text, DateTime, JSON, select
from ..config import settings


class Base(DeclarativeBase):
    pass


class Event(Base):
    __tablename__ = "events"
    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    title: Mapped[str] = mapped_column(String(500))
    description: Mapped[str] = mapped_column(Text, nullable=True)
    event_type: Mapped[str] = mapped_column(String(100))
    severity: Mapped[float] = mapped_column(Float, default=0.0)
    timestamp: Mapped[datetime] = mapped_column(DateTime)
    location: Mapped[Optional[str]] = mapped_column(String(200), nullable=True)
    sectors: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    extra_meta: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)


class Article(Base):
    __tablename__ = "articles"
    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    title: Mapped[str] = mapped_column(String(500))
    content: Mapped[str] = mapped_column(Text)
    source: Mapped[str] = mapped_column(String(200))
    url: Mapped[Optional[str]] = mapped_column(String(1000), nullable=True)
    published_at: Mapped[datetime] = mapped_column(DateTime)
    topics: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    sentiment: Mapped[Optional[float]] = mapped_column(Float, nullable=True)


class Signal(Base):
    __tablename__ = "signals"
    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    signal_type: Mapped[str] = mapped_column(String(100))
    asset: Mapped[str] = mapped_column(String(50))
    direction: Mapped[str] = mapped_column(String(10))
    strength: Mapped[float] = mapped_column(Float)
    confidence: Mapped[float] = mapped_column(Float)
    reasoning: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime)


class Price(Base):
    __tablename__ = "prices"
    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    symbol: Mapped[str] = mapped_column(String(20))
    price: Mapped[float] = mapped_column(Float)
    volume: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime)


class Conversation(Base):
    __tablename__ = "conversations"
    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    query: Mapped[str] = mapped_column(Text)
    response: Mapped[str] = mapped_column(Text)
    intent: Mapped[str] = mapped_column(String(50))
    agents_used: Mapped[str] = mapped_column(String(500))
    confidence: Mapped[float] = mapped_column(Float)
    created_at: Mapped[datetime] = mapped_column(DateTime)


engine = None
session_factory = None


async def init_db():
    global engine, session_factory
    engine = create_async_engine(settings.postgres_dsn, echo=False)
    session_factory = async_sessionmaker(engine, expire_on_commit=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def get_session() -> AsyncSession:
    if session_factory is None:
        await init_db()
    return session_factory()


async def store_conversation(conversation_id: str, query: str, response: str, intent: str, agents: list[str], confidence: float):
    async with await get_session() as session:
        conv = Conversation(
            id=conversation_id,
            query=query,
            response=response,
            intent=intent,
            agents_used=",".join(agents),
            confidence=confidence,
            created_at=datetime.utcnow(),
        )
        session.add(conv)
        await session.commit()
        return conv


async def get_conversation_history(limit: int = 20) -> list[Conversation]:
    async with await get_session() as session:
        result = await session.execute(
            select(Conversation).order_by(Conversation.created_at.desc()).limit(limit)
        )
        return result.scalars().all()

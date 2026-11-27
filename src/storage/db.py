"""
Database engine and sessions. SQLite by default (data/research.db); any SQLAlchemy URL
can be supplied through RESEARCH_DB_URL.
"""

import os
from contextlib import contextmanager
from typing import Iterator, Optional

from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from .models import Base

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_DB_PATH = os.path.join(ROOT, "data", "research.db")

_engines: dict = {}


def database_url() -> str:
    return os.getenv("RESEARCH_DB_URL") or f"sqlite:///{DEFAULT_DB_PATH}"


def get_engine(url: Optional[str] = None) -> Engine:
    """Create (once per URL) an engine and ensure all tables exist."""
    url = url or database_url()
    if url in _engines:
        return _engines[url]
    if url.startswith("sqlite:///") and not url.endswith(":memory:"):
        os.makedirs(os.path.dirname(url.replace("sqlite:///", "", 1)) or ".", exist_ok=True)
    engine = create_engine(url, future=True)
    if url.startswith("sqlite"):
        @event.listens_for(engine, "connect")
        def _sqlite_pragmas(dbapi_conn, _):
            cur = dbapi_conn.cursor()
            cur.execute("PRAGMA foreign_keys=ON")
            cur.execute("PRAGMA journal_mode=WAL")
            cur.close()
    Base.metadata.create_all(engine)
    _engines[url] = engine
    return engine


@contextmanager
def session_scope(url: Optional[str] = None) -> Iterator[Session]:
    """Transactional session: commits on success, rolls back on error."""
    session = sessionmaker(bind=get_engine(url), expire_on_commit=False, future=True)()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()

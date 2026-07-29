"""Shared database session, used by the training/import scripts."""
from sqlalchemy import create_engine
from sqlalchemy.orm import scoped_session, sessionmaker

db_session = scoped_session(sessionmaker())


def init_db(database_url: str):
    engine = create_engine(database_url, future=True)
    db_session.configure(bind=engine)
    return engine

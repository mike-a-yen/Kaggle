import os

from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

engine = create_engine(os.environ['DATABASE_URL'])
_Session = sessionmaker(bind=engine)

Base = declarative_base()


def make_session():
    Base.metadata.create_all(engine)
    return _Session()

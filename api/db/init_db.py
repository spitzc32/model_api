# PROGRAM:      init_db.py --> Program for calling a database
# PROGRAMMER:   Jayra Gaile Ortiz
# VERSION 1:    08-29-22 Initial API setup and i2b2 converted sets
# PURPOSE:      Calls a database
# DATA:
# *GENERATOR:   Uses Generator to return a lazy iterator so that contents are not stored in the memory
# ALGORITHM:    Returns a database


from typing import Generator
from api.db.session import SessionLocal


def get_db() -> Generator:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
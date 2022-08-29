"""
Models To be separated upon change of requirements
"""

from enum import unique
from sqlalchemy import Integer, String, Column, Float, TIMESTAMP
from sqlalchemy.orm import relationship
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.sql.schema import ForeignKey, UniqueConstraint

base = declarative_base()

class User(base):
    __tablename__ = 'user'
    id = Column(Integer, primary_key=True, index=True)
    first_name = Column(String(256), nullable=False)
    last_name = Column(String(256), nullable=False)
    email = Column(String(256),unique=True, nullable=False)
    password = Column(String(256), nullable=False)
    role_id = Column(ForeignKey("role.id"), nullable=False)
    role = relationship("Role", backref="role_id", lazy=True)


class Role(base):
   __tablename__ = 'role'
   id = Column(Integer, primary_key=True, index=True)
   role_name = Column(String(100), nullable=True)


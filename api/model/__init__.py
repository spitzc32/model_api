# PROGRAM:      __init__.py --> Separates models upon change of requirements
# PROGRAMMER:   Jayra Gaile Ortiz
# VERSION 1:    08-29-22 Initial API setup and i2b2 converted sets
# REVISION 1.1: 11-30-22 Added Feature for CSV and model trainer output
# PURPOSE:      Creates a base class containing different components for Users and Roles
# ALGORITHM:    Creates a base class using declarative_base() then assigns data to respective columns


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

   


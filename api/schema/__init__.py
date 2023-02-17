# PROGRAM:      __init__.py --> Program for defining schemas
# PROGRAMMER:   Jayra Gaile Ortiz
# VERSION 1:    08-29-22 Initial API setup and i2b2 converted sets
# REVISION 1.1: 11-30-22 Added Feature for CSV and model trainer output
# PURPOSE:      Define schema's data types
# ALGORTHIM:    Define the components for each Base Model         


from datetime import datetime
from typing import List, Optional, Sequence

from pydantic import BaseModel, Field, HttpUrl
from pydantic.types import UUID4, condecimal, constr


class UserSchema(BaseModel):
    user_id: Optional[int]
    first_name: Optional[str]
    last_name: Optional[str]
    email: Optional[str]
    role: Optional[str]

    class Config:
        orm_mode = True

class RoleSchema(BaseModel):
    id: Optional[str]
    role_name: Optional[str]

class Token(BaseModel):
    accessToken: str
    refreshToken: str
    tokenType: str


class TokenData(BaseModel):
    role: Optional[str] = None

class WordSchema(BaseModel):
    word_text: str




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




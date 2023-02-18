"""
PROGRAM:      root > main.py
PROGRAMMER:   Jayra Gaile Ortiz
VERSION 1:    Sept. 20, 2022 
VERSION 2:    Jan. 09, 2023
PURPOSE:      Main Runner of the API for login, signup and model runner
ALGORITHM:    For Login, Sign Up and model algorithm response for REST API

"""

import uvicorn
from fastapi import  APIRouter, Query, Depends

from typing import Optional
from sqlalchemy.orm import Session
import sqlalchemy
from flair.data import Sentence
import pandas as pd

from api.db.settings import app
from api.db.init_db import get_db
from api.model import User
from api.schema import UserSchema, RoleSchema,  WordSchema
from api.modules.encryption import *
from environment import tagger, roles

from starlette import status
from starlette.responses import JSONResponse
api_router = APIRouter()



@api_router.post("/api/user/create", status_code=200)
def create_user(
    *, 
    first_name: str = Query(None, min_length=3, max_length=100),
    last_name: str = Query(None, min_length=3, max_length=50),
    email: str = Query(None, min_length=3, max_length=50),
    password: str = Query(None, min_length=3, max_length=30),
    role_id: int,
    user_license: str,
    db: Session = Depends(get_db)
): 
    """
    An api for user creation within the database.
    :param first_name: the first name of the user
    :param last_name: last name of the user.
    :param email: email of the user
    :param password: password of the user
    :param role_id: role number of user
    :param db: database connection initialized in PSQL

    :returns: json response that contains entered 
        credentials except password.
    """
    try:
        user = User(
            first_name=first_name,
            last_name=last_name,
            email=email,
            password=get_password_hash(password),
            role_id=role_id,
            user_license=user_license
        )
        db.add(user)
        db.commit()

        return JSONResponse(
                status_code=status.HTTP_201_CREATED,
                content={
                    "code": 200,
                    "data": {
                        "user_id": user.id,
                        "first_name": user.first_name,
                        "last_name": user.last_name,
                        "email": user.email,
                        "role": roles[role_id-1],
                        "license": user.user_license,
                        "is_verified": user.is_verified,
                    }
                },
            )
    except Exception as e:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "code": 400,
                "error": {
                    "message": f"User already exists. please log in, {e}",
                }
            },
        )

@api_router.get("/api/user/login", status_code=200)
def login(
    *, 
    email: str = Query(None, min_length=3, max_length=50),
    password: str = Query(None, min_length=3, max_length=30),
    db: Session = Depends(get_db)
):
    """
    An api for user retrieval of informtion using credentials
        within the database.
    
    :param email: email of the user
    :param password: password of the user
    :param role_id: role number of user
    :param db: database connection initialized in PSQL

    :returns: json response that contains entered 
        credentials except password.
    """
    try:
        user = db.query(User).filter(User.email == email).one()
        
        if verify_password(password, user.password):
            if user.is_verified:
                return JSONResponse(
                    status_code=status.HTTP_201_CREATED,
                    content={
                        "code": 200,
                        "data": {
                            "user_id": user.id,
                            "first_name": user.first_name,
                            "last_name": user.last_name,
                            "email": user.email,
                            "role": roles[user.role_id-1],
                            "license": user.user_license,
                            "is_verified": user.is_verified,
                        }
                    },
                )
            else:
                return JSONResponse(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    content={
                        "code": 400,
                        "error": {
                            "message": "User not yet verified. please contact mail.arniefraga@gmail.com to verify",
                        }
                    },
                )
        else:
            return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "code": 400,
                "error": {
                    "message": "Wrong password for this user. Try again",
                }
            },
        )         
    
    except sqlalchemy.orm.exc.NoResultFound:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "code": 400,
                "error": {
                    "message": "No mailing address for this user",
                }
            },
        )

@api_router.get("/api/model", response_model=WordSchema)
def model(
    *, 
    word: str,
    file_name: str
    ):
    """
    An api for serving the model for the PHI classification.
    :param word: list of word tokens in a paragraph.
    :param file_name: name of the wile.

    :returns: json response that contains labeled 
        tags their respective classification.
    """
    txt = Sentence(word)
    tagger.predict(txt)
    labels, tags = [], []

    for entity in txt.get_spans('ner'):
        labels.append(entity.text)
        tags.append(entity.get_label("ner").value)
    
    pd.DataFrame({'Entity': labels,
        'Tags': tags}
    ).to_csv(f"./data/CSVs/{file_name}.csv")

    return JSONResponse(
            status_code=status.HTTP_201_CREATED,
            content={
                "code": 200,
                "data": {
                    "labels": labels,
                    "tags": tags
                }
            },
        )



app.include_router(api_router)

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True, debug=True)

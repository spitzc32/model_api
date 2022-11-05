from jose import jwt
from datetime import datetime, timedelta
from api.schema import TokenData

from starlette import status
from starlette.responses import JSONResponse
from starlette.status import *


import configparser

config = configparser.ConfigParser()
config.read('.env')

SECRET_KEY = "3729ewodhwoefw09ef80e9wdfhf3049890aso!082sqwodwy09"
ALGORITHM = "SHA-256"


def create_access_token(data: dict, expires_delta: timedelta = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

    return encoded_jwt


# This part will create the refresh token
def create_refresh_token(data: dict, expires_delta: timedelta = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(days=1)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

    return encoded_jwt


async def check_access_token(token: str):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=['HS256'])
        email: str = payload.get("sub")
        return TokenData(email=email)
    except:
        error = {
            "code": 401,
            "error": {
                "message": "Invalid Token",
            },
        }
        return JSONResponse(
            status_code=status.HTTP_401_UNAUTHORIZED,
            content=error,
        )

# This functions checks if the refresh token is valid or not
def check_refresh_token(refresh):
    try:
        token = jwt.decode(refresh, SECRET_KEY, algorithms=['HS256'])
        email = token['sub']
        return email
    except:
        error = {
            "code": 401,
            "error": {
                "message": "Invalid Refresh Token",
            }
        }
        return JSONResponse(
            status_code=status.HTTP_401_UNAUTHORIZED,
            content=error,
        )



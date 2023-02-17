# PROGRAM:      encryption.py --> Program for encryption of passwords
# PROGRAMMER:   Jayra Gaile Ortiz
# VERSION 1:    08-29-22 Initial API setup and i2b2 converted sets
# PURPOSE:      Encrypts protectected texts like passwords
# ALGORITHM:    Uses CryptContext to configure default algorithm, verifies the password,
#               re-hash deprecated passwords, then load the configurated file


from passlib.context import CryptContext
import uuid


pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def verify_password(plain_password, password):
    return pwd_context.verify(plain_password, password)

def get_password_hash(password):
    return pwd_context.hash(password)

def get_id():
    return uuid.uuid1().int

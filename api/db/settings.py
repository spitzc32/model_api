# PROGRAM:      settings.py --> Program to handle all configurations inside the API
# PROGRAMMER:   Jayra Gaile Ortiz
# VERSION 1:    08-29-22 Initial API setup and i2b2 converted sets
# REVISION 1.1: 11-05-22 Fizes to port web
# PURPOSE:      Handles configurations for the API
# ALGORITHM:    Stores app documentations, handles API configurations


"""
    This file includes all the configuration file for API
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware


# Handles documentation title
app = FastAPI(
    version="version 1.0.0",
    title="Redaction Backend",
    description="List of all endpoints for the Backend API",
    docs_url="/", redoc_url= None
)

# Handles cors
origins = [
    "http://localhost:8000"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
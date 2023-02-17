# PROGRAM:      controllers.py --> Program for requesting and submitting the data
# PROGRAMMER:   Jayra Gaile Ortiz
# VERSION 1:    08-29-22 Initial API setup and i2b2 converted sets
# PURPOSE:      Requests and submits data to the server
# ALGORITHM:    Handles data to be requested and submitted


from fastapi import FastAPI

app = FastAPI(
    version="version 1.0.0",
    title="Redaction Backend",
    description="List of all endpoints for the Backend API",
    docs_url="/", redoc_url=None
)

@app.get("/", tags=["Home"])
def index() -> dict:
    return {
        "code": 200,
        "data": "Redaction Backend Version 1.0.0"
    }

@app.post("/user/create", tags=["User", "Create"])
def create()-> dict:
    return {
        "code":200,
        "data": {
            "email":  "johndoe"
        }
    }

@app.post("/user/login", tags=["User", "Create"])
def login()-> dict:
    return {}

@app.post("/redact", tags=["User", "Create"])
def redact()-> dict:
    return {}



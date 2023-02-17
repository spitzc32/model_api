# PROGRAM:      session.py --> Program to create a session configuration 
# PROGRAMMER:   Jayra Gaile Ortiz
# VERSION 1:    08-29-22 Initial API setup and i2b2 converted sets
# REVISION 1.1: 11-05-22 Added Docker configuration setup and fixes to some APIs in 
#               accordance to docker setup
# PURPOSE:      Used to create a session that can be used throughout the application without 
#               repeating configurational arguments
# ALGORITHM:    Fetches the database, creates an engine from the database URL which is then binded
#               inside the sessionmaker to create a connection        


from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import databases

DATABASE_URI = "postgresql://postgres:117l3vi@db:5432/thesis"
database = databases.Database(DATABASE_URI, min_size=2, max_size=10)

engine = create_engine(DATABASE_URI, pool_pre_ping=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
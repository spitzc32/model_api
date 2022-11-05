from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import databases

DATABASE_URI = "postgresql://postgres:117l3vi@db:5432/thesis"
database = databases.Database(DATABASE_URI, min_size=2, max_size=10)

engine = create_engine(DATABASE_URI, pool_pre_ping=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import databases
import configparser

config = configparser.ConfigParser()
config.read('.env')

DATABASE_URI = config['DB']['DATABASE_URL']
database = databases.Database(DATABASE_URI, min_size=2, max_size=10)

engine = create_engine(DATABASE_URI, pool_pre_ping=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
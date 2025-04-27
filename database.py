# database.py
from sqlalchemy import create_engine

DATABASE_URL = "postgresql://postgres:******s@localhost:5432/GYK2-Northwind"
engine = create_engine(DATABASE_URL)

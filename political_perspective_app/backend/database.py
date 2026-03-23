# database.py : DB 연결 설정

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from models import Base

# 로컬 SQLite 파일(spectrum.db) 생성
SQLALCHEMY_DATABASE_URL = "sqlite:///./spectrum.db"

engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# 테이블 생성
Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
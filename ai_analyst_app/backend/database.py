from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

# SQLite 데이터베이스 파일 경로 설정 (루트 디렉토리에 ai_analyst.db 생성)
SQLALCHEMY_DATABASE_URL = "sqlite:///./ai_analyst.db"

# SQLite는 기본적으로 하나의 스레드에서만 통신을 허용하므로, 
# FastAPI의 비동기/다중 요청 처리를 위해 check_same_thread 옵션을 False로 설정합니다.
engine = create_engine(
    SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}
)

# 데이터베이스 세션 팩토리 생성
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# ORM 모델들이 상속받을 Base 클래스 생성
Base = declarative_base()

# FastAPI 라우터에서 사용할 DB 세션 의존성 함수
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
# config.py
import os
from pydantic_settings import BaseSettings, SettingsConfigDict
from dotenv import load_dotenv

load_dotenv()

class Settings(BaseSettings):
    # 프로젝트 기본 설정
    PROJECT_NAME: str = "AI-Analyst PB Project"
    PROJECT_VERSION: str = "1.0.0"
    API_V1_STR: str = "/api/v1"

    # API Keys (런타임에 업데이트 가능하도록 초기값은 .env나 None)
    OPENAI_API_KEY: str | None = None
    LANGCHAIN_API_KEY: str | None = None
    NAVER_CLIENT_ID: str | None = None      # 필드 추가
    NAVER_CLIENT_SECRET: str | None = None  # 필드 추가

    # LangSmith 설정
    LANGCHAIN_TRACING_V2: str = "true"
    LANGCHAIN_ENDPOINT: str = "https://api.smith.langchain.com"
    LANGCHAIN_PROJECT: str = "ai_analyst_app"

    model_config = SettingsConfigDict(
        env_file=".env", 
        env_file_encoding="utf-8", 
        case_sensitive=True,
        extra="ignore"
    )

settings = Settings()

# 런타임에 설정을 업데이트하는 헬퍼 함수
def update_runtime_settings(new_keys: dict):
    """
    UI(Sidebar) 등에서 입력받은 API 키를 
    1) settings 인스턴스에 반영하고 
    2) os.environ에도 강제 세팅하여 라이브러리(LangChain 등)가 인식하게 함
    """
    for key, value in new_keys.items():
        if value and value.strip():  # 빈 값이 아닐 때만 업데이트
            setattr(settings, key, value)
            os.environ[key] = value
            
    # LangSmith 전용 추가 처리 (필요시)
    if settings.LANGCHAIN_API_KEY:
        os.environ["LANGCHAIN_TRACING_V2"] = settings.LANGCHAIN_TRACING_V2
        os.environ["LANGCHAIN_API_KEY"] = settings.LANGCHAIN_API_KEY
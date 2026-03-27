# Python 3.11 슬림 이미지 사용
FROM python:3.11-slim

# 환경 변수 설정 (파이썬 출력 버퍼링 방지)
ENV PYTHONUNBUFFERED=1

# Selenium 웹 크롤링을 위한 Chromium 및 필수 패키지 설치
RUN apt-get update && apt-get install -y \
    wget \
    gnupg \
    unzip \
    chromium \
    chromium-driver \
    && rm -rf /var/lib/apt/lists/*

# 작업 디렉토리 설정
WORKDIR /app

# 의존성 파일 복사 및 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 백엔드 소스 코드 복사
COPY ./backend /app/backend

# FastAPI 포트 노출
EXPOSE 8000

# 서버 실행 명령어
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
# backend.Dockerfile : 백엔드 이미지 설계도

# 경량화된 파이썬 3.11 이미지 사용
FROM python:3.11-slim

# 작업 디렉토리 설정
WORKDIR /app

# SQLite 및 컴파일에 필요한 시스템 도구 설치
RUN apt-get update && apt-get install -y gcc

# 요구사항 파일 복사 및 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 백엔드 소스 코드 복사
COPY backend/ ./backend/

# 작업 공간을 백엔드 폴더로 이동
WORKDIR /app/backend

# FastAPI 포트 노출
EXPOSE 8000

# 서버 자동 실행 (main.py의 포트 탐색 로직 활용)
CMD ["python", "main.py"]
# 1. 베이스 이미지 설정 (Python 3.10 Slim 버전 사용으로 이미지 크기 최적화)
FROM python:3.10-slim

# 2. 작업 디렉토리 설정
WORKDIR /app

# 3. 시스템 필수 패키지 설치 (PyTorch 등 컴파일에 필요할 수 있음)
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 4. 파이썬 라이브러리 설치
# 캐시를 사용하지 않아 이미지 용량을 줄입니다.
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 5. 프로젝트 소스 코드 및 모델 가중치 복사
COPY . .

# 6. 로그 폴더 생성 및 권한 부여
RUN mkdir -p /app/logs && chmod 777 /app/logs

# 7. 포트 개방 안내 (8000: FastAPI, 8501: Streamlit)
EXPOSE 8000 8501

# 8. 기본 실행 명령어 세팅
# 기본값으로 Streamlit 앱을 실행하도록 설정합니다. (0.0.0.0 바인딩 필수)
CMD ["streamlit", "run", "app.py", "--server.address", "0.0.0.0"]
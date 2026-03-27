# Python 3.11 슬림 이미지 사용
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1

WORKDIR /app

# 프론트엔드 구동에 필요한 패키지 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 프론트엔드 소스 코드 복사
COPY ./frontend /app/frontend

# Streamlit 포트 노출
EXPOSE 8501

# Streamlit 앱 실행 명령어
CMD ["streamlit", "run", "frontend/0_📈_Home.py", "--server.port=8501", "--server.address=0.0.0.0"]
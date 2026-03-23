# frontend.Dockerfile : 프론트엔드 이미지 설계도

FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY frontend/ ./frontend/

WORKDIR /app/frontend

EXPOSE 8501

# Streamlit 설정 (변경된 파일명 0_🏠_홈.py 적용, 외부 접속 허용)
CMD ["streamlit", "run", "0_🏠_홈.py", "--server.port=8501", "--server.address=0.0.0.0"]
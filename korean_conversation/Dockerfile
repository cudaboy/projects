# 1. Base Image: 가벼운 리눅스(Debian) 위에 파이썬 3.11이 설치된 버전을 사용합니다.
FROM python:3.11-slim

# 2. 작업 디렉토리 설정: 컨테이너 내부의 기준 폴더를 /app으로 잡습니다.
WORKDIR /app

# 3. 시스템 필수 패키지 설치: C++ 컴파일러 등 일부 파이썬 패키지 설치에 필요한 도구들을 깝니다.
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 4. 패키지 설치: 파이썬 패키지 명세서를 먼저 복사하고 설치합니다. 
# (코드 수정 시 매번 패키지를 다시 다운받지 않도록 캐싱 효과를 누리기 위함입니다.)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. 소스 코드 및 데이터 복사: 현재 폴더의 모든 파일(rag_db, pth, py 등)을 컨테이너 안으로 복사합니다.
# (이때 .dockerignore에 적힌 파일들은 알아서 걸러집니다.)
COPY . .

# 6. 포트 노출: 이 컨테이너가 8000번(FastAPI)과 8501번(Streamlit) 포트를 사용한다는 것을 명시합니다.
EXPOSE 8000 8501

# 7. 실행 명령어: 컨테이너가 켜질 때 우리가 만든 'app.py' 지휘자를 실행합니다!
CMD ["python", "app.py"]
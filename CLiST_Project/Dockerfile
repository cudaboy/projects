# 1. Base Image: CUDA 12.8 환경을 지원하는 NVIDIA 공식 이미지를 기반으로 합니다.
# Ubuntu 24.04 환경에서 Python 3.11이 기본으로 탑재됩니다.
FROM nvidia/cuda:12.8.0-runtime-ubuntu24.04

# 2. 시스템 환경 변수 설정
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
WORKDIR /app

# 3. 필수 시스템 패키지 및 Python 설치
# OpenCV(이미지 처리)와 MLflow 등에 필요한 기본 라이브러리를 포함합니다.
RUN apt-get update && apt-get install -y \
    python3-dev \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# 4. uv 패키지 매니저 설치
# 현재 사용 중인 초고속 uv 매니저를 컨테이너 안에서도 그대로 사용합니다.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uv/bin/uv
ENV PATH="/uv/bin:$PATH"

# 5. 의존성 파일 복사 및 초고속 설치
# --no-cache 옵션을 통해 설치 후 용량을 잡아먹는 캐시 파일을 즉시 제거합니다.
# --index-strategy unsafe-best-match 옵션으로 PyTorch와 PyPI 서버 간의 충돌을 방지합니다.
COPY requirements.txt .
RUN uv pip install --no-cache -r requirements.txt \
     --system \
     --break-system-packages \
     --index-strategy unsafe-best-match

# 6. 프로젝트 전체 소스 코드 및 가중치 복사
# .dockerignore 파일을 만들어 데이터셋(processed_data)이 복사되지 않게 주의해야 합니다.
COPY . .

# 7. 진입점(Entrypoint) 설정
# 사용자가 docker run 실행 시 인자값만 주면 바로 run.py가 작동하도록 설계합니다.
ENTRYPOINT ["python3", "run.py"]
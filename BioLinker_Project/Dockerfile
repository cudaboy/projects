# 1. Base Image: CUDA 12.8 환경을 지원하는 NVIDIA 공식 이미지를 기반으로 합니다.
# Ubuntu 24.04 환경에서 Python 3.11이 기본으로 탑재됩니다.
FROM nvidia/cuda:12.8.0-runtime-ubuntu24.04

# 2. 시스템 환경 변수 설정
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
WORKDIR /app

# 3. 필수 시스템 패키지 및 Python 설치
RUN apt-get update && apt-get install -y \
    python3-dev \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# 4. uv 패키지 매니저 설치
# 초고속 uv 매니저를 컨테이너 안에서도 그대로 사용합니다.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uv/bin/uv
ENV PATH="/uv/bin:$PATH"

# 5. 의존성 파일 복사 및 초고속 설치
# --no-cache 옵션을 통해 캐시 파일을 즉시 제거하여 이미지 용량을 최소화합니다.
COPY requirements.txt .
RUN uv pip install --no-cache -r requirements.txt \
     --system \
     --break-system-packages \
     --index-strategy unsafe-best-match

# 6. 프로젝트 전체 소스 코드 복사
# .dockerignore 파일을 통해 무거운 로컬 DB나 원본 데이터셋은 복사되지 않습니다.
COPY . .

# 7. 진입점(Entrypoint) 설정
# 사용자가 docker run 실행 시 인자값(예: --start, --build)을 주어 run.py를 실행하도록 설계합니다.
ENTRYPOINT ["python3", "run.py"]
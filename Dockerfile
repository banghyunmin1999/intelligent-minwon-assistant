# =================================================================
# STAGE 1: Builder
# - 애플리케이션 실행에 필요한 모든 의존성을 설치하고 빌드하는 역할
# =================================================================
FROM python:3.12-slim as builder

WORKDIR /app

# 시스템 패키지 업데이트 및 빌드에 필요한 모든 도구 설치
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    autoconf \
    automake \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# 파이썬 가상 환경 생성
ENV VIRTUAL_ENV=/app/venv
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# requirements.txt 파일을 먼저 복사하여 캐시 활용
COPY requirements.txt .

# requirements.txt의 모든 라이브러리 설치
RUN pip install --no-cache-dir -r requirements.txt

# *** 핵심 수정: 특정 GPU 아키텍처를 하드코딩하는 대신, CUDA 지원만 활성화 ***
# 이렇게 하면 빌드 환경의 GPU를 자동으로 감지하여 최적의 라이브러리를 생성합니다.
RUN CMAKE_ARGS="-DLLAMA_CUDA=ON" \
    pip install --no-cache-dir --force-reinstall --no-binary :all: llama-cpp-python

# 애플리케이션 소스 코드 복사
COPY ./ai-engine /app

# =================================================================
# STAGE 2: Final
# - 실제 서비스 운영에 사용될 최종 이미지를 만드는 역할
# =================================================================
FROM python:3.12-slim

WORKDIR /app

# 보안을 위한 non-root 사용자 생성
RUN groupadd --gid 1001 appuser && \
    useradd --uid 1001 --gid 1001 -m appuser

# 필요한 시스템 라이브러리 설치
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libpq5 \
    && rm -rf /var/lib/apt/lists/*

USER appuser

# Builder 스테이지에서 생성된 결과물만 복사
COPY --from=builder /app/venv ./venv
COPY --from=builder /app .

# 환경 변수 설정
ENV PATH="/app/venv/bin:$PATH"

# 애플리케이션 실행
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

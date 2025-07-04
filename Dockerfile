# =================================================================
# STAGE 1: Builder
# - 애플리케이션 실행에 필요한 모든 의존성을 설치하고 빌드하는 역할
# - 컴파일러 등 무거운 도구들이 이 스테이지에만 포함됩니다.
# =================================================================
FROM python:3.12-slim as builder

# 작업 디렉토리 설정
WORKDIR /app

# 시스템 패키지 업데이트 및 빌드에 필요한 도구 설치
# *** 핵심 수정: PostgreSQL 연결에 필요한 시스템 라이브러리 'libpq-dev'를 추가합니다. ***
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    autoconf \
    automake \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# 파이썬 가상 환경 생성 및 활성화 경로 설정
ENV VIRTUAL_ENV=/app/venv
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# requirements.txt 파일을 먼저 복사하여, 의존성 변경이 없을 시 캐시를 활용하도록 함
COPY requirements.txt .

# pip를 사용하여 모든 파이썬 라이브러리를 설치합니다.
RUN pip install --no-cache-dir -r requirements.txt

# CPU/GPU 아키텍처에 맞춰 llama-cpp-python을 소스코드부터 직접 컴파일합니다.
RUN CMAKE_ARGS="-DLLAMA_AVX=OFF -DLLAMA_AVX2=OFF -DLLAMA_FMA=OFF -DLLAMA_CUDA_TARGET_ARCHS=86" \
    pip install --no-cache-dir --force-reinstall --no-binary :all: llama-cpp-python

# 애플리케이션 소스 코드를 복사합니다.
COPY ./ai-engine /app

# =================================================================
# STAGE 2: Final
# - 실제 서비스 운영에 사용될 최종 이미지를 만드는 역할
# - Builder 스테이지에서 생성된 결과물만 가져와 매우 가볍고 안전합니다.
# =================================================================
FROM python:3.12-slim

# 작업 디렉토리 설정
WORKDIR /app

# 시스템에 non-root 사용자를 생성하여 보안을 강화합니다.
RUN groupadd --gid 1001 appuser && \
    useradd --uid 1001 --gid 1001 -m appuser

# *** 핵심 수정: libpq 런타임 라이브러리 설치 ***
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libpq5 \
    && rm -rf /var/lib/apt/lists/*

USER appuser

# Builder 스테이지에서 생성된 가상 환경(venv)과 소스 코드를 복사합니다.
COPY --from=builder /app/venv ./venv
COPY --from=builder /app .

# 환경 변수 설정 (가상 환경 활성화)
ENV PATH="/app/venv/bin:$PATH"

# 애플리케이션 실행
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

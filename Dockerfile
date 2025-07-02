FROM nvidia/cuda:12.2.0-cudnn8-runtime-ubuntu22.04

WORKDIR /app

# 설치할 패키지들
COPY requirements.txt .
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir -r requirements.txt \
    && pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 애플리케이션 코드 복사
COPY . .

# 환경 변수 설정
ENV PYTHONPATH=/app

# 애플리케이션 실행
CMD ["uvicorn", "ai-engine.main:app", "--host", "0.0.0.0", "--port", "8000"]

# --- 1단계: 빌더 (Builder) 스테이지 ---
    FROM python:3.12-slim as builder

    RUN apt-get update && apt-get install -y --no-install-recommends gcc build-essential
    WORKDIR /app
    COPY requirements.txt .
    RUN pip install --no-cache-dir -r requirements.txt
    
    # --- 2단계: 최종 (Final) 스테이지 ---
    FROM python:3.12-slim
    
    RUN apt-get update && apt-get install -y libgomp1 && rm -rf /var/lib/apt/lists/*
    WORKDIR /app
    
    COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
    COPY --from=builder /usr/local/bin /usr/local/bin
    
    COPY server.py .
    COPY .env .
    
    # [수정] 새로운 1.5B 모델 파일을 복사하도록 경로 변경
    COPY model/gemma-3-1b-thinking-v2-q4_k_m.gguf ./model/
    
    EXPOSE 8000
    
    CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
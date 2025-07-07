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
    
    # [추가] 이 지점부터 캐시를 무효화할 수 있는 스위치를 만듭니다.
    ARG CACHE_BUSTER
    
    # 애플리케이션 파일 복사 (이제 이 부분은 --build-arg 값에 따라 캐시가 무효화됩니다)
    COPY server.py .
    COPY .env .
    COPY model/hyperclovax-seed-text-instruct-1.5b-q4_k_m.gguf ./model/
    
    EXPOSE 8000
    
    CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
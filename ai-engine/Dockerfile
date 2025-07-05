# 1. 베이스 이미지 선택 (파이썬 3.9 슬림 버전)
FROM python:3.9-slim

# 2. 작업 디렉토리 설정
WORKDIR /app

# 3. 의존성 파일 복사 및 설치
# (나중에 requirements.txt 파일을 만들어야 합니다)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. 소스 코드 복사
# server.py를 WORKDIR(/app)로 복사
COPY ../server.py .

# 5. 서버 실행
# 컨테이너가 시작될 때 실행할 명령어
# FastAPI 기본 포트는 8000번입니다.
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
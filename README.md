# Intelligent Minwon Assistant

Intelligent Minwon Assistant는 민원 처리를 위한 인공지능 기반의 도우미 시스템입니다.

## 프로젝트 구조

```
intelligent-minwon-assistant/
├── .env                 # 환경 변수 설정 파일
├── ai-engine/           # AI 엔진 관련 코드
├── models/             # 모델 관련 파일
├── scripts/            # 스크립트 파일
├── test.py            # 테스트 파일
└── README.md          # 프로젝트 설명
```

## 설치 및 실행

1. PostgreSQL 설치 및 설정
   ```bash
   # PostgreSQL 설치
   sudo apt-get update
   sudo apt-get install postgresql postgresql-contrib
   
   # PostgreSQL 시작
   sudo systemctl start postgresql
   
   # PostgreSQL 사용자 및 데이터베이스 생성
   sudo -u postgres psql -c "CREATE USER minwon_user WITH PASSWORD '1234';"
   sudo -u postgres createdb minwon_pg_db
   sudo -u postgres psql -c "GRANT ALL PRIVILEGES ON DATABASE minwon_pg_db TO minwon_user;"
   
   # vector 확장 설치
   sudo apt-get install postgresql-16-pgvector
   sudo -u postgres psql minwon_pg_db -c "CREATE EXTENSION vector;"
   ```

2. 환경 변수 설정
   - `.env` 파일을 생성하고 필요한 환경 변수를 설정합니다.
   - 주요 환경 변수: PG_HOST, PG_PORT, PG_USER, PG_PASSWORD, PG_DB_NAME

3. 의존성 설치
   ```bash
   # Python 패키지 설치
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

4. 데이터베이스 테이블 생성
   ```bash
   sudo -u postgres psql minwon_pg_db -c "CREATE TABLE minwon_cases_pg (create_date BIGINT NOT NULL PRIMARY KEY, title VARCHAR(500), content TEXT, institution VARCHAR(255), department VARCHAR(255), search_keyword VARCHAR(255), collected_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP, embedding vector(768));"
   ```

5. 데이터 수집 실행
   ```bash
   python3 scripts/pg_data_collector.py
   ```

6. 서버 실행
   ```bash
   # ai-engine 디렉토리로 이동
   cd ai-engine
   
   # 서버 실행 (개발 모드)
   uvicorn main:app --reload
   
   # 또는 프로덕션 모드로 실행
   uvicorn main:app --host 0.0.0.0 --port 8000
   ```

   서버가 실행되면:
   - API 문서: http://localhost:8000/docs
   - 테스트 엔드포인트: http://localhost:8000/ask

## 데이터베이스 초기화 및 데이터 삽입

### 데이터베이스 초기화
```bash
# PostgreSQL 접속
sudo -u postgres psql minwon_pg_db

# 테이블 삭제
DROP TABLE IF EXISTS minwon_cases_pg;

# 테이블 생성
CREATE TABLE minwon_cases_pg (
    create_date BIGINT NOT NULL PRIMARY KEY,
    title VARCHAR(500),
    content TEXT,
    institution VARCHAR(255),
    department VARCHAR(255),
    search_keyword VARCHAR(255),
    collected_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    embedding vector(768)
);

# 벡터 확장 설치 확인
CREATE EXTENSION IF NOT EXISTS vector;
```

### 데이터 삽입
```bash
# 데이터 수집 실행
python3 scripts/pg_data_collector.py

# 데이터베이스에 데이터 확인
sudo -u postgres psql minwon_pg_db -c "SELECT COUNT(*) FROM minwon_cases_pg;"
```

## 사용법

프로젝트의 주요 기능과 사용 방법을 설명합니다.

## 기술 스택

- Python
- AI/ML
- Database
- 환경 변수 관리 (python-dotenv)

## 버전 관리

- v3.0.0: PostgreSQL 설치 및 데이터베이스 설정 지침 업데이트

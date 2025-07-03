-- PostgreSQL용 데이터베이스 설정 및 테이블 생성 스크립트

-- 1. pgvector 확장 활성화 (DB 슈퍼유저 권한 필요)
-- 이 확장은 벡터 유사도 검색 기능을 제공합니다.
CREATE EXTENSION IF NOT EXISTS vector;

-- 2. 'minwon_cases_pg' 테이블 생성
-- 기존 MySQL 테이블 구조에 벡터 저장을 위한 컬럼을 추가합니다.
CREATE TABLE IF NOT EXISTS minwon_cases_pg (
    -- 민원 생성 날짜 (YYYYMMDDHHMMSS 형식의 숫자). 기본 키로 사용됩니다.
    create_date BIGINT NOT NULL PRIMARY KEY,

    -- 민원의 제목.
    title VARCHAR(500),

    -- 민원의 상세 내용.
    content TEXT,

    -- 담당 기관명.
    institution VARCHAR(255),

    -- 담당 부서명.
    department VARCHAR(255),

    -- 데이터 수집 시 사용된 검색 키워드.
    search_keyword VARCHAR(255),

    -- 데이터가 데이터베이스에 수집된 시간.
    collected_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,

    -- 텍스트 내용의 벡터 표현을 저장하는 컬럼.
    -- 768은 'jhgan/ko-sbert-nli' 모델의 임베딩 차원(dimension)입니다.
    -- 사용하는 임베딩 모델에 따라 이 숫자는 달라져야 합니다.
    embedding vector(768)
);

-- 3. (선택 사항) 벡터 검색 성능 향상을 위한 인덱스 생성
-- 데이터 양이 많아지면 HNSW 인덱스를 생성하여 검색 속도를 크게 향상시킬 수 있습니다.
-- 코사인 유사도(cosine distance)를 사용하여 인덱스를 생성합니다.
-- CREATE INDEX ON minwon_cases_pg USING hnsw (embedding vector_cosine_ops);
import os
from dotenv import load_dotenv
import mysql.connector
import psycopg2
from psycopg2.extras import execute_values
import numpy as np
from langchain_community.embeddings import HuggingFaceEmbeddings

# --- 1. 설정 (Configuration) ---
print("환경 변수를 로드합니다...")
load_dotenv()

# 소스 DB (MySQL) 연결 정보
MYSQL_CONFIG = {
    "host": os.getenv("DB_HOST"),
    "port": int(os.getenv("DB_PORT")),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
    "database": os.getenv("DB_NAME")
}

# 타겟 DB (PostgreSQL) 연결 정보
POSTGRES_CONFIG = {
    "host": os.getenv("PG_HOST", "localhost"),
    "port": os.getenv("PG_PORT", "5432"),
    "user": os.getenv("PG_USER", "minwon_user"),
    "password": os.getenv("PG_PASSWORD", "1234"),
    "dbname": os.getenv("PG_DB_NAME", "minwon_pg_db")
}

# 임베딩 모델 설정
EMBEDDING_MODEL_CONFIG = {
    "model_name": "jhgan/ko-sbert-nli",
    "model_kwargs": {'device': 'cuda'},
    "encode_kwargs": {'normalize_embeddings': True}
}

# --- 2. 데이터 처리 함수 ---

def migrate_and_embed():
    """
    MySQL에서 데이터를 로드하여 임베딩한 후 PostgreSQL에 저장합니다.
    """
    # 1. MySQL에서 데이터 로드
    print("MySQL에서 원본 데이터를 로드합니다...")
    try:
        mysql_conn = mysql.connector.connect(**MYSQL_CONFIG)
        cursor = mysql_conn.cursor(dictionary=True)
        cursor.execute("SELECT create_date, title, content, institution, department, search_keyword FROM minwon_cases WHERE content IS NOT NULL AND content != ''")
        rows = cursor.fetchall()
        cursor.close()
        mysql_conn.close()
        if not rows:
            print("MySQL에 데이터가 없습니다. 프로세스를 종료합니다.")
            return
        print(f"✅ 총 {len(rows)}개의 문서를 MySQL에서 로드했습니다.")
    except mysql.connector.Error as e:
        print(f"❌ MySQL 처리 중 오류 발생: {e}")
        return

    # 2. 임베딩 모델 초기화
    print(f"임베딩 모델 '{EMBEDDING_MODEL_CONFIG['model_name']}'을(를) 초기화합니다...")
    try:
        embeddings = HuggingFaceEmbeddings(**EMBEDDING_MODEL_CONFIG)
    except Exception as e:
        print(f"❌ 임베딩 모델 로딩 실패: {e}")
        return

    # 3. 텍스트 임베딩
    contents = [row['content'] for row in rows]
    print(f"총 {len(contents)}개의 문서를 벡터로 변환합니다. (시간이 소요될 수 있습니다)")
    try:
        embedded_vectors = embeddings.embed_documents(contents)
        print("✅ 문서 임베딩 완료.")
    except Exception as e:
        print(f"❌ 문서 임베딩 중 오류 발생: {e}")
        return

    # 4. PostgreSQL에 데이터 저장
    print("PostgreSQL에 데이터를 저장합니다...")
    try:
        with psycopg2.connect(**POSTGRES_CONFIG) as pg_conn:
            with pg_conn.cursor() as cur:
                insert_query = """
                INSERT INTO minwon_cases_pg (create_date, title, content, institution, department, search_keyword, embedding)
                VALUES %s
                ON CONFLICT (create_date) DO UPDATE SET
                    title = EXCLUDED.title,
                    content = EXCLUDED.content,
                    institution = EXCLUDED.institution,
                    department = EXCLUDED.department,
                    search_keyword = EXCLUDED.search_keyword,
                    embedding = EXCLUDED.embedding;
                """
                
                data_to_insert = [
                    (
                        row['create_date'], row['title'], row['content'],
                        row['institution'], row['department'], row['search_keyword'],
                        np.array(vec).tolist() # 벡터를 리스트 형태로 변환
                    ) for row, vec in zip(rows, embedded_vectors)
                ]

                execute_values(cur, insert_query, data_to_insert)
                pg_conn.commit()
                print(f"✅ {len(data_to_insert)}개의 데이터가 PostgreSQL에 성공적으로 이전 및 저장되었습니다.")
    except psycopg2.Error as e:
        print(f"❌ PostgreSQL 처리 중 오류 발생: {e}")

if __name__ == "__main__":
    migrate_and_embed()

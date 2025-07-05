import os
import sys
import json
import psycopg2
from dotenv import load_dotenv

# --- 1. 설정 (Configuration) ---

# 프로젝트 루트 디렉토리를 명확하게 정의합니다.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# .env 파일 경로를 지정합니다.
DOTENV_PATH = os.path.join(PROJECT_ROOT, ".env")

print("환경 변수를 로드합니다...")
if not os.path.exists(DOTENV_PATH):
    print(f"❌ [오류] .env 파일을 찾을 수 없습니다: {DOTENV_PATH}")
    sys.exit()

load_dotenv(dotenv_path=DOTENV_PATH)

# PostgreSQL DB 설정
DB_CONFIG = {
    "host": os.getenv("PG_HOST", "localhost"),
    "port": os.getenv("PG_PORT", "5432"),
    "dbname": os.getenv("PG_DB_NAME", "minwon_pg_db"),
    "user": os.getenv("PG_USER", "minwon_user"),
    "password": os.getenv("PG_PASSWORD", "1234"),
}
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "minwon_qna_cases")

# --- 2. 데이터베이스 확인 함수 ---

def check_vector_database():
    """
    PostgreSQL에 연결하여 LangChain이 저장한 데이터 샘플을 확인합니다.
    """
    print("\n" + "="*50)
    print(" PostgreSQL 벡터 데이터베이스 데이터 확인 시작")
    print("="*50)

    connection = None
    try:
        # 데이터베이스에 연결
        print("1. 데이터베이스에 연결 중...")
        connection = psycopg2.connect(**DB_CONFIG)
        cursor = connection.cursor()
        print("✅ 연결 성공!")

        # LangChain 컬렉션(테이블) 존재 여부 확인
        print("\n2. LangChain 관리 테이블 확인 중...")
        cursor.execute("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_name = 'langchain_pg_collection'
            );
        """)
        if not cursor.fetchone()[0]:
            print("❌ [오류] 'langchain_pg_collection' 테이블이 존재하지 않습니다.")
            print("   - 데이터 저장 스크립트(pg_data_parse.py)를 먼저 실행해야 합니다.")
            return

        cursor.execute("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_name = 'langchain_pg_embedding'
            );
        """)
        if not cursor.fetchone()[0]:
            print("❌ [오류] 'langchain_pg_embedding' 테이블이 존재하지 않습니다.")
            print("   - 데이터 저장 스크립트(pg_data_parse.py)를 먼저 실행해야 합니다.")
            return
        
        print("✅ LangChain 관리 테이블 확인 완료.")

        # 데이터 샘플 가져오기
        print(f"\n3. '{COLLECTION_NAME}' 컬렉션에서 데이터 샘플 5개 가져오는 중...")
        
        query = """
            SELECT
                e.document,
                e.cmetadata,
                e.embedding::text
            FROM langchain_pg_embedding AS e
            JOIN langchain_pg_collection AS c ON e.collection_id = c.uuid
            WHERE c.name = %s
            LIMIT 5;
        """
        cursor.execute(query, (COLLECTION_NAME,))
        rows = cursor.fetchall()

        if not rows:
            print("❌ [오류] 데이터베이스에 저장된 데이터가 없습니다.")
            print("   - 데이터 저장 스크립트(pg_data_parse.py)가 정상적으로 실행되었는지 확인해주세요.")
            return
        
        print(f"✅ 총 {len(rows)}개의 데이터 샘플을 성공적으로 가져왔습니다.")

        # 결과 출력
        print("\n" + "-"*50)
        for i, row in enumerate(rows):
            document, cmetadata, embedding = row
            print(f"\n--- 샘플 데이터 {i+1} ---")
            print(f"  [질문 (document)]: {document[:100]}...")
            
            # 메타데이터(답변 포함)를 예쁘게 출력
            if isinstance(cmetadata, str):
                cmetadata = json.loads(cmetadata)
            
            answer = cmetadata.get('answer', 'N/A')
            source = cmetadata.get('source', 'N/A')

            print(f"  [답변 (metadata.answer)]: {answer[:100]}...")
            print(f"  [출처 (metadata.source)]: {source}")
            print(f"  [벡터 (embedding)]: {embedding[:100]}...")
        print("\n" + "-"*50)

        print("\n[최종 진단]")
        print("위 내용이 정상적으로 보인다면, 데이터베이스에는 문제가 없을 가능성이 높습니다.")
        print("답변이 깨지는 현상은 DB 데이터가 아닌, AI 모델 라이브러리와 GPU의 호환성 문제일 확률이 매우 높습니다.")


    except psycopg2.Error as e:
        print(f"\n❌ 데이터베이스 처리 중 오류 발생: {e}")
    finally:
        if connection:
            connection.close()
            print("\n데이터베이스 연결을 닫았습니다.")

# --- 3. 메인 실행 블록 ---
if __name__ == "__main__":
    check_vector_database()

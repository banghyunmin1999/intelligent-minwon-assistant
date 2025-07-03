import os
from dotenv import load_dotenv
import mysql.connector
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document
from langchain_community.vectorstores import FAISS

print("환경 변수를 로드합니다...")
load_dotenv()

DB_CONFIG = {
    "host": os.getenv("DB_HOST"),
    "port": int(os.getenv("DB_PORT")),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
    "database": os.getenv("DB_NAME")
}

def load_docs_from_db():
    print("MySQL 데이터베이스에 연결 중...")
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor(dictionary=True)
        print("데이터 로딩 중...")
        cursor.execute("SELECT title, content, create_date FROM minwon_cases WHERE content IS NOT NULL AND content != ''")
        rows = cursor.fetchall()
        conn.close()
        print(f"총 {len(rows)}개의 문서를 데이터베이스에서 로드했습니다.")
        return [
            Document(
                page_content=row['content'],
                metadata={'title': row['title'], 'create_date': str(row['create_date'])}
            ) for row in rows
        ]
    except mysql.connector.Error as e:
        print(f"데이터베이스 오류: {e}")
        return []

def create_and_save_vector_store():
    docs = load_docs_from_db()
    if not docs:
        print("벡터 스토어를 생성할 문서가 없습니다. (데이터 수집이 필요할 수 있습니다)")
        return

    print("오픈소스 임베딩 모델을 초기화합니다...")
    model_name = "jhgan/ko-sbert-nli"
    model_kwargs = {'device': 'cuda'}
    encode_kwargs = {'normalize_embeddings': True}
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )

    print("문서를 벡터로 변환하고 FAISS 인덱스를 생성합니다...")
    vector_store = FAISS.from_documents(docs, embeddings)
    vector_store.save_local("ai-engine/faiss_index")
    print("✅ FAISS 인덱스를 'ai-engine/faiss_index' 폴더에 성공적으로 저장했습니다.")

if __name__ == "__main__":
    create_and_save_vector_store()

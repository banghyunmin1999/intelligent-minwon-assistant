# ingestion.py

import os
from langchain_community.document_loaders import PyPDFLoader # PDF 로더로 변경
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_postgres.vectorstores import PGVector
from langchain_text_splitters import RecursiveCharacterTextSplitter

# --- VectorDB (PostgreSQL) Settings ---
PG_HOST = "localhost"
PG_PORT = 5432
PG_DATABASE = "minwon_pg_db"
PG_USER = "minwon_user"
PG_PASSWORD = "1234"
# ★ 요청하신 대로 새로운 테이블(컬렉션) 이름을 사용합니다.
COLLECTION_NAME = "minwon_pdf_cases_v1" 
CONNECTION_STRING = f"postgresql+psycopg2://{PG_USER}:{PG_PASSWORD}@{PG_HOST}:{PG_PORT}/{PG_DATABASE}"

# --- Source Document ---
# ★ 알려주신 PDF 파일 경로를 사용합니다.
PDF_FILE_PATH = "/home/bang/intelligent-minwon-assistant/data/2023년 농지민원 사례집(최종).pdf"
loader = PyPDFLoader(PDF_FILE_PATH)
documents = loader.load()
print(f"✅ PDF 문서 로드 완료. 총 {len(documents)} 페이지입니다.")

# --- 1. Chunking ---
# 텍스트를 의미있는 단위로 분할합니다.
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100,
    length_function=len,
)
chunks = text_splitter.split_documents(documents)
print(f"✅ 문서가 {len(chunks)}개의 청크로 분할되었습니다.")

# --- 2. Embedding Model ---
# 한국어에 더 적합한 임베딩 모델을 사용합니다.
print("⏳ 임베딩 모델을 로드합니다...")
embeddings = HuggingFaceEmbeddings(
    model_name="jhgan/ko-sroberta-multitask",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)
print("✅ 임베딩 모델 로드 완료.")

# --- 3. Store to VectorDB ---
# 새 컬렉션에 데이터를 저장합니다.
print("⏳ 벡터 데이터베이스에 연결하고 데이터를 저장합니다...")
db = PGVector.from_documents(
    embedding=embeddings,
    documents=chunks,
    collection_name=COLLECTION_NAME,
    connection=CONNECTION_STRING,
    pre_delete_collection=True, # 스크립트 실행 시마다 이 컬렉션을 새로 만듭니다.
)
print(f"✅ {len(chunks)}개의 청크가 '{COLLECTION_NAME}' 컬렉션에 성공적으로 저장되었습니다.")
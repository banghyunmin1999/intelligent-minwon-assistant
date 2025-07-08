import os
import re
import torch
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_postgres.vectorstores import PGVector
from langchain_core.documents import Document

# .env 파일 로드
load_dotenv()

# --- 설정 값 읽기 및 자동 계산 ---
# GPU 가용 여부에 따라 임베딩 모델 장치 자동 설정
EMBEDDING_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"임베딩 장치를 '{EMBEDDING_DEVICE}'로 설정합니다.")

# .env에서 임베딩 모델 이름 읽어오기
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL", "jhgan/ko-sroberta-multitask")

# VectorDB (PostgreSQL) Settings
PG_HOST = os.getenv("DB_HOST", "localhost")
PG_PORT = os.getenv("DB_PORT", 5432)
PG_DATABASE = os.getenv("DB_NAME", "minwon_pg_db")
PG_USER = os.getenv("DB_USER", "minwon_user")
PG_PASSWORD = os.getenv("DB_PASSWORD", "1234")
COLLECTION_NAME = "minwon_pdf_cases_v3"
CONNECTION_STRING = f"postgresql+psycopg2://{PG_USER}:{PG_PASSWORD}@{PG_HOST}:{PG_PORT}/{PG_DATABASE}"

# --- Source Document ---
PDF_FILE_PATH = "/home/bang/intelligent-minwon-assistant/data/2023년 농지민원 사례집(최종).pdf"
loader = PyPDFLoader(PDF_FILE_PATH)
documents = loader.load()
print(f"✅ PDF 문서 로드 완료. 총 {len(documents)} 페이지입니다.")


# --- 1. 텍스트 전처리 함수 ---
def clean_text(page_content):
    cleaned_text = re.sub(r'\d*\s*농지민원 사례집\s*❘\s*www\.mafra\.go\.kr', '', page_content)
    cleaned_text = re.sub(r'I\.\s*농지 정의 및 취득·처분', '', cleaned_text)
    cleaned_text = re.sub(r' 농지 정의', '', cleaned_text)
    cleaned_text = re.sub(r'농지 소유·농취증', '', cleaned_text)
    cleaned_text = re.sub(r'^\s*\d+\s*$', '', cleaned_text, flags=re.MULTILINE)
    cleaned_text = re.sub(r'^\s*iv\s*$', '', cleaned_text, flags=re.MULTILINE)
    cleaned_text = re.sub(r'\s+\d+$', '', cleaned_text)
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
    return cleaned_text

# --- 2. 의미 기반 Chunking ---
all_text = " ".join([clean_text(doc.page_content) for doc in documents])
pattern = re.compile(r'(문\s*\d+\s*.*?)(?=문\s+\d+|$)', re.DOTALL)
matches = pattern.findall(all_text)

new_chunks = []
if matches:
    for match in matches:
        if '답변' in match:
            question_part = match.split('답변')[0]
            answer_part = match.split('답변', 1)[1]
            
            question_text = re.sub(r'^문\s*\d+\s*', '', question_part).strip()
            answer_text = answer_part.strip()
            
            full_qa_text = f"질문: {question_text}\n답변: {answer_text}"
            new_chunks.append(Document(page_content=full_qa_text, metadata={"source": "농지민원 사례집"}))

print(f"✅ 문서가 {len(new_chunks)}개의 의미 기반 청크(Q&A)로 분할되었습니다.")

# --- 3. Embedding Model ---
print(f"⏳ 임베딩 모델({EMBEDDING_MODEL_NAME})을 로드합니다...")
embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL_NAME,
    model_kwargs={'device': EMBEDDING_DEVICE},
    encode_kwargs={'normalize_embeddings': True}
)
print("✅ 임베딩 모델 로드 완료.")

# --- 4. Store to VectorDB ---
print("⏳ 벡터 데이터베이스에 연결하고 데이터를 저장합니다...")
if new_chunks:
    db = PGVector.from_documents(
        embedding=embeddings,
        documents=new_chunks,
        collection_name=COLLECTION_NAME,
        connection=CONNECTION_STRING,
        pre_delete_collection=True,
    )
    print(f"✅ {len(new_chunks)}개의 청크가 '{COLLECTION_NAME}' 컬렉션에 성공적으로 저장되었습니다.")
else:
    print("⚠️ 저장할 청크가 없습니다. 전처리 또는 분할 로직을 확인하세요.")
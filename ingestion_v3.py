import os
import re
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_postgres.vectorstores import PGVector
from langchain_core.documents import Document

# --- VectorDB (PostgreSQL) Settings ---
PG_HOST = "localhost"
PG_PORT = 5432
PG_DATABASE = "minwon_pg_db"
PG_USER = "minwon_user"
PG_PASSWORD = "1234"
COLLECTION_NAME = "minwon_pdf_cases_v3" 
CONNECTION_STRING = f"postgresql+psycopg2://{PG_USER}:{PG_PASSWORD}@{PG_HOST}:{PG_PORT}/{PG_DATABASE}"

# --- Source Document ---
PDF_FILE_PATH = "/home/bang/intelligent-minwon-assistant/data/2023년 농지민원 사례집(최종).pdf"
loader = PyPDFLoader(PDF_FILE_PATH)
documents = loader.load()
print(f"✅ PDF 문서 로드 완료. 총 {len(documents)} 페이지입니다.")


# --- 1. 텍스트 전처리 함수 ---
def clean_text(page_content):
    # 불필요한 머리글/바닥글 및 페이지 번호 관련 텍스트 제거
    cleaned_text = re.sub(r'\d*\s*농지민원 사례집\s*❘\s*www\.mafra\.go\.kr', '', page_content)
    cleaned_text = re.sub(r'I\.\s*농지 정의 및 취득·처분', '', cleaned_text)
    cleaned_text = re.sub(r' 농지 정의', '', cleaned_text)
    cleaned_text = re.sub(r'농지 소유·농취증', '', cleaned_text)
    # ... 기타 불필요한 패턴들을 여기에 추가 ...
    
    # 줄바꿈을 공백으로 바꾸고 여러 공백을 하나로 합침
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
    return cleaned_text

# --- 2. [수정] 더 안정적인 의미 기반 Chunking ---
all_text = " ".join([clean_text(doc.page_content) for doc in documents])

# '문 00' 패턴을 사용하여 Q&A 블록을 찾기 위한 정규표현식
pattern = re.compile(r'(문\s*\d+\s*.*?)(?=문\s+\d+|$)', re.DOTALL)
matches = pattern.findall(all_text)

new_chunks = []
for match in matches:
    if '답변' in match:
        # 질문과 답변 부분을 분리
        question_part = match.split('답변')[0]
        answer_part = match.split('답변', 1)[1]
        
        # '문 00' 부분을 정리
        question_text = re.sub(r'^문\s*\d+\s*', '', question_part).strip()
        answer_text = answer_part.strip()

        # 최종 청크 텍스트 생성
        full_qa_text = f"질문: {question_text}\n답변: {answer_text}"
        
        # 메타데이터와 함께 Document 객체 생성
        # 페이지 번호 등 더 유용한 정보를 메타데이터에 추가할 수 있습니다.
        new_chunks.append(Document(page_content=full_qa_text, metadata={"source": "농지민원 사례집"}))

print(f"✅ 문서가 {len(new_chunks)}개의 의미 기반 청크(Q&A)로 분할되었습니다.")

# --- 3. Embedding Model ---
print("⏳ 임베딩 모델을 로드합니다...")
embeddings = HuggingFaceEmbeddings(
    model_name="jhgan/ko-sroberta-multitask",
    model_kwargs={'device': 'cpu'},
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
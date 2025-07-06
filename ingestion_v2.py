# ingestion.py (개선된 버전)

import os
import re # 정규표현식 라이브러리 추가
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_postgres.vectorstores import PGVector
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document # Document 객체 생성을 위해 추가

# --- VectorDB (PostgreSQL) Settings ---
PG_HOST = "localhost"
PG_PORT = 5432
PG_DATABASE = "minwon_pg_db"
PG_USER = "minwon_user"
PG_PASSWORD = "1234"
COLLECTION_NAME = "minwon_pdf_cases_v2" # 새 전략을 적용했으므로 버전업
CONNECTION_STRING = f"postgresql+psycopg2://{PG_USER}:{PG_PASSWORD}@{PG_HOST}:{PG_PORT}/{PG_DATABASE}"

# --- Source Document ---
PDF_FILE_PATH = "/home/bang/intelligent-minwon-assistant/data/2023년 농지민원 사례집(최종).pdf"
loader = PyPDFLoader(PDF_FILE_PATH)
documents = loader.load()
print(f"✅ PDF 문서 로드 완료. 총 {len(documents)} 페이지입니다.")


# --- [개선 1] 텍스트 전처리 함수 ---
# 머리글, 바닥글 등 불필요한 텍스트를 정규표현식으로 제거합니다.
def clean_text(page_content):
    # PDF 머리글/바닥글 패턴 예시 (실제 내용에 맞게 수정 필요)
    cleaned_text = re.sub(r'농지민원 사례집\s*\|\s*www\.mafra\.go\.kr', '', page_content)
    cleaned_text = re.sub(r'\d+I\.\s*농지 정의 및 취득·처분\s*\d+', '', cleaned_text)
    # 기타 불필요한 공백이나 줄바꿈 정리
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
    return cleaned_text

# --- [개선 2] 의미 기반 Chunking ---
# "문"으로 시작해서 "답변"으로 끝나는 Q&A 쌍을 하나의 청크로 묶습니다.
all_text = "".join([clean_text(doc.page_content) for doc in documents])

# "문"을 기준으로 텍스트를 분할하여 Q&A 리스트 생성
qa_pairs = re.split(r'(문\s+\d+)', all_text)
new_chunks = []
# 분할된 텍스트를 다시 "문 + 내용" 형태로 조합
for i in range(1, len(qa_pairs), 2):
    question_part = qa_pairs[i]
    answer_part = qa_pairs[i+1]
    
    # "답변" 키워드를 기준으로 질문과 답변을 분리
    if '답변' in answer_part:
        question_content = answer_part.split('답변')[0].strip()
        answer_content = answer_part.split('답변')[1].strip()
        
        # Q&A를 합쳐서 하나의 의미 있는 청크로 만듦
        full_qa_text = f"질문: {question_content}\n답변: {answer_content}"
        
        # 원본 문서 정보(메타데이터)를 유지하며 새로운 Document 객체 생성
        # 여기서는 간단하게 메타데이터를 비워두지만, 실제로는 페이지 번호 등을 넣으면 좋습니다.
        new_chunks.append(Document(page_content=full_qa_text, metadata={"source": "농지민원 사례집"}))

print(f"✅ 문서가 {len(new_chunks)}개의 의미 기반 청크(Q&A)로 분할되었습니다.")

# --- Embedding Model ---
print("⏳ 임베딩 모델을 로드합니다...")
embeddings = HuggingFaceEmbeddings(
    model_name="jhgan/ko-sroberta-multitask",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)
print("✅ 임베딩 모델 로드 완료.")

# --- Store to VectorDB ---
print("⏳ 벡터 데이터베이스에 연결하고 데이터를 저장합니다...")
if new_chunks: # 생성된 청크가 있을 경우에만 실행
    db = PGVector.from_documents(
        embedding=embeddings,
        documents=new_chunks, # 새로 만든 청크 사용
        collection_name=COLLECTION_NAME,
        connection=CONNECTION_STRING,
        pre_delete_collection=True,
    )
    print(f"✅ {len(new_chunks)}개의 청크가 '{COLLECTION_NAME}' 컬렉션에 성공적으로 저장되었습니다.")
else:
    print("⚠️ 저장할 청크가 없습니다. 전처리 또는 분할 로직을 확인하세요.")
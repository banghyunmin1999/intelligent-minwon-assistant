import sys
from llama_cpp import Llama
from langchain_postgres.vectorstores import PGVector
from langchain_huggingface import HuggingFaceEmbeddings

# VectorDB (PostgreSQL) Settings
PG_DRIVER = "postgresql+asyncpg"
PG_HOST = "localhost"
PG_PORT = 5432
PG_DATABASE = "minwon_pg_db"
PG_USER = "minwon_user"
PG_PASSWORD = "1234"
COLLECTION_NAME = "minwon_qna_cases"

# 모델 경로 설정
model_path = "/home/bang/intelligent-minwon-assistant/models/llama-3-Korean-Bllossom-8B-Q2_K-GGUF/llama-3-korean-bllossom-8b-q2_k.gguf"

# 모델 로드
llm = Llama(
    model_path=model_path,
    n_ctx=2048,  # 컨텍스트 크기
    n_threads=4,  # 스레드 수 설정
    max_tokens=256,  # 생성할 토큰 수 (더 긴 답변을 위해 증가)
    temperature=0.5,  # 온도 조정 (더 낮게 설정하여 안정적인 출력)
    verbose=False,  # 디버그 정보 출력
    n_gpu_layers=32,  # GPU 사용
    n_batch=16,  # 배치 크기 설정
    repeat_penalty=1.1  # 반복 패널티 추가
)

# 데이터베이스 연결 설정
connection_string = f"postgresql://{PG_USER}:{PG_PASSWORD}@{PG_HOST}:{PG_PORT}/{PG_DATABASE}"

# 벡터 스토어 연결 및 초기화
try:
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/xlm-r-100langs-bert-base-nli-stsb-mean-tokens",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    print("✅ 임베딩 모델 로드 완료.")
except Exception as e:
    print(f"❌ 임베딩 모델 로드 실패: {e}")
    sys.exit()

try:
    vector_store = PGVector(
        embeddings=embeddings,
        collection_name=COLLECTION_NAME,
        connection=f"postgresql+psycopg://{PG_USER}:{PG_PASSWORD}@{PG_HOST}:{PG_PORT}/{PG_DATABASE}",
        use_jsonb=True,
        create_extension=True
    )
    print("✅ PostgreSQL 벡터 스토어 연결 완료.")
except Exception as e:
    print(f"❌ PostgreSQL 벡터 스토어 연결 실패: {e}")
    sys.exit()

# 테스트 데이터 준비
from langchain_core.documents import Document

# 데이터베이스에서 PDF 데이터를 가져옵니다.
retriever = vector_store.as_retriever(search_kwargs={"k": 5})

# 테스트 프롬프트
prompt = "농지 소유권을 양도할 때 필요한 절차와 주의사항은 무엇인가요? 농지관리법에서 정한 농사 짓기 의무는 무엇인가요?"

# 데이터베이스에서 관련 문서 검색
print("\n=== 관련 문서 검색 ===")
prompt = "농지 소유권을 양도할 때 필요한 절차와 주의사항은 무엇인가요? 농지관리법에서 정한 농사 짓기 관련 의무는 무엇인가요?"

# 관련 문서 검색
docs = retriever.get_relevant_documents(prompt)

# 검색된 문서를 프롬프트에 추가
context = "\n".join([doc.page_content for doc in docs])
prompt_with_context = """질문: {question}

관련 사례:
{context}

이 사례들을 바탕으로 농지 소유권 양도와 농사 짓기 의무에 대해 설명해주세요.""".format(question=prompt, context=context)

# 모델 추론 실행
response = llm(prompt_with_context)

print("\n=== 모델 출력 ===")
print("응답:", response['choices'][0]['text'].strip())

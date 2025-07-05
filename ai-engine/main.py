import os
import sys
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

# LangChain 및 관련 라이브러리 임포트
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import LlamaCpp
from langchain_postgres.vectorstores import PGVector
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
from langchain.schema.output_parser import StrOutputParser

# .env 파일에서 환경 변수를 로드합니다.
# 이 스크립트는 ai-engine 폴더에 있으므로, 상위 폴더의 .env 파일을 참조합니다.
load_dotenv(dotenv_path="../.env")

# --- 1. 설정 및 전역 변수 ---
# 복잡한 설정 클래스 대신, os.getenv로 직접 설정을 읽어옵니다.
LLM_MODEL_PATH = os.getenv("LLM_MODEL_PATH")
LLM_N_GPU_LAYERS = int(os.getenv("LLM_N_GPU_LAYERS", "0"))
PG_CONNECTION_STRING = f"postgresql+psycopg://{os.getenv('PG_USER')}:{os.getenv('PG_PASSWORD')}@{os.getenv('PG_HOST')}:{os.getenv('PG_PORT')}/{os.getenv('PG_DATABASE')}"
COLLECTION_NAME = os.getenv("COLLECTION_NAME")

# --- 2. FastAPI 앱 및 모델 로딩 ---
# lifespan 대신, 서버 시작 시 모든 리소스를 직접 로드합니다.
app = FastAPI(
    title="지능형 농지민원 답변 API (안정화 버전)",
    description="가장 안정적이고 단순한 RAG 파이프라인을 사용하는 최종 버전입니다.",
    version="8.0.0",
)

print("AI 엔진 초기화를 시작합니다...")

# 임베딩 모델 로드
try:
    embeddings = HuggingFaceEmbeddings(
        model_name=os.getenv("EMBEDDING_MODEL", "jhgan/ko-sbert-nli"),
        model_kwargs={'device': os.getenv("EMBEDDING_DEVICE", "cpu")},
        encode_kwargs={'normalize_embeddings': True}
    )
    print("✅ 임베딩 모델 로드 완료.")
except Exception as e:
    print(f"❌ 임베딩 모델 로드 실패: {e}")
    sys.exit()

# 벡터 스토어 연결
try:
    vector_store = PGVector(
        embeddings=embeddings,
        collection_name=COLLECTION_NAME,
        connection=PG_CONNECTION_STRING,
        use_jsonb=True,
        create_extension=False
    )
    print("✅ PostgreSQL 벡터 스토어 연결 완료.")
except Exception as e:
    print(f"❌ PostgreSQL 벡터 스토어 연결 실패: {e}")
    sys.exit()

# LLM 로드
try:
    # *** 핵심 수정: .env 파일에서 추가 파라미터를 읽어와 LlamaCpp 모델에 적용 ***
    llm = LlamaCpp(
        model_path=LLM_MODEL_PATH,
        n_gpu_layers=LLM_N_GPU_LAYERS,
        n_ctx=int(os.getenv("LLM_N_CTX", 4096)),
        max_tokens=int(os.getenv("LLM_MAX_TOKENS", 1024)),
        temperature=float(os.getenv("LLM_TEMPERATURE", 0.3)),
        n_threads=int(os.getenv("LLM_N_THREADS", 4)),
        n_batch=int(os.getenv("LLM_N_BATCH", 16)),
        repeat_penalty=float(os.getenv("LLM_REPEAT_PENALTY", 1.1)),
        verbose=False,
        stop=["<|eot_id|>", "[사용자 질문]", "[전문가 답변]", "질문:", "답변:"]
    )
    print("✅ LlamaCpp 모델 로드 완료.")
except Exception as e:
    print(f"❌ LlamaCpp 모델 로드 실패: {e}")
    sys.exit()


# --- 3. RAG 체인 구성 ---
retriever = vector_store.as_retriever(search_kwargs={"k": 4}) # 참고 자료를 4개로 늘려 정확도 향상

template = """
당신은 대한민국 농지법을 담당하는 전문 공무원입니다. 사용자의 질문에 대해, 아래에 주어진 '유사 민원 사례와 공식 답변'을 바탕으로 가장 정확하고 전문적인 답변을 생성해야 합니다.
주어진 사례에서 답을 찾을 수 없다면, "관련된 사례를 찾을 수 없어 정확한 답변이 어렵습니다." 라고 솔직하게 답변하세요.
절대 질문을 반복하거나, 불필요한 인사말, 서론, 결론을 덧붙이지 마세요. 즉시 답변의 핵심 내용으로 시작하세요.

[유사 민원 사례와 공식 답변]
{context}

[사용자 질문]
{question}

[전문가 답변]
"""
prompt = ChatPromptTemplate.from_template(template)

def format_docs(docs: List[Document]) -> str:
    """검색된 Document 객체들을 프롬프트에 넣기 좋은 형태로 변환합니다."""
    return "\n\n".join(
        f"--- 사례 {i+1} ---\n사례 질문: {doc.page_content}\n공식 답변: {doc.metadata.get('answer', '답변 정보 없음')}"
        for i, doc in enumerate(docs)
    )

# 가장 단순하고 직관적인 RAG 체인 구조
rag_chain = (
    {"context": retriever | RunnableLambda(format_docs), "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

print("✅ AI 엔진 초기화 완료. API 서버가 준비되었습니다.")


# --- 4. API 엔드포인트 정의 ---
class Question(BaseModel):
    query: str

@app.post("/ask", summary="질문/답변 생성")
def ask_question(question: Question):
    """사용자의 질문을 받아 RAG 체인을 통해 답변을 생성합니다."""
    print(f"수신된 질문: {question.query}")
    try:
        # 단순하고 안정적인 동기(sync) 방식으로 체인을 실행합니다.
        answer = rag_chain.invoke(question.query)
        print(f"생성된 답변: {answer}")
        return {"answer": answer.strip()}
    except Exception as e:
        print(f"오류 발생: {e}")
        return {"error": "답변을 생성하는 중 오류가 발생했습니다."}

@app.get("/")
def read_root():
    return {"message": "지능형 농지민원 답변 API 서버가 정상적으로 실행 중입니다."}

import os
import sys
import re
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings
from typing import List, Dict
import torch # GPU 체크를 위해 torch 임포트

# LangChain 및 관련 라이브러리 임포트
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import LlamaCpp
from langchain_postgres.vectorstores import PGVector
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda, RunnableConfig
from langchain.schema.output_parser import StrOutputParser

# --- 1. 설정 중앙화 (Pydantic BaseSettings 사용) ---
class Settings(BaseSettings):
    # Embedding Model
    EMBEDDING_MODEL: str
    EMBEDDING_DEVICE: str
    
    # LLM Model
    LLM_MODEL_PATH: str
    LLM_N_GPU_LAYERS: int
    LLM_N_CTX: int
    LLM_MAX_TOKENS: int
    LLM_TEMPERATURE: float

    # VectorDB (PostgreSQL)
    PG_DRIVER: str
    PG_HOST: str
    PG_PORT: int
    PG_DATABASE: str
    PG_USER: str
    PG_PASSWORD: str
    COLLECTION_NAME: str

    class Config:
        env_file = "../.env"
        env_file_encoding = "utf-8"

try:
    settings = Settings()
except Exception as e:
    print(f"❌ 설정 파일(.env) 로딩 중 오류 발생: {e}")
    print("   - 프로젝트 최상위 폴더에 '.env' 파일이 있는지, 모든 설정값이 포함되어 있는지 확인해주세요.")
    sys.exit()


# --- 2. 애플리케이션 생명주기(Lifespan) 관리 ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("AI 엔진 초기화를 시작합니다...")
    app.state.settings = settings
    
    # 임베딩 모델 로드
    app.state.embeddings = HuggingFaceEmbeddings(
        model_name=settings.EMBEDDING_MODEL,
        model_kwargs={'device': settings.EMBEDDING_DEVICE},
        encode_kwargs={'normalize_embeddings': True}
    )
    print("✅ 임베딩 모델 로드 완료.")

    # 벡터 스토어 연결
    connection_string = PGVector.connection_string_from_db_params(
        driver=settings.PG_DRIVER,
        host=settings.PG_HOST, port=settings.PG_PORT,
        database=settings.PG_DATABASE, user=settings.PG_USER, password=settings.PG_PASSWORD
    )
    app.state.vector_store = PGVector(
        embeddings=app.state.embeddings,
        collection_name=settings.COLLECTION_NAME,
        connection=connection_string,
    )
    print("✅ PostgreSQL 벡터 스토어 연결 완료.")

    # LLM 로드
    app.state.llm = LlamaCpp(
        model_path=settings.LLM_MODEL_PATH,
        n_gpu_layers=settings.LLM_N_GPU_LAYERS,
        n_ctx=settings.LLM_N_CTX,
        max_tokens=settings.LLM_MAX_TOKENS,
        temperature=settings.LLM_TEMPERATURE,
        verbose=False,
        # *** 핵심 수정: 모델이 불필요한 생성을 멈추도록 중단 단어 설정 ***
        stop=["<|eot_id|>", "[사용자 답변]", "[전문가 답변]"] 
    )
    print("✅ LlamaCpp 모델 로드 완료.")

    # --- GPU 사용 여부 확인 로직 (다시 추가) ---
    print("\n--- GPU 사용 상태 확인 ---")
    if settings.LLM_N_GPU_LAYERS > 0:
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            print(f"✅ GPU 활성화 확인: {gpu_count}개의 CUDA 장치가 감지되었습니다.")
            print(f"   - 현재 모델은 '{settings.LLM_N_GPU_LAYERS}'개의 레이어를 GPU에 오프로딩하여 실행됩니다.")
        else:
            print("⚠️ GPU 사용 불가: PyTorch에서 CUDA 장치를 찾을 수 없습니다.")
            print("   - n_gpu_layers 설정이 적용되지 않으며, 모델이 CPU로만 실행됩니다.")
    else:
        print("ℹ️ GPU 비활성화 상태: n_gpu_layers가 0으로 설정되어 CPU로만 실행됩니다.")
    print("------------------------\n")
    # --- 추가된 부분 끝 ---
    
    # RAG 체인 구성
    retriever = app.state.vector_store.as_retriever(search_kwargs={"k": 3})
    template = """
    당신은 대한민국 농지법 전문 공무원입니다. 사용자의 질문에 대해, 아래에 주어진 '유사 민원 사례와 공식 답변'을 바탕으로 가장 정확하고 전문적인 답변을 생성해야 합니다.
    반드시 주어진 공식 답변의 내용과 법적 근거를 활용하여 답변을 구성하고, 원본 사례의 어조를 참고하여 친절하게 설명하세요.
    단, 불필요한 안내문, 날짜, 이메일, 유효기간 등은 답변에 포함하지 마세요.
    주어진 사례에서 답을 찾을 수 없다면, "관련된 사례를 찾을 수 없어 정확한 답변이 어렵습니다." 라고 솔직하게 답변하세요.

    [유사 민원 사례와 공식 답변]
    {context}

    [사용자 질문]
    {question}

    [전문가 답변]
    """
    prompt = ChatPromptTemplate.from_template(template)

    def format_docs(docs: List[Document]) -> str:
        return "\n\n".join(
            f"--- 사례 {i+1} ---\n사례 질문: {doc.page_content}\n공식 답변: {doc.metadata.get('answer', '답변 정보 없음')}"
            for i, doc in enumerate(docs)
        )

    app.state.rag_chain = (
        {"context": retriever | RunnableLambda(format_docs), "question": RunnablePassthrough()}
        | prompt
        | app.state.llm
        | StrOutputParser()
    )
    print("✅ RAG 체인 구성 완료. API 서버가 준비되었습니다.")
    
    yield
    
    print("AI 엔진을 종료합니다...")

# --- 3. FastAPI 앱 및 엔드포인트 정의 ---
app = FastAPI(
    title="지능형 농지민원 답변 API (전문가용)",
    description="Lifespan과 중앙화된 설정을 사용하여 운영 환경에 최적화된 RAG API입니다.",
    version="4.0.0",
    lifespan=lifespan
)

class Question(BaseModel):
    query: str

@app.post("/ask", summary="질문/답변 생성")
def ask_question(request: Request, question: Question):
    """사용자의 질문을 받아 RAG 체인을 통해 답변을 생성합니다."""
    rag_chain = request.app.state.rag_chain
    
    print(f"수신된 질문: {question.query}")
    try:
        answer = rag_chain.invoke(question.query)
        print(f"생성된 답변: {answer}")
        return {"answer": answer.strip()}
    except Exception as e:
        print(f"오류 발생: {e}")
        return {"error": "답변을 생성하는 중 오류가 발생했습니다."}

@app.get("/")
def read_root():
    return {"message": "지능형 농지민원 답변 API 서버가 정상적으로 실행 중입니다."}

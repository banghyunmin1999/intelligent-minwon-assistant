import os
from dotenv import load_dotenv
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field
from typing import List, Dict, Any
import traceback
import time

from langchain_postgres.vectorstores import PGVector
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import EmbeddingsFilter

from llama_cpp import Llama
import torch
from contextlib import asynccontextmanager
from sqlalchemy.ext.asyncio import create_async_engine
from fastapi.middleware.cors import CORSMiddleware

# --- 1. .env 파일 로드 ---
dotenv_path = os.path.join(os.path.dirname(__file__), '.env')
if os.path.exists(dotenv_path):
    load_dotenv(dotenv_path=dotenv_path)

# --- 2. 설정 값 읽기 및 자동 계산 ---
LLM_N_THREADS = int(os.getenv("LLM_N_THREADS", 4)) # 기본값 4
EMBEDDING_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = os.getenv("MODEL_PATH")
LLM_N_GPU_LAYERS = int(os.getenv("LLM_N_GPU_LAYERS", -1))
LLM_N_CTX = int(os.getenv("LLM_N_CTX", 2048))
LLM_N_BATCH = int(os.getenv("LLM_N_BATCH", 1))
LLM_REPEAT_PENALTY = float(os.getenv("LLM_REPEAT_PENALTY", 1.1))
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", 0.2))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", 512))
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "jhgan/ko-sroberta-multitask")
PG_HOST = os.getenv("DB_HOST")
PG_PORT = os.getenv("DB_PORT")
PG_DATABASE = os.getenv("DB_NAME")
PG_USER = os.getenv("DB_USER")
PG_PASSWORD = os.getenv("DB_PASSWORD")
COLLECTION_NAME = "minwon_pdf_cases_v3"
CONNECTION_STRING = f"postgresql+asyncpg://{PG_USER}:{PG_PASSWORD}@{PG_HOST}:{PG_PORT}/{PG_DATABASE}"

async def do_nothing(*args, **kwargs):
    pass
PGVector.acreate_vector_extension = do_nothing

# --- 전역 변수 선언 ---
llm = None
# retriever = None # RAG를 사용하지 않으므로 주석 처리

# --- lifespan 이벤트 핸들러 ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    global llm #, retriever
    
    llm = Llama(
        model_path=MODEL_PATH,
        n_ctx=LLM_N_CTX,
        n_threads=LLM_N_THREADS,
        n_gpu_layers=LLM_N_GPU_LAYERS,
        n_batch=LLM_N_BATCH,
        verbose=False,
        temperature=LLM_TEMPERATURE,
        repeat_penalty=LLM_REPEAT_PENALTY
    )
    
    # RAG를 사용하지 않으므로 DB 관련 초기화는 불필요
    # embeddings = HuggingFaceEmbeddings(...)
    # engine = create_async_engine(CONNECTION_STRING)
    # vector_store = await PGVector.afrom_existing_index(...)
    # base_retriever = vector_store.as_retriever(...)
    # retriever = ContextualCompressionRetriever(...)

    print("\n>> AI 모델 워밍업을 시작합니다...", flush=True)
    try:
        _ = llm("warmup", max_tokens=2)
        print("✅ AI 모델 워밍업 완료.", flush=True)
    except Exception as e:
        print(f"⚠️ 워밍업 중 오류 발생: {e}", flush=True)
    
    startup_logs = []
    startup_logs.append("-------------------- SERVER STATUS (LLM Only Test Mode) --------------------")
    startup_logs.append(f"✅ LLM 모델: {os.path.basename(MODEL_PATH) if MODEL_PATH else 'N/A'}")
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        startup_logs.append(f"✅ GPU 상태: CUDA 사용 가능 ({gpu_name})")
        startup_logs.append(f"✅ LLM 오프로드: {LLM_N_GPU_LAYERS}개 레이어")
    else:
        startup_logs.append("⚠️ GPU 상태: CUDA 사용 불가")
    startup_logs.append(f"✅ LLM 스레드 수: {LLM_N_THREADS}개")
    startup_logs.append(f"✅ LLM 컨텍스트 크기: {LLM_N_CTX}")
    startup_logs.append("-----------------------------------------------------------------------")

    for log in startup_logs:
        print(log, flush=True)
    
    print("\n✅ 모든 리소스 로드 완료. 서버가 요청을 받을 준비가 되었습니다.", flush=True)
    yield
    print("\n✅ 서버를 종료합니다.", flush=True)

# --- FastAPI 앱 생성 ---
app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

# --- API 모델 정의 ---
class QuestionRequest(BaseModel):
    question: str

class SourceDocument(BaseModel):
    page_content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)

class AnswerResponse(BaseModel):
    question: str
    answer: str
    sources: List[SourceDocument] = []

# --- API 엔드포인트 ---
@app.post("/ask", response_model=AnswerResponse)
async def ask_question(request: QuestionRequest):
    user_question = request.question
    print(f"\n[입력 질문]: {user_question}", flush=True)

    t_start_total = time.time()
    performance_logs = []

    try:
        # [수정] RAG 비활성화에 맞춰 프롬프트 지시 변경
        answer_generation_prompt = f"""[지시]
당신은 사용자의 '[질문]'에 대해 아는 대로 상세히 답변하는 AI입니다.

[질문]
{user_question}

[답변]
"""
        # 답변 생성(Inference) 시간 측정
        t_start_inference = time.time()
        response = await run_in_threadpool(
            llm, 
            answer_generation_prompt, 
            max_tokens=LLM_MAX_TOKENS, 
            stop=["[지시]", "[정보]", "[질문]"]
        )
        t_end_inference = time.time()
        performance_logs.append(f"  - 답변 생성(LLM Inference): {t_end_inference - t_start_inference:.4f} 초")
        
        raw_answer = response['choices'][0]['text'].strip()
        
        # 후처리 로직
        clean_answer = raw_answer.replace("[답변]", "").strip()
        last_period_index = clean_answer.rfind('다.')
        if last_period_index != -1:
            answer = clean_answer[:last_period_index + 2]
        else:
            answer = clean_answer
        
        # 최종 성능 로그 출력
        t_end_total = time.time()
        performance_logs.append(f"  - 총 처리 시간: {t_end_total - t_start_total:.4f} 초")
        print("\n[성능 측정 결과 (LLM Only)]", flush=True)
        for log in performance_logs:
            print(log, flush=True)
        print("---------------", flush=True)

        print(f"[생성된 답변]: {answer[:150]}...", flush=True)

        return AnswerResponse(question=user_question, answer=answer, sources=[])

    except Exception as e:
        error_details = traceback.format_exc()
        print(f"!!!!!!!!!!!!! API 처리 중 심각한 오류 발생 !!!!!!!!!!!!!", flush=True)
        print(error_details, flush=True)
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}\n{error_details}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
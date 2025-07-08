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
LLM_N_THREADS = int(os.getenv("LLM_N_THREADS", 4))
EMBEDDING_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = os.getenv("MODEL_PATH")
LLM_N_GPU_LAYERS = int(os.getenv("LLM_N_GPU_LAYERS", -1))
LLM_N_CTX = int(os.getenv("LLM_N_CTX", 2048))
LLM_N_BATCH = int(os.getenv("LLM_N_BATCH", 1))
LLM_REPEAT_PENALTY = float(os.getenv("LLM_REPEAT_PENALTY", 1.1))
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", 0.2))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", 256))
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
retriever = None

# --- lifespan 이벤트 핸들러 ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    global llm, retriever
    
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
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={'device': EMBEDDING_DEVICE},
        encode_kwargs={'normalize_embeddings': True}
    )
    engine = create_async_engine(CONNECTION_STRING)
    vector_store = await PGVector.afrom_existing_index(
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        connection=engine,
        use_jsonb=True,
    )
    
    base_retriever = vector_store.as_retriever(search_kwargs={"k": 5})
    embeddings_filter = EmbeddingsFilter(embeddings=embeddings, similarity_threshold=0.75)
    retriever = ContextualCompressionRetriever(
        base_compressor=embeddings_filter, base_retriever=base_retriever
    )

    print("\n>> AI 모델 워밍업을 시작합니다...", flush=True)
    try:
        _ = llm("warmup", max_tokens=2)
        print("✅ AI 모델 워밍업 완료.", flush=True)
    except Exception as e:
        print(f"⚠️ 워밍업 중 오류 발생: {e}", flush=True)
    
    startup_logs = []
    startup_logs.append("-------------------- SERVER STATUS --------------------")
    startup_logs.append(f"✅ LLM 모델: {os.path.basename(MODEL_PATH) if MODEL_PATH else 'N/A'}")
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        startup_logs.append(f"✅ GPU 상태: CUDA 사용 가능 ({gpu_name})")
        startup_logs.append(f"✅ LLM 오프로드: {LLM_N_GPU_LAYERS}개 레이어")
    else:
        startup_logs.append("⚠️ GPU 상태: CUDA 사용 불가")
    startup_logs.append(f"✅ LLM 스레드 수: {LLM_N_THREADS}개")
    startup_logs.append(f"✅ LLM 컨텍스트 크기: {LLM_N_CTX}")
    startup_logs.append(f"✅ 임베딩 모델: {EMBEDDING_MODEL}")
    startup_logs.append(f"✅ VectorDB 컬렉션: {COLLECTION_NAME}")
    startup_logs.append("-----------------------------------------------------")

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
        civil_complaint_keywords = [
            "법", "규정", "절차", "신고", "허가", "과태료", "문의", "소유권", "양도", 
            "상속", "취득", "요건", "자격", "제한", "대상", "처분", "의무", "설치",
            "농지", "토지", "지목", "임야", "농업", "영농", "농막", "농업인", "재배",
            "경우", "무엇인가요", "어떻게", "가능", "불가능"
        ]
        if not any(keyword in user_question for keyword in civil_complaint_keywords):
            answer = "저는 법률 및 민원 관련 질문에만 답변할 수 있습니다."
            print(f"[판단] 민원 관련 질문이 아님. 기본 응답 반환.", flush=True)
            return AnswerResponse(question=user_question, answer=answer, sources=[])

        if retriever is None:
            return AnswerResponse(question=user_question, answer="오류: 서버가 아직 준비되지 않았습니다.", sources=[])

        t_start_retrieval = time.time()
        docs = await retriever.ainvoke(user_question)
        t_end_retrieval = time.time()
        performance_logs.append(f"  - 문서 검색(Retrieval): {t_end_retrieval - t_start_retrieval:.4f} 초")

        if not docs:
            answer = "관련된 정보를 찾을 수 없습니다. 좀 더 구체적으로 질문해주시겠어요?"
            print("[결과] 벡터DB에서 관련된 문서를 찾지 못함.", flush=True)
            return AnswerResponse(question=user_question, answer=answer, sources=[])

        print("\n[필터링 후 검색된 문서]", flush=True)
        for i, doc in enumerate(docs, 1):
            print(f"--- 문서 {i} ---", flush=True)
            print(doc.page_content, flush=True)
            if hasattr(doc, "metadata"):
                print(f"[metadata]: {doc.metadata}", flush=True)
            print("---------------", flush=True)

        context_parts = []
        for doc in docs:
            content = doc.page_content
            if "답변:" in content:
                context_parts.append(content.split("답변:", 1)[1].strip())
            else:
                context_parts.append(content)
        context = "\n---\n".join(context_parts)
        
        answer_generation_prompt = f"""[지시]
당신은 법률 AI입니다. '[정보]'에 있는 내용을 바탕으로, 사용자의 '[질문]'에 대한 답변을 3~5 문장으로 요약해서 설명해주세요.
'[정보]'에 없는 내용은 절대로 언급하지 마세요.

[정보]
{context}

[질문]
{user_question}

[답변]
"""
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
        
        cannot_answer_phrase = "제공된 정보로는 답변을 찾을 수 없습니다."
        if cannot_answer_phrase in raw_answer:
            answer = cannot_answer_phrase
        else:
            # --- [수정] '다.'로 자르는 로직 비활성화 ---
            # '---' 패턴이나 '[답변]' 마커만 제거하고, 모델의 원본 출력을 최대한 그대로 둡니다.
            main_answer_part = raw_answer.split('---')[0].strip()
            answer = main_answer_part.replace("[답변]", "").strip()
            # --- [수정 끝] ---
        
        t_end_total = time.time()
        performance_logs.append(f"  - 총 처리 시간: {t_end_total - t_start_total:.4f} 초")
        print("\n[성능 측정 결과]", flush=True)
        for log in performance_logs:
            print(log, flush=True)
        print("---------------", flush=True)

        print(f"[생성된 답변]: {answer[:250]}...", flush=True) # 로그 길이를 조금 늘려서 확인

        source_documents = [SourceDocument(page_content=doc.page_content, metadata=doc.metadata) for doc in docs]
        return AnswerResponse(question=user_question, answer=answer, sources=source_documents)

    except Exception as e:
        error_details = traceback.format_exc()
        print(f"!!!!!!!!!!!!! API 처리 중 심각한 오류 발생 !!!!!!!!!!!!!", flush=True)
        print(error_details, flush=True)
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}\n{error_details}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
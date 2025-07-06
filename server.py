# server.py (후처리 수정 최종 버전)

import os
from dotenv import load_dotenv
import uvicorn
from fastapi import FastAPI
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field
from typing import List, Dict, Any

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

# --- 2. 설정 값 읽기 ---
PG_HOST = os.getenv("DB_HOST")
PG_PORT = os.getenv("DB_PORT")
PG_DATABASE = os.getenv("DB_NAME")
PG_USER = os.getenv("DB_USER")
PG_PASSWORD = os.getenv("DB_PASSWORD")
COLLECTION_NAME = "minwon_pdf_cases_v2" 
CONNECTION_STRING = f"postgresql+asyncpg://{PG_USER}:{PG_PASSWORD}@{PG_HOST}:{PG_PORT}/{PG_DATABASE}"
model_path = os.getenv("MODEL_PATH")
N_GPU_LAYERS = int(os.getenv("N_GPU_LAYERS", 0))

async def do_nothing(*args, **kwargs):
    pass
PGVector.acreate_vector_extension = do_nothing

# --- 4. 전역 변수 선언 ---
llm = None
retriever = None

# --- 5. lifespan 이벤트 핸들러 ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    global llm, retriever

    startup_logs = []
    
    llm = Llama(
        model_path=model_path, n_ctx=4096, n_threads=4, n_gpu_layers=N_GPU_LAYERS,
        n_batch=16, verbose=False, temperature=0.2, repeat_penalty=1.2
    )
    embeddings = HuggingFaceEmbeddings(
        model_name="jhgan/ko-sroberta-multitask",
        model_kwargs={'device': 'cpu'}, encode_kwargs={'normalize_embeddings': True}
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
    
    startup_logs.append("-------------------- SERVER STATUS --------------------")
    if os.path.exists(dotenv_path):
        startup_logs.append(f"✅ .env 파일 로드: '{dotenv_path}'")
    else:
        startup_logs.append(f"⚠️ .env 파일 없음: '{dotenv_path}'")
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        startup_logs.append(f"✅ GPU 상태: CUDA 사용 가능 ({gpu_name})")
        if N_GPU_LAYERS != 0:
            startup_logs.append(f"✅ LLM 오프로드: {N_GPU_LAYERS}개 레이어 GPU에 설정")
        else:
            startup_logs.append("ℹ️ LLM 오프로드: 설정되지 않음 (CPU로만 동작)")
    else:
        startup_logs.append("⚠️ GPU 상태: CUDA 사용 불가")
    startup_logs.append(f"✅ VectorDB 연결: postgresql+asyncpg://.../{PG_DATABASE} ({COLLECTION_NAME})")
    startup_logs.append("-----------------------------------------------------")

    for log in startup_logs:
        print(log)
    
    print("\n✅ 모든 리소스 로드 완료. 서버가 요청을 받을 준비가 되었습니다.")
    yield
    print("\n✅ 서버를 종료합니다.")

# --- 6. FastAPI 앱 생성 ---
app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

# --- 7. API 모델 정의 ---
class QuestionRequest(BaseModel):
    question: str

class SourceDocument(BaseModel):
    page_content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)

class AnswerResponse(BaseModel):
    question: str
    answer: str
    sources: List[SourceDocument] = []

# --- 8. API 엔드포인트 ---
# server.py의 ask_question 함수

@app.post("/ask", response_model=AnswerResponse)
async def ask_question(request: QuestionRequest):
    user_question = request.question
    print(f"\n[입력 질문]: {user_question}")

    # 가드레일: 질문 의도 분류
    civil_complaint_keywords = ["법", "규정", "절차", "신고", "허가", "과태료", "문의", "소유권", "양도", "상속", "취득"]
    if not any(keyword in user_question for keyword in civil_complaint_keywords):
        answer = "저는 법률 및 민원 관련 질문에만 답변할 수 있습니다."
        print(f"[판단] 민원 관련 질문이 아님. 기본 응답 반환.")
        return AnswerResponse(question=user_question, answer=answer, sources=[])

    if retriever is None:
        return AnswerResponse(question=user_question, answer="오류: 서버가 아직 준비되지 않았습니다.", sources=[])

    docs = await retriever.ainvoke(user_question)

    if not docs:
        answer = "관련된 정보를 찾을 수 없습니다. 좀 더 구체적으로 질문해주시겠어요?"
        print("[결과] 벡터DB에서 관련된 문서를 찾지 못함.")
        return AnswerResponse(question=user_question, answer=answer, sources=[])

    print("\n[필터링 후 검색된 문서]")
    for i, doc in enumerate(docs, 1):
        print(f"--- 문서 {i} ---")
        print(doc.page_content)
        if hasattr(doc, "metadata"):
            print(f"[metadata]: {doc.metadata}")
        print("---------------")

    context = "\n---\n".join([doc.page_content for doc in docs])
    
    answer_generation_prompt = f"""[지시]
당신은 오직 '[정보]' 섹션에 제공된 법률 문서 내용만을 사용하여 '[질문]'에 답변하는 법률 AI입니다.
주어진 '[정보]'에 질문에 대한 답변 근거가 전혀 없다면, **"제공된 정보만으로는 답변할 수 없습니다."** 라고만 답변하세요.
절대로 당신의 지식을 사용하거나 정보를 추측해서 답변하면 안 됩니다.
답변은 민원인이 이해하기 쉽게 핵심 내용을 요약하여 3~5문장 이내로 간결하게 작성하세요.

[정보]
{context}

[질문]
{user_question}

[답변]
"""

    response = await run_in_threadpool(
        llm, answer_generation_prompt, max_tokens=512, stop=["[지시]", "[정보]", "[질문]"]
    )
    raw_answer = response['choices'][0]['text'].strip()
    
    # --- [최종 수정] 강력한 후처리 로직 ---
    # 1. 기본적인 [답변] 마커 제거
    clean_answer = raw_answer.replace("[답변]", "").strip()

    # 2. 반복적으로 나타나는 불필요한 상용구나 쓰레기 문자열의 시작점을 기준으로 분리하여, 그 앞부분(정상 답변)만 사용
    stop_phrases = ["제시된 정보만으로는", "](본문)"]
    for phrase in stop_phrases:
        if phrase in clean_answer:
            clean_answer = clean_answer.split(phrase)[0].strip()
    
    answer = clean_answer
    
    print(f"[생성된 답변]: {answer[:150]}...")

    source_documents = [SourceDocument(page_content=doc.page_content, metadata=doc.metadata) for doc in docs]
    return AnswerResponse(question=user_question, answer=answer, sources=source_documents)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
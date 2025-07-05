# server.py (클래스 레벨 몽키 패칭 적용 최종 버전)

import os
from dotenv import load_dotenv
import uvicorn
from fastapi import FastAPI
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel
from llama_cpp import Llama
from langchain_postgres.vectorstores import PGVector
from langchain_huggingface import HuggingFaceEmbeddings
import torch
from contextlib import asynccontextmanager
from sqlalchemy.ext.asyncio import create_async_engine

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
COLLECTION_NAME = "minwon_pdf_cases_v1"
CONNECTION_STRING = f"postgresql+asyncpg://{PG_USER}:{PG_PASSWORD}@{PG_HOST}:{PG_PORT}/{PG_DATABASE}"
model_path = os.getenv("MODEL_PATH")
N_GPU_LAYERS = int(os.getenv("N_GPU_LAYERS", 0))

# --- 3. 몽키 패칭: PGVector 클래스 전체에 적용 ---
async def do_nothing(*args, **kwargs):
    pass
PGVector.acreate_vector_extension = do_nothing

# --- 4. 전역 변수 선언 ---
llm = None
retriever = None
vector_store = None

# --- 5. lifespan 이벤트 핸들러 ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    global llm, retriever, vector_store

    startup_logs = []

    # LLM, Embeddings 모델 로드
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

    retriever = vector_store.as_retriever(search_kwargs={"k": 3})

    # -- 상태 확인 및 로그 메시지 생성 --
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
    startup_logs.append(f"✅ VectorDB 연결: postgresql+asyncpg://.../{PG_DATABASE}")
    startup_logs.append("-----------------------------------------------------")

    for log in startup_logs:
        print(log)

    print("\n✅ 모든 리소스 로드 완료. 서버가 요청을 받을 준비가 되었습니다.")
    yield
    print("\n✅ 서버를 종료합니다.")

# --- 6. FastAPI 앱 생성 및 lifespan 연결 ---
app = FastAPI(lifespan=lifespan)

# --- 7. API 엔드포인트 ---
class QuestionRequest(BaseModel):
    question: str
class AnswerResponse(BaseModel):
    question: str
    answer: str

@app.post("/ask", response_model=AnswerResponse)
async def ask_question(request: QuestionRequest):
    user_question = request.question
    print(f"\n[입력 질문]: {user_question}")

    if retriever is None:
        return {"error": "서버가 아직 준비 중입니다. 잠시 후 다시 시도해주세요."}

    docs = await retriever.ainvoke(user_question)
    context = "\n---\n".join([doc.page_content for doc in docs])

    # 프롬프트 개선
    answer_generation_prompt = f"""[지시]
당신은 `[정보]`를 바탕으로 `[질문]`에 대해 민원인이 이해하기 쉽도록
핵심만 간결하게(3~5문장 이내) 요약해 답변하는 AI입니다.
불필요한 서론, 반복, 부연설명 없이 바로 답변만 출력하세요.

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
    # 후처리(불필요한 마커 제거)
    for marker in ["[정보]", "[질문]", "[답변]"]:
        if marker in raw_answer:
            raw_answer = raw_answer.split(marker)[-1].strip()
    answer = raw_answer

    print(f"[생성된 답변]: {answer[:150]}...")

    return AnswerResponse(question=user_question, answer=answer)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

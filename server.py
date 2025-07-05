# server.py (최종 .env 적용 버전)

import os
from dotenv import load_dotenv
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from llama_cpp import Llama
from langchain_postgres.vectorstores import PGVector
from langchain_huggingface import HuggingFaceEmbeddings

# --- 1. .env 파일 로드 (가장 먼저 실행) ---
# 스크립트 시작 시 .env 파일의 변수들을 환경 변수로 로드합니다.
load_dotenv()

# --- 2. 설정 값들을 환경 변수에서 읽어오기 ---
PG_HOST = os.getenv("DB_HOST")
PG_PORT = os.getenv("DB_PORT")
PG_DATABASE = os.getenv("DB_NAME")
PG_USER = os.getenv("DB_USER")
PG_PASSWORD = os.getenv("DB_PASSWORD")
COLLECTION_NAME = "minwon_pdf_cases_v1"
CONNECTION_STRING = f"postgresql+psycopg2://{PG_USER}:{PG_PASSWORD}@{PG_HOST}:{PG_PORT}/{PG_DATABASE}"
model_path = os.getenv("MODEL_PATH")

# --- 3. 모델 및 리소스 로드 (이하 로직은 동일) ---
print("✅ 서버 시작... 리소스를 로드합니다.")
llm = Llama(
    model_path=model_path, n_ctx=4096, n_threads=4, n_gpu_layers=32,
    n_batch=16, verbose=False, temperature=0.2, repeat_penalty=1.2
)
embeddings = HuggingFaceEmbeddings(
    model_name="jhgan/ko-sroberta-multitask",
    model_kwargs={'device': 'cpu'}, encode_kwargs={'normalize_embeddings': True}
)
vector_store = PGVector(
    embeddings=embeddings, collection_name=COLLECTION_NAME, connection=CONNECTION_STRING,
)
retriever = vector_store.as_retriever(search_kwargs={"k": 3})
print("✅ 모든 리소스 로드 완료. 서버가 요청을 받을 준비가 되었습니다.")

# --- 4. FastAPI 앱 및 API 엔드포인트 생성 (이하 로직은 동일) ---
app = FastAPI()

class QuestionRequest(BaseModel):
    question: str

class AnswerResponse(BaseModel):
    question: str
    answer: str

@app.post("/ask", response_model=AnswerResponse)
def ask_question(request: QuestionRequest):
    user_question = request.question
    print(f"\n[입력 질문]: {user_question}")
    
    docs = retriever.invoke(user_question)
    context = "\n---\n".join([doc.page_content for doc in docs])

    answer_generation_prompt = f"""[지시]
당신은 `[정보]`를 분석하여 `[질문]`에 답변하는 AI입니다.
답변은 반드시 `[정보]`에 명시된 사실에만 근거해야 합니다.
서론이나 부연 설명 없이, 질문에 대한 답변의 핵심 내용부터 바로 시작하세요.

[정보]
{context}

[질문]
{user_question}

[답변]
"""
    response = llm(answer_generation_prompt, max_tokens=1024, stop=["[지시]", "[정보]", "[질문]"])
    answer = response['choices'][0]['text'].strip()
    
    print(f"[생성된 답변]: {answer[:150]}...")
    
    return AnswerResponse(question=user_question, answer=answer)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
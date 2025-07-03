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
load_dotenv()

# --- 1. 전역 설정 및 모델/DB 연결 ---
print("AI 엔진 초기화를 시작합니다...")

# 임베딩 모델 로드
try:
    embeddings = HuggingFaceEmbeddings(
        model_name="jhgan/ko-sbert-nli",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
except Exception as e:
    print(f"임베딩 모델 로드 중 오류 발생: {e}")
    sys.exit()

# PostgreSQL 연결 정보
try:
    PG_CONNECTION_STRING = PGVector.connection_string_from_db_params(
        driver="psycopg",
        host=os.getenv("PG_HOST", "localhost"),
        port=int(os.getenv("PG_PORT", "5432")),
        database=os.getenv("PG_DB_NAME", "minwon_pg_db"),
        user=os.getenv("PG_USER", "minwon_user"),
        password=os.getenv("PG_PASSWORD", "1234"),
    )
except Exception as e:
    print(f"DB 연결 문자열 생성 중 오류 발생: {e}")
    sys.exit()


# PGVector 스토어 초기화
# *** 핵심 수정: 구버전과 호환되는 직접 생성자 방식으로 변경 ***
try:
    vector_store = PGVector(
        connection_string=PG_CONNECTION_STRING,
        embedding_function=embeddings, # 구버전 파라미터 이름 사용
        collection_name="minwon_qna_cases",
    )
    print("✅ PostgreSQL Q&A 벡터 스토어에 성공적으로 연결했습니다.")
except Exception as e:
    print(f"❌ PostgreSQL 벡터 스토어 연결 실패: {e}")
    sys.exit()


# Retriever(검색기) 생성
retriever = vector_store.as_retriever(search_kwargs={"k": 3})

# Prompt Template 정의
template = """
당신은 대한민국 농지법 전문 공무원입니다. 사용자의 질문에 대해, 아래에 주어진 '유사 민원 사례와 공식 답변'을 바탕으로 가장 정확하고 전문적인 답변을 생성해야 합니다.
반드시 주어진 공식 답변의 내용과 법적 근거를 활용하여 답변을 구성하고, 원본 사례의 어조를 참고하여 친절하게 설명하세요.
주어진 사례에서 답을 찾을 수 없다면, "관련된 사례를 찾을 수 없어 정확한 답변이 어렵습니다." 라고 솔직하게 답변하세요.

[유사 민원 사례와 공식 답변]
{context}

[사용자 질문]
{question}

[전문가 답변]
"""
prompt = ChatPromptTemplate.from_template(template)

# LlamaCpp LLM 모델 로드
try:
    llm = LlamaCpp(
        model_path="/home/bang/intelligent-minwon-assistant/models/llama-3-Korean-Bllossom-8B-gguf-Q4_K_M/llama-3-Korean-Bllossom-8B-Q4_K_M.gguf",
        n_gpu_layers=40,
        n_batch=2048,
        n_ctx=4096,
        max_tokens=1024,
        temperature=0.7,
        verbose=True,
    )
except Exception as e:
    print(f"LlamaCpp 모델 로드 중 오류 발생: {e}")
    exit()

# --- 2. RAG 체인 업그레이드 ---

def format_docs(docs: List[Document]) -> str:
    """검색된 Document 객체들을 프롬프트에 넣기 좋은 형태로 변환합니다."""
    formatted_strings = []
    for i, doc in enumerate(docs):
        question_content = doc.page_content
        answer_content = doc.metadata.get('answer', '답변 정보 없음')
        
        formatted_strings.append(f"--- 사례 {i+1} ---\n사례 질문: {question_content}\n공식 답변: {answer_content}")
    return "\n\n".join(formatted_strings)

rag_chain = (
    {"context": retriever | RunnableLambda(format_docs), "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

print("✅ AI 엔진 초기화 완료. API 서버가 준비되었습니다.")

# --- 3. FastAPI 앱 설정 ---
app = FastAPI(
    title="지능형 농지민원 답변 API (사례집 기반)",
    description="농지민원 사례집 PDF의 지식을 기반으로 답변하는 RAG API입니다.",
    version="3.0.0",
)

class Question(BaseModel):
    query: str

@app.post("/ask", summary="질문/답변 생성")
async def ask_question(question: Question):
    print(f"수신된 질문: {question.query}")
    try:
        answer = rag_chain.invoke(question.query)
        print(f"생성된 답변: {answer}")
        return {"answer": answer}
    except Exception as e:
        print(f"오류 발생: {e}")
        return {"error": "답변을 생성하는 중 오류가 발생했습니다."}

@app.get("/", summary="API 상태 확인")
def read_root():
    return {"message": "지능형 농지민원 답변 API 서버가 정상적으로 실행 중입니다."}

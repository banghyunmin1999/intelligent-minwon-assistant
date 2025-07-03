import os
import sys
import re
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

print("AI 엔진 초기화를 시작합니다...")

# 임베딩 모델 로드
try:
    embeddings_model = HuggingFaceEmbeddings(
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

try:
    vector_store = PGVector(
        embeddings=embeddings_model,
        collection_name="minwon_qna_cases",
        connection=PG_CONNECTION_STRING,
    )
    print("✅ PostgreSQL Q&A 벡터 스토어에 성공적으로 연결했습니다.")
except Exception as e:
    print(f"❌ PostgreSQL 벡터 스토어 연결 실패: {e}")
    sys.exit()

retriever = vector_store.as_retriever(search_kwargs={"k": 3})

# 프롬프트 정의 (불필요한 안내문/날짜/이메일/유효기간 등은 포함하지 말라고 명시)
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

# LlamaCpp LLM 모델 로드
try:
    llm = LlamaCpp(
        model_path="/home/bang/intelligent-minwon-assistant/models/llama-3-Korean-Bllossom-8B-gguf-Q4_K_M/llama-3-Korean-Bllossom-8B-Q4_K_M.gguf",
        n_gpu_layers=40,
        n_batch=2048,
        n_ctx=4096,
        max_tokens=768,
        temperature=0.5,
        verbose=True,
    )
except Exception as e:
    print(f"LlamaCpp 모델 로드 중 오류 발생: {e}")
    sys.exit()

def format_docs(docs: List[Document]) -> str:
    formatted_strings = []
    for i, doc in enumerate(docs):
        question_content = doc.page_content
        answer_content = doc.metadata.get('answer', '답변 정보 없음')
        formatted_strings.append(f"--- 사례 {i+1} ---\n사례 질문: {question_content}\n공식 답변: {answer_content}")
    return "\n\n".join(formatted_strings)

def clean_output(text: str) -> str:
    # 반복, 불필요한 줄, 안내문, 날짜/이메일/유효기간 등 제거
    lines = text.splitlines()
    seen = set()
    result = []
    for line in lines:
        line = line.strip()
        # 불필요한 안내문/날짜/이메일/유효기간/참고/발송/끝/유효 등 제거
        if not line or line.startswith("[START]"):
            continue
        if re.search(r"\d{4}년|\d{4}\.\d{2}\.\d{2}|이메일|유효합니다|발송|참고 사항|끝\.|끝|법률 전문가|업데이트|정보는|작성되었습니다", line):
            continue
        if line not in seen:
            result.append(line)
            seen.add(line)
    # 종료 신호에서 자르기
    for stop_token in ["[END]", "### End", "###", "</s>"]:
        if stop_token in "\n".join(result):
            result = "\n".join(result).split(stop_token)[0].splitlines()
    return "\n".join(result).strip()

rag_chain = (
    {"context": retriever | RunnableLambda(format_docs), "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
    | RunnableLambda(clean_output)
)

print("✅ AI 엔진 초기화 완료. API 서버가 준비되었습니다.")

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

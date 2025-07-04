import os
import sys
import re
import logging
import asyncio
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings
from typing import List, Dict, Any

# LangChain 및 관련 라이브러리 임포트
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import LlamaCpp
from langchain_postgres.vectorstores import PGVector
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain.schema.runnable import Runnable, RunnablePassthrough, RunnableLambda, RunnableBranch
from langchain.schema.output_parser import StrOutputParser

# SQLAlchemy 비동기 엔진 임포트
from sqlalchemy.ext.asyncio import create_async_engine

# --------------------------------------------------------------------------
# 1. 설정 관리 (config.py)
# --------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Settings(BaseSettings):
    """애플리케이션 설정을 관리하는 클래스"""
    EMBEDDING_MODEL: str
    EMBEDDING_DEVICE: str
    LLM_MODEL_PATH: str
    LLM_N_GPU_LAYERS: int
    LLM_N_CTX: int
    LLM_MAX_TOKENS: int
    LLM_TEMPERATURE: float
    PG_DRIVER: str = "postgresql+asyncpg"
    PG_HOST: str
    PG_PORT: int
    PG_DATABASE: str
    PG_USER: str
    PG_PASSWORD: str
    COLLECTION_NAME: str

    class Config:
        env_file = "../.env"
        env_file_encoding = "utf-8"

# --------------------------------------------------------------------------
# 2. 서비스 로직 (services.py)
# --------------------------------------------------------------------------
def create_rag_chain(llm: LlamaCpp, retriever: PGVector.as_retriever) -> Runnable:
    """검색, 프롬프트, LLM, 파서를 연결하여 RAG 체인을 생성합니다."""
    
    def format_docs(docs: List[Document]) -> str:
        return "\n\n".join(
            f"--- 사례 {i+1} ---\n사례 질문: {doc.page_content}\n공식 답변: {doc.metadata.get('answer', '답변 정보 없음')}"
            for i, doc in enumerate(docs)
        )

    # 1. 최종 답변 생성 프롬프트 (Llama 3 형식 및 지시사항 강화)
    answer_prompt = ChatPromptTemplate.from_template("""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

당신은 대한민국 농지법 질의응답 데이터베이스입니다. 당신의 유일한 임무는 사용자의 질문에 대해, 주어진 [참고 자료]의 '공식 답변'을 바탕으로 핵심 정보를 요약하여 전달하는 것입니다.

**엄격한 출력 규칙:**
1. 절대 사용자의 질문을 반복하거나 되묻지 마세요.
2. '답변:', '설명:', '네, ...에 대해 질문해 주셨습니다.' 와 같은 서론을 절대 사용하지 마세요.
3. 즉시 답변의 핵심 내용으로 시작하세요.
4. 주어진 [참고 자료]에 없는 내용은 절대 언급하지 마세요.
5. 오직 최종 답변 내용만 출력하세요. 불필요한 태그나 추가 설명은 절대 포함하지 마세요.<|eot_id|><|start_header_id|>user<|end_header_id|>

[참고 자료: 실제 민원 처리 사례]
{context}

[사용자 질문]
{question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
""")

    # 2. 관련성 판단 프롬프트 (개선)
    relevance_prompt = PromptTemplate.from_template("""
    당신은 사용자의 질문이 주어진 컨텍스트와 관련이 있는지 판단하는 AI 필터입니다.
    컨텍스트는 '대한민국 농지법'에 대한 민원 사례들입니다.
    사용자의 질문이 '농지', '농업', '농지법' 등과 명백히 관련이 있다면 'yes'를,
    '점심 메뉴', '날씨', '스포츠'와 같이 전혀 관련 없는 주제라면 'no'를 반환해야 합니다.
    오직 'yes' 또는 'no'로만 대답하세요.
    [컨텍스트]
    {context}
    [사용자 질문]
    {question}
    [관련성 여부 (yes/no)]
    """)

    # 3. 최종 답변 후처리 함수 (강력한 안전장치)
    def clean_response(text: str) -> str:
        """LLM의 출력에서 불필요한 반복, 태그, 자기 대화 등을 정리합니다."""
        # 제거할 불필요한 문구 목록
        stop_phrases = [
            "[공식 답변]", "[법적 근거]", "[기타 참고사항]", "[전문가의 추가 설명]",
            "[질문에 대한 추가 답변]", "[사용자에게 주는 추가 정보 및 조언]",
            "[전문가의 최종 진단 및 제안]", "[질문에 대한 완벽한 답변]", "사용자 질문:",
            "Human:", "답변:", "질문:", "끝."
        ]
        
        # 정규표현식을 사용하여 각 문구를 찾고, 가장 먼저 나오는 위치를 기준으로 자름
        # 대소문자 무시, 앞뒤 공백 무시
        escaped_phrases = [re.escape(phrase) for phrase in stop_phrases]
        pattern = re.compile(r'|'.join(escaped_phrases), re.IGNORECASE)
        
        match = pattern.search(text)
        if match:
            # 매칭된 부분 이전의 텍스트만 반환
            return text[:match.start()].strip()
            
        return text.strip()

    # 4. 각 체인 구성 요소 정의
    answer_chain = answer_prompt | llm | StrOutputParser() | RunnableLambda(clean_response)
    relevance_chain = relevance_prompt | llm | StrOutputParser()

    # 5. 조건부 분기 로직 구성
    branch = RunnableBranch(
        (lambda x: "yes" in x["relevance"].lower(), answer_chain),
        lambda x: "관련된 사례를 찾을 수 없어 정확한 답변이 어렵습니다."
    )

    # 6. 전체 RAG 체인 통합
    full_rag_chain = {
        "context": retriever | RunnableLambda(format_docs),
        "question": RunnablePassthrough()
    } | RunnableLambda(
        lambda x: {
            "context": x["context"],
            "question": x["question"],
            "relevance": relevance_chain.invoke({"context": x["context"], "question": x["question"]})
        }
    ) | branch
    
    return full_rag_chain

# --------------------------------------------------------------------------
# 3. 메인 애플리케이션 (main.py)
# --------------------------------------------------------------------------
state: Dict[str, Any] = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI 애플리케이션의 시작과 종료 시점에 실행될 로직을 관리합니다."""
    logger.info("AI 엔진 초기화를 시작합니다...")
    
    try:
        settings = Settings()
        state["settings"] = settings
    except Exception as e:
        logger.critical(f"설정 파일(.env) 로딩 중 치명적 오류 발생: {e}", exc_info=True)
        sys.exit("설정 파일을 로드할 수 없습니다. 프로그램을 종료합니다.")

    try:
        state["embeddings"] = HuggingFaceEmbeddings(
            model_name=settings.EMBEDDING_MODEL,
            model_kwargs={'device': settings.EMBEDDING_DEVICE},
            encode_kwargs={'normalize_embeddings': True}
        )
        logger.info("✅ 임베딩 모델 로드 완료.")

        connection_string = f"{settings.PG_DRIVER}://{settings.PG_USER}:{settings.PG_PASSWORD}@{settings.PG_HOST}:{settings.PG_PORT}/{settings.PG_DATABASE}"
        async_engine = create_async_engine(connection_string)
        state["async_engine"] = async_engine

        state["vector_store"] = PGVector(
            embeddings=state["embeddings"],
            collection_name=settings.COLLECTION_NAME,
            connection=async_engine,
            use_jsonb=True,
            create_extension=False
        )
        logger.info("✅ PostgreSQL 벡터 스토어 연결 완료.")

        state["llm"] = LlamaCpp(
            model_path=settings.LLM_MODEL_PATH,
            n_gpu_layers=settings.LLM_N_GPU_LAYERS, n_ctx=settings.LLM_N_CTX,
            max_tokens=settings.LLM_MAX_TOKENS, temperature=settings.LLM_TEMPERATURE,
            verbose=False, stop=["<|eot_id|>", "[사용자 질문]", "[전문가 답변]", "사용자 질문:", "전문가 답변:"]
        )
        logger.info("✅ LlamaCpp 모델 로드 완료.")
    except Exception as e:
        logger.critical(f"AI 모델 또는 DB 연결 초기화 중 치명적 오류 발생: {e}", exc_info=True)
        sys.exit("초기화 실패. 프로그램을 종료합니다.")

    retriever = state["vector_store"].as_retriever(search_kwargs={"k": 3})
    state["rag_chain"] = create_rag_chain(llm=state["llm"], retriever=retriever)
    logger.info("✅ RAG 체인 구성 완료. API 서버가 준비되었습니다.")
    
    yield
    
    logger.info("AI 엔진을 종료합니다...")
    await state["async_engine"].dispose()
    state.clear()

app = FastAPI(
    title="지능형 농지민원 답변 API (최종)",
    description="최적화된 프롬프트와 비동기 처리를 적용한 최종 버전의 RAG API입니다.",
    version="5.0.0",
    lifespan=lifespan
)

class QuestionRequest(BaseModel):
    query: str = Field(..., description="사용자의 민원 질문 내용", example="상속받은 농지는 직접 농사를 짓지 않아도 되나요?")

class AnswerResponse(BaseModel):
    answer: str = Field(..., description="AI 엔진이 생성한 답변", example="네, 상속받으신 농지는 직접 농사를 짓지 않으시더라도...")

@app.post("/ask", response_model=AnswerResponse, summary="질문/답변 생성")
async def ask_question(request: Request, question_body: QuestionRequest):
    """사용자의 질문을 받아 RAG 체인을 통해 답변을 생성합니다."""
    rag_chain = state.get("rag_chain")
    if not rag_chain:
        logger.error("RAG 체인이 초기화되지 않았습니다.")
        return {"answer": "서버 내부 오류: RAG 체인이 준비되지 않았습니다."}
        
    query = question_body.query
    logger.info(f"수신된 질문: {query}")
    try:
        answer = await rag_chain.ainvoke(query)
        logger.info(f"생성된 답변: {answer[:100]}...")
        return {"answer": answer.strip()}
    except Exception as e:
        logger.error(f"답변 생성 중 오류 발생: {e}", exc_info=True)
        return {"answer": "답변을 생성하는 중 오류가 발생했습니다."}

@app.get("/", summary="API 상태 확인")
def read_root():
    return {"message": "지능형 농지민원 답변 API 서버가 정상적으로 실행 중입니다."}

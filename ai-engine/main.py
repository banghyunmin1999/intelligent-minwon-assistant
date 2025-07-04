import os
import sys
import re
import logging
import asyncio
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from fastapi import FastAPI, Request, HTTPException, Depends
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings
from typing import List, Dict, Any

# LangChain 및 관련 라이브러리 임포트
import torch
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import LlamaCpp
from langchain_postgres.vectorstores import PGVector
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain.schema.runnable import Runnable, RunnablePassthrough, RunnableLambda, RunnableBranch
from langchain.schema.output_parser import StrOutputParser

# SQLAlchemy 비동기 엔진 임포트
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

# --------------------------------------------------------------------------
# 1. 설정 및 스키마 정의 (config.py & schemas.py)
# --------------------------------------------------------------------------
def setup_logging():
    """애플리케이션의 로깅 설정을 초기화합니다."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        stream=sys.stdout,
    )

class Settings(BaseSettings):
    """애플리케이션 설정을 .env 파일에서 로드합니다."""
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

class QuestionRequest(BaseModel):
    query: str = Field(..., description="사용자의 민원 질문 내용", example="상속받은 농지는 직접 농사를 짓지 않아도 되나요?")

class AnswerResponse(BaseModel):
    answer: str = Field(..., description="AI 엔진이 생성한 답변", example="네, 상속받으신 농지는 직접 농사를 짓지 않으시더라도...")

# --------------------------------------------------------------------------
# 2. 서비스 로직 정의 (services.py)
# --------------------------------------------------------------------------
class RAGService:
    """RAG 파이프라인과 관련된 모든 서비스를 캡슐화하는 클래스입니다."""
    def __init__(self, settings: Settings):
        self.settings = settings
        self.embeddings: HuggingFaceEmbeddings = None
        self.vector_store: PGVector = None
        self.llm: LlamaCpp = None
        self.rag_chain: Runnable = None
        self.async_engine: AsyncEngine = None
        self.logger = logging.getLogger(self.__class__.__name__)

    async def initialize(self):
        """AI 모델과 DB 연결 등 무거운 리소스를 비동기적으로 초기화합니다."""
        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.settings.EMBEDDING_MODEL,
            model_kwargs={'device': self.settings.EMBEDDING_DEVICE},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        connection_string = f"{self.settings.PG_DRIVER}://{self.settings.PG_USER}:{self.settings.PG_PASSWORD}@{self.settings.PG_HOST}:{self.settings.PG_PORT}/{self.settings.PG_DATABASE}"
        self.async_engine = create_async_engine(connection_string)

        self.vector_store = PGVector(
            embeddings=self.embeddings,
            collection_name=self.settings.COLLECTION_NAME,
            connection=self.async_engine,
            use_jsonb=True,
            create_extension=False
        )
        
        self.llm = LlamaCpp(
            model_path=self.settings.LLM_MODEL_PATH,
            n_gpu_layers=self.settings.LLM_N_GPU_LAYERS, n_ctx=self.settings.LLM_N_CTX,
            max_tokens=self.settings.LLM_MAX_TOKENS, temperature=self.settings.LLM_TEMPERATURE,
            verbose=False, stop=["<|eot_id|>", "[사용자 질문]", "[전문가 답변]", "사용자 질문:", "전문가 답변:"]
        )
        
        self._check_gpu_status()
        self._build_rag_chain()

    def _check_gpu_status(self):
        self.logger.info("--- GPU 사용 상태 확인 ---")
        if self.settings.LLM_N_GPU_LAYERS > 0 and torch.cuda.is_available():
            self.logger.info(f"✅ GPU 활성화 확인: {torch.cuda.device_count()}개의 CUDA 장치가 감지되었습니다.")
            self.logger.info(f"   - 현재 모델은 '{self.settings.LLM_N_GPU_LAYERS}'개의 레이어를 GPU에 오프로딩하여 실행됩니다.")
        else:
            self.logger.warning("ℹ️ GPU 비활성화 상태 또는 CUDA 사용 불가. 모델이 CPU로만 실행됩니다.")
        self.logger.info("------------------------")

    def _build_rag_chain(self):
        """RAG 체인의 모든 구성 요소를 조립합니다."""
        retriever = self.vector_store.as_retriever(search_kwargs={"k": 3})

        # --- 체인 구성 요소들 ---
        query_expansion_prompt = PromptTemplate.from_template("당신은 사용자의 질문을 더 효과적으로 검색할 수 있도록, 다양한 동의어와 관점을 사용하여 3가지 대체 질문을 생성하는 AI 어시스턴트입니다.\n각 질문은 한 줄로 작성하고, 다른 어떤 설명도 추가하지 마세요.\n\n[원본 질문]\n{question}\n\n[대체 질문 3가지]")
        answer_prompt = ChatPromptTemplate.from_template("""<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n당신은 대한민국 농지법을 담당하는 최고 수준의 전문 공무원입니다. 당신의 임무는 주어진 실제 민원 처리 사례를 바탕으로 사용자의 질문에 대한 답변만 생성하는 것입니다.\n**엄격한 출력 규칙:**\n1. 절대 사용자의 질문을 반복하거나 되묻지 마세요.\n2. '답변:', '설명:', '네, ...에 대해 질문해 주셨습니다.' 와 같은 서론을 절대 사용하지 마세요.\n3. 즉시 답변의 핵심 내용으로 시작하세요.\n4. 주어진 [참고 자료]에 없는 내용은 절대 언급하지 마세요.\n5. 오직 최종 답변 내용만 출력하세요. 불필요한 태그나 추가 설명은 절대 포함하지 마세요.<|eot_id|><|start_header_id|>user<|end_header_id|>\n[참고 자료: 실제 민원 처리 사례]\n{context}\n\n[사용자 질문]\n{question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>""")
        relevance_prompt = PromptTemplate.from_template("당신은 사용자의 질문이 주어진 컨텍스트와 관련이 있는지 판단하는 AI 필터입니다.\n컨텍스트는 '대한민국 농지법'에 대한 민원 사례들입니다.\n사용자의 질문이 '농지', '농업', '농지법' 등과 명백히 관련이 있다면 'yes'를, 전혀 관련 없는 주제라면 'no'를 반환해야 합니다.\n오직 'yes' 또는 'no'로만 대답하세요.\n\n[컨텍스트]\n{context}\n\n[사용자 질문]\n{question}\n\n[관련성 여부 (yes/no)]")
        
        query_expansion_chain = query_expansion_prompt | self.llm | StrOutputParser()
        answer_chain = answer_prompt | self.llm | StrOutputParser()
        relevance_chain = relevance_prompt | self.llm | StrOutputParser()

        # --- 비동기 함수 정의 ---
        async def get_expanded_documents(question: str) -> str:
            """질문을 확장하고, 검색하고, 결과를 포맷팅하는 비동기 함수."""
            expanded_queries_str = await query_expansion_chain.ainvoke({"question": question})
            queries = [question] + parse_expanded_queries(expanded_queries_str)
            self.logger.info(f"정제된 확장 질문: {queries}")

            tasks = [retriever.ainvoke(q) for q in queries]
            results = await asyncio.gather(*tasks)
            
            all_docs = [doc for sublist in results for doc in sublist]
            unique_docs = list({doc.page_content: doc for doc in all_docs}.values())
            
            return "\n\n".join(
                f"--- 사례 {i+1} ---\n사례 질문: {doc.page_content}\n공식 답변: {doc.metadata.get('answer', '답변 정보 없음')}"
                for i, doc in enumerate(unique_docs)
            )

        def parse_expanded_queries(raw_output: str) -> List[str]:
            """LLM이 생성한 확장 질문에서 코드나 불필요한 설명을 제거하고 순수한 질문 텍스트만 추출합니다."""
            lines = raw_output.strip().split('\n')
            questions = []
            for line in lines:
                cleaned_line = re.sub(r'^\s*\d+\.\s*', '', line).strip()
                if cleaned_line and not cleaned_line.startswith(('```', '#', '//')):
                    questions.append(cleaned_line)
            return questions[:3]

        async def check_relevance_and_pass_through(input_dict: Dict) -> Dict:
            """비동기적으로 관련성을 체크하고, 결과를 원본 딕셔너리에 추가하여 반환합니다."""
            relevance = await relevance_chain.ainvoke(input_dict)
            input_dict["relevance"] = relevance
            return input_dict

        # --- 최종 체인 조립 ---
        context_and_question = {
            "context": RunnableLambda(get_expanded_documents),
            "question": RunnablePassthrough()
        }

        branch = RunnableBranch(
            (lambda x: "yes" in x["relevance"].lower(), answer_chain),
            lambda x: "관련된 사례를 찾을 수 없어 정확한 답변이 어렵습니다."
        )

        self.rag_chain = context_and_question | RunnableLambda(check_relevance_and_pass_through) | branch

    async def get_answer(self, query: str) -> str:
        """주어진 질문에 대해 RAG 체인을 비동기적으로 실행하여 답변을 반환합니다."""
        if not self.rag_chain:
            raise RuntimeError("RAG chain is not initialized.")
        return await self.rag_chain.ainvoke(query)

    async def dispose(self):
        """애플리케이션 종료 시 DB 연결 등 리소스를 해제합니다."""
        if self.async_engine:
            await self.async_engine.dispose()

# --------------------------------------------------------------------------
# 3. 메인 애플리케이션 (main.py)
# --------------------------------------------------------------------------
setup_logging()
logger = logging.getLogger(__name__)
state: Dict[str, Any] = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI 애플리케이션의 시작과 종료 시점에 실행될 로직을 관리합니다."""
    logger.info("AI 엔진 초기화를 시작합니다...")
    try:
        settings = Settings()
        state["rag_service"] = RAGService(settings)
        await state["rag_service"].initialize()
        logger.info("✅ AI 엔진 초기화 완료. API 서버가 준비되었습니다.")
    except Exception as e:
        logger.critical(f"AI 엔진 초기화 중 치명적 오류 발생: {e}", exc_info=True)
        sys.exit("초기화 실패. 프로그램을 종료합니다.")
    
    yield
    
    logger.info("AI 엔진을 종료합니다...")
    if "rag_service" in state:
        await state["rag_service"].dispose()
    state.clear()

def get_rag_service() -> RAGService:
    """API 엔드포인트에 RAGService 인스턴스를 주입하는 함수입니다."""
    service = state.get("rag_service")
    if not service:
        raise HTTPException(status_code=503, detail="서버가 아직 준비되지 않았습니다. 잠시 후 다시 시도해주세요.")
    return service

app = FastAPI(
    title="지능형 농지민원 답변 API (최종 전문가 버전)",
    description="모듈화, 의존성 주입, 상세 오류 처리 등 전문가 수준의 패턴을 적용한 최종 버전의 RAG API입니다.",
    version="6.0.0",
    lifespan=lifespan
)

@app.post("/ask", response_model=AnswerResponse, summary="질문/답변 생성")
async def ask_question(
    question_body: QuestionRequest,
    rag_service: RAGService = Depends(get_rag_service)
):
    """
    사용자의 질문을 받아 RAG 체인을 통해 답변을 생성합니다.
    """
    query = question_body.query
    logger.info(f"수신된 질문: {query}")
    try:
        answer = await rag_service.get_answer(query)
        logger.info(f"생성된 답변: {answer[:100]}...")
        return AnswerResponse(answer=answer)
    except Exception as e:
        logger.error(f"답변 생성 중 예상치 못한 오류 발생: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="답변을 생성하는 중 서버 내부 오류가 발생했습니다.")

@app.get("/", summary="API 상태 확인")
def read_root():
    return {"message": "지능형 농지민원 답변 API 서버가 정상적으로 실행 중입니다."}

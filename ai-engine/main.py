"""
민원 AI 엔진 v2 (FastAPI 기반)
-----------------------------
- 버전: 2.0
- 주요 기능: 시스템/리소스 체크, GPU 자동 감지 및 fallback, LangChain+FAISS+Ollama 기반 RAG, 헬스체크 및 질의응답 API 제공
- 작성일: 2025-07-03
"""

import os
import logging
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import psutil
import time
import platform
from datetime import datetime
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain_community.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

# =========================
# 로깅 및 시스템 정보 함수
# =========================

# 로깅 설정 (INFO 레벨, 포맷 지정)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 시스템 정보 체크 함수: OS/CPU/메모리 등 로그 출력
def check_system_info():
    try:
        # 시스템 정보
        logger.info(f"시스템: {platform.system()} {platform.release()}")
        logger.info(f"프로세서: {platform.processor()}")
        # 메모리 정보
        mem = psutil.virtual_memory()
        logger.info(f"총 메모리: {mem.total / (1024**3):.2f}GB")
        logger.info(f"사용 중 메모리: {mem.used / (1024**3):.2f}GB")
        logger.info(f"메모리 사용률: {mem.percent}%")
    except Exception as e:
        logger.error(f"시스템 정보 체크 중 오류 발생: {str(e)}")

def check_gpu_availability():
    try:
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            logger.info(f"GPU 사용 가능. 총 {device_count}개의 GPU가 발견되었습니다.")
            for i in range(device_count):
                logger.info(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            return True, "GPU 사용 가능"
        else:
            logger.warning("GPU를 사용할 수 없습니다. CPU로 실행됩니다.")
            # 주요 원인 진단
            import subprocess
            try:
                result = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
                if result.returncode != 0:
                    reason = "nvidia-smi 미동작 (드라이버/호스트 설정 문제)"
                else:
                    reason = "CUDA 드라이버는 있으나 torch에서 인식 불가 (torch 설치 혹은 --gpus 옵션 미사용 등)"
            except FileNotFoundError:
                reason = "nvidia-smi 명령어 없음 (NVIDIA 드라이버 미설치)"
            return False, reason
    except Exception as e:
        logger.error(f"GPU 체크 중 오류 발생: {str(e)}")
        return False, f"GPU 체크 중 오류: {str(e)}"

        # 시스템 정보
        logger.info(f"시스템: {platform.system()} {platform.release()}")
        logger.info(f"프로세서: {platform.processor()}")
        
        # 메모리 정보
        mem = psutil.virtual_memory()
        logger.info(f"총 메모리: {mem.total / (1024**3):.2f}GB")
        logger.info(f"사용 중 메모리: {mem.used / (1024**3):.2f}GB")
        logger.info(f"메모리 사용률: {mem.percent}%")
        
        # 디스크 정보
        disk = psutil.disk_usage('/')
        logger.info(f"디스크 사용률: {disk.percent}%")
        
        # 네트워크 연결 상태
        if psutil.net_if_stats():
            logger.info("네트워크 인터페이스가 활성화되어 있습니다.")
        else:
            logger.warning("네트워크 인터페이스가 비활성화되어 있습니다.")
    except Exception as e:
        logger.error(f"시스템 정보 체크 중 오류 발생: {str(e)}")
        raise HTTPException(status_code=500, detail="시스템 정보 체크 중 오류 발생")

# 모델 로드 시간 체크
def check_model_load_time():
    try:
        start_time = time.time()
        # 모델 로드 코드
        # 이 부분은 실제 모델 로드 코드로 대체되어야 합니다
        time.sleep(2)  # 예시로 2초 대기
        load_time = time.time() - start_time
        logger.info(f"모델 로드 시간: {load_time:.2f}초")
        return load_time
    except Exception as e:
        logger.error(f"모델 로드 중 오류 발생: {str(e)}")
        raise HTTPException(status_code=500, detail="모델 로드 중 오류 발생")

# 리소스 사용량 모니터링
def monitor_resources():
    try:
        mem = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        cpu_percent = psutil.cpu_percent()
        
        logger.info(f"CPU 사용률: {cpu_percent}%")
        logger.info(f"메모리 사용률: {mem.percent}%")
        logger.info(f"디스크 사용률: {disk.percent}%")
        
        status = {
            'cpu_percent': cpu_percent,
            'memory_percent': mem.percent,
            'disk_percent': disk.percent,
            'gpu_enabled': USE_GPU,
            'gpu_reason': GPU_REASON
        }
        return status
    except Exception as e:
        logger.error(f"리소스 모니터링 중 오류 발생: {str(e)}")
        raise HTTPException(status_code=500, detail="리소스 모니터링 중 오류 발생")

# FastAPI 앱 초기화
app = FastAPI()

# API 요청 시 받을 데이터 모델을 정의합니다.
class Question(BaseModel):
    query: str

# --- LangChain RAG 파이프라인 설정 ---

print("AI 엔진 초기화 중...")

# 1. 임베딩 모델 로드 (create_index.py와 동일한 모델 사용)
embeddings = HuggingFaceEmbeddings(
    model_name="jhgan/ko-sbert-nli",
    model_kwargs={'device': 'cuda'},
    encode_kwargs={'normalize_embeddings': True}
)

# 2. 저장된 FAISS 인덱스 로드
vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
# 3. Retriever(검색기) 생성: 유사도가 높은 상위 5개의 문서를 검색하도록 설정
retriever = vector_store.as_retriever(search_kwargs={"k": 5})
# 4. Prompt Template 정의
template = """
당신은 대한민국 민원 담당 공무원입니다. 주어진 '유사 민원 사례'를 바탕으로 사용자의 '질문'에 대해 친절하고 논리적으로 답변을 생성해야 합니다.
근거가 되는 사례를 문장에 포함하여 답변하면 좋습니다. 만약 유사 사례에서 답을 찾을 수 없다면, "관련된 유사 사례를 찾을 수 없어 정확한 답변이 어렵습니다." 라고 솔직하게 답변하세요.

[유사 민원 사례]
{context}

[질문]
{question}

[답변]
"""
prompt = ChatPromptTemplate.from_template(template)
# 5. LLM 모델 정의 (Ollama 로컬 모델 사용)
llm = ChatOllama(model="bllossom:latest") 
# 6. RAG 체인(Chain) 구성
rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)
print("✅ AI 엔진 초기화 완료.")

# --- API 엔드포인트 정의 ---
@app.get("/health")
async def health_check():
    """시스템 상태 체크 엔드포인트"""
    try:
        system_info = check_system_info()
        resources = monitor_resources()
        return {
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'system_info': system_info,
            'resources': resources
        }
    except Exception as e:
        logger.error(f"헬스 체크 중 오류 발생: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ask")
async def ask_question(question: Question):
    """사용자의 질문을 받아 RAG 체인을 통해 답변을 생성합니다."""
    print(f"수신된 질문: {question.query}")
    try:
        answer = rag_chain.invoke(question.query)
        print(f"생성된 답변: {answer}")
        return {"answer": answer}
    except Exception as e:
        print(f"오류 발생: {e}")
        return {"error": "답변을 생성하는 중 오류가 발생했습니다."}

@app.get("/")
def read_root():
    return {"message": "민원분석 AI 엔진 API 서버입니다. /docs 로 접속하여 테스트할 수 있습니다."}

# --- 시스템 초기화 및 리소스 체크 ---
logger.info("시스템 초기화 시작")
check_system_info()
USE_GPU, GPU_REASON = check_gpu_availability()
model_load_time = check_model_load_time()
if USE_GPU:
    logger.info("GPU를 사용하여 모델을 로드합니다.")
    device = "cuda"
else:
    logger.info("CPU를 사용하여 모델을 로드합니다.")
    device = "cpu"
load_dotenv()

# --- LangChain RAG 파이프라인 설정 ---

print("AI 엔진 초기화 중...")

# 1. 임베딩 모델 로드 (create_index.py와 동일한 모델 사용)
embeddings = HuggingFaceEmbeddings(
    model_name="jhgan/ko-sbert-nli",
    model_kwargs={'device': 'cuda'},
    encode_kwargs={'normalize_embeddings': True}
)

# 2. 저장된 FAISS 인덱스 로드
# allow_dangerous_deserialization=True 옵션은 로컬 인덱스를 신뢰하고 로드하기 위해 필요합니다.
vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

# 3. Retriever(검색기) 생성: 유사도가 높은 상위 5개의 문서를 검색하도록 설정
retriever = vector_store.as_retriever(search_kwargs={"k": 5})

# 4. Prompt Template 정의
template = """
당신은 대한민국 민원 담당 공무원입니다. 주어진 '유사 민원 사례'를 바탕으로 사용자의 '질문'에 대해 친절하고 논리적으로 답변을 생성해야 합니다.
근거가 되는 사례를 문장에 포함하여 답변하면 좋습니다. 만약 유사 사례에서 답을 찾을 수 없다면, "관련된 유사 사례를 찾을 수 없어 정확한 답변이 어렵습니다." 라고 솔직하게 답변하세요.

[유사 민원 사례]
{context}

[질문]
{question}

[답변]
"""
prompt = ChatPromptTemplate.from_template(template)

# 5. LLM 모델 정의 (Ollama 로컬 모델 사용)
# 중요: model 이름은 'ollama create' 명령어로 최종 성공한 모델 이름과 정확히 일치해야 합니다.
llm = ChatOllama(model="bllossom:latest") 

# 6. RAG 체인(Chain) 구성
rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

print("✅ AI 엔진 초기화 완료.")

# --- API 엔드포인트 정의 ---

@app.post("/ask")
async def ask_question(question: Question):
    """사용자의 질문을 받아 RAG 체인을 통해 답변을 생성합니다."""
    print(f"수신된 질문: {question.query}")
    try:
        answer = rag_chain.invoke(question.query)
        print(f"생성된 답변: {answer}")
        return {"answer": answer}
    except Exception as e:
        print(f"오류 발생: {e}")
        return {"error": "답변을 생성하는 중 오류가 발생했습니다."}

@app.get("/")
def read_root():
    return {"message": "민원분석 AI 엔진 API 서버입니다. /docs 로 접속하여 테스트할 수 있습니다."}
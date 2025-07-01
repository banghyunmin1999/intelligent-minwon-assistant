import os
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain_community.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

# .env 파일에서 환경 변수를 로드합니다.
load_dotenv()

# FastAPI 앱을 초기화합니다.
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
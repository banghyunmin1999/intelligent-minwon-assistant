# final_system.py

import sys
import re
from llama_cpp import Llama
from langchain_postgres.vectorstores import PGVector
from langchain_huggingface import HuggingFaceEmbeddings

# --- 1. 설정 (Configuration) ---

# VectorDB (PostgreSQL) Settings
PG_HOST = "localhost"
PG_PORT = 5432
PG_DATABASE = "minwon_pg_db"
PG_USER = "minwon_user"
PG_PASSWORD = "1234"
COLLECTION_NAME = "minwon_pdf_cases_v1" # ingestion.py에서 생성한 컬렉션
CONNECTION_STRING = f"postgresql+psycopg2://{PG_USER}:{PG_PASSWORD}@{PG_HOST}:{PG_PORT}/{PG_DATABASE}"

# Model Path
model_path = "/home/bang/intelligent-minwon-assistant/models/llama-3-Korean-Bllossom-8B-gguf-Q4_K_M/llama-3-Korean-Bllossom-8B-Q4_K_M.gguf"

# --- 2. 초기화 (Initialization) ---

# 모델 로드
print("⏳ LLM 모델을 로드합니다...")
llm = Llama(
    model_path=model_path, n_ctx=4096, n_threads=4, n_gpu_layers=32,
    n_batch=16, verbose=False, temperature=0.2, repeat_penalty=1.2
)
print("✅ LLM 모델 로드 완료.")

# 임베딩 및 벡터 스토어 연결
print("⏳ 임베딩 모델 및 벡터 스토어에 연결합니다...")
embeddings = HuggingFaceEmbeddings(
    model_name="jhgan/ko-sroberta-multitask",
    model_kwargs={'device': 'cpu'}, encode_kwargs={'normalize_embeddings': True}
)
vector_store = PGVector(
    embeddings=embeddings, collection_name=COLLECTION_NAME, connection=CONNECTION_STRING,
)
retriever = vector_store.as_retriever(search_kwargs={"k": 3}) # 컨텍스트는 3개로 집중
print("✅ 임베딩 모델 및 벡터 스토어 연결 완료.")


# --- 3. 핵심 로직 (Core Logic) ---

def main():
    # 사용자의 원본 복합 질문
    complex_question = "농지 소유권을 양도할 때 필요한 절차와 주의사항은 무엇인가요? 그리고 농지법에서 정한 농사 짓기 의무는 무엇인지 알려주세요."
    print(f"\n[원본 질문]\n{complex_question}\n" + "="*50)

    # --- 단계 1: 복합 질문을 단일 질문들로 분해 ---
    print("\n⏳ 1단계: 복합 질문을 단일 질문으로 분해합니다...")
    
    sub_question_generation_prompt = f"""사용자의 복잡한 질문을 검색과 답변에 용이한 단순한 단일 질문 여러 개로 분해하는 역할을 맡았습니다.
각 질문은 하나의 주제만 다루어야 합니다. 번호가 매겨진 목록 형태로 질문만 간결하게 출력하세요.

[사용자 질문]
{complex_question}

[분해된 단일 질문 목록]
"""
    
    response = llm(sub_question_generation_prompt, max_tokens=256, stop=["["])
    raw_sub_questions = response['choices'][0]['text'].strip()
    
    # 생성된 질문 목록 파싱 (예: "1. 첫번째 질문\n2. 두번째 질문" -> ["첫번째 질문", "두번째 질문"])
    sub_questions = [re.sub(r'^\d+\.\s*', '', line).strip() for line in raw_sub_questions.split('\n') if line.strip()]
    
    print("✅ 분해된 질문 목록:")
    for i, sq in enumerate(sub_questions):
        print(f"   {i+1}. {sq}")
    print("="*50)

    # --- 단계 2: 각 단일 질문에 대해 답변 생성 ---
    print("\n⏳ 2단계: 각 단일 질문에 대해 관련 문서를 찾고 답변을 생성합니다...")
    
    final_answers = []
    for sub_q in sub_questions:
        print(f"\n  [처리 중인 질문]: {sub_q}")
        
        # a. 관련 문서 검색
        docs = retriever.invoke(sub_q)
        context = "\n---\n".join([doc.page_content for doc in docs])
        
        # b. 답변 생성을 위한 프롬프트
        answer_generation_prompt = f"""## 지시사항
- 당신은 대한민국 농지법 전문가입니다.
- 반드시 `[관련 사례]` 내용만을 이용해서 `[질문]`에 대한 답변을 생성하세요.
- 간결하고 명확하게, 사실만을 기반으로 설명하세요.

## 관련 사례
{context}

## 질문
{sub_q}

## 답변
"""
        # c. 답변 생성
        response = llm(answer_generation_prompt, max_tokens=512, stop=["##", "\n\n"])
        answer = response['choices'][0]['text'].strip()
        final_answers.append({"question": sub_q, "answer": answer})
        print(f"  [생성된 답변]: {answer[:100]}...") # 답변 일부만 출력
        
    print("✅ 모든 질문에 대한 답변 생성 완료.")
    print("="*50)

    # --- 단계 3: 결과 취합 및 최종 출력 ---
    print("\n\n✅ 최종 답변\n" + "="*50)
    for item in final_answers:
        print(f"### ❓ {item['question']}")
        print(f"답변: {item['answer']}\n")

# --- 4. 실행 ---
if __name__ == "__main__":
    main()
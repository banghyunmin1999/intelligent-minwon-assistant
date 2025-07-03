import os
import re
import sys
from dotenv import load_dotenv
import psycopg2
from psycopg2.extras import execute_values
import numpy as np
from langchain_huggingface import HuggingFaceEmbeddings
# PDF 처리를 위해 PyPDF2 라이브러리가 필요합니다.
# pip install PyPDF2
try:
    from PyPDF2 import PdfReader
except ImportError:
    print("❌ 'PyPDF2' 라이브러리가 설치되지 않았습니다.")
    print("   - 해결 방법: 터미널에 'pip install PyPDF2'를 입력하세요.")
    sys.exit()

# --- 1. 설정 (Configuration) ---
print("환경 변수를 로드합니다...")
load_dotenv()

# 처리할 PDF 파일 경로 (스크립트와 같은 위치에 있다고 가정)
PDF_FILE_PATH = "2023년 농지민원 사례집(최종).pdf"
SOURCE_NAME = "2023년 농지민원 사례집"

# PostgreSQL DB 설정
DB_CONFIG = {
    "host": os.getenv("PG_HOST", "localhost"),
    "port": os.getenv("PG_PORT", "5432"),
    "user": os.getenv("PG_USER", "minwon_user"),
    "password": os.getenv("PG_PASSWORD", "1234"),
    "dbname": os.getenv("PG_DB_NAME", "minwon_pg_db")
}

# 임베딩 모델 설정
EMBEDDING_MODEL_CONFIG = {
    "model_name": "jhgan/ko-sbert-nli",
    "model_kwargs": {'device': 'cuda'},
    "encode_kwargs": {'normalize_embeddings': True}
}

# --- 2. PDF 파싱 및 데이터 처리 함수 ---

def parse_qna_from_pdf(file_path):
    """PDF 파일에서 질문과 답변 쌍을 추출하는 개선된 함수."""
    print(f"'{file_path}' 파일 분석을 시작합니다...")
    if not os.path.exists(file_path):
        print(f"❌ [오류] 파일을 찾을 수 없습니다: {file_path}")
        return []

    try:
        reader = PdfReader(file_path)
        full_text = ""
        for page in reader.pages:
            # 페이지 사이에 공백을 추가하여 단어가 합쳐지는 것을 방지
            full_text += page.extract_text() + " "

        # '목차' 및 앞부분 제외
        toc_end_index = full_text.find("농지의 정의")
        if toc_end_index != -1:
            full_text = full_text[toc_end_index:]

        qna_list = []
        # 더 유연한 정규표현식으로 각 질문의 시작점을 찾음
        # (예: '문 1', '문 22' 등)
        question_starts = list(re.finditer(r'문\s+\d{1,3}\s+', full_text))

        for i, match in enumerate(question_starts):
            start_pos = match.end()
            # 다음 질문 시작점 또는 문서 끝까지를 하나의 Q&A 블록으로 간주
            end_pos = question_starts[i+1].start() if i + 1 < len(question_starts) else len(full_text)
            
            qa_block = full_text[start_pos:end_pos]
            
            # '답변' 키워드를 기준으로 질문과 답변으로 나눔
            answer_split = re.split(r'\s+답변\s+', qa_block, 1)
            
            if len(answer_split) == 2:
                # 공백, 줄바꿈 등을 정리
                question_text = " ".join(answer_split[0].strip().split())
                answer_text = " ".join(answer_split[1].strip().split())
                
                # 페이지 바닥글 등 불필요한 텍스트 제거
                answer_text = re.sub(r'\s*\w*\s*농지민원 사례집\s*\|.*', '', answer_text)
                
                if question_text and answer_text:
                    qna_list.append({
                        "question": question_text,
                        "answer": answer_text
                    })

        print(f"✅ 총 {len(qna_list)}개의 Q&A 사례를 성공적으로 추출했습니다.")
        if qna_list:
            print(f"  - 첫 번째 추출된 질문: {qna_list[0]['question'][:70]}...")
            print(f"  - 첫 번째 추출된 답변: {qna_list[0]['answer'][:70]}...")

        return qna_list

    except Exception as e:
        print(f"❌ [오류] PDF 파싱 중 예외 발생: {e}")
        return []

def embed_and_store_qna():
    """PDF에서 Q&A를 추출하고, 임베딩하여 PostgreSQL에 저장합니다."""
    qna_data = parse_qna_from_pdf(PDF_FILE_PATH)
    if not qna_data:
        print("처리할 데이터가 없어 프로세스를 종료합니다.")
        return

    # 임베딩 모델 초기화
    print("임베딩 모델을 초기화합니다...")
    try:
        embeddings = HuggingFaceEmbeddings(**EMBEDDING_MODEL_CONFIG)
    except Exception as e:
        if "CUDA" in str(e):
            print("⚠️ GPU(CUDA) 로딩 실패. CPU로 전환합니다.")
            EMBEDDING_MODEL_CONFIG['model_kwargs'] = {'device': 'cpu'}
            embeddings = HuggingFaceEmbeddings(**EMBEDDING_MODEL_CONFIG)
        else:
            print(f"❌ 임베딩 모델 로딩 실패: {e}")
            return

    # 질문(question) 텍스트를 배치로 임베딩
    questions = [item['question'] for item in qna_data]
    print(f"총 {len(questions)}개의 질문을 벡터로 변환합니다...")
    try:
        embedded_vectors = embeddings.embed_documents(questions)
        print("✅ 문서 임베딩 완료.")
    except Exception as e:
        print(f"❌ 문서 임베딩 중 오류 발생: {e}")
        return

    # PostgreSQL에 데이터 저장
    print("PostgreSQL에 Q&A 데이터를 저장합니다...")
    try:
        with psycopg2.connect(**DB_CONFIG) as conn:
            with conn.cursor() as cur:
                insert_query = """
                INSERT INTO minwon_qna_cases (question, answer, source, embedding)
                VALUES %s
                """
                
                data_to_insert = [
                    (
                        item['question'],
                        item['answer'],
                        SOURCE_NAME,
                        np.array(vec).tolist()
                    ) for item, vec in zip(qna_data, embedded_vectors)
                ]

                print("- 기존 'minwon_qna_cases' 테이블 데이터를 삭제하고 새로 삽입합니다.")
                cur.execute("TRUNCATE TABLE minwon_qna_cases RESTART IDENTITY;")
                
                execute_values(cur, insert_query, data_to_insert)
                conn.commit()
                print(f"✅ {len(data_to_insert)}개의 Q&A 데이터가 PostgreSQL에 성공적으로 저장되었습니다.")

    except psycopg2.Error as e:
        print(f"❌ PostgreSQL 처리 중 오류가 발생했습니다: {e}")
    except Exception as e:
        print(f"❌ 예기치 않은 오류가 발생했습니다: {e}")

# --- 3. 메인 실행 블록 ---
if __name__ == "__main__":
    embed_and_store_qna()

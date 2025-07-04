import os
import re
import sys
import asyncio
from dotenv import load_dotenv
from pydantic_settings import BaseSettings
from PyPDF2 import PdfReader
import numpy as np
from typing import List

# LangChain 및 관련 라이브러리 임포트
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_postgres.vectorstores import PGVector

# --- 1. 설정 관리 (Configuration) ---

# 프로젝트 루트 디렉토리를 명확하게 정의합니다.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# .env 파일 또는 환경 변수에서 설정을 자동으로 읽어오는 클래스
class Settings(BaseSettings):
    # PDF 및 컬렉션 정보
    PDF_FILE_PATH: str = "data/2023년 농지민원 사례집(최종).pdf"
    SOURCE_NAME: str = "2023년 농지민원 사례집"
    COLLECTION_NAME: str

    # PostgreSQL DB 설정
    PG_DRIVER: str = "postgresql+asyncpg"
    PG_HOST: str
    PG_PORT: int
    PG_DATABASE: str
    PG_USER: str
    PG_PASSWORD: str

    # 임베딩 모델 설정
    EMBEDDING_MODEL: str
    EMBEDDING_DEVICE: str

    class Config:
        env_file = os.path.join(PROJECT_ROOT, ".env")
        env_file_encoding = "utf-8"
        extra = 'ignore'

print("환경 변수를 로드합니다...")
try:
    settings = Settings()
except Exception as e:
    print(f"❌ 설정 파일(.env) 로딩 중 오류 발생: {e}")
    print("   - 프로젝트 최상위 폴더에 '.env' 파일이 있는지, 모든 설정값이 포함되어 있는지 확인해주세요.")
    sys.exit()


# --- 2. PDF 파싱 및 데이터 처리 함수 ---

def parse_qna_from_pdf(file_path: str, source_name: str) -> List[Document]:
    """PDF 파일에서 질문과 답변 쌍을 추출하여 Document 객체 리스트로 반환합니다."""
    abs_file_path = os.path.join(PROJECT_ROOT, file_path)
    print(f"'{abs_file_path}' 파일 분석을 시작합니다...")
    if not os.path.exists(abs_file_path):
        print(f"❌ [오류] 파일을 찾을 수 없습니다: {abs_file_path}")
        return []

    try:
        reader = PdfReader(abs_file_path)
        full_text = "".join(page.extract_text() + " " for page in reader.pages)
        
        toc_end_index = full_text.find("농지의 정의")
        if toc_end_index != -1:
            full_text = full_text[toc_end_index:]

        qna_list = []
        question_starts = list(re.finditer(r'문\s+\d{1,3}\s+', full_text))

        for i, match in enumerate(question_starts):
            start_pos = match.end()
            end_pos = question_starts[i+1].start() if i + 1 < len(question_starts) else len(full_text)
            qa_block = full_text[start_pos:end_pos]
            
            answer_split = re.split(r'\s+답변\s+', qa_block, 1)
            
            if len(answer_split) == 2:
                question_text = " ".join(answer_split[0].strip().split())
                answer_text = " ".join(answer_split[1].strip().split())
                answer_text = re.sub(r'\s*\w*\s*농지민원 사례집\s*\|.*', '', answer_text)
                
                if question_text and answer_text:
                    qna_list.append(
                        Document(
                            page_content=question_text,
                            metadata={"answer": answer_text, "source": source_name}
                        )
                    )
        
        print(f"✅ 총 {len(qna_list)}개의 Q&A 문서를 성공적으로 추출했습니다.")
        return qna_list

    except Exception as e:
        print(f"❌ [오류] PDF 파싱 중 예외 발생: {e}")
        return []

async def setup_vector_database():
    """
    PDF에서 문서를 추출하고, LangChain의 from_documents 메서드를 사용하여
    데이터베이스에 테이블을 생성하고 데이터를 저장합니다.
    """
    documents = parse_qna_from_pdf(settings.PDF_FILE_PATH, settings.SOURCE_NAME)
    if not documents:
        print("처리할 문서가 없어 프로세스를 종료합니다.")
        return

    print("임베딩 모델을 초기화합니다...")
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name=settings.EMBEDDING_MODEL,
            model_kwargs={'device': settings.EMBEDDING_DEVICE},
            encode_kwargs={'normalize_embeddings': True}
        )
    except Exception as e:
        print(f"❌ 임베딩 모델 로딩 실패: {e}")
        return

    connection_string = (
        f"{settings.PG_DRIVER}://{settings.PG_USER}:{settings.PG_PASSWORD}"
        f"@{settings.PG_HOST}:{settings.PG_PORT}/{settings.PG_DATABASE}"
    )
    
    print("PostgreSQL에 벡터 데이터베이스를 설정합니다...")
    print(f"'{settings.COLLECTION_NAME}' 컬렉션에 데이터를 저장합니다.")

    try:
        # *** 핵심 수정: 확장 기능 자동 생성을 비활성화합니다. ***
        await PGVector.afrom_documents(
            embedding=embeddings,
            documents=documents,
            collection_name=settings.COLLECTION_NAME,
            connection=connection_string,
            pre_delete_collection=True,
            create_extension=False, # 이 옵션을 추가하여 오류를 해결합니다.
        )
        print(f"✅ {len(documents)}개의 문서가 DB에 성공적으로 저장되었습니다.")
        print("   - 테이블: langchain_pg_collection, langchain_pg_embedding")

    except Exception as e:
        print(f"❌ 벡터 데이터베이스 설정 중 오류 발생: {e}")

# --- 3. 메인 실행 블록 ---
if __name__ == "__main__":
    asyncio.run(setup_vector_database())

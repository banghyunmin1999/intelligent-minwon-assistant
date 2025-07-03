import os
import requests
import json
import psycopg2
from psycopg2.extras import execute_values
import time
from dotenv import load_dotenv
import numpy as np
from langchain_huggingface import HuggingFaceEmbeddings

# --- 설정 (Configuration) ---
print("환경 변수를 로드합니다...")
load_dotenv()

# 공공데이터 API 설정
API_CONFIG = {
    "service_key": os.getenv("API_SERVICE_KEY"),
    "base_url": "http://apis.data.go.kr/1140100/minAnalsInfoView5/minSimilarInfo5"
}

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

# 검색할 키워드 목록
SEARCH_KEYWORDS = [
    "층간소음", "불법주차", "쓰레기 무단투기", "공동주택 하자보수", "재개발 보상", "전세사기"
]

# --- 전역 변수 ---
print("임베딩 모델을 초기화합니다...")
try:
    embeddings_model = HuggingFaceEmbeddings(**EMBEDDING_MODEL_CONFIG)
except Exception as e:
    if "CUDA" in str(e):
        print("⚠️ GPU(CUDA) 로딩 실패. CPU로 전환합니다.")
        EMBEDDING_MODEL_CONFIG['model_kwargs'] = {'device': 'cpu'}
        embeddings_model = HuggingFaceEmbeddings(**EMBEDDING_MODEL_CONFIG)
    else:
        print(f"❌ 임베딩 모델 로딩 실패: {e}")
        embeddings_model = None

# --- 데이터베이스 관련 함수 ---
def create_db_connection():
    """PostgreSQL 데이터베이스에 연결하고 connection 객체를 반환합니다."""
    try:
        connection = psycopg2.connect(**DB_CONFIG)
        print("✅ PostgreSQL 데이터베이스 연결 성공!")
        return connection
    except psycopg2.Error as e:
        print(f"❌ 데이터베이스 연결 오류: '{e}'")
        return None

def insert_minwon_data(connection, cases, search_keyword):
    """수집한 민원 데이터를 임베딩하여 데이터베이스에 삽입합니다."""
    if not cases:
        print("  - 삽입할 데이터가 없습니다.")
        return

    unique_cases = []
    seen_dates = set()
    for case in cases:
        create_date_str = case.get('create_date')
        if create_date_str and create_date_str.isdigit():
            if create_date_str not in seen_dates:
                unique_cases.append(case)
                seen_dates.add(create_date_str)

    if len(unique_cases) < len(cases):
        print(f"  - ⚠️ 중복 데이터 발견. 원본 {len(cases)}개 -> 고유 {len(unique_cases)}개로 필터링됨.")

    if not unique_cases:
        print("  - 중복 제거 후 삽입할 유효 데이터가 없습니다.")
        return

    contents = [case.get('content', '') for case in unique_cases]
    print(f"  - {len(contents)}개의 고유 문서를 벡터로 변환 중...")
    try:
        embedded_vectors = embeddings_model.embed_documents(contents)
    except Exception as e:
        print(f"❌ 문서 임베딩 중 오류 발생: {e}")
        return

    data_to_insert = []
    for case, vector in zip(unique_cases, embedded_vectors):
        create_date_str = case.get('create_date')
        if create_date_str and create_date_str.isdigit():
            data_to_insert.append((
                int(create_date_str),
                case.get('title'),
                case.get('content'),
                case.get('main_sub_name'),
                case.get('dep_name'),
                search_keyword,
                np.array(vector).tolist()
            ))

    if not data_to_insert:
        print("  - 삽입할 유효한 데이터가 없습니다.")
        return

    cursor = connection.cursor()
    insert_query = """
    INSERT INTO minwon_cases_pg (create_date, title, content, institution, department, search_keyword, embedding)
    VALUES %s
    ON CONFLICT (create_date) DO UPDATE SET
        title = EXCLUDED.title,
        content = EXCLUDED.content,
        embedding = EXCLUDED.embedding;
    """
    try:
        execute_values(cursor, insert_query, data_to_insert)
        connection.commit()
        print(f"  - ✔️ {len(data_to_insert)}개의 데이터를 DB에 성공적으로 삽입/업데이트했습니다.")
    except psycopg2.Error as e:
        print(f"❌ 데이터 삽입 오류: '{e}'")
        connection.rollback()
    finally:
        cursor.close()

# --- API 호출 관련 함수 (최종 오류 처리 강화) ---
def fetch_api_data(keyword, start_position):
    """API를 호출하여 유사사례 데이터를 가져옵니다. (JSON 파싱 오류 직접 처리)"""
    if not API_CONFIG['service_key']:
        print("❌ .env 파일에 API_SERVICE_KEY가 설정되지 않았습니다.")
        return None
    
    params = {
        "serviceKey": API_CONFIG['service_key'], "searchword": keyword, 
        "retCount": 100, "startPos": start_position,
        "target": "qna,qna_origin", "dataType": "json"
    }
    
    try:
        response = requests.get(API_CONFIG['base_url'], params=params, timeout=20)
        
        if response.status_code != 200:
            print(f"  - ❌ API 에러. 상태 코드: {response.status_code}, 내용: {response.text}")
            return None
        
        # *** 핵심 수정 부분: JSON 파싱을 try...except로 직접 감싸기 ***
        try:
            data = response.json()
        except json.JSONDecodeError:
            # API가 성공(200)했지만 내용이 비어있거나 JSON이 아니면, 데이터가 없는 것으로 간주
            return []

        if isinstance(data, list):
            return data
        
        if 'response' in data and 'body' in data['response'] and 'items' in data['response']['body']:
            return data['response']['body']['items'].get('item', [])
            
        print(f"  - ⚠️ 예상치 못한 API 응답 구조입니다: {data}")
        return []
        
    except requests.exceptions.RequestException as e:
        print(f"❌ API 요청 네트워크 오류: {e}")
        return None

# --- 메인 실행 로직 ---
def main():
    """데이터 수집 및 임베딩 프로세스를 총괄합니다."""
    if not embeddings_model:
        print("임베딩 모델이 로드되지 않아 프로그램을 종료합니다.")
        return

    db_conn = create_db_connection()
    if not db_conn:
        return

    try:
        for keyword in SEARCH_KEYWORDS:
            print(f"\n{'='*50}\n▶️ '{keyword}' 키워드에 대한 데이터 수집을 시작합니다.\n{'='*50}")
            current_pos = 1
            total_collected = 0
            while True:
                print(f" - 페이지 {current_pos // 100 + 1} (시작 위치: {current_pos}) 데이터 요청 중...")
                cases = fetch_api_data(keyword, current_pos)
                
                if cases is None:
                    print(" - API 요청에 실패하여 다음 키워드로 넘어갑니다.")
                    break
                if not cases:
                    print(" - 더 이상 데이터가 없어 이 키워드에 대한 수집을 마칩니다.")
                    break
                
                print(f"  - {len(cases)}개의 데이터를 API로부터 수신했습니다.")
                total_collected += len(cases)
                insert_minwon_data(db_conn, cases, keyword)

                current_pos += 100
                time.sleep(1)
            print(f"▶️ '{keyword}' 키워드로 총 {total_collected}개의 데이터를 처리했습니다.")
    finally:
        if db_conn:
            db_conn.close()
            print("\n✅ 데이터베이스 연결을 닫았습니다. 모든 작업이 완료되었습니다.")

if __name__ == "__main__":
    main()

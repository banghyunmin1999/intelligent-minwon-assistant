import requests
import json
import mysql.connector
from mysql.connector import Error
import time
import os
from dotenv import load_dotenv

# --- 설정 (Configuration) ---
load_dotenv()

API_CONFIG = {
    "service_key": os.getenv("API_SERVICE_KEY"), # .env 파일에 추가 필요
    "base_url": "http://apis.data.go.kr/1140100/minAnalsInfoView5/minSimilarInfo5"
}

DB_CONFIG = {
    "host": os.getenv("DB_HOST"),
    "port": int(os.getenv("DB_PORT")),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
    "database": os.getenv("DB_NAME"),
    "connection_timeout": 10  # 10초로 제한
}

SEARCH_KEYWORDS = [
    "층간소음", "불법주차", "쓰레기 무단투기", "공동주택 하자보수", "재개발 보상", "전세사기"
]

# --- 데이터베이스 관련 함수 ---
def create_db_connection():
    """MySQL 데이터베이스에 연결하고 connection 객체를 반환합니다."""
    try:
        connection = mysql.connector.connect(**DB_CONFIG)
        print("✅ MySQL 데이터베이스 연결 성공!")
        return connection
    except Error as e:
        print(f"❌ 데이터베이스 연결 오류: '{e}'")
        return None

def create_table_if_not_exists(connection):
    """'minwon_cases' 테이블이 없으면 생성합니다."""
    cursor = connection.cursor()
    try:
        create_table_query = """
        CREATE TABLE IF NOT EXISTS minwon_cases (
            create_date BIGINT PRIMARY KEY,
            title VARCHAR(500),
            content TEXT,
            institution VARCHAR(255),
            department VARCHAR(255),
            search_keyword VARCHAR(255),
            collected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
        """
        cursor.execute(create_table_query)
        connection.commit()
        print("✅ 'minwon_cases' 테이블 확인 및 준비 완료.")
    except Error as e:
        print(f"❌ 테이블 생성 오류: '{e}'")
    finally:
        cursor.close()

def insert_minwon_data(connection, cases, search_keyword):
    """수집한 민원 데이터를 데이터베이스에 삽입합니다."""
    cursor = connection.cursor()
    data_to_insert = []
    for case in cases:
        create_date_str = case.get('create_date')
        if create_date_str and create_date_str.isdigit():
            data_to_insert.append((
                int(create_date_str), 
                case.get('title'), 
                case.get('content'), 
                case.get('main_sub_name'), 
                case.get('dep_name'),
                search_keyword
            ))
        else:
            print(f"⚠️ 'create_date' 값이 유효하지 않아 다음 데이터를 건너뜁니다: {case}")
            continue
    
    if not data_to_insert:
        print("✔️ 삽입할 유효한 데이터가 없습니다.")
        return

    insert_query = """
    INSERT IGNORE INTO minwon_cases 
    (create_date, title, content, institution, department, search_keyword) 
    VALUES (%s, %s, %s, %s, %s, %s)
    """
    try:
        cursor.executemany(insert_query, data_to_insert)
        connection.commit()
        print(f"✔️ {cursor.rowcount}개의 새 데이터를 삽입했습니다.")
    except Error as e:
        print(f"❌ 데이터 삽입 오류: '{e}'")
    finally:
        cursor.close()

# --- API 호출 관련 함수 ---
def fetch_api_data(keyword, start_position):
    """API를 호출하여 유사사례 데이터를 가져옵니다."""
    if not API_CONFIG['service_key']:
        print("❌ .env 파일에 API_SERVICE_KEY가 설정되지 않았습니다.")
        return None

    params = {
        "serviceKey": API_CONFIG['service_key'],
        "searchword": keyword,
        "retCount": 100,
        "startPos": start_position,
        "target": "qna,qna_origin",
        "dataType": "json"
    }
    try:
        response = requests.get(API_CONFIG['base_url'], params=params, timeout=20)
        response.raise_for_status()
        data = response.json()
        if 'response' in data and 'body' in data['response'] and 'items' in data['response']['body']:
            return data['response']['body']['items'].get('item', [])
        elif isinstance(data, list):
            return data
        return []
    except requests.exceptions.RequestException as e:
        print(f"❌ API 요청 오류: {e}")
        return None
    except json.JSONDecodeError:
        print("❌ API 응답 JSON 파싱 오류")
        return None

# --- 메인 실행 로직 ---
def main():
    """데이터 수집 프로세스를 총괄합니다."""
    db_conn = create_db_connection()
    if not db_conn:
        return

    create_table_if_not_exists(db_conn)

    try:
        for keyword in SEARCH_KEYWORDS:
            print(f"\n{'='*50}\n▶️ '{keyword}' 키워드에 대한 데이터 수집을 시작합니다.\n{'='*50}")
            current_pos = 1
            while True:
                print(f"  - 페이지 {current_pos // 100 + 1} (시작 위치: {current_pos}) 데이터 요청 중...")
                cases = fetch_api_data(keyword, current_pos)
                if cases is None:
                    print("  - API 요청에 실패하여 다음 키워드로 넘어갑니다.")
                    break
                if not cases:
                    print("  - 더 이상 데이터가 없어 이 키워드에 대한 수집을 마칩니다.")
                    break
                insert_minwon_data(db_conn, cases, keyword)
                current_pos += 100
                time.sleep(1)
    finally:
        if db_conn and db_conn.is_connected():
            db_conn.close()
            print("\n✅ 데이터베이스 연결을 닫았습니다. 모든 작업이 완료되었습니다.")

if __name__ == "__main__":
    main()

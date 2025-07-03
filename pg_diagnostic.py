import os
import sys
import json
import requests
from dotenv import load_dotenv
import psycopg2

# --- 진단 설정 ---
# 테스트할 API 키워드와 요청 개수 (실제 수집 스크립트와 동일한 키워드 사용)
TEST_KEYWORD = "층간소음"
TEST_COUNT = 5 # 테스트를 위해 적은 수의 데이터만 요청

# --- 스크립트 시작 ---
print("\n" + "="*50)
print(" 지능형 민원 시스템 데이터 파이프라인 통합 진단 시작")
print("="*50)

# --- 1. 환경 설정 진단 ---
print("\n[1단계] 환경 변수(.env) 로딩 진단")
try:
    load_dotenv()
    # API 키 확인
    api_key = os.getenv("API_SERVICE_KEY")
    if not api_key:
        print("❌ [실패] .env 파일에서 'API_SERVICE_KEY'를 찾을 수 없습니다.")
        sys.exit()
    print("✅ 'API_SERVICE_KEY' 로딩 성공")

    # DB 정보 확인
    db_host = os.getenv("PG_HOST")
    if not db_host:
        print("❌ [실패] .env 파일에서 'PG_HOST'를 찾을 수 없습니다.")
        sys.exit()
    print("✅ PostgreSQL 접속 정보 로딩 성공")

except Exception as e:
    print(f"❌ [실패] .env 파일 로딩 중 예외 발생: {e}")
    sys.exit()


# --- 2. 공공데이터 API 호출 진단 ---
print("\n[2단계] 공공데이터 API 호출 진단")
API_CONFIG = {
    "service_key": api_key,
    "base_url": "http://apis.data.go.kr/1140100/minAnalsInfoView5/minSimilarInfo5"
}
params = {
    "serviceKey": API_CONFIG['service_key'],
    "searchword": TEST_KEYWORD,
    "retCount": TEST_COUNT,
    "startPos": 1,
    "target": "qna,qna_origin",
    "dataType": "json"
}
try:
    print(f"- '{TEST_KEYWORD}' 키워드로 API 요청 시도...")
    response = requests.get(API_CONFIG['base_url'], params=params, timeout=20)

    print(f"- HTTP 응답 코드: {response.status_code}")
    if response.status_code != 200:
        print("❌ [실패] API 서버가 200 성공 코드를 반환하지 않았습니다.")
        print("   - 원인: 서비스 키가 잘못되었거나, 서버 점검 중일 수 있습니다.")
        print(f"   - 서버 응답 내용: {response.text}")
        sys.exit()

    # JSON 파싱 시도
    data = response.json()
    print("- JSON 파싱 성공")

    # 데이터 구조 확인
    if 'response' in data and 'body' in data['response'] and 'items' in data['response']['body']:
        items = data['response']['body']['items'].get('item', [])
        print(f"- 데이터 구조 정상 확인. 수신된 데이터 개수: {len(items)}")
        if not items:
            print("⚠️ [경고] API 호출은 성공했으나, 반환된 데이터가 없습니다.")
            print("   - 원인: 해당 키워드에 대한 데이터가 없거나, API 서버의 일시적인 문제일 수 있습니다.")
        else:
            print("✅ API 호출 및 데이터 수신 성공")
            # print(f"   - 수신 데이터 샘플: {json.dumps(items[0], ensure_ascii=False, indent=2)}")
    else:
        print("❌ [실패] API 응답의 JSON 구조가 예상과 다릅니다.")
        print(f"   - 전체 응답 내용: {json.dumps(data, ensure_ascii=False, indent=2)}")
        sys.exit()

except requests.exceptions.RequestException as e:
    print(f"❌ [실패] API 요청 중 네트워크 오류 발생: {e}")
    sys.exit()
except json.JSONDecodeError:
    print("❌ [실패] API 응답이 유효한 JSON 형식이 아닙니다.")
    print(f"   - 서버 응답 내용: {response.text}")
    sys.exit()


# --- 3. 임베딩 모델 로딩 진단 ---
print("\n[3단계] 임베딩 모델 로딩 진단")
try:
    from langchain_huggingface import HuggingFaceEmbeddings
    print("- HuggingFaceEmbeddings 라이브러리 로딩 성공")
    embeddings_model = HuggingFaceEmbeddings(
        model_name="jhgan/ko-sbert-nli",
        model_kwargs={'device': 'cuda'}, # GPU 사용 시도
        encode_kwargs={'normalize_embeddings': True}
    )
    print("✅ 임베딩 모델 초기화 성공 (GPU 사용 가능)")
except ImportError:
    print("❌ [실패] 'langchain_huggingface' 라이브러리를 찾을 수 없습니다.")
    print("   - 해결: pip install langchain-huggingface sentence-transformers torch")
    sys.exit()
except Exception as e:
    # CUDA 오류 발생 시 CPU로 재시도
    if "CUDA" in str(e):
        print("⚠️ [경고] GPU(CUDA)로 모델 로딩 실패. CPU로 재시도합니다.")
        try:
            embeddings_model = HuggingFaceEmbeddings(
                model_name="jhgan/ko-sbert-nli",
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
            print("✅ 임베딩 모델 초기화 성공 (CPU 사용)")
        except Exception as e2:
            print(f"❌ [실패] CPU로도 모델 로딩에 실패했습니다: {e2}")
            sys.exit()
    else:
        print(f"❌ [실패] 임베딩 모델 로딩 중 예외 발생: {e}")
        sys.exit()


# --- 4. PostgreSQL 연결 및 상태 진단 ---
print("\n[4단계] PostgreSQL 데이터베이스 진단")
DB_CONFIG = {
    "host": os.getenv("PG_HOST", "localhost"),
    "port": os.getenv("PG_PORT", "5432"),
    "user": os.getenv("PG_USER", "minwon_user"),
    "password": os.getenv("PG_PASSWORD", "1234"),
    "dbname": os.getenv("PG_DB_NAME", "minwon_pg_db")
}
try:
    print("- DB 연결 시도...")
    connection = psycopg2.connect(**DB_CONFIG)
    cursor = connection.cursor()
    print("✅ DB 연결 성공")

    # 테이블 존재 여부 확인
    cursor.execute("SELECT to_regclass('public.minwon_cases_pg');")
    if cursor.fetchone()[0]:
        print("- 'minwon_cases_pg' 테이블 존재 확인")
    else:
        print("❌ [실패] 'minwon_cases_pg' 테이블이 데이터베이스에 존재하지 않습니다.")
        connection.close()
        sys.exit()

    # 현재 데이터 개수 확인
    cursor.execute("SELECT COUNT(*) FROM minwon_cases_pg;")
    count = cursor.fetchone()[0]
    print(f"- 현재 저장된 데이터 개수: {count}")

    connection.close()
    print("✅ 데이터베이스 상태 진단 완료")

except psycopg2.Error as e:
    print(f"❌ [실패] 데이터베이스 연결 또는 쿼리 중 오류 발생: {e}")
    sys.exit()


print("\n" + "="*50)
print(" 진단 완료. '❌ [실패]' 또는 '⚠️ [경고]' 메시지를 확인하세요.")
print("="*50)
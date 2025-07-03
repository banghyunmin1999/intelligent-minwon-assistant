# inspect_pgvector.py
import sys
try:
    from langchain_postgres.vectorstores import PGVector
    print("--- PGVector 모듈 로딩 성공 ---")
    print("아래는 현재 설치된 PGVector의 공식 사용법입니다.")
    print("="*50)
    help(PGVector)
    print("="*50)

except ImportError:
    print("❌ [오류] langchain-postgres 라이브러리를 찾을 수 없습니다.")
    print("   - 해결 방법: pip install --upgrade langchain-postgres")
except Exception as e:
    print(f"❌ [오류] 예상치 못한 오류 발생: {e}")
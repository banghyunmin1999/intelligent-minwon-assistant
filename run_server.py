import os
import sys

# 프로젝트 루트 디렉토리 추가
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# 모듈을 올바르게 임포트하도록 설정
import sys
import os

# 프로젝트 루트 디렉토리 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 프로젝트 패키지 임포트
from intelligent_minwon_assistant.api.main import app

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

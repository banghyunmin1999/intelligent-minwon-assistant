# Intelligent Minwon Assistant

Intelligent Minwon Assistant는 민원 처리를 위한 인공지능 기반의 도우미 시스템입니다.

## 프로젝트 구조

```
intelligent-minwon-assistant/
├── .env                 # 환경 변수 설정 파일
├── ai-engine/           # AI 엔진 관련 코드
├── models/             # 모델 관련 파일
├── scripts/            # 스크립트 파일
├── test.py            # 테스트 파일
└── README.md          # 프로젝트 설명
```

## 설치 및 실행

1. 환경 변수 설정
   - `.env` 파일을 생성하고 필요한 환경 변수를 설정합니다.
   - 주요 환경 변수: DB_HOST, DB_PORT 등

2. 의존성 설치
   - Python 3.8 이상이 필요합니다.
   - 필요한 패키지는 `requirements.txt`에 정의되어 있습니다.

3. llama.cpp 설치 및 빌드
   ```bash
   # 필요한 패키지들은 이미 설치되어 있습니다:
   # - build-essential
   # - cmake
   # - make
   # - libcurl

   # llama.cpp 클론 및 빌드
   git clone https://github.com/ggerganov/llama.cpp
   cd llama.cpp
   mkdir -p build
   cd build
   cmake ..
   make -j$(nproc)
   ```

4. 모델 실행
   ```bash
   # 모델 실행 명령어
   /home/bang/llama.cpp/build/bin/llama-cli --model /home/bang/intelligent-minwon-assistant/models/llama-3-Korean-Bllossom-8B-gguf-Q4_K_M/llama-3-Korean-Bllossom-8B-Q4_K_M.gguf
   ```

## 사용법

프로젝트의 주요 기능과 사용 방법을 설명합니다.

## 기술 스택

- Python
- AI/ML
- Database
- 환경 변수 관리 (python-dotenv)

## 버전 관리

- v2.1.0: 모델 실행 명령어 및 설치 가이드 추가

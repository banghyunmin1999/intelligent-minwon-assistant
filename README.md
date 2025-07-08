

# 지능형 농지민원 어시스턴트

**지능형 농지민원 답변 시스템 (Intelligent Civil Complaint Assistant)**

[](https://kubernetes.io/) [](https://www.docker.com/) [](https://fastapi.tiangolo.com/) [](https://www.java.com/) [](https://www.postgresql.org/)

## 🎯 프로젝트 개요

본 프로젝트는 전자정부 표준 프레임워크 기반의 민원 게시판 웹 애플리케이션과, RAG(검색 증강 생성) 기술을 활용한 AI 답변 서버를 연동하는 시스템입니다. 두 애플리케이션은 각각 독립적인 Docker 컨테이너로 패키징되며, 로컬 개발 환경에서는 Minikube를 사용하여 쿠버네티스 클러스터 위에 배포 및 운영됩니다.

## 🏗️ 아키텍처 (로컬 개발 환경)

데이터베이스는 호스트 PC에서 직접 실행하고, 애플리케이션 서버들만 쿠버네티스 위에서 실행합니다.


<img src="https://github.com/user-attachments/assets/ca519a87-06ea-4be7-9002-c94eeca0c62e" width="600">


## 디렉토리 구조

```
intelligent-minwon-assistant/
├── kubernetes-manifests/            # 쿠버네티스 리소스 정의
│   ├── ai-deployment.yaml           # AI 서버 Deployment (GPU 리소스 요청 포함)
│   ├── egov-deployment.yaml         # eGov 서버 Deployment (Tomcat + WAR 이미지)
│   ├── service.yaml                 # 두 서비스(NodePort 30001/30002)
│   ├── ingress.yaml                 # nginx-ingress 경로(/ai, /egov)
│   ├── config-and-secrets.yaml      # ConfigMap & Secret (DB, 모델 경로 등)
│   └── start_servers.sh             # 로컬 포트포워딩(8000/8080) 스크립트
│
├── minwon-ai-server/                # AI FastAPI 애플리케이션
│   ├── Dockerfile                   # 다단계 빌드 + CUDA 기반 이미지
│   ├── requirements.txt             # Python 의존성
│   ├── server.py                    # FastAPI 엔트리포인트
│   ├── ingestion.py                 # 임베딩/벡터화 파이프라인
│   └── .env                         # AI 서버용 예시 환경변수
│
├── egov-server/                     # 전자정부프레임워크(Spring) 애플리케이션
│   ├── Dockerfile                   # Tomcat 기반 이미지 빌드
│   ├── ROOT.war                     # 빌드 산출물
│   ├── pom.xml                      # Maven 설정
│   └── src/                         # Java 소스 코드
└── README.md
```

## 🛠️ 기술 스택

| 구분 | 기술 |
| :--- | :--- |
| **AI 서버** | Python, FastAPI, LangChain, Llama.cpp |
| **웹 서버** | Java 8, 전자정부 표준 프레임워크(Spring), Tomcat |
| **데이터베이스** | PostgreSQL, MySQL |
| **인프라/DevOps**| Docker, Minikube, Kubernetes |

## 🚀 로컬 환경 구축 및 실행 가이드

#### **1단계: 사전 준비**

1.  **필수 프로그램 설치**: `Docker Desktop`, `WSL2`, `Minikube`, `kubectl`을 PC에 설치합니다.
2.  **GPU 드라이버**: 최신 NVIDIA 드라이버를 설치합니다.
3.  **데이터베이스 실행**: PC(호스트 또는 WSL)에 **PostgreSQL**과 **MySQL**을 설치하고 실행합니다.
      * PostgreSQL에는 `CREATE EXTENSION IF NOT EXISTS vector;` 명령으로 pgvector 확장을 활성화해야 합니다.

#### **2단계: 데이터 인덱싱**

AI가 참고할 데이터를 벡터DB에 저장합니다.

```bash
# minwon-ai-server 폴더로 이동
cd minwon-ai-server
# 필요한 라이브러리 설치
pip install -r requirements.txt
# 데이터 인덱싱 스크립트 실행
python ingestion.py
```

#### **3단계: Docker 이미지 빌드 및 푸시**

두 개의 서버를 각각 Docker 이미지로 만들고 Docker Hub에 업로드합니다. (`<ID>`는 본인의 Docker Hub ID로 변경)

```bash
# 1. AI 서버 이미지 빌드 및 푸시
cd ~/intelligent-minwon-assistant/minwon-ai-server
docker build -t <ID>/minwon-ai-server:v1 .
docker push <ID>/minwon-ai-server:v1

# 2. 전자정부 서버 이미지 빌드 및 푸시
cd ~/intelligent-minwon-assistant/egov-project
docker build -t <ID>/egov-server:v1 .
docker push <ID>/egov-server:v1
```

#### **4단계: Minikube 클러스터 시작 및 설정**

GPU를 사용하는 쿠버네티스 클러스터를 시작하고, 외부 접속을 위한 설정을 합니다.

```bash
# 1. Minikube 클러스터 시작 (GPU 활성화)
minikube start --driver=docker --gpus=all --cpus=4 --memory=8192

# 2. NVIDIA Device Plugin 설치 (GPU 인식용)
kubectl create -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/v0.14.1/nvidia-device-plugin.yml

# 3. Ingress Controller 애드온 활성화
minikube addons enable ingress
```

#### **5단계: 쿠버네티스에 애플리케이션 배포**

작성된 YAML 파일들을 사용하여 클러스터에 두 서버를 배포합니다.

```bash
# kubernetes-manifests 폴더로 이동
cd ~/intelligent-minwon-assistant/kubernetes-manifests

# 모든 YAML 파일 적용
kubectl apply -f .
```

#### **5단계: 상태 확인 및 애플리케이션 접속**

`kubectl apply`로 배포를 완료한 후, 아래 절차에 따라 파드들의 상태를 확인하고, 포트포워딩을 통해 외부에서 애플리케이션에 접속합니다.

 **1. 파드(Pod) 상태 확인**

아래 명령어로 두 개의 애플리케이션 파드가 생성되고 `Running` 상태로 전환되는 것을 실시간으로 확인합니다.

```bash
kubectl get pods -w
```

> **[확인]** `ai-server-deployment-...`와 `egov-server-deployment-...` 두 파드가 모두 `STATUS` `Running`, `READY` `1/1`이 될 때까지 기다립니다.

 **2. 로컬 포트포워딩 실행**

`start_servers.sh` 스크립트를 실행하여, 로컬 PC의 포트와 쿠버네티스 파드의 포트를 직접 연결합니다.

1.  **별도의 새 터미널**을 엽니다.
2.  아래 명령어를 실행합니다.
    ```bash
    cd kubernetes-manifests
    ./start_servers.sh
    ```

> **[중요]** 이 스크립트가 실행 중인 터미널은 접속하는 동안 **계속 켜두어야 합니다.**

 **3. 웹 브라우저 및 API 클라이언트로 최종 접속**

이제 `localhost`를 통해 각 서버에 직접 접속할 수 있습니다.

  * **전자정부 서버 접속**:

      * 웹 브라우저를 열고 주소창에 **`http://localhost:8080/main.do`** 를 입력합니다.

  * **AI 서버 API 테스트**:

      * API 테스트 도구(Postman, curl 등)를 사용하여 **`http://localhost:8000/docs`** 로 테스트를 진행합니다.

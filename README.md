

# 지능형 농지민원 어시스턴트

**지능형 농지민원 답변 시스템 (Intelligent Civil Complaint Assistant)**

[](https://kubernetes.io/) [](https://www.docker.com/) [](https://fastapi.tiangolo.com/) [](https://www.java.com/) [](https://www.postgresql.org/)

## 🎯 프로젝트 개요

본 프로젝트는 전자정부 표준 프레임워크 기반의 민원 게시판 웹 애플리케이션과, RAG(검색 증강 생성) 기술을 활용한 AI 답변 서버를 연동하는 시스템입니다. 두 애플리케이션은 각각 독립적인 Docker 컨테이너로 패키징되며, 로컬 개발 환경에서는 Minikube를 사용하여 쿠버네티스 클러스터 위에 배포 및 운영됩니다.

## 🏗️ 아키텍처 (로컬 개발 환경)

데이터베이스는 호스트 PC에서 직접 실행하고, 애플리케이션 서버들만 쿠버네티스 위에서 실행합니다.

```
[ Developer's PC (Windows + WSL2) ]
+------------------------------------------------------+
| [ PostgreSQL ]      [ MySQL ]                        |  <-- DB는 PC(호스트)에서 직접 실행
+------------------------------------------------------+
       ^                                  ^
       | (host.minikube.internal)         | (host.minikube.internal)
       |                                  |
+------------------------------------------------------+
| [ Minikube Kubernetes Node (Docker 기반 VM) ]          |
|                                                      |
|   +-------------------+  <---------->  +-------------------------+
|   |  Pod: egov-server |  (내부 서비스명)    |  Pod: ai-server         |
|   |  (Java/Tomcat)   |                  |  (Python/FastAPI/GPU)   |
|   +-------------------+                  +-------------------------+
|          ^
|          | Ingress Controller (minwon.local)
|          |
+----------+-------------------------------------------+
           |
[ 개발자 / 웹 브라우저 ]
```

## 디렉토리 구조

```
intelligent-minwon-assistant/
├── egov-project/                  # 전자정부프레임워크 프로젝트
│   ├── Dockerfile
│   ├── pom.xml
│   ├── docker.globals.properties  # (선택) Docker 볼륨 마운트용 설정
│   └── src/
├── minwon-ai-server/              # AI 서버 프로젝트
│   ├── Dockerfile
│   ├── .env
│   ├── ingestion.py
│   ├── server.py
│   └── requirements.txt
├── kubernetes-manifests/          # 쿠버네티스 YAML 파일
│   ├── config-and-secrets.yaml
│   ├── deployment.yaml
│   └── network.yaml
└── README.md
```

## 🛠️ 기술 스택

| 구분 | 기술 |
| :--- | :--- |
| **AI 서버** | Python, FastAPI, LangChain, Llama.cpp |
| **웹 서버** | Java 8, 전자정부 표준 프레임워크(Spring), Tomcat |
| **데이터베이스** | PostgreSQL (with PGVector), MySQL |
| **인프라/DevOps**| Docker, Minikube, Kubernetes, NGINX Ingress |

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

#### **6단계: 상태 확인 및 접속**

1.  **파드 상태 확인**: 모든 파드가 `Running` 상태가 될 때까지 기다립니다.
    ```bash
    kubectl get pods -w
    ```
2.  **`hosts` 파일 설정**:
      * `minikube ip` 명령어로 클러스터 IP를 확인합니다.
      * Windows의 `C:\Windows\System32\drivers\etc\hosts` 파일을 관리자 권한으로 열어, 맨 아래에 `<minikube_ip> minwon.local` 한 줄을 추가합니다. (예: `192.168.49.2 minwon.local`)
3.  **터널 실행**: **별도의 새 터미널**을 열고 아래 명령어를 실행합니다. **이 터미널은 접속하는 동안 계속 켜두어야 합니다.**
    ```bash
    minikube tunnel
    ```
4.  **최종 접속**: 웹 브라우저에서 `http://minwon.local` 주소로 접속합니다.

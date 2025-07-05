네, 알겠습니다. `intelligent-minwon-assistant` 프로젝트의 시작을 위해 `README.md` 파일을 생성해 드리겠습니다. 이 파일은 프로젝트의 목표, 아키텍처, 기술 스택, 그리고 시작 방법을 명확하게 설명하는 것을 목표로 합니다.

-----

# 지능형 민원 처리 어시스턴트 (Intelligent Minwon Assistant)

[](https://www.gitops.tech/) [](https://kubernetes.io/) [](https://github.com/features/actions) [](https://argoproj.github.io/argo-cd/)

이 프로젝트는 전자정부프레임워크 기반의 민원 게시판과 지능형 AI 엔진을 연동하여, 민원 처리를 자동화하고 효율화하는 것을 목표로 합니다. 모든 인프라와 애플리케이션 배포는 **GitOps 철학**에 기반하여 완전 자동으로 관리됩니다.

## 📌 주요 기능

  * **민원 게시판**: 전자정부 표준프레임워크로 개발된 웹 기반 민원 접수 및 조회 시스템.
  * **AI 어시스턴트**: 접수된 민원의 내용을 분석하고, 관련 규정이나 답변을 추천하는 FastAPI 기반 AI 엔진.
  * **완전 자동화된 배포**: `Git`에 코드를 푸시하는 것만으로 테스트, 빌드, 배포가 자동으로 이루어지는 CI/CD 파이프라인.

-----

## 🏗️ 아키텍처

본 프로젝트는 Git 저장소를 '단일 진실 공급원(Single Source of Truth)'으로 사용하는 GitOps 방식을 따릅니다. 개발자가 코드를 변경하면, 인프라와 애플리케이션이 자동으로, 그리고 예측 가능하게 업데이트됩니다.

**배포 흐름:**
`개발자 코드 Push` ➡️ `Git 저장소` ➡️ `① GitHub Actions (CI)` ➡️ `② 컨테이너 레지스트리` ➡️ `③ Argo CD (CD)` ➡️ `④ 쿠버네티스 클러스터`

1.  **CI (GitHub Actions)**: 코드 변경을 감지하여 Docker 이미지를 빌드하고 레지스트리에 푸시합니다.
2.  **CD (Argo CD)**: Git 저장소의 Kubernetes 설정 파일 변경을 감지하고, 클러스터의 상태를 Git에 정의된 상태와 자동으로 일치시킵니다.

-----

## 🛠️ 기술 스택 (Tech Stack)

모든 컴포넌트는 비용 부담이 없는 오픈소스 및 자체 구축 가능한(Self-hosted) 도구로 구성됩니다.

| 구분                  | 기술                                         | 설명                                   |
| --------------------- | -------------------------------------------- | -------------------------------------- |
| **애플리케이션** | FastAPI, e-Gov Framework (Spring/Java)       | AI 엔진 및 민원 게시판 서버            |
| **데이터베이스** | PostgreSQL                                   | 민원 데이터 저장                       |
| **컨테이너** | Docker                                       | 애플리케이션 컨테이너화                |
| **오케스트레이션** | Kubernetes (Minikube, k3s)                   | 컨테이너 관리 및 오케스트레이션        |
| **CI/CD** | GitHub Actions, Argo CD                      | 지속적 통합 및 배포 (GitOps)           |
| **인프라 설정 관리** | Kustomize                                    | 환경별 Kubernetes 설정 관리            |
| **이미지 레지스트리** | Harbor, GitHub Packages (GHCR)               | Docker 이미지 저장소 (자체 구축)       |
| **파일 스토리지** | MinIO                                        | AI 모델 등 대용량 파일 저장소 (자체 구축) |
| **모니터링** | Prometheus, Grafana                          | 시스템 메트릭 수집 및 시각화           |

-----

## 📁 디렉토리 구조

```
intelligent-minwon-assistant/
├── ai-engine/
│   └── Dockerfile              # ✅ AI 서버 Dockerfile (완료)
├── egov-server/
│   ├── src/                    # 📋 전자정부프레임워크 소스
│   ├── pom.xml                 # (또는 build.gradle)
│   └── Dockerfile              # 📝 전자정부 서버 Dockerfile (생성 예정)
├── k8s/
│   ├── deployment.yaml         # 🚀 쿠버네티스 배포 설정 (생성 예정)
│   └── ingress.yaml            # 🌐 외부 접속 설정 (생성 예정)
├── server.py                   # ✅ AI 서버 파이썬 코드 (완료)
└── requirements.txt            # ✅ AI 서버 파이썬 의존성 (완료)
```

-----

## ⚙️ 시작하기 (Getting Started)

### 사전 요구사항

  * Git
  * Docker
  * Kubernetes (로컬 환경에서는 **Minikube** 또는 **k3s** 사용을 권장합니다.)
  * kubectl

### 로컬 환경 배포 절차 (요약)

1.  **리포지토리 클론:**

    ```bash
    git clone [저장소 URL]
    cd intelligent-minwon-assistant
    ```

2.  **쿠버네티스 클러스터 실행:**

      * Minikube, k3s 등 로컬 쿠버네티스 클러스터를 시작합니다.

3.  **인프라 서비스 배포:**

      * 클러스터에 Argo CD, MinIO, Harbor 등 기본 인프라를 배포합니다. (Helm 또는 Kustomize 사용)

4.  **Argo CD 설정:**

      * Argo CD가 이 Git 리포지토리의 `k8s/` 디렉토리를 바라보도록 Application을 생성합니다.

5.  **자동 동기화:**

      * Argo CD가 리포지토리의 내용을 클러스터에 자동으로 동기화하며 모든 애플리케이션(AI 엔진, 민원 게시판)을 배포합니다. 이후 `git push`만으로 모든 변경사항이 자동으로 반영됩니다.
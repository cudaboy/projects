# 🤖 Korean Conversation Chatbot (Transformer)

  📅 문서 버전: 본 README는 2026년 3월 16일 기준으로 작성 및 업데이트되었습니다.

이 프로젝트는 기존의 Transformer 기반 한국어 챗봇에 RAG(Retrieval-Augmented Generation) 파이프라인을 결합하여, 외부 지식(Chroma Vector DB)을 참조해 더욱 정확하고 풍부한 실시간 대화 서비스를 제공하는 고도화된 챗봇 시스템입니다.

백엔드(FastAPI)와 프론트엔드(Streamlit)가 완벽하게 분리된 마이크로서비스 아키텍처를 채택하였습니다.

<br>

# ✔️ Tech Stack (개발 환경)

- **Deep Learning Framework**
  - ![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)

- **RAG & NLP Ecosystem**
  - ![LangChain](https://img.shields.io/badge/LangChain-1C3C3C?style=for-the-badge&logo=langchain&logoColor=white)
  - ![Chroma](https://img.shields.io/badge/Chroma_DB-FF6F00?style=for-the-badge&logo=chroma&logoColor=white)
  - ![HuggingFace](https://img.shields.io/badge/HuggingFace-F9AB00?style=for-the-badge&logo=huggingface&logoColor=white)

- **API Server**
  - ![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)
  - ![Uvicorn](https://img.shields.io/badge/Uvicorn-499848?style=for-the-badge)

- **Frontend UI**
  - ![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)

- **DevOps & Environment**
  - ![Linux](https://img.shields.io/badge/Linux-FCC624?style=for-the-badge&logo=linux&logoColor=black)
  - ![Docker](https://img.shields.io/badge/docker-%230db7ed.svg?style=for-the-badge&logo=docker&logoColor=white)
  - ![Minikube](https://img.shields.io/badge/Minikube-326CE5?style=for-the-badge&logo=kubernetes&logoColor=white)

<br>

# 📊 Data Source (데이터 출처)

본 프로젝트의 학습에는 **AI 허브**(AI Hub)에서 제공하는 공공 데이터를 활용하였습니다.

- 데이터셋 명: 소상공인 고객 주문 질의-응답 데이터

- 제공처: [AI 허브 바로가기](https://aihub.or.kr/aihubdata/data/view.do?dataSetSn=116)

- 데이터 특징:

  음식점(중식, 카페, 한식 등) 및 소상공인 업종에서 발생하는 고객의 질문과 그에 대한 실제 응답 대화로 구성되어 있습니다.

  MQ(질문), SA(답변) 등의 구조화된 엑셀 데이터를 정제하여 자연스러운 대화 모델링을 수행했습니다.

<br>

# 📂 Project Structure

프로젝트의 주요 파일 구성과 역할은 다음과 같습니다.
```
 .
 │
 ├── 🚀 Orchestration
 │   └── app.py                  # 백엔드/프론트엔드 동시 구동 지휘자 (Entry Point)
 │
 ├── ⚙️ Backend & AI Core
 │   ├── chatbot_api.py          # FastAPI 서버 (Port 8000)
 │   └── rag_handler.py          # 모델 로드 및 LangChain RAG 체인 조립
 │
 ├── 🖥️ Frontend
 │   ├── main.py                 # Streamlit 채팅 메인 화면 (Port 8501)
 │   └── sidebar.py              # 설정 및 대화 다운로드 UI
 │
 ├── 🧠 Model & Data Assets
 │   ├── model.py                # Transformer 아키텍처 클래스
 │   ├── transformer_best.pth    # 파인튜닝 완료된 챗봇 가중치 파일
 │   └── rag_db/                 # 로컬 Chroma Vector DB 폴더
 │
 ├── 🛠️ Utils & Logs
 │   ├── utils.py                # 시스템 로깅 설정
 │   └── logs/                   # 일별 로그 저장 폴더
 │
 └── 🐳 Deployment
     ├── requirements.txt        # Python 패키지 의존성 목록
     ├── .dockerignore           # Docker 빌드 제외 목록
     ├── Dockerfile              # Docker 이미지 생성 명세서
     └── k8s/                    # Kubernetes 배포 설정
         ├── deployment.yaml     # Pod 관리 및 배포 설정
         └── service.yaml        # 네트워크 노출 설정 (NodePort 30501)
```

<br>

# ✨ Key Features (주요 기능)

- RAG (Retrieval-Augmented Generation) 결합

  사용자의 질문과 가장 유사한 외부 문서 문맥을 Chroma DB에서 실시간으로 검색하여 Transformer 모델에 제공합니다.

- Subword Tokenizer 디코딩 최적화

  `#` 기호가 생성되는 WordPiece 토크나이저의 한계를 극복하기 위해, 예측 ID를 배열로 모아 한 번에 디코딩(`skip_special_tokens=True`)하여 깔끔한 한국어를 출력합니다.

- 완벽한 MSA(마이크로서비스) 분리

  UI 렌더링을 담당하는 프론트엔드(`main.py`)와 AI 추론을 담당하는 백엔드(`chatbot_api.py`)를 분리하여 확장성과 유지보수성을 극대화했습니다.

- 통합 실행기 (`app.py`)

  단 한 줄의 명령어로 FastAPI와 Streamlit을 동시에 백그라운드에서 구동하고 안전하게 종료(Graceful Shutdown)합니다.

<br>

# 🛠 Data Pipeline & Preprocessing

다양한 형태의 엑셀 데이터를 정제하여 고품질의 학습 데이터를 추출합니다.

- 데이터 추출 로직

  1. 발화 컬럼 병합: raw data에 흩어져 있는 세부 발화 컬럼(`MQ`, `SQ`, `UA`, `SA`)의 텍스트를 하나의 발화 열(Turn_Text)로 통합하여 데이터의 결측을 보완합니다.

  2. 멀티턴(Multi-turn) 기반 세션 그룹화: 대화 순번(`MAIN` 컬럼)을 기준으로 개별 전화 통화 세션을 식별합니다.

  3. 문맥 유지 Q&A 구성: 화자(`SPEAKER`)를 기준으로 고객의 꼬리를 무는 연속된 발화는 하나의 '질문'으로 누적 병합하고, 이에 대한 점원의 연속된 대답은 '답변'으로 묶어내어 실제 대화 흐름이 온전히 반영된 Q&A 쌍을 구성합니다.

- 데이터 필터링:
  1. 결측치 및 예외 데이터 제거: 병합된 텍스트 데이터가 `NULL`(결측치)이거나, 숫자로만 구성된 무의미한 행을 사전에 제외하여 학습 에러를 방지합니다.

  2. 민감 정보 필터링: 개인정보(전화번호, 주소 등) 마스킹 처리를 위해 `#` 기호가 포함된 질문-답변 쌍은 노이즈 방지를 위해 최종 분석 및 학습 데이터에서 완전히 제외합니다.

<br>

# 🧠 Model Architecture

Google의 Transformer 모델을 기반으로 구현되었습니다.

- Encoder/Decoder: 다중 레이어 구조를 통한 복잡한 문맥 파악

- Multi-Head Attention: 문장 내 단어 간의 관계를 여러 관점에서 병렬로 학습

- Positional Encoding: 단어의 순서 정보를 모델에 전달

- Hyperparameters: ```D_MODEL=256, NUM_HEADS=8, DFF=512, NUM_LAYERS=2```

<br>

# 💻 Getting Started (Locl Environment)

1. Requirements
  Python 3.11 이상의 환경을 권장합니다. 필수 패키지를 설치합니다.

    ```Bash
    pip install -r requirements.txt
    ```

2. Run the Server (통합 구동)
  프로젝트 루트 폴더에서 아래 명령어를 실행하면, 백엔드(8000)와 프론트엔드(8501)가 동시에 구동됩니다.

    ```Bash
    python app.py
    ```
    - 챗봇 UI 접속: http://localhost:8501

    - FastAPI API 문서 (Swagger): http://localhost:8000/docs


<br>

# 🐳 Docker Deployment

로컬 환경을 완벽하게 컨테이너화하여 일관된 실행 환경을 보장합니다. `.dockerignore`를 통해 불필요한 파일 로드를 방지합니다.

1. Docker Image Build
  Linux 터미널에서 Dockerfile이 위치한 디렉토리로 이동한 후, 아래 명령어를 통해 이미지를 빌드합니다.
  
    (PyTorch와 가중치 파일의 용량 때문에 시간이 조금 걸릴 수 있습니다.)
  
    ```bash
    docker build -t transformer-chatbot:v3 .
    ```

2. Docker Container Run
  포트 8000(API)과 8501(UI)을 모두 개방하여 백그라운드(`-d`)에서 실행합니다.
  
    ```bash
    docker run -d --name chatbot-v3 -p 8000:8000 -p 8501:8501 transformer-chatbot:v3
    ```
      👉 실행 후 브라우저에서 http://localhost:8501로 접속하세요.

<br>

# ☸️ Kubernetes (Minikube) Deployment

로컬 도커 이미지를 활용하여 Kubernetes 클러스터에 컨테이너 오케스트레이션을 구성할 수 있습니다.

1. Minikube 도커 환경 연결 및 이미지 빌드

    ```bash
    eval $(minikube docker-env)

    docker build -t transformer-chatbot:v3 .
    ```

2. Kubernetes 리소스 적용 (`k8s/deployment.yaml`, `k8s/service.yaml`)

    ```bash
    kubectl apply -f deployment.yaml

    kubectl apply -f service.yaml
    ```

3. 서비스 노출 및 접속 (NodePort 활용)

    ```bash
    minikube service chatbot-ui-service
    ```

<br>

# 📝 Logging & Monitoring

시스템의 안정적인 운영을 위해 `logs/` 폴더 내에 일별 로그를 기록합니다.

- 요청 기록: 모든 사용자의 질문(`USER_REQUEST`)과 챗봇의 답변(`BOT_RESPONSE`)을 모두 기록합니다.

- 에러 추적: FastAPI 내부 에러나 서버 구동 실패 원인 등이 시간대별로 정확하게 기록됩니다.

<br>

# ⚠️ Troubleshooting

개발 및 운영 중 발생할 수 있는 주요 이슈와 해결 방법입니다.

- Size Mismatch: 모델 로드 시 레이어 크기가 다르면 ```VOCAB_SIZE```를 가중치 파일의 기준($8,096$ 등)에 맞게 조정해야 합니다.

- TypeError: 전처리 중 숫자가 섞여 있다면 ```astype(str)```을 통해 문자열로 강제 변환 후 처리합니다.

<br>

# 💡 Future Work (향후 추가 및 개선 계획)

본 프로젝트는 기본적인 챗봇 기능 구현 이후, 대화 품질 향상 및 모델 세밀 조정을 위해 다음과 같은 업데이트를 계획하고 있습니다.

- 모델 추론 로직(Inference) 개선

  현재의 Greedy Search 방식을 보완하여, 답변의 다양성 및 창의성을 제어할 수 있는 확률적 샘플링(Temperature, Top-K) 파라미터 조절 UI 도입 및 평가(`evaluate`) 로직 수정
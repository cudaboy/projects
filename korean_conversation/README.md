# 🤖 Korean Conversation Chatbot (Transformer)

  📅 문서 버전: 본 README는 2026년 3월 10일 기준으로 작성 및 업데이트되었습니다.

이 프로젝트 실습은 Transformer 아키텍처를 활용하여 한국어 대화 데이터를 학습하고, FastAPI를 통해 실시간 대화 서비스를 제공하는 챗봇 시스템입니다.

<br>

# ✔️ Tech Stack (개발 환경)

- **Deep Learning Framework**
  - ![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)

- **API Server**
  - ![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)
  - ![Uvicorn](https://img.shields.io/badge/Uvicorn-499848?style=for-the-badge)

- **Frontend UI**
  - ![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)

- **Environment**
  - ![Linux](https://img.shields.io/badge/Linux-FCC624?style=for-the-badge&logo=linux&logoColor=black)
  - ![Docker](https://img.shields.io/badge/docker-%230db7ed.svg?style=for-the-badge&logo=docker&logoColor=white)

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
├── logs/                     # 생성된 로그 파일들이 저장되는 폴더
│   ├── chatbot_2026-03-06.log # 일자별 대화(USER/BOT) 및 에러 기록
│   └ ...
├── app.py                    # Streamlit 웹 UI 앱의 메인 진입점 (실행 파일)
├── main.py                   # Streamlit 채팅 화면 UI 구성 및 대화 상태(Session) 관리
├── sidebar.py                # Streamlit 사이드바 UI (최대 길이 설정 및 대화 초기화)
├── model_handler.py          # 모델 캐싱(@st.cache_resource) 로드 및 텍스트 생성 추론 로직
├── chatbot_api.py            # FastAPI 기반 챗봇 백엔드 API 서버 구동 스크립트
├── model.py                  # PyTorch 기반 Transformer 모델 아키텍처(Encoder/Decoder) 정의
├── utils.py                  # 파일 및 콘솔 로깅 설정 등 공통 유틸리티 함수
├── requirements.txt          # 프로젝트 실행에 필요한 파이썬 라이브러리 목록 (의존성)
├── ko-con.pth                # 학습이 완료된 모델의 가중치(Weight) 파일
├── ko-con_tokenizer.subwords # 텍스트 인코딩/디코딩을 위한 단어 사전 데이터
└── training_260306.ipynb     # 데이터 정제 및 모델 학습 프로세스를 담은 Jupyter Notebook
```

<br>

# 🛠 Data Pipeline & Preprocessing

다양한 형태의 엑셀 데이터를 정제하여 고품질의 학습 데이터를 추출합니다.

- 데이터 추출 로직:

  raw data에서 MQ(질문)와 다음 행의 SA(답변)가 연결되는 답변인 경우 질문-답변 쌍을 구성합니다.

- 데이터 필터링:
  질문 또는 답변이 **NULL**(결측치)인 경우 삭제합니다.

  데이터 타입이 **정수**(int)로 인식된 행을 제외하여 전처리 에러를 방지합니다.

  데이터 중 개인정보 masking을 위해 '#' 문자가 포함된 문장은 분석에서 제외합니다.

<br>

# 🧠 Model Architecture

Google의 Transformer 모델을 기반으로 구현되었습니다.

- Encoder/Decoder: 다중 레이어 구조를 통한 복잡한 문맥 파악

- Multi-Head Attention: 문장 내 단어 간의 관계를 여러 관점에서 병렬로 학습

- Positional Encoding: 단어의 순서 정보를 모델에 전달

- Hyperparameters: ```D_MODEL=256, NUM_HEADS=8, DFF=512, NUM_LAYERS=2```

<br>

# 🚀 API Usage (FastAPI)

학습된 모델을 기반으로 실시간 대화 API를 제공합니다.

- 서버 실행
```uvicorn chatbot_api:app --reload```

- API Endpoint (POST)

URL: ```http://127.0.0.1:8000/chat/```

- Request Body:
  ```
  {
    "question": "오늘 점심 메뉴 추천해줘"
  }
  ```

- Response:
  ```
  {
    "question": "오늘 점심 메뉴 추천해줘",
    "answer": "따끈한 김치찌개 어떠신가요?"
  }
  ```

<br>

# 🚀 Web UI Usage (Streamlit)

사용자가 더 직관적으로 챗봇과 대화할 수 있도록 Streamlit 기반의 인터랙티브 웹 인터페이스를 제공합니다.

- Web App 실행:
  ```
  streamlit run app.py
  ```

- 접속 주소:
  브라우저에서 http://localhost:8501로 접속합니다.

  (Docker 컨테이너로 실행 시 -p 8501:8501 포트 매핑 옵션을 반드시 추가해 주세요.)

- 주요 기능:
  실시간 채팅: 웹 화면에서 챗봇과 즉각적으로 대화할 수 있습니다.
  
  사이드바 설정: 챗봇의 최대 답변 길이(Max Length)를 실시간으로 조절할 수 있습니다.
  
  대화 초기화: 기존 대화 기록을 지우고 새로운 세션을 시작할 수 있습니다.

<br>

# 🚀 Docker 빌드 및 실행 가이드

1. 이미지 빌드하기
  WSL 터미널에서 Dockerfile이 위치한 디렉토리로 이동한 후, 아래 명령어를 통해 이미지를 빌드합니다.
  
    (PyTorch와 가중치 파일의 용량 때문에 시간이 조금 걸릴 수 있습니다.)
  
    ```
    docker build -t transformer-chatbot:v1 .
    ```

2. 컨테이너 실행하기
  앞서 만든 이미지 하나로 두 가지 방식을 모두 실행할 수 있습니다.
  
    옵션 A: Streamlit 웹 UI 실행 (기본값)
      Streamlit을 실행하려면 8501 포트를 연결합니다.
  
    ```
    docker run -d --name chatbot-ui -p 8501:8501 transformer-chatbot:v1
    ```
      👉 실행 후 브라우저에서 http://localhost:8501로 접속하세요.

    옵션 B: FastAPI 백엔드 서버 실행
      FastAPI 서버로 실행하고 싶다면, docker run 명령어의 맨 끝에 실행 명령어를 덮어씌워 줍니다.
      
      포트는 `8000`을 연결합니다.

    ```
    docker run -d --name chatbot-api -p 8000:8000 transformer-chatbot:v1 uvicorn chatbot_api:app --host 0.0.0.0 --port 8000
    ```
      👉 실행 후 브라우저에서 http://localhost:8000/docs로 접속하여 API를 테스트하세요.

<br>

# 📝 Logging & Monitoring

시스템의 안정적인 운영을 위해 logs/ 폴더 내에 일별 로그를 기록합니다.

- 요청 기록: 모든 사용자의 질문(USER_REQUEST)과 챗봇의 답변(BOT_RESPONSE)을 기록합니다.

- 에러 추적: 시스템 오류 발생 시 상세 내용을 기록하여 즉각적인 디버깅이 가능합니다.

<br>

# ⚠️ Troubleshooting

개발 및 운영 중 발생할 수 있는 주요 이슈와 해결 방법입니다.

- Size Mismatch: 모델 로드 시 레이어 크기가 다르면 ```VOCAB_SIZE```를 가중치 파일의 기준($8,096$ 등)에 맞게 조정해야 합니다.

- TypeError: 전처리 중 숫자가 섞여 있다면 ```astype(str)```을 통해 문자열로 강제 변환 후 처리합니다.

<br>

# 💡 Future Work (향후 추가 및 개선 계획)

본 프로젝트는 기본적인 챗봇 기능 구현 이후, 사용자 편의성 및 운영 환경 안정화를 위해 다음과 같은 업데이트를 계획하고 있습니다.

- Streamlit UI 확장 (sidebar.py)

  대화 내역 다운로드(Export Chat) 기능 추가

  답변의 다양성 및 창의성을 제어할 수 있는 확률적 샘플링(Temperature, Top-K) 파라미터 조절 UI 도입

- Kubernetes (Minikube) 배포 파이프라인 완성

  로컬 Docker 이미지를 활용한 컨테이너 오케스트레이션 구성

  deployment.yaml 및 service.yaml을 통한 Pod 스케일링 및 NodePort 네트워크 연결 테스트 진행
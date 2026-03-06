# 🤖 Korean Conversation Chatbot (Transformer)

이 프로젝트 실습은 Transformer 아키텍처를 활용하여 한국어 대화 데이터를 학습하고, FastAPI를 통해 실시간 대화 서비스를 제공하는 챗봇 시스템입니다.

<br>

# 📊 Data Source (데이터 출처)

본 프로젝트의 학습에는 **AI 허브(AI Hub)**에서 제공하는 공공 데이터를 활용하였습니다.

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
│   ├── chatbot_2026-03-06.log # 오늘 날짜의 대화 및 에러 기록
│   └ ...
├── chatbot_api.py            # FastAPI 서버 실행 및 모델 배포 스크립트
├── model.py                  # Transformer 모델 아키텍처 정의 클래스
├── ko-con.pth                # 학습된 모델의 가중치(Weight) 파일
├── ko-con_tokenizer.subwords # 텍스트 인코딩을 위한 단어 사전 데이터
├── 20260306-1.ipynb          # 데이터 정제 및 모델 학습 프로세스 정리 notebook
└── 20260306-2.ipynb          # 모델 성능 테스트 및 추론 실험 notebook
```

<br>

# 🛠 Data Pipeline & Preprocessing

다양한 형태의 엑셀 데이터를 정제하여 고품질의 학습 데이터를 추출합니다.

- 데이터 추출 로직:

MQ(질문)와 다음 행의 SA(답변)가 쌍인 경우 ```shift(-1)``` 연산을 통해 질문-답변 쌍을 구성합니다.

- 데이터 필터링:
질문 또는 답변이 **NULL(결측치)**인 경우 삭제합니다.

데이터 타입이 **정수(int)**로 인식된 행을 제외하여 전처리 에러를 방지합니다.

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

# 📝 Logging & Monitoring

시스템의 안정적인 운영을 위해 logs/ 폴더 내에 일별 로그를 기록합니다.

- 요청 기록: 모든 사용자의 질문(USER_REQUEST)과 챗봇의 답변(BOT_RESPONSE)을 기록합니다.

- 에러 추적: 시스템 오류 발생 시 상세 내용을 기록하여 즉각적인 디버깅이 가능합니다.

<br>

# ⚠️ Troubleshooting

개발 및 운영 중 발생할 수 있는 주요 이슈와 해결 방법입니다.

- Size Mismatch: 모델 로드 시 레이어 크기가 다르면 ```VOCAB_SIZE```를 가중치 파일의 기준($8,096$ 등)에 맞게 조정해야 합니다.

- TypeError: 전처리 중 숫자가 섞여 있다면 ```astype(str)```을 통해 문자열로 강제 변환 후 처리합니다.
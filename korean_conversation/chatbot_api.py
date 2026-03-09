from fastapi import FastAPI
from pydantic import BaseModel
import torch
import re
import tensorflow_datasets as tfds

# [모듈 분리 반영] 
# 이전 단계에서 정의한 Transformer 클래스와 방금 분리한 로깅 유틸리티를 불러옵니다.
from model import Transformer 
from utils import setup_logger

# FastAPI 앱 객체 생성
app = FastAPI()

# 1. 로깅(Logging) 설정 (utils.py에서 불러와 한 줄로 깔끔하게 처리)
logger = setup_logger()

# 2. 데이터 모델 및 전역 설정
class ChatRequest(BaseModel):
    """API 요청 바디를 정의하는 Pydantic 모델입니다."""
    question: str

# 하이퍼파라미터 및 장치 설정
MAX_LENGTH = 40
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 토크나이저 로드 (학습 시 사용한 SubwordTextEncoder)
tokenizer = tfds.deprecated.text.SubwordTextEncoder.load_from_file("./ko-con_tokenizer")

# 시작 토큰(SOS)과 종료 토큰(EOS) 인덱스
START_TOKEN = [tokenizer.vocab_size]
END_TOKEN = [tokenizer.vocab_size + 1]

# 모델 구조 설정 (학습 환경과 동일해야 함)
VOCAB_SIZE = 8096 
NUM_LAYERS = 2
D_MODEL = 256
NUM_HEADS = 8
DFF = 512
DROPOUT = 0.1

# 3. 모델 초기화 및 가중치 로드
model = Transformer(
    vocab_size=VOCAB_SIZE,
    num_layers=NUM_LAYERS,
    dff=DFF,
    d_model=D_MODEL,
    num_heads=NUM_HEADS,
    dropout=DROPOUT
).to(device)

# 가중치 파일 로드 및 추론 모드 전환
model.load_state_dict(torch.load("./ko-con.pth", map_location=device))
model.eval()
print("✅ 챗봇 모델 및 토크나이저 로드 완료")

# 4. 텍스트 전처리 및 추론 함수
def preprocess_sentence(sentence):
    """입력 문장의 기호 앞뒤에 공백을 추가하여 전처리합니다."""
    sentence = re.sub(r"([?.!,])", r" \1 ", sentence)
    return sentence.strip()

def evaluate(sentence):
    """사용자 질문을 텐서로 변환하고 Transformer 모델로 답변을 생성합니다."""
    sentence = preprocess_sentence(sentence)
    
    # 인코딩: 시작 토큰 + 내용 + 종료 토큰
    sentence_ids = START_TOKEN + tokenizer.encode(sentence) + END_TOKEN
    inputs = torch.tensor(sentence_ids, dtype=torch.long).unsqueeze(0).to(device)

    # 디코딩 시작점 준비
    output_predict = torch.tensor(START_TOKEN).unsqueeze(0).to(device)

    for i in range(MAX_LENGTH):
        with torch.no_grad():
            predictions = model(inputs, output_predict)
        
        # 마지막 생성 단어 선택 (Greedy Search)
        predictions = predictions[:, -1:, :]
        predicted_id = torch.argmax(predictions, dim=-1)

        # 종료 토큰(EOS)을 만나면 생성 중단
        if predicted_id.item() == END_TOKEN[0]:
            break

        # 예측 단어를 다음 입력에 이어붙임
        output_predict = torch.cat([output_predict, predicted_id], dim=-1)

    return output_predict.squeeze(0).cpu().numpy()

# 5. API 엔드포인트 생성
@app.post('/chat/')
async def get_chatbot_response(item: ChatRequest):
    """POST 요청으로 들어온 사용자 질문에 대해 챗봇 답변을 반환합니다."""
    user_question = item.question
    logger.info(f"USER_REQUEST: {user_question}") # 모듈화된 로거 사용

    try:
        prediction = evaluate(user_question) 
        
        # 특수 토큰을 제외하고 텍스트로 디코딩
        predicted_sentence = tokenizer.decode(
            [i for i in prediction if i < tokenizer.vocab_size]
        )
        
        logger.info(f"BOT_RESPONSE: {predicted_sentence}") # 모듈화된 로거 사용
        
        return {
            'question': user_question,
            'answer': predicted_sentence
        }
    except Exception as e:
        logger.error(f"ERROR: {str(e)}") # 모듈화된 로거 사용
        return {"error": "Internal Server Error"}
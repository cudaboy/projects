from fastapi import FastAPI
from pydantic import BaseModel
import torch
import re
import tensorflow_datasets as tfds
import logging
import os
from datetime import datetime

# 이전 단계에서 정의한 Transformer 클래스들이 포함된 model.py가 있다고 가정하거나, 
# 아래에 해당 클래스 정의(Transformer, EncoderLayer 등)가 포함되어야 합니다.
from model import Transformer 

app = FastAPI()

## Log 정보
# 로그 폴더 생성
LOG_DIR = "./logs"
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

# 로그 파일명 설정 (날짜별로 저장)
log_filename = datetime.now().strftime("chatbot_%Y-%m-%d.log")
log_path = os.path.join(LOG_DIR, log_filename)

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(log_path, encoding='utf-8'), # 파일에 저장
        logging.StreamHandler() # 터미널에 출력
    ]
)
logger = logging.getLogger(__name__)




# 1. 입력 데이터 모델 정의
class ChatRequest(BaseModel):
    question: str

# 2. 전역 변수 및 모델 설정
MAX_LENGTH = 40
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 토크나이저 로드
# 파일 경로에 .subwords 확장자를 제외한 이름을 넣습니다.
tokenizer = tfds.deprecated.text.SubwordTextEncoder.load_from_file("./ko-con_tokenizer")

START_TOKEN = [tokenizer.vocab_size]
END_TOKEN = [tokenizer.vocab_size + 1]

# 에러 방지를 위해 가중치 파일(pth)과 일치하는 VOCAB_SIZE 설정 (이전 에러 메시지 기준 8096)
# 만약 토크나이저와 완벽히 일치한다면 tokenizer.vocab_size + 2를 사용하세요.
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

# 가중치 로드 (경로 확인 필수)
model.load_state_dict(torch.load("./ko-con.pth", map_location=device))
model.eval()
print("챗봇 모델 및 토크나이저 로드 완료")

# 4. 문장 전처리 함수
def preprocess_sentence(sentence):
    sentence = re.sub(r"([?.!,])", r" \1 ", sentence)
    sentence = sentence.strip()
    return sentence

# 5. 예측(답변 생성) 로직
def evaluate(sentence):
    sentence = preprocess_sentence(sentence)
    # 시작 토큰과 종료 토큰 추가 및 텐서 변환
    sentence_ids = START_TOKEN + tokenizer.encode(sentence) + END_TOKEN
    inputs = torch.tensor(sentence_ids, dtype=torch.long).unsqueeze(0).to(device)

    # 디코더의 입력으로 사용할 시작 토큰 준비
    output_predict = torch.tensor(START_TOKEN).unsqueeze(0).to(device)

    for i in range(MAX_LENGTH):
        # 모델 예측
        with torch.no_grad():
            predictions = model(inputs, output_predict)
        
        # 마지막 단어 선택 (Greedy Search)
        predictions = predictions[:, -1:, :]
        predicted_id = torch.argmax(predictions, dim=-1)

        # 종료 토큰을 만나면 중단
        if predicted_id.item() == END_TOKEN[0]:
            break

        # 예측된 단어를 디코더의 입력에 추가
        output_predict = torch.cat([output_predict, predicted_id], dim=-1)

    return output_predict.squeeze(0).cpu().numpy()

# 6. POST 방식으로 API 엔드포인트 생성
@app.post('/chat/')
async def get_chatbot_response(item: ChatRequest):
    user_question = item.question
    logger.info(f"USER_REQUEST: {user_question}") # 요청 로그 기록

    try:
        prediction = evaluate(user_question) # 모델 예측
        predicted_sentence = tokenizer.decode(
            [i for i in prediction if i < tokenizer.vocab_size]
        )
        
        logger.info(f"BOT_RESPONSE: {predicted_sentence}") # 답변 로그 기록
        
        return {
            'question': user_question,
            'answer': predicted_sentence
        }
    except Exception as e:
        logger.error(f"ERROR: {str(e)}") # 에러 발생 시 로그 기록
        return {"error": "Internal Server Error"}
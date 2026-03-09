import torch
import re
import tensorflow_datasets as tfds
import streamlit as st

# 이전 단계에서 정의한 Transformer 클래스들이 포함된 model.py를 불러옵니다.
from model import Transformer  

# ==========================================
# 1. 설정값 및 하이퍼파라미터
# ==========================================
# GPU 연산이 가능하면 CUDA를, 그렇지 않으면 CPU를 사용하도록 설정합니다.
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 모델 아키텍처 설정 (학습 시 사용했던 값과 반드시 동일해야 합니다)
VOCAB_SIZE = 8096  # 단어 사전의 크기
NUM_LAYERS = 2     # 인코더와 디코더의 층 수
D_MODEL = 256      # 임베딩 차원 및 모델 내부의 차원
NUM_HEADS = 8      # 멀티 헤드 어텐션의 헤드 개수
DFF = 512          # 피드 포워드 신경망의 은닉층 크기
DROPOUT = 0.1      # 드롭아웃 비율

# ==========================================
# 2. 모델 및 토크나이저 로드 (캐싱 적용)
# ==========================================
# @st.cache_resource 데코레이터를 사용하여 모델과 토크나이저를 전역 메모리에 캐싱합니다.
# 이렇게 하면 Streamlit 앱에서 사용자가 채팅을 입력해 화면이 새로고침되더라도, 
# 최초 1회만 모델을 로드하고 이후에는 메모리에 올려둔 객체를 재사용하여 속도가 매우 빨라집니다.
@st.cache_resource
def load_model_and_tokenizer():
    # 1. 토크나이저 로드 (SubwordTextEncoder 사용)
    # 확장자(.subwords)를 제외한 파일 경로를 입력합니다.
    tokenizer = tfds.deprecated.text.SubwordTextEncoder.load_from_file("./ko-con_tokenizer")
    
    # 2. Transformer 모델 초기화
    model = Transformer(
        vocab_size=VOCAB_SIZE,
        num_layers=NUM_LAYERS,
        dff=DFF,
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        dropout=DROPOUT
    ).to(DEVICE)
    
    # 3. 학습된 가중치 파일(.pth) 로드
    # map_location을 설정하여 저장된 환경(GPU/CPU)과 현재 구동 환경이 달라도 안전하게 로드합니다.
    model.load_state_dict(torch.load("./ko-con.pth", map_location=DEVICE))
    model.eval() # 모델을 추론(평가) 모드로 전환하여 드롭아웃 등을 비활성화합니다.
    
    return model, tokenizer

# ==========================================
# 3. 텍스트 전처리
# ==========================================
def preprocess_sentence(sentence):
    """
    입력된 문장의 구두점(?, ., !, ,) 앞뒤에 공백을 추가하여 
    단어와 구두점이 분리되어 토큰화되도록 돕습니다.
    """
    sentence = re.sub(r"([?.!,])", r" \1 ", sentence)
    return sentence.strip()

# ==========================================
# 4. 모델 추론 (응답 생성) 로직
# ==========================================
def evaluate(sentence, model, tokenizer, max_length=40):
    """
    사용자의 입력 문장을 받아 Transformer 모델을 통해 답변을 생성합니다.
    """
    # 1. 입력 문장 전처리
    sentence = preprocess_sentence(sentence)
    
    # 2. 시작 토큰과 종료 토큰의 ID 정의
    START_TOKEN = [tokenizer.vocab_size]
    END_TOKEN = [tokenizer.vocab_size + 1]

    # 3. 입력 문장 토큰화 및 텐서 변환
    # [시작 토큰] + [정수 인코딩된 문장] + [종료 토큰] 순서로 결합합니다.
    sentence_ids = START_TOKEN + tokenizer.encode(sentence) + END_TOKEN
    inputs = torch.tensor(sentence_ids, dtype=torch.long).unsqueeze(0).to(DEVICE)
    
    # 4. 디코더의 첫 입력으로 사용될 시작 토큰 텐서 생성
    output_predict = torch.tensor(START_TOKEN).unsqueeze(0).to(DEVICE)

    # 5. 최대 길이(max_length)만큼 반복하며 단어를 하나씩 예측 (자기회귀)
    for _ in range(max_length):
        with torch.no_grad(): # 추론 과정이므로 기울기(Gradient) 계산을 비활성화하여 메모리를 절약합니다.
            predictions = model(inputs, output_predict)
        
        # 현재 예측한 결과 중 가장 마지막 단어의 확률 분포를 가져옵니다.
        predictions = predictions[:, -1:, :]
        
        # 가장 확률이 높은 단어의 인덱스를 선택합니다. (Greedy Search)
        predicted_id = torch.argmax(predictions, dim=-1)

        # 예측된 단어가 종료 토큰(END_TOKEN)이라면 문장 생성을 즉시 중단합니다.
        if predicted_id.item() == END_TOKEN[0]:
            break

        # 예측된 단어를 다음 단계의 디코더 입력 시퀀스에 이어 붙입니다.
        output_predict = torch.cat([output_predict, predicted_id], dim=-1)

    # 6. 생성된 결과 디코딩
    # 텐서를 numpy 배열로 변환하고 배치 차원을 제거합니다.
    prediction = output_predict.squeeze(0).cpu().numpy()
    
    # 예측된 ID 리스트를 실제 텍스트로 변환 (시작/종료 등 특수 토큰 제외)
    predicted_sentence = tokenizer.decode([i for i in prediction if i < tokenizer.vocab_size])
    
    return predicted_sentence
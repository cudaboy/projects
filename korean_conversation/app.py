import streamlit as st
# 다른 파일(모듈)에서 정의한 함수들을 가져옵니다.
from sidebar import render_sidebar
from main import run_chat
from model_handler import load_model_and_tokenizer

# [Streamlit 페이지 기본 설정]
# 이 함수는 반드시 다른 Streamlit 명령어보다 먼저 실행되어야 합니다.
st.set_page_config(
    page_title="Korean Conversation Chatbot", # 브라우저 탭에 표시될 제목
    page_icon="💬",                   # 브라우저 탭에 표시될 아이콘
    layout="centered"                 # 화면 레이아웃 (centered 또는 wide)
)

def main():
    """
    앱의 전체 실행 흐름을 제어하는 메인 함수입니다.
    """
    
    # 1. 모델과 토크나이저 로드
    # model_handler.py의 load_model_and_tokenizer 함수는 @st.cache_resource가 적용되어 있어
    # 앱이 재실행되어도 모델을 다시 로드하지 않고 메모리에 유지된 객체를 재사용합니다.
    model, tokenizer = load_model_and_tokenizer()
    
    # 2. 사이드바 UI 렌더링 및 설정값 받아오기
    # 사용자가 슬라이더로 조절한 max_length 등의 설정값을 반환받습니다.
    max_length = render_sidebar()
    
    # 3. 메인 채팅 인터페이스 실행
    # 실제 채팅 UI를 출력하고 사용자의 입력에 따라 모델 추론(Inference)을 수행합니다.
    run_chat(model, tokenizer, max_length)

# 파이썬 스크립트가 직접 실행될 때만 main() 함수를 호출합니다.
if __name__ == "__main__":
    main()
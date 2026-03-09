import streamlit as st
from model_handler import evaluate
from utils import setup_logger

logger = setup_logger()

def run_chat(model, tokenizer, max_length):
    st.title("🤖 나만의 Transformer 챗봇")

    # 대화 기록을 저장할 세션 상태 초기화
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # 기존 대화 기록 화면에 렌더링
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # 사용자 입력 처리
    if prompt := st.chat_input("챗봇에게 메시지를 보내보세요."):
        logger.info(f"USER_REQUEST: {prompt}")
        
        # 사용자 메시지 화면에 출력 및 상태 저장
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # 챗봇 응답 생성 및 출력
        with st.chat_message("assistant"):
            with st.spinner("답변을 생성하는 중입니다..."):
                try:
                    response = evaluate(prompt, model, tokenizer, max_length=max_length)
                    st.markdown(response)
                    
                    # 챗봇 응답 상태 저장 및 로깅
                    st.session_state.messages.append({"role": "assistant", "content": response})
                    logger.info(f"BOT_RESPONSE: {response}")
                except Exception as e:
                    st.error("응답을 생성하는 중 오류가 발생했습니다.")
                    logger.error(f"ERROR: {str(e)}")
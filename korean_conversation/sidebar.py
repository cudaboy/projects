import streamlit as st

def render_sidebar():
    with st.sidebar:
        st.title("⚙️ 챗봇 설정")
        
        # 모델의 max_length 파라미터 조절
        max_length = st.slider(
            "최대 생성 길이 (Max Length)", 
            min_value=10, 
            max_value=100, 
            value=40, 
            step=5
        )
        
        # 세션 상태 초기화 (대화 기록 삭제)
        if st.button("대화 기록 초기화", use_container_width=True):
            st.session_state.messages = []
            st.rerun() # 화면 새로고침
            
        return max_length
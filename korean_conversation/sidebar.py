import streamlit as st
from datetime import datetime

def render_sidebar():
    with st.sidebar:
        st.title("⚙️ 챗봇 설정")
        
        # 세션 상태 초기화 (대화 기록 삭제)
        if st.button("🗑️ 대화 기록 초기화", use_container_width=True):
            st.session_state.messages = []
            st.rerun() # 화면 새로고침
            
        st.divider()
    
        # 대화 내역 다운로드
        st.subheader("💾 대화 내역 저장")
        
        # 세션 상태에 메시지 변수가 없으면 에러 방지를 위해 빈 리스트로 초기화
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # 대화 내역이 존재할 때만 다운로드 버튼을 활성화합니다.
        if st.session_state.messages:
            # 다운로드할 텍스트 데이터 포맷팅
            chat_text = "🤖 Transformer 챗봇 대화 내역\n"
            chat_text += "=" * 30 + "\n\n"
            
            for msg in st.session_state.messages:
                # 역할을 알아보기 쉽게 한글로 변환
                role = "사용자" if msg["role"] == "user" else "챗봇"
                chat_text += f"[{role}] {msg['content']}\n"
                
            # 파일명에 현재 시간을 포함하여 중복 방지 (예: chat_history_20260310_204527.txt)
            current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_name = f"chat_history_{current_time}.txt"

            # Streamlit 다운로드 버튼 생성
            st.download_button(
                label="📥 텍스트 파일(.txt)로 다운로드",
                data=chat_text,
                file_name=file_name,
                mime="text/plain",
                use_container_width=True
            )
        else:
            # 대화가 없을 때 표시할 안내 문구
            st.info("아직 대화 내역이 없습니다. 챗봇과 대화를 시작해 보세요!")
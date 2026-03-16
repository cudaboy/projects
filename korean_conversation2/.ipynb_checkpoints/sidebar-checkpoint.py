import streamlit as st
from datetime import datetime

def render_sidebar():
    """
    Streamlit 화면 왼쪽의 사이드바(Sidebar) UI를 구성하는 함수입니다.
    대화 기록 초기화 및 대화 내역 다운로드 기능을 제공합니다.
    """
    with st.sidebar:
        st.title("⚙️ RAG 챗봇 설정")
        
        # =====================================================================
        # 1. 대화 기록 초기화 버튼
        # =====================================================================
        # 버튼이 클릭되면 세션 상태(session_state)에 저장된 대화 목록을 비우고 화면을 새로고침합니다.
        if st.button("🗑️ 대화 기록 초기화", use_container_width=True):
            st.session_state.messages = []
            st.rerun() # 화면을 즉시 새로고침하여 변경사항(빈 화면)을 반영합니다.
            
        st.divider() # 시각적인 구분선을 긋습니다.
    
        # =====================================================================
        # 2. 대화 내역 다운로드 기능
        # =====================================================================
        st.subheader("💾 대화 내역 저장")
        
        # 세션 메모리에 'messages' 공간이 아예 없다면(최초 접속 시) 에러 방지를 위해 빈 리스트로 만듭니다.
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # 대화를 한 번이라도 나누어 messages 리스트에 데이터가 있을 때만 다운로드 버튼을 활성화하여 보여줍니다.
        if st.session_state.messages:
            
            # 다운로드될 텍스트 파일의 윗부분(헤더)을 예쁘게 꾸밉니다.
            chat_text = "🤖 Transformer RAG 챗봇 대화 내역\n"
            chat_text += "=" * 30 + "\n\n"
            
            # 저장된 대화 기록을 처음부터 끝까지 하나씩 꺼내서 텍스트로 쭉 연결합니다.
            for msg in st.session_state.messages:
                # 'user'는 '사용자'로, 'assistant'는 '챗봇'으로 역할을 알기 쉽게 한글로 바꿉니다.
                role = "사용자" if msg["role"] == "user" else "챗봇"
                chat_text += f"[{role}] {msg['content']}\n"
                
            # 다운로드 파일 이름이 겹치지 않도록 현재 시간을 파일명에 붙여줍니다.
            # (예: rag_chat_history_20260316_094417.txt)
            current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Streamlit의 내장 다운로드 버튼 컴포넌트입니다. 클릭하면 즉시 txt 파일이 다운로드됩니다.
            st.download_button(
                label="📥 텍스트 파일로 다운로드",
                data=chat_text,
                file_name=f"rag_chat_history_{current_time}.txt",
                mime="text/plain",
                use_container_width=True
            )
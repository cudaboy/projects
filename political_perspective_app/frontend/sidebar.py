import streamlit as st

def render_sidebar():
    """Streamlit 좌측 사이드바를 구성하고 선택된 설정들을 반환합니다."""
    with st.sidebar:
        st.title("⚙️ 설정 및 정보")
        st.markdown("---")
        
        st.subheader("🤖 AI 모델 설정")
        provider = st.selectbox("LLM 제공자", ["OpenAI", "Anthropic", "Google"])
        
        if provider == "OpenAI":
            model_name = st.selectbox("모델 선택", ["gpt-4o-mini", "gpt-4o", "gpt-4-turbo"])
        elif provider == "Anthropic":
            model_name = st.selectbox("모델 선택", ["claude-3-haiku-20240307", "claude-3-sonnet-20240229", "claude-3-opus-20240229"])
        elif provider == "Google":
            model_name = st.selectbox("모델 선택", ["gemini-1.5-flash", "gemini-1.5-pro"])
            
        st.markdown("---")
        
        # 🌟 LangSmith 추적 활성화 토글 (기본값: False)
        st.subheader("🛠️ 개발자 옵션")
        use_langsmith = st.checkbox("LangSmith 모니터링 활성화", value=False)
        
        st.markdown("---")
        st.info("💡 **Spectrum View**는 12대 핵심 의제를 바탕으로 세 가지 정치적 관점을 심층 분석합니다.")
        
        with st.expander("📌 12대 핵심 의제 보기"):
            st.markdown("""
            1. 사회  2. 경제  3. 분배  4. 복지
            5. 정치  6. 문화  7. 국방  8. 북한/통일
            9. 외교 10. 사법 11. 교육관 12. 개헌
            """)
        
        st.caption("© 2026 Spectrum View App")
        
        # 🌟 provider, model_name에 더해 use_langsmith까지 3개의 값을 반환
        return provider.lower(), model_name, use_langsmith
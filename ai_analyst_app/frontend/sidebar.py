import streamlit as st

def render_sidebar():
    """
    사이드바 UI를 렌더링하고 사용자가 입력한 설정값들을 딕셔너리 형태로 반환합니다.
    """
    with st.sidebar:
        st.title("⚙️ 분석 모델 설정")
        st.caption("에이전트들이 사용할 AI 모델과 파라미터를 세팅합니다.")
        
        st.markdown("---")
        
        # ==========================================
        # 1. LLM 제공자 및 세부 모델 선택
        # ==========================================
        provider = st.selectbox(
            "LLM 제공자", 
            ["OpenAI", "Anthropic", "Google Gemini"],
            help="분석을 수행할 핵심 AI 모델의 제공자를 선택하세요."
        )
        
        # 제공자에 따라 세부 모델 리스트 동적 변경
        if provider == "OpenAI":
            model_options = ["gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"]
            key_env_name = "OPENAI_API_KEY"
        elif provider == "Anthropic":
            model_options = ["claude-3-opus-20240229", "claude-3-sonnet-20240229", "claude-3-haiku-20240307"]
            key_env_name = "ANTHROPIC_API_KEY"
        else:  # Google Gemini
            model_options = ["gemini-1.5-pro-latest", "gemini-1.5-flash-latest"]
            key_env_name = "GOOGLE_API_KEY"
            
        model_name = st.selectbox("세부 모델", model_options)
        
        temperature = st.slider(
            "창의성 (Temperature)", 
            min_value=0.0, max_value=1.0, value=0.2, step=0.1,
            help="0에 가까울수록 사실적이고 일관된 답변을, 1에 가까울수록 창의적인 답변을 제공합니다."
        )
        
        # 🌟 선택된 제공자의 이름으로 API Key 입력창 라벨 동적 변경
        st.markdown("### 🔑 LLM API 키 입력")
        llm_api_key = st.text_input(
            f"[{provider}] API Key 입력", 
            type="password",
            help="기본 서버 환경변수를 사용하려면 비워두세요."
        )

        st.markdown("---")

        # ==========================================
        # 2. 검색 엔진 설정 (네이버 API & 구글 Fallback)
        # ==========================================
        st.header("🔍 검색 엔진 설정")
        st.caption("※ 네이버 API 키를 입력하지 않으면 무료 구글 검색 엔진으로 자동 대체됩니다.")
        
        naver_client_id = st.text_input("Naver Client ID", type="password")
        naver_client_secret = st.text_input("Naver Client Secret", type="password")

        st.markdown("---")

        # ==========================================
        # 3. 개발자 옵션 (LangSmith)
        # ==========================================
        st.header("🛠️ 개발자 옵션")
        use_langsmith = st.checkbox("LANGSMITH 추적 활성화")
        
        langsmith_api_key = None
        # 🌟 체크박스가 활성화되었을 때만 입력창 노출 (조건부 렌더링)
        if use_langsmith:
            langsmith_api_key = st.text_input(
                "LANGSMITH API KEY를 입력하세요", 
                type="password"
            )

        # 수집된 모든 설정값을 딕셔너리로 묶어서 반환
        return {
            "provider": provider,
            "model_name": model_name,
            "temperature": temperature,
            "custom_api_key": llm_api_key,
            "llm_key_env_name": key_env_name,  # 백엔드에서 어떤 키인지 식별하기 위한 변수명
            "use_langsmith": use_langsmith,
            "langsmith_api_key": langsmith_api_key,
            "naver_client_id": naver_client_id,
            "naver_client_secret": naver_client_secret
        }
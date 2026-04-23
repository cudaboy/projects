"""
===============================================================================
[File Role]
이 파일(sidebar.py)은 BioLinker 프로젝트 프론트엔드의 '전역 설정 및 보안 인증'을 담당합니다.

[상세 기능]
1. 멀티 LLM 엔진 관리: OpenAI, Anthropic, Google, Grok 모델을 동적으로 선택할 수 있습니다.
2. 실시간 보안 인증: 사용자가 직접 UI에서 API 키를 입력하여 백엔드로 전달합니다.
3. 시스템 추적 제어: LangSmith 활성화 여부 및 관련 API 키 설정을 관리합니다.
===============================================================================
"""

import streamlit as st
import os

def render_sidebar():
    """Streamlit 좌측 사이드바를 구성하고 사용자 설정값을 반환합니다."""
    with st.sidebar:
        # 로고 이미지를 제거하고 텍스트 타이틀만 깔끔하게 표시합니다.
        st.title("🧬 BioLinker")
        
        st.subheader("⚙️ 시스템 설정")
        st.markdown("---")
        
        # 1. AI 모델 설정 섹션
        st.subheader("🤖 LLM 추론 모델 설정")
        provider = st.selectbox(
            "LLM 제공자 (Provider)", 
            ["OpenAI", "Anthropic", "Google", "Grok"],
            help="답변 합성을 담당할 메인 언어 모델의 제조사를 선택하세요."
        )
        
        # 제공자별 세부 모델 매핑 및 API 키 입력 (환경 변수 자동 인식 포함)
        if provider == "OpenAI":
            model_name = st.selectbox("모델 선택", ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo"])
            api_key = st.text_input(
                "OpenAI API Key", 
                type="password", 
                value=os.getenv("OPENAI_API_KEY", ""),
                placeholder="sk-..."
            )
        elif provider == "Anthropic":
            model_name = st.selectbox("모델 선택", ["claude-3-5-sonnet-20240620", "claude-3-opus-20240229"])
            api_key = st.text_input(
                "Anthropic API Key", 
                type="password", 
                value=os.getenv("ANTHROPIC_API_KEY", ""),
                placeholder="sk-ant-..."
            )
        elif provider == "Google":
            model_name = st.selectbox("모델 선택", ["gemini-1.5-pro", "gemini-1.5-flash"])
            api_key = st.text_input(
                "Google AI API Key", 
                type="password",
                value=os.getenv("GOOGLE_API_KEY", "")
            )
        elif provider == "Grok":
            model_name = st.selectbox("모델 선택", ["grok-2-1212", "grok-2-latest", "grok-beta"])
            api_key = st.text_input(
                "xAI (Grok) API Key", 
                type="password", 
                value=os.getenv("XAI_API_KEY", os.getenv("GROK_API_KEY", "")),
                placeholder="xai-..."
            )
            
        st.markdown("---")
        
        # 2. LangSmith 및 모니터링 설정
        st.subheader("🛠️ 모니터링 옵션")
        use_langsmith = st.checkbox(
            "LangSmith 추적 활성화", 
            value=False,
            help="에이전트의 사고 과정을 LangSmith 대시보드에서 실시간 모니터링합니다."
        )
        
        langsmith_api_key = ""
        if use_langsmith:
            langsmith_api_key = st.text_input(
                "LangSmith API Key", 
                type="password", 
                value=os.getenv("LANGCHAIN_API_KEY", ""),
                placeholder="lsv2_pt_..."
            )
            
        st.markdown("---")
        
        # 3. 프로젝트 정보
        st.info(
            "💡 **BioLinker**는 ModernBERT을 이용한 의료 데이터 특화 임베딩과 "
            "Multi-hop 그래프 추론을 결합하여 지식을 탐색합니다."
        )
        
        with st.expander("📌 시스템 아키텍처"):
            st.markdown("""
            - **Brain:** LangGraph Multi-Agent
            - **Memory:** ChromaDB (Abstracts)
            - **Structure:** NetworkX (Knowledge Graph)
            """)
        
        st.caption("© 2026 BioLinker System | AI-Powered Bio Intelligence")
        
        return {
            "provider": provider.lower(),
            "model_name": model_name,
            "api_key": api_key,
            "use_langsmith": use_langsmith,
            "langsmith_api_key": langsmith_api_key
        }
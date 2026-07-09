"""BioLinker Streamlit sidebar."""

from __future__ import annotations

import os
import streamlit as st


def render_sidebar():
    with st.sidebar:
        st.title("🧬 BioLinker")
        st.subheader("⚙️ 시스템 설정")
        st.markdown("---")

        st.subheader("🤖 LLM 추론 모델 설정")
        provider = st.selectbox(
            "LLM 제공자 (Provider)",
            ["OpenAI", "Anthropic", "Google", "Grok"],
            help="최종 답변 합성을 담당할 언어 모델 제공자를 선택합니다.",
        )

        if provider == "OpenAI":
            auth_mode = st.selectbox(
                "인증 방식",
                ["api_key", "hermes_openai_key", "oauth"],
                index=0,
                help="api_key: 직접 입력/API 요청값 사용, hermes_openai_key: 개발용으로 환경변수·BioLinker .env·Hermes .env의 OPENAI_API_KEY 자동 사용, oauth: Hermes OAuth 토큰 사용(별도 scope 필요)",
            )
            model_name = st.selectbox("모델 선택", ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo"])
            default_key = "" if auth_mode in {"hermes_openai_key", "oauth"} else os.getenv("OPENAI_API_KEY", "")
            api_key = st.text_input("OpenAI API Key", type="password", value=default_key, placeholder="sk-...")
        elif provider == "Anthropic":
            auth_mode = "api_key"
            model_name = st.selectbox("모델 선택", ["claude-3-5-sonnet-20240620", "claude-3-opus-20240229"])
            api_key = st.text_input("Anthropic API Key", type="password", value=os.getenv("ANTHROPIC_API_KEY", ""), placeholder="sk-ant-...")
        elif provider == "Google":
            auth_mode = "api_key"
            model_name = st.selectbox("모델 선택", ["gemini-1.5-pro", "gemini-1.5-flash"])
            api_key = st.text_input("Google AI API Key", type="password", value=os.getenv("GOOGLE_API_KEY", ""))
        else:
            auth_mode = "api_key"
            model_name = st.selectbox("모델 선택", ["grok-2-1212", "grok-2-latest", "grok-beta"])
            api_key = st.text_input(
                "xAI (Grok) API Key",
                type="password",
                value=os.getenv("XAI_API_KEY", os.getenv("GROK_API_KEY", "")),
                placeholder="xai-...",
            )

        st.markdown("---")
        st.subheader("🧪 검색 실험 옵션")
        retrieval_mode = st.selectbox(
            "Retrieval Mode",
            ["auto", "vector", "graph", "both"],
            index=0,
            help="auto는 라우터가 결정하고, 나머지는 강제로 해당 retrieval 경로를 사용합니다.",
        )
        top_k = st.slider("Top-K 문헌 수", min_value=1, max_value=12, value=6, step=1)
        show_raw_payload = st.checkbox("Raw API Response 보기", value=False)

        st.markdown("---")
        st.subheader("🛠️ 모니터링 옵션")
        use_langsmith = st.checkbox("LangSmith 추적 활성화", value=False)
        langsmith_api_key = ""
        if use_langsmith:
            langsmith_api_key = st.text_input(
                "LangSmith API Key",
                type="password",
                value=os.getenv("LANGCHAIN_API_KEY", ""),
                placeholder="lsv2_pt_...",
            )

        st.markdown("---")
        st.info(
            "💡 BioLinker는 vector 문헌 검색과 knowledge graph 탐색을 결합합니다.\n\n"
            "- route confidence\n- evidence panel\n- citation trace\n- retrieval mode 비교"
        )
        with st.expander("📌 시스템 아키텍처"):
            st.markdown(
                """
- **Brain:** LangGraph Multi-Agent
- **Memory:** ChromaDB (chunked literature)
- **Structure:** NetworkX (knowledge graph)
- **API:** FastAPI readiness / health endpoints
                """.strip()
            )
        st.caption("© 2026 BioLinker System | AI-Powered Bio Intelligence")

        return {
            "provider": provider.lower(),
            "model_name": model_name,
            "api_key": api_key,
            "auth_mode": auth_mode,
            "use_langsmith": use_langsmith,
            "langsmith_api_key": langsmith_api_key,
            "retrieval_mode": retrieval_mode,
            "top_k": top_k,
            "show_raw_payload": show_raw_payload,
        }

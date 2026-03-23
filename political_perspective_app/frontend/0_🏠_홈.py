# 0_🏠_홈.py : 사용자의 입력을 가장 먼저 처리
# 사이드바에서 설정한 내용, 백엔드의 상세 에러 메시지를 화면에 띄워주는 기능

import streamlit as st
import requests
import os
from sidebar import render_sidebar
from ui_components import render_analysis_result

st.set_page_config(page_title="Spectrum View", page_icon="⚖️", layout="wide")

# 🌟 사이드바에서 3개의 변수를 모두 받아옵니다.
provider, model_name, use_langsmith = render_sidebar()

st.title("⚖️ 정치 스펙트럼 심층 뷰어")
st.markdown("제시된 이슈에 대하여 12가지 핵심 의제에 대해 진영별(진보·중도·보수) 심층 분석 결과를 도출합니다.")

@st.cache_resource(ttl=10)
def get_active_backend_url():
    env_url = os.getenv("BACKEND_API_URL")
    if env_url:
        return env_url
        
    for port in [8000, 8001, 8002]:
        base_url = f"http://localhost:{port}"
        try:
            res = requests.get(base_url, timeout=1)
            if res.status_code == 200 and "Spectrum View Backend API" in res.text:
                return f"{base_url}/analyze"
        except requests.exceptions.ConnectionError:
            continue
    return None

with st.form(key="search_form"):
    query = st.text_input("분석할 이슈나 키워드를 입력하세요 (예: 전 국민 25만 원 민생회복지원금 지급안)")
    submit_button = st.form_submit_button(label="관점 분석하기")

if submit_button:
    if query:
        BACKEND_URL = get_active_backend_url()
        
        if not BACKEND_URL:
            st.error("🚨 열려있는 백엔드 서버를 찾을 수 없습니다. 서버 실행 상태를 확인해 주세요.")
        else:
            with st.spinner(f"[{provider.upper()} - {model_name}] 모델이 핵심 의제를 바탕으로 이슈를 분석 중입니다..."):
                try:
                    # 🌟 JSON 데이터에 use_langsmith 추가
                    payload = {
                        "question": query,
                        "provider": provider,
                        "model_name": model_name,
                        "use_langsmith": use_langsmith
                    }
                    response = requests.post(BACKEND_URL, json=payload)
                    response.raise_for_status() 
                    
                    result = response.json()
                    render_analysis_result(result)
                        
                except requests.exceptions.RequestException as e:
                    error_msg = str(e)
                    if getattr(e, 'response', None) is not None:
                        try:
                            error_msg = e.response.json().get("detail", str(e))
                        except:
                            pass
                    st.error(f"🚨 분석 중 오류가 발생했습니다.\n\n원인: {error_msg}")
    else:
        st.warning("분석할 키워드를 입력해주세요.")
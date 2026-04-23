"""
===============================================================================
[File Role]
이 파일(main.py)은 BioLinker 프로젝트의 '프론트엔드 진입점(Main Entry Point)'입니다.

[전체 프로젝트 내 역할]
1. UI 오케스트레이션: sidebar.py와 협력하여 전체 웹 인터페이스의 레이아웃과 설정을 관리합니다.
2. 상태 유지(Stateful Management): Streamlit의 세션 상태를 활용해 대화 흐름을 끊김 없이 유지합니다.
3. 백엔드 통신(Bridge): 사용자가 입력한 보안 API 키와 질의 내용을 FastAPI 백엔드(api.py)로 
   안전하게 전달하고 응답을 수신합니다.
4. 사용자 경험(UX): AI의 추론 과정(어떤 DB를 탐색했는지 등)을 투명하게 시각화하여 
   사용자에게 근거 중심의 신뢰할 수 있는 답변을 제공합니다.
5. 대화 내용 저장: 대화 내역을 로컬 파일(json)에 지속적으로 저장하고 재시작 시 불러오는 로깅 기능 추가
   대화 내역 초기화 및 다운로드 버튼 UI 추가
===============================================================================
"""

import streamlit as st
import requests
import time
import sys
import os
import json
from pathlib import Path
from dotenv import load_dotenv

# 1. 환경 변수 로드
env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

# [경로 설정] sidebar.py 인식을 위한 경로 추가
current_dir = Path(__file__).resolve().parent
if str(current_dir) not in sys.path:
    sys.path.append(str(current_dir))

from sidebar import render_sidebar 

# ---------------------------------------------------------
# 로컬 로그 저장 기능 구현
# ---------------------------------------------------------
# 채팅 기록을 저장할 JSON 파일 경로 지정 (data 폴더 내)
CHAT_LOG_FILE = Path(__file__).resolve().parent.parent / "data" / "chat_history.json"

def save_chat_history():
    """현재 세션의 채팅 기록을 로컬 JSON 파일에 저장합니다."""
    CHAT_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(CHAT_LOG_FILE, "w", encoding="utf-8") as f:
        json.dump(st.session_state.messages, f, ensure_ascii=False, indent=2)

def clear_chat_history():
    """채팅 기록을 초기화하고 파일에서도 삭제합니다."""
    st.session_state.messages = [
        {"role": "assistant", "content": "반갑습니다! 오늘 어떤 약물의 기전이나 질환 정보를 찾아드릴까요?"}
    ]
    save_chat_history()

# ---------------------------------------------------------
# 페이지 설정 및 UI 초기화
# ---------------------------------------------------------
st.set_page_config(
    page_title="BioLinker | AI-Powered Bio RAG",
    page_icon="🧬",
    layout="wide"
)

# FastAPI 백엔드 주소
API_URL = "http://localhost:8000/api/v1/query"

# 사이드바 렌더링 및 설정값 수신
config = render_sidebar()

# 메인 화면 헤더
st.title("🔬 BioLinker AI Searcher")
st.markdown("사용자가 직접 입력하거나 환경 변수에 설정된 API 키를 사용하여 분석을 수행합니다.")

# ---------------------------------------------------------
# 세션 상태 초기화 및 과거 로그 불러오기
# ---------------------------------------------------------
if "messages" not in st.session_state:
    if CHAT_LOG_FILE.exists():
        try:
            with open(CHAT_LOG_FILE, "r", encoding="utf-8") as f:
                st.session_state.messages = json.load(f)
        except Exception:
            clear_chat_history()
    else:
        clear_chat_history()

# ---------------------------------------------------------
# [사이드바 확장] 대화 기록 관리 도구 추가
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("---")
    st.subheader("💾 검색 기록 관리")
    col1, col2 = st.columns(2)
    
    # 초기화 버튼
    with col1:
        if st.button("🔄 기록 초기화", use_container_width=True):
            clear_chat_history()
            st.rerun()
            
    # 다운로드 버튼
    with col2:
        chat_json = json.dumps(st.session_state.messages, ensure_ascii=False, indent=2)
        st.download_button(
            label="📥 다운로드",
            data=chat_json,
            file_name="biolinker_chat_log.json",
            mime="application/json",
            use_container_width=True
        )

# ---------------------------------------------------------
# 기존 대화 기록 출력
# ---------------------------------------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "route" in msg:
            st.caption(f"🧭 탐색 경로: **{msg['route'].upper()}**")

# ---------------------------------------------------------
# 사용자 질문 입력 및 응답 생성 로직
# ---------------------------------------------------------
if prompt := st.chat_input("질문을 입력하세요..."):
    # API Key 검증
    if not config["api_key"]:
        st.error("⚠️ 사이드바에서 선택한 모델의 API Key를 먼저 입력해주세요. (.env 파일 확인 필요)")
        st.stop()

    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})
    save_chat_history() # 질문 저장

    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        
        with st.spinner(f"{config['model_name']} 분석 중..."):
            try:
                payload = {
                    "question": prompt,
                    "provider": config["provider"],
                    "model_name": config["model_name"],
                    "api_key": config["api_key"],
                    "use_langsmith": config["use_langsmith"],
                    "langsmith_api_key": config["langsmith_api_key"]
                }
                
                start_time = time.time()
                response = requests.post(API_URL, json=payload, timeout=60)
                response.raise_for_status()
                
                result = response.json()
                final_answer = result.get("final_answer", "결과를 가져오지 못했습니다.")
                route = result.get("route", "unknown")
                logs = result.get("logs", [])
                elapsed_time = time.time() - start_time
                
                st.markdown(f"<span style='color:green; font-size:0.8em;'>✓ {route.upper()} 탐색 및 합성 완료 ({elapsed_time:.2f}초)</span>", unsafe_allow_html=True)
                
                # 최종 답변 출력
                response_placeholder.markdown(final_answer)
                
                if logs:
                    with st.expander("🛠️ 에이전트 사고 과정 (Trace Logs)", expanded=False):
                        for log in logs:
                            # 에러나 경고 기호가 있으면 warning으로 시각적 구분
                            if "⚠️" in log or "❌" in log:
                                st.warning(log)
                            else:
                                st.info(log)
                
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": final_answer,
                    "route": route
                })
                save_chat_history() # 답변 완료 후 상태 저장
                
            except Exception as e:
                st.error(f"❌ 오류가 발생했습니다: {e}")

# 하단 푸터
st.markdown("---")
st.caption(f"Connected to: {config['provider'].upper()} ({config['model_name']})")
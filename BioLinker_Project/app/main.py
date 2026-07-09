"""
BioLinker Streamlit frontend.

개선 사항
- 세션별 로그 저장
- backend readiness 상태 표시
- 답변 / citation / graph / raw response 패널 분리
- retrieval mode / top_k 실험 옵션 전달
"""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path

import requests
import streamlit as st
from dotenv import load_dotenv

try:
    from biolinker import config as app_config
except ImportError:
    import sys

    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from biolinker import config as app_config

from sidebar import render_sidebar

env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

API_URL = f"http://localhost:{app_config.API_PORT}/api/v1/query"
READY_URL = f"http://localhost:{app_config.API_PORT}/health/ready"

st.set_page_config(page_title="BioLinker | AI-Powered Bio RAG", page_icon="🧬", layout="wide")


def get_session_id() -> str:
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())[:8]
    return st.session_state.session_id


def session_log_file() -> Path:
    return app_config.SESSION_LOG_DIR / f"chat_{get_session_id()}.json"


def save_chat_history():
    path = session_log_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(st.session_state.messages, handle, ensure_ascii=False, indent=2)


def clear_chat_history():
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "반갑습니다! 약물 기전, 질환-유전자 관계, 관련 문헌 근거를 함께 탐색해드릴게요.",
        }
    ]
    save_chat_history()


def load_readiness() -> dict:
    try:
        response = requests.get(READY_URL, timeout=5)
        response.raise_for_status()
        return response.json()
    except Exception as exc:
        return {"status": "unreachable", "ready": False, "error": str(exc)}


def render_assistant_message(msg: dict, show_raw_payload: bool = False):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("role") != "assistant":
            return
        meta = msg.get("meta") or {}
        if not meta:
            return
        cols = st.columns(4)
        cols[0].metric("Route", meta.get("route", "unknown").upper())
        cols[1].metric("Confidence", f"{float(meta.get('route_confidence', 0.0)):.2f}")
        cols[2].metric("Latency", f"{float(meta.get('latency_ms', 0.0)):.0f} ms")
        cols[3].metric("Safety", meta.get("safety_flag", "ok"))

        tab_answer, tab_citations, tab_graph, tab_logs, tab_raw = st.tabs(
            ["Answer", "Citations", "Graph", "Trace Logs", "Raw JSON"]
        )
        with tab_answer:
            if meta.get("no_answer_reason"):
                st.warning(f"근거 부족 사유: {meta['no_answer_reason']}")
            st.markdown(msg["content"])
        with tab_citations:
            citations = meta.get("citations", [])
            if not citations:
                st.info("표시할 문헌 citation이 없습니다.")
            else:
                for citation in citations:
                    st.markdown(
                        f"- **{citation.get('title','제목 없음')}**  \
출처 ID: `{citation.get('doc_id','')}` | 저널: {citation.get('journal','')} | 연도: {citation.get('year','')} | score: {float(citation.get('score',0.0)):.4f}"
                    )
        with tab_graph:
            graph_edges = meta.get("graph_edges", [])
            if not graph_edges:
                st.info("표시할 그래프 관계가 없습니다.")
            else:
                for edge in graph_edges:
                    st.markdown(
                        f"- `{edge.get('subject','')}` --**{edge.get('relation','')}``→ `{edge.get('object','')}` "
                        f"(hop={edge.get('hop',1)}, doc={edge.get('doc_id','')}, year={edge.get('year','')})"
                    )
                    if edge.get("evidence_text"):
                        st.caption(edge["evidence_text"][:240])
        with tab_logs:
            for log in meta.get("logs", []):
                if "⚠️" in log or "insufficient" in log.lower():
                    st.warning(log)
                else:
                    st.info(log)
        with tab_raw:
            if show_raw_payload:
                st.json(meta)
            else:
                st.caption("사이드바에서 'Raw API Response 보기'를 켜면 전체 payload를 볼 수 있습니다.")


sidebar_config = render_sidebar()
readiness = load_readiness()

st.title("🔬 BioLinker AI Searcher")
st.markdown("문헌 검색, 그래프 탐색, route confidence, citation trace를 한 화면에서 확인합니다.")

if readiness.get("ready"):
    st.success(
        f"Backend Ready | embedding={readiness.get('embedding_model')} | device={readiness.get('embedding_device')}"
    )
else:
    st.warning(f"Backend not ready: {readiness.get('status')} | {readiness.get('error', '')}")

if "messages" not in st.session_state:
    log_file = session_log_file()
    if log_file.exists():
        try:
            with open(log_file, "r", encoding="utf-8") as handle:
                st.session_state.messages = json.load(handle)
        except Exception:
            clear_chat_history()
    else:
        clear_chat_history()

with st.sidebar:
    st.markdown("---")
    st.subheader("💾 세션 로그 관리")
    st.caption(f"session_id: `{get_session_id()}`")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔄 기록 초기화", use_container_width=True):
            clear_chat_history()
            st.rerun()
    with col2:
        st.download_button(
            label="📥 다운로드",
            data=json.dumps(st.session_state.messages, ensure_ascii=False, indent=2),
            file_name=f"biolinker_chat_{get_session_id()}.json",
            mime="application/json",
            use_container_width=True,
        )

for message in st.session_state.messages:
    render_assistant_message(message, show_raw_payload=sidebar_config["show_raw_payload"])

if prompt := st.chat_input("질문을 입력하세요..."):
    if sidebar_config.get("auth_mode") not in {"hermes_openai_key", "oauth"} and not sidebar_config["api_key"]:
        st.error("⚠️ 사이드바에서 선택한 모델의 API Key를 먼저 입력해주세요.")
        st.stop()

    user_message = {"role": "user", "content": prompt}
    st.session_state.messages.append(user_message)
    save_chat_history()
    render_assistant_message(user_message)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        with st.spinner(f"{sidebar_config['model_name']} 분석 중..."):
            try:
                payload = {
                    "question": prompt,
                    "provider": sidebar_config["provider"],
                    "model_name": sidebar_config["model_name"],
                    "api_key": sidebar_config["api_key"],
                    "auth_mode": sidebar_config.get("auth_mode", "api_key"),
                    "use_langsmith": sidebar_config["use_langsmith"],
                    "langsmith_api_key": sidebar_config["langsmith_api_key"],
                    "retrieval_mode": sidebar_config["retrieval_mode"],
                    "top_k": sidebar_config["top_k"],
                    "session_id": get_session_id(),
                }
                start_time = time.time()
                response = requests.post(API_URL, json=payload, timeout=app_config.REQUEST_TIMEOUT_SECONDS)
                response.raise_for_status()
                result = response.json()
                elapsed = time.time() - start_time
                placeholder.markdown(result.get("final_answer", "결과를 가져오지 못했습니다."))
                st.caption(
                    f"✓ route={result.get('route','unknown').upper()} | confidence={float(result.get('route_confidence',0.0)):.2f} | "
                    f"latency={result.get('latency_ms', elapsed * 1000):.0f} ms"
                )
                assistant_message = {
                    "role": "assistant",
                    "content": result.get("final_answer", "결과를 가져오지 못했습니다."),
                    "meta": result,
                }
                st.session_state.messages.append(assistant_message)
                save_chat_history()
                render_assistant_message(assistant_message, show_raw_payload=sidebar_config["show_raw_payload"])
            except Exception as exc:
                st.error(f"❌ 오류가 발생했습니다: {exc}")

st.markdown("---")
st.caption(
    f"Connected to: {sidebar_config['provider'].upper()} ({sidebar_config['model_name']}) | session={get_session_id()}"
)

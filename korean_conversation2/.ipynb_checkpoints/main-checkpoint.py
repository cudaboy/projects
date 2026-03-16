import streamlit as st
import requests
from sidebar import render_sidebar

# =====================================================================
# 1. 백엔드(FastAPI) 통신 주소 설정
# =====================================================================
# chatbot_api.py가 8000번 포트에서 돌아가고 있으므로 해당 주소를 지정합니다.
# 주의: chatbot_api.py의 엔드포인트가 "@app.post('/chat/')" 이므로 끝에 슬래시(/)가 꼭 있어야 합니다.
API_URL = "http://127.0.0.1:8000/chat/"

# =====================================================================
# 2. Streamlit 기본 페이지 설정
# =====================================================================
st.set_page_config(page_title="한국어 RAG 챗봇", page_icon="🤖")
st.title("🤖 한국어 RAG 챗봇 (Transformer)")

# =====================================================================
# 3. 사이드바(Sidebar) 렌더링
# =====================================================================
# 프로젝트 1에서 만드셨던 sidebar.py의 함수를 호출하여 화면 왼쪽에 설정 창을 붙입니다.
render_sidebar()

# =====================================================================
# 4. 세션 상태 (Session State) 초기화
# =====================================================================
# 사용자가 새로고침을 하거나 화면이 바뀌어도 기존 대화 기록이 날아가지 않도록
# Streamlit의 'session_state'라는 특수 메모리 공간에 대화 내역을 리스트 형태로 저장합니다.
if "messages" not in st.session_state:
    st.session_state.messages = []

# =====================================================================
# 5. 기존 채팅 기록 화면에 출력
# =====================================================================
# 저장된 대화 내역(messages)을 하나씩 꺼내서, 사용자는 오른쪽, 챗봇은 왼쪽 등
# 역할(role)에 맞는 채팅 말풍선(chat_message)으로 화면에 그려줍니다.
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# =====================================================================
# 6. 채팅 입력창 및 백엔드 통신 로직
# =====================================================================
# 화면 맨 아래에 채팅 입력창(chat_input)을 만듭니다. 사용자가 엔터를 치면 prompt 변수에 텍스트가 담깁니다.
if prompt := st.chat_input("질문을 입력해주세요..."):
    
    # 1) 사용자 질문을 화면에 표시하고 세션 메모리에 저장합니다.
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # 2) 챗봇(assistant) 말풍선을 미리 띄워두고 답변을 기다립니다.
    with st.chat_message("assistant"):
        # 답변이 올 때까지 뱅글뱅글 도는 로딩 애니메이션(spinner)을 띄웁니다.
        with st.spinner("문서를 검색하고 답변을 생성하는 중입니다..."):
            try:
                # 💡 [핵심] 챗봇의 뇌(rag_handler)가 있는 FastAPI 백엔드 서버로 질문을 전송(POST)합니다!
                # 딕셔너리 형태로 {"question": "사용자 질문"}을 만들어 JSON으로 보냅니다.
                response = requests.post(API_URL, json={"question": prompt})
                
                # 서버가 정상(200 OK)적으로 답변을 돌려주었을 때의 처리
                if response.status_code == 200:
                    # JSON 응답 데이터에서 'answer' 키의 값을 뽑아옵니다.
                    answer = response.json().get("answer", "오류: 답변이 없습니다.")
                    
                    # 화면에 답변을 출력하고, 세션 메모리에 챗봇의 답변을 기록합니다.
                    st.markdown(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                else:
                    # FastAPI 서버에서 에러(예: 422 양식 오류, 500 내부 오류 등)가 났을 때 화면에 붉은색으로 띄웁니다.
                    st.error(f"서버 응답 오류: {response.status_code}")
            
            except requests.exceptions.ConnectionError:
                # 아예 FastAPI 서버(8000번 포트)가 꺼져있어서 연결조차 안 될 때의 에러 처리입니다.
                st.error("🔌 FastAPI 서버에 연결할 수 없습니다. 백엔드 서버가 켜져 있는지 확인해주세요!")
            except Exception as e:
                # 그 외의 알 수 없는 에러 처리입니다.
                st.error(f"알 수 없는 에러가 발생했습니다: {e}")
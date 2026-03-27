import streamlit as st
import sys
import requests
import os

# 파이썬이 프로젝트 최상위 폴더를 인식할 수 있도록 경로 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ==========================================
# 1. 페이지 및 환경 설정 (반드시 가장 먼저 호출!)
# ==========================================
st.set_page_config(
    page_title="AI 주식 애널리스트", 
    page_icon="📈", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# 모듈화된 UI 컴포넌트 및 사이드바 함수 불러오기
from frontend.sidebar import render_sidebar
from frontend.ui_components import (
    render_fund_manager_report, 
    render_cfo_analysis, 
    render_analyst_news, 
    render_trader_chart
)

# 사이드바 렌더링 및 유저가 설정한 동적 LLM 파라미터 값 받아오기
user_settings = render_sidebar()

# 백엔드 API 주소 (Docker 환경에서는 컨테이너명 'backend' 사용, 로컬 테스트 시 localhost)
API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000/api/v1")

# ==========================================
# 2. 메인 UI (헤더 및 입력창)
# ==========================================
st.title("📈 Multi-Agent AI PB")
st.markdown("""
**LangGraph 기반의 3명의 AI 전문가(CFO, 애널리스트, 트레이더)가 종목을 다각도로 분석하고, 
총괄 펀드매니저가 최종 투자 리포트를 작성해 드립니다.**
""")

st.divider()

# 사용자 입력부 (st.form 적용으로 엔터키 인식)
with st.form(key="search_form"):
    col1, col2 = st.columns([3, 1])
    with col1:
        company_name = st.text_input("🔍 분석할 종목명을 입력하세요 (예: 삼성전자, SK텔레콤, 현대차)", placeholder="종목명 입력...")
    with col2:
        st.write("") # 버튼 높이 맞춤용 빈 공백
        st.write("")
        # st.button 대신 st.form_submit_button 사용
        analyze_btn = st.form_submit_button("심층 분석 시작", use_container_width=True, type="primary")

# ==========================================
# 3. 분석 실행 및 결과 렌더링
# ==========================================
if analyze_btn:
    if not company_name.strip():
        st.warning("⚠️ 종목명을 입력해주세요.")
    else:
        # 분석 진행 상태 표시 (LangGraph 에이전트들이 도구를 사용하는 동안 대기)
        with st.status(f"🚀 '{company_name}'에 대한 AI 에이전트 분석을 진행 중입니다...", expanded=True) as status:
            st.write("🕵️‍♂️ CFO 에이전트가 재무제표를 스크래핑 중입니다...")
            st.write("📰 Analyst 에이전트가 최신 뉴스를 검색 중입니다...")
            st.write("📈 Trader 에이전트가 최근 300일 차트 데이터를 분석 중입니다...")
            
            try:
                # FastAPI 백엔드로 POST 요청 전송 (종목명과 유저 설정값 모두 전달)
                response = requests.post(
                    f"{API_BASE_URL}/analyze", 
                    json={
                        "company_name": company_name,
                        "model_settings": user_settings
                    },
                    timeout=120 # 크롤링 및 LLM 생성 시간을 고려해 타임아웃을 넉넉히 부여
                )
                response.raise_for_status() # HTTP 400~500번대 에러 발생 시 예외 처리
                
                result_json = response.json()
                
                if result_json.get("status") == "error":
                    status.update(label="분석 실패", state="error", expanded=True)
                    st.error(result_json.get("error_message", "알 수 없는 오류가 발생했습니다."))
                else:
                    status.update(label="분석 완료!", state="complete", expanded=False)
                    data = result_json.get("data", {})
                    
                    st.success(f"✅ '{company_name}'에 대한 종합 분석 리포트가 완성되었습니다.")
                    
                    # 4개의 탭으로 결과 분리하여 모듈화된 UI 컴포넌트로 예쁘게 렌더링
                    tab1, tab2, tab3, tab4 = st.tabs([
                        "💡 최종 투자 의견", 
                        "💼 재무 분석 (CFO)", 
                        "📰 뉴스 모멘텀 (Analyst)", 
                        "📊 기술적 진단 (Trader)"
                    ])
                    
                    with tab1:
                        render_fund_manager_report(data.get("final_report", "결과 없음"))
                        
                    with tab2:
                        render_cfo_analysis(data.get("company_finance", "결과 없음"))
                        
                    with tab3:
                        render_analyst_news(data.get("company_news", "결과 없음"))
                        
                    with tab4:
                        render_trader_chart(data.get("company_stock", "결과 없음"))

            except requests.exceptions.Timeout:
                status.update(label="시간 초과", state="error", expanded=True)
                st.error("요청 시간이 초과되었습니다. 데이터 수집이나 AI 생성에 시간이 너무 오래 걸렸습니다.")
            except Exception as e:
                status.update(label="오류 발생", state="error", expanded=True)
                st.error(f"서버 통신 중 오류가 발생했습니다: {str(e)}")
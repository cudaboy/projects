import streamlit as st
import sys
import requests
import pandas as pd
import os

# 파이썬이 프로젝트 최상위 폴더를 인식할 수 있도록 경로 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ==========================================
# 1. 페이지 및 환경 설정
# ==========================================
st.set_page_config(page_title="분석 히스토리", page_icon="📊", layout="wide")

# 백엔드 API 주소 (Docker 환경에서는 컨테이너명 'backend' 사용, 로컬 테스트 시 localhost)
API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000/api/v1")

st.title("📊 AI 분석 히스토리 대시보드")
st.markdown("과거에 AI 에이전트들이 분석했던 종목들의 통계를 확인하고, 상세 리포트를 다시 열람할 수 있습니다.")

# ==========================================
# 2. 데이터 불러오기 (캐싱 적용)
# ==========================================
@st.cache_data(ttl=60) # 데이터베이스 부하를 줄이기 위해 1분간 결과를 캐싱합니다.
def load_history_data():
    try:
        # 백엔드에서 전체 히스토리 데이터를 가져오는 API 호출
        response = requests.get(f"{API_BASE_URL}/history", timeout=10)
        response.raise_for_status()
        return response.json().get("data", [])
    except Exception as e:
        st.error(f"히스토리 데이터를 불러오는 중 오류가 발생했습니다: {str(e)}")
        return []

history_data = load_history_data()

# ==========================================
# 3. 대시보드 렌더링
# ==========================================
if not history_data:
    st.info("아직 분석된 종목 히스토리가 없습니다. 홈 화면에서 첫 분석을 시작해 보세요!")
else:
    # 데이터를 Pandas DataFrame으로 변환하여 다루기 쉽게 만듭니다.
    df = pd.DataFrame(history_data)
    
    # 시간 데이터 포맷팅 (예: 2026-03-25T14:30:00 -> 2026-03-25 14:30:00)
    if 'created_at' in df.columns:
        df['created_at'] = pd.to_datetime(df['created_at']).dt.strftime('%Y-%m-%d %H:%M:%S')

    # --- 상단: 핵심 통계 지표 (Metrics) ---
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(label="총 분석 건수", value=f"{len(df)} 건")
    with col2:
        st.metric(label="분석된 고유 종목 수", value=f"{df['company_name'].nunique()} 개")
    with col3:
        latest_date = df['created_at'].max() if 'created_at' in df.columns else "-"
        st.metric(label="최근 분석일", value=latest_date)

    st.divider()

    # --- 중단: 분석 히스토리 검색 및 상세 열람 ---
    st.subheader("📂 과거 분석 리포트 열람")
    
    # Selectbox를 이용해 특정 날짜의 특정 종목 분석 결과를 선택
    # 최신순으로 정렬하여 보여주기 위해 [::-1] 적용
    selected_record = st.selectbox(
        "열람할 분석 기록을 선택하세요:", 
        options=df.to_dict('records')[::-1],
        format_func=lambda x: f"[{x.get('created_at', '')}] {x.get('company_name', '알 수 없음')}"
    )

    if selected_record:
        st.markdown(f"### 💡 {selected_record['company_name']} 최종 리포트")
        st.info(selected_record.get('final_report', '최종 리포트 내용이 없습니다.'))
        
        # 세부 에이전트들의 리포트는 공간을 많이 차지하므로 Expander(접기/펴기)로 숨겨둡니다.
        with st.expander("🔍 세부 에이전트 분석 내용 보기"):
            tab1, tab2, tab3 = st.tabs(["💼 재무 분석 (CFO)", "📰 뉴스 모멘텀 (Analyst)", "📈 기술적 진단 (Trader)"])
            with tab1:
                st.write(selected_record.get('finance_summary', '내용 없음'))
            with tab2:
                st.write(selected_record.get('news_summary', '내용 없음'))
            with tab3:
                st.write(selected_record.get('stock_summary', '내용 없음'))
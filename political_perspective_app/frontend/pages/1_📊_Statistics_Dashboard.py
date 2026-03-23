# 1_📊_Statistics_Dashboard.py : 전처리 기능을 활용해 DB에 쌓인 로그를 분석하고 시각화

import streamlit as st
import pandas as pd
import sqlite3

st.set_page_config(page_title="통계 대시보드", page_icon="📊", layout="wide")
st.title("📊 이슈 트렌드 및 통계 대시보드")
st.markdown("사용자들이 검색한 키워드와 도출된 평가 기조의 통계적 분포를 분석합니다.")

# DB에서 Pandas DataFrame으로 데이터 로드
@st.cache_data(ttl=60) # 1분마다 캐시 갱신
def load_data():
    conn = sqlite3.connect("../backend/spectrum.db")
    query = """
        SELECT q.keyword, q.created_at, r.categories
        FROM user_queries q
        JOIN analysis_results r ON q.id = r.query_id
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df

try:
    df = load_data()
    
    if not df.empty:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🔍 주요 연관 범주 빈도 분석")
            # "사회, 경제" -> ['사회', '경제'] 로 분리 후 개수 카운트
            category_series = df['categories'].str.split(', ').explode()
            cat_counts = category_series.value_counts().reset_index()
            cat_counts.columns = ['범주', '빈도수']
            
            st.bar_chart(cat_counts.set_index('범주'))
            
        with col2:
            st.subheader("📈 최근 검색 키워드 트렌드")
            st.dataframe(df[['created_at', 'keyword']].sort_values(by='created_at', ascending=False).head(10), use_container_width=True)
            
        st.divider()
        st.subheader("💡 텍스트 마이닝 요약")
        st.info(f"현재까지 총 **{len(df)}건**의 이슈 분석이 수행되었습니다. 가장 많이 분석된 기조는 **{cat_counts.iloc[0]['범주']}**입니다.")
        
    else:
        st.info("아직 누적된 데이터가 없습니다. 메인 페이지에서 이슈를 분석해 보세요!")

except Exception as e:
    st.error(f"데이터베이스를 불러오는 중 오류가 발생했습니다: {e}")
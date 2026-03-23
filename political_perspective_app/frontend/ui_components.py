# ui_components.py : ui_components.py

import streamlit as st

def render_analysis_result(result: dict):
    """백엔드에서 받은 결과 JSON을 Streamlit 화면에 렌더링하는 함수입니다."""
    
    # 상단: 분석 대상 및 주요 연관 범주 출력
    st.success(f"📌 **분석 대상 이슈:** {result['issue_target']}")
    categories_str = ", ".join(result['key_categories'])
    st.info(f"🔍 **주요 연관 범주:** {categories_str}")
    
    st.divider()
    
    # 하단: 3개 진영 평가 결과 가로로 나란히 출력
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("🔴 진보 논객")
        prog = result['progressive']
        st.markdown(f"**핵심 논조:**\n> {prog['core_tone']}")
        st.markdown(f"**상세 분석:**\n{prog['detailed_analysis']}")
        
    with col2:
        st.subheader("🟣 중도 논객")
        cent = result['centrist']
        st.markdown(f"**핵심 논조:**\n> {cent['core_tone']}")
        st.markdown(f"**상세 분석:**\n{cent['detailed_analysis']}")
        
    with col3:
        st.subheader("🔵 보수 논객")
        cons = result['conservative']
        st.markdown(f"**핵심 논조:**\n> {cons['core_tone']}")
        st.markdown(f"**상세 분석:**\n{cons['detailed_analysis']}")
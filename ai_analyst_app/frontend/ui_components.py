import streamlit as st
import re

def render_fund_manager_report(report_text: str):
    """
    총괄 펀드매니저의 최종 리포트를 눈에 띄게 렌더링합니다.
    (매수/관망/매도 의견을 시각적으로 강조)
    """
    st.markdown("---")
    
    # 정규식을 통해 Buy / Hold / Sell 키워드가 있으면 태그 형태로 강조
    if re.search(r'(?i)\b(buy|매수)\b', report_text):
        st.success("🎯 **최종 투자의견: BUY (매수)**")
    elif re.search(r'(?i)\b(sell|매도)\b', report_text):
        st.error("⚠️ **최종 투자의견: SELL (매도)**")
    elif re.search(r'(?i)\b(hold|관망)\b', report_text):
        st.warning("⚖️ **최종 투자의견: HOLD (관망)**")
        
    # 본문 내용을 깔끔한 박스 안에 배치
    with st.container(border=True):
        st.markdown("#### 📝 총괄 요약 리포트")
        st.markdown(report_text)

def render_cfo_analysis(finance_text: str):
    """
    CFO 에이전트의 재무 분석 결과를 렌더링합니다.
    """
    with st.container(border=True):
        st.markdown("### 💼 재무 건전성 및 수익성 분석")
        st.caption("기업의 최근 연간 및 분기별 재무제표 데이터를 바탕으로 분석된 결과입니다.")
        st.divider()
        
        # LLM이 마크다운 표나 불릿 포인트를 출력할 때 깨지지 않도록 처리
        st.markdown(finance_text)

def render_analyst_news(news_text: str):
    """
    애널리스트 에이전트의 최신 뉴스 모멘텀 분석을 렌더링합니다.
    """
    with st.container(border=True):
        st.markdown("### 📰 최신 뉴스 모멘텀 & 리스크")
        st.caption("최근 보도된 뉴스 기사들을 스크래핑하여 기업의 현재 상태와 미래 전망을 도출했습니다.")
        st.divider()
        
        # 만약 Pydantic으로 구조화된 JSON이 문자열 형태로 들어왔다면 
        # 이를 딕셔너리로 변환해서 예쁘게 뿌려줄 수도 있습니다. (여기서는 마크다운 가정)
        st.markdown(news_text)

def render_trader_chart(stock_text: str):
    """
    트레이더 에이전트의 차트 및 거래량 기술적 분석을 렌더링합니다.
    """
    with st.container(border=True):
        st.markdown("### 📈 기술적 진단 (차트 & 거래량)")
        st.caption("최근 300일간의 주가 흐름, 지지선/저항선 및 투자자 심리를 분석했습니다.")
        st.divider()
        
        st.markdown(stock_text)

def render_error_state(error_message: str):
    """
    에러 발생 시 사용자에게 보여줄 친절한 UI
    """
    st.error("🚨 분석 중 문제가 발생했습니다.")
    with st.expander("에러 상세 내용 보기"):
        st.code(error_message)
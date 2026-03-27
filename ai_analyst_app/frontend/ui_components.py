import streamlit as st
import re

def render_fund_manager_report(report_text: str):
    st.markdown("---")
    
    # '최종 투자의견:' 또는 '최종 투자 의견:' 뒤에 나오는 단어 그룹을 추출하는 정규식
    match = re.search(r'최종\s*투자\s*의견\s*:\s*(.*)', report_text, re.IGNORECASE)
    
    if match:
        verdict = match.group(1).upper()
        if "BUY" in verdict or "매수" in verdict:
            st.success("🎯 **최종 투자의견: BUY (매수)**")
        elif "SELL" in verdict or "매도" in verdict:
            st.error("⚠️ **최종 투자의견: SELL (매도)**")
        elif "HOLD" in verdict or "관망" in verdict:
            st.warning("⚖️ **최종 투자의견: HOLD (관망)**")
        else:
            st.info(f"💡 **최종 투자의견: {match.group(1)}**") # 예외적인 단어일 경우
    else:
        # 혹시라도 양식에 안 맞춰서 출력했을 경우를 대비한 기본값
        st.info("💡 **최종 투자의견: 본문 참조**")
        
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
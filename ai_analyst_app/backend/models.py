from sqlalchemy import Column, Integer, String, Text, DateTime
from sqlalchemy.sql import func
from backend.database import Base

class StockAnalysisHistory(Base):
    """
    사용자가 검색한 종목명과 각 에이전트들의 분석 결과, 
    그리고 생성 시간을 저장하는 히스토리 테이블입니다.
    """
    __tablename__ = "stock_analysis_history"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    
    # 검색한 종목명 (예: 광전자, 삼성전자)
    company_name = Column(String(100), index=True, nullable=False)
    
    # 1. 재무 에이전트 (CFO) 분석 결과
    finance_summary = Column(Text, nullable=True)
    
    # 2. 뉴스 에이전트 (Analyst) 분석 결과
    news_summary = Column(Text, nullable=True)
    
    # 3. 주가/차트 에이전트 (Trader) 분석 결과
    stock_summary = Column(Text, nullable=True)
    
    # 4. Valuation Agent의 DCF-lite/상대가치 요약
    valuation_summary = Column(Text, nullable=True)

    # 5. Risk Manager의 결정론적 점수/리스크 요약
    risk_summary = Column(Text, nullable=True)

    # 6. 알고리즘 버전 및 점수 JSON
    investment_score = Column(Text, nullable=True)

    # 7. Phase 5 백테스트/성과 추적용 분석 시점 메타데이터
    stock_code = Column(String(20), nullable=True)
    analysis_price = Column(String(50), nullable=True)
    analysis_price_date = Column(String(20), nullable=True)

    # 8. 알고리즘 버전/평점 스냅샷
    algorithm_version = Column(String(100), nullable=True)
    rating = Column(String(20), nullable=True)

    # 9. 총괄 펀드매니저의 최종 리포트 및 투자의견
    final_report = Column(Text, nullable=True)
    
    # 분석이 실행된 시간 (서버 시간 기준 자동 생성)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
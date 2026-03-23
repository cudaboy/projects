# models.py : 질문(Query)과 세 가지 관점의 분석 결과(Result)를 1:1 관계로 매핑하는 구조

from sqlalchemy import Column, Integer, String, Text, DateTime, ForeignKey
from sqlalchemy.orm import declarative_base, relationship
from datetime import datetime

Base = declarative_base()

class UserQuery(Base):
    __tablename__ = "user_queries"
    
    id = Column(Integer, primary_key=True, index=True)
    keyword = Column(String, index=True, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # 분석 결과와 1:1 관계
    analysis_result = relationship("AnalysisResult", back_populates="query", uselist=False)

class AnalysisResult(Base):
    __tablename__ = "analysis_results"
    
    id = Column(Integer, primary_key=True, index=True)
    query_id = Column(Integer, ForeignKey("user_queries.id"))
    categories = Column(String) # 예: "사회, 경제"
    
    prog_tone = Column(Text)
    prog_analysis = Column(Text)
    cent_tone = Column(Text)
    cent_analysis = Column(Text)
    cons_tone = Column(Text)
    cons_analysis = Column(Text)
    
    query = relationship("UserQuery", back_populates="analysis_result")
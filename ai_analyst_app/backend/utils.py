import logging
import re
from functools import wraps
from sqlalchemy.orm import Session
from typing import Dict, Any

from backend.models import StockAnalysisHistory

# 로거 설정
logger = logging.getLogger(__name__)

# ==========================================
# 1. 크롤링 및 API 호출 예외 처리 데코레이터
# ==========================================
def safe_crawl(default_return: str = "데이터를 불러오는 데 실패했습니다."):
    """
    스크래핑이나 외부 API 호출 시 발생할 수 있는 각종 예외(Timeout, ConnectionError 등)를 
    안전하게 잡아내고 로깅한 뒤, 에이전트가 당황하지 않도록 기본 문자열을 반환하는 데코레이터입니다.
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.error(f"[{func.__name__}] 실행 중 오류 발생: {str(e)}\nArgs: {args}\nKwargs: {kwargs}")
                return default_return
        return wrapper
    return decorator

# ==========================================
# 2. 데이터 전처리 (텍스트 정제) 함수
# ==========================================
def clean_text(text: str) -> str:
    """
    크롤링한 텍스트에서 불필요한 공백, 탭, 줄바꿈, 특수기호 등을 제거하여 
    LLM이 토큰을 효율적으로 소모하고 문맥을 잘 파악할 수 있도록 정제합니다.
    """
    if not text:
        return ""
    
    # 여러 개의 줄바꿈이나 공백을 하나의 띄어쓰기로 압축
    text = re.sub(r'\s+', ' ', text)
    # HTML 태그 잔재미나 불필요한 특수문자 제거 (필요에 따라 정규식 조정)
    text = re.sub(r'<[^>]+>', '', text)
    
    return text.strip()

# ==========================================
# 3. DB 저장 헬퍼 함수
# ==========================================
def save_analysis_history(db: Session, company_name: str, analysis_data: Dict[str, Any]) -> StockAnalysisHistory:
    """
    LangGraph 에이전트들의 분석이 모두 완료된 후, 
    그 결과(딕셔너리)를 받아 데이터베이스(SQLite/PostgreSQL 등)에 안전하게 저장합니다.
    """
    try:
        # 모델 객체 생성
        history_record = StockAnalysisHistory(
            company_name=company_name,
            finance_summary=analysis_data.get('company_finance', ''),
            news_summary=analysis_data.get('company_news', ''),
            stock_summary=analysis_data.get('company_stock', ''),
            final_report=analysis_data.get('final_report', '')
        )
        
        # DB 세션에 추가 및 커밋
        db.add(history_record)
        db.commit()
        db.refresh(history_record) # 생성된 ID 등 최신 상태 반영
        
        logger.info(f"[{company_name}] 분석 결과 DB 저장 성공 (ID: {history_record.id})")
        return history_record
        
    except Exception as e:
        db.rollback() # 오류 발생 시 롤백하여 DB 락 방지
        logger.error(f"[{company_name}] 분석 결과 DB 저장 실패: {str(e)}")
        raise e
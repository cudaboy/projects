import os
import logging
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session

# 🌟 LangSmith 실시간 추적을 위한 컨텍스트 매니저 추가
from langchain_core.tracers.context import tracing_v2_enabled

# 🌟 수정됨: update_runtime_settings 함수 임포트
from backend.core.config import settings, update_runtime_settings
from backend.core.schemas import StockRequest, StockResponse, AgentAnalysisData
from backend.services.graph import app as workflow_app

# DB 연결 및 테이블 구조(models)를 가져옵니다.
from backend.database import get_db, engine
from backend.models import StockAnalysisHistory
from backend import models

# ==========================================
# 🚨 DB 테이블 자동 생성
# ==========================================
models.Base.metadata.create_all(bind=engine)

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.PROJECT_VERSION,
    description="다중 에이전트(CFO, Analyst, Trader, Fund Manager) 기반 주식 분석 API"
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", tags=["Health Check"])
async def root():
    return {"message": f"{settings.PROJECT_NAME} API Server is running!"}

# ==========================================
# 메인 분석 API 엔드포인트
# ==========================================
@app.post(f"{settings.API_V1_STR}/analyze", response_model=StockResponse, tags=["Analysis"])
async def analyze_stock(request: StockRequest, db: Session = Depends(get_db)):
    """
    사이드바에서 입력받은 API 키들을 런타임에 업데이트하고 분석을 수행합니다.
    """
    logger.info(f"분석 요청 수신 - 종목명: {request.company_name}")
    
    ms = request.model_settings
    
    # 🌟 [중요] 런타임 설정 업데이트 로직
    # 프론트엔드(사이드바)에서 넘어온 키들을 시스템 환경 변수 및 settings 객체에 동기화합니다.
    if ms:
        runtime_keys = {
            "OPENAI_API_KEY": ms.openai_api_key or ms.custom_api_key,
            "LANGCHAIN_API_KEY": ms.langsmith_api_key,
            # 스키마(StockRequest/ModelSettings)에 아래 필드들이 정의되어 있다고 가정합니다.
            "NAVER_CLIENT_ID": getattr(ms, "naver_client_id", None),
            "NAVER_CLIENT_SECRET": getattr(ms, "naver_client_secret", None)
        }
        update_runtime_settings(runtime_keys)
        logger.info("사이드바 입력값으로 런타임 환경 변수 업데이트 완료")

    # LangSmith 추적 활성화 여부 확인
    use_tracing = ms and ms.use_langsmith and ms.langsmith_api_key
    
    try:
        initial_state = {
            "question": request.company_name,
            "model_settings": ms.model_dump() if ms else {}
        }
        
        # 🌟 LangSmith 추적 로직 (컨텍스트 매니저 사용)
        if use_tracing:
            logger.info(f"LangSmith 추적 활성화: 프로젝트명 '{settings.LANGCHAIN_PROJECT}'")
            with tracing_v2_enabled(project_name=settings.LANGCHAIN_PROJECT):
                result = await workflow_app.ainvoke(initial_state)
        else:
            logger.info("LangSmith 추적이 비활성화되었습니다.")
            result = await workflow_app.ainvoke(initial_state)
        
        # 결과 매핑
        analysis_data = AgentAnalysisData(
            company_finance=result.get("company_finance", "재무 데이터 분석 결과를 불러오지 못했습니다."),
            company_news=result.get("company_news", "뉴스 데이터 분석 결과를 불러오지 못했습니다."),
            company_stock=result.get("company_stock", "차트 데이터 분석 결과를 불러오지 못했습니다."),
            final_report=result.get("final_report", "최종 펀드매니저 리포트 생성에 실패했습니다.")
        )
        
        # DB 저장
        new_history = StockAnalysisHistory(
            company_name=request.company_name,
            finance_summary=analysis_data.company_finance,
            news_summary=analysis_data.company_news,
            stock_summary=analysis_data.company_stock,
            final_report=analysis_data.final_report
        )
        db.add(new_history)
        db.commit()
        db.refresh(new_history)
        
        logger.info(f"분석 완료 및 DB 저장 성공 - 종목명: {request.company_name} (ID: {new_history.id})")
        
        return StockResponse(
            status="success",
            company_name=request.company_name,
            data=analysis_data
        )
        
    except Exception as e:
        logger.error(f"분석 중 오류 발생 ({request.company_name}): {str(e)}")
        return StockResponse(
            status="error",
            company_name=request.company_name,
            error_message=f"서버 내부 오류가 발생했습니다: {str(e)}"
        )
        
@app.get(f"{settings.API_V1_STR}/history", tags=["History"])
async def get_analysis_history(db: Session = Depends(get_db)):
    """DB에 저장된 모든 과거 주식 분석 히스토리를 반환합니다."""
    records = db.query(StockAnalysisHistory).all()
    data = [
        {
            "id": r.id,
            "company_name": r.company_name,
            "finance_summary": r.finance_summary,
            "news_summary": r.news_summary,
            "stock_summary": r.stock_summary,
            "final_report": r.final_report,
            "created_at": r.created_at.isoformat() if r.created_at else None
        }
        for r in records
    ]
    return {"status": "success", "data": data}
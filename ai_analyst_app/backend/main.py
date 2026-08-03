import os
import json
import logging
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from sqlalchemy import text

# 🌟 LangSmith 실시간 추적을 위한 컨텍스트 매니저 추가
from langchain_core.tracers.context import tracing_v2_enabled

# 🌟 수정됨: update_runtime_settings 함수 임포트
from backend.core.config import settings, update_runtime_settings
from backend.core.schemas import StockRequest, StockResponse, AgentAnalysisData
from backend.services.graph import app as workflow_app
from backend.services.backtest import price_snapshot, summarize_algorithm_performance
from backend.services.tools import get_code

# DB 연결 및 테이블 구조(models)를 가져옵니다.
from backend.database import get_db, engine
from backend.models import StockAnalysisHistory
from backend import models

# ==========================================
# 🚨 DB 테이블 자동 생성 및 경량 SQLite 마이그레이션
# ==========================================
models.Base.metadata.create_all(bind=engine)

def ensure_algorithm_columns():
    """기존 SQLite DB에도 신규 알고리즘 컬럼을 안전하게 추가합니다."""
    with engine.begin() as conn:
        existing = {row[1] for row in conn.execute(text("PRAGMA table_info(stock_analysis_history)"))}
        for column_name in (
            "valuation_summary",
            "risk_summary",
            "investment_score",
            "stock_code",
            "analysis_price",
            "analysis_price_date",
            "algorithm_version",
            "rating",
        ):
            if column_name not in existing:
                conn.execute(text(f"ALTER TABLE stock_analysis_history ADD COLUMN {column_name} TEXT"))

ensure_algorithm_columns()

def _extract_score_field(score_json: str | None, field: str):
    try:
        payload = json.loads(score_json or "{}")
        return payload.get(field)
    except Exception:
        return None

def _build_analysis_price_metadata(company_name: str) -> dict:
    """Resolve stock code and latest close snapshot for Phase 5 backtesting."""
    try:
        code = get_code.invoke({"company_name": company_name})
        if not isinstance(code, str) or not code.isdigit() or len(code) != 6:
            return {"stock_code": None, "analysis_price": None, "analysis_price_date": None}
        snapshot = price_snapshot(code)
        if snapshot.get("status") != "success":
            return {"stock_code": code, "analysis_price": None, "analysis_price_date": None}
        return {
            "stock_code": code,
            "analysis_price": snapshot.get("price"),
            "analysis_price_date": snapshot.get("price_date"),
        }
    except Exception as exc:
        logger.warning("Phase 5 가격 메타데이터 생성 실패: %s", exc)
        return {"stock_code": None, "analysis_price": None, "analysis_price_date": None}

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
        provider = (ms.provider or "OpenAI").lower()
        runtime_keys = {
            "LANGCHAIN_API_KEY": ms.langsmith_api_key,
            "NAVER_CLIENT_ID": getattr(ms, "naver_client_id", None),
            "NAVER_CLIENT_SECRET": getattr(ms, "naver_client_secret", None),
        }
        if provider == "openai":
            runtime_keys["OPENAI_API_KEY"] = ms.openai_api_key or ms.custom_api_key
        elif provider in {"grok", "xai"}:
            runtime_keys["XAI_API_KEY"] = ms.custom_api_key
        elif provider == "openrouter":
            runtime_keys["OPENROUTER_API_KEY"] = ms.custom_api_key
        elif provider == "anthropic":
            runtime_keys["ANTHROPIC_API_KEY"] = ms.custom_api_key
        elif provider == "google gemini":
            runtime_keys["GOOGLE_API_KEY"] = ms.custom_api_key
        elif provider == "ollama" and ms.base_url:
            runtime_keys["OLLAMA_BASE_URL"] = ms.base_url
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

        # Phase 5: 분석 시점 종목코드/가격 스냅샷 저장
        price_meta = _build_analysis_price_metadata(request.company_name)

        # 결과 매핑
        analysis_data = AgentAnalysisData(
            company_finance=result.get("company_finance", "재무 데이터 분석 결과를 불러오지 못했습니다."),
            company_news=result.get("company_news", "뉴스 데이터 분석 결과를 불러오지 못했습니다."),
            company_stock=result.get("company_stock", "차트 데이터 분석 결과를 불러오지 못했습니다."),
            company_valuation=result.get("company_valuation"),
            stock_code=price_meta.get("stock_code"),
            analysis_price=price_meta.get("analysis_price"),
            analysis_price_date=price_meta.get("analysis_price_date"),
            company_risk=result.get("company_risk"),
            investment_score=result.get("investment_score"),
            final_report=result.get("final_report", "최종 펀드매니저 리포트 생성에 실패했습니다.")
        )

        # DB 저장
        new_history = StockAnalysisHistory(
            company_name=request.company_name,
            finance_summary=analysis_data.company_finance,
            news_summary=analysis_data.company_news,
            stock_summary=analysis_data.company_stock,
            valuation_summary=analysis_data.company_valuation,
            stock_code=analysis_data.stock_code,
            analysis_price=str(analysis_data.analysis_price) if analysis_data.analysis_price is not None else None,
            analysis_price_date=analysis_data.analysis_price_date,
            risk_summary=analysis_data.company_risk,
            investment_score=analysis_data.investment_score,
            algorithm_version=_extract_score_field(analysis_data.investment_score, "algorithm_version"),
            rating=_extract_score_field(analysis_data.investment_score, "rating_hint"),
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
            "valuation_summary": r.valuation_summary,
            "stock_code": r.stock_code,
            "analysis_price": r.analysis_price,
            "analysis_price_date": r.analysis_price_date,
            "risk_summary": r.risk_summary,
            "investment_score": r.investment_score,
            "algorithm_version": r.algorithm_version,
            "rating": r.rating,
            "final_report": r.final_report,
            "created_at": r.created_at.isoformat() if r.created_at else None
        }
        for r in records
    ]
    return {"status": "success", "data": data}

@app.get(f"{settings.API_V1_STR}/performance", tags=["History"])
async def get_algorithm_performance(db: Session = Depends(get_db)):
    """알고리즘 버전별 분석 건수, 평점 분포, 평균 점수/리스크를 반환합니다."""
    records = db.query(StockAnalysisHistory).all()
    return summarize_algorithm_performance(records)

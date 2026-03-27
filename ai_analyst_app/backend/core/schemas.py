from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

# ==========================================
# 1. API 통신용 스키마 (FastAPI Request/Response)
# ==========================================

class ModelSettings(BaseModel):
    """
    프론트엔드 사이드바에서 설정한 LLM 동적 파라미터 스키마
    """
    provider: str = Field(default="OpenAI", description="LLM 제공자 (OpenAI, Anthropic, Google Gemini)")
    model_name: str = Field(default="gpt-4o", description="사용할 세부 모델명")
    temperature: float = Field(default=0.2, description="창의성 조절 파라미터 (0.0 ~ 1.0)")
    
    # 🌟 [추가] OpenAI API 키 필드 (main.py의 AttributeError 해결용)
    openai_api_key: Optional[str] = Field(default=None, description="OpenAI 전용 API 키")
    
    # 🌟 기존 LLM API 키 관련 필드 유지
    custom_api_key: Optional[str] = Field(default=None, description="개별 API 키 (설정하지 않으면 서버 .env 키 사용)")
    llm_key_env_name: Optional[str] = Field(default=None, description="입력된 API 키의 환경 변수명 (예: OPENAI_API_KEY)")
    
    # 🌟 LangSmith 설정 필드
    use_langsmith: Optional[bool] = Field(default=False, description="LangSmith 추적 활성화 여부")
    langsmith_api_key: Optional[str] = Field(default=None, description="LangSmith 개별 API 키")
    
    # 🌟 Naver Search 설정 필드
    naver_client_id: Optional[str] = Field(default=None, description="네이버 뉴스 검색용 Client ID")
    naver_client_secret: Optional[str] = Field(default=None, description="네이버 뉴스 검색용 Client Secret")

class StockRequest(BaseModel):
    """
    프론트엔드(Streamlit)에서 백엔드(FastAPI)로 
    분석을 요청할 때 사용하는 입력 데이터 스키마
    """
    company_name: str = Field(
        ..., 
        description="분석할 기업의 이름", 
        examples=["광전자", "삼성전자"]
    )
    model_settings: Optional[ModelSettings] = Field(
        default=None, 
        description="사용자가 선택한 모델 설정값"
    )

class AgentAnalysisData(BaseModel):
    """
    LangGraph 워크플로우를 거쳐 나온 각 에이전트들의 분석 결과 데이터
    """
    company_finance: str = Field(description="CFO 에이전트의 재무제표 종합 평가")
    company_news: str = Field(description="애널리스트 에이전트의 뉴스 기반 모멘텀 분석")
    company_stock: str = Field(description="트레이더 에이전트의 과거 주가 기반 기술적 진단")
    final_report: str = Field(description="펀드매니저 에이전트의 최종 종합 리포트 및 투자 의견")

class StockResponse(BaseModel):
    """
    백엔드에서 프론트엔드로 최종 반환하는 응답 스키마
    """
    status: str = Field(description="API 처리 상태 (예: success, error)")
    company_name: str = Field(description="분석을 수행한 기업명")
    data: Optional[AgentAnalysisData] = Field(default=None, description="각 에이전트별 분석 결과 데이터")
    error_message: Optional[str] = Field(default=None, description="에러 발생 시 에러 메시지")


# ==========================================
# 2. LLM 구조화된 출력(Structured Output)용 스키마
# ==========================================

class CompanyAnalysisNews(BaseModel):
    """
    뉴스 에이전트(Analyst)가 스크래핑한 텍스트를 읽고, 
    Pydantic을 통해 강제로 정형화된 JSON 형태로 파싱하기 위한 스키마
    """
    current_status: str = Field(description="현재 기업의 상태 요약")
    future_outlook: str = Field(description="미래 전망 및 성장 가능성")
    business_health: str = Field(description="사업의 건전성 및 리스크 요인")
    core_keyword: List[str] = Field(description="기업과 관련된 핵심 뉴스 키워드 리스트")
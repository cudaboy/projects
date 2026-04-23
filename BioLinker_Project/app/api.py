"""
===============================================================================
[File Role]
이 파일(api.py)은 BioLinker 프로젝트의 '백엔드 API 서버(Backend API Server)' 역할을 담당합니다.

[상세 설명]
1. 멀티 LLM 동적 수용: 프론트엔드에서 사용자가 입력한 API Key와 선택한 모델(OpenAI, Anthropic, Google, Grok)을 
   전달받아 실시간으로 해당 LLM 객체를 생성하고 에이전트 워크플로우에 주입합니다.
2. 하이브리드 리소스 관리: 
   - 서버 시작 시(Startup): 로컬 임베딩 모델(ModernBERT)과 벡터 DB를 메모리에 1회 로드합니다. (약 14분 소요)
   - 쿼리 요청 시(Request): 전달받은 인증 정보를 바탕으로 추론용 LLM을 즉석에서 구성하여 답변을 합성합니다.
3. 보안 및 관찰성: 사용자의 API 키가 서버 로그에 남지 않도록 주의하며, 선택적으로 LangSmith 추적을 활성화합니다.
===============================================================================
"""

import os
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, List

# BioLinker 코어 모듈 임포트
try:
    from biolinker.database import BioDatabaseManager
    from biolinker.agents import BioAgentManager
    from biolinker.workflow import create_workflow
except ImportError:
    import sys
    from pathlib import Path
    # 프로젝트 루트 경로를 sys.path에 추가하여 모듈 참조 해결
    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from biolinker.database import BioDatabaseManager
    from biolinker.agents import BioAgentManager
    from biolinker.workflow import create_workflow

# LangChain 기반 LLM 클래스들
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ---------------------------------------------------------
# 1. API 입출력 데이터 스키마 (Schema)
# ---------------------------------------------------------
class QueryRequest(BaseModel):
    """프론트엔드(main.py)로부터 전달받는 요청 규격"""
    question: str
    provider: str  # openai, anthropic, google, grok
    model_name: str
    api_key: str
    use_langsmith: bool = False
    langsmith_api_key: Optional[str] = ""

class QueryResponse(BaseModel):
    """프론트엔드로 반환하는 결과 규격"""
    question: str
    route: str
    final_answer: str
    logs: List[str] = []

# ---------------------------------------------------------
# 2. FastAPI 초기화 및 글로벌 상태 관리
# ---------------------------------------------------------
app = FastAPI(
    title="BioLinker API (Dynamic LLM Mode)",
    description="사용자 인증 기반 멀티 LLM 하이브리드 RAG API",
    version="1.2.0"
)

# 무거운 로컬 리소스(Vector DB, Embedding)는 전역 변수로 관리하여 재사용
db_manager = None

@app.on_event("startup")
async def startup_event():
    """서버 기동 시 로컬 임베딩 모델 및 DB 로드 (최초 1회 실행)"""
    global db_manager
    logging.info("🚀 BioLinker 백엔드 시작. 로컬 리소스 초기화 중 (ModernBERT 로딩 포함)...")
    try:
        # 이 과정에서 ModernBERT 모델을 메모리에 올리므로 사양에 따라 시간이 소요됩니다.
        db_manager = BioDatabaseManager()
        logging.info("✅ 로컬 데이터베이스 및 임베딩 엔진 준비 완료.")
    except Exception as e:
        logging.error(f"❌ 시스템 초기화 실패: {e}")
        raise e

# ---------------------------------------------------------
# 3. LLM 동적 생성 헬퍼 함수
# ---------------------------------------------------------
def get_dynamic_llm(request: QueryRequest):
    """사용자가 UI에서 입력한 정보를 바탕으로 LangChain LLM 객체를 생성합니다."""
    provider = request.provider.lower()
    
    # LangSmith 모니터링 활성화 시 환경 변수 설정
    if request.use_langsmith and request.langsmith_api_key:
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_API_KEY"] = request.langsmith_api_key
        os.environ["LANGCHAIN_PROJECT"] = "BioLinker-Production"
    else:
        os.environ["LANGCHAIN_TRACING_V2"] = "false"

    try:
        if provider == "openai":
            return ChatOpenAI(
                model=request.model_name,
                openai_api_key=request.api_key,
                temperature=0
            )
        elif provider == "anthropic":
            return ChatAnthropic(
                model=request.model_name,
                anthropic_api_key=request.api_key,
                temperature=0
            )
        elif provider == "google":
            return ChatGoogleGenerativeAI(
                model=request.model_name,
                google_api_key=request.api_key,
                temperature=0
            )
        elif provider == "grok":
            # Grok(xAI)은 OpenAI 호환 API를 사용하므로 ChatOpenAI로 구성 가능
            return ChatOpenAI(
                model=request.model_name,
                openai_api_key=request.api_key,
                openai_api_base="https://api.x.ai/v1",
                temperature=0
            )
        else:
            raise ValueError(f"지원하지 않는 제공자입니다: {provider}")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"LLM 초기화 실패: {str(e)}")

# ---------------------------------------------------------
# 4. API 엔드포인트
# ---------------------------------------------------------
@app.get("/", tags=["Health"])
def health_check():
    return {"status": "running", "local_db": "connected" if db_manager else "loading"}

@app.post("/api/v1/query", response_model=QueryResponse, tags=["Search"])
async def process_query(request: QueryRequest):
    """프론트엔드의 요청을 받아 실시간 추론 워크플로우를 실행합니다."""
    
    if db_manager is None:
        raise HTTPException(status_code=503, detail="시스템이 아직 초기화 중입니다. 잠시 후 다시 시도하세요.")

    logging.info(f"🔍 쿼리 수신: {request.question} (Model: {request.model_name})")

    try:
        # 1. 요청에 맞는 LLM 동적 생성
        dynamic_llm = get_dynamic_llm(request)
        
        # 2. 에이전트 매니저 생성 및 LLM 주입
        # BioAgentManager를 요청 시마다 생성하여 주입된 LLM을 사용하게 함
        agent_manager = BioAgentManager(db_manager)
        agent_manager.llm = dynamic_llm  # 에이전트의 두뇌를 동적으로 교체
        
        # 3. 워크플로우 생성 및 실행
        workflow_app = create_workflow(agent_manager)
        
        # LangGraph State 실행
        initial_state = {"question": request.question}
        final_state = workflow_app.invoke(initial_state)
        
        return QueryResponse(
            question=request.question,
            route=final_state.get("route", "unknown"),
            final_answer=final_state.get("final_answer", "답변 생성에 실패했습니다."),
            logs=final_state.get("logs", [])
        )

    except Exception as e:
        logging.error(f"❌ 추론 실패: {e}")
        raise HTTPException(status_code=500, detail=f"에이전트 처리 중 오류 발생: {str(e)}")

# 서버 직접 실행 로직
if __name__ == "__main__":
    import uvicorn
    # python app/api.py 명령어로 실행
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=False)
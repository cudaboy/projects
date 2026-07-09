"""
BioLinker FastAPI backend.

개선 사항
- /health/live, /health/ready 분리
- request_id / latency / structured response 추가
- retrieval_mode / top_k 실험 파라미터 지원
- startup readiness 상태 추적
"""

from __future__ import annotations

import logging
import os
import subprocess
import time
import uuid
from typing import Any, List, Optional

from dotenv import dotenv_values
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from langchain_core.runnables import RunnableLambda

try:
    from biolinker.database import BioDatabaseManager
    from biolinker.agents import BioAgentManager
    from biolinker.workflow import create_workflow
    from biolinker import config
except ImportError:
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from biolinker.database import BioDatabaseManager
    from biolinker.agents import BioAgentManager
    from biolinker.workflow import create_workflow
    from biolinker import config

from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("biolinker.api")


class QueryRequest(BaseModel):
    question: str
    provider: str
    model_name: str
    api_key: str = ""
    auth_mode: str = Field(default="api_key", description="api_key, env, hermes_openai_key, oauth 중 하나")
    use_langsmith: bool = False
    langsmith_api_key: Optional[str] = ""
    retrieval_mode: str = Field(default=config.DEFAULT_RETRIEVAL_MODE)
    top_k: int = Field(default=config.DEFAULT_TOP_K, ge=1, le=20)
    session_id: Optional[str] = None


class Citation(BaseModel):
    doc_id: str
    title: str
    journal: str = ""
    year: str = ""
    score: float = 0.0
    chunk_id: str = ""


class GraphEdge(BaseModel):
    source: str = ""
    subject: str
    object: str
    relation: str
    doc_id: str = ""
    journal: str = ""
    year: str = ""
    evidence_text: str = ""
    confidence: str = ""
    hop: int = 1


class QueryResponse(BaseModel):
    request_id: str
    question: str
    route: str
    route_confidence: float = 0.0
    final_answer: str
    citations: List[Citation] = []
    retrieved_doc_ids: List[str] = []
    graph_edges: List[GraphEdge] = []
    logs: List[str] = []
    latency_ms: float = 0.0
    safety_flag: str = "ok"
    no_answer_reason: str = ""


app = FastAPI(
    title="BioLinker API (Production-Ready Hybrid RAG)",
    description="사용자 인증 기반 멀티 LLM 하이브리드 RAG API",
    version="2.0.0",
)

db_manager: Optional[BioDatabaseManager] = None
startup_error: Optional[str] = None
startup_started_at: float = 0.0
startup_completed_at: float = 0.0


@app.on_event("startup")
async def startup_event():
    global db_manager, startup_error, startup_started_at, startup_completed_at
    startup_started_at = time.time()
    logger.info("🚀 BioLinker backend startup 시작")
    try:
        db_manager = BioDatabaseManager()
        startup_completed_at = time.time()
        startup_error = None
        logger.info("✅ 로컬 데이터베이스 및 임베딩 엔진 준비 완료")
    except Exception as exc:
        startup_error = str(exc)
        logger.exception("❌ 시스템 초기화 실패: %s", exc)


@app.get("/health/live", tags=["Health"])
def health_live():
    return {"status": "live", "uptime_ready": startup_completed_at > 0}


@app.get("/health/ready", tags=["Health"])
def health_ready():
    if startup_error:
        return {"status": "error", "ready": False, "error": startup_error}
    return {
        "status": "ready" if db_manager else "initializing",
        "ready": db_manager is not None,
        "embedding_model": config.EMBEDDING_MODEL,
        "embedding_device": config.EMBEDDING_DEVICE,
        "startup_seconds": round((startup_completed_at or time.time()) - startup_started_at, 3) if startup_started_at else None,
    }


@app.get("/", tags=["Health"])
def health_check():
    ready = db_manager is not None and not startup_error
    return {"status": "running", "ready": ready, "local_db": "connected" if ready else "loading"}


def _oauth_responses_invoke(model: str, access_token: str, input_value: Any, base_url: str = "https://api.openai.com/v1") -> str:
    from openai import OpenAI

    text = _prompt_value_to_text(input_value)
    client = OpenAI(api_key=access_token, base_url=base_url)
    response = client.responses.create(
        model=model,
        input=text,
        temperature=0,
    )
    if getattr(response, "output_text", None):
        return response.output_text
    return str(response)


def _prompt_value_to_text(value: Any) -> str:
    if hasattr(value, "to_messages"):
        parts = []
        for msg in value.to_messages():
            role = getattr(msg, "type", "message")
            content = getattr(msg, "content", "")
            parts.append(f"[{role}] {content}")
        return "\n".join(parts)
    if isinstance(value, list):
        return "\n".join(str(item) for item in value)
    return str(value)


def _mask_secret(value: str) -> str:
    if not value:
        return "missing"
    return f"set(len={len(value)}, prefix={value[:3]}...)"


def _load_dotenv_key(path: str, key_name: str) -> str:
    env_path = os.path.expanduser(path)
    if not os.path.exists(env_path):
        return ""
    return str(dotenv_values(env_path).get(key_name) or "").strip()


def _resolve_openai_api_key(request: QueryRequest) -> str:
    """Resolve an OpenAI-compatible API key for development and production checks.

    Precedence:
    1. explicit request.api_key
    2. process OPENAI_API_KEY
    3. BioLinker project .env
    4. Hermes biolinker profile .env
    5. Hermes default .env
    """
    candidates = [
        ("request.api_key", request.api_key),
        ("process OPENAI_API_KEY", os.getenv("OPENAI_API_KEY", "")),
        ("project .env OPENAI_API_KEY", _load_dotenv_key(str(config.BASE_DIR / ".env"), "OPENAI_API_KEY")),
        ("biolinker profile .env OPENAI_API_KEY", _load_dotenv_key("~/.hermes/profiles/biolinker/.env", "OPENAI_API_KEY")),
        ("hermes default .env OPENAI_API_KEY", _load_dotenv_key("~/.hermes/.env", "OPENAI_API_KEY")),
    ]
    for source, value in candidates:
        key = str(value or "").strip()
        if key:
            logger.info("OpenAI API key resolved from %s: %s", source, _mask_secret(key))
            return key
    return ""


def _run_hermes_json(script: str) -> dict:
    """Run a tiny helper inside the Hermes checkout so OAuth tokens are resolved by Hermes itself."""
    hermes_root = os.getenv("HERMES_AGENT_DIR", os.path.expanduser("~/.hermes/hermes-agent"))
    result = subprocess.run(
        ["python3", "-c", script],
        cwd=hermes_root,
        text=True,
        capture_output=True,
        timeout=60,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError((result.stderr or result.stdout or "Hermes OAuth helper failed").strip())
    import json

    return json.loads(result.stdout)


def _resolve_hermes_openai_oauth_token() -> str:
    script = r'''
import json
from agent.credential_pool import load_pool
pool = load_pool("openai-codex")
entry = None
if hasattr(pool, "try_refresh_current"):
    refreshed = pool.try_refresh_current()
    entry = refreshed[0] if isinstance(refreshed, tuple) else refreshed
if entry is None and hasattr(pool, "select"):
    selected = pool.select()
    entry = selected[0] if isinstance(selected, tuple) else selected
if entry is None and hasattr(pool, "peek"):
    entry = pool.peek()
if entry is None and hasattr(pool, "current"):
    entry = pool.current
if entry is None and hasattr(pool, "entries"):
    entries = list(pool.entries)
    entry = entries[0] if entries else None
if entry is None:
    raise SystemExit("openai-codex OAuth credential not found. Run `hermes auth add openai-codex` or `hermes login --provider openai-codex`.")
token = getattr(entry, "access_token", None) or getattr(entry, "runtime_api_key", None)
if not token and isinstance(entry, dict):
    token = entry.get("access_token") or entry.get("runtime_api_key")
if not token:
    raise SystemExit("openai-codex OAuth access token could not be resolved")
print(json.dumps({"access_token": token}))
'''
    return _run_hermes_json(script)["access_token"]


def get_dynamic_llm(request: QueryRequest):
    provider = request.provider.lower()
    auth_mode = (request.auth_mode or "api_key").lower()
    if request.use_langsmith and request.langsmith_api_key:
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_API_KEY"] = request.langsmith_api_key
        os.environ["LANGCHAIN_PROJECT"] = "BioLinker-Production"
    else:
        os.environ["LANGCHAIN_TRACING_V2"] = "false"

    if provider in {"openai-oauth", "openai_codex", "openai-codex"} or (provider == "openai" and auth_mode == "oauth"):
        token = request.api_key or _resolve_hermes_openai_oauth_token()
        return RunnableLambda(lambda input_value: _oauth_responses_invoke(request.model_name, token, input_value))

    if provider == "openai":
        resolved_api_key = _resolve_openai_api_key(request) if auth_mode in {"api_key", "env", "hermes_openai_key", "hermes-openai-key", "hermes"} else request.api_key
        if not resolved_api_key:
            raise HTTPException(
                status_code=400,
                detail="OpenAI API key가 필요합니다. request.api_key, OPENAI_API_KEY, BioLinker .env, Hermes profile .env, ~/.hermes/.env 중 하나에 설정하세요.",
            )
        return ChatOpenAI(model=request.model_name, openai_api_key=resolved_api_key, temperature=0)
    if provider == "anthropic":
        if not request.api_key:
            raise HTTPException(status_code=400, detail="Anthropic API key가 필요합니다.")
        return ChatAnthropic(model=request.model_name, anthropic_api_key=request.api_key, temperature=0)
    if provider == "google":
        if not request.api_key:
            raise HTTPException(status_code=400, detail="Google API key가 필요합니다.")
        return ChatGoogleGenerativeAI(model=request.model_name, google_api_key=request.api_key, temperature=0)
    if provider == "grok":
        if not request.api_key:
            raise HTTPException(status_code=400, detail="xAI/Grok API key가 필요합니다.")
        return ChatOpenAI(
            model=request.model_name,
            openai_api_key=request.api_key,
            openai_api_base="https://api.x.ai/v1",
            temperature=0,
        )
    raise HTTPException(status_code=400, detail=f"지원하지 않는 제공자입니다: {provider}")


@app.post("/api/v1/query", response_model=QueryResponse, tags=["Search"])
async def process_query(request: QueryRequest):
    global db_manager
    if db_manager is None:
        raise HTTPException(status_code=503, detail="시스템이 아직 초기화 중입니다. 잠시 후 다시 시도하세요.")

    request_id = str(uuid.uuid4())[:8]
    start = time.perf_counter()
    logger.info(
        "request_id=%s provider=%s model=%s retrieval_mode=%s top_k=%s question=%s",
        request_id,
        request.provider,
        request.model_name,
        request.retrieval_mode,
        request.top_k,
        request.question[:180],
    )
    try:
        dynamic_llm = get_dynamic_llm(request)
        agent_manager = BioAgentManager(db_manager)
        agent_manager.llm = dynamic_llm
        workflow_app = create_workflow(agent_manager)
        final_state = workflow_app.invoke(
            {
                "question": request.question,
                "route_override": request.retrieval_mode,
                "top_k": request.top_k,
            }
        )
        latency_ms = round((time.perf_counter() - start) * 1000, 2)
        logger.info(
            "request_id=%s route=%s confidence=%.2f latency_ms=%.2f safety=%s",
            request_id,
            final_state.get("route", "unknown"),
            float(final_state.get("route_confidence", 0.0)),
            latency_ms,
            final_state.get("safety_flag", "ok"),
        )
        return QueryResponse(
            request_id=request_id,
            question=request.question,
            route=final_state.get("route", "unknown"),
            route_confidence=float(final_state.get("route_confidence", 0.0)),
            final_answer=final_state.get("final_answer", "답변 생성에 실패했습니다."),
            citations=final_state.get("citations", []),
            retrieved_doc_ids=final_state.get("retrieved_doc_ids", []),
            graph_edges=final_state.get("graph_edges", []),
            logs=final_state.get("logs", []),
            latency_ms=latency_ms,
            safety_flag=final_state.get("safety_flag", "ok"),
            no_answer_reason=final_state.get("no_answer_reason", ""),
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("❌ request_id=%s 추론 실패: %s", request_id, exc)
        raise HTTPException(status_code=500, detail=f"에이전트 처리 중 오류 발생: {exc}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api:app", host=config.API_HOST, port=config.API_PORT, reload=False)

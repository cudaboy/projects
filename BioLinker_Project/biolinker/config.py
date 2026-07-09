"""
BioLinker 전역 설정.

- 환경 변수 로드
- 데이터/로그/평가 디렉토리 관리
- 임베딩/검색/그래프/평가 하이퍼파라미터 관리
- GPU/CPU fallback 지원
"""

from __future__ import annotations

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

if os.getenv("HF_TOKEN"):
    os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")


def _get_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


def _get_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


def _get_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _resolve_embedding_device() -> str:
    explicit = os.getenv("BIO_EMBEDDING_DEVICE")
    if explicit:
        return explicit
    try:
        import torch  # type: ignore

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "02.라벨링데이터"
RAW_JSON_PATH = RAW_DATA_DIR
PROCESSED_DATA_DIR = DATA_DIR / "processed"
PARSED_CSV_PATH = PROCESSED_DATA_DIR / "parsed_entities_relations.csv"
PARSED_DOCUMENTS_PATH = PROCESSED_DATA_DIR / "parsed_documents.csv"
CHROMA_DB_DIR = DATA_DIR / "chroma_db"
KNOWLEDGE_GRAPH_PATH = DATA_DIR / "knowledge_graph.gml"
SESSION_LOG_DIR = DATA_DIR / "session_logs"
EVAL_DIR = DATA_DIR / "eval"
EVAL_DATASET_PATH = EVAL_DIR / "queries.jsonl"
EVAL_REPORT_PATH = PROCESSED_DATA_DIR / "ragas_evaluation_report.csv"
EVAL_SUMMARY_PATH = PROCESSED_DATA_DIR / "ragas_evaluation_summary.json"

for directory in [
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    CHROMA_DB_DIR,
    SESSION_LOG_DIR,
    EVAL_DIR,
]:
    directory.mkdir(parents=True, exist_ok=True)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
LLM_MODEL = os.getenv("BIO_LLM_MODEL", "gpt-4o-mini")
LLM_TEMPERATURE = _get_float("BIO_LLM_TEMPERATURE", 0.0)
MAX_TOKENS = _get_int("BIO_MAX_TOKENS", 1800)
EMBEDDING_MODEL = os.getenv(
    "BIO_EMBEDDING_MODEL",
    "thomas-sounack/BioClinical-ModernBERT-base",
)
EMBEDDING_DEVICE = _resolve_embedding_device()
EMBEDDING_BATCH_SIZE = _get_int("BIO_EMBEDDING_BATCH_SIZE", 8 if EMBEDDING_DEVICE == "cuda" else 2)
EMBEDDING_NORMALIZE = _get_bool("BIO_EMBEDDING_NORMALIZE", True)

CHUNK_SIZE = _get_int("BIO_CHUNK_SIZE", 1200)
CHUNK_OVERLAP = _get_int("BIO_CHUNK_OVERLAP", 150)
MAX_CHUNKS_PER_DOC = _get_int("BIO_MAX_CHUNKS_PER_DOC", 8)
CHROMA_COLLECTION_NAME = os.getenv("BIO_CHROMA_COLLECTION", "bio_literature_collection")
RETRIEVER_K = _get_int("BIO_RETRIEVER_K", 6)
RERANK_TOP_K = _get_int("BIO_RERANK_TOP_K", 5)
MIN_VECTOR_SCORE = _get_float("BIO_MIN_VECTOR_SCORE", 0.05)

GRAPH_MAX_HOPS = _get_int("BIO_GRAPH_MAX_HOPS", 2)
GRAPH_MAX_EDGES = _get_int("BIO_GRAPH_MAX_EDGES", 12)
GRAPH_PREVIEW_EDGES = _get_int("BIO_GRAPH_PREVIEW_EDGES", 3)
ROUTE_LOW_CONFIDENCE_THRESHOLD = _get_float("BIO_ROUTE_LOW_CONFIDENCE_THRESHOLD", 0.55)
NO_ANSWER_ENTITY_HIT_THRESHOLD = _get_int("BIO_NO_ANSWER_ENTITY_HIT_THRESHOLD", 1)

RECURSION_LIMIT = _get_int("BIO_RECURSION_LIMIT", 12)

API_HOST = os.getenv("BIO_API_HOST", "0.0.0.0")
API_PORT = _get_int("BIO_API_PORT", 8000)
APP_TITLE = "Bio-Linker: 논문 연계 질병-약물 타겟 추적 시스템"
REQUEST_TIMEOUT_SECONDS = _get_int("BIO_REQUEST_TIMEOUT_SECONDS", 120)

DEFAULT_RETRIEVAL_MODE = os.getenv("BIO_DEFAULT_RETRIEVAL_MODE", "auto")
DEFAULT_TOP_K = _get_int("BIO_DEFAULT_TOP_K", RETRIEVER_K)

SYNTHESIS_TEMPLATE = """
당신은 바이오메디컬 도메인 전문 연구 보조 AI입니다.
반드시 제공된 검색 근거만 사용해 답변하세요.
출력은 다음 순서를 따르세요.
1. 핵심 결론
2. 기전 설명
3. 문헌 근거
4. 그래프 근거
5. 한계 및 추가 확인 포인트

근거가 부족하면 확신하는 표현을 금지하고, 부족한 이유를 명확히 적으세요.
문헌 근거는 논문 제목 또는 출처 ID를 괄호로 명시하세요.
"""

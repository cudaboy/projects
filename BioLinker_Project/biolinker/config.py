"""
===============================================================================
[File Role]
이 파일(config.py)은 BioLinker 프로젝트의 전역 설정(Global Configuration)을 담당합니다.

[상세 설명]
1. 경로 관리: 원천 데이터, 전처리된 데이터, Vector DB, 지식 그래프 파일의 저장 경로를
   일괄적으로 관리하여, 다른 모듈(data_parser, agents 등)에서 경로를 하드코딩하지 않도록 합니다.
2. 환경 변수: .env 파일과 연동하여 API Key 등 민감한 정보를 안전하게 로드합니다.
3. 모델 파라미터: RAG 시스템 및 LangGraph 멀티에이전트에서 사용할 LLM 모델명, 임베딩 모델,
   Chunk 사이즈 등의 하이퍼파라미터를 중앙 통제합니다.
4. 확장성: 추후 한미약품의 내부 로컬 LLM이나 다른 Open-source 모델로 교체할 때, 
   비즈니스 로직 수정 없이 이 파일의 'LLM_MODEL' 변수만 변경하여 즉시 대응할 수 있도록 설계되었습니다.
===============================================================================
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# 1. 환경 변수 로드
load_dotenv()

if os.getenv("HF_TOKEN"):
    os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")

# ---------------------------------------------------------
# 2. 디렉토리 및 파일 경로 설정
# ---------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent

# 데이터 저장소 기본 경로
DATA_DIR = BASE_DIR / "data"

# 원천 데이터 (AI-Hub 라벨링 데이터 .zip 파일들이 위치한 경로)
RAW_DATA_DIR = DATA_DIR / "02.라벨링데이터"
# build_index.py 와의 호환성을 위한 별칭(Alias)
RAW_JSON_PATH = RAW_DATA_DIR 

# 전처리 완료된 데이터 경로
PROCESSED_DATA_DIR = DATA_DIR / "processed"
PARSED_CSV_PATH = PROCESSED_DATA_DIR / "parsed_entities_relations.csv"

# 데이터베이스 경로
CHROMA_DB_DIR = DATA_DIR / "chroma_db"
KNOWLEDGE_GRAPH_PATH = DATA_DIR / "knowledge_graph.gml"

# 디렉토리 자동 생성
for directory in [RAW_DATA_DIR, PROCESSED_DATA_DIR, CHROMA_DB_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------
# 3. LLM 및 임베딩 모델 설정
# ---------------------------------------------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
LLM_MODEL = "gpt-4o-mini"
LLM_TEMPERATURE = 0.0
MAX_TOKENS = 1500
EMBEDDING_MODEL = "thomas-sounack/BioClinical-ModernBERT-base" # 의학 특화 임베딩 모델 (BioClinical ModernBERT) 적용

# ---------------------------------------------------------
# 4. 데이터 전처리 및 Vector DB (RAG) 설정
# ---------------------------------------------------------
CHUNK_SIZE = 2000
CHUNK_OVERLAP = 200
CHROMA_COLLECTION_NAME = "bio_literature_collection"
RETRIEVER_K = 5

# ---------------------------------------------------------
# 5. LangGraph 에이전트 프롬프트 설정
# ---------------------------------------------------------
RECURSION_LIMIT = 10

ROUTER_PROMPT = """
당신은 제약 R&D 임상이행 파트의 인텐트 라우터(Intent Router)입니다.
사용자의 질문이 논문 문헌 검색이 필요한지, 질병-유전자-약물 간의 관계(Graph) 탐색이 필요한지,
혹은 둘 다 필요한지 판단하여 다음 단계를 지시하세요.
단, 사용자의 질문이 의학, 약학, 생물학, 신약 개발과 전혀 관련 없는 일상적인 질문이거나 엉뚱한 내용이라면 반드시 'irrelevant'라고 출력하세요.
"""

SYNTHESIZER_PROMPT = """
당신은 한미약품의 데이터 기반 임상 연구원입니다.
Vector DB에서 검색된 [문헌 정보]와 Knowledge Graph에서 검색된 [관계 정보]를 종합하여,
사용자가 타겟 기전(MoA)을 쉽게 이해할 수 있도록 구조화된 리포트 형태로 답변을 작성하세요.
반드시 출처(Reference)를 명시해야 합니다.
"""

API_HOST = "0.0.0.0"
API_PORT = 8000
APP_TITLE = "Bio-Linker: 논문 연계 질병-약물 타겟 추적 시스템"
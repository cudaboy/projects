"""
===============================================================================
[File Role]
이 파일(build_index.py)은 BioLinker 프로젝트의 '초기 데이터 파이프라인 자동화'를 담당합니다.

[상세 설명]
1. 목적: 프로젝트를 처음 세팅할 때, 원천 데이터(JSON)에서부터 RAG를 위한 Vector DB와
   지식 그래프(Graph DB) 구축까지의 전 과정을 한 번에 실행하는 원클릭 스크립트입니다.
2. 주요 흐름 (Workflow):
   - Step 1: data_parser.py를 호출하여 원천 논문 데이터를 정형화(CSV)
   - Step 2: database.py를 호출하여 문헌 텍스트를 Chroma DB에 임베딩
   - Step 3: database.py를 호출하여 개체-관계 데이터를 NetworkX 그래프로 구축
===============================================================================
"""

import sys
import logging
import time
from pathlib import Path

# 스크립트를 프로젝트 루트 위치에서 실행하지 않더라도, 
# 프로젝트 최상단 디렉토리를 인식하여 biolinker 모듈을 정상적으로 import 할 수 있도록 경로 추가
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

# 핵심 코어 모듈들 불러오기
try:
    from biolinker import config
    from biolinker.data_parser import BioDataParser
    from biolinker.database import BioDatabaseManager
except ImportError as e:
    logging.error(f"모듈을 불러오는 중 에러가 발생했습니다: {e}")
    logging.info("💡 팁: PYTHONPATH가 설정되어 있는지, 프로젝트 루트 디렉토리에서 실행 중인지 확인하세요.")
    sys.exit(1)

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    logging.info("🚀 BioLinker 초기 데이터 인덱싱 파이프라인을 시작합니다.")
    start_time = time.time()

    # ---------------------------------------------------------
    # 1. 데이터 파싱 (Data Parsing & Curation)
    # ---------------------------------------------------------
    logging.info("==================================================")
    logging.info("[Phase 1] 원천 데이터 파싱 및 정제")
    logging.info("==================================================")
    parser = BioDataParser(
        raw_json_path=config.RAW_JSON_PATH,
        parsed_csv_path=config.PARSED_CSV_PATH
    )
    # 파이프라인 실행: 로드 -> 파싱 -> CSV 저장
    parser.run_pipeline()

    # ---------------------------------------------------------
    # 2. 하이브리드 DB 구축 (Vector DB & Knowledge Graph)
    # ---------------------------------------------------------
    logging.info("==================================================")
    logging.info("[Phase 2] 하이브리드 데이터베이스 구축")
    logging.info("==================================================")
    db_manager = BioDatabaseManager()
    
    # 문헌 데이터 경로와 관계 데이터 경로는 data_parser가 저장한 경로를 사용
    docs_csv_path = config.PROCESSED_DATA_DIR / "parsed_documents.csv"
    relations_csv_path = config.PARSED_CSV_PATH

    # Vector DB 구축 (Chroma)
    logging.info("-> 문헌 데이터 임베딩 시작...")
    db_manager.build_vector_db(docs_csv_path)

    # Knowledge Graph 구축 (NetworkX)
    logging.info("-> 관계 데이터 그래프 변환 시작...")
    db_manager.build_knowledge_graph(relations_csv_path)

    # ---------------------------------------------------------
    # 3. 인덱싱 완료 보고
    # ---------------------------------------------------------
    elapsed_time = time.time() - start_time
    logging.info("==================================================")
    logging.info(f"✅ 모든 데이터 파싱 및 DB 구축(Indexing)이 성공적으로 완료되었습니다! (소요 시간: {elapsed_time:.2f}초)")
    logging.info("💡 Next Step: `python run.py --api` 와 `python run.py --ui`를 실행하여 챗봇을 테스트해 보세요.")
    logging.info("==================================================")

if __name__ == "__main__":
    main()
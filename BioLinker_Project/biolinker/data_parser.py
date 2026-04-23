"""
===============================================================================
[File Role]
이 파일(data_parser.py)은 BioLinker 프로젝트의 '데이터 추출 및 정제(Parsing & Curation)'를 담당합니다.

[상세 설명]
1. 목적: AI-Hub의 '바이오·의료 논문 간 연계분석 데이터(JSON)' 구조를 파싱하여,
   RAG 검색용 문헌 텍스트와 지식 그래프(Knowledge Graph) 구축용 관계(Relation) 데이터로 분리합니다.
2. 기능:
   - 원천 JSON 데이터 로드 및 무결성 검증
   - 논문 메타데이터(제목, 초록 등) 추출 -> Vector DB 임베딩용 텍스트 확보
   - 개체(Entity: 질병, 약물, 유전자) 및 관계(Relation: 기전) 추출 -> Graph DB용 엣지 확보
===============================================================================
"""

import json
import logging
import zipfile
import pandas as pd
from pathlib import Path

try:
    from biolinker import config
except ImportError:
    import config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class BioDataParser:
    def __init__(self, raw_json_path: Path, parsed_csv_path: Path):
        """
        :param raw_json_path: '02.라벨링데이터' 디렉토리 경로 (Training, Validation 포함)
        """
        self.raw_data_dir = raw_json_path
        self.parsed_csv_path = parsed_csv_path
        self.documents = []  
        self.relations = []  

    def process_zip_files(self):
        """디렉토리 및 하위 디렉토리(Training, Validation 등) 내의 모든 .zip 파일을 순회하며 JSON 데이터를 파싱합니다."""
        if not self.raw_data_dir.exists() or not self.raw_data_dir.is_dir():
            logging.error(f"데이터 폴더를 찾을 수 없습니다: {self.raw_data_dir}")
            return

        # 🔥 [수정됨] glob 대신 rglob을 사용하여 하위 폴더(Training, Validation)까지 모두 재귀 탐색합니다.
        zip_files = list(self.raw_data_dir.rglob("*.zip"))
        
        if not zip_files:
            logging.warning(f"'{self.raw_data_dir}' 경로 및 하위 폴더에 ZIP 파일이 없습니다.")
            return

        logging.info(f"🔍 총 {len(zip_files)}개의 ZIP 파일(Training/Validation 포함)을 발견했습니다. 파싱을 시작합니다.")

        for zip_path in zip_files:
            # 파일 경로에서 Training인지 Validation인지 파악하여 로그에 표시
            folder_type = zip_path.parent.name
            logging.info(f"📦 압축 파일 파싱 중... [{folder_type}] : {zip_path.name}")
            try:
                with zipfile.ZipFile(zip_path, 'r') as z:
                    # 압축 파일 내의 모든 .json 파일 찾기
                    json_files = [f for f in z.namelist() if f.endswith('.json')]
                    
                    for json_filename in json_files:
                        with z.open(json_filename) as f:
                            # 바이트 데이터를 utf-8로 디코딩하여 JSON 로드
                            data = json.loads(f.read().decode('utf-8'))
                            self.parse_single_json(data)
            except Exception as e:
                logging.error(f"{zip_path.name} 파일 처리 중 오류 발생: {e}")

        logging.info(f"✅ 전체 데이터 파싱 완료: 총 {len(self.documents)}개 문헌, {len(self.relations)}개 관계 도출.")

    def parse_single_json(self, data: dict):
        """제공된 AI-Hub JSON 스키마에 맞춰 문헌과 관계 데이터를 추출합니다."""
        # 1. 메타데이터 및 텍스트 추출
        doc_id = data.get('sourcid', 'unknown_id')
        text = data.get('text', '')
        
        # 명시적인 title 필드가 없으므로, 저널 이름이나 텍스트 첫 문장을 활용
        journal = data.get('journal', 'Unknown Journal')
        title = f"[{journal}] {text[:100]}..." if len(text) > 100 else text

        if text:
            self.documents.append({
                'doc_id': doc_id,
                'title': title,
                'text': text
            })

        # 2. 개체(Entity) 정보를 ID를 Key로 하는 딕셔너리로 구축
        entity_dict = {}
        for ent in data.get('entities', []):
            ent_id = ent.get('entityId')
            if ent_id:
                entity_dict[ent_id] = {
                    'name': ent.get('entityName'),
                    'type': ent.get('entityType')
                }

        # 3. 관계(Relation) 정보 추출 및 Graph Edge 생성
        for rel in data.get('relation_info', []):
            subj_id = rel.get('subjectID')
            obj_id = rel.get('objectId')
            rel_name = rel.get('rel_name')

            # 주어와 목적어 개체가 모두 정의되어 있는 경우에만 유효한 관계로 취급
            if subj_id in entity_dict and obj_id in entity_dict:
                self.relations.append({
                    'doc_id': doc_id,
                    'subject_name': entity_dict[subj_id]['name'],
                    'subject_type': entity_dict[subj_id]['type'],
                    'relation_type': rel_name,
                    'object_name': entity_dict[obj_id]['name'],
                    'object_type': entity_dict[obj_id]['type']
                })

    def save_processed_data(self):
        """파싱된 데이터를 CSV로 저장합니다."""
        if not self.documents and not self.relations:
            logging.warning("저장할 데이터가 없습니다.")
            return

        self.parsed_csv_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 관계(Graph) 데이터 저장
        relations_df = pd.DataFrame(self.relations)
        relations_df.to_csv(self.parsed_csv_path, index=False, encoding='utf-8-sig')
        logging.info(f"📁 관계 데이터 CSV 저장 완료: {self.parsed_csv_path}")
        
        # 문헌(Vector) 데이터 저장
        documents_df = pd.DataFrame(self.documents)
        docs_path = self.parsed_csv_path.parent / "parsed_documents.csv"
        documents_df.to_csv(docs_path, index=False, encoding='utf-8-sig')
        logging.info(f"📁 문헌 텍스트 CSV 저장 완료: {docs_path}")

    def run_pipeline(self):
        """ZIP 파일 스캔 -> 메모리 파싱 -> CSV 저장 파이프라인"""
        logging.info("데이터 파싱 파이프라인(Curation)을 시작합니다...")
        self.process_zip_files()
        self.save_processed_data()

if __name__ == "__main__":
    parser = BioDataParser(
        raw_json_path=config.RAW_JSON_PATH,
        parsed_csv_path=config.PARSED_CSV_PATH
    )
    parser.run_pipeline()
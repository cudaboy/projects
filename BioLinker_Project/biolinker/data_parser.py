"""
AI-Hub 바이오·의료 JSON 파서를 기반으로 문헌/관계 CSV를 생성한다.

개선 사항
- 문헌 메타데이터 확장(journal/year/study_type/entity tags 등)
- source_zip, entity_count, relation_count 저장
- 그래프 후처리를 위한 normalized subject/object 필드 추가
"""

from __future__ import annotations

import json
import logging
import re
import zipfile
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd

try:
    from biolinker import config
except ImportError:
    import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

DISEASE_HINTS = {"disease", "diseases", "질환", "암", "cancer", "tumor", "syndrome"}
DRUG_HINTS = {"drug", "compound", "chemical", "medicine", "약물", "치료제", "inhibitor", "antibody"}
GENE_HINTS = {"gene", "protein", "target", "biomarker", "유전자", "단백질", "receptor", "mutation"}


class BioDataParser:
    def __init__(self, raw_json_path: Path, parsed_csv_path: Path):
        self.raw_data_dir = raw_json_path
        self.parsed_csv_path = parsed_csv_path
        self.documents: List[dict] = []
        self.relations: List[dict] = []

    @staticmethod
    def normalize_text(value: Optional[str]) -> str:
        value = str(value or "").strip().lower()
        value = re.sub(r"[^0-9a-z가-힣]+", " ", value)
        return re.sub(r"\s+", " ", value).strip()

    @staticmethod
    def _first_non_empty(*values: object) -> str:
        for value in values:
            if value is None:
                continue
            if isinstance(value, str) and value.strip():
                return value.strip()
            if isinstance(value, (int, float)):
                return str(value)
        return ""

    def _infer_study_type(self, text: str) -> str:
        lowered = text.lower()
        if any(keyword in lowered for keyword in ["clinical trial", "phase i", "phase ii", "phase iii", "randomized"]):
            return "clinical_trial"
        if any(keyword in lowered for keyword in ["review", "systematic review", "meta-analysis"]):
            return "review"
        if any(keyword in lowered for keyword in ["in vivo", "in vitro", "mouse model", "cell line", "preclinical"]):
            return "preclinical"
        return "unknown"

    def _infer_language(self, text: str) -> str:
        if re.search(r"[가-힣]", text):
            return "ko"
        return "en"

    def _extract_year(self, data: dict) -> Optional[int]:
        candidate = self._first_non_empty(
            data.get("year"),
            data.get("publication_year"),
            data.get("pub_year"),
            data.get("date"),
            data.get("published_at"),
        )
        match = re.search(r"(19|20)\d{2}", candidate)
        return int(match.group(0)) if match else None

    def _extract_document_text(self, data: dict) -> Tuple[str, str]:
        title = self._first_non_empty(data.get("title"), data.get("article_title"), data.get("paper_title"))
        abstract = self._first_non_empty(data.get("abstract"), data.get("summary"))
        main_text = self._first_non_empty(data.get("text"), data.get("body"), data.get("content"))

        if not title and main_text:
            title = main_text.split(".")[0][:180].strip()
        if abstract and main_text and abstract in main_text:
            text = main_text
        else:
            text = "\n\n".join(part for part in [title, abstract, main_text] if part)
        return title or "Untitled", text.strip()

    def _partition_entities(self, entities: Sequence[dict]) -> Tuple[List[str], List[str], List[str]]:
        diseases: List[str] = []
        drugs: List[str] = []
        genes: List[str] = []
        for ent in entities:
            name = self._first_non_empty(ent.get("entityName"), ent.get("name"))
            ent_type = self.normalize_text(self._first_non_empty(ent.get("entityType"), ent.get("type")))
            if not name:
                continue
            if any(token in ent_type for token in DISEASE_HINTS):
                diseases.append(name)
            elif any(token in ent_type for token in DRUG_HINTS):
                drugs.append(name)
            elif any(token in ent_type for token in GENE_HINTS):
                genes.append(name)
        return sorted(set(diseases)), sorted(set(drugs)), sorted(set(genes))

    def process_zip_files(self):
        if not self.raw_data_dir.exists() or not self.raw_data_dir.is_dir():
            logging.error(f"데이터 폴더를 찾을 수 없습니다: {self.raw_data_dir}")
            return

        zip_files = list(self.raw_data_dir.rglob("*.zip"))
        if not zip_files:
            logging.warning(f"'{self.raw_data_dir}' 경로 및 하위 폴더에 ZIP 파일이 없습니다.")
            return

        logging.info(f"🔍 총 {len(zip_files)}개의 ZIP 파일을 발견했습니다. 파싱을 시작합니다.")
        for zip_path in zip_files:
            folder_type = zip_path.parent.name
            logging.info(f"📦 압축 파일 파싱 중... [{folder_type}] : {zip_path.name}")
            try:
                with zipfile.ZipFile(zip_path, "r") as zf:
                    for json_filename in [name for name in zf.namelist() if name.endswith(".json")]:
                        with zf.open(json_filename) as handle:
                            payload = json.loads(handle.read().decode("utf-8"))
                            self.parse_single_json(payload, source_zip=zip_path.name)
            except Exception as exc:
                logging.error(f"{zip_path.name} 파일 처리 중 오류 발생: {exc}")

        logging.info(f"✅ 전체 데이터 파싱 완료: 총 {len(self.documents)}개 문헌, {len(self.relations)}개 관계 도출.")

    def parse_single_json(self, data: dict, source_zip: str = "unknown.zip"):
        doc_id = self._first_non_empty(data.get("sourcid"), data.get("doc_id"), data.get("id"), "unknown_id")
        journal = self._first_non_empty(data.get("journal"), data.get("journal_name"), "Unknown Journal")
        title, text = self._extract_document_text(data)
        if not text:
            return

        entities = list(data.get("entities", []))
        relations = list(data.get("relation_info", []))
        diseases, drugs, genes = self._partition_entities(entities)
        year = self._extract_year(data)
        study_type = self._infer_study_type(text)
        language = self._infer_language(text)

        self.documents.append(
            {
                "doc_id": doc_id,
                "title": title,
                "journal": journal,
                "year": year,
                "study_type": study_type,
                "language": language,
                "disease": " | ".join(diseases),
                "drug": " | ".join(drugs),
                "gene": " | ".join(genes),
                "source_zip": source_zip,
                "entity_count": len(entities),
                "relation_count": len(relations),
                "text": text,
            }
        )

        entity_dict: Dict[str, dict] = {}
        for ent in entities:
            ent_id = self._first_non_empty(ent.get("entityId"), ent.get("entity_id"), ent.get("id"))
            if ent_id:
                entity_dict[ent_id] = {
                    "name": self._first_non_empty(ent.get("entityName"), ent.get("name")),
                    "type": self._first_non_empty(ent.get("entityType"), ent.get("type")),
                }

        for rel in relations:
            subj_id = self._first_non_empty(rel.get("subjectID"), rel.get("subjectId"), rel.get("subject_id"))
            obj_id = self._first_non_empty(rel.get("objectId"), rel.get("objectID"), rel.get("object_id"))
            rel_name = self._first_non_empty(rel.get("rel_name"), rel.get("relation"), rel.get("type"), "related_to")
            evidence_text = self._first_non_empty(rel.get("sentence"), rel.get("evidence"), rel.get("context"))
            confidence = self._first_non_empty(rel.get("confidence"), rel.get("score"), "")
            if subj_id in entity_dict and obj_id in entity_dict:
                subject_name = entity_dict[subj_id]["name"]
                object_name = entity_dict[obj_id]["name"]
                self.relations.append(
                    {
                        "doc_id": doc_id,
                        "subject_name": subject_name,
                        "subject_norm": self.normalize_text(subject_name),
                        "subject_type": entity_dict[subj_id]["type"],
                        "relation_type": rel_name,
                        "object_name": object_name,
                        "object_norm": self.normalize_text(object_name),
                        "object_type": entity_dict[obj_id]["type"],
                        "evidence_text": evidence_text,
                        "confidence": confidence,
                        "source_zip": source_zip,
                        "year": year,
                        "journal": journal,
                    }
                )

    def save_processed_data(self):
        if not self.documents and not self.relations:
            logging.warning("저장할 데이터가 없습니다.")
            return

        self.parsed_csv_path.parent.mkdir(parents=True, exist_ok=True)
        relations_df = pd.DataFrame(self.relations)
        relations_df.to_csv(self.parsed_csv_path, index=False, encoding="utf-8-sig")
        logging.info(f"📁 관계 데이터 CSV 저장 완료: {self.parsed_csv_path}")

        documents_df = pd.DataFrame(self.documents)
        docs_path = self.parsed_csv_path.parent / "parsed_documents.csv"
        documents_df.to_csv(docs_path, index=False, encoding="utf-8-sig")
        logging.info(f"📁 문헌 텍스트 CSV 저장 완료: {docs_path}")

    def run_pipeline(self):
        logging.info("데이터 파싱 파이프라인(Curation)을 시작합니다...")
        self.process_zip_files()
        self.save_processed_data()


if __name__ == "__main__":
    parser = BioDataParser(raw_json_path=config.RAW_JSON_PATH, parsed_csv_path=config.PARSED_CSV_PATH)
    parser.run_pipeline()

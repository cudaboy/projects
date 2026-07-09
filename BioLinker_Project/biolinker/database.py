"""
BioLinker 하이브리드 데이터베이스 관리 모듈.

개선 사항
- 임베딩 디바이스 자동 fallback(cuda -> cpu)
- 문헌 chunking + metadata 확장
- vector search score 포함 반환
- knowledge graph provenance/metadata 적재
"""

from __future__ import annotations

import logging
import math
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import networkx as nx
import pandas as pd
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

try:
    from biolinker import config
except ImportError:
    import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class SimpleTextChunker:
    def __init__(self, chunk_size: int, chunk_overlap: int):
        self.chunk_size = max(200, chunk_size)
        self.chunk_overlap = max(0, min(chunk_overlap, self.chunk_size // 2))

    def split_text(self, text: str) -> List[str]:
        normalized = " ".join(str(text or "").split())
        if not normalized:
            return []
        chunks: List[str] = []
        start = 0
        step = self.chunk_size - self.chunk_overlap
        while start < len(normalized):
            end = min(len(normalized), start + self.chunk_size)
            chunk = normalized[start:end].strip()
            if chunk:
                chunks.append(chunk)
            if end >= len(normalized):
                break
            start += max(1, step)
        return chunks[: config.MAX_CHUNKS_PER_DOC]


class BioDatabaseManager:
    def __init__(self):
        model_kwargs = {"device": config.EMBEDDING_DEVICE}
        encode_kwargs = {
            "normalize_embeddings": config.EMBEDDING_NORMALIZE,
            "batch_size": config.EMBEDDING_BATCH_SIZE,
        }
        self.embeddings = HuggingFaceEmbeddings(
            model_name=config.EMBEDDING_MODEL,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs,
        )
        self.chunker = SimpleTextChunker(config.CHUNK_SIZE, config.CHUNK_OVERLAP)
        self.chroma_persist_dir = str(config.CHROMA_DB_DIR)
        self.knowledge_graph_path = config.KNOWLEDGE_GRAPH_PATH
        logging.info(
            "✅ 임베딩 모델 로드 완료: %s (device=%s)",
            config.EMBEDDING_MODEL,
            config.EMBEDDING_DEVICE,
        )

    @staticmethod
    def _metadata_from_row(row: pd.Series) -> Dict[str, Any]:
        metadata = {
            "doc_id": str(row.get("doc_id", "unknown")),
            "title": str(row.get("title", "")),
            "journal": str(row.get("journal", "")),
            "year": str(row.get("year", "")),
            "study_type": str(row.get("study_type", "")),
            "language": str(row.get("language", "")),
            "disease": str(row.get("disease", "")),
            "drug": str(row.get("drug", "")),
            "gene": str(row.get("gene", "")),
            "source_zip": str(row.get("source_zip", "")),
            "entity_count": str(row.get("entity_count", "")),
            "relation_count": str(row.get("relation_count", "")),
        }
        return {key: value for key, value in metadata.items() if value not in {"", "nan", "None"}}

    def build_vector_db(self, parsed_docs_csv: Path):
        if not parsed_docs_csv.exists():
            logging.error(f"문헌 데이터가 없습니다: {parsed_docs_csv}")
            return None

        logging.info("Vector DB(Chroma) 구축을 시작합니다...")
        df = pd.read_csv(parsed_docs_csv).fillna("")
        documents: List[Document] = []
        for _, row in df.iterrows():
            chunks = self.chunker.split_text(str(row.get("text", "")))
            metadata = self._metadata_from_row(row)
            if not chunks:
                continue
            for idx, chunk in enumerate(chunks, start=1):
                chunk_metadata = dict(metadata)
                chunk_metadata.update(
                    {
                        "chunk_id": f"{metadata.get('doc_id', 'unknown')}-CH-{idx:03d}",
                        "chunk_index": idx,
                        "parent_doc_id": metadata.get("doc_id", "unknown"),
                        "section": "abstract_or_body",
                    }
                )
                documents.append(Document(page_content=chunk, metadata=chunk_metadata))

        if Path(self.chroma_persist_dir).exists():
            shutil.rmtree(self.chroma_persist_dir, ignore_errors=True)

        vector_db = Chroma(
            embedding_function=self.embeddings,
            persist_directory=self.chroma_persist_dir,
            collection_name=config.CHROMA_COLLECTION_NAME,
        )

        batch_size = 32 if config.EMBEDDING_DEVICE == "cuda" else 8
        for start in range(0, len(documents), batch_size):
            batch_docs = documents[start : start + batch_size]
            vector_db.add_documents(batch_docs)

        logging.info(
            "✅ Vector DB 구축 완료: 총 %s개 chunk 임베딩됨. (경로: %s)",
            len(documents),
            self.chroma_persist_dir,
        )
        return vector_db

    def build_knowledge_graph(self, parsed_relations_csv: Path):
        if not parsed_relations_csv.exists():
            logging.error(f"관계 데이터가 없습니다: {parsed_relations_csv}")
            return None

        logging.info("Knowledge Graph 구축을 시작합니다...")
        df = pd.read_csv(parsed_relations_csv).fillna("")
        graph = nx.DiGraph()
        for _, row in df.iterrows():
            subject_name = str(row.get("subject_name", "")).strip()
            object_name = str(row.get("object_name", "")).strip()
            if not subject_name or not object_name:
                continue
            relation_type = str(row.get("relation_type", "related_to")).strip() or "related_to"
            graph.add_node(
                subject_name,
                entity_type=str(row.get("subject_type", "")),
                normalized_name=str(row.get("subject_norm", "")),
            )
            graph.add_node(
                object_name,
                entity_type=str(row.get("object_type", "")),
                normalized_name=str(row.get("object_norm", "")),
            )
            graph.add_edge(
                subject_name,
                object_name,
                relation=relation_type,
                doc_id=str(row.get("doc_id", "")),
                journal=str(row.get("journal", "")),
                year=str(row.get("year", "")),
                evidence_text=str(row.get("evidence_text", "")),
                confidence=str(row.get("confidence", "")),
                source_zip=str(row.get("source_zip", "")),
            )

        nx.write_gml(graph, self.knowledge_graph_path)
        logging.info(
            "✅ Knowledge Graph 구축 완료: %s개 노드, %s개 엣지 생성. (저장: %s)",
            graph.number_of_nodes(),
            graph.number_of_edges(),
            self.knowledge_graph_path,
        )
        return graph

    def get_vector_db(self) -> Chroma:
        return Chroma(
            persist_directory=self.chroma_persist_dir,
            embedding_function=self.embeddings,
            collection_name=config.CHROMA_COLLECTION_NAME,
        )

    def get_vector_retriever(self):
        vector_db = self.get_vector_db()
        return vector_db.as_retriever(search_kwargs={"k": config.RETRIEVER_K})

    def search_vector(self, question: str, k: Optional[int] = None, filters: Optional[dict] = None) -> List[Document]:
        vector_db = self.get_vector_db()
        top_k = k or config.RETRIEVER_K
        search_kwargs: Dict[str, Any] = {"k": top_k}
        if filters:
            search_kwargs["filter"] = filters

        docs: List[Document] = []
        try:
            results = vector_db.similarity_search_with_relevance_scores(question, **search_kwargs)
            for doc, score in results:
                doc.metadata = dict(doc.metadata)
                doc.metadata["score"] = float(score)
                docs.append(doc)
        except Exception:
            results = vector_db.similarity_search_with_score(question, **search_kwargs)
            for doc, distance in results:
                score = 1.0 / (1.0 + float(distance))
                doc.metadata = dict(doc.metadata)
                doc.metadata["score"] = score
                docs.append(doc)
        filtered = [doc for doc in docs if float(doc.metadata.get("score", 0.0)) >= config.MIN_VECTOR_SCORE]
        return filtered[:top_k]

    def load_knowledge_graph(self):
        if self.knowledge_graph_path.exists():
            return nx.read_gml(self.knowledge_graph_path)
        logging.warning("저장된 지식 그래프가 없습니다. 새로 구축이 필요합니다.")
        return nx.DiGraph()


if __name__ == "__main__":
    manager = BioDatabaseManager()
    manager.build_vector_db(config.PARSED_DOCUMENTS_PATH)
    manager.build_knowledge_graph(config.PARSED_CSV_PATH)

"""
BioLinker 멀티에이전트 로직.

개선 사항
- route confidence / rationale 반환
- graph entity normalization + alias matching
- 1-hop/2-hop 그래프 탐색
- citation 추출 구조화
"""

from __future__ import annotations

import json
import logging
import re
from collections import defaultdict, deque
from typing import Any, Dict, List, Optional, Sequence, Tuple

import networkx as nx
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

try:
    from biolinker import config
    from biolinker.database import BioDatabaseManager
except ImportError:
    import config
    from database import BioDatabaseManager

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class BioAgentManager:
    def __init__(self, db_manager: BioDatabaseManager) -> None:
        self.db_manager = db_manager
        self.llm: Any = None
        self.knowledge_graph: nx.DiGraph = self.db_manager.load_knowledge_graph()
        self.alias_map = self._build_alias_map()

    @staticmethod
    def normalize_text(value: str) -> str:
        value = str(value or "").strip().lower()
        value = re.sub(r"[^0-9a-z가-힣]+", " ", value)
        return re.sub(r"\s+", " ", value).strip()

    def _build_alias_map(self) -> Dict[str, str]:
        alias_map: Dict[str, str] = {}
        for node, attrs in self.knowledge_graph.nodes(data=True):
            normalized = self.normalize_text(node)
            alias_map[normalized] = str(node)
            if attrs.get("normalized_name"):
                alias_map[self.normalize_text(attrs["normalized_name"])] = str(node)
            compact = normalized.replace(" ", "")
            if compact:
                alias_map[compact] = str(node)
        return alias_map

    def route_query(self, question: str) -> Dict[str, Any]:
        if not self.llm:
            raise ValueError("LLM이 초기화되지 않았습니다. api.py에서 주입되었는지 확인하세요.")

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
당신은 바이오메디컬 RAG 시스템의 라우터입니다.
사용자의 질문을 vector, graph, both, irrelevant 중 하나로 분류하고 confidence(0~1), rationale, rewritten_query를 JSON으로 반환하세요.
규칙:
- 문헌 근거 탐색이 핵심이면 vector
- 관계/기전/경로 탐색이 핵심이면 graph
- 둘 다 필요하면 both
- 의학/약학/생물학과 무관한 질문일 때만 irrelevant
- rewritten_query는 검색 친화적으로 짧고 명확하게 정리
응답 예시:
{"route":"both","confidence":0.82,"rationale":"약물 기전과 논문 근거가 모두 필요함","rewritten_query":"aspirin cardiovascular mechanism clinical evidence"}
JSON 외 텍스트 금지.
                    """.strip(),
                ),
                ("user", f"질문: {question}"),
            ]
        )
        raw = (prompt | self.llm | StrOutputParser()).invoke({}).strip()
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            parsed = {"route": "both", "confidence": 0.5, "rationale": raw[:200], "rewritten_query": question}
        route = str(parsed.get("route", "both")).strip().lower()
        if route not in {"vector", "graph", "both", "irrelevant"}:
            route = "both"
        try:
            confidence = float(parsed.get("confidence", 0.5))
        except (TypeError, ValueError):
            confidence = 0.5
        return {
            "route": route,
            "confidence": max(0.0, min(1.0, confidence)),
            "rationale": str(parsed.get("rationale", ""))[:240],
            "rewritten_query": str(parsed.get("rewritten_query", question)).strip() or question,
        }

    def retrieve_vector_context(self, question: str, top_k: Optional[int] = None) -> List[Document]:
        try:
            return self.db_manager.search_vector(question, k=top_k or config.RETRIEVER_K)
        except Exception as exc:
            logging.error(f"Vector DB 검색 오류: {exc}")
            return []

    def _match_nodes(self, question: str) -> List[str]:
        normalized_question = self.normalize_text(question)
        compact_question = normalized_question.replace(" ", "")
        matched: List[str] = []
        for alias, node in self.alias_map.items():
            if not alias:
                continue
            if alias in normalized_question or alias in compact_question:
                matched.append(node)
        # 짧은 노드명보다 긴 노드명을 우선 유지
        unique = sorted(set(matched), key=lambda item: (-len(self.normalize_text(item)), item))
        filtered: List[str] = []
        for node in unique:
            node_norm = self.normalize_text(node)
            if any(node_norm != self.normalize_text(other) and node_norm in self.normalize_text(other) for other in filtered):
                continue
            filtered.append(node)
        return filtered[:8]

    def retrieve_graph_context(self, question: str, max_hops: Optional[int] = None) -> Tuple[str, List[str], List[dict]]:
        logs: List[str] = []
        if not self.knowledge_graph or self.knowledge_graph.number_of_nodes() == 0:
            logs.append("⚠️ [Graph DB] 그래프 데이터베이스가 비어있어 탐색을 건너뜁니다.")
            return "그래프 데이터베이스가 비어있습니다.", logs, []

        matched_nodes = self._match_nodes(question)
        if not matched_nodes:
            logs.append("⚠️ [Graph DB] 질문과 일치하는 정규화된 의료 entity를 찾지 못했습니다.")
            return "질문과 직접적으로 연관된 그래프 관계 정보가 없습니다.", logs, []

        hop_limit = max_hops or config.GRAPH_MAX_HOPS
        logs.append(f"📍 [Graph DB] entity 매칭 완료: {', '.join(matched_nodes)}")

        ranked_edges: List[dict] = []
        visited_paths = set()
        for root in matched_nodes:
            queue = deque([(root, 0)])
            seen = {root}
            while queue:
                current, depth = queue.popleft()
                if depth >= hop_limit:
                    continue
                for neighbor in self.knowledge_graph.successors(current):
                    edge = self.knowledge_graph.get_edge_data(current, neighbor) or {}
                    edge_key = (root, current, neighbor)
                    if edge_key in visited_paths:
                        continue
                    visited_paths.add(edge_key)
                    record = {
                        "source": root,
                        "subject": current,
                        "object": neighbor,
                        "relation": edge.get("relation", "연관됨"),
                        "doc_id": edge.get("doc_id", "출처미상"),
                        "journal": edge.get("journal", ""),
                        "year": edge.get("year", ""),
                        "evidence_text": edge.get("evidence_text", ""),
                        "confidence": edge.get("confidence", ""),
                        "hop": depth + 1,
                    }
                    ranked_edges.append(record)
                    if neighbor not in seen:
                        seen.add(neighbor)
                        queue.append((neighbor, depth + 1))

        ranked_edges = sorted(
            ranked_edges,
            key=lambda item: (item["hop"], 0 if item.get("doc_id") else 1, str(item["relation"])),
        )[: config.GRAPH_MAX_EDGES]

        if not ranked_edges:
            logs.append("⚠️ [Graph DB] 매칭된 entity는 있으나 연결 관계를 찾지 못했습니다.")
            return "연결된 그래프 관계 정보가 없습니다.", logs, []

        preview = [f"[{edge['subject']}] → [{edge['object']}]" for edge in ranked_edges[: config.GRAPH_PREVIEW_EDGES]]
        logs.append(
            f"🕸️ [Graph DB] {len(ranked_edges)}개의 관계 추출 완료 (예시: {', '.join(preview)})"
        )

        lines = []
        for edge in ranked_edges:
            evidence = f" | evidence: {edge['evidence_text'][:160]}" if edge.get("evidence_text") else ""
            lines.append(
                f"[{edge['subject']}] --({edge['relation']})--> [{edge['object']}] "
                f"(hop={edge['hop']}, doc_id={edge['doc_id']}, year={edge['year']}){evidence}"
            )
        return "\n".join(lines), logs, ranked_edges

    @staticmethod
    def build_citations(vector_docs: Sequence[Document]) -> List[dict]:
        citations: List[dict] = []
        seen = set()
        for doc in vector_docs:
            doc_id = str(doc.metadata.get("doc_id", "unknown"))
            if doc_id in seen:
                continue
            seen.add(doc_id)
            citations.append(
                {
                    "doc_id": doc_id,
                    "title": str(doc.metadata.get("title", "제목 없음")),
                    "journal": str(doc.metadata.get("journal", "")),
                    "year": str(doc.metadata.get("year", "")),
                    "score": float(doc.metadata.get("score", 0.0)),
                    "chunk_id": str(doc.metadata.get("chunk_id", "")),
                }
            )
        return citations

    def synthesize_answer(
        self,
        question: str,
        vector_docs: Sequence[Document],
        graph_context: str,
        no_answer_reason: str = "",
    ) -> str:
        if not self.llm:
            raise ValueError("LLM이 초기화되지 않았습니다.")

        if no_answer_reason:
            return (
                "## 답변 보류\n"
                f"- 사유: {no_answer_reason}\n"
                "- 현재 검색 결과만으로는 신뢰할 수 있는 임상/기전 결론을 제시하기 어렵습니다.\n"
                "- 질환명/약물명/유전자명을 더 구체화하거나 데이터 적재 범위를 확인해 주세요."
            )

        if not vector_docs and (not graph_context or "없습니다" in graph_context or "비어있습니다" in graph_context):
            return "## 답변 보류\n- 검색된 문헌 및 그래프 근거가 모두 부족합니다."

        formatted_docs = []
        for idx, doc in enumerate(vector_docs, start=1):
            formatted_docs.append(
                "\n".join(
                    [
                        f"[문헌 {idx}]",
                        f"- 제목: {doc.metadata.get('title', '제목 없음')}",
                        f"- 출처 ID: {doc.metadata.get('doc_id', 'unknown')}",
                        f"- 저널/연도: {doc.metadata.get('journal', '')} / {doc.metadata.get('year', '')}",
                        f"- score: {doc.metadata.get('score', 0.0):.4f}",
                        f"- 내용: {doc.page_content}",
                    ]
                )
            )
        vector_text = "\n\n".join(formatted_docs) if formatted_docs else "검색된 문헌 없음"

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", config.SYNTHESIS_TEMPLATE),
                (
                    "user",
                    f"""
[사용자 질문]
{question}

[Vector DB 검색 결과]
{vector_text}

[Graph DB 관계 검색 결과]
{graph_context}

반드시 markdown으로 답변하세요.
""".strip(),
                ),
            ]
        )
        return (prompt | self.llm | StrOutputParser()).invoke({})

"""
BioLinker LangGraph 오케스트레이션.

개선 사항
- route confidence / rationale / query rewrite 상태 관리
- rerank -> evidence_check -> synthesize -> citation_verify 단계 분리
- retrieval mode override(vector/graph/both) 지원
- no-answer / safety flag 구조화
"""

from __future__ import annotations

import logging
import operator
from typing import Annotated, Dict, List, Optional, TypedDict

from langchain_core.documents import Document
from langgraph.graph import END, START, StateGraph

try:
    from biolinker.agents import BioAgentManager
    from biolinker import config
except ImportError:
    from agents import BioAgentManager
    import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class GraphState(TypedDict, total=False):
    question: str
    rewritten_question: str
    route_override: str
    route: str
    route_confidence: float
    route_reason: str
    top_k: int
    vector_context: List[Document]
    graph_context: str
    graph_edges: List[dict]
    citations: List[dict]
    retrieved_doc_ids: List[str]
    no_answer_reason: str
    final_answer: str
    safety_flag: str
    logs: Annotated[List[str], operator.add]


def create_workflow(agent_manager: BioAgentManager):
    workflow = StateGraph(GraphState)

    def router_node(state: GraphState):
        override = str(state.get("route_override", "auto") or "auto").lower()
        question = state["question"]
        if override in {"vector", "graph", "both", "irrelevant"}:
            return {
                "route": override,
                "route_confidence": 1.0,
                "route_reason": f"사용자 지정 retrieval mode: {override}",
                "rewritten_question": question,
                "logs": [f"🧭 [Router] override 적용: {override}"],
            }

        route_payload = agent_manager.route_query(question)
        route = route_payload["route"]
        confidence = float(route_payload["confidence"])
        rationale = route_payload.get("rationale", "")
        rewritten = route_payload.get("rewritten_query", question)
        return {
            "route": route,
            "route_confidence": confidence,
            "route_reason": rationale,
            "rewritten_question": rewritten,
            "logs": [f"🧭 [Router] route={route}, confidence={confidence:.2f}, rationale={rationale}"],
        }

    def vector_node(state: GraphState):
        query = state.get("rewritten_question") or state["question"]
        top_k = int(state.get("top_k") or config.RETRIEVER_K)
        docs = agent_manager.retrieve_vector_context(query, top_k=top_k)
        doc_ids = [str(doc.metadata.get("doc_id", "unknown")) for doc in docs]
        return {
            "vector_context": docs,
            "logs": [f"📚 [Vector DB] {len(docs)}개 chunk 검색 완료: {', '.join(doc_ids[:top_k]) or '없음'}"],
        }

    def graph_node(state: GraphState):
        query = state.get("rewritten_question") or state["question"]
        g_context, g_logs, edges = agent_manager.retrieve_graph_context(query, max_hops=config.GRAPH_MAX_HOPS)
        return {"graph_context": g_context, "graph_edges": edges, "logs": g_logs}

    def both_node(state: GraphState):
        vector_result = vector_node(state)
        graph_result = graph_node(state)
        return {
            "vector_context": vector_result.get("vector_context", []),
            "graph_context": graph_result.get("graph_context", ""),
            "graph_edges": graph_result.get("graph_edges", []),
            "logs": vector_result.get("logs", []) + graph_result.get("logs", []),
        }

    def rerank_node(state: GraphState):
        docs = state.get("vector_context", [])
        if not docs:
            return {"logs": ["🧪 [Rerank] vector 결과 없음 - rerank 생략"]}
        deduped: Dict[str, Document] = {}
        for doc in docs:
            doc_id = str(doc.metadata.get("doc_id", "unknown"))
            score = float(doc.metadata.get("score", 0.0))
            if doc_id not in deduped or score > float(deduped[doc_id].metadata.get("score", 0.0)):
                deduped[doc_id] = doc
        ranked = sorted(deduped.values(), key=lambda item: float(item.metadata.get("score", 0.0)), reverse=True)
        reranked = ranked[: config.RERANK_TOP_K]
        return {
            "vector_context": reranked,
            "logs": [f"🧪 [Rerank] {len(docs)} -> {len(reranked)}개 문헌으로 축약"],
        }

    def evidence_check_node(state: GraphState):
        docs = state.get("vector_context", [])
        graph_edges = state.get("graph_edges", [])
        route = state.get("route", "unknown")
        confidence = float(state.get("route_confidence", 0.0))
        no_answer_reason = ""

        if route == "vector" and not docs:
            no_answer_reason = "문헌 검색 결과가 없어 답변 근거를 확보하지 못했습니다."
        elif route == "graph" and not graph_edges:
            no_answer_reason = "그래프 관계 검색 결과가 없어 기전 근거를 확보하지 못했습니다."
        elif route == "both" and not docs and not graph_edges:
            no_answer_reason = "문헌과 그래프 모두에서 근거를 찾지 못했습니다."
        elif confidence < config.ROUTE_LOW_CONFIDENCE_THRESHOLD and not docs:
            no_answer_reason = "라우터 confidence가 낮고 문헌 근거도 부족합니다."

        safety_flag = "ok" if not no_answer_reason else "insufficient_evidence"
        log = f"🛡️ [Evidence Check] safety_flag={safety_flag}"
        if no_answer_reason:
            log += f" | reason={no_answer_reason}"
        return {"no_answer_reason": no_answer_reason, "safety_flag": safety_flag, "logs": [log]}

    def irrelevant_node(state: GraphState):
        return {
            "final_answer": "🚫 의학·약학·생물학 관련 질문만 답변할 수 있습니다.",
            "safety_flag": "irrelevant",
            "no_answer_reason": "도메인 외 질문",
        }

    def synthesize_node(state: GraphState):
        answer = agent_manager.synthesize_answer(
            question=state["question"],
            vector_docs=state.get("vector_context", []),
            graph_context=state.get("graph_context", ""),
            no_answer_reason=state.get("no_answer_reason", ""),
        )
        return {"final_answer": answer}

    def citation_verify_node(state: GraphState):
        docs = state.get("vector_context", [])
        citations = agent_manager.build_citations(docs)
        retrieved_doc_ids = [citation["doc_id"] for citation in citations]
        return {
            "citations": citations,
            "retrieved_doc_ids": retrieved_doc_ids,
            "logs": [f"🔎 [Citation Verify] {len(citations)}개 citation 구조화 완료"],
        }

    workflow.add_node("router", router_node)
    workflow.add_node("search_vector", vector_node)
    workflow.add_node("search_graph", graph_node)
    workflow.add_node("search_both", both_node)
    workflow.add_node("rerank", rerank_node)
    workflow.add_node("evidence_check", evidence_check_node)
    workflow.add_node("irrelevant", irrelevant_node)
    workflow.add_node("synthesize", synthesize_node)
    workflow.add_node("citation_verify", citation_verify_node)

    workflow.add_edge(START, "router")

    def route_condition(state: GraphState) -> str:
        return state.get("route", "both")

    workflow.add_conditional_edges(
        "router",
        route_condition,
        {
            "vector": "search_vector",
            "graph": "search_graph",
            "both": "search_both",
            "irrelevant": "irrelevant",
        },
    )

    workflow.add_edge("search_vector", "rerank")
    workflow.add_edge("search_graph", "evidence_check")
    workflow.add_edge("search_both", "rerank")
    workflow.add_edge("rerank", "evidence_check")
    workflow.add_edge("evidence_check", "synthesize")
    workflow.add_edge("synthesize", "citation_verify")
    workflow.add_edge("citation_verify", END)
    workflow.add_edge("irrelevant", END)

    app = workflow.compile()
    logging.info("✅ LangGraph 워크플로우 컴파일 완료")
    return app


if __name__ == "__main__":
    from pprint import pprint
    from biolinker.database import BioDatabaseManager

    db_manager = BioDatabaseManager()
    agent_manager = BioAgentManager(db_manager)
    app = create_workflow(agent_manager)
    pprint(app)

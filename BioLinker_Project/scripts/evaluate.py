"""
BioLinker 하이브리드 RAG 평가 스크립트.

개선 사항
- 평가셋 외부 파일(data/eval/queries.jsonl) 사용
- retrieval_mode / top_k 실험 파라미터 지원
- retrieval 지표 + no-answer 정확도 저장
- CSV + JSON summary 동시 저장
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
from datasets import Dataset
from dotenv import load_dotenv
from langchain_core.runnables import RunnableLambda

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))
load_dotenv(dotenv_path=PROJECT_ROOT / ".env")

from ragas import evaluate
from ragas.metrics import AnswerRelevancy, ContextPrecision, ContextRecall, Faithfulness
from langchain_openai import ChatOpenAI

from biolinker import config
from biolinker.agents import BioAgentManager
from biolinker.database import BioDatabaseManager
from biolinker.workflow import create_workflow

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("biolinker.evaluate")


DEFAULT_DATASET = [
    {
        "question": "BRCA1 또는 BRCA2 유전자 변이가 있는 유방암 환자에게 PARP 억제제를 투여했을 때의 작용 기전은 무엇인가요?",
        "ground_truth": "PARP 억제제는 BRCA1/2 변이로 HRR 복구가 손상된 암세포에서 합성 치사를 유도합니다.",
        "expected_entities": ["BRCA1", "BRCA2", "PARP", "유방암"],
        "retrieval_mode": "both",
    },
    {
        "question": "아스피린이 COX 효소를 억제하여 심혈관계 질환 예방에 기여하는 기전을 설명해주세요.",
        "ground_truth": "아스피린은 COX-1/COX-2를 억제해 TXA2 생성을 줄이고 혈소판 응집을 억제합니다.",
        "expected_entities": ["아스피린", "COX", "심혈관계 질환"],
        "retrieval_mode": "both",
    },
    {
        "question": "완전한 가상의 물질인 Vitamin Z가 뇌종양 세포 사멸에 미치는 임상적 효과를 알려주세요.",
        "ground_truth": "Vitamin Z는 가상의 물질이므로 신뢰할 수 있는 근거가 없습니다.",
        "expected_entities": ["Vitamin Z", "뇌종양"],
        "expect_no_answer": True,
        "retrieval_mode": "vector",
    },
]


def ensure_eval_dataset(path: Path):
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for row in DEFAULT_DATASET:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    logger.info("기본 평가셋 생성: %s", path)


def load_eval_dataset(path: Path, limit: int | None = None) -> List[Dict[str, Any]]:
    ensure_eval_dataset(path)
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if limit and len(rows) >= limit:
                break
    return rows


def compute_retrieval_metrics(results: List[Dict[str, Any]]) -> Dict[str, float]:
    citation_counts = [len(item.get("citations", [])) for item in results]
    graph_counts = [len(item.get("graph_edges", [])) for item in results]
    entity_hits = [item.get("entity_hit_rate", 0.0) for item in results]
    no_answer_expected = [item for item in results if item.get("expect_no_answer")]
    no_answer_correct = [
        item for item in no_answer_expected if item.get("safety_flag") == "insufficient_evidence" or item.get("no_answer_reason")
    ]
    return {
        "avg_citation_count": statistics.mean(citation_counts) if citation_counts else 0.0,
        "avg_graph_edge_count": statistics.mean(graph_counts) if graph_counts else 0.0,
        "avg_entity_hit_rate": statistics.mean(entity_hits) if entity_hits else 0.0,
        "no_answer_accuracy": (len(no_answer_correct) / len(no_answer_expected)) if no_answer_expected else 0.0,
    }


def run_ragas_evaluation(retrieval_mode: str, top_k: int, limit: int | None = None):
    logger.info("🔬 BioLinker 평가 시작 | mode=%s top_k=%s", retrieval_mode, top_k)
    dataset_rows = load_eval_dataset(config.EVAL_DATASET_PATH, limit=limit)

    db_manager = BioDatabaseManager()
    agent_manager = BioAgentManager(db_manager)
    has_openai_key = bool(config.OPENAI_API_KEY)
    if has_openai_key:
        eval_llm = ChatOpenAI(model=config.LLM_MODEL, temperature=0.0)
    else:
        logger.warning("OPENAI_API_KEY가 없어 offline smoke mode로 평가를 수행합니다. RAGAS 점수는 생략됩니다.")
        eval_llm = RunnableLambda(lambda _: "## Offline smoke answer\n- 로컬 retrieval smoke test 모드입니다.")
    agent_manager.llm = eval_llm
    workflow_app = create_workflow(agent_manager)
    evaluator_llm = ChatOpenAI(model=config.LLM_MODEL, temperature=0.0) if has_openai_key else None

    questions: List[str] = []
    answers: List[str] = []
    contexts: List[List[str]] = []
    ground_truths: List[str] = []
    experiment_rows: List[Dict[str, Any]] = []

    for idx, item in enumerate(dataset_rows, start=1):
        query = item["question"]
        mode = item.get("retrieval_mode", retrieval_mode)
        final_state = workflow_app.invoke({"question": query, "route_override": mode, "top_k": top_k})

        v_docs = final_state.get("vector_context", [])
        g_edges = final_state.get("graph_edges", [])
        retrieved_texts = [doc.page_content for doc in v_docs]
        if final_state.get("graph_context"):
            retrieved_texts.append(final_state["graph_context"])

        expected_entities = [str(entity).lower() for entity in item.get("expected_entities", [])]
        retrieved_blob = " ".join(retrieved_texts).lower()
        entity_hits = sum(1 for entity in expected_entities if entity.lower() in retrieved_blob)
        entity_hit_rate = (entity_hits / len(expected_entities)) if expected_entities else 0.0

        row = {
            "question": query,
            "route": final_state.get("route", "unknown"),
            "route_confidence": float(final_state.get("route_confidence", 0.0)),
            "answer": final_state.get("final_answer", ""),
            "citations": final_state.get("citations", []),
            "graph_edges": g_edges,
            "retrieved_doc_ids": final_state.get("retrieved_doc_ids", []),
            "no_answer_reason": final_state.get("no_answer_reason", ""),
            "safety_flag": final_state.get("safety_flag", "ok"),
            "expect_no_answer": bool(item.get("expect_no_answer", False)),
            "entity_hit_rate": entity_hit_rate,
        }
        experiment_rows.append(row)
        questions.append(query)
        answers.append(row["answer"])
        contexts.append(retrieved_texts)
        ground_truths.append(item["ground_truth"])
        logger.info("[%s/%s] route=%s confidence=%.2f citations=%s graph_edges=%s", idx, len(dataset_rows), row["route"], row["route_confidence"], len(row["citations"]), len(g_edges))

    ragas_scores = {}
    if has_openai_key:
        dataset = Dataset.from_dict(
            {
                "question": questions,
                "answer": answers,
                "contexts": contexts,
                "ground_truth": ground_truths,
            }
        )
        ragas_result = evaluate(
            dataset=dataset,
            metrics=[Faithfulness(), AnswerRelevancy(), ContextPrecision(), ContextRecall()],
            llm=evaluator_llm,
        )

        ragas_df = ragas_result.to_pandas()
        ragas_scores = ragas_df.mean(numeric_only=True).to_dict()
    retrieval_scores = compute_retrieval_metrics(experiment_rows)

    report_df = pd.DataFrame(experiment_rows)
    report_path = config.EVAL_REPORT_PATH
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_df.to_csv(report_path, index=False, encoding="utf-8-sig")

    summary = {
        "retrieval_mode": retrieval_mode,
        "top_k": top_k,
        "sample_count": len(dataset_rows),
        "ragas": ragas_scores,
        "retrieval": retrieval_scores,
        "report_path": str(report_path),
    }
    with open(config.EVAL_SUMMARY_PATH, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    logger.info("📈 평가 요약: %s", json.dumps(summary, ensure_ascii=False))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BioLinker evaluation runner")
    parser.add_argument("--retrieval-mode", default="auto", choices=["auto", "vector", "graph", "both"])
    parser.add_argument("--top-k", type=int, default=config.RETRIEVER_K)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    run_ragas_evaluation(args.retrieval_mode, args.top_k, args.limit)

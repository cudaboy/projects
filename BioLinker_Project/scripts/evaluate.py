"""
===============================================================================
[File Role]
이 파일(evaluate.py)은 BioLinker 프로젝트의 'RAG 파이프라인 정량적 성능 평가'를 담당합니다.

[상세 설명]
1. 목적: Ragas 프레임워크를 활용하여 LangGraph 멀티에이전트가 생성한 답변의 
   신뢰성 및 검색 품질을 4가지 다차원 지표로 엄격하게 자동 평가합니다.
2. 핵심 평가 지표 (Metrics):
   - Faithfulness (사실 부합성): 생성된 답변이 환각 없이 오직 검색된 문헌/그래프에만 기반하는가?
   - Answer Relevancy (답변 관련성): 답변이 임상 연구원의 질문 의도에 정확히 부합하는가?
   - Context Precision (검색 정밀도): 검색된 하이브리드 컨텍스트 중 정답에 유용한 정보가 상위에 있는가?
   - Context Recall (검색 재현율): 참조할 만한 Ground Truth 정보가 검색 결과에 빠짐없이 포함되었는가?
===============================================================================
"""

import sys
import logging
import pandas as pd
from pathlib import Path
from datasets import Dataset
from dotenv import load_dotenv

# 프로젝트 최상단 디렉토리 경로 추가 및 환경 변수(.env) 명시적 로드
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))
load_dotenv(dotenv_path=PROJECT_ROOT / ".env")

# Ragas 프레임워크 및 평가지표 불러오기
from ragas import evaluate
from ragas.metrics import (
    Faithfulness,
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
)
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings

try:
    from biolinker import config
    from biolinker.database import BioDatabaseManager
    from biolinker.agents import BioAgentManager
    from biolinker.workflow import create_workflow
except ImportError as e:
    logging.error(f"모듈 로드 에러: {e}")
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def run_ragas_evaluation():
    logging.info("🔬 BioLinker 하이브리드 RAG 시스템에 대한 Ragas 정량 평가를 시작합니다.")
    
    # 1. 모델 및 시스템 초기화
    db_manager = BioDatabaseManager()
    agent_manager = BioAgentManager(db_manager)

    # evaluate.py는 api.py를 거치지 않는 단독 실행 스크립트이므로,
    # 에이전트가 사용할 LLM을 여기서 직접 주입(Injection)해 주어야 합니다.
    eval_agent_llm = ChatOpenAI(model=config.LLM_MODEL, temperature=0.0)
    agent_manager.llm = eval_agent_llm  # 에이전트에게 두뇌 장착
    
    workflow_app = create_workflow(agent_manager)
    
    # Ragas 평가용 모델 명시적 매핑 (안정성 확보)
    evaluator_llm = ChatOpenAI(model=config.LLM_MODEL, temperature=0.0)
    evaluator_embeddings = HuggingFaceEmbeddings(
        model_name=config.EMBEDDING_MODEL,
        model_kwargs={'device': 'cuda'},  # GPU가 있다면 'cuda'
        encode_kwargs={'normalize_embeddings': True, 'batch_size': 4}
    )

    # 2. 임상 도메인 특화 테스트셋 (Test Queries & Ground Truth)
    # 실제 환경에서는 전문 연구원이 레이블링한 50~100개의 QA셋을 엑셀로 불러와 사용합니다.
    evaluation_dataset = [
        {
            "question": "BRCA1 또는 BRCA2 유전자 변이가 있는 유방암 환자에게 파프(PARP) 억제제를 투여했을 때의 작용 기전(MoA)은 무엇이며, 관련 임상적 근거를 설명해주세요.",
            "ground_truth": "PARP 억제제는 BRCA1/2 변이로 인해 상동재조합 복구(HRR)가 결핍된 암세포에서 DNA 단일 가닥 손상 복구를 차단하여 합성 치사(Synthetic lethality)를 유도함으로써 암세포 사멸을 촉진합니다. 임상적으로 Olaparib 등이 유방암 치료에 유효한 것으로 입증되었습니다."
        },
        {
            "question": "아스피린(Aspirin)이 COX 효소를 억제하여 심혈관계 질환 예방에 기여하는 생물학적 기전을 설명하고, 이에 대한 최신 논문 문헌을 찾아주세요.",
            "ground_truth": "아스피린은 비스테로이드성 항염증제(NSAID)로 COX-1 및 COX-2 효소를 비가역적으로 억제하여 프로스타글란딘 및 트롬복산 A2(TXA2)의 생성을 감소시킵니다. 이를 통해 혈소판 응집을 억제하고 혈전 생성을 방지하여 심혈관계 질환을 예방합니다."
        },
        {
            "question": "비소세포폐암(NSCLC) 치료에 사용되는 특정 표적항암제의 주요 내성(Resistance) 기전에는 어떤 것들이 있나요?",
            "ground_truth": "비소세포폐암(NSCLC)의 EGFR 표적항암제 등 주요 내성 기전으로는 EGFR T790M 등 2차 돌연변이 발생, MET 유전자 증폭, HER2 변이, 그리고 PI3K/AKT/mTOR 등 대체 신호전달 경로의 우회 활성화가 있습니다."
        },
        {
            "question": "완전한 가상의 물질인 비타민 Z(Vitamin Z)가 뇌종양 세포 사멸에 미치는 임상적 효과와 관련 논문을 찾아주세요.",
            "ground_truth": "비타민 Z는 가상의 물질이므로 뇌종양 세포 사멸에 미치는 임상적 효과나 관련 논문 근거는 당사 데이터베이스 및 문헌에 존재하지 않습니다." # 환각 방지 테스트 1
        },
        {
            "question": "특정 유전자 변이가 발견되지 않은 환자에게 가짜약물(FakeDrug_X)을 투여했을 때의 효과는?",
            "ground_truth": "해당 가짜약물(FakeDrug_X)에 대한 유의미한 임상적 효과나 연관성은 문헌 및 데이터베이스에 보고된 바 없습니다." # 환각 방지 테스트 2
        }
    ]

    questions = []
    answers = []
    contexts = []
    ground_truths = []

    # 3. 데이터 추론 및 컨텍스트 수집 루프
    for idx, item in enumerate(evaluation_dataset, 1):
        query = item["question"]
        logging.info(f"[{idx}/{len(evaluation_dataset)}] 질의 추론 중: {query}")
        
        # 워크플로우 실행
        final_state = workflow_app.invoke({"question": query})

        # 워크플로우가 수집한 에이전트 사고 과정(Logs)을 터미널에 출력
        logs = final_state.get("logs", [])
        if logs:
            logging.info("--- 🛠️ [Agent Trace Logs] ---")
            for log in logs:
                logging.info(f"  {log}")
            logging.info("------------------------------")
        
        # 검색된 Vector DB 및 Graph DB 컨텍스트 추출
        v_docs = final_state.get("vector_context", [])
        retrieved_texts = [doc.page_content for doc in v_docs]
        
        g_text = final_state.get("graph_context", "")
        if g_text:
            retrieved_texts.append(f"Graph DB 관계 정보: {g_text}")
            
        # Ragas 입력 형식에 맞게 저장
        questions.append(query)
        answers.append(final_state.get("final_answer", ""))
        contexts.append(retrieved_texts)
        ground_truths.append(item["ground_truth"])

    # 4. HuggingFace Dataset 포맷으로 변환 (Ragas 필수 규격)
    data = {
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths
    }
    dataset = Dataset.from_dict(data)

    # 5. Ragas 지표 평가 수행
    logging.info("📊 Ragas 평가 지표 산출 중... (LLM-as-a-Judge 작동)")
    result = evaluate(
        dataset=dataset,
        metrics=[
            Faithfulness(),     # 사실 부합성 (환각 방지)
            AnswerRelevancy(),  # 답변 관련성
            ContextPrecision(), # 검색 정밀도
            ContextRecall()     # 검색 재현율
            ],
        llm=evaluator_llm,
        embeddings=evaluator_embeddings
    )

    # 6. 결과 리포팅
    logging.info("\n==================================================")
    logging.info("📈 BioLinker RAG 성능 평가 최종 리포트")
    logging.info("==================================================")
    
    # Ragas는 기본적으로 0 ~ 1.0 사이의 점수를 반환합니다.
    df_result = result.to_pandas()
    metrics_score = df_result.mean(numeric_only=True).to_dict()

    for metric, score in metrics_score.items():
        logging.info(f" - {metric.replace('_', ' ').title()}: {score:.4f}")

    logging.info("\n💡 [평가 진단 및 인사이트]")
    faith_score = metrics_score.get('faithfulness', 0)

    if faith_score > 0.8:
        logging.info("✅ 우수: 환각(Hallucination) 통제력이 높아 임상 데이터로서의 신뢰성을 갖추었습니다.")
    else:
        logging.info("⚠️ 주의: 사실 부합성이 낮습니다. Prompt 제약 조건을 강화하거나 Temperature를 점검하세요.")
        
    if metrics_score.get('context_precision', 0) < 0.5:
        logging.info("🔧 개선 권고: 검색 정밀도가 낮습니다. 의학 도메인 특화 임베딩(PubMedBERT 등) 도입을 고려하세요.")
        
    logging.info("==================================================")
    
    # 세부 평가 결과를 CSV로 저장 (나중에 분석용)
    result_csv_path = PROJECT_ROOT / "data" / "processed" / "ragas_evaluation_report.csv"
    result_csv_path.parent.mkdir(parents=True, exist_ok=True)
    df_result.to_csv(result_csv_path, index=False, encoding='utf-8-sig')
    logging.info(f"📁 상세 평가 데이터가 저장되었습니다: {result_csv_path}")

if __name__ == "__main__":
    run_ragas_evaluation()
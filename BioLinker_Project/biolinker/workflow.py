"""
===============================================================================
[File Role]
이 파일(workflow.py)은 BioLinker 프로젝트의 '멀티에이전트 파이프라인 오케스트레이션(Orchestration)'을 담당합니다.

[전체 프로젝트 내 역할 및 상세 설명]
1. 모듈 역할: agents.py에서 정의한 개별 에이전트(Router, Retriever, Synthesizer)들을 LangGraph의 StateGraph로 연결하여 데이터가 순차적/조건부로 흐르도록 워크플로우를 조립합니다.
2. 상태 관리 (State Management): GraphState 구조체를 통해 에이전트 간 주고받는 변수(질문, 라우팅 경로, 검색 결과, 최종 답변 등)의 상태를 안전하게 관리합니다.
3. 조건부 라우팅 및 종료:
   - 시작 노드인 Router의 결정에 따라 Vector 탐색, Graph 탐색, 혹은 둘 다(Both)를 수행하는 노드로 동적 분기(Conditional Edges)합니다.
   - 만약 질문이 의학/약학과 관련 없는 일상적인 질문('irrelevant')일 경우, 데이터베이스 검색이나 응답 합성(Synthesize) 단계를 거치지 않고 거절 메시지를 반환한 뒤 워크플로우를 즉시 종료(END)하여 불필요한 API 비용을 절감합니다.
===============================================================================
"""

import logging
import operator
from typing import Annotated, TypedDict, List, Optional
from langchain_core.documents import Document
from langgraph.graph import StateGraph, START, END

# agents.py 모듈 불러오기
try:
    from biolinker.agents import BioAgentManager
    from biolinker.database import BioDatabaseManager
except ImportError:
    from agents import BioAgentManager
    from database import BioDatabaseManager

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ---------------------------------------------------------
# 1. 그래프 상태(State) 정의
# ---------------------------------------------------------
class GraphState(TypedDict):
    """
    LangGraph 워크플로우 내에서 에이전트들이 공유할 상태(State)를 정의합니다.
    각 노드는 이 상태를 읽고, 자신이 맡은 필드를 업데이트합니다.
    """
    question: str                            # 사용자의 원래 질문
    route: str                               # Router가 결정한 경로 ('vector', 'graph', 'both', 'irrelevant')
    vector_context: List[Document]           # Vector DB에서 검색된 문헌 조각들
    graph_context: str                       # Graph DB에서 검색된 개체 관계 텍스트
    final_answer: str                        # Synthesizer가 최종 생성한 리포트 답변
    logs: Annotated[List[str], operator.add] # 누적 로그

# ---------------------------------------------------------
# 2. 워크플로우 생성 및 컴파일 함수
# ---------------------------------------------------------
def create_workflow(agent_manager: BioAgentManager):
    """
    BioAgentManager의 기능들을 LangGraph의 노드(Node)와 엣지(Edge)로 매핑하여
    실행 가능한 형태의 그래프로 컴파일합니다.
    """
    # 1) 상태 그래프 초기화
    workflow = StateGraph(GraphState)

    # 2) 그래프 노드(Node) 정의 (각 에이전트의 작업 단위)
    def router_node(state: GraphState):
        """질문을 분석하여 탐색할 데이터베이스를 결정합니다."""
        logging.info("▶️ [Node: Router] 실행 중...")
        route_decision = agent_manager.route_query(state["question"])
        
        route_descriptions = {
            "vector": "논문 문헌 검색을 진행합니다",
            "graph": "질병-유전자-약물 간의 관계 탐색을 진행합니다",
            "both": "논문 문헌 검색과 질병-유전자-약물 간의 관계 탐색을 진행합니다"
        }
        
        # 'irrelevant' 등 예외 상황 대비 기본 문구 설정
        description = route_descriptions.get(route_decision, f"'{route_decision}' 경로로 탐색을 진행합니다")
        
        return {
            "route": route_decision, 
            "logs": [f"🧭 [Router] 판단 결과: {description}"]
        }

    def vector_retriever_node(state: GraphState):
        """Vector DB에서 유사 논문을 검색합니다."""
        logging.info("▶️ [Node: Vector Retriever] 실행 중...")
        
        # 1. docs 변수를 먼저 정의해야 합니다 (이 줄이 누락되면 NameError 발생)
        docs = agent_manager.retrieve_vector_context(state["question"])
        
        # 2. 정의된 docs를 바탕으로 ID 리스트 생성
        doc_ids = [f"[{doc.metadata.get('doc_id', 'Unknown')}]" for doc in docs]
        id_list_str = ", ".join(doc_ids) if doc_ids else "검색 결과 없음"
        
        return {
            "vector_context": docs, 
            "logs": [f"📚 [Vector DB] 의미 검색 완료: {len(docs)}건의 문헌 확보 : {id_list_str}"]
        }

    def graph_retriever_node(state: GraphState):
        """Knowledge Graph에서 개체 간 관계를 탐색합니다."""
        logging.info("▶️ [Node: Graph Retriever] 실행 중...")
        g_context, g_logs = agent_manager.retrieve_graph_context(state["question"])
        
        return {"graph_context": g_context, "logs": g_logs}

    def both_retriever_node(state: GraphState):
        """'both' 라우팅 시, Vector와 Graph 탐색을 모두 수행하여 상태를 업데이트합니다."""
        logging.info("▶️ [Node: Hybrid Retriever (Both)] 실행 중...")
        
        # 1. Vector DB 검색 및 ID 리스트 추출
        docs = agent_manager.retrieve_vector_context(state["question"])
        doc_ids = [f"[{doc.metadata.get('doc_id', 'Unknown')}]" for doc in docs]
        id_list_str = ", ".join(doc_ids) if doc_ids else "검색 결과 없음"
        
        # 2. Graph DB 검색 (g_context: 텍스트 기전 정보, g_logs: 에이전트 사고 로그 리스트)
        g_context, g_logs = agent_manager.retrieve_graph_context(state["question"])
        
        # 3. 로그 결합 (Vector 요약 + ID 리스트 + Graph 상세 로그들)
        # 리스트 합치기 연산을 통해 순서대로 로그가 쌓입니다.
        vector_log = [f"📚 [Vector DB] 의미 검색 완료: {len(docs)}건의 문헌 확보 : {id_list_str}"]
        combined_logs = vector_log + g_logs
    
        return {
            "vector_context": docs, 
            "graph_context": g_context, 
            "logs": combined_logs
        }

    def irrelevant_node(state: GraphState):
        """의학 분야와 무관한 질문일 경우 처리하는 노드입니다."""
        logging.info("▶️ [Node: Irrelevant Query] 실행 중... (검색 차단)")
        return {"final_answer": "🚫 의학, 약학, 생물학 관련 질문만 답변할 수 있습니다. (예: 아스피린의 기전은 무엇인가요?)"}

    def synthesizer_node(state: GraphState):
        """검색된 컨텍스트를 종합하여 최종 답변을 도출합니다."""
        logging.info("▶️ [Node: Synthesizer] 실행 중...")
        # 상태에서 검색 결과 가져오기 (없으면 빈 값으로 처리)
        v_docs = state.get("vector_context", [])
        g_ctx = state.get("graph_context", "")
        
        # 에이전트 매니저를 호출하여 최종 답변 생성 
        # (이전의 무조건 '참고 문헌' 리스트를 덧붙이던 코드를 완전히 삭제하여 에러 및 출력 오류를 방지)
        answer = agent_manager.synthesize_answer(state["question"], v_docs, g_ctx)
        return {"final_answer": answer}

    # 3) 노드를 워크플로우에 등록
    workflow.add_node("router", router_node)
    workflow.add_node("search_vector", vector_retriever_node)
    workflow.add_node("search_graph", graph_retriever_node)
    workflow.add_node("search_both", both_retriever_node)
    workflow.add_node("irrelevant", irrelevant_node)
    workflow.add_node("synthesize", synthesizer_node)

    # 4) 엣지(Edge) 및 제어 흐름 정의
    # 시작점 설정
    workflow.add_edge(START, "router")

    # 조건부 라우팅 로직
    def route_condition(state: GraphState) -> str:
        return state["route"]

    # Router의 결정에 따라 다른 검색 노드로 분기 (Conditional Edges)
    workflow.add_conditional_edges(
        "router",
        route_condition,
        {
            "vector": "search_vector",
            "graph": "search_graph",
            "both": "search_both",
            "irrelevant": "irrelevant"
        }
    )

    # 검색이 완료되면 모두 답변 합성(Synthesize) 노드로 모임
    workflow.add_edge("search_vector", "synthesize")
    workflow.add_edge("search_graph", "synthesize")
    workflow.add_edge("search_both", "synthesize")

    # 무관한 질문은 합성을 거치지 않고 바로 종료
    workflow.add_edge("irrelevant", END)

    # 최종 노드에서 종료
    workflow.add_edge("synthesize", END)

    # 5) 그래프 컴파일
    app = workflow.compile()
    logging.info("✅ LangGraph 워크플로우 컴파일이 완료되었습니다.")
    return app


# 모듈 단독 테스트 로직
if __name__ == "__main__":
    from pprint import pprint

    # 매니저 객체 초기화 (DB 및 Agent)
    db_manager = BioDatabaseManager()
    agent_manager = BioAgentManager(db_manager)
    
    # 워크플로우 앱 컴파일
    app = create_workflow(agent_manager)

    # 테스트 질의 (그래프와 벡터 모두 탐색해야 하는 복합 질문)
    test_state = {"question": "아스피린(Aspirin)과 연관된 질환은 무엇이며, 관련 기전이 언급된 최신 논문 근거를 찾아주세요."}
    
    print("\n[워크플로우 실행 시작]")
    # 스트리밍 형태로 상태 변화 관찰
    for output in app.stream(test_state):
        for node_name, state_update in output.items():
            print(f"\n--- 현재 실행 노드: {node_name} ---")
            # 상태가 어떻게 업데이트되었는지 간략히 출력
            if "route" in state_update:
                print(f"👉 라우팅 경로: {state_update['route']}")
            if "vector_context" in state_update:
                print(f"👉 벡터 검색 완료: {len(state_update['vector_context'])}개의 문헌 찾음.")
            if "graph_context" in state_update:
                print(f"👉 그래프 탐색 완료: 텍스트 추출됨.")
            if "final_answer" in state_update:
                print("👉 최종 답변 도출 완료.")

    # 최종 결과 출력
    final_state = app.invoke(test_state)
    print("\n====================================")
    print("[최종 응답 결과]")
    print(final_state["final_answer"])
    print("====================================")
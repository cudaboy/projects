from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langchain_community.tools import DuckDuckGoSearchRun
from core.config import get_llm, set_langsmith_tracking
from core.prompts import analysis_prompt, parser

# 무료 웹 검색 툴 초기화
search_tool = DuckDuckGoSearchRun()

# 1. Graph State 정의 (context 추가)
class GraphState(TypedDict):
    question: str
    provider: str
    model_name: str
    use_langsmith: bool
    context: str  # 🌟 검색된 뉴스 기사/웹 문서를 담을 바구니
    analysis_result: dict

# 2. Node 함수 정의: 🔍 검색 노드 (새로 추가됨)
def retrieve_node(state: GraphState):
    """DuckDuckGo를 이용해 질문과 관련된 최신 뉴스나 웹 문서를 검색합니다."""
    query = state["question"]
    
    # 검색 정확도를 높이기 위해 쿼리 가공 가능
    search_query = f"{query} 최신 쟁점 논란 기사"
    
    try:
        # 웹 검색 실행하여 텍스트 데이터 추출
        retrieved_docs = search_tool.invoke(search_query)
    except Exception as e:
        retrieved_docs = f"검색 중 오류 발생 또는 결과 없음. 기존 지식을 바탕으로 분석합니다. 오류: {e}"
        
    return {"context": retrieved_docs}

# 3. Node 함수 정의: 🧠 분석 노드 (수정됨)
def analyze_node(state: GraphState):
    """검색된 데이터(context)와 질문을 LLM에 넘겨 진영별 시각을 분석합니다."""
    set_langsmith_tracking(state.get("use_langsmith", False))
    
    dynamic_llm = get_llm(
        provider=state.get("provider", "openai"),
        model_name=state.get("model_name", "gpt-4o-mini")
    )
    
    chain = analysis_prompt | dynamic_llm | parser
    
    # 🌟 프롬프트에 question과 함께, 방금 검색해 온 context를 같이 주입합니다.
    result = chain.invoke({
        "question": state["question"],
        "context": state["context"]
    })
    
    return {"analysis_result": result.model_dump()}

# 4. Graph 조립 (워크플로우 재설계)
workflow = StateGraph(GraphState)

# 노드 등록
workflow.add_node("retrieve_node", retrieve_node)
workflow.add_node("analyze_node", analyze_node)

# 🌟 흐름 연결: 시작 -> 검색 -> 분석 -> 끝
workflow.add_edge(START, "retrieve_node")
workflow.add_edge("retrieve_node", "analyze_node")
workflow.add_edge("analyze_node", END)

app_graph = workflow.compile()
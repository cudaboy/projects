import os
from typing import TypedDict, Dict, Any
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import create_react_agent
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage

# 사전에 정의한 프롬프트와 도구들 가져오기
from backend.core.prompts import (
    FINANCE_AGENT_PROMPT, 
    NEWS_AGENT_PROMPT, 
    STOCK_AGENT_PROMPT, 
    FUND_MANAGER_PROMPT
)
from backend.services.tools import finance_report, get_news, get_data, get_code

# ==========================================
# 1. 상태(State) 정의
# ==========================================
class CompanyState(TypedDict):
    """
    LangGraph가 실행되는 동안 각 노드 간에 데이터를 주고받을 메모리 구조입니다.
    """
    question: str               # 사용자가 입력한 종목명
    model_settings: Dict[str, Any] # 프론트엔드에서 넘어온 LLM 동적 설정값
    company_finance: str        # CFO 에이전트의 재무 분석 결과
    company_news: str           # Analyst 에이전트의 뉴스 분석 결과
    company_stock: str          # Trader 에이전트의 차트 분석 결과
    final_report: str           # 펀드매니저의 최종 요약 리포트

# ==========================================
# 2. LLM 동적 생성 헬퍼 함수
# ==========================================
def get_llm(settings: dict):
    """
    전달받은 설정값에 따라 적절한 제공자의 LLM 객체를 동적으로 생성하여 반환합니다.
    """
    provider = settings.get("provider", "OpenAI")
    model_name = settings.get("model_name", "gpt-4o")
    temperature = settings.get("temperature", 0.2)
    custom_api_key = settings.get("custom_api_key")
    
    if provider == "Anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(model=model_name, temperature=temperature, api_key=custom_api_key)
        
    elif provider == "Google Gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(model=model_name, temperature=temperature, google_api_key=custom_api_key)
        
    else:  # 기본값은 OpenAI
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=model_name, temperature=temperature, api_key=custom_api_key)

# ==========================================
# 3. 에이전트 노드(Node) 정의
# ==========================================
async def finance_node(state: CompanyState):
    """사용자가 입력한 회사의 재무제표 정보를 분석하는 노드"""
    llm = get_llm(state.get("model_settings", {}))
    tools = [finance_report, get_code]
    
    agent = create_react_agent(llm, tools)
    
    inputs = {
        "messages": [
            SystemMessage(content=FINANCE_AGENT_PROMPT),
            HumanMessage(content=state['question'])
        ]
    }
    
    result = await agent.ainvoke(inputs)
    return {'company_finance': result["messages"][-1].content}

async def news_node(state: CompanyState):
    """최신 뉴스 기사를 검색하여 모멘텀을 분석하는 노드"""
    ms = state.get("model_settings", {})
    
    # 🌟 UI에서 전달받은 네이버 API 키를 환경 변수에 동적 주입
    naver_id = ms.get("naver_client_id")
    naver_secret = ms.get("naver_client_secret")
    
    if naver_id and naver_secret:
        os.environ["NAVER_CLIENT_ID"] = naver_id
        os.environ["NAVER_CLIENT_SECRET"] = naver_secret
    else:
        # 키가 입력되지 않았다면, 기존 환경 변수에 남아있을 수 있는 값도 제거하여 DuckDuckGo Fallback 유도
        os.environ.pop("NAVER_CLIENT_ID", None)
        os.environ.pop("NAVER_CLIENT_SECRET", None)

    llm = get_llm(ms)
    tools = [get_news]
    
    agent = create_react_agent(llm, tools)
    
    inputs = {
        "messages": [
            SystemMessage(content=NEWS_AGENT_PROMPT),
            HumanMessage(content=state['question'])
        ]
    }
    
    result = await agent.ainvoke(inputs)
    return {'company_news': result["messages"][-1].content}

async def stock_node(state: CompanyState):
    """과거 주가 및 거래량 데이터를 가져와 기술적 진단을 수행하는 노드"""
    llm = get_llm(state.get("model_settings", {}))
    tools = [get_data, get_code]
    
    agent = create_react_agent(llm, tools)
    
    query = f"현재 날짜 기준으로 {state['question']} 주가 정보 분석해줘"
    inputs = {
        "messages": [
            SystemMessage(content=STOCK_AGENT_PROMPT),
            HumanMessage(content=query)
        ]
    }
    
    result = await agent.ainvoke(inputs)
    return {'company_stock': result["messages"][-1].content}

async def summarize_node(state: CompanyState):
    """3명의 전문가가 정리한 자료를 취합하여 최종 펀드매니저 리포트를 작성하는 노드"""
    llm = get_llm(state.get("model_settings", {}))
    
    prompt_template = PromptTemplate.from_template(FUND_MANAGER_PROMPT)
    chain = prompt_template | llm
    
    result = await chain.ainvoke({
        'company_finance': state.get('company_finance', ""),
        'company_news': state.get('company_news', ""),
        'company_stock': state.get('company_stock', ""),
        'question': state['question']
    })
    return {'final_report': result.content}

# ==========================================
# 4. 워크플로우(Graph) 구성 및 컴파일
# ==========================================
workflow = StateGraph(CompanyState)

workflow.add_node('finance_node', finance_node)
workflow.add_node('news_node', news_node)
workflow.add_node('stock_node', stock_node)
workflow.add_node('summarize_node', summarize_node)

workflow.add_edge(START, 'finance_node')
workflow.add_edge(START, 'news_node')
workflow.add_edge(START, 'stock_node')

workflow.add_edge('finance_node', 'summarize_node')
workflow.add_edge('news_node', 'summarize_node')
workflow.add_edge('stock_node', 'summarize_node')

workflow.add_edge('summarize_node', END)

app = workflow.compile()
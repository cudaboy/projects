import json
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
from backend.services.tools import finance_report, get_news, get_data, get_code, technical_report
from backend.services.scoring.investment_score import build_risk_summary
from backend.services.valuation import build_valuation_summary

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
    company_valuation: str      # Valuation Agent의 DCF-lite/상대가치 요약
    company_risk: str           # Risk Manager의 결정론적 점수/리스크 요약
    investment_score: str       # 최종 판단 보조용 알고리즘 점수 JSON
    final_report: str           # 펀드매니저의 최종 요약 리포트

# ==========================================
# 2. LLM 동적 생성 헬퍼 함수
# ==========================================
def _is_thinking_model(provider: str, model_name: str) -> bool:
    """Best-effort routing for models that expose explicit reasoning/thinking."""
    provider_key = (provider or "").lower()
    model_key = (model_name or "").lower()
    if provider_key == "openai":
        return model_key.startswith(("o1", "o3", "o4")) or "gpt-5" in model_key
    if provider_key in {"grok", "xai"}:
        return "grok-3-mini" in model_key or "grok-4" in model_key or "reasoning" in model_key
    if provider_key == "anthropic":
        return "claude-3-7" in model_key or "claude-4" in model_key or "sonnet-4" in model_key or "opus-4" in model_key
    # Local Ollama models vary; enable only by model naming convention.
    if provider_key == "ollama":
        return any(token in model_key for token in ("think", "reason", "r1", "qwq"))
    return False


def _reasoning_kwargs(provider: str, model_name: str, settings: dict) -> dict:
    """Translate UI thinking settings into provider-specific kwargs.

    Unsupported combinations intentionally return {} so existing providers keep
    working even when the UI sends enable_thinking=False.
    """
    if not settings.get("enable_thinking") or not _is_thinking_model(provider, model_name):
        return {}

    effort = settings.get("reasoning_effort") or "medium"
    provider_key = (provider or "").lower()
    if provider_key == "openai":
        return {"reasoning_effort": effort}
    if provider_key in {"grok", "xai"}:
        return {"extra_body": {"reasoning_effort": effort}}
    if provider_key == "anthropic":
        budget = {"low": 1024, "medium": 4096, "high": 8192}.get(effort, 4096)
        return {"thinking": {"type": "enabled", "budget_tokens": budget}, "max_tokens": max(2048, budget + 1024)}
    if provider_key == "google gemini":
        return {"thinking_level": effort, "include_thoughts": True}
    if provider_key == "ollama":
        return {"extra_body": {"think": True}}
    return {}


# ==========================================
# 2. LLM 동적 생성 헬퍼 함수
# ==========================================
def get_llm(settings: dict):
    """
    전달받은 설정값에 따라 적절한 제공자의 LLM 객체를 동적으로 생성하여 반환합니다.
    OpenAI-compatible provider(Grok/xAI, Ollama, OpenRouter)는 ChatOpenAI의
    base_url을 활용하여 동일한 LangChain tool-calling 경로를 사용합니다.
    """
    provider = settings.get("provider", "OpenAI")
    model_name = settings.get("model_name", "gpt-4o")
    temperature = settings.get("temperature", 0.2)
    custom_api_key = settings.get("custom_api_key")
    base_url = settings.get("base_url")
    provider_key = (provider or "OpenAI").lower()

    if provider == "Anthropic":
        from langchain_anthropic import ChatAnthropic
        kwargs = _reasoning_kwargs(provider, model_name, settings)
        return ChatAnthropic(model=model_name, temperature=temperature, api_key=custom_api_key, **kwargs)

    elif provider == "Google Gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        kwargs = _reasoning_kwargs(provider, model_name, settings)
        return ChatGoogleGenerativeAI(model=model_name, temperature=temperature, google_api_key=custom_api_key, **kwargs)

    else:
        from langchain_openai import ChatOpenAI
        kwargs = _reasoning_kwargs(provider, model_name, settings)
        if provider_key in {"grok", "xai"}:
            return ChatOpenAI(
                model=model_name,
                temperature=temperature,
                api_key=custom_api_key or os.getenv("XAI_API_KEY"),
                base_url=base_url or "https://api.x.ai/v1",
                **kwargs,
            )
        if provider_key == "ollama":
            return ChatOpenAI(
                model=model_name,
                temperature=temperature,
                api_key=custom_api_key or "ollama",
                base_url=base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1"),
                **kwargs,
            )
        if provider_key == "openrouter":
            return ChatOpenAI(
                model=model_name,
                temperature=temperature,
                api_key=custom_api_key or os.getenv("OPENROUTER_API_KEY"),
                base_url=base_url or "https://openrouter.ai/api/v1",
                **kwargs,
            )
        return ChatOpenAI(model=model_name, temperature=temperature, api_key=custom_api_key, base_url=base_url, **kwargs)

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
    tools = [get_data, get_code, technical_report]
    
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

async def risk_node(state: CompanyState):
    """전문가 결과를 바탕으로 결정론적 점수와 리스크 요약을 생성하는 노드"""
    risk_summary = build_risk_summary(
        finance_text=state.get('company_finance', ""),
        news_text=state.get('company_news', ""),
        technical_text=state.get('company_stock', ""),
        valuation_text=state.get('company_valuation', ""),
    )
    risk_json = json.dumps(risk_summary, ensure_ascii=False, indent=2)
    return {
        'company_risk': risk_json,
        'investment_score': risk_json,
    }


async def valuation_node(state: CompanyState):
    """재무/기술 분석 결과에서 DCF-lite 및 상대가치 보조 지표를 생성하는 노드"""
    valuation_summary = build_valuation_summary(
        finance_text=state.get('company_finance', ""),
        technical_text=state.get('company_stock', ""),
    )
    return {'company_valuation': json.dumps(valuation_summary, ensure_ascii=False, indent=2)}


async def summarize_node(state: CompanyState):
    """3명의 전문가가 정리한 자료와 리스크 점수를 취합하여 최종 펀드매니저 리포트를 작성하는 노드"""
    llm = get_llm(state.get("model_settings", {}))
    
    prompt_template = PromptTemplate.from_template(FUND_MANAGER_PROMPT)
    chain = prompt_template | llm
    
    result = await chain.ainvoke({
        'company_finance': state.get('company_finance', ""),
        'company_news': state.get('company_news', ""),
        'company_stock': state.get('company_stock', ""),
        'company_valuation': state.get('company_valuation', ""),
        'company_risk': state.get('company_risk', ""),
        'investment_score': state.get('investment_score', ""),
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
workflow.add_node('valuation_node', valuation_node)
workflow.add_node('risk_node', risk_node)
workflow.add_node('summarize_node', summarize_node)

workflow.add_edge(START, 'finance_node')
workflow.add_edge(START, 'news_node')
workflow.add_edge(START, 'stock_node')

workflow.add_edge('finance_node', 'valuation_node')
workflow.add_edge('news_node', 'risk_node')
workflow.add_edge('stock_node', 'valuation_node')
workflow.add_edge('valuation_node', 'risk_node')
workflow.add_edge('risk_node', 'summarize_node')

workflow.add_edge('summarize_node', END)

app = workflow.compile()
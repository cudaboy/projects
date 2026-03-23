import os
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

# 🌟 외부(LangGraph 노드)에서 모니터링 활성화 여부를 세팅할 수 있는 함수
def set_langsmith_tracking(enable: bool):
    if enable:
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_PROJECT"] = "Spectrum_View_Project"
    else:
        os.environ["LANGCHAIN_TRACING_V2"] = "false"

def get_llm(provider="openai", model_name="gpt-4o-mini", temperature=0.7):
    """
    프론트엔드에서 선택된 모델 정보에 맞추어 동적으로 LLM을 생성합니다.
    """
    provider = provider.lower()
    
    if provider == "openai":
        return ChatOpenAI(model=model_name, temperature=temperature)
    elif provider in ["claude", "anthropic"]:
        return ChatAnthropic(model_name=model_name, temperature=temperature)
    elif provider in ["gemini", "google"]:
        return ChatGoogleGenerativeAI(model=model_name, temperature=temperature)
    else:
        raise ValueError(f"지원하지 않는 LLM 제공자입니다: {provider}")
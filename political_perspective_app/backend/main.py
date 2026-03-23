# main.py : 에러 추적(Traceback) 기능과 LangSmith 토글 기능, 예비 포트 자동 할당 기능

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import sys
import os
import socket
import traceback  # 🌟 에러 상세 출력을 위한 모듈 추가

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from services.graph import app_graph
from utils import save_to_history

app = FastAPI(title="Spectrum View API", description="정치 스펙트럼 분석 백엔드 API")

# 🌟 요청 데이터에 프론트엔드에서 보낸 설정값들 추가
class AnalyzeRequest(BaseModel):
    question: str
    provider: str = "openai"
    model_name: str = "gpt-4o-mini"
    use_langsmith: bool = False

@app.get("/")
def read_root():
    return {"message": "Spectrum View Backend API is running!"}

@app.post("/analyze")
def analyze_issue(request: AnalyzeRequest):
    try:
        # 🌟 LangGraph에 질문과 설정값을 함께 전달
        initial_state = {
            "question": request.question,
            "provider": request.provider,
            "model_name": request.model_name,
            "use_langsmith": request.use_langsmith
        }
        result_state = app_graph.invoke(initial_state)
        
        final_result = result_state['analysis_result']
        save_to_history(request.question, final_result)
        
        return final_result
    except Exception as e:
        # 🌟 에러 발생 시 백엔드 터미널에 붉은 글씨로 상세 원인 출력
        traceback.print_exc() 
        raise HTTPException(status_code=500, detail=f"LLM 분석 중 오류 발생: {str(e)}")

def find_free_port(ports):
    """주어진 포트 리스트 중 사용 가능한 첫 번째 포트를 반환합니다."""
    for port in ports:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(('localhost', port)) != 0:
                return port
    return None

if __name__ == "__main__":
    import uvicorn
    ports_to_try = [8000, 8001, 8002]
    target_port = find_free_port(ports_to_try)
    
    if target_port:
        print(f"🚀 사용 가능한 포트({target_port})를 찾아 백엔드 서버를 시작합니다.")
        uvicorn.run("main:app", host="0.0.0.0", port=target_port, reload=True)
    else:
        print("🚨 지정된 예비 포트(8000, 8001, 8002)가 모두 사용 중입니다.")
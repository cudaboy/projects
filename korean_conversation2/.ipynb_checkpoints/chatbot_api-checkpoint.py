import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from rag_handler import initialize_rag_system
from utils import setup_logger

# =====================================================================
# 1. 로깅(Logging) 설정
# =====================================================================
# utils.py에서 만들어둔 로거를 불러옵니다. 
# 이제 이 파일에서 logger.info() 등을 쓰면 파일과 화면에 동시에 기록됩니다.
logger = setup_logger()

# =====================================================================
# 2. FastAPI 앱 객체 생성
# =====================================================================
# 'app'이라는 이름의 FastAPI 서버 인스턴스를 만듭니다.
# app.py에서 uvicorn을 실행할 때 'chatbot_api:app'이라고 적었던 이유가 바로 이 객체 때문입니다.
app = FastAPI(title="Korean Conversation RAG API")

# =====================================================================
# 3. RAG 시스템(모델 & 벡터DB) 초기화
# =====================================================================
# 서버가 구동될 때 단 한 번만 실행되어 무거운 모델과 DB를 메모리에 적재합니다.
# 이 작업은 rag_handler.py에 위임하여 코드를 깔끔하게 유지합니다.
logger.info("FastAPI 서버 가동 준비: RAG 시스템을 초기화합니다...")
rag_chain = initialize_rag_system()

# =====================================================================
# 4. 데이터 양식(Schema) 정의
# =====================================================================
# 프론트엔드(UI)에서 백엔드로 데이터를 보낼 때 반드시 지켜야 하는 규칙입니다.
# 무조건 'question'이라는 키(Key) 값에 문자열(String) 형태로 질문을 담아 보내야 합니다.
class ChatRequest(BaseModel):
    question: str

# =====================================================================
# 5. API 엔드포인트 생성 (채팅 요청 처리 창구)
# =====================================================================
# @app.post("/chat/")는 "누군가 이 서버의 /chat/ 주소로 POST 요청을 보내면 아래 함수를 실행해라!"라는 뜻입니다.
@app.post("/chat/")
def get_chatbot_response(request: ChatRequest):
    """
    프론트엔드에서 들어온 질문을 받아 RAG 체인에 던지고, 완성된 답변을 반환합니다.
    """
    user_question = request.question
    
    # 1단계: 어떤 질문이 들어왔는지 로그에 기록합니다.
    logger.info(f"USER_REQUEST: {user_question}") 

    try:
        # 2단계: rag_handler에서 만들어둔 rag_chain에 질문을 통과시킵니다.
        # 문서 검색 -> 프롬프트 생성 -> 모델 추론 -> 텍스트 디코딩이 이 한 줄에서 모두 일어납니다.
        answer = rag_chain.invoke(user_question)
        
        # 3단계: 모델이 어떤 답변을 내놓았는지 로그에 기록합니다.
        logger.info(f"BOT_RESPONSE: {answer}")
        
        # 4단계: 최종 결과물을 딕셔너리(JSON) 형태로 프론트엔드에 돌려줍니다.
        return {"question": user_question, "answer": answer}
        
    except Exception as e:
        # 에러가 발생하면 서버가 죽지 않도록 방어하고, 500(서버 내부 오류)
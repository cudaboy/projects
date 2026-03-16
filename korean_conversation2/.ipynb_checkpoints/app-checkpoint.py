import subprocess
import sys
import time
from utils import setup_logger

# utils.py에 정의된 로거를 불러와 화면과 파일에 로그를 기록할 준비를 합니다.
logger = setup_logger()

def main():
    logger.info("🚀 프로젝트 2 RAG 시스템 통합 구동을 시작합니다...")

    # =====================================================================
    # 1. FastAPI 백엔드 서버 구동 (chatbot_api.py)
    # =====================================================================
    logger.info("📡 FastAPI 백엔드 서버를 시작합니다 (Port: 8000)...")
    
    # subprocess.Popen을 사용하여 백그라운드에서 백엔드 서버를 독립적으로 실행합니다.
    # sys.executable은 현재 파이썬 가상환경의 python.exe 경로를 자동으로 찾아줍니다.
    backend_process = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "chatbot_api:app", "--host", "0.0.0.0", "--port", "8000"],
        stdout=sys.stdout, # 백엔드의 출력(로그)을 현재 터미널 화면에 그대로 보여줍니다.
        stderr=sys.stderr  # 에러 메시지도 현재 터미널 화면에 보여줍니다.
    )

    # ⏳ 백엔드 서버가 완전히 켜질 때까지 3초 정도 대기합니다.
    # (rag_handler.py가 무거운 모델과 ChromaDB를 메모리에 올릴 시간을 벌어주기 위함입니다.)
    time.sleep(3)

    # =====================================================================
    # 2. Streamlit 프론트엔드 서버 구동 (main.py)
    # =====================================================================
    logger.info("🖥️ Streamlit 프론트엔드 UI를 시작합니다...")
    
    # 마찬가지로 백그라운드에서 프론트엔드(UI) 서버를 독립적으로 실행합니다.
    # 실행 대상 파일이 'main.py'이며, 포트는 8501번을 사용합니다.
    frontend_process = subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run", "main.py", "--server.port", "8501"],
        stdout=sys.stdout,
        stderr=sys.stderr
    )

    # =====================================================================
    # 3. 프로세스 유지 및 안전한 종료(Graceful Shutdown) 처리
    # =====================================================================
    try:
        # wait() 함수를 통해 백엔드와 프론트엔드 서버가 켜진 상태로 무한정 대기합니다.
        # 사용자가 수동으로 끄기 전까지 프로그램이 종료되지 않게 막아줍니다.
        backend_process.wait()
        frontend_process.wait()
        
    except KeyboardInterrupt:
        # 사용자가 터미널에서 'Ctrl + C'를 눌러 강제 종료 신호를 보냈을 때 작동합니다.
        logger.info("🛑 종료 신호(Ctrl+C)를 받았습니다. 서버들을 안전하게 종료합니다...")
        
        # 좀비 프로세스가 남지 않도록 두 서버에 각각 종료(terminate) 명령을 내립니다.
        backend_process.terminate()
        frontend_process.terminate()
        
        # 완전히 꺼질 때까지 잠시 기다려줍니다.
        backend_process.wait()
        frontend_process.wait()
        
        logger.info("✅ 모든 서버가 깔끔하게 종료되었습니다.")

if __name__ == "__main__":
    main()
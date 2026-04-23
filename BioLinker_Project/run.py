"""
===============================================================================
[File Role]
이 파일(run.py)은 BioLinker 프로젝트의 전체 수명 주기를 관리하는 '최상위 통합 컨트롤러(CLI Entry Point)'입니다.

[상세 기능 및 역할]
1. 인프라 일괄 구축 (--build): 
   - RAW 데이터를 파싱하고, BioClinical-ModernBERT를 이용해 Vector DB 및 Graph DB를 자동으로 생성합니다.
2. 성능 검증 및 리포팅 (--eval): 
   - Ragas 프레임워크를 구동하여 구축된 RAG 시스템의 신뢰도와 사실 부합성을 평가합니다.
3. 통합 서비스 구동 (--start): 
   - FastAPI 백엔드(api.py)와 Streamlit 프론트엔드(main.py)를 병렬 프로세스로 실행하여 
     즉시 사용 가능한 풀스택 환경을 제공합니다.
4. 프로세스 생명주기 관리: 
   - 시스템 종료 시(Ctrl+C) 백그라운드에서 실행 중인 API 서버까지 안전하게 중단시켜 리소스 누수를 방지합니다.
===============================================================================
"""

import argparse
import subprocess
import sys
import time
import logging
import os
from pathlib import Path

# 로깅 설정: 시스템의 진행 상황을 터미널에 명확히 출력
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 프로젝트 루트 경로 확보 (어떤 디렉토리에서 실행하든 절대 경로 기반으로 동작하도록 설정)
PROJECT_ROOT = Path(__file__).resolve().parent

def run_build():
    """[Step 1] 데이터 전처리 및 하이브리드 DB(Vector+Graph) 구축 스크립트 실행"""
    logging.info("🚀 데이터 인덱싱 파이프라인(build_index.py)을 시작합니다...")
    script_path = PROJECT_ROOT / "scripts" / "build_index.py"
    try:
        subprocess.run([sys.executable, str(script_path)], check=True)
    except subprocess.CalledProcessError as e:
        logging.error(f"❌ 빌드 도중 오류가 발생했습니다: {e}")

def run_eval():
    """[Step 2] RAG 시스템의 신뢰도 및 할루시네이션 평가 스크립트 실행"""
    logging.info("🔬 RAG 시스템 신뢰도 평가(evaluate.py)를 시작합니다...")
    script_path = PROJECT_ROOT / "scripts" / "evaluate.py"
    try:
        subprocess.run([sys.executable, str(script_path)], check=True)
    except subprocess.CalledProcessError as e:
        logging.error(f"❌ 평가 도중 오류가 발생했습니다: {e}")

def run_api():
    """[Option] FastAPI 백엔드 서버 단독 구동"""
    logging.info("⚙️ FastAPI 백엔드 서버를 구동합니다 (포트: 8000)...")
    api_path = "app.api:app"
    try:
        # uvicorn 모듈을 사용하여 백엔드 실행
        subprocess.run([sys.executable, "-m", "uvicorn", api_path, "--host", "0.0.0.0", "--port", "8000", "--reload"], cwd=PROJECT_ROOT)
    except KeyboardInterrupt:
        logging.info("🛑 API 서버가 사용자에 의해 중단되었습니다.")

def run_ui():
    """[Option] Streamlit 프론트엔드 대시보드 단독 구동"""
    logging.info("🎨 Streamlit 프론트엔드 대시보드를 구동합니다...")
    # [수정 사항] streamlit_app.py 대신 분리된 main.py를 실행합니다.
    app_path = PROJECT_ROOT / "app" / "main.py"
    try:
        subprocess.run(["streamlit", "run", str(app_path)], cwd=PROJECT_ROOT)
    except KeyboardInterrupt:
        logging.info("🛑 UI 서버가 사용자에 의해 중단되었습니다.")

def run_start():
    """[Final] 백엔드 API와 프론트엔드 UI를 동시에 실행 (풀스택 데모 모드)"""
    logging.info("🌟 BioLinker 풀스택 시스템(API + UI) 통합 구동을 시작합니다...")
    
    # 1. API 서버를 백그라운드 프로세스로 실행
    api_cmd = [sys.executable, "-m", "uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]
    api_process = subprocess.Popen(api_cmd, cwd=PROJECT_ROOT)
    
    # 2. 서버가 완전히 뜰 때까지 잠시 대기 (모델 로딩 시간 고려)
    logging.info("⏳ 백엔드 서버 초기화 대기 중... (최초 실행 시 모델 로딩으로 인해 시간이 소요될 수 있습니다)")
    time.sleep(5)
    
    # 3. UI 서버를 포그라운드 프로세스로 실행
    ui_app_path = PROJECT_ROOT / "app" / "main.py"
    ui_cmd = ["streamlit", "run", str(ui_app_path)]
    
    try:
        # 사용자가 브라우저에서 확인할 수 있도록 UI를 메인으로 실행
        subprocess.run(ui_cmd, cwd=PROJECT_ROOT, check=True)
    except KeyboardInterrupt:
        logging.info("\n🛑 시스템 종료 요청을 받았습니다. 모든 서버를 종료합니다.")
    finally:
        # 종료 시 백그라운드에 떠 있는 API 프로세스도 함께 종료
        api_process.terminate()
        api_process.wait()
        logging.info("✅ BioLinker 시스템이 안전하게 종료되었습니다.")

def main():
    parser = argparse.ArgumentParser(
        description="BioLinker 프로젝트 통합 실행 도구",
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    # 상호 배타적 그룹 설정: 한 번에 하나의 기능만 수행하도록 설계
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--build", action="store_true", help="초기 데이터 전처리 및 하이브리드 DB 인덱싱 실행")
    group.add_argument("--eval", action="store_true", help="시스템 신뢰도 자동 평가(Ragas) 실행")
    group.add_argument("--api", action="store_true", help="FastAPI 백엔드 서버만 실행")
    group.add_argument("--ui", action="store_true", help="Streamlit 프론트엔드 UI만 실행")
    group.add_argument("--start", action="store_true", help="API와 UI를 통합 모드로 동시 실행")

    args = parser.parse_args()

    # 필수 디렉토리 확인 가드 로직
    if not (PROJECT_ROOT / "app").exists():
        logging.error("❌ 'app' 디렉토리가 존재하지 않습니다. 프로젝트 루트 경로를 확인하세요.")
        sys.exit(1)

    if args.build:
        run_build()
    elif args.eval:
        run_eval()
    elif args.api:
        run_api()
    elif args.ui:
        run_ui()
    elif args.start:
        run_start()

if __name__ == "__main__":
    main()
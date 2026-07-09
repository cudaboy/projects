"""BioLinker CLI entry point."""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
import time
from pathlib import Path
from urllib.request import urlopen
from urllib.error import URLError

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
PROJECT_ROOT = Path(__file__).resolve().parent


def run_build():
    logging.info("🚀 데이터 인덱싱 파이프라인(build_index.py)을 시작합니다...")
    script_path = PROJECT_ROOT / "scripts" / "build_index.py"
    subprocess.run([sys.executable, str(script_path)], check=True)


def run_eval():
    logging.info("🔬 RAG 시스템 신뢰도 평가(evaluate.py)를 시작합니다...")
    script_path = PROJECT_ROOT / "scripts" / "evaluate.py"
    subprocess.run([sys.executable, str(script_path)], check=True)


def run_api():
    logging.info("⚙️ FastAPI 백엔드 서버를 구동합니다 (포트: 8000)...")
    subprocess.run(
        [sys.executable, "-m", "uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000", "--reload"],
        cwd=PROJECT_ROOT,
    )


def run_ui():
    logging.info("🎨 Streamlit 프론트엔드 대시보드를 구동합니다...")
    subprocess.run(["streamlit", "run", str(PROJECT_ROOT / "app" / "main.py")], cwd=PROJECT_ROOT)


def wait_for_ready(url: str, timeout_seconds: int = 120) -> bool:
    start = time.time()
    while time.time() - start < timeout_seconds:
        try:
            with urlopen(url, timeout=5) as response:
                payload = response.read().decode("utf-8")
                if '"ready": true' in payload or '"status": "ready"' in payload:
                    return True
        except URLError:
            pass
        time.sleep(2)
    return False


def run_start():
    logging.info("🌟 BioLinker 풀스택 시스템(API + UI) 통합 구동을 시작합니다...")
    api_cmd = [sys.executable, "-m", "uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]
    api_process = subprocess.Popen(api_cmd, cwd=PROJECT_ROOT)
    try:
        logging.info("⏳ 백엔드 readiness 확인 중...")
        if not wait_for_ready("http://127.0.0.1:8000/health/ready", timeout_seconds=180):
            raise RuntimeError("백엔드 readiness 확인에 실패했습니다.")
        logging.info("✅ 백엔드 readiness 확인 완료. Streamlit UI를 시작합니다.")
        subprocess.run(["streamlit", "run", str(PROJECT_ROOT / "app" / "main.py")], cwd=PROJECT_ROOT, check=True)
    finally:
        api_process.terminate()
        api_process.wait()
        logging.info("✅ BioLinker 시스템이 안전하게 종료되었습니다.")


def main():
    parser = argparse.ArgumentParser(description="BioLinker 프로젝트 통합 실행 도구", formatter_class=argparse.RawTextHelpFormatter)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--build", action="store_true", help="초기 데이터 전처리 및 하이브리드 DB 인덱싱 실행")
    group.add_argument("--eval", action="store_true", help="시스템 신뢰도 자동 평가(Ragas) 실행")
    group.add_argument("--api", action="store_true", help="FastAPI 백엔드 서버만 실행")
    group.add_argument("--ui", action="store_true", help="Streamlit 프론트엔드 UI만 실행")
    group.add_argument("--start", action="store_true", help="API와 UI를 통합 모드로 동시 실행")
    args = parser.parse_args()

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

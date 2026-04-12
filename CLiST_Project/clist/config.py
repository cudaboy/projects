"""
=============================================================================================
[Configuration Module] CLiST Project Central Control Tower
=============================================================================================

1. 개요 및 설계 목적 (Overview & Design Intent)
본 모듈은 CLiST(Multimodal Time-Frequency Fusion Network) 모델의 모든 실행 환경과 
하이퍼파라미터를 중앙 집중식으로 관리하는 '프로젝트 제어 센터'입니다. 

실무 딥러닝 프로젝트에서 코드 곳곳에 하드코딩된 숫자(Magic Numbers)는 유지보수를 어렵게 하고 
실험의 재현성을 떨어뜨립니다. 본 파일은 모든 변수를 한곳에 모아 관리함으로써, 
데이터 사이언티스트가 소스 코드 로직을 건드리지 않고도 오직 이 파일의 수치만 변경하여 
다양한 실험을 수행할 수 있도록 돕습니다.

2. 주요 기능 (Main Features)
  - 실험 일관성 유지: 하드웨어 설정, 데이터 경로, 모델 구조 파라미터를 통합 관리합니다.
  - 파이프라인 자동화: tune.py에서 산출된 최적의 하이퍼파라미터를 자동으로 탐색하고 로드합니다.
  - 환경 독립성: Pathlib을 사용하여 Windows, Linux 등 운영체제에 상관없이 경로를 올바르게 처리합니다.
  - 가독성 및 문서화: 각 설정값이 모델 학습의 어느 단계에 영향을 주는지 명확한 주석을 제공합니다.

3. 데이터 흐름 (Data Flow)
  - Default: 본 파일에 선언된 기본값 사용.
  - Auto-Update: tune.py 실행 완료 시 생성되는 'best_params.json'을 감지하여 
                LEARNING_RATE와 WEIGHT_DECAY를 실시간으로 갱신.
=============================================================================================
"""

import torch
import os
import json
from pathlib import Path

class Config:
    """
    프로젝트의 전역 설정을 담고 있는 정적 클래스입니다.
    다른 모듈에서 'from config import Config' 후 'Config.변수명'으로 즉시 접근 가능합니다.
    """
    
    # ==========================================
    # 🖥️ 1. 하드웨어 및 실행 환경 (Hardware & Env)
    # ==========================================
    # GPU 가속 여부 결정 (NVIDIA GPU가 있으면 cuda, 없으면 cpu 사용)
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 데이터 로딩 병렬 처리를 위한 CPU 코어 수 (DataLoader의 num_workers에 할당)
    # vCPU 개수에 따라서 숫자를 설정하면 GPU 연산 중 데이터 공급 병목을 방지할 수 있습니다.
    NUM_WORKERS = 8
    
    # FP16 혼합 정밀도 연산 사용 여부 (학습 속도 가속 및 VRAM 절약)
    USE_AMP = True
    
    # CPU 메모리에 상주하는 데이터를 GPU로 보낼 때 고정 메모리(Pinned Memory)를 사용하여 전송 효율 증대
    PIN_MEMORY = False

    # ==========================================
    # 📂 2. 데이터 및 출력 경로 (Paths & Directories)
    # ==========================================
    # 프로젝트 루트 디렉토리 설정
    BASE_DIR = Path('/workspace/work4')
    
    # 전처리된 데이터가 저장된 기본 폴더
    DATA_DIR = BASE_DIR / 'processed_data'
    
    # 학습 및 검증용 JSON/CSV/BIN 데이터셋 경로
    TRAIN_PATH = DATA_DIR / 'Training'
    VAL_PATH = DATA_DIR / 'Validation'
    
    # 결과물(가중치, 리포트, 그래프)이 저장될 출력 폴더
    OUTPUT_DIR = BASE_DIR / 'outputs'
    
    # 학습 완료 후 저장될 최고 성능의 모델 가중치 파일 경로
    MODEL_SAVE_PATH = OUTPUT_DIR / 'best_clist_model.pth'
    
    # 최종 검증 성능(Classification Report)이 텍스트로 기록될 경로
    REPORT_SAVE_PATH = OUTPUT_DIR / 'final_evaluation_report.txt'
    
    # 출력 폴더가 존재하지 않을 경우 자동 생성 (런타임 에러 방지)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ==========================================
    # 🧠 3. 모델 아키텍처 및 하이퍼파라미터 (Model & Hyperparams)
    # ==========================================
    # 타겟 변수의 클래스 개수 (0:정상, 1:관심, 2:경고, 3:위험)
    NUM_CLASSES = 4
    
    # 전체 데이터셋을 반복 학습할 횟수
    EPOCHS = 50
    
    # 한 번의 연산에 투입되는 샘플 개수 (GPU 메모리 사양에 맞춰 조절)
    BATCH_SIZE = 128
    
    # 시계열 통계량(평균, 분산 등)을 추출하기 위한 롤링 윈도우 크기 (단위: 프레임 또는 초)
    WINDOW_SIZE = 5
    
    # ----------------------------------------------------
    # 🌟 [학습 제어 변수 - tune.py 미실행 시 적용되는 기본값]
    # ----------------------------------------------------
    # 가중치 업데이트 크기를 결정하는 학습률
    LEARNING_RATE = 1e-4
    # 모델의 복잡도를 제어하여 과적합을 방지하는 가중치 감쇠(L2 Regularization)
    WEIGHT_DECAY = 1e-2
    # 가중치를 어떤 방향과 속도로 업데이트할지 결정하는 최적화 알고리즘 엔진
    OPTIMIZER = "AdamW"    
    # 모델이 데이터의 복잡한 패턴을 기억하고 학습할 수 있는 용량(Capacity)을 결정하는 은닉층 차원 수
    HIDDEN_DIM = 64    
    # 학습 중 뉴런을 무작위로 꺼버림으로써, 특정 변수에만 과도하게 의존하는 과적합을 방지하는 확률
    DROPOUT_RATE = 0.4

    # ==========================================
    # 🛑 4. 학습 전략 및 조기 종료 (Training Strategy)
    # ==========================================
    # 검증 점수가 개선되지 않을 때 몇 번의 Epoch를 더 기다릴지 설정
    EARLY_STOPPING_PATIENCE = 5
    
    # 성능 정체 시 학습률을 줄여주는 스케줄러(Scheduler)의 대기 시간
    SCHEDULER_PATIENCE = 2

    # ==========================================
    # 📈 5. MLOps 로깅 및 튜닝 (Monitoring & Tuning)
    # ==========================================
    # MLflow 대시보드에서 실험을 구분하기 위한 고유 이름
    EXPERIMENT_NAME = "CLiST_Predictive_Maintenance_v2"
    
    # Optuna 자동 튜닝 시 수행할 실험 횟수
    OPTUNA_TRIALS = 50
    
    # Optuna가 탐색할 학습률의 하한값과 상한값 (Log Scale 기반)
    OPTUNA_LR_RANGE = (1e-5, 1e-2)
    # Optuna가 탐색할 가중치 감쇠의 범위
    OPTUNA_WD_RANGE = (1e-4, 1e-1)
    
    # 튜닝 시 시간 절약을 위해 사용할 전체 데이터 대비 샘플링 비율 (0.1 = 10%)
    TUNE_DATA_RATIO = 0.1

    # ==========================================
    # ⚙️ 6. 자동화 로직 (Automation Logic)
    # ==========================================
    @classmethod
    def load_best_params(cls):
        """
        [자동 파라미터 및 실험명 동기화 함수]
        tune.py 실행 결과로 생성된 'best_params.json' 파일을 탐색합니다.
        파일이 존재할 경우 최적의 학습률(LR)과 가중치 감쇠(WD) 수치를 덮어쓰고,
        MLflow에 기록될 실험 이름(EXPERIMENT_NAME)도 자동으로 연동시킵니다.
        """
        param_file = cls.OUTPUT_DIR / "best_params.json"
        
        if param_file.exists():
            try:
                with open(param_file, 'r', encoding='utf-8') as f:
                    best_params = json.load(f)
                
                # JSON에 저장된 최적값 및 저장된 실험 이름을 클래스 변수에 동적으로 할당
                cls.LEARNING_RATE = best_params.get("LEARNING_RATE", cls.LEARNING_RATE)
                cls.WEIGHT_DECAY = best_params.get("WEIGHT_DECAY", cls.WEIGHT_DECAY)
                cls.OPTIMIZER = best_params.get("OPTIMIZER", cls.OPTIMIZER)
                cls.HIDDEN_DIM = best_params.get("HIDDEN_DIM", cls.HIDDEN_DIM)
                cls.DROPOUT_RATE = best_params.get("DROPOUT_RATE", cls.DROPOUT_RATE)
                cls.EXPERIMENT_NAME = best_params.get("EXPERIMENT_NAME", cls.EXPERIMENT_NAME)
                
                print(f"✅ [Config] tune.py의 최적 파라미터를 자동 적용했습니다: LR={cls.LEARNING_RATE:.6f}, WD={cls.WEIGHT_DECAY:.6f}, EXP_NAME={cls.EXPERIMENT_NAME}")
            except Exception as e:
                print(f"⚠️ [Config] 최적 파라미터 파일 읽기 실패. 기본값을 사용합니다. 에러: {e}")
        else:
            print(f"ℹ️ [Config] 최적 파라미터 기록이 발견되지 않았습니다. 기본 설정값으로 진행합니다.")

    @staticmethod
    def get_summary():
        """설정된 현재 파이프라인의 주요 수치를 터미널에 요약 출력합니다."""
        print("-" * 50)
        print(f"PROJECT: {Config.EXPERIMENT_NAME}")
        print(f"HARDWARE: {Config.DEVICE} (AMP: {Config.USE_AMP})")
        print(f"HYPERPARAMS: LR={Config.LEARNING_RATE:.6f}, BATCH={Config.BATCH_SIZE}, WD={Config.WEIGHT_DECAY:.6f}")
        print(f"DATA PATH: {Config.TRAIN_PATH}")
        print("-" * 50)

# ---------------------------------------------------------
# 모듈 로드 시점에 '최적 파라미터'를 자동으로 확인하고 반영합니다.
# 이 코드가 파일 하단에 위치함으로써 다른 파일에서 import만 해도 자동 업데이트가 일어납니다.
# ---------------------------------------------------------
Config.load_best_params()

if __name__ == "__main__":
    # 파일 단독 실행 시 현재 설정 상태를 출력하여 확인 용도로 사용합니다.
    Config.get_summary()
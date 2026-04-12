"""
=============================================================================================
[Hyperparameter Tuning Module] CLiST Optuna Optimization
=============================================================================================

1. 기능 및 목적 (Overview)
본 모듈은 Optuna 프레임워크를 활용하여 CLiST 모델의 최적 하이퍼파라미터
(Learning Rate, Weight Decay 등)를 자동으로 탐색합니다.
탐색 시간을 획기적으로 단축하기 위해 전체 데이터의 일부(예: 10%)만 샘플링하여 학습하며,
가망이 없는 파라미터 조합은 초반에 강제로 종료시키는 '가지치기(Pruning)' 기법이 적용되어 있습니다.

2. train.py와의 차이점
  - [데이터 경량화]: SubsetRandomSampler를 사용하여 Config.TUNE_DATA_RATIO 비율만큼만 학습합니다.
  - [기록 최소화]: 수십 번의 실험마다 Confusion Matrix나 그래프를 그리면 디스크가 낭비되므로,
                  오직 '검증 점수(Macro F1)'만 추적하여 최적의 파라미터 조합만 텍스트로 반환합니다.
  - [Pruning]: 매 에포크마다 Optuna에 점수를 보고하여, 가망이 없으면 TrialPruned 예외를 발생시킵니다.
=============================================================================================
"""

import optuna
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, SubsetRandomSampler
from sklearn.metrics import f1_score
import numpy as np
from tqdm import tqdm
import json
import mlflow
from optuna.integration.mlflow import MLflowCallback

# 내부 모듈 연동
from config import Config
from dataset import CLiSTDataset
from model import CLiST

def tune_and_evaluate(model, train_loader, val_loader, trial):
    """
    [튜닝 전용 학습 루프] 
    본 학습(train.py)과 달리 파일 저장이나 시각화를 생략하고, Optuna의 가지치기(Pruning)를 지원합니다.
    """
    # 다중 GPU 환경 지원 (고속 탐색)
    # if torch.cuda.device_count() > 1:
    #     model = nn.DataParallel(model)
    model = model.to(Config.DEVICE)
    
    # 클래스 불균형 가중치 및 Loss 함수 설정
    class_weights = torch.tensor([1.0, 2.0, 5.0, 10.0]).to(Config.DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # Optuna가 제안한 파라미터가 적용된 옵티마이저 (학습률, 가중치 감쇠, 옵티마이저 종류 선택)
    lr = trial.suggest_float("lr", Config.OPTUNA_LR_RANGE[0], Config.OPTUNA_LR_RANGE[1], log=True)
    
    weight_decay = trial.suggest_float("weight_decay", Config.OPTUNA_WD_RANGE[0], Config.OPTUNA_WD_RANGE[1], log=True)
    
    optimizer_name = trial.suggest_categorical("optimizer", ["AdamW", "Adam", "RMSprop"])
    # 선택된 옵티마이저에 따라 분기 처리
    if optimizer_name == "AdamW":
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        optimizer = optim.RMSprop(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # AMP(Mixed Precision) 스케일러 적용 (학습 가속)
    scaler = torch.amp.GradScaler('cuda', enabled=Config.USE_AMP)

    best_macro_f1 = 0.0
    
    # 튜닝 시에는 전체 에포크를 다 돌지 않고, 초반 잠재력만 확인하기 위해 짧게(예: 5 Epoch) 돕니다.
    tune_epochs = min(5, Config.EPOCHS) 

    for epoch in range(1, tune_epochs + 1):
        # -------------------------
        # 1. 학습 (Train Phase)
        # -------------------------
        model.train()
        for sensor_x, vision_x, labels in tqdm(train_loader, desc=f"Trial {trial.number} | Epoch {epoch}/{tune_epochs}", leave=False):
            if torch.isnan(sensor_x).any() or torch.isnan(vision_x).any():
                continue 
            
            sensor_x, vision_x, labels = sensor_x.to(Config.DEVICE), vision_x.to(Config.DEVICE), labels.to(Config.DEVICE)
            optimizer.zero_grad()
            
            with torch.amp.autocast('cuda', enabled=Config.USE_AMP):
                outputs = model(sensor_x, vision_x)
                loss = criterion(outputs, labels)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
        # -------------------------
        # 2. 검증 (Validation Phase)
        # -------------------------
        model.eval()
        all_preds, all_labels = [], []
        
        with torch.no_grad():
            for sensor_x, vision_x, labels in val_loader:
                sensor_x, vision_x, labels = sensor_x.to(Config.DEVICE), vision_x.to(Config.DEVICE), labels.to(Config.DEVICE)
                
                with torch.amp.autocast('cuda', enabled=Config.USE_AMP):
                    outputs = model(sensor_x, vision_x)
                    
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
        macro_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
        
        if macro_f1 > best_macro_f1:
            best_macro_f1 = macro_f1
            
        # -------------------------
        # 3. Optuna Pruning (가지치기)
        # -------------------------
        # 매 에포크의 결과를 Optuna에 보고합니다.
        trial.report(macro_f1, epoch)
        
        # 만약 다른 실험들에 비해 현재 점수가 너무 형편없다면, 남은 에포크를 포기하고 강제 종료합니다.
        if trial.should_prune():
            print(f"✂️ Trial {trial.number} pruned at epoch {epoch} (Macro F1: {macro_f1:.4f})")
            raise optuna.exceptions.TrialPruned()

    return best_macro_f1


def objective(trial):
    """
    [Optuna 목적 함수]
    Optuna 스터디(Study)가 1번의 실험(Trial)을 수행할 때마다 호출되는 메인 함수입니다.
    """

    with mlflow.start_run(run_name=f"optuna_run_{trial.number}"):

        print(f"\n🔍 [Trial {trial.number}] 새로운 하이퍼파라미터 탐색 시작...")

        try:
        
            # 1. 하이퍼파라미터 탐색 : 모델 구조 (Dropout, Hidden Dim)
            dropout_rate = trial.suggest_float("dropout_rate", 0.3, 0.5)
            # MLP 차원은 보통 2의 거수제곱을 사용하므로 categorical로 제안하는 것이 안전
            hidden_dim = trial.suggest_categorical("hidden_dim", [32, 64, 128])

            # 2. 전체 데이터셋 로드
            train_dataset = CLiSTDataset(Config.TRAIN_PATH)
            val_dataset = CLiSTDataset(Config.VAL_PATH)
            
            # 3. 고속 탐색을 위한 데이터 샘플링 (SubsetRandomSampler 사용)
            # 전체 데이터의 TUNE_DATA_RATIO (예: 10%) 분량만 무작위로 추출할 인덱스 생성
            num_train = len(train_dataset)
            indices = list(range(num_train))
            np.random.shuffle(indices)
            split = int(np.floor(Config.TUNE_DATA_RATIO * num_train))
            train_idx = indices[:split]
            
            train_sampler = SubsetRandomSampler(train_idx)
            
            # 4. 튜닝용 데이터 로더 생성 (샘플러 적용)
            train_loader = DataLoader(
                train_dataset, batch_size=Config.BATCH_SIZE, sampler=train_sampler, 
                num_workers=Config.NUM_WORKERS, pin_memory=Config.PIN_MEMORY
            )
            # 검증은 과적합 여부를 정확히 판단하기 위해 샘플링 없이 전체 검증 데이터를 사용
            val_loader = DataLoader(
                val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, 
                num_workers=Config.NUM_WORKERS, pin_memory=Config.PIN_MEMORY
            )
            
            # 5. 모델 초기화 및 평가 수행
            # 찾아낸 파라미터를 적용하여 모델 인스턴스 생성
            model = CLiST(num_classes=Config.NUM_CLASSES, hidden_dim=hidden_dim, dropout_rate=dropout_rate)
            best_f1 = tune_and_evaluate(model, train_loader, val_loader, trial)

            # 모델 평가가 끝나면, 사용된 파라미터와 점수를 직접 기록
            mlflow.log_params(trial.params)
            mlflow.log_metric("macro_f1", best_f1)
            
            return best_f1

        except optuna.exceptions.TrialPruned as e:
                # 가지치기(Pruning) 당한 경우에도 파라미터는 기록
                mlflow.log_params(trial.params)
                mlflow.set_tag("status", "pruned")
                raise e  # Optuna에게 가지치기 당했다고 다시 알려줌


if __name__ == "__main__":
    user_input = input(f"📝 기록할 실험 이름을 입력하세요 (엔터 시 기본값 '{Config.EXPERIMENT_NAME}' 사용): ")
    current_exp_name = user_input.strip() if user_input.strip() else Config.EXPERIMENT_NAME

    print(f"🚀 CLiST 모델 Optuna 하이퍼파라미터 튜닝을 시작합니다.")
    
    # 폴더 대신 SQLite DB를 사용하도록 MLflow 기본 설정 변경
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment(current_exp_name)
    
    # Info 메시지 방지: Optuna 스터디 이름도 실험 이름과 통일
    study = optuna.create_study(
        study_name=current_exp_name,  # 이름 강제 지정
        direction="maximize", 
        storage="sqlite:///optuna_history.db", # 탐색 기록을 저장할 로컬 DB 파일 생성
        load_if_exists=True,                   # 같은 이름의 스터디가 있으면 과거 기록 불러오기
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=2)
    )
    
    # 최적화 수행
    study.optimize(objective, n_trials=Config.OPTUNA_TRIALS)
    
    # 최종 결과 출력
    print("\n" + "="*60)
    print("🏆 [Optuna 탐색 완료] 최적의 하이퍼파라미터를 찾았습니다!")
    print("="*60)
    print(f"최고 검증 Macro F1 점수 : {study.best_value:.4f}")
    print(f"최적 학습률(LR)          : {study.best_params['lr']:.6f}")
    print(f"최적 가중치 감쇠(WD)     : {study.best_params['weight_decay']:.6f}")
    print("="*60)

    # 💡 4. JSON에 저장할 때, 방금 입력받은 실험 이름(current_exp_name)도 같이 묶어서 저장합니다!
    best_params = {
        "LEARNING_RATE": study.best_params['lr'],
        "WEIGHT_DECAY": study.best_params['weight_decay'],
        "OPTIMIZER": study.best_params['optimizer'],
        "HIDDEN_DIM": study.best_params['hidden_dim'],
        "DROPOUT_RATE": study.best_params['dropout_rate'],
        "EXPERIMENT_NAME": current_exp_name
    }
    
    param_file_path = Config.OUTPUT_DIR / "best_params.json"
    with open(param_file_path, "w", encoding="utf-8") as f:
        json.dump(best_params, f, indent=4)
        
    print(f"✅ 최적 파라미터와 실험 이름({current_exp_name})이 자동으로 저장되었습니다!")
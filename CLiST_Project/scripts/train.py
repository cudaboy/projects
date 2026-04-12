"""
=============================================================================================
[Orchestrator Module] CLiST Main Training Pipeline
=============================================================================================

1. 기능 및 목적 (Overview)
본 모듈은 CLiST 프로젝트의 전체 학습 프로세스를 지휘(Orchestration)하는 메인 실행 파일입니다.
설정(Config), 데이터(Dataset), 모델(Model), 부가 기능(Utils) 모듈을 모두 불러와 결합하며,
현업 MLOps 표준인 MLflow를 통해 모든 실험 과정과 결과물을 자동으로 기록합니다.

2. 적용된 핵심 최적화 기술 (Core Technologies)
  A. 데이터 로딩 병목(I/O Bottleneck) 해소를 통한 단일 GPU 학습 효율 최적화:
     - 분산 학습 시 발생하는 통신 오버헤드를 배제하고, 멀티 프로세싱(8 Workers) 기반의 데이터 프리페칭(Pre-fetching)을 통해
       고해상도 열화상 이미지 데이터의 공급 속도를 GPU 연산 속도에 동기화하였습니다.
  B. 혼합 정밀도 학습 (Native AMP):
     - PyTorch Native GradScaler를 활용하여 FP16 연산을 수행, VRAM을 절약하고 속도를 극대화합니다.
  C. MLOps 트래킹 (MLflow) & 스마트 로깅:
     - train_run_0, train_run_1 등으로 자동으로 이름을 넘버링하여 아카이빙합니다.
=============================================================================================
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, classification_report
from tqdm import tqdm
import mlflow
import os

# 모듈화된 프로젝트 내부 파일 로드
from config import Config
from dataset import CLiSTDataset
from model import CLiST
from utils import EarlyStopping, save_learning_curve, save_confusion_matrix


def main():
    """
    [스크립트 실행 진입점 및 핵심 학습 루프]
    메모리 누수 및 전역 변수 충돌을 방지하기 위해 전체 로직을 main()으로 감싸서 실행합니다.
    """
    
    # 환경 설정 서머리 출력
    Config.get_summary()
    print("🚀 데이터셋을 로딩하고 메모리에 올리는 중입니다...")
        
    # ---------------------------------------------------------
    # 1. 데이터셋 및 데이터 로더 준비
    # ---------------------------------------------------------
    train_dataset = CLiSTDataset(Config.TRAIN_PATH)
    val_dataset = CLiSTDataset(Config.VAL_PATH)

    train_loader = DataLoader(
        train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True, 
        num_workers=Config.NUM_WORKERS, pin_memory=Config.PIN_MEMORY
    )
    val_loader = DataLoader(
        val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, 
        num_workers=Config.NUM_WORKERS, pin_memory=Config.PIN_MEMORY
    )

    # ---------------------------------------------------------
    # 2. 모델, 손실 함수, 옵티마이저, 스케줄러 세팅
    # ---------------------------------------------------------
    # tune.py에서 찾아낸 동적 파라미터(Hidden Dim, Dropout) 적용하여 모델 초기화
    model = CLiST(num_classes=Config.NUM_CLASSES, hidden_dim=Config.HIDDEN_DIM, dropout_rate=Config.DROPOUT_RATE)
    
    # 💡 [핵심] 1-GPU 모드이므로 명시적으로 모델을 해당 장치(cuda)로 보냅니다.
    model = model.to(Config.DEVICE)
    
    # 불균형 데이터 해소를 위한 클래스별 가중치 적용 (정상:1, 관심:2, 경고:5, 위험:10)
    class_weights = torch.tensor([1.0, 2.0, 5.0, 10.0]).to(Config.DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # 설정 파일(Config)에서 가져온 파라미터 적용 (tune.py에서 찾은 최적값)
    if Config.OPTIMIZER == "AdamW":
        optimizer = optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY)
    elif Config.OPTIMIZER == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY)
    else:
        optimizer = optim.RMSprop(model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY)
        
    # 성능이 정체될 때 학습률을 절반으로 줄여 미세 조정 유도
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=Config.SCHEDULER_PATIENCE
    )
    
    # 조기 종료 객체 및 혼합 정밀도(AMP) 스케일러 초기화
    early_stopping = EarlyStopping(patience=Config.EARLY_STOPPING_PATIENCE)
    scaler = torch.amp.GradScaler('cuda', enabled=Config.USE_AMP)

    # ---------------------------------------------------------
    # 3. 자동 네이밍 및 MLflow 안전 설정
    # ---------------------------------------------------------
    # 단일 프로세스이므로 DB Lock 충돌 걱정 없이 안전하게 방을 만듭니다.
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment(Config.EXPERIMENT_NAME)
    
    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name(Config.EXPERIMENT_NAME)
    
    run_idx = 0
    if experiment is not None:
        runs = client.search_runs(experiment_ids=[experiment.experiment_id])
        train_run_numbers = []
        for r in runs:
            name = r.data.tags.get("mlflow.runName", "")
            if name.startswith("train_run_"):
                try:
                    num = int(name.split("_")[-1])
                    train_run_numbers.append(num)
                except ValueError:
                    pass
        if train_run_numbers:
            run_idx = max(train_run_numbers) + 1
            
    current_run_name = f"train_run_{run_idx}"
    
    # MLflow 실행 및 하이퍼파라미터 로깅
    mlflow.start_run(run_name=current_run_name)
    mlflow.log_params({
        "learning_rate": Config.LEARNING_RATE, "weight_decay": Config.WEIGHT_DECAY,
        "optimizer": Config.OPTIMIZER, "hidden_dim": Config.HIDDEN_DIM,
        "dropout_rate": Config.DROPOUT_RATE, "batch_size": Config.BATCH_SIZE,
        "epochs": Config.EPOCHS, "use_amp": Config.USE_AMP, "num_workers": Config.NUM_WORKERS
    })
    
    print("🔥 본격적인 모델 학습을 시작합니다...\n")

    # 최고 성능 기록용 변수 및 시각화를 위한 히스토리 딕셔너리
    best_macro_f1 = 0.0
    history = {'train_loss': [], 'val_loss': [], 'val_f1': []}

    # ---------------------------------------------------------
    # 4. 에포크(Epoch) 반복 루프
    # ---------------------------------------------------------
    for epoch in range(1, Config.EPOCHS + 1):
        
        # [A. Training Phase]
        model.train()
        train_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch [{epoch}/{Config.EPOCHS}] Train", leave=False)

        for sensor_x, vision_x, labels in pbar:
            # 결측치 텐서 필터링 (방어 코드)
            if torch.isnan(sensor_x).any() or torch.isnan(vision_x).any():
                continue 
            
            # 💡 [핵심] 1-GPU 모드이므로 데이터를 직접 GPU 메모리로 올려줍니다.
            sensor_x = sensor_x.to(Config.DEVICE)
            vision_x = vision_x.to(Config.DEVICE)
            labels = labels.to(Config.DEVICE)
            
            optimizer.zero_grad()
            
            # PyTorch Native AMP 적용 (가속 및 메모리 절약)
            with torch.amp.autocast('cuda', enabled=Config.USE_AMP):
                outputs = model(sensor_x, vision_x)
                loss = criterion(outputs, labels)
            
            # Scaler를 통한 안전한 역전파 및 가중치 업데이트
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            # 진행 바 옆에 현재 배치의 Loss를 실시간으로 띄워줍니다!
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
            
            train_loss += loss.item()
            
        # [B. Validation Phase]
        model.eval()
        val_loss = 0.0
        all_preds, all_labels = [], []
        
        with torch.no_grad():
            for sensor_x, vision_x, labels in val_loader:
                sensor_x = sensor_x.to(Config.DEVICE)
                vision_x = vision_x.to(Config.DEVICE)
                labels = labels.to(Config.DEVICE)
                
                with torch.amp.autocast('cuda', enabled=Config.USE_AMP):
                    outputs = model(sensor_x, vision_x)
                    loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
        # [C. Metrics Calculation]
        macro_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        # [D. 메인 작업] 프린트, 모델 저장, MLflow 로깅
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['val_f1'].append(macro_f1)
        
        print(f"Epoch {epoch:02d} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Macro F1: {macro_f1:.4f}")
        
        if mlflow.active_run():
            mlflow.log_metric("train_loss", avg_train_loss, step=epoch)
            mlflow.log_metric("val_loss", avg_val_loss, step=epoch)
            mlflow.log_metric("val_macro_f1", macro_f1, step=epoch)
            mlflow.log_metric("learning_rate", optimizer.param_groups[0]['lr'], step=epoch)

        # [E. 전략 업데이트 및 조기 종료]
        scheduler.step(macro_f1)
        
        if macro_f1 > best_macro_f1:
            best_macro_f1 = macro_f1
            # 💡 [핵심] 순수 PyTorch 모델이므로 unwrap 같은 번거로운 과정 없이 다이렉트로 저장!
            torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)
            print(f"   ✨ 최고 성능 갱신! 모델 가중치가 저장되었습니다. (Macro F1: {best_macro_f1:.4f})")
            
        early_stopping(macro_f1)
        if early_stopping.early_stop:
            print(f"🛑 과적합 감지: Epoch {epoch}에서 조기 종료(Early Stopping) 되었습니다!")
            break

    # ---------------------------------------------------------
    # 5. 최종 결과물 산출 및 로깅 종료 (Post-Processing)
    # ---------------------------------------------------------
    print("\n📊 학습 완료! 최종 평가 리포트 및 시각화 이미지를 생성합니다.")
    
    save_learning_curve(history['train_loss'], history['val_loss'], history['val_f1'])
    save_confusion_matrix(all_labels, all_preds)
    
    report_text = classification_report(
        all_labels, all_preds, 
        target_names=['Normal', 'Attention', 'Warning', 'Danger'], 
        zero_division=0
    )
    
    with open(Config.REPORT_SAVE_PATH, 'w', encoding='utf-8') as f:
        f.write("="*60 + "\nCLiST 모델 최종 검증 리포트\n" + "="*60 + "\n")
        f.write(report_text)
        
    print(report_text)
        
    if mlflow.active_run():
        mlflow.log_artifact(str(Config.OUTPUT_DIR / "learning_curve.png"))
        mlflow.log_artifact(str(Config.OUTPUT_DIR / "confusion_matrix.png"))
        mlflow.log_artifact(str(Config.REPORT_SAVE_PATH))
        mlflow.log_artifact(str(Config.MODEL_SAVE_PATH))
        mlflow.end_run() 
        
    print(f"\n🎉 모든 파이프라인이 종료되었습니다.")
    print(f"터미널에 'mlflow ui --backend-store-uri sqlite:///mlflow.db --host 0.0.0.0 --port 5000'를 입력하여 웹 대시보드를 확인하세요!")

# ==============================================================================
# 🚀 스크립트 실행
# ==============================================================================
if __name__ == "__main__":
    main()

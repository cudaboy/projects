"""
=============================================================================================
[Utility Module] CLiST Helper Functions & Visualization
=============================================================================================

1. 기능 및 목적 (Overview)
본 모듈은 모델 학습 과정에서 필요한 부가적인 기능(조기 종료)과 
학습 완료 후 결과를 분석하기 위한 시각화(Learning Curve, Confusion Matrix) 도구들을 제공합니다.
이러한 헬퍼(Helper) 함수들을 분리함으로써 메인 스크립트(`train.py`)가 
오직 '학습 플로우'에만 집중할 수 있도록 관심사를 분리(Separation of Concerns)합니다.

2. 주요 클래스 및 함수 구성
  A. EarlyStopping (Class):
     - 모델의 검증 성능(Macro F1)을 매 에포크마다 추적합니다.
     - 설정된 횟수(Patience) 동안 성능 개선이 없으면 과적합(Overfitting)으로 간주하고
       학습을 강제 종료시켜 클라우드 컴퓨팅 비용(GPU 타임)을 절약합니다.

  B. save_learning_curve (Function):
     - Train Loss, Validation Loss, Validation F1-Score의 변화 추이를 
       하나의 꺾은선 그래프(Dual-axis)로 시각화하여 저장합니다.
       
  C. save_confusion_matrix (Function):
     - 모델의 예측값과 실제 정답의 교차표를 히트맵(Heatmap) 형태로 시각화합니다.
     - 소수 클래스(위험, 경고)에 대한 모델의 오탐지(False Positive/Negative) 성향을 
       직관적으로 분석할 수 있게 돕습니다.
=============================================================================================
"""

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import os
from config import Config  # 시각화 이미지 저장 경로를 불러오기 위해 연동

class EarlyStopping:
    """
    [조기 종료 제어기]
    검증 데이터셋의 성능 지표(Macro F1)가 일정 기간(Patience) 동안 개선되지 않으면
    학습 루프에 중단 신호(early_stop = True)를 보냅니다.
    """
    def __init__(self, patience=Config.EARLY_STOPPING_PATIENCE, delta=0.001):
        """
        초기화 함수
        Args:
            patience (int): 성능이 개선되지 않아도 몇 에포크를 더 참을 것인지 설정 (기본값: config.py의 5)
            delta (float): '개선되었다'고 인정할 최소한의 점수 변화폭 (노이즈 방지용)
        """
        self.patience = patience
        self.delta = delta
        self.counter = 0              # 인내심 카운터 (개선 안될 때마다 1씩 증가)
        self.best_score = None        # 지금까지 관측된 최고 성능 점수
        self.early_stop = False       # 학습 중단 여부를 나타내는 플래그 (True면 중단)

    def __call__(self, val_macro_f1):
        """
        매 에포크가 끝날 때마다 호출되어 성능을 평가합니다.
        Args:
            val_macro_f1 (float): 현재 에포크의 검증 데이터 Macro F1 점수 (높을수록 좋음)
        """
        # 첫 번째 에포크인 경우 (기준점이 없으므로 현재 점수를 최고 점수로 등록)
        if self.best_score is None:
            self.best_score = val_macro_f1
            
        # 성능이 이전 최고 점수 + 최소 요구치(delta)보다 낮거나 같은 경우 (개선 실패)
        elif val_macro_f1 < self.best_score + self.delta:
            self.counter += 1
            print(f"⚠️ [EarlyStopping] 성능 개선 정체. 카운트: {self.counter} / {self.patience}")
            # 설정한 인내심 한계에 도달하면 조기 종료 플래그 활성화
            if self.counter >= self.patience:
                self.early_stop = True
                
        # 성능이 확실하게 개선된 경우 (기록 갱신 및 카운터 초기화)
        else:
            self.best_score = val_macro_f1
            self.counter = 0


def save_learning_curve(train_losses, val_losses, val_f1s, filename="learning_curve.png"):
    """
    [학습 곡선 시각화]
    모델이 안정적으로 수렴(Convergence)하고 있는지, 과적합이 발생하지 않았는지 
    증명하기 위한 듀얼 Y축 그래프를 그려 이미지로 저장합니다.
    
    Args:
        train_losses (list): 매 에포크의 Train Loss 리스트
        val_losses (list): 매 에포크의 Validation Loss 리스트
        val_f1s (list): 매 에포크의 Validation Macro F1 점수 리스트
        filename (str): 저장할 파일명
    """
    epochs = range(1, len(train_losses) + 1)
    
    # 가로 10, 세로 5 비율의 도화지 생성
    fig, ax1 = plt.subplots(figsize=(10, 5))
    
    # [왼쪽 Y축] Loss 곡선 그리기 (Train은 원, Validation은 사각형 마커 사용)
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss', color='tab:red')
    ax1.plot(epochs, train_losses, label='Train Loss', color='tab:red', marker='o')
    ax1.plot(epochs, val_losses, label='Val Loss', color='tab:orange', marker='s')
    ax1.tick_params(axis='y', labelcolor='tab:red')
    
    # [오른쪽 Y축] 동일한 X축을 공유하는 두 번째 Y축 생성 (F1 Score 용도)
    ax2 = ax1.twinx()
    ax2.set_ylabel('Macro F1 Score', color='tab:blue')
    ax2.plot(epochs, val_f1s, label='Val Macro F1', color='tab:blue', marker='^', linestyle='--')
    ax2.tick_params(axis='y', labelcolor='tab:blue')
    
    # 레이아웃 정리 및 제목 설정
    fig.tight_layout()
    plt.title("CLiST Multimodal Model - Training Curve")
    plt.grid(True, alpha=0.3)  # 배경 모눈종이 연하게 추가
    
    # Config.OUTPUT_DIR (예: /workspace/work4/outputs) 경로에 파일 저장
    save_path = Config.OUTPUT_DIR / filename
    plt.savefig(save_path, dpi=300, bbox_inches='tight')  # dpi=300으로 고화질 저장
    plt.close()  # 메모리 누수 방지를 위해 도화지 닫기
    print(f"📊 학습 곡선이 저장되었습니다: {save_path}")


def save_confusion_matrix(y_true, y_pred, classes=['Normal', 'Attention', 'Warning', 'Danger'], filename="confusion_matrix.png"):
    """
    [혼동 행렬 시각화]
    실제 라벨과 모델의 예측값을 비교하여, 어떤 클래스에서 주로 오답을 냈는지 
    직관적으로 파악할 수 있는 히트맵(Heatmap)을 저장합니다.
    
    Args:
        y_true (list or ndarray): 실제 정답 라벨
        y_pred (list or ndarray): 모델이 예측한 라벨
        classes (list): 시각화 텍스트로 사용할 클래스 이름 리스트
        filename (str): 저장할 파일명
    """
    # 1. Scikit-learn을 이용하여 Confusion Matrix 수치 데이터 생성
    cm = confusion_matrix(y_true, y_pred)
    
    # 2. Seaborn을 이용하여 예쁜 색상의 히트맵 그리기
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes,
                cbar_kws={'label': 'Number of Samples'})
    
    # 3. 축 라벨 및 타이틀 설정
    plt.title('CLiST Confusion Matrix')
    plt.ylabel('Actual Label (Ground Truth)')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    # Config.OUTPUT_DIR 경로에 파일 저장
    save_path = Config.OUTPUT_DIR / filename
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"📊 혼동 행렬이 저장되었습니다: {save_path}")
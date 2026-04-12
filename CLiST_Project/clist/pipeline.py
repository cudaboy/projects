"""
=============================================================================================
[Inference Pipeline Module] CLiST Integrated API Wrapper
=============================================================================================

1. 개요 및 목적 (Overview)
본 모듈은 CLiST(Multimodal Time-Frequency Fusion Network) 모델의 실시간 추론(Inference)을 
담당하는 '통합 파이프라인 래퍼(Wrapper)' 클래스를 정의합니다.
데이터 엔지니어나 외부 시스템(FastAPI 서버, 공장 제어 대시보드 등) 개발자가 
복잡한 PyTorch 로직이나 전처리 과정을 알 필요 없이, 오직 파일 경로만 전달하여 
즉각적인 위험도 판정 결과를 얻을 수 있도록 사용자 친화적인 인터페이스를 제공합니다.

2. 주요 역할 및 로직 (Core Responsibilities)
  A. 원터치 초기화 (Auto Initialization):
     - 모델 아키텍처 로드, 학습된 가중치(.pth) 주입, GPU/CPU 환경 자동 할당을 한 번에 수행합니다.
     - 학습 시 사용된 글로벌 통계량(`domain_stats.json`)을 자동으로 불러와 추론의 정확성을 보장합니다.

  B. 실시간 멀티모달 전처리 (Real-time Preprocessing):
     - `_preprocess_sensor`: 1D 센서 CSV 파일의 결측치를 방어적으로 보간(ffill, bfill)하고 
       Z-Score 정규화를 적용하여 (Batch=1, Seq=1, Features=24) 텐서로 변환합니다.
     - `_preprocess_vision`: 2D 열화상 BIN 파일을 읽어 Min-Max 스케일링을 수행하고, 
       Swin Transformer 입력 규격에 맞춰 3채널 복제 및 224x224 리사이즈를 적용합니다.

  C. 직관적인 결과 포맷팅 (User-Friendly Output):
     - 텐서 형태의 모델 출력값(Logits)을 Softmax 함수를 통해 확률(%)로 변환합니다.
     - 최종 예측 클래스와 신뢰도(Confidence), 전체 클래스별 확률 분포를 
       JSON 형태(Dictionary)로 깔끔하게 포장하여 반환합니다.

3. 사용 예시 (Usage Example)
    pipeline = CLiSTPipeline(weight_path='weights/best_model.pth')
    result = pipeline.predict(sensor_csv_path='sensor.csv', vision_bin_path='vision.bin')
    print(result['predicted_status']) # "Danger(위험)"
=============================================================================================
"""

import torch
import pandas as pd
import numpy as np
import json
from torchvision import transforms
from pathlib import Path

# 기존 구현된 모듈 로드
from .model import CLiST
from .config import Config

class CLiSTPipeline:
    """
    사용자 친화적인 CLiST 모델 통합 API (Wrapper Class)
    사용자는 이 클래스 하나만 선언하면 복잡한 전처리 및 추론을 한 번에 수행할 수 있습니다.
    """
    def __init__(self, weight_path='weights/best_clist_model.pth', stats_path='weights/domain_stats.json'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.classes = ['Normal(정상)', 'Attention(관심)', 'Warning(경고)', 'Danger(위험)']
        
        # 1. 모델 아키텍처 로드 및 가중치 삽입
        self.model = CLiST(num_classes=Config.NUM_CLASSES, hidden_dim=Config.HIDDEN_DIM, dropout_rate=0.0)
        self.model.load_state_dict(torch.load(weight_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval() # 추론 모드로 전환 (Dropout, BatchNorm 비활성화)
        
        # 2. 전처리에 필수적인 학습 데이터 통계량(Mean, Std) 로드
        with open(stats_path, 'r') as f:
            self.stats = json.load(f)
            
        # 3. 비전 데이터 리사이즈 도구
        self.resize_op = transforms.Resize((224, 224), antialias=True)
        print("✅ CLiST Pipeline이 성공적으로 로드되었습니다.")

    def _preprocess_sensor(self, csv_path):
        """1개의 CSV 파일을 읽어 1D 텐서로 변환"""
        df = pd.read_csv(csv_path).interpolate().fillna(0)
        
        # Z-Score 정규화
        for col, s in self.stats.items():
            if col in df.columns:
                df[col] = (df[col] - s['mean']) / (s['std'] + 1e-8)
                
        # 롤링 윈도우 특징 추출 (주의: 입력 CSV는 최소 5행 이상이어야 함)
        feat = df.select_dtypes(include=[np.number]).rolling(Config.WINDOW_SIZE, min_periods=1).agg(['mean', 'var', 'kurt'])
        final_feat = feat.fillna(0).values[-1:] 
        
        # (Batch=1, Seq=1, Features=24)
        return torch.tensor(final_feat, dtype=torch.float32).unsqueeze(0).to(self.device)

    def _preprocess_vision(self, bin_path):
        """1개의 BIN 파일을 읽어 2D 텐서로 변환"""
        img = np.load(bin_path).astype(np.float32)
        img = (img - img.min()) / (img.max() - img.min() + 1e-8) # Min-Max
        
        tensor_img = torch.tensor(img).unsqueeze(0).repeat(3, 1, 1) # 3채널 복제
        tensor_img = self.resize_op(tensor_img)
        
        # (Batch=1, Channels=3, H=224, W=224)
        return tensor_img.unsqueeze(0).to(self.device)

    def predict(self, sensor_csv_path, vision_bin_path):
        """
        [핵심 추론 함수] 파일 경로 2개를 받아 위험도를 예측합니다.
        """
        # 1. 개별 전처리 수행
        sensor_tensor = self._preprocess_sensor(sensor_csv_path)
        vision_tensor = self._preprocess_vision(vision_bin_path)
        
        # 2. 모델 추론
        with torch.no_grad():
            logits = self.model(sensor_tensor, vision_tensor)
            probabilities = torch.softmax(logits, dim=1).squeeze().cpu().numpy()
            pred_class_idx = np.argmax(probabilities)
            
        # 3. 결과 반환 (딕셔너리 형태)
        result = {
            "predicted_status": self.classes[pred_class_idx],
            "confidence": float(probabilities[pred_class_idx]),
            "all_probabilities": {self.classes[i]: float(probabilities[i]) for i in range(4)}
        }
        return result

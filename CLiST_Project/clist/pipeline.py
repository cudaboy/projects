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
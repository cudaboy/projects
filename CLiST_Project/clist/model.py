"""
=============================================================================================
[Model Architecture Module] CLiST: Multimodal Time-Frequency Fusion Network
=============================================================================================

1. 기능 및 목적 (Overview)
본 모듈은 스마트 팩토리 이송장치(AGV/OHT)의 실시간 탄화 위험도를 예측하는 
멀티모달 융합 딥러닝 아키텍처(CLiST)를 정의합니다.
데이터 파이프라인(`dataset.py`)에서 전처리된 1D 센서 특징과 2D 비전 특징을 
각각 독립적으로 인코딩한 후, '지연 융합(Late Fusion)' 방식으로 결합하여 
최종 분류(`config.py`의 NUM_CLASSES)를 수행합니다.

2. 데이터 흐름도 (Data Flow Architecture)
  [dataset.py]                                [model.py]                                   [Output]
  1D (Batch, 1, 24)       -> LightTSEncoder  -> 1D Vector (Batch, 64)   -\
                                                                          => Concat(192) => Logits (Batch, 4)
  2D (Batch, 3, 224, 224) -> LiteSwinEncoder -> 2D Vector (Batch, 128)  -/
=============================================================================================
"""

import torch
import torch.nn as nn
from torchvision.models import swin_t, Swin_T_Weights

# 중앙 집중식 환경 설정 파일에서 분류 클래스 개수 로드
from config import Config  

class LightTSEncoder(nn.Module):
    """
    [1D Stream] LightTS Encoder
    
    센서 데이터(온도, 진동, 전류 등)의 시계열적 불안정성을 파악하는 인코더입니다.
    무거운 RNN/LSTM을 배제하고, dataset.py에서 미리 추출한 통계적 피처(Mean, Var, Kurt)를 
    가벼운 MLP(Multi-Layer Perceptron) 네트워크로 분석하여 초고속 추론(Low Latency)을 달성합니다.
    """
    def __init__(self, input_dim=24, hidden_dim=64, dropout_rate=0.3):
        super().__init__()
        # input_dim=24: 8개 센서 변수 * 3개 통계량 (Rolling Mean, Var, Kurt)
        
        self.mlp = nn.Sequential(
            # [Layer 1: 차원 확장 및 비선형성 학습]
            # 24차원의 입력 변수들 사이에 숨겨진 복잡한 상호작용(예: 온도가 높고 진동이 클 때)을 128차원으로 펼쳐서 포착
            nn.Linear(input_dim, hidden_dim * 2),
            
            # 배치 정규화(Batch Normalization): 
            # 센서 데이터의 스케일 차이로 인한 Internal Covariate Shift(내부 공변량 이동) 방지 및 학습 가속화
            nn.BatchNorm1d(hidden_dim * 2),
            
            # GELU (Gaussian Error Linear Unit): 
            # 기존 ReLU 모델이 0 이하의 값을 완전히 죽여버리는(Dying ReLU) 한계를 극복한 최신 활성화 함수
            nn.GELU(),
            
            # Dropout: 
            # 특정 센서 값 하나에 모델이 과도하게 의존하는 과적합(Overfitting)을 방지 (30% 무작위 비활성화)
            nn.Dropout(dropout_rate),
            
            # [Layer 2: 특징 벡터 압축]
            # 128차원으로 펼쳐진 정보 중 노이즈를 제거하고 핵심 정보만 64차원 벡터로 요약 압축
            nn.Linear(hidden_dim * 2, hidden_dim)
        )

    def forward(self, x):
        """
        Args:
            x (Tensor): dataset.py의 _process_1d_lightts 반환값. 
                        Shape: (Batch, Sequence=1, Features=24)
        Returns:
            Tensor: 압축된 1D 센서 지식 벡터. 
                    Shape: (Batch, 64)
        """
        # (Batch, 1, 24) -> (Batch, 24) 크기로 텐서 평탄화(Flatten)
        # nn.Linear 레이어는 2D 텐서 입력을 기대하므로 불필요한 Sequence 차원(1)을 제거합니다.
        x = x.view(x.size(0), -1) 
        
        return self.mlp(x)


class LiteSwinEncoder(nn.Module):
    """
    [2D Stream] LiteSwin Encoder
    
    열화상 이미지의 공간적 온도 분포와 특정 부품의 국소 발열(Hot Spot) 패턴을 파악하는 인코더입니다.
    CNN 계열이 아닌 계층적 Vision Transformer(Swin-T)를 사용하여, 
    화면 전체의 온도 균형과 특정 부품의 국소 온도를 동시에 Attention(집중)하여 분석합니다.
    """
    def __init__(self, embed_dim=128):
        super().__init__()
        # 전이 학습(Transfer Learning): 
        # 수백만 장의 ImageNet 데이터로 시각적 기본기(선, 모서리, 질감 인식 등)가 사전 학습된 가중치를 로드
        self.swin = swin_t(weights=Swin_T_Weights.DEFAULT)
        
        # 모델 커스터마이징 (Head Replacement):
        # 원본 Swin-T는 1000개의 사물을 분류하는 출력층(head)을 가지고 있습니다.
        # 이를 제거하고, 우리가 융합에 사용할 128차원 크기의 임베딩(Embedding) 벡터를 반환하도록 레이어를 교체합니다.
        in_features = self.swin.head.in_features
        self.swin.head = nn.Linear(in_features, embed_dim)

    def forward(self, x):
        """
        Args:
            x (Tensor): dataset.py의 _process_2d_liteswin 반환값. 
                        Shape: (Batch, Channels=3, Height=224, Width=224)
        Returns:
            Tensor: 공간적 열화 패턴이 압축된 2D 비전 지식 벡터. 
                    Shape: (Batch, 128)
        """
        return self.swin(x)


class CLiST(nn.Module):
    """
    [Fusion & Classifier] 최종 멀티모달 융합 모델
    
    1D 센서 지식(64차원)과 2D 열화상 지식(128차원)을 병합하여 시너지를 창출하고,
    설정 파일(config.py)에 정의된 최종 위험도 클래스 중 하나로 상태를 판정합니다.
    """
    def __init__(self, num_classes=Config.NUM_CLASSES, hidden_dim=64, dropout_rate=0.4):
        """
        Args:
            num_classes (int): config.py에서 불러온 분류 대상의 개수 (기본값: 4)
        """
        super().__init__()
        
        # 1. 독립적인 특징 추출기(Encoder) 객체 생성
        self.sensor_encoder = LightTSEncoder(input_dim=24, hidden_dim=hidden_dim, dropout_rate=dropout_rate)
        self.vision_encoder = LiteSwinEncoder(embed_dim=128)
        
        # 2. 특징 결합 차원 계산 (ex: 64 + 128 = 192 차원)
        fusion_dim = hidden_dim + 128
        
        # 3. 최종 분류기 (Classifier Network)
        self.classifier = nn.Sequential(
            # [시너지 레이어] 두 모달리티 간의 교차 특성(Cross-modal feature)을 지정한 차원으로 결합
            nn.Linear(fusion_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            
            # [강력한 과적합 억제]
            # 위험/경고 클래스 데이터가 희소(Imbalanced)하므로 다수 클래스(정상)에 
            # 편향되는 것을 막기 위해 분류기 바로 직전에서 드롭아웃 적용
            nn.Dropout(dropout_rate),
            
            # [출력층] (ex: Batch, 64) -> (Batch, NUM_CLASSES)
            # Softmax 함수는 학습 시 nn.CrossEntropyLoss() 내부에 포함되므로 여기서는 순수 Logit 값을 반환
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, sensor_x, vision_x):
        """
        메인 순전파 (Main Forward Pass) - 데이터 로더의 배치가 입력으로 들어오는 최종 관문
        
        Args:
            sensor_x (Tensor): 1D 시계열 센서 미니배치 (Batch, 1, 24)
            vision_x (Tensor): 2D 열화상 비전 미니배치 (Batch, 3, 224, 224)
            
        Returns:
            out (Tensor): 최종 4단계 위험도별 예측 Logit 값 (Batch, 4)
        """
        # Step 1: Feature Extraction (모달리티별 독립적인 특징 추출)
        feat_1d = self.sensor_encoder(sensor_x)  # -> (Batch, 64)
        feat_2d = self.vision_encoder(vision_x)  # -> (Batch, 128)
        
        # Step 2: Late Fusion (지연 융합)
        # 피처 차원(dim=1)을 기준으로 두 벡터를 가로로 나란히 이어 붙임 (Concatenation)
        fused_features = torch.cat((feat_1d, feat_2d), dim=1)  # -> (Batch, 192)
        
        # Step 3: Classification (최종 판정)
        # 융합된 192차원의 멀티모달 피처를 기반으로 클래스 예측 수행
        out = self.classifier(fused_features)    # -> (Batch, 4)
        
        return out
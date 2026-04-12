"""
=============================================================================================
[Data Pipeline Module] CLiST Dataset & Preprocessing
=============================================================================================

1. 기능 및 목적 (Overview)
본 모듈은 저장된 원시 데이터(CSV, BIN, JSON)를 읽어와 CLiST 모델이 학습할 수 있는 
PyTorch Tensor 형태로 변환하는 '데이터 로더 및 전처리 전담 클래스'입니다.
데이터 전처리 로직을 분리함으로써 메인 학습 코드의 가독성을 극대화하고 재사용성을 높입니다.

2. 주요 전처리 파이프라인 (Processing Pipeline)
  A. 1D Sensor Data (_process_1d_lightts):
     - 결측치 보간(Interpolation) 및 글로벌 통계량 기반 Z-Score 정규화.
     - 시계열 롤링 윈도우(Rolling Window)를 적용해 평균, 분산, 첨도 특징을 추출.
  
  B. 2D Vision Data (_process_2d_liteswin):
     - 열화상 BIN 파일을 읽어 Min-Max 정규화 수행.
     - Swin Transformer의 입력 규격(224x224, 3 Channel)에 맞게 텐서 복제 및 리사이즈.
  
  C. Label Data (_process_label):
     - UTF-8-BOM 인코딩 문제 및 중첩된 JSON 구조(annotations -> tagging -> state)를 
       안전하게 파싱하여 4단계 위험도(0~3) 정답 라벨(Target) 추출.
=============================================================================================
"""

import torch
import json
import pandas as pd
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset
from torchvision import transforms
from config import Config  # 중앙 설정 모듈 연동

class CLiSTDataset(Dataset):
    """
    CLiST 멀티모달 학습을 위한 PyTorch 호환 커스텀 데이터셋 클래스입니다.
    """
    def __init__(self, split_dir, window_size=Config.WINDOW_SIZE):
        """
        초기화 함수: 캐시 파일과 통계량 파일을 메모리에 로드합니다.
        
        Args:
            split_dir (str or Path): 'Training' 또는 'Validation' 폴더 경로
            window_size (int): 시계열 통계 추출에 사용할 윈도우 크기 (config.py에서 상속)
        """
        self.split_dir = Path(split_dir)
        self.window_size = window_size
        
        # 1. 파일 경로가 매핑된 캐시 데이터 로드
        with open(self.split_dir / "file_pairs_cache.json", 'r') as f:
            self.samples = json.load(f)
            
        # 2. Z-Score 정규화를 위한 전체 도메인 통계량(평균, 표준편차) 로드
        with open(self.split_dir / "domain_stats.json", 'r') as f:
            self.stats = json.load(f)

    def _process_1d_lightts(self, csv_path):
        """
        [1D 시계열 전처리] CSV 파일을 읽어 24차원의 센서 특징 벡터를 추출합니다.
        
        Args: csv_path (str): CSV 파일 절대/상대 경로
        Returns: torch.Tensor shape (1, 24)
        """
        # 1. 결측치 보간 및 0 채우기
        df = pd.read_csv(csv_path).interpolate().fillna(0)
        
        # 2. 글로벌 통계량 기반 Z-Score 정규화 (스케일 통일)
        for col, s in self.stats.items():
            if col in df.columns:
                df[col] = (df[col] - s['mean']) / (s['std'] + 1e-8)
        
        # 3. 이동 윈도우(Rolling Window)를 이용해 기계의 트렌드(mean)와 불안정성(var, kurt) 추출
        feat = df.select_dtypes(include=[np.number]).rolling(self.window_size, min_periods=1).agg(['mean', 'var', 'kurt'])
        feat.columns = ['_'.join(c) for c in feat.columns]
        
        # 4. 시퀀스 전체가 아닌 '가장 최신 시점(마지막 행)'의 피처만 모델의 입력으로 사용
        final_feat = feat.fillna(0).values[-1:] 
        
        return torch.tensor(final_feat, dtype=torch.float32)

    def _process_2d_liteswin(self, bin_path):
        """
        [2D 비전 전처리] BIN 열화상 데이터를 Swin-T 모델 규격에 맞게 변환합니다.
        
        Args: bin_path (str): BIN 파일 절대/상대 경로
        Returns: torch.Tensor shape (3, 224, 224)
        """
        # 1. 넘파이 배열 로드 및 0~1 사이로 Min-Max 정규화
        img = np.load(bin_path).astype(np.float32)
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        
        # 2. 1채널 흑백 이미지를 3채널(RGB) 구조로 복제: (1, 120, 160) -> (3, 120, 160)
        tensor_img = torch.tensor(img).unsqueeze(0).repeat(3, 1, 1)
        
        # 3. Swin Transformer 패치 시스템이 요구하는 224x224 사이즈로 강제 리사이즈
        resize_op = transforms.Resize((224, 224), antialias=True)
        
        return resize_op(tensor_img)

    def _process_label(self, json_path):
        """
        [라벨 전처리] JSON 파일을 파싱하여 4단계 탄화 위험도(0~3)를 정수로 반환합니다.
        
        Args: json_path (str): JSON 파일 절대/상대 경로
        Returns: torch.Tensor shape () - 스칼라 값
        """
        try:
            # 윈도우 환경 생성 파일의 BOM 문자 충돌을 막기 위해 'utf-8-sig' 사용
            with open(json_path, 'r', encoding='utf-8-sig') as f:
                data = json.load(f)
            
            # 실제 데이터 구조에 맞춘 다중 계층 안전 탐색 (annotations -> tagging -> state)
            ann = data.get('annotations', [])
            if isinstance(ann, list) and len(ann) > 0:
                tagging = ann[0].get('tagging', [])
                if isinstance(tagging, list) and len(tagging) > 0:
                    state = tagging[0].get('state', 0)
                else:
                    state = 0
            else:
                state = 0
                
            return torch.tensor(int(state), dtype=torch.long)
            
        except Exception:
            # 파일 손상 시 방어적 처리 (기본값인 정상(0) 반환)
            return torch.tensor(0, dtype=torch.long)

    def __len__(self):
        """전체 데이터셋의 샘플 개수를 반환합니다."""
        return len(self.samples)

    def __getitem__(self, idx):
        """
        DataLoader가 배치를 구성할 때 호출되는 핵심 함수입니다.
        인덱스(idx)에 해당하는 1D 데이터, 2D 데이터, 라벨을 튜플로 반환합니다.
        """
        s = self.samples[idx]
        
        # 운영체제 간 파일 경로 호환성 보장 (Windows 역슬래시 '\'를 Linux 슬래시 '/'로 치환)
        csv_path = s['csv'].replace('\\', '/')
        bin_path = s['bin'].replace('\\', '/')
        json_path = s['json'].replace('\\', '/')
        
        # 💡 [디버깅용 로그 추가] 현재 어떤 파일을 읽으려 시도 중인지 터미널에 출력합니다.
        # print(f"🔍 [데이터 로드 시도] 인덱스: {idx} | 파일: {csv_path}")
        
        try:
            feat_1d = self._process_1d_lightts(csv_path)
            feat_2d = self._process_2d_liteswin(bin_path)
            label = self._process_label(json_path)
            
            # 💡 [디버깅용 로그 추가] 로드가 성공하면 완료 메시지 출력
            # print(f"✅ [데이터 로드 성공] 인덱스: {idx}")
            
            return feat_1d, feat_2d, label
        except Exception as e:
            # 💡 혹시라도 특정 파일이 깨져서 에러가 나면 숨기지 않고 터미널에 경고를 띄웁니다.
            print(f"❌ [데이터 로드 실패!] 파일: {csv_path} | 에러원인: {e}")
            # 에러가 난 경우 임시로 빈 텐서와 정상(0) 라벨을 반환하여 멈춤 방지
            return torch.zeros((1, 24)), torch.zeros((3, 224, 224)), torch.tensor(0)
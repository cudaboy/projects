import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ==========================================
# 1. 위치 인코딩 (Positional Encoding)
# ==========================================
class PositionalEncoding(nn.Module):
    """
    Transformer는 RNN이나 CNN처럼 단어의 순서 정보를 자체적으로 알 수 없으므로,
    입력 시퀀스의 각 위치(단어 순서)에 대한 고유한 정보를 인베딩 벡터에 더해주는 역할을 합니다.
    """
    def __init__(self, d_model, max_len=9000):
        super(PositionalEncoding, self).__init__()
        # 0으로 채워진 (최대 길이, 임베딩 차원) 텐서 생성
        pe = torch.zeros(max_len, d_model)
        
        # 위치 인덱스 (0, 1, 2, ...) 생성
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        # 주파수(Frequency) 계산: 10000^(2i/d_model)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        # 짝수 인덱스에는 사인(sin) 함수 적용
        pe[:, 0::2] = torch.sin(position * div_term)
        # 홀수 인덱스에는 코사인(cos) 함수 적용
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # 배치 차원 추가 (1, max_len, d_model)
        pe = pe.unsqueeze(0)
        
        # 학습되는 파라미터가 아니라 버퍼로 등록하여 상태(state_dict)에 저장되게 함
        self.register_buffer('pe', pe)

    def forward(self, x):
        # 입력 텐서 길이에 맞춰 위치 인코딩을 잘라서 더해줌
        return x + self.pe[:, :x.size(1), :]

# ==========================================
# 2. 마스킹 (Masking)
# ==========================================
def create_padding_mask(x):
    """
    패딩 마스크: 문장의 길이를 맞추기 위해 넣은 0(PAD 토큰)을 어텐션 연산에서 무시(가리기) 위해 사용합니다.
    결과 크기: (batch_size, 1, 1, seq_len)
    """
    return (x == 0).unsqueeze(1).unsqueeze(2)

def create_look_ahead_mask(x):
    """
    룩 어헤드(Look-ahead) 마스크: 디코더에서 미래의 단어를 미리 보고 예측하는(Cheating) 것을 방지합니다.
    자기 자신과 과거의 단어들만 어텐션할 수 있도록 상삼각행렬을 이용해 미래 시점을 마스킹합니다.
    """
    seq_len = x.size(1)
    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
    if x.is_cuda:
        mask = mask.cuda()
    return mask

# ==========================================
# 3. 어텐션 매커니즘 (Attention)
# ==========================================
def scaled_dot_product_attention(query, key, value, mask=None):
    """
    스케일드 닷 프로덕트 어텐션: Query, Key, Value를 이용하여 어텐션 스코어를 계산합니다.
    """
    d_k = query.size(-1)
    
    # Q와 K의 전치 행렬을 내적(Dot Product)하고, sqrt(d_k)로 나누어 스케일링(Scaling)
    # 크기: (batch, num_heads, seq_len_q, seq_len_k)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    
    # 마스크가 존재하면 마스킹된 위치를 매우 작은 값(-1e9)으로 채워 Softmax에서 0이 되게 함
    if mask is not None:
        scores = scores.masked_fill(mask, -1e9)
        
    # Softmax를 적용하여 어텐션 가중치(확률 분포) 생성
    p_attn = F.softmax(scores, dim=-1)
    
    # 어텐션 가중치와 Value 텐서를 행렬 곱하여 최종 결과 반환
    return torch.matmul(p_attn, value), p_attn

class MultiHeadAttention(nn.Module):
    """
    멀티 헤드 어텐션: d_model 차원을 num_heads 개수만큼 나누어 
    서로 다른 시각(Perspective)에서 어텐션을 병렬 수행한 후 다시 하나로 합칩니다.
    """
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        assert d_model % self.num_heads == 0
        self.depth = d_model // self.num_heads # 각 헤드가 담당할 차원 수

        # Q, K, V 생성을 위한 선형 층(Linear Layer)
        self.query_dense = nn.Linear(d_model, d_model)
        self.key_dense = nn.Linear(d_model, d_model)
        self.value_dense = nn.Linear(d_model, d_model)
        # 여러 헤드의 결과를 합친 후 마지막으로 통과시킬 선형 층
        self.dense = nn.Linear(d_model, d_model)

    def split_heads(self, x, batch_size):
        """텐서의 차원을 분리하여 멀티 헤드 연산용 (batch_size, num_heads, seq_len, depth)로 변환"""
        x = x.view(batch_size, -1, self.num_heads, self.depth)
        return x.transpose(1, 2)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        # 1. Q, K, V에 각각 Linear 적용
        query = self.query_dense(query)
        key = self.key_dense(key)
        value = self.value_dense(value)

        # 2. 멀티 헤드를 위해 텐서 분할
        query = self.split_heads(query, batch_size)
        key = self.split_heads(key, batch_size)
        value = self.split_heads(value, batch_size)

        # 3. 스케일드 닷 프로덕트 어텐션 적용
        scaled_attention, _ = scaled_dot_product_attention(query, key, value, mask)
        
        # 4. 분할되었던 멀티 헤드를 다시 합침 (Concatenation)
        scaled_attention = scaled_attention.transpose(1, 2).contiguous()
        concat_attention = scaled_attention.view(batch_size, -1, self.d_model)

        # 5. 최종 Linear 층 통과
        return self.dense(concat_attention)

# ==========================================
# 4. 인코더 및 디코더 레이어 (Layers)
# ==========================================
class EncoderLayer(nn.Module):
    """인코더의 단일 레이어입니다. (논문에서는 이를 N번 반복)"""
    def __init__(self, dff, d_model, num_heads, dropout):
        super(EncoderLayer, self).__init__()
        self.mha = MultiHeadAttention(d_model, num_heads) # 멀티 헤드 어텐션
        self.ffn = nn.Sequential(                         # 피드 포워드 신경망
            nn.Linear(d_model, dff),
            nn.ReLU(),
            nn.Linear(dff, d_model)
        )
        self.layernorm1 = nn.LayerNorm(d_model, eps=1e-6)
        self.layernorm2 = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask):
        # 1. 셀프 어텐션 (Q, K, V가 모두 x)
        attn_output = self.mha(x, x, x, mask)
        attn_output = self.dropout1(attn_output)
        # 잔차 연결(Residual Connection) & 레이어 정규화
        out1 = self.layernorm1(x + attn_output)
        
        # 2. 피드 포워드 신경망
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output)
        # 잔차 연결 & 레이어 정규화
        return self.layernorm2(out1 + ffn_output)

class DecoderLayer(nn.Module):
    """디코더의 단일 레이어입니다. (논문에서는 이를 N번 반복)"""
    def __init__(self, dff, d_model, num_heads, dropout):
        super(DecoderLayer, self).__init__()
        self.mha1 = MultiHeadAttention(d_model, num_heads) # 마스크드 셀프 어텐션용
        self.mha2 = MultiHeadAttention(d_model, num_heads) # 인코더-디코더 어텐션용
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dff),
            nn.ReLU(),
            nn.Linear(dff, d_model)
        )
        self.layernorm1 = nn.LayerNorm(d_model, eps=1e-6)
        self.layernorm2 = nn.LayerNorm(d_model, eps=1e-6)
        self.layernorm3 = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, x, enc_output, look_ahead_mask, padding_mask):
        # 1. 마스크드 셀프 어텐션 (디코더 자신의 시퀀스에 대한 어텐션, 미래 단어 블록)
        attn1 = self.mha1(x, x, x, look_ahead_mask)
        attn1 = self.dropout1(attn1)
        out1 = self.layernorm1(x + attn1)

        # 2. 인코더-디코더 어텐션 (Q는 디코더, K와 V는 인코더의 출력)
        attn2 = self.mha2(out1, enc_output, enc_output, padding_mask)
        attn2 = self.dropout2(attn2)
        out2 = self.layernorm2(out1 + attn2)

        # 3. 피드 포워드 신경망
        ffn_output = self.ffn(out2)
        ffn_output = self.dropout3(ffn_output)
        return self.layernorm3(out2 + ffn_output)

# ==========================================
# 5. 최종 Transformer 모델 (Model)
# ==========================================
class Transformer(nn.Module):
    """
    모든 요소들을 조립하여 완성한 최종 Transformer 모델 구조입니다.
    """
    def __init__(self, vocab_size, num_layers, dff, d_model, num_heads, dropout):
        super(Transformer, self).__init__()
        self.d_model = d_model
        
        # 임베딩 층
        self.embedding = nn.Embedding(vocab_size, d_model)
        # 위치 인코딩
        self.pos_encoding = PositionalEncoding(d_model)
        
        # 인코더와 디코더를 num_layers만큼 쌓음
        self.enc_layers = nn.ModuleList([EncoderLayer(dff, d_model, num_heads, dropout) for _ in range(num_layers)])
        self.dec_layers = nn.ModuleList([DecoderLayer(dff, d_model, num_heads, dropout) for _ in range(num_layers)])
        
        self.dropout = nn.Dropout(dropout)
        # 최종 단어 예측을 위한 선형 층
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, inputs, dec_inputs):
        # 마스크 생성
        enc_padding_mask = create_padding_mask(inputs)
        dec_padding_mask = create_padding_mask(inputs)
        
        # 디코더의 룩어헤드 마스크는 패딩 마스크와 결합하여 사용 (max 연산)
        look_ahead_mask = torch.max(
            create_look_ahead_mask(dec_inputs),
            create_padding_mask(dec_inputs)
        )

        # --- 인코더(Encoder) 파트 ---
        # 1. 단어 임베딩에 스케일링(sqrt(d_model)) 적용 후 위치 인코딩 더함
        enc_out = self.embedding(inputs) * math.sqrt(self.d_model)
        enc_out = self.dropout(self.pos_encoding(enc_out))
        
        # 2. 인코더 레이어를 차례로 통과
        for layer in self.enc_layers:
            enc_out = layer(enc_out, enc_padding_mask)

        # --- 디코더(Decoder) 파트 ---
        # 1. 단어 임베딩 + 위치 인코딩
        dec_out = self.embedding(dec_inputs) * math.sqrt(self.d_model)
        dec_out = self.dropout(self.pos_encoding(dec_out))
        
        # 2. 디코더 레이어를 차례로 통과 (인코더의 최종 출력값 enc_out 전달)
        for layer in self.dec_layers:
            dec_out = layer(dec_out, enc_out, look_ahead_mask, dec_padding_mask)

        # --- 최종 출력 ---
        # (batch_size, dec_seq_len, vocab_size) 형태로 단어 확률 예측을 위한 로짓(Logit) 반환
        return self.fc_out(dec_out)
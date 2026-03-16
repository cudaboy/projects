### Transformer 모델 관련 코드
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# =====================================================================
# 1. Positional Encoding (위치 인코딩)
# =====================================================================
# 트랜스포머는 문장의 단어를 한꺼번에 병렬로 처리하기 때문에 단어의 '순서'를 모릅니다.
# 따라서 각 단어의 위치 정보(사인/코사인 함수 활용)를 단어 벡터에 더해주는 역할을 합니다.
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=9000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        # 수학적인 주기를 만들어 위치마다 고유한 패턴을 가지게 합니다.
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term) # 짝수 인덱스에는 사인 함수
        pe[:, 1::2] = torch.cos(position * div_term) # 홀수 인덱스에는 코사인 함수
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe) # 모델 저장 시 가중치로 업데이트되지 않도록 버퍼로 등록

    def forward(self, x):
        # 들어온 단어 벡터(x)에 위치 정보(pe)를 더해서 반환합니다.
        return x + self.pe[:, :x.size(1), :]

# =====================================================================
# 2. Masking Functions (마스킹 함수들)
# =====================================================================
# 패딩 마스크: 문장 길이를 맞추기 위해 넣은 0([PAD] 토큰)을 모델이 무시하도록 가립니다.
def create_padding_mask(x):
    return (x == 0).unsqueeze(1).unsqueeze(2)

# 룩어헤드 마스크: 챗봇이 다음 단어를 예측할 때, 미래의 단어를 미리 '컨닝'하지 못하게 뒤를 가립니다.
def create_look_ahead_mask(x):
    seq_len = x.size(1)
    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
    if x.is_cuda:
        mask = mask.cuda()
    return mask

# =====================================================================
# 3. Multi-Head Attention (멀티 헤드 어텐션)
# =====================================================================
# 어떤 단어가 다른 어떤 단어와 연관이 깊은지 '집중도'를 계산하는 트랜스포머의 핵심 엔진입니다.
def scaled_dot_product_attention(query, key, value, mask=None):
    d_k = query.size(-1)
    # Query와 Key를 곱해서 연관성 점수를 계산합니다.
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    # 마스크가 있다면 쓸데없는 곳(패딩 등)의 점수를 마이너스 무한대(-1e9)로 깎아버립니다.
    if mask is not None:
        scores = scores.masked_fill(mask, -1e9)
    # Softmax를 통과시켜 점수를 0~1 사이의 확률값으로 바꿉니다.
    p_attn = F.softmax(scores, dim=-1)
    # 최종적으로 Value와 곱해서 문맥이 반영된 벡터를 만듭니다.
    return torch.matmul(p_attn, value), p_attn

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        assert d_model % self.num_heads == 0
        self.depth = d_model // self.num_heads

        # 머리(Head)를 여러 개로 쪼개기 전의 가중치 변환 층들입니다.
        self.query_dense = nn.Linear(d_model, d_model)
        self.key_dense = nn.Linear(d_model, d_model)
        self.value_dense = nn.Linear(d_model, d_model)
        self.dense = nn.Linear(d_model, d_model)
    
    # 텐서를 쪼개서 여러 개의 헤드가 각자 다른 관점에서 문장을 보게 합니다.
    def split_heads(self, x, batch_size):
        x = x.view(batch_size, -1, self.num_heads, self.depth)
        return x.transpose(1, 2)
    
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        query = self.query_dense(query)
        key = self.key_dense(key)
        value = self.value_dense(value)

        query = self.split_heads(query, batch_size)
        key = self.split_heads(key, batch_size)
        value = self.split_heads(value, batch_size)

        # 쪼개진 헤드별로 어텐션을 수행합니다.
        scaled_attention, _ = scaled_dot_product_attention(query, key, value, mask)
        scaled_attention = scaled_attention.transpose(1, 2).contiguous()
        # 다시 하나로 합칩니다.
        concat_attention = scaled_attention.view(batch_size, -1, self.d_model)

        return self.dense(concat_attention)

# =====================================================================
# 4. Encoder / Decoder Layers (인코더/디코더 블록)
# =====================================================================
# 인코더: 사용자의 질문을 이해하는 파트입니다. (셀프 어텐션 -> 신경망)
class EncoderLayer(nn.Module):
    def __init__(self, dff, d_model, num_heads, dropout):
        super(EncoderLayer, self).__init__()
        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dff),
            nn.ReLU(),
            nn.Linear(dff, d_model)
        )
        self.layernorm1 = nn.LayerNorm(d_model, eps=1e-6)
        self.layernorm2 = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x ,  mask):
        attn_output = self.mha(x, x, x, mask)
        attn_output = self.dropout1(attn_output)
        out1 = self.layernorm1(x + attn_output) # 잔차 연결(Residual Connection) 후 정규화

        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output)
        out2 = self.layernorm2(out1 + ffn_output)
        return out2

# 디코더: 이해한 내용을 바탕으로 답변을 한 글자씩 만들어내는 파트입니다.
class DecoderLayer(nn.Module):
    def __init__(self, dff, d_model, num_heads, dropout):
        super(DecoderLayer, self).__init__()
        # 첫 번째 어텐션: 자기가 지금까지 쓴 글을 다시 봅니다 (Look-ahead mask 적용).
        self.mha1 = MultiHeadAttention(d_model, num_heads)
        # 두 번째 어텐션: 인코더가 이해한 '사용자 질문'을 참고합니다 (Cross-attention).
        self.mha2 = MultiHeadAttention(d_model, num_heads)
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
        attn1 = self.mha1(x, x, x, look_ahead_mask)
        attn1 = self.dropout1(attn1)
        out1 = self.layernorm1(x + attn1)

        attn2 = self.mha2(out1, enc_output, enc_output, padding_mask)
        attn2 = self.dropout2(attn2)
        out2 = self.layernorm2(out1 + attn2)

        ffn_output = self.ffn(out2)
        ffn_output = self.dropout3(ffn_output)
        out3 = self.layernorm3(out2 + ffn_output)
        return out3

# =====================================================================
# 5. Transformer Class (최종 조립 모델)
# =====================================================================
# 위에서 만든 부품들을 모두 조립하여 최종 챗봇 모델을 완성합니다.
class Transformer(nn.Module):
    def __init__(self, vocab_size, num_layers, dff, d_model, num_heads, dropout):
        super(Transformer, self).__init__()
        self.d_model = d_model
        
        # 단어 번호를 벡터로 바꿔주는 임베딩 층
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model)
        
        # 인코더와 디코더를 설정한 층 수(num_layers)만큼 반복해서 쌓습니다.
        self.enc_layers = nn.ModuleList([EncoderLayer(dff, d_model, num_heads, dropout) for _ in range(num_layers)])
        self.dec_layers = nn.ModuleList([DecoderLayer(dff, d_model, num_heads, dropout) for _ in range(num_layers)])
        
        self.dropout = nn.Dropout(dropout)
        
        # 최종적으로 어떤 단어(vocab_size)가 나올지 확률을 계산하는 마지막 선형 층
        self.fc_out = nn.Linear(d_model, vocab_size)
    
    def forward(self, inputs, dec_inputs):
        # 입력 문장과 답변 문장에 맞는 마스크를 생성합니다.
        enc_padding_mask = create_padding_mask(inputs)
        dec_padding_mask = create_padding_mask(inputs)
        look_ahead_mask = torch.max(
            create_look_ahead_mask(dec_inputs),
            create_padding_mask(dec_inputs)
        )

        # 1. 인코더 처리 과정
        enc_out = self.embedding(inputs) * math.sqrt(self.d_model)
        enc_out = self.dropout(self.pos_encoding(enc_out))
        for layer in self.enc_layers:
            enc_out = layer(enc_out, enc_padding_mask)

        # 2. 디코더 처리 과정
        dec_out = self.embedding(dec_inputs) * math.sqrt(self.d_model)
        dec_out = self.dropout(self.pos_encoding(dec_out))
        for layer in self.dec_layers:
            dec_out = layer(dec_out, enc_out, look_ahead_mask, dec_padding_mask)

        # 3. 단어 확률 반환
        return self.fc_out(dec_out)
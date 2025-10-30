import torch
import torch.nn as nn

class Concat(nn.Module):
    """(1) 단순 Concat (기존 방식)"""
    def __init__(self, v_dim=512, q_dim=512):
        super().__init__()
        self.output_dim = v_dim + q_dim

    def forward(self, v_seq, v_global, q_seq, q_global):
        # 시퀀스(v_seq, q_seq)는 무시하고 전역 특징만 사용
        return torch.cat([v_global, q_global], dim=1)


class Attention(nn.Module):
    """(2) 간단한 어텐션 (텍스트(질문)가 이미지를 쿼리)"""
    def __init__(self, embed_dim=512, num_heads=8):
        super().__init__()
        # MultiheadAttention은 Query, Key, Value의 embed_dim이 같아야 함
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        # 어텐션 결과(embed_dim)와 텍스트 전역 특징(embed_dim)을 concat
        self.output_dim = embed_dim * 2

    def forward(self, v_seq, v_global, q_seq, q_global):
        # 1. Q = q_global (질문 전역 특징)
        # (B, 512) -> (B, 1, 512)
        q_global_unsq = q_global.unsqueeze(1)
        
        # 2. K, V = v_seq (이미지 영역 시퀀스)
        # (B, 784, 512)
        
        # 3. 어텐션 수행: "질문(Q)이 이미지(K, V)의 어떤 영역을 봐야 하는가?"
        attended_v, _ = self.attention(
            query=q_global_unsq, 
            key=v_seq, 
            value=v_seq
        )
        
        # 4. 어텐션 결과 (B, 1, 512) -> (B, 512)
        attended_v = attended_v.squeeze(1)
        
        # 5. 어텐션으로 정제된 이미지 특징(attended_v)과 텍스트 특징(q_global)을 융합
        fused = torch.cat([attended_v, q_global], dim=1) # (B, 1024)
        return fused
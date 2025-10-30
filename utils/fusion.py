import torch
import torch.nn as nn
import torch.nn.functional as F

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


class CoAttention(nn.Module):
    """(1) Co-Attention (양방향 어텐션)"""
    def __init__(self, embed_dim=512, num_heads=8):
        super().__init__()
        
        # 1. 텍스트가 이미지를 쿼리 (Text-to-Vision Attention)
        self.txt_to_vis_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        
        # 2. 이미지가 텍스트를 쿼리 (Vision-to-Text Attention)
        self.vis_to_txt_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        
        # [attended_v, attended_q]를 concat하므로 output_dim은 2배
        self.output_dim = embed_dim * 2

    def forward(self, v_seq, v_global, q_seq, q_global):
        # MultiheadAttention은 (B, Seq_Len, Dim) 형태의 입력을 기대
        q_global_unsq = q_global.unsqueeze(1) # (B, 1, 512)
        v_global_unsq = v_global.unsqueeze(1) # (B, 1, 512)

        # 1. 텍스트 -> 이미지 어텐션 (Q=q_global, K=v_seq, V=v_seq)
        attended_v, _ = self.txt_to_vis_attn(
            query=q_global_unsq, 
            key=v_seq, 
            value=v_seq
        )
        attended_v = attended_v.squeeze(1) # (B, 512)

        # 2. 이미지 -> 텍스트 어텐션 (Q=v_global, K=q_seq, V=q_seq)
        attended_q, _ = self.vis_to_txt_attn(
            query=v_global_unsq, 
            key=q_seq, 
            value=q_seq
        )
        attended_q = attended_q.squeeze(1) # (B, 512)
        
        # 3. 양방향 어텐션 결과를 융합
        fused = torch.cat([attended_v, attended_q], dim=1) # (B, 1024)
        return fused


class MFBFusion(nn.Module):
    """(2) Multimodal Factorized Bilinear Pooling (MFB)"""
    def __init__(self, embed_dim=512, k_factor=5, output_dim=1024, dropout_prob=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.k_factor = k_factor
        self.output_dim = output_dim # VQA 모델의 concat (512*2)와 맞춤

        # 1. 각 모달리티를 (output_dim * k_factor) 차원으로 확장
        self.v_proj = nn.Linear(embed_dim, output_dim * k_factor)
        self.q_proj = nn.Linear(embed_dim, output_dim * k_factor)
        
        self.dropout = nn.Dropout(dropout_prob)
        # self.output_dim은 VQAModel의 classifier 입력 차원과 일치해야 함

    def forward(self, v_seq, v_global, q_seq, q_global):
        # 이 모듈은 전역 특징(v_global, q_global)만 사용합니다.
        # v_seq와 q_seq는 사용되지 않습니다.
        
        # 1. 프로젝션
        v_proj = self.v_proj(v_global) # (B, output_dim * k)
        q_proj = self.q_proj(q_global) # (B, output_dim * k)
        
        # 2. Hadamard product (element-wise 곱)
        fused_vq = v_proj * q_proj
        fused_vq = self.dropout(fused_vq)
        
        # 3. Sum pooling (Factorized Bilinear Pooling)
        # (B, output_dim * k) -> (B, output_dim, k)
        fused_vq = fused_vq.view(-1, self.output_dim, self.k_factor)
        fused_vq = fused_vq.sum(dim=-1) # (B, output_dim)
        
        # 4. 정규화 (Power-L2)
        fused_vq = torch.sqrt(F.relu(fused_vq)) - torch.sqrt(F.relu(-fused_vq))
        fused_vq = F.normalize(fused_vq, p=2, dim=1) # L2 Norm
        
        return fused_vq # (B, 1024)

class GatedFusion(nn.Module):
    """(3) Gated Fusion (기존 Attention 모듈에 Gating 적용)"""
    def __init__(self, embed_dim=512, num_heads=8):
        super().__init__()
        
        # 1. 기존 Attention 모듈 (Text-to-Vision)
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        
        # 2. 융합 특징의 차원
        self.fusion_dim = embed_dim * 2 # attended_v(512) + q_global(512)
        
        # 3. Gating 메커니즘
        # 융합된 특징을 기반으로 게이트 값을 생성
        self.gate_linear = nn.Linear(self.fusion_dim, self.fusion_dim)
        
        # 4. Gating을 통과할 특징을 생성
        self.proj_linear = nn.Linear(self.fusion_dim, self.fusion_dim)
        
        self.output_dim = self.fusion_dim # (B, 1024)

    def forward(self, v_seq, v_global, q_seq, q_global):
        # 이 모듈은 Gated 'Attention'이므로 v_global, q_seq는 사용하지 않습니다.
        
        # 1. 어텐션 수행 (기존 모듈과 동일)
        q_global_unsq = q_global.unsqueeze(1) # (B, 1, 512)
        attended_v, _ = self.attention(
            query=q_global_unsq, 
            key=v_seq, 
            value=v_seq
        )
        attended_v = attended_v.squeeze(1) # (B, 512)
        
        # 2. 어텐션 결과와 텍스트 특징을 concat (기존 모듈과 동일)
        concat_features = torch.cat([attended_v, q_global], dim=1) # (B, 1024)
        
        # 3. Gating 적용
        # (B, 1024) -> (B, 1024)
        gate = torch.sigmoid(self.gate_linear(concat_features))
        
        # (B, 1024) -> (B, 1024)
        projection = torch.tanh(self.proj_linear(concat_features))
        
        # 4. 게이트를 통과한 최종 융합 특징
        fused = gate * projection # (B, 1024)
        
        return fused
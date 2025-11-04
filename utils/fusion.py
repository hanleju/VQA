import torch
import torch.nn as nn

class Concat(nn.Module):
    """Concat"""
    def __init__(self, v_dim=512, q_dim=512):
        super().__init__()
        self.output_dim = v_dim + q_dim

    def forward(self, v_seq, v_global, q_seq, q_global):
        # 시퀀스(v_seq, q_seq)는 무시하고 전역 특징만 사용
        return torch.cat([v_global, q_global], dim=1)


class Attention(nn.Module):
    """text->image (query)"""
    def __init__(self, embed_dim=512, num_heads=8):
        super().__init__()

        self.attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.output_dim = embed_dim * 2

    def forward(self, v_seq, v_global, q_seq, q_global):
        # Q = q_global 
        # (B, 512) -> (B, 1, 512)
        q_global_unsq = q_global.unsqueeze(1)
        
        # K, V = v_seq 
        # (B, 784, 512)
        
        # 3. 어텐션 수행: "질문(Q)이 이미지(K, V)의 어떤 영역을 봐야 하는가?"
        attended_v, _ = self.attention(
            query=q_global_unsq, 
            key=v_seq, 
            value=v_seq
        )
        
        # attention result: (B, 1, 512) -> (B, 512)
        attended_v = attended_v.squeeze(1)
        
        # 정제된 이미지 특징(attended_v)과 텍스트 특징(q_global)을 융합
        fused = torch.cat([attended_v, q_global], dim=1) # (B, 1024)
        return fused


class CoAttention(nn.Module):
    """(1) Co-Attention"""
    def __init__(self, embed_dim=512, num_heads=8):
        super().__init__()
        
        # Text-to-Vision Attention
        self.txt_to_vis_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        
        # Vision-to-Text Attention
        self.vis_to_txt_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        
        # [attended_v, attended_q]를 concat하므로 output_dim은 2배
        self.output_dim = embed_dim * 2

    def forward(self, v_seq, v_global, q_seq, q_global):
        # MultiheadAttention: (B, Seq_Len, Dim) 
        q_global_unsq = q_global.unsqueeze(1) # (B, 1, 512)
        v_global_unsq = v_global.unsqueeze(1) # (B, 1, 512)

        # text->image (Q=q_global, K=v_seq, V=v_seq)
        attended_v, _ = self.txt_to_vis_attn(
            query=q_global_unsq, 
            key=v_seq, 
            value=v_seq
        )
        attended_v = attended_v.squeeze(1) # (B, 512)

        # image->text (Q=v_global, K=q_seq, V=q_seq)
        attended_q, _ = self.vis_to_txt_attn(
            query=v_global_unsq, 
            key=q_seq, 
            value=q_seq
        )
        attended_q = attended_q.squeeze(1) # (B, 512)
        
        # attention concat
        fused = torch.cat([attended_v, attended_q], dim=1) # (B, 1024)
        return fused

class GatedFusion(nn.Module):
    """(3) Gated Fusion (기존 Attention 모듈에 Gating 적용)"""
    def __init__(self, embed_dim=512, num_heads=8):
        super().__init__()
        
        # Text-to-Vision
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        
        self.fusion_dim = embed_dim * 2 # attended_v(512) + q_global(512)
        
        # Gating (융합된 특징을 기반으로 게이트 값을 생성)
        self.gate_linear = nn.Linear(self.fusion_dim, self.fusion_dim)
        
        self.proj_linear = nn.Linear(self.fusion_dim, self.fusion_dim)
        
        self.output_dim = self.fusion_dim # (B, 1024)

    def forward(self, v_seq, v_global, q_seq, q_global):
        # 이 모듈은 Gated 'Attention'이므로 v_global, q_seq는 사용하지 않습니다.
        
        # 어텐션 수행 (기존 모듈과 동일)
        q_global_unsq = q_global.unsqueeze(1) # (B, 1, 512)
        attended_v, _ = self.attention(
            query=q_global_unsq, 
            key=v_seq, 
            value=v_seq
        )
        attended_v = attended_v.squeeze(1) # (B, 512)
        
        # 어텐션 결과와 텍스트 특징을 concat (기존 모듈과 동일)
        concat_features = torch.cat([attended_v, q_global], dim=1) # (B, 1024)
        
        # Gating (B, 1024) -> (B, 1024)
        gate = torch.sigmoid(self.gate_linear(concat_features))
        
        # (B, 1024) -> (B, 1024)
        projection = torch.tanh(self.proj_linear(concat_features))
        
        # 게이트를 통과한 최종 융합 특징
        fused = gate * projection # (B, 1024)
        
        return fused
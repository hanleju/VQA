import torch
import torch.nn as nn


class PrivacyAwareTokenPruning(nn.Module):
    """
    Privacy-Aware Token Pruning Module
    
    Attention score 기반으로 token을 pruning하되, privacy를 위해:
    1. Stochastic noise 추가 (같은 입력에도 다른 패턴)
    2. Temperature scaling (확신도 낮춤)
    3. Mixup (제거된 정보를 약간 섞음)
    """
    def __init__(self, prune_ratio=0.5, noise_scale=0.1, mixup_alpha=0.05, temperature=0.5):
        super().__init__()
        self.prune_ratio = prune_ratio  # 제거할 비율 (0.5 = 50% 제거)
        self.noise_scale = noise_scale  # Stochastic noise 크기
        self.mixup_alpha = mixup_alpha  # Mixup 비율
        self.temperature = temperature  # Temperature scaling
    
    def forward(self, seq, attn_weights):
        """
        Args:
            seq: (B, N, D) - N개의 토큰
            attn_weights: (B, 1, N) 또는 (B, N) - 각 토큰의 attention score
        
        Returns:
            pruned_seq: (B, K, D) - K개의 선택된 토큰 (K = N * (1 - prune_ratio))
        """
        B, N, D = seq.shape
        
        if attn_weights.dim() == 3:
            attn_weights = attn_weights.squeeze(1)  # (B, N)
        
        # 유지할 토큰 수
        k = max(1, int(N * (1 - self.prune_ratio)))
        
        # ===== 1. Stochastic Noise 추가 =====
        if self.training:
            # Gumbel noise (미분 가능)
            gumbel_noise = -torch.log(-torch.log(torch.rand_like(attn_weights) + 1e-10) + 1e-10)
            noisy_weights = attn_weights + self.noise_scale * gumbel_noise
        else:
            # Gaussian noise (inference)
            noise = torch.randn_like(attn_weights) * self.noise_scale * 0.5
            noisy_weights = attn_weights + noise
        
        # ===== 2. Temperature Scaling =====
        soft_weights = torch.softmax(noisy_weights / self.temperature, dim=-1)
        
        # ===== 3. Top-k Selection =====
        _, top_indices = torch.topk(soft_weights, k=k, dim=-1)
        
        # 선택된 토큰들
        top_indices_expanded = top_indices.unsqueeze(-1).expand(-1, -1, D)
        selected_tokens = torch.gather(seq, 1, top_indices_expanded)
        
        # ===== 4. Mixup (제거된 토큰의 정보를 약간 섞음) =====
        if self.training and self.mixup_alpha > 0:
            # 제거될 토큰들의 mask
            mask = torch.ones(B, N, device=seq.device)
            mask.scatter_(1, top_indices, 0.0)
            
            # 제거될 토큰들의 평균
            remaining_mask = mask.unsqueeze(-1).expand(-1, -1, D)
            remaining_sum = (seq * remaining_mask).sum(dim=1, keepdim=True)
            remaining_count = mask.sum(dim=1, keepdim=True).unsqueeze(-1) + 1e-10
            remaining_mean = remaining_sum / remaining_count
            
            # Mixup: selected + α * remaining
            selected_tokens = selected_tokens + self.mixup_alpha * remaining_mean
        
        return selected_tokens


class Concat(nn.Module):
    """Concat"""
    def __init__(self, v_dim=512, q_dim=512):
        super().__init__()
        self.output_dim = v_dim + q_dim

    def forward(self, v_seq, v_global, q_seq, q_global):
        # 시퀀스(v_seq, q_seq)는 무시하고 전역 특징만 사용
        return torch.cat([v_global, q_global], dim=1)


class Attention(nn.Module):
    """text->image (query) with optional token pruning"""
    def __init__(self, embed_dim=512, num_heads=8, use_pruning=False, prune_ratio=0.5,
                 noise_scale=0.1, mixup_alpha=0.05, temperature=0.5):
        super().__init__()

        self.attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.output_dim = embed_dim * 2
        
        # Privacy-aware token pruning
        self.use_pruning = use_pruning
        if use_pruning:
            self.pruning = PrivacyAwareTokenPruning(
                prune_ratio=prune_ratio,
                noise_scale=noise_scale,
                mixup_alpha=mixup_alpha,
                temperature=temperature
            )

    def forward(self, v_seq, v_global, q_seq, q_global):
        # Q = q_global 
        # (B, 512) -> (B, 1, 512)
        q_global_unsq = q_global.unsqueeze(1)
        
        # K, V = v_seq 
        # (B, 784, 512)
        
        # Privacy-aware pruning (선택적)
        if self.use_pruning:
            # Attention weights 먼저 계산
            attended_v, attn_weights = self.attention(
                query=q_global_unsq, 
                key=v_seq, 
                value=v_seq
            )
            
            # Token pruning 적용
            v_seq_pruned = self.pruning(v_seq, attn_weights)
            
            # Pruned tokens로 다시 attention
            attended_v, _ = self.attention(
                query=q_global_unsq,
                key=v_seq_pruned,
                value=v_seq_pruned
            )
        else:
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
    """Co-Attention with privacy-aware token pruning"""
    def __init__(self, embed_dim=512, num_heads=8, use_pruning=False, prune_ratio=0.5,
                 noise_scale=0.1, mixup_alpha=0.05, temperature=0.5):
        super().__init__()
        
        # Text-to-Vision Attention
        self.txt_to_vis_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        
        # Vision-to-Text Attention
        self.vis_to_txt_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        
        # [attended_v, attended_q]를 concat하므로 output_dim은 2배
        self.output_dim = embed_dim * 2
        
        # Privacy-aware token pruning
        self.use_pruning = use_pruning
        if use_pruning:
            self.v_pruning = PrivacyAwareTokenPruning(
                prune_ratio=prune_ratio,
                noise_scale=noise_scale,
                mixup_alpha=mixup_alpha,
                temperature=temperature
            )
            self.q_pruning = PrivacyAwareTokenPruning(
                prune_ratio=prune_ratio,
                noise_scale=noise_scale,
                mixup_alpha=mixup_alpha,
                temperature=temperature
            )

    def forward(self, v_seq, v_global, q_seq, q_global):
        # MultiheadAttention: (B, Seq_Len, Dim) 
        q_global_unsq = q_global.unsqueeze(1) # (B, 1, 512)
        v_global_unsq = v_global.unsqueeze(1) # (B, 1, 512)

        if self.use_pruning:
            # ===== Text->Image with pruning =====
            # 1. 먼저 attention weights 계산
            _, v_attn_weights = self.txt_to_vis_attn(
                query=q_global_unsq, 
                key=v_seq, 
                value=v_seq
            )
            
            # 2. Vision tokens pruning
            v_seq_pruned = self.v_pruning(v_seq, v_attn_weights)
            
            # 3. Pruned tokens로 다시 attention
            attended_v, _ = self.txt_to_vis_attn(
                query=q_global_unsq,
                key=v_seq_pruned,
                value=v_seq_pruned
            )
            
            # ===== Image->Text with pruning =====
            # 1. 먼저 attention weights 계산
            _, q_attn_weights = self.vis_to_txt_attn(
                query=v_global_unsq,
                key=q_seq,
                value=q_seq
            )
            
            # 2. Text tokens pruning
            q_seq_pruned = self.q_pruning(q_seq, q_attn_weights)
            
            # 3. Pruned tokens로 다시 attention
            attended_q, _ = self.vis_to_txt_attn(
                query=v_global_unsq,
                key=q_seq_pruned,
                value=q_seq_pruned
            )
        else:
            # text->image (Q=q_global, K=v_seq, V=v_seq)
            attended_v, _ = self.txt_to_vis_attn(
                query=q_global_unsq, 
                key=v_seq, 
                value=v_seq
            )

            # image->text (Q=v_global, K=q_seq, V=q_seq)
            attended_q, _ = self.vis_to_txt_attn(
                query=v_global_unsq, 
                key=q_seq, 
                value=q_seq
            )
        
        attended_v = attended_v.squeeze(1) # (B, 512)
        attended_q = attended_q.squeeze(1) # (B, 512)
        
        # attention concat
        fused = torch.cat([attended_v, attended_q], dim=1) # (B, 1024)
        return fused
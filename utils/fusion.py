import torch
import torch.nn as nn


class PrivacyAwareTokenPruning(nn.Module):
    """
    Privacy-Aware Token Pruning Module with Adversarial Mixing
    
    Attention score 기반으로 token을 pruning하되, privacy를 위해:
    1. Adversarial mixing (중요 토큰과 덜 중요한 토큰 교체)
    2. Mixup (제거된 정보를 약간 섞음)
    """
    def __init__(self, prune_ratio=0.5, mixup_alpha=0.05, 
                 adversarial_mix=True, adv_ratio=0.2):
        super().__init__()
        self.prune_ratio = prune_ratio  # 제거할 비율 (0.5 = 50% 제거)
        self.mixup_alpha = mixup_alpha  # Mixup 비율
        self.adversarial_mix = adversarial_mix  # Adversarial mixing 활성화
        self.adv_ratio = adv_ratio  # Top-k 중 교체할 비율 (0.2 = 20% 교체)
    
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
        
        # ===== 1. Top-k Selection (attention score 기반) =====
        _, top_indices = torch.topk(attn_weights, k=k, dim=-1)
        
        # ===== 2. Adversarial Mixing (Privacy 향상) =====
        if self.training and self.adversarial_mix:
            # 중요하지 않은 토큰(bottom-k) 찾기
            remaining_k = N - k
            _, bottom_indices = torch.topk(attn_weights, k=remaining_k, dim=-1, largest=False)
            
            # Top-k 중 교체할 개수
            k_adv = max(1, int(k * self.adv_ratio))
            
            for b in range(B):
                if len(bottom_indices[b]) >= k_adv:
                    # 랜덤하게 교체할 위치 선택
                    replace_positions = torch.randperm(k, device=seq.device)[:k_adv]
                    
                    # Bottom tokens 중 랜덤 선택
                    random_bottom = bottom_indices[b][torch.randperm(remaining_k, device=seq.device)[:k_adv]]
                    
                    # Top-k 인덱스 교체
                    top_indices[b, replace_positions] = random_bottom
        
        # 선택된 토큰들
        top_indices_expanded = top_indices.unsqueeze(-1).expand(-1, -1, D)
        selected_tokens = torch.gather(seq, 1, top_indices_expanded)
        
        # ===== 3. 버려지는 토큰들의 평균을 하나의 토큰으로 추가 =====
        if self.mixup_alpha > 0:
            # 제거될 토큰들의 mask
            mask = torch.ones(B, N, device=seq.device)
            mask.scatter_(1, top_indices, 0.0)
            
            # 제거될 토큰들의 평균 (하나의 토큰으로)
            remaining_mask = mask.unsqueeze(-1).expand(-1, -1, D)
            remaining_sum = (seq * remaining_mask).sum(dim=1, keepdim=True)  # (B, 1, D)
            remaining_count = mask.sum(dim=1, keepdim=True).unsqueeze(-1) + 1e-10
            remaining_token = remaining_sum / remaining_count  # (B, 1, D)
            
            # Mixup alpha 적용
            remaining_token = self.mixup_alpha * remaining_token
            
            # 선택된 토큰들에 버려지는 토큰의 평균을 하나의 토큰으로 추가
            selected_tokens = torch.cat([selected_tokens, remaining_token], dim=1)  # (B, K+1, D)
        
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
                 mixup_alpha=0.05, adversarial_mix=True, adv_ratio=0.2):
        super().__init__()

        self.attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.output_dim = embed_dim * 2
        
        # Privacy-aware token pruning
        self.use_pruning = use_pruning
        if use_pruning:
            self.pruning = PrivacyAwareTokenPruning(
                prune_ratio=prune_ratio,
                mixup_alpha=mixup_alpha,
                adversarial_mix=adversarial_mix,
                adv_ratio=adv_ratio
            )

    def forward(self, v_seq, v_global, q_seq, q_global):
        # Q = q_global 
        # (B, 512) -> (B, 1, 512)
        q_global_unsq = q_global.unsqueeze(1)
        
        # K, V = v_seq 
        # (B, 784, 512)
        
        # Privacy-aware pruning (선택적)
        if self.use_pruning:
            # Attention weights로 중요한 vision tokens 파악
            _, attn_weights = self.attention(
                query=q_global_unsq, 
                key=v_seq, 
                value=v_seq
            )
            
            # 중요한 tokens만 선택 (privacy-aware)
            v_seq_pruned = self.pruning(v_seq, attn_weights)
            
            # 이미 중요한 토큰만 선택했으므로 mean pooling
            # (추가 attention은 redundant)
            attended_v = v_seq_pruned.mean(dim=1)  # (B, 512)
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
                 mixup_alpha=0.05, adversarial_mix=True, adv_ratio=0.2):
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
                mixup_alpha=mixup_alpha,
                adversarial_mix=adversarial_mix,
                adv_ratio=adv_ratio
            )
            self.q_pruning = PrivacyAwareTokenPruning(
                prune_ratio=prune_ratio,
                mixup_alpha=mixup_alpha,
                adversarial_mix=adversarial_mix,
                adv_ratio=adv_ratio
            )

    def forward(self, v_seq, v_global, q_seq, q_global):
        # MultiheadAttention: (B, Seq_Len, Dim) 
        q_global_unsq = q_global.unsqueeze(1) # (B, 1, 512)
        v_global_unsq = v_global.unsqueeze(1) # (B, 1, 512)

        if self.use_pruning:
            # ===== Attention-based pruning =====
            # Text->Vision: q가 v의 어떤 토큰에 주목하는가?
            _, v_attn_weights = self.txt_to_vis_attn(
                query=q_global_unsq, 
                key=v_seq, 
                value=v_seq
            )
            # 중요한 vision tokens만 선택 (privacy-aware)
            v_seq_pruned = self.v_pruning(v_seq, v_attn_weights)
            
            # Vision->Text: v가 q의 어떤 토큰에 주목하는가?
            _, q_attn_weights = self.vis_to_txt_attn(
                query=v_global_unsq,
                key=q_seq,
                value=q_seq
            )
            # 중요한 text tokens만 선택 (privacy-aware)
            q_seq_pruned = self.q_pruning(q_seq, q_attn_weights)
            
            # ===== Pruned tokens aggregation =====
            # 이미 중요한 토큰만 선택했으므로 mean pooling으로 집계
            # (추가 attention은 redundant하므로 제거)
            attended_v = v_seq_pruned.mean(dim=1)  # (B, 512)
            attended_q = q_seq_pruned.mean(dim=1)  # (B, 512)
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
        
        attended_v = attended_v.squeeze(1) if attended_v.dim() == 3 else attended_v  # (B, 512)
        attended_q = attended_q.squeeze(1) if attended_q.dim() == 3 else attended_q  # (B, 512)
        
        # attention concat
        fused = torch.cat([attended_v, attended_q], dim=1) # (B, 1024)
        return fused
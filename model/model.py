import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.fusion import Concat, Attention, CoAttention

class VQAModel(nn.Module):
    def __init__(self, vision = "VisionEncoder_ResNet50", text="TextEncoder_Bert",
                 fusion_type="concat", hidden_dim=1024, num_classes=13,
                 use_token_pruning=False, prune_ratio=0.5, 
                 noise_scale=0.1, mixup_alpha=0.05, temperature=0.5):
        super().__init__()
        
        self.fusion_type = fusion_type

        self.vision_encoder = vision(out_features=512)
        self.text_encoder = text(hidden_dim=512)
        
        # Fusion module with optional privacy-aware token pruning
        if fusion_type == "concat":
            self.fusion_module = Concat(
                v_dim=self.vision_encoder.output_dim,
                q_dim=self.text_encoder.output_dim
            )

        elif fusion_type == "attention":
            self.fusion_module = Attention(
                embed_dim=512, num_heads=8,
                use_pruning=use_token_pruning,
                prune_ratio=prune_ratio,
                noise_scale=noise_scale,
                mixup_alpha=mixup_alpha,
                temperature=temperature
            )

        elif self.fusion_type == "co_attention":
            self.fusion_module = CoAttention(
                embed_dim=512, num_heads=8,
                use_pruning=use_token_pruning,
                prune_ratio=prune_ratio,
                noise_scale=noise_scale,
                mixup_alpha=mixup_alpha,
                temperature=temperature
            )

        else:
            raise ValueError(f"Unknown fusion type: {fusion_type}")

        fusion_output_dim = self.fusion_module.output_dim
        
        self.classifier = nn.Sequential(
            nn.Linear(fusion_output_dim, hidden_dim*2),
            nn.BatchNorm1d(hidden_dim*2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim*2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, images, input_ids, attention_mask):
        v_seq, v_global = self.vision_encoder(images)
        q_seq, q_global = self.text_encoder(input_ids, attention_mask)
        
        fused_feat = self.fusion_module(v_seq, v_global, q_seq, q_global)
        
        return self.classifier(fused_feat)
    

class VQAModel_IB(nn.Module):
    """
    VQA Model with Information Bottleneck (IB)
    
    구조:
    1. Fusion: 비전과 텍스트 특징 융합 (Token Pruning 옵션)
    2. IB Encoder: 융합된 특징을 압축된 표현으로 인코딩
    3. IB Decoder: 압축된 표현을 원래 차원으로 복원 (재구성 손실 계산용)
    4. Classifier: 압축된 표현으로부터 분류
    """
    def __init__(self, vision = "VisionEncoder_ResNet50", text="TextEncoder_Bert", 
                 fusion_type="concat", hidden_dim=1024, num_classes=13, 
                 bottleneck_dim=256, beta=0.1,
                 use_token_pruning=False, prune_ratio=0.5,
                 noise_scale=0.1, mixup_alpha=0.05, temperature=0.5):
        super().__init__()
        
        self.fusion_type = fusion_type
        self.beta = beta  # IB 정규화 가중치 (클수록 압축 강함)
        self.bottleneck_dim = bottleneck_dim

        self.vision_encoder = vision(out_features=512)
        self.text_encoder = text(hidden_dim=512)
        
        # Fusion module with optional privacy-aware token pruning
        if fusion_type == "concat":
            self.fusion_module = Concat(
                v_dim=self.vision_encoder.output_dim,
                q_dim=self.text_encoder.output_dim
            )

        elif fusion_type == "attention":
            self.fusion_module = Attention(
                embed_dim=512, num_heads=8,
                use_pruning=use_token_pruning,
                prune_ratio=prune_ratio,
                noise_scale=noise_scale,
                mixup_alpha=mixup_alpha,
                temperature=temperature
            )

        elif self.fusion_type == "co_attention":
            self.fusion_module = CoAttention(
                embed_dim=512, num_heads=8,
                use_pruning=use_token_pruning,
                prune_ratio=prune_ratio,
                noise_scale=noise_scale,
                mixup_alpha=mixup_alpha,
                temperature=temperature
            )

        else:
            raise ValueError(f"Unknown fusion type: {fusion_type}")

        fusion_output_dim = self.fusion_module.output_dim

        # ===== Information Bottleneck 구성 =====
        # 1. Encoder: 융합 특징 -> 압축된 표현 (평균, 분산 파라미터 학습)
        self.encoder = nn.Sequential(
            nn.Linear(fusion_output_dim, fusion_output_dim // 2),
            nn.BatchNorm1d(fusion_output_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(fusion_output_dim // 2, fusion_output_dim // 4),
            nn.BatchNorm1d(fusion_output_dim // 4),
            nn.ReLU()
        )
        
        # 평균과 분산 파라미터화 (VAE 스타일)
        self.mu_layer = nn.Linear(fusion_output_dim // 4, bottleneck_dim)
        self.logvar_layer = nn.Linear(fusion_output_dim // 4, bottleneck_dim)
        
        # 2. Decoder: 압축된 표현 -> 원래 차원 복원 (재구성 손실 계산)
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, fusion_output_dim // 4),
            nn.BatchNorm1d(fusion_output_dim // 4),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(fusion_output_dim // 4, fusion_output_dim // 2),
            nn.BatchNorm1d(fusion_output_dim // 2),
            nn.ReLU(),
            nn.Linear(fusion_output_dim // 2, fusion_output_dim)
        )
        
        # 3. 분류기: 압축된 표현으로부터 분류
        self.classifier = nn.Sequential(
            nn.Linear(bottleneck_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_classes)
        )

    def reparameterize(self, mu, logvar, training=True):
        """
        Reparameterization trick: z = mu + eps * std
        
        Args:
            mu: 평균
            logvar: 로그 분산
            training: 학습 중인지 여부
        
        Returns:
            z: 샘플링된 잠재 변수
        """
        if training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            z = mu + eps * std
        else:
            # 테스트 시에는 평균만 사용
            z = mu
        return z

    def forward(self, images, input_ids, attention_mask):
        """
        Forward pass with Information Bottleneck
        
        Returns:
            logits: 분류 로짓
            aux_dict: 재구성 손실 계산용 보조 정보
                - z: 압축된 표현
                - mu: 평균
                - logvar: 로그 분산
                - fused_feat: 원본 융합 특징 (재구성 손실 계산용)
        """
        # 1. 인코더에서 특징 추출 및 융합
        v_seq, v_global = self.vision_encoder(images)
        q_seq, q_global = self.text_encoder(input_ids, attention_mask)
        fused_feat = self.fusion_module(v_seq, v_global, q_seq, q_global)
        
        # 2. Information Bottleneck 인코딩
        encoded = self.encoder(fused_feat)
        mu = self.mu_layer(encoded)
        logvar = self.logvar_layer(encoded)
        
        # 3. Reparameterization
        z = self.reparameterize(mu, logvar, training=self.training)
        
        # 4. 분류
        logits = self.classifier(z)
        
        # 5. 학습용 보조 정보 (IB 손실 계산용)
        aux_dict = {
            'z': z,
            'mu': mu,
            'logvar': logvar,
            'fused_feat': fused_feat,
            'reconstructed': self.decoder(z)
        }
        
        return logits, aux_dict
    
    def get_ib_loss(self, aux_dict):
        """
        Information Bottleneck 정규화 손실
        
        IB 손실 = KL divergence + beta * 재구성 손실
        
        KL divergence: 압축된 표현을 표준 정규분포에 가깝게 유지
        재구성 손실: 압축된 표현이 충분한 정보를 유지하도록 강제
        
        Args:
            aux_dict: forward()에서 반환한 보조 정보
        
        Returns:
            ib_loss: Information Bottleneck 정규화 손실
        """
        mu = aux_dict['mu']
        logvar = aux_dict['logvar']
        fused_feat = aux_dict['fused_feat']
        reconstructed = aux_dict['reconstructed']
        
        # KL divergence: N(mu, logvar) vs N(0, 1)
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        kl_loss = kl_loss / mu.size(0)  # Batch 정규화CoAttention
        
        # 재구성 손실
        reconstruction_loss = F.mse_loss(reconstructed, fused_feat)
        
        # 전체 IB 손실
        ib_loss = kl_loss + self.beta * reconstruction_loss
        
        return {
            'ib_loss': ib_loss,
            'kl_loss': kl_loss,
            'reconstruction_loss': reconstruction_loss
        }
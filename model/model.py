import torch.nn as nn
from utils.fusion import Concat, Attention, CoAttention, GatedFusion

class VQAModel(nn.Module):
    def __init__(self, vision = "VisionEncoder_ResNet50", text="TextEncoder_Bert",fusion_type="concat", hidden_dim=1024, num_classes=13):
        super().__init__()
        
        self.fusion_type = fusion_type

        self.vision_encoder = vision(out_features=512)
        self.text_encoder = text(hidden_dim=512)
        
        if fusion_type == "concat":
            self.fusion_module = Concat(
                v_dim=self.vision_encoder.output_dim,
                q_dim=self.text_encoder.output_dim
            )

        elif fusion_type == "attention":
            self.fusion_module = Attention(embed_dim=512, num_heads=8)

        elif self.fusion_type == "co_attention":
            self.fusion_module = CoAttention(embed_dim=512, num_heads=8)

        elif self.fusion_type == "gated_fusion":
            self.fusion_module = GatedFusion(embed_dim=512, num_heads=8)

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
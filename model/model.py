import torch.nn as nn
from utils.fusion import Concat, Attention

class VQAModel(nn.Module):
    def __init__(self, vision = "VisionEncoder_ResNet50", text="TextEncoder_Bert",fusion_type="concat", hidden_dim=1024, num_classes=13):
        super().__init__()
        
        # 1. (수정된) 커스텀 인코더 초기화
        # self.vision_encoder = VisionEncoder_CNN(out_features=512)
        self.vision_encoder = vision(out_features=512)
        self.text_encoder = text(hidden_dim=512)
        
        # 2. 퓨전 모듈 선택
        if fusion_type == "concat":
            self.fusion_module = Concat(
                v_dim=self.vision_encoder.output_dim,
                q_dim=self.text_encoder.output_dim
            )
        elif fusion_type == "attention":
            # (embed_dim이 512로 동일하다고 가정)
            self.fusion_module = Attention(embed_dim=512, num_heads=8)
        else:
            raise ValueError(f"Unknown fusion type: {fusion_type}")

        # 3. 퓨전 모듈에서 출력 차원을 가져옴
        fusion_output_dim = self.fusion_module.output_dim
        
        # 4. 요청하신 classifier (입력 차원이 자동으로 설정됨)
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
        
        # (인코더 동결 코드는 없음. 모두 학습)

    def forward(self, images, input_ids, attention_mask):
        # 1. 인코더에서 (시퀀스, 전역) 특징 모두 추출
        v_seq, v_global = self.vision_encoder(images)
        q_seq, q_global = self.text_encoder(input_ids, attention_mask)
        
        # 2. 선택된 퓨전 모듈에 전달
        fused_feat = self.fusion_module(v_seq, v_global, q_seq, q_global)
        
        # 3. 분류기 통과
        return self.classifier(fused_feat)
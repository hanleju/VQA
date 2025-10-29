import torch
import torch.nn as nn
import torchvision.models as models

from fusion import Concat, Attention
from transformers import BertModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class VisionEncoder_CNN(nn.Module):
    def __init__(self, out_features=512):
        super().__init__()
        # (layer1, layer2는 이전과 동일)
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2) # 112
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2) # 56
        )
        # 1. 시퀀스 출력을 위해 layer3의 채널을 256 -> 512로 변경 (텍스트와 맞추기 위함)
        self.layer3 = nn.Sequential(
            nn.Conv2d(128, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512), nn.ReLU(), nn.MaxPool2d(2) # (B, 512, 28, 28)
        )
        
        # 2. 전역(Global) 특징 추출 경로 (기존과 동일)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(512, out_features) # (B, 512)
        
        self.output_dim = out_features

    def forward(self, images):
        x = self.layer1(images)
        x = self.layer2(x)
        feature_map = self.layer3(x) # (B, 512, 28, 28)
        
        # (1) 시퀀스 특징 반환
        # (B, 512, 28, 28) -> (B, 512, 784) -> (B, 784, 512)
        v_seq = feature_map.flatten(2).permute(0, 2, 1) 
        
        # (2) 전역 특징 반환
        pooled = self.pool(feature_map) # (B, 512, 1, 1)
        flattened = self.flatten(pooled) # (B, 512)
        v_global = self.fc(flattened)    # (B, 512)
        
        return v_seq, v_global

class VisionEncoder_ResNet50(nn.Module):
    """
    사전 학습된 ResNet-50을 Vision Encoder로 사용합니다.
    """
    def __init__(self, out_features=512):
        super().__init__()
        # 1. ResNet-50 로드 (마지막 FC 레이어 제외)
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        modules = list(resnet.children())[:-2] # avgpool과 fc 제외
        self.resnet_features = nn.Sequential(*modules)
        
        # ResNet-50의 layer4 출력 차원은 2048 (B, 2048, 7, 7)
        resnet_output_dim = 2048
        
        # 2. 시퀀스 특징 추출을 위한 Projection
        # (B, 2048, 7, 7) -> (B, 512, 7, 7)
        self.seq_projection = nn.Conv2d(resnet_output_dim, out_features, kernel_size=1)
        
        # 3. 전역 특징 추출을 위한 경로
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.global_projection = nn.Linear(resnet_output_dim, out_features)
        
        self.output_dim = out_features

    def forward(self, images):
        # (B, 3, 224, 224) -> (B, 2048, 7, 7)
        feature_map = self.resnet_features(images)
        
        # (1) 전역 특징 (Global Feature)
        pooled = self.pool(feature_map)
        flattened = self.flatten(pooled)
        v_global = self.global_projection(flattened) # (B, 512)
        
        # (2) 시퀀스 특징 (Sequence Feature)
        projected_map = self.seq_projection(feature_map)
        v_seq = projected_map.flatten(2).permute(0, 2, 1) # (B, 49, 512)
        
        return v_seq, v_global

class TextEncoder(nn.Module):
    """
    사전 학습된 BERT 모델을 Text Encoder로 사용합니다.
    """
    def __init__(self, model_name='bert-base-uncased', hidden_dim=512):
        super().__init__()
        
        # 1. BERT 모델 로드
        self.bert = BertModel.from_pretrained(model_name)
        
        # 2. BERT의 기본 출력 차원 (bert-base-uncased는 768)
        bert_output_dim = self.bert.config.hidden_size # 768
        
        # 3. (선택 사항) BERT 출력을 VQA 모델의 공통 차원(hidden_dim)으로 매핑
        #    (VisionEncoder와 차원을 맞추기 위함)
        self.seq_projection = nn.Linear(bert_output_dim, hidden_dim)
        self.global_projection = nn.Linear(bert_output_dim, hidden_dim)
        
        self.output_dim = hidden_dim # 최종 출력 차원은 512

    def forward(self, input_ids, attention_mask):
        """
        데이터 로더에서 collate_fn을 통해 생성된
        input_ids와 attention_mask를 직접 받습니다.
        """
        
        # BERT 모델은 device 처리가 필요 없습니다.
        # 모델 자체가 .to(device)로 이동하면 내부 텐서도 모두 이동합니다.
        
        # 1. BERT 순전파
        outputs = self.bert(input_ids=input_ids, 
                            attention_mask=attention_mask)
        
        # 2. BERT 출력에서 시퀀스 특징과 전역 특징 추출
        last_hidden_state = outputs.last_hidden_state # (B, L, 768)
        pooler_output = outputs.pooler_output     # (B, 768) [CLS] 토큰의 특징
        
        # 3. (선택 사항) Projection Layer 통과
        q_seq = self.seq_projection(last_hidden_state)     # (B, L, 512)
        q_global = self.global_projection(pooler_output) # (B, 512)
        
        # (만약 hidden_dim=768로 모델 전체를 통일한다면 projection은 필요 없습니다)

        return q_seq, q_global
    

class VQAModel(nn.Module):
    def __init__(self, fusion_type="concat", hidden_dim=1024, num_classes=13):
        super().__init__()
        
        # 1. (수정된) 커스텀 인코더 초기화
        # self.vision_encoder = VisionEncoder_CNN(out_features=512)
        self.vision_encoder = VisionEncoder_ResNet50(out_features=512)
        self.text_encoder = TextEncoder(hidden_dim=512)
        
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
import torch
import torch.nn as nn
import torchvision.models as models

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
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from transformers import BertTokenizer
from functools import partial
import os

from data.data import VQADataset, collate_fn_with_tokenizer
from utils.src import validate

from model.vision_encoder import VisionEncoder_ResNet50
from model.text_encoder import TextEncoder_Bert
from model.model import VQAModel

# --- 1. 하이퍼파라미터 및 설정 ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATASET_ROOT = 'D:/VQA/easyVQA'
NUM_CLASSES = 13  # 테스트에서 확인된 값
BATCH_SIZE = 32   # 배치 크기 (테스트는 2였지만, 학습은 32~64 권장)
NUM_EPOCHS = 20   # 총 학습 에포크
LEARNING_RATE = 1e-5
VAL_SPLIT_RATIO = 0.1 # 전체 데이터 중 10%를 검증용으로 사용
NUM_WORKERS = 4   # 데이터 로딩 워커 수 (환경에 맞게 조절)
MODEL_SAVE_PATH = './checkpoints' # 모델 저장 경로
FUSION_TYPE = 'attention' # 'concat' 또는 'attention' 선택 (attention 권장)
Vision = VisionEncoder_ResNet50
Text = TextEncoder_Bert


# --- 5. 메인 실행 함수 ---
def main():
    print(f"Using device: {DEVICE}")
    os.makedirs(MODEL_SAVE_PATH, exist_ok=True) # 모델 저장 폴더 생성

    # --- 데이터셋 및 로더 준비 ---
    image_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225])
    ])
    
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    test_dataset = VQADataset(root_dir=DATASET_ROOT, 
                                split='test',
                                transform=image_transform)

    # 3. DataLoader 생성
    collate_fn = partial(collate_fn_with_tokenizer, tokenizer=tokenizer)
    
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=NUM_WORKERS,
        pin_memory=True # GPU 사용 시 메모리 고정
    )

    # --- 모델, 옵티마이저, 손실함수 준비 ---
    model = VQAModel(vision=Vision, text=Text, fusion_type=FUSION_TYPE, num_classes=NUM_CLASSES).to(DEVICE)
    criterion = nn.CrossEntropyLoss()

    model.load_state_dict(torch.load('./checkpoints/best_model.pth', map_location=DEVICE))

    val_loss, val_acc = validate(model, test_loader, criterion, DEVICE)
    
    print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")


# --- 7. 스크립트 실행 ---
if __name__ == '__main__':
    main()
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from transformers import BertTokenizer
from functools import partial
import os
import traceback

from model import VQAModel
from data import VQADataset, collate_fn_with_tokenizer
from src import train, validate

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
    
    # 1. 전체 'train' 데이터셋 로드
    try:
        full_dataset = VQADataset(root_dir=DATASET_ROOT, 
                                  split='train', 
                                  transform=image_transform)

        # (NUM_CLASSES가 일치하는지 재확인)
        assert full_dataset.num_answers == NUM_CLASSES, \
            f"NUM_CLASSES({NUM_CLASSES})가 데이터셋의 답변 수({full_dataset.num_answers})와 일치하지 않습니다."

    except Exception as e:
        print(f"데이터셋 로드 중 오류 발생: {e}")
        traceback.print_exc()
        return

    # 2. Train / Validation 분할
    total_size = len(full_dataset)
    val_size = int(total_size * VAL_SPLIT_RATIO)
    train_size = total_size - val_size
    
    print(f"Data Split -> Train: {train_size} samples, Validation: {val_size} samples")
    
    # random_split으로 데이터셋 분리
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    # 3. DataLoader 생성
    collate_fn = partial(collate_fn_with_tokenizer, tokenizer=tokenizer)
    
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=NUM_WORKERS,
        pin_memory=True # GPU 사용 시 메모리 고정
    )
    
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False, # 검증 시에는 섞지 않음
        collate_fn=collate_fn,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )

    # --- 모델, 옵티마이저, 손실함수 준비 ---
    model = VQAModel(fusion_type=FUSION_TYPE, num_classes=NUM_CLASSES).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()
    
    # (선택 사항) 학습률 스케줄러
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    print(f"모델({FUSION_TYPE} 퓨전) 및 학습 설정 완료. Start Train...")
    
    best_val_acc = 0.0 # 최고 검증 정확도 추적

    # --- 6. 전체 학습/검증 루프 ---
    for epoch in range(NUM_EPOCHS):
        print(f"\n--- Epoch {epoch+1}/{NUM_EPOCHS} ---")
        
        # 1. 학습
        train_loss, train_acc = train(model, train_loader, optimizer, criterion, DEVICE)
        
        # 2. 검증
        val_loss, val_acc = validate(model, val_loader, criterion, DEVICE)
        
        print(f"Epoch {epoch+1} | Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

        # 3. 학습률 스케줄러 업데이트
        scheduler.step()

        # 4. 최고 성능 모델 저장
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_path = os.path.join(MODEL_SAVE_PATH, f"best_model_epoch_{epoch+1}_acc_{val_acc:.2f}.pth")
            torch.save(model.state_dict(), save_path)
            print(f"새로운 최고 검증 정확도 달성! 모델 저장: {save_path}")

    print(f"\n--- Train End ---")
    print(f"최고 검증 정확도: {best_val_acc:.2f}%")

# --- 7. 스크립트 실행 ---
if __name__ == '__main__':
    main()
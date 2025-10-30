import torch
from tqdm import tqdm

# --- 2. 정확도 계산 헬퍼 함수 ---
def calc_accuracy(preds, labels):
    """
    VQA 정확도를 계산합니다.
    preds: 모델의 로짓 출력 (Batch, Num_Classes)
    labels: 정답 인덱스 (Batch,)
    """
    # 
    # 가장 높은 로짓(확률)을 가진 인덱스를 예측값으로 선택
    pred_indices = torch.argmax(preds, dim=1)
    
    # 예측과 정답이 일치하는 개수 계산
    correct_count = (pred_indices == labels).sum().item()
    return correct_count

# --- 3. 1 에포크 학습 함수 ---
def train(model, train_loader, optimizer, criterion, device):
    model.train() # 모델을 학습 모드로 설정 (Dropout, BatchNorm 활성화)
    
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    
    # tqdm을 사용하여 진행률 표시
    train_pbar = tqdm(train_loader, desc=f"Training", leave=False)
    
    for batch in train_pbar:
        # 1. 데이터 준비 및 device로 이동
        images = batch['image'].to(device)
        inputs = batch['inputs'].to(device)
        answers = batch['answer'].to(device)

        # 2. Gradient 초기화
        optimizer.zero_grad()
        
        # 3. 모델 순전파
        outputs = model(
            images=images,
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask']
        )
        
        # 4. 손실 계산
        loss = criterion(outputs, answers)
        
        # 5. 역전파 및 가중치 업데이트
        loss.backward()
        optimizer.step()
        
        # 6. 통계 업데이트
        total_loss += loss.item() * images.size(0) # 배치 크기만큼 곱해줌
        total_correct += calc_accuracy(outputs, answers)
        total_samples += images.size(0)
        
        # 진행률 표시줄에 현재 배치 손실 업데이트
        train_pbar.set_postfix(loss=f"{loss.item():.4f}")
        
    avg_loss = total_loss / total_samples
    avg_acc = (total_correct / total_samples) * 100
    
    return avg_loss, avg_acc

# --- 4. 1 에포크 검증 함수 ---
def validate(model, val_loader, criterion, device):
    model.eval() # 모델을 평가 모드로 설정 (Dropout, BatchNorm 비활성화)
    
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    
    val_pbar = tqdm(val_loader, desc=f"Validating", leave=False)
    
    with torch.no_grad(): # Gradient 계산 중지
        for batch in val_pbar:
            images = batch['image'].to(device)
            inputs = batch['inputs'].to(device)
            answers = batch['answer'].to(device)
            
            outputs = model(
                images=images,
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask']
            )
            
            loss = criterion(outputs, answers)
            
            total_loss += loss.item() * images.size(0)
            total_correct += calc_accuracy(outputs, answers)
            total_samples += images.size(0)
            
            val_pbar.set_postfix(loss=f"{loss.item():.4f}")

    avg_loss = total_loss / total_samples
    avg_acc = (total_correct / total_samples) * 100
    
    return avg_loss, avg_acc
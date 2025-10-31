import torch
from tqdm import tqdm
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import os
from pathlib import Path

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
def validate(model, val_loader, criterion, device, mode="val", weight_path=None):
    model.eval()
    
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    
    # 예측값과 정답을 모두 저장
    all_preds = []
    all_labels = []
    
    desc = "Testing" if mode == "test" else "Validating"
    val_pbar = tqdm(val_loader, desc=desc, leave=False)
    
    with torch.no_grad():
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
            
            # 예측값과 정답 저장
            pred_indices = torch.argmax(outputs, dim=1)
            all_preds.extend(pred_indices.cpu().numpy())
            all_labels.extend(answers.cpu().numpy())
            
            total_loss += loss.item() * images.size(0)
            total_correct += calc_accuracy(outputs, answers)
            total_samples += images.size(0)
            
            val_pbar.set_postfix(loss=f"{loss.item():.4f}")

    avg_loss = total_loss / total_samples
    avg_acc = (total_correct / total_samples) * 100

    metrics = {
        'loss': avg_loss,
        'accuracy': avg_acc
    }

    if mode == "test" and weight_path:
        # 메트릭 계산
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, 
            average='weighted',
            zero_division=0  # 예측이 없는 클래스의 경우 0으로 처리
        )
        
        # 혼동 행렬 계산 및 시각화
        cm = confusion_matrix(all_labels, all_preds)
        
        # 결과 저장 경로 설정
        weight_dir = os.path.dirname(weight_path)
        result_dir = os.path.join(weight_dir, 'results')
        os.makedirs(result_dir, exist_ok=True)
        
        # 가중치 파일 이름에서 확장자 제거
        weight_name = Path(weight_path).stem
        
        # 혼동 행렬 플롯 저장
        plt.figure(figsize=(1200, 1000))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix - {weight_name}')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.savefig(os.path.join(result_dir, f'{weight_name}_confusion_matrix.png'))
        plt.close()
        
        # 테스트 결과를 텍스트 파일로 저장
        with open(os.path.join(result_dir, f'{weight_name}_results.txt'), 'w') as f:
            f.write(f"Test Results for {weight_name}\n")
            f.write("-" * 50 + "\n")
            f.write(f"Loss: {avg_loss:.4f}\n")
            f.write(f"Accuracy: {avg_acc:.2f}%\n")
            f.write(f"Precision: {precision:.4f}\n")
            f.write(f"Recall: {recall:.4f}\n")
            f.write(f"F1 Score: {f1:.4f}\n")
        
        # 메트릭 딕셔너리에 추가
        metrics.update({
            'precision': precision,
            'recall': recall,
            'f1': f1
        })
    
    return avg_loss, avg_acc, metrics
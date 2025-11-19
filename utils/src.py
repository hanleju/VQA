import torch
from tqdm import tqdm
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import os
from pathlib import Path


def calc_accuracy(preds, labels):
    """
    VQA 정확도를 계산합니다.
    preds: 모델의 로짓 출력 (Batch, Num_Classes)
    labels: 정답 인덱스 (Batch,)
    """

    pred_indices = torch.argmax(preds, dim=1)
    
    correct_count = (pred_indices == labels).sum().item()
    return correct_count

def train(model, train_loader, optimizer, criterion, device, model_type="VQAModel", use_dp_sgd=False, lambda_ib=0.01):
    """
    통합 학습 함수 - 모든 모델 타입과 학습 방식을 지원
    
    Args:
        model: VQA 모델 (VQAModel, VQAModel_IB 등)
        train_loader: 학습 데이터 로더
        optimizer: 옵티마이저 (dict 또는 단일 옵티마이저)
        criterion: 손실 함수
        device: GPU/CPU 디바이스
        model_type: 모델 타입 ("VQAModel", "VQAModel_IB" 등)
        use_dp_sgd: DP-SGD 사용 여부
        lambda_ib: IB 모델에서 사용할 IB 손실의 가중치
    
    Returns:
        avg_loss: 평균 손실
        avg_acc: 평균 정확도 (%)
    """
    model.train()
    
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    
    # optimizer가 dict인지 확인 (DP-SGD의 경우)
    is_split_optimizer = isinstance(optimizer, dict)
    
    # Progress bar 설정
    desc = "Training"
    if use_dp_sgd:
        desc = "Training (DP-SGD)"
    elif model_type == "VQAModel_IB":
        desc = "Training (IB)"
    
    train_pbar = tqdm(train_loader, desc=desc, leave=False)
    
    for batch in train_pbar:
        images = batch['image'].to(device)
        inputs = batch['inputs'].to(device)
        answers = batch['answer'].to(device)

        # Zero gradients
        if is_split_optimizer:
            optimizer['main'].zero_grad()
            optimizer['classifier'].zero_grad()
        else:
            optimizer.zero_grad()
        
        # Forward pass - 모델 타입에 따라 출력 형식이 다름
        if model_type == "VQAModel_IB":
            outputs, aux_dict = model(
                images=images,
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask']
            )
            
            # 분류 손실
            ce_loss = criterion(outputs, answers)
            
            # Information Bottleneck 손실
            ib_losses = model.get_ib_loss(aux_dict)
            ib_loss = ib_losses['ib_loss']
            
            # 전체 손실 = 분류 손실 + lambda_ib * IB 손실
            loss = ce_loss + lambda_ib * ib_loss
            
        else:  # VQAModel 또는 기타 모델
            outputs = model(
                images=images,
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask']
            )
            loss = criterion(outputs, answers)
        
        # Backward pass
        loss.backward()
        
        # Optimizer step
        if is_split_optimizer:
            optimizer['main'].step()
            optimizer['classifier'].step()
        else:
            optimizer.step()
        
        # Metrics 계산
        total_loss += loss.item() * images.size(0)
        total_correct += calc_accuracy(outputs, answers)
        total_samples += images.size(0)
        
        # Progress bar 업데이트
        if model_type == "VQAModel_IB":
            train_pbar.set_postfix(
                ce_loss=f"{ce_loss.item():.4f}",
                kl_loss=f"{ib_losses['kl_loss'].item():.4f}",
                recon_loss=f"{ib_losses['reconstruction_loss'].item():.4f}"
            )
        else:
            train_pbar.set_postfix(loss=f"{loss.item():.4f}")
        
    avg_loss = total_loss / total_samples
    avg_acc = (total_correct / total_samples) * 100
    
    return avg_loss, avg_acc


def validate(model, val_loader, criterion, device, model_type="VQAModel", mode="val", weight_path=None):
    """
    통합 검증 함수 - 모든 모델 타입을 지원
    
    Args:
        model: VQA 모델 (VQAModel, VQAModel_IB 등)
        val_loader: 검증 데이터 로더
        criterion: 손실 함수
        device: GPU/CPU 디바이스
        model_type: 모델 타입 ("VQAModel", "VQAModel_IB" 등)
        mode: 검증 모드 ("val" 또는 "test")
        weight_path: 가중치 경로 (test 모드에서 결과 저장용)
    
    Returns:
        avg_loss: 평균 손실
        avg_acc: 평균 정확도 (%)
        metrics: 평가 지표 dict
    """
    model.eval()
    
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    all_preds = []
    all_labels = []
    
    # Progress bar 설정
    desc = "Testing" if mode == "test" else "Validating"
    if model_type == "VQAModel_IB":
        desc = desc + " (IB)"
    
    val_pbar = tqdm(val_loader, desc=desc, leave=False)
    
    with torch.no_grad():
        for batch in val_pbar:
            images = batch['image'].to(device)
            inputs = batch['inputs'].to(device)
            answers = batch['answer'].to(device)
            
            # Forward pass - 모델 타입에 따라 출력 형식이 다름
            if model_type == "VQAModel_IB":
                outputs, aux_dict = model(
                    images=images,
                    input_ids=inputs['input_ids'],
                    attention_mask=inputs['attention_mask']
                )
            else:  # VQAModel 또는 기타 모델
                outputs = model(
                    images=images,
                    input_ids=inputs['input_ids'],
                    attention_mask=inputs['attention_mask']
                )
            
            loss = criterion(outputs, answers)
            
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

    # Test 모드에서 결과 저장
    if mode == "test" and weight_path:
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, 
            average='weighted',
            zero_division=0
        )
        
        weight_dir = os.path.dirname(weight_path)
        result_dir = os.path.join(weight_dir, 'results')
        os.makedirs(result_dir, exist_ok=True)
        
        weight_name = Path(weight_path).stem
        
        # Pandas DataFrame으로 테이블 형식 생성 (2x4: Metrics x Values)
        results_data = {
            'Accuracy': [f"{avg_acc:.2f}%"],
            'Precision': [f"{precision:.4f}"],
            'Recall': [f"{recall:.4f}"],
            'F1-Score': [f"{f1:.4f}"]
        }
        results_df = pd.DataFrame(results_data)
        
        # 파일 저장
        result_file = os.path.join(result_dir, f'{weight_name}_results.txt')
        with open(result_file, 'w') as f:
            f.write(f"Test Results for {weight_name}\n")
            f.write(f"Loss: {avg_loss:.4f}\n")
            f.write("=" * 60 + "\n\n")
            f.write(results_df.to_string(index=False))
            f.write("\n")
        
        # CSV 형식으로도 저장
        csv_file = os.path.join(result_dir, f'{weight_name}_results.csv')
        results_df.to_csv(csv_file, index=False)
        
        metrics.update({
            'precision': precision,
            'recall': recall,
            'f1': f1
        })
    
    return avg_loss, avg_acc, metrics
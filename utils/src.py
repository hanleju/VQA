import torch
from tqdm import tqdm
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
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

def train(model, train_loader, optimizer, criterion, device):
    model.train()
    
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    
    train_pbar = tqdm(train_loader, desc=f"Training", leave=False)
    
    for batch in train_pbar:
        images = batch['image'].to(device)
        inputs = batch['inputs'].to(device)
        answers = batch['answer'].to(device)

        optimizer.zero_grad()
        
        outputs = model(
            images=images,
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask']
        )

        loss = criterion(outputs, answers)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * images.size(0)
        total_correct += calc_accuracy(outputs, answers)
        total_samples += images.size(0)
        
        train_pbar.set_postfix(loss=f"{loss.item():.4f}")
        
    avg_loss = total_loss / total_samples
    avg_acc = (total_correct / total_samples) * 100
    
    return avg_loss, avg_acc

def validate(model, val_loader, criterion, device, mode="val", weight_path=None):
    model.eval()
    
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

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
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, 
            average='weighted',
            zero_division=0
        )
        
        # confusion matrix
        # cm = confusion_matrix(all_labels, all_preds)
        # plt.figure(figsize=(500, 300))
        # sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        # plt.title(f'Confusion Matrix - {weight_name}')
        # plt.xlabel('Predicted')
        # plt.ylabel('True')
        # plt.savefig(os.path.join(result_dir, f'{weight_name}_confusion_matrix.png'))
        # plt.close()

        weight_dir = os.path.dirname(weight_path)
        result_dir = os.path.join(weight_dir, 'results')
        os.makedirs(result_dir, exist_ok=True)
        
        weight_name = Path(weight_path).stem
        
        with open(os.path.join(result_dir, f'{weight_name}_results.txt'), 'w') as f:
            f.write(f"Test Results for {weight_name}\n")
            f.write("-" * 50 + "\n")
            f.write(f"Loss: {avg_loss:.4f}\n")
            f.write(f"Accuracy: {avg_acc:.2f}%\n")
            f.write(f"Precision: {precision:.4f}\n")
            f.write(f"Recall: {recall:.4f}\n")
            f.write(f"F1 Score: {f1:.4f}\n")
        
        metrics.update({
            'precision': precision,
            'recall': recall,
            'f1': f1
        })
    
    return avg_loss, avg_acc, metrics


def train_epoch_ib(model, train_loader, optimizer, criterion, device, lambda_ib=0.01):
    """
    Information Bottleneck을 적용하여 한 에포크 학습
    
    Args:
        model: VQAModel_IB 모델
        train_loader: 학습 데이터 로더
        optimizer: 옵티마이저
        criterion: 손실 함수
        device: GPU/CPU 디바이스
        lambda_ib: IB 손실의 가중치 (클수록 정보 압축이 강함)
    
    Returns:
        epoch_loss: 에포크 전체 손실
        epoch_acc: 에포크 정확도
    """
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    
    train_pbar = tqdm(train_loader, desc="Training IB", leave=False)
    
    for batch in train_pbar:
        images = batch['image'].to(device)
        inputs = batch['inputs'].to(device)
        answers = batch['answer'].to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        logits, aux_dict = model(
            images=images,
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask']
        )
        
        # 분류 손실
        ce_loss = criterion(logits, answers)
        
        # Information Bottleneck 손실
        ib_losses = model.get_ib_loss(aux_dict)
        ib_loss = ib_losses['ib_loss']
        
        # 전체 손실 = 분류 손실 + lambda_ib * IB 손실
        total_ce_loss = ce_loss + lambda_ib * ib_loss
        
        total_ce_loss.backward()
        optimizer.step()
        
        total_loss += total_ce_loss.item() * images.size(0)
        total_correct += calc_accuracy(logits, answers)
        total_samples += images.size(0)
        
        train_pbar.set_postfix(
            ce_loss=f"{ce_loss.item():.4f}",
            kl_loss=f"{ib_losses['kl_loss'].item():.4f}",
            recon_loss=f"{ib_losses['reconstruction_loss'].item():.4f}"
        )
    
    epoch_loss = total_loss / total_samples
    epoch_acc = (total_correct / total_samples) * 100
    
    return epoch_loss, epoch_acc


def validate_ib(model, val_loader, criterion, device):
    """
    Information Bottleneck 모델 검증
    
    Args:
        model: VQAModel_IB 모델
        val_loader: 검증 데이터 로더
        criterion: 손실 함수
        device: GPU/CPU 디바이스
    
    Returns:
        val_acc: 검증 정확도 (%)
        val_loss: 검증 손실
    """
    model.eval()
    total_acc = 0.0
    total_loss = 0.0
    total_samples = 0
    
    val_pbar = tqdm(val_loader, desc="Validating IB", leave=False)
    
    with torch.no_grad():
        for batch in val_pbar:
            images = batch['image'].to(device)
            inputs = batch['inputs'].to(device)
            answers = batch['answer'].to(device)
            
            # Forward pass
            logits, aux_dict = model(
                images=images,
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask']
            )
            
            # 손실 계산
            ce_loss = criterion(logits, answers)
            total_loss += ce_loss.item() * images.size(0)
            
            # 정확도 계산
            total_correct = calc_accuracy(logits, answers)
            total_acc += total_correct
            total_samples += images.size(0)
            
            val_pbar.set_postfix(loss=f"{ce_loss.item():.4f}")
    
    val_acc = (total_acc / total_samples) * 100
    avg_loss = total_loss / total_samples
    
    return val_acc, avg_loss
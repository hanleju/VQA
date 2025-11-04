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
import os
import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
from PIL import Image

torch.manual_seed(42)


def save_importance_dict(importance_dict, save_path):
    """importance_dict를 파일로 저장"""
    import pickle
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cpu_importance = {name: tensor.cpu() for name, tensor in importance_dict.items()}
    with open(save_path, 'wb') as f:
        pickle.dump(cpu_importance, f)
    print(f"✓ Importance dict saved to {save_path}")

def load_importance_dict(load_path, device):
    """importance_dict를 파일에서 로드"""
    import pickle
    with open(load_path, 'rb') as f:
        importance_dict = pickle.load(f)
    importance_dict = {name: tensor.to(device) for name, tensor in importance_dict.items()}
    print(f"✓ Importance dict loaded from {load_path}")
    return importance_dict

def save_metrics(metrics, save_path):
    """평가 메트릭을 JSON으로 저장"""
    import json
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"✓ Metrics saved to {save_path}")

def load_metrics(load_path):
    """평가 메트릭을 JSON에서 로드"""
    import json
    with open(load_path, 'r') as f:
        metrics = json.load(f)
    print(f"✓ Metrics loaded from {load_path}")
    return metrics


def compute_weight_importance_mas(model, dataloader, processor, model_name, dataset_root, split, device):
    """
    MAS (Memory Aware Synapses)를 사용하여 weight importance 계산
    
    Args:
        model: VQA 모델 (BLIP or ViLT)
        dataloader: 데이터 로더
        processor: Hugging Face processor
        model_name: 모델 이름
        dataset_root: 데이터셋 루트 경로
        split: 'train' or 'test'
        device: 디바이스
        
    Returns:
        importance_dict: 각 파라미터의 importance (normalized)
    """
    model.eval()
    importance_dict = {}
    
    # Initialize importance to zeros
    for name, param in model.named_parameters():
        if param.requires_grad:
            importance_dict[name] = torch.zeros_like(param).to(device)
    
    print(f"Computing weight importance for {model_name}...")
    num_samples = 0
    
    for batch in tqdm(dataloader, desc="Computing importance", leave=False):
        try:
            # 이미지 로드
            image_paths = []
            for img_id in batch['image_id']:
                base_path = os.path.join(dataset_root, split, 'images', img_id)
                found = False
                for ext in ['.jpg', '.png', '.jpeg']:
                    if os.path.exists(base_path + ext):
                        image_paths.append(base_path + ext)
                        found = True
                        break
                if not found:
                    if os.path.exists(base_path):
                        image_paths.append(base_path)
                    else:
                        raise FileNotFoundError(f"Image not found: {base_path}")
            
            images = [Image.open(path).convert('RGB') for path in image_paths]
            questions = batch['question'][:len(images)]
            answers = batch['answer'][:len(images)].to(device)
            answer_texts = batch['answer_text'][:len(images)] if 'answer_text' in batch else [str(a.item()) for a in answers]
            
            # Forward pass
            model.zero_grad()
            
            if "blip" in model_name.lower():
                inputs = processor(images=images, text=questions, return_tensors="pt", 
                                 padding=True, truncation=True).to(device)
                labels = processor(text=answer_texts, return_tensors="pt", padding=True, truncation=True).input_ids.to(device)
                outputs = model(**inputs, labels=labels)
                loss = outputs.loss
                
            elif "vilt" in model_name.lower():
                inputs = processor(images=images, text=questions, return_tensors="pt", 
                                 padding=True, truncation=True, max_length=40).to(device)
                outputs = model(**inputs)
                logits = outputs.logits
                # MAS: gradient of L2 norm of output
                loss = torch.sum(logits ** 2)
            
            # Backward to compute gradients
            loss.backward()
            
            # Accumulate absolute gradients as importance
            for name, param in model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    importance_dict[name] += param.grad.abs()
            
            num_samples += len(images)
            
        except Exception as e:
            print(f"\n⚠ Error in importance computation: {e}")
            continue
    
    # Normalize by number of samples
    for name in importance_dict.keys():
        importance_dict[name] /= num_samples
    
    # Normalize to [0, 1] per layer
    layer_groups = {}
    for name in importance_dict.keys():
        layer_prefix = name.split('.')[0]
        if layer_prefix not in layer_groups:
            layer_groups[layer_prefix] = []
        layer_groups[layer_prefix].append(name)
    
    normalized_importance = {}
    for layer_prefix, param_names in layer_groups.items():
        all_importances = torch.cat([importance_dict[name].flatten() for name in param_names])
        min_imp = all_importances.min()
        max_imp = all_importances.max()
        
        if max_imp > min_imp:
            for name in param_names:
                normalized_importance[name] = (importance_dict[name] - min_imp) / (max_imp - min_imp)
        else:
            for name in param_names:
                normalized_importance[name] = torch.zeros_like(importance_dict[name])
    
    print(f"✓ Importance computed for {len(normalized_importance)} parameters")
    
    return normalized_importance


def compute_difficulty_aware_consistency_loss(logits, answer_indices, temperature=2.0):
    """
    난이도를 고려한 일관성 loss
    
    핵심 차이점:
    - unlearning.py: 같은 레이블끼리만 일관성 → 난이도 무시 ❌
    - 이 함수: 같은 (난이도, 레이블)끼리만 일관성 → 난이도 보존 ✅
    
    난이도 측정: Loss 값 (높을수록 어려움)
    그룹화: Percentile 기반 (Easy/Medium/Hard)
    
    Args:
        logits: 모델 출력 (batch_size, num_classes)
        answer_indices: 각 샘플의 정답 인덱스 (batch_size,)
        temperature: softmax temperature
        
    Returns:
        consistency_loss: 같은 (난이도, 레이블)을 가진 샘플들 간의 일관성 loss
    """
    # 1. Loss로 난이도 측정
    criterion = nn.CrossEntropyLoss(reduction='none')
    losses = criterion(logits, answer_indices)  # (batch_size,)
    
    # 2. Percentile로 난이도 그룹 할당
    if len(losses) < 3:
        # 샘플이 너무 적으면 기본 label consistency
        probs = torch.softmax(logits / temperature, dim=-1)
        unique_answers = torch.unique(answer_indices)
        consistency_loss = torch.tensor(0.0, device=logits.device)
        num_groups = 0
        
        for answer in unique_answers:
            mask = (answer_indices == answer)
            if mask.sum() > 1:
                group_probs = probs[mask]
                mean_prob = group_probs.mean(dim=0, keepdim=True)
                kl_div = (group_probs * (torch.log(group_probs + 1e-10) - torch.log(mean_prob + 1e-10))).sum(dim=-1)
                consistency_loss += kl_div.mean()
                num_groups += 1
        
        if num_groups > 0:
            consistency_loss = consistency_loss / num_groups
        
        return consistency_loss
    
    # Percentile 계산 (CPU에서)
    losses_cpu = losses.detach().cpu().numpy()
    p33 = np.percentile(losses_cpu, 33)
    p66 = np.percentile(losses_cpu, 66)
    
    # 난이도 그룹 할당 (0=Easy, 1=Medium, 2=Hard)
    difficulty_groups = torch.zeros(len(losses), dtype=torch.long, device=logits.device)
    difficulty_groups[losses >= p33] = 1
    difficulty_groups[losses >= p66] = 2
    
    # 3. Softmax with temperature
    probs = torch.softmax(logits / temperature, dim=-1)
    
    # 4. 같은 (난이도, 레이블) 그룹별로 일관성 강화
    consistency_loss = torch.tensor(0.0, device=logits.device)
    num_groups = 0
    
    for difficulty in [0, 1, 2]:  # Easy, Medium, Hard
        difficulty_mask = (difficulty_groups == difficulty)
        
        if difficulty_mask.sum() > 0:
            # 이 난이도 내에서 레이블별로 그룹화
            difficulty_answers = answer_indices[difficulty_mask]
            difficulty_probs = probs[difficulty_mask]
            
            unique_answers = torch.unique(difficulty_answers)
            
            for answer in unique_answers:
                answer_mask = (difficulty_answers == answer)
                if answer_mask.sum() > 1:  # 같은 (난이도, 레이블)이 2개 이상
                    group_probs = difficulty_probs[answer_mask]
                    
                    # Mean distribution
                    mean_prob = group_probs.mean(dim=0, keepdim=True)
                    
                    # KL divergence from mean
                    kl_div = (group_probs * (torch.log(group_probs + 1e-10) - torch.log(mean_prob + 1e-10))).sum(dim=-1)
                    consistency_loss += kl_div.mean()
                    num_groups += 1
    
    if num_groups > 0:
        consistency_loss = consistency_loss / num_groups
    
    return consistency_loss


def save_evaluation_results(original_metrics, trained_metrics, model_key, output_dir):
    """평가 결과 저장 및 비교"""
    import json
    
    comparison = {
        'original': original_metrics,
        'trained': trained_metrics,
        'improvement': {}
    }
    
    # Overall improvement
    orig_overall = original_metrics['overall']
    train_overall = trained_metrics['overall']
    
    variance_improvement = (orig_overall['label_wise_loss_variance'] - 
                           train_overall['label_wise_loss_variance']) / \
                           (orig_overall['label_wise_loss_variance'] + 1e-10) * 100
    
    cv_improvement = (orig_overall['label_consistency_cv'] - 
                      train_overall['label_consistency_cv']) / \
                     (orig_overall['label_consistency_cv'] + 1e-10) * 100
    
    mean_loss_change = (train_overall['label_wise_mean_loss'] - 
                       orig_overall['label_wise_mean_loss']) / \
                       (orig_overall['label_wise_mean_loss'] + 1e-10) * 100
    
    comparison['improvement']['overall'] = {
        'label_variance_reduction': f"{variance_improvement:.2f}%",
        'consistency_cv_reduction': f"{cv_improvement:.2f}%",
        'mean_loss_change': f"{mean_loss_change:+.2f}%"
    }
    
    # Difficulty-wise improvement
    difficulty_improvement = {}
    for diff_name in ['Easy', 'Medium', 'Hard']:
        if diff_name in original_metrics['difficulty_wise'] and diff_name in trained_metrics['difficulty_wise']:
            orig_cv = original_metrics['difficulty_wise'][diff_name]['consistency_cv']
            train_cv = trained_metrics['difficulty_wise'][diff_name]['consistency_cv']
            
            cv_imp = (orig_cv - train_cv) / (orig_cv + 1e-10) * 100
            difficulty_improvement[diff_name] = f"{cv_imp:.2f}%"
    
    comparison['improvement']['difficulty_wise'] = difficulty_improvement
    
    # Save to JSON
    save_path = os.path.join(output_dir, model_key, 'evaluation_metrics.json')
    with open(save_path, 'w') as f:
        json.dump(comparison, f, indent=4)
    
    print(f"✓ Evaluation results saved to {save_path}")
    
    # Print comparison
    print(f"\n{'='*60}")
    print(f"IMPROVEMENT SUMMARY for {model_key.upper()}")
    print(f"{'='*60}")
    print(f"Overall:")
    print(f"  Label Variance Reduction: {variance_improvement:+.2f}%")
    print(f"  Label Consistency (CV) Reduction: {cv_improvement:+.2f}%")
    print(f"  Mean Loss Change: {mean_loss_change:+.2f}%")
    print(f"\nDifficulty-wise CV Reduction:")
    for diff_name, improvement in difficulty_improvement.items():
        print(f"  {diff_name}: {improvement}")
    print(f"{'='*60}\n")
"""
Unlearning for Reference Models (BLIP, ViLT)
목표: Pretrained VQA 모델을 더 general하게 만들기

핵심 아이디어:
- 동일한 레이블(답변)을 가지는 데이터들의 출력은 비슷한 경향으로 나와야 함
- Weight importance regularization으로 특정 샘플에 대한 편향 제거
- 더 general한 reference model → 더 정확한 difficulty calibration

사용 모델:
- dandelin/vilt-b32-finetuned-vqa
- Salesforce/blip-vqa-base

# BLIP과 ViLT 모두 unlearning
python unlearning_reference.py --models blip

"""
import os
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Subset
from torchvision import transforms
from transformers import (
    BlipProcessor, BlipForQuestionAnswering,
    ViltProcessor, ViltForQuestionAnswering,
    BertTokenizer
)
from functools import partial
from tqdm import tqdm
import numpy as np
from PIL import Image
import argparse

from data.data import VQADataset, collate_fn_with_tokenizer

torch.manual_seed(42)


def compute_weight_importance_mas(model, dataloader, processor, model_name, dataset_root, split, device):
    """
    MAS (Memory Aware Synapses)를 사용하여 weight importance 계산
    Hugging Face 모델용
    
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
                # BLIP needs labels for forward pass
                labels = processor(text=answer_texts, return_tensors="pt", padding=True, truncation=True).input_ids.to(device)
                outputs = model(**inputs, labels=labels)
                # Use loss for MAS instead of logits
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


def save_importance_dict(importance_dict, save_path):
    """
    importance_dict를 파일로 저장
    """
    import pickle
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    # CPU로 이동하여 저장
    cpu_importance = {name: tensor.cpu() for name, tensor in importance_dict.items()}
    with open(save_path, 'wb') as f:
        pickle.dump(cpu_importance, f)
    print(f"✓ Importance dict saved to {save_path}")


def load_importance_dict(load_path, device):
    """
    importance_dict를 파일에서 로드
    """
    import pickle
    with open(load_path, 'rb') as f:
        importance_dict = pickle.load(f)
    # 디바이스로 이동
    importance_dict = {name: tensor.to(device) for name, tensor in importance_dict.items()}
    print(f"✓ Importance dict loaded from {load_path}")
    return importance_dict


def save_metrics(metrics, save_path):
    """
    평가 메트릭을 JSON으로 저장
    """
    import json
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"✓ Metrics saved to {save_path}")


def load_metrics(load_path):
    """
    평가 메트릭을 JSON에서 로드
    """
    import json
    with open(load_path, 'r') as f:
        metrics = json.load(f)
    print(f"✓ Metrics loaded from {load_path}")
    return metrics


def compute_label_consistency_loss(logits, answer_indices, temperature=2.0):
    """
    동일한 레이블을 가진 샘플들의 출력 분포를 일관되게 만드는 loss
    
    Args:
        logits: 모델 출력 (batch_size, num_classes)
        answer_indices: 각 샘플의 정답 인덱스 (batch_size,)
        temperature: softmax temperature (높을수록 smooth)
        
    Returns:
        consistency_loss: 같은 답변을 가진 샘플들 간의 일관성 loss
    """
    # Softmax with temperature
    probs = torch.softmax(logits / temperature, dim=-1)
    
    # Group by answer
    unique_answers = torch.unique(answer_indices)
    consistency_loss = 0.0
    num_groups = 0
    
    for answer in unique_answers:
        mask = (answer_indices == answer)
        if mask.sum() > 1:  # 같은 답변이 2개 이상 있을 때만
            group_probs = probs[mask]  # (group_size, num_classes)
            
            # 같은 답변을 가진 샘플들의 출력 분포가 비슷해지도록
            # Mean distribution
            mean_prob = group_probs.mean(dim=0, keepdim=True)  # (1, num_classes)
            
            # KL divergence from mean
            kl_div = (group_probs * (torch.log(group_probs + 1e-10) - torch.log(mean_prob + 1e-10))).sum(dim=-1)
            consistency_loss += kl_div.mean()
            num_groups += 1
    
    if num_groups > 0:
        consistency_loss /= num_groups
    
    return consistency_loss


def generate_adversarial_examples_hf(model, processor, images, questions, targets, 
                                     model_name, device, epsilon=0.05):
    """
    Hugging Face 모델용 adversarial examples 생성
    
    Args:
        model: VQA 모델
        processor: Hugging Face processor
        images: PIL Images 리스트
        questions: 질문 텍스트 리스트
        targets: 정답 레이블
        model_name: 모델 이름
        device: 디바이스
        epsilon: perturbation 크기
        
    Returns:
        adv_images: adversarial images (텐서)
    """
    model.eval()
    
    # PIL Image를 tensor로 변환
    if "blip" in model_name.lower():
        inputs = processor(images=images, text=questions, return_tensors="pt", 
                         padding=True, truncation=True).to(device)
    elif "vilt" in model_name.lower():
        inputs = processor(images=images, text=questions, return_tensors="pt", 
                         padding=True, truncation=True, max_length=40).to(device)
    
    # pixel_values에 gradient 필요
    inputs['pixel_values'].requires_grad = True
    
    # Forward pass
    outputs = model(**inputs)
    logits = outputs.logits
    
    # Compute loss
    criterion = nn.CrossEntropyLoss()
    loss = criterion(logits, targets)
    
    # Backward pass
    model.zero_grad()
    loss.backward()
    
    # Generate adversarial examples (FGSM)
    sign_grad = inputs['pixel_values'].grad.sign()
    adv_pixel_values = inputs['pixel_values'] + epsilon * sign_grad
    adv_pixel_values = torch.clamp(adv_pixel_values, 0, 1)
    
    return adv_pixel_values.detach()


def unlearning_reference_model(model, processor, model_name, dataloader, 
                               importance_dict, initial_params, args, device):
    """
    Reference model에 대한 unlearning 수행
    
    목표:
    1. Adversarial examples로 특정 샘플에 대한 overfitting 제거
    2. Label consistency로 같은 답변에 대한 일관된 출력 유도
    3. Weight importance로 중요한 지식 보존
    
    Args:
        model: Hugging Face VQA 모델
        processor: Processor
        model_name: 모델 이름
        dataloader: 데이터 로더
        importance_dict: Weight importance
        initial_params: 초기 파라미터
        args: 설정
        device: 디바이스
    """
    model.train()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()
    
    lambda_imp = args.lambda_unlearn
    lambda_consistency = args.lambda_consistency
    
    print(f"\n{'='*60}")
    print(f"Unlearning {model_name}")
    print(f"Lambda (importance): {lambda_imp}")
    print(f"Lambda (consistency): {lambda_consistency}")
    print(f"Learning rate: {args.lr}")
    print(f"Epochs: {args.unlearn_epochs}")
    print(f"{'='*60}\n")
    
    for epoch in range(args.unlearn_epochs):
        epoch_loss = 0.0
        epoch_adv_loss = 0.0
        epoch_cons_loss = 0.0
        epoch_reg_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.unlearn_epochs}")
        
        for batch in pbar:
            try:
                # 이미지 로드
                image_paths = []
                for img_id in batch['image_id']:
                    base_path = os.path.join(args.dataset_root, 'train', 'images', img_id)
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
                            continue
                
                if len(image_paths) == 0:
                    continue
                
                images = [Image.open(path).convert('RGB') for path in image_paths]
                questions = batch['question'][:len(images)]
                answers = batch['answer'][:len(images)].to(device)
                
                # === 1. Adversarial loss ===
                # Generate adversarial examples
                with torch.enable_grad():
                    adv_pixel_values = generate_adversarial_examples_hf(
                        model, processor, images, questions, answers, model_name, device
                    )
                
                # Forward pass on adversarial examples
                model.train()
                if "blip" in model_name.lower():
                    inputs = processor(images=images, text=questions, return_tensors="pt", 
                                     padding=True, truncation=True).to(device)
                    inputs['pixel_values'] = adv_pixel_values
                elif "vilt" in model_name.lower():
                    inputs = processor(images=images, text=questions, return_tensors="pt", 
                                     padding=True, truncation=True, max_length=40).to(device)
                    inputs['pixel_values'] = adv_pixel_values
                
                outputs = model(**inputs)
                logits = outputs.logits
                
                # Adversarial loss (maximize loss on adversarial examples)
                adv_loss = -criterion(logits, answers)
                
                # === 2. Label consistency loss ===
                # 같은 답변을 가진 샘플들의 출력을 일관되게
                cons_loss = compute_label_consistency_loss(logits, answers)
                
                # === 3. Weight importance regularization ===
                reg_loss = 0.0
                for name, param in model.named_parameters():
                    if name in importance_dict and name in initial_params:
                        omega_bar = 1.0 - importance_dict[name]
                        param_change = (param - initial_params[name]) ** 2
                        reg_loss += torch.sum(omega_bar * param_change)
                
                # === 4. Combined loss ===
                total_loss = adv_loss + lambda_consistency * cons_loss + lambda_imp * reg_loss
                
                # Backward and optimize
                optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                # Accumulate losses
                epoch_loss += total_loss.item()
                epoch_adv_loss += adv_loss.item()
                epoch_cons_loss += cons_loss.item()
                epoch_reg_loss += reg_loss.item()
                num_batches += 1
                
                pbar.set_postfix({
                    'Loss': f'{total_loss.item():.4f}',
                    'Adv': f'{adv_loss.item():.4f}',
                    'Cons': f'{cons_loss.item():.4f}',
                    'Reg': f'{reg_loss.item():.4f}'
                })
                
            except Exception as e:
                print(f"\n⚠ Error in batch: {e}")
                continue
        
        # Epoch summary
        if num_batches > 0:
            avg_loss = epoch_loss / num_batches
            avg_adv = epoch_adv_loss / num_batches
            avg_cons = epoch_cons_loss / num_batches
            avg_reg = epoch_reg_loss / num_batches
            
            print(f"Epoch {epoch+1}/{args.unlearn_epochs} - "
                  f"Loss: {avg_loss:.4f}, Adv: {avg_adv:.4f}, "
                  f"Cons: {avg_cons:.4f}, Reg: {avg_reg:.4f}")


def evaluate_generalization(model, processor, model_name, dataloader, 
                           dataset_root, split, device):
    """
    모델의 generalization 평가
    
    핵심 지표:
    1. Label-wise Loss Variance: 같은 답변을 가진 샘플들의 loss 분산
       - 낮을수록 일관된 출력 → 더 general
    2. Label Consistency (CV): 같은 레이블 내 loss의 변동계수 (Coefficient of Variation)
       - 낮을수록 일관된 출력 → 더 general
       - CV = std / mean (평균 대비 표준편차 비율)
    
    Args:
        model: VQA 모델
        processor: Processor
        model_name: 모델 이름
        dataloader: 데이터 로더
        dataset_root: 데이터셋 루트
        split: 'train' or 'test'
        device: 디바이스
        
    Returns:
        metrics: 평가 지표 dict
    """
    model.eval()
    
    # 각 레이블별로 loss와 output 저장
    label_losses = {}  # {label: [loss1, loss2, ...]}
    label_outputs = {}  # {label: [output1, output2, ...]}
    
    criterion = nn.CrossEntropyLoss(reduction='none')
    
    print(f"Evaluating generalization for {model_name}...")
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
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
                            continue
                
                if len(image_paths) == 0:
                    continue
                
                images = [Image.open(path).convert('RGB') for path in image_paths]
                questions = batch['question'][:len(images)]
                answers = batch['answer'][:len(images)].to(device)
                answer_texts = batch['answer_text'][:len(images)] if 'answer_text' in batch else [str(a.item()) for a in answers]
                
                # Forward pass - 모든 모델을 loss 기반으로 통일
                if "blip" in model_name.lower():
                    inputs = processor(images=images, text=questions, return_tensors="pt", 
                                     padding=True, truncation=True).to(device)
                    # BLIP needs labels for loss computation
                    labels = processor(text=answer_texts, return_tensors="pt", padding=True, truncation=True).input_ids.to(device)
                    outputs = model(**inputs, labels=labels)
                    
                    # BLIP은 generative model이므로 batch loss를 각 샘플에 할당
                    batch_loss = outputs.loss.item()
                    losses_batch = np.array([batch_loss] * len(images))
                    
                    # Output은 loss 값 자체를 사용 (1D)
                    outputs_batch = [[loss] for loss in losses_batch]
                    
                elif "vilt" in model_name.lower():
                    inputs = processor(images=images, text=questions, return_tensors="pt", 
                                     padding=True, truncation=True, max_length=40).to(device)
                    outputs = model(**inputs)
                    logits = outputs.logits
                    
                    # Loss per sample (일관성을 위해 loss 사용)
                    losses_batch = criterion(logits, answers).cpu().numpy()
                    
                    # Output은 loss 값으로 통일 (BLIP과 일관성)
                    outputs_batch = [[loss] for loss in losses_batch]
                
                # Group by label
                for i, (answer, loss, output) in enumerate(zip(answers.cpu().numpy(), losses_batch, outputs_batch)):
                    answer = int(answer)
                    if answer not in label_losses:
                        label_losses[answer] = []
                        label_outputs[answer] = []
                    label_losses[answer].append(loss)
                    label_outputs[answer].append(output)  # 1D array [loss]
                
            except Exception as e:
                print(f"\n⚠ Error: {e}")
                continue
    
    # === Compute metrics ===
    
    # 1. Label-wise Loss Variance (핵심!)
    label_variances = []
    label_means = []
    label_stds = []
    for label, losses in label_losses.items():
        if len(losses) > 1:
            label_variances.append(np.var(losses))
            label_means.append(np.mean(losses))
            label_stds.append(np.std(losses))
    
    avg_label_variance = np.mean(label_variances) if label_variances else 0.0
    avg_label_mean_loss = np.mean(label_means) if label_means else 0.0
    avg_label_std = np.mean(label_stds) if label_stds else 0.0
    
    # 2. Label Consistency: 표준편차 기반 (낮을수록 일관적)
    # Normalize by mean to get coefficient of variation (CV)
    label_cvs = []
    for label, losses in label_losses.items():
        if len(losses) > 1:
            mean_loss = np.mean(losses)
            std_loss = np.std(losses)
            if mean_loss > 0:
                cv = std_loss / mean_loss  # Coefficient of Variation
                label_cvs.append(cv)
    
    avg_label_cv = np.mean(label_cvs) if label_cvs else 0.0
    
    metrics = {
        'label_wise_loss_variance': avg_label_variance,  # 낮을수록 좋음
        'label_wise_mean_loss': avg_label_mean_loss,
        'label_wise_std': avg_label_std,  # 낮을수록 좋음
        'label_consistency_cv': avg_label_cv,  # 낮을수록 일관적 (Coefficient of Variation)
        'num_labels_evaluated': len(label_losses)
    }
    
    print(f"\n{'='*60}")
    print(f"Generalization Metrics for {model_name}")
    print(f"{'='*60}")
    print(f"Label-wise Loss Variance: {avg_label_variance:.6f} (↓ better)")
    print(f"Label-wise Loss Std: {avg_label_std:.6f} (↓ better)")
    print(f"Label-wise Mean Loss: {avg_label_mean_loss:.4f}")
    print(f"Label Consistency (CV): {avg_label_cv:.4f} (↓ better)")
    print(f"Number of labels: {len(label_losses)}")
    print(f"{'='*60}\n")
    
    return metrics


def save_evaluation_results(original_metrics, unlearned_metrics, model_key, output_dir):
    """
    평가 결과 저장 및 비교
    
    Args:
        original_metrics: 원본 모델 메트릭
        unlearned_metrics: Unlearned 모델 메트릭
        model_key: 모델 키 (vilt, blip)
        output_dir: 출력 디렉토리
    """
    import json
    
    comparison = {
        'original': original_metrics,
        'unlearned': unlearned_metrics,
        'improvement': {}
    }
    
    # Compute improvement
    # Label-wise variance: 낮을수록 좋음 (감소율)
    variance_improvement = (original_metrics['label_wise_loss_variance'] - 
                           unlearned_metrics['label_wise_loss_variance']) / \
                           (original_metrics['label_wise_loss_variance'] + 1e-10) * 100
    
    # Label consistency (CV): 낮을수록 좋음 (감소율)
    cv_improvement = (original_metrics['label_consistency_cv'] - 
                      unlearned_metrics['label_consistency_cv']) / \
                     (original_metrics['label_consistency_cv'] + 1e-10) * 100
    
    comparison['improvement'] = {
        'label_variance_reduction': f"{variance_improvement:.2f}%",
        'consistency_cv_reduction': f"{cv_improvement:.2f}%"
    }
    
    # Save to JSON
    save_path = os.path.join(output_dir, model_key, 'evaluation_metrics.json')
    with open(save_path, 'w') as f:
        json.dump(comparison, f, indent=4)
    
    print(f"✓ Evaluation results saved to {save_path}")
    
    # Print comparison
    print(f"\n{'='*60}")
    print(f"IMPROVEMENT SUMMARY for {model_key.upper()}")
    print(f"{'='*60}")
    print(f"Label Variance Reduction: {variance_improvement:+.2f}%")
    print(f"Label Consistency (CV) Reduction: {cv_improvement:+.2f}%")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description='Unlearning for Reference Models')
    parser.add_argument('--dataset_root', type=str, default='D:/VQA/cocoqa', help='Dataset root path')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers')
    parser.add_argument('--unlearn_epochs', type=int, default=10, help='Unlearning epochs')
    parser.add_argument('--lr', type=float, default=1e-5, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.0001, help='Weight decay')
    parser.add_argument('--lambda_unlearn', type=float, default=0.01, help='Importance regularization weight')
    parser.add_argument('--lambda_consistency', type=float, default=1.0, help='Label consistency weight')
    parser.add_argument('--forget_ratio', type=float, default=0.3, help='Ratio of data to unlearn')
    parser.add_argument('--output_dir', type=str, default='./unlearned_reference_models', help='Output directory')
    parser.add_argument('--models', type=str, default='vilt,blip', help='Models to unlearn (comma-separated): vilt, blip')
    parser.add_argument('--recompute_importance', action='store_true', help='Recompute importance dict even if it exists')
    parser.add_argument('--recompute_metrics', action='store_true', help='Recompute original metrics even if it exists')
    
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Model names
    model_dict = {
        'vilt': 'dandelin/vilt-b32-finetuned-vqa',
        'blip': 'Salesforce/blip-vqa-base'
    }
    
    selected_models = [m.strip() for m in args.models.split(',')]
    models_to_unlearn = {k: v for k, v in model_dict.items() if k in selected_models}
    
    print(f"\n{'='*60}")
    print(f"UNLEARNING REFERENCE MODELS FOR DIFFICULTY CALIBRATION")
    print(f"{'='*60}")
    print(f"Dataset: {args.dataset_root}")
    print(f"Forget ratio: {args.forget_ratio}")
    print(f"Models: {list(models_to_unlearn.values())}")
    print(f"Output: {args.output_dir}")
    print(f"{'='*60}\n")
    
    # === Load data ===
    image_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225])
    ])
    
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    collate_fn = partial(collate_fn_with_tokenizer, tokenizer=tokenizer)
    
    full_dataset = VQADataset(root_dir=args.dataset_root, split='train', transform=image_transform)
    
    # Split into forget and retain
    total_size = len(full_dataset)
    forget_size = int(total_size * args.forget_ratio)
    retain_size = total_size - forget_size
    
    print(f"Dataset Split -> Forget: {forget_size} samples, Retain: {retain_size} samples\n")
    
    forget_dataset, _ = random_split(full_dataset, [forget_size, retain_size])
    
    forget_loader = DataLoader(
        dataset=forget_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=args.num_workers
    )
    
    # === Unlearn each model ===
    for model_key, model_name in models_to_unlearn.items():
        print(f"\n{'='*60}")
        print(f"Processing {model_name}")
        print(f"{'='*60}\n")
        
        # Load model
        if "vilt" in model_name.lower():
            processor = ViltProcessor.from_pretrained(model_name)
            original_model = ViltForQuestionAnswering.from_pretrained(model_name, use_safetensors=True).to(device)
        elif "blip" in model_name.lower():
            processor = BlipProcessor.from_pretrained(model_name)
            original_model = BlipForQuestionAnswering.from_pretrained(model_name).to(device)
        
        print(f"✓ Model loaded")
        
        # === Evaluate BEFORE unlearning ===
        original_metrics_path = os.path.join(args.output_dir, model_key, 'original_metrics.json')
        if os.path.exists(original_metrics_path) and not args.recompute_metrics:
            print(f"\n>>> Loading pre-computed ORIGINAL metrics...")
            original_metrics = load_metrics(original_metrics_path)
            print(f"\nOriginal Model Metrics:")
            print(f"  Label-wise Loss Variance: {original_metrics['label_wise_loss_variance']:.6f}")
            print(f"  Label Consistency (CV): {original_metrics['label_consistency_cv']:.4f}")
        else:
            print(f"\n>>> Evaluating ORIGINAL model...")
            original_metrics = evaluate_generalization(
                original_model, processor, model_name, forget_loader,
                args.dataset_root, 'train', device
            )
            # Save for future use
            save_metrics(original_metrics, original_metrics_path)
        
        # Clone model for unlearning
        model = copy.deepcopy(original_model)
        
        # Compute or load weight importance
        importance_path = os.path.join(args.output_dir, model_key, 'importance_dict.pkl')
        if os.path.exists(importance_path) and not args.recompute_importance:
            print(f"Loading pre-computed importance dict...")
            importance_dict = load_importance_dict(importance_path, device)
        else:
            print(f"Computing weight importance...")
            importance_dict = compute_weight_importance_mas(
                model, forget_loader, processor, model_name, 
                args.dataset_root, 'train', device
            )
            # Save for future use
            save_importance_dict(importance_dict, importance_path)
        
        # Save initial parameters
        initial_params = {}
        for name, param in model.named_parameters():
            initial_params[name] = param.data.clone()
        
        # Perform unlearning
        unlearning_reference_model(
            model, processor, model_name, forget_loader,
            importance_dict, initial_params, args, device
        )
        
        # === Evaluate AFTER unlearning ===
        print(f"\n>>> Evaluating UNLEARNED model...")
        unlearned_metrics = evaluate_generalization(
            model, processor, model_name, forget_loader,
            args.dataset_root, 'train', device
        )
        
        # Save unlearned model
        save_dir = os.path.join(args.output_dir, model_key)
        os.makedirs(save_dir, exist_ok=True)
        
        model.save_pretrained(save_dir)
        processor.save_pretrained(save_dir)
        
        print(f"\n✓ Unlearned {model_name} saved to {save_dir}")
        
        # Save evaluation comparison
        save_evaluation_results(original_metrics, unlearned_metrics, model_key, args.output_dir)
        
        # Clean up
        del model, original_model, processor, importance_dict, initial_params
        torch.cuda.empty_cache()
    
    print(f"\n{'='*60}")
    print(f"All models unlearned successfully!")
    print(f"Results saved to {args.output_dir}")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()

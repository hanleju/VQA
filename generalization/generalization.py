"""
python generalization.py --models vilt --batch_size 32 --epochs 2 --lr 1e-6 --lambda_consistency 0.5
"""
import os
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from transformers import (BlipProcessor, BlipForQuestionAnswering, 
                          ViltProcessor, ViltForQuestionAnswering,
                          BertTokenizer)
from functools import partial
from tqdm import tqdm
import numpy as np
from PIL import Image
import argparse

from data import VQADataset, collate_fn_with_tokenizer
from src import (
    compute_weight_importance_mas, compute_difficulty_aware_consistency_loss, 
    save_importance_dict, load_importance_dict, save_metrics, load_metrics, save_evaluation_results)

torch.manual_seed(42)

def train_consistency_model(model, processor, model_name, dataloader, 
                            importance_dict, initial_params, args, device):
    """
    Difficulty-aware consistency training
    """
    model.train()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()
    
    lambda_reg = args.lambda_reg
    lambda_consistency = args.lambda_consistency
    
    print(f"\n{'='*60}")
    print(f"Training Difficulty-Aware Consistency for {model_name}")
    print(f"{'='*60}")
    print(f"Lambda (consistency): {lambda_consistency}")
    print(f"Lambda (regularization): {lambda_reg}")
    print(f"Learning rate: {args.lr}")
    print(f"Epochs: {args.epochs}")
    print(f"{'='*60}\n")
    
    for epoch in range(args.epochs):
        epoch_total_loss = 0.0
        epoch_task_loss = 0.0
        epoch_cons_loss = 0.0
        epoch_reg_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")
        
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
                answer_texts = batch['answer_text'][:len(images)] if 'answer_text' in batch else [str(a.item()) for a in answers]
                
                # === 1. Task Loss (VQA training) ===
                model.train()
                if "blip" in model_name.lower():
                    inputs = processor(images=images, text=questions, return_tensors="pt", 
                                     padding=True, truncation=True).to(device)
                    labels = processor(text=answer_texts, return_tensors="pt", padding=True, truncation=True).input_ids.to(device)
                    outputs = model(**inputs, labels=labels)
                    task_loss = outputs.loss
                    cons_loss = torch.tensor(0.0, device=device)  # BLIP: no logits
                    
                elif "vilt" in model_name.lower():
                    inputs = processor(images=images, text=questions, return_tensors="pt", 
                                     padding=True, truncation=True, max_length=40).to(device)
                    outputs = model(**inputs)
                    logits = outputs.logits
                    
                    # Task loss
                    task_loss = criterion(logits, answers)
                    
                    # === 2. Difficulty-Aware Consistency Loss ===
                    cons_loss = compute_difficulty_aware_consistency_loss(logits, answers)
                
                # === 3. Weight Importance Regularization ===
                reg_loss = torch.tensor(0.0, device=device)
                for name, param in model.named_parameters():
                    if name in importance_dict and name in initial_params:
                        omega = importance_dict[name]
                        param_change = (param - initial_params[name]) ** 2
                        reg_loss += torch.sum(omega * param_change)
                
                # === 4. Combined Loss ===
                total_loss = task_loss + lambda_consistency * cons_loss + lambda_reg * reg_loss
                
                # Backward and optimize
                optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                # Accumulate losses
                epoch_total_loss += total_loss.item()
                epoch_task_loss += task_loss.item()
                epoch_cons_loss += cons_loss.item()
                epoch_reg_loss += reg_loss.item()
                num_batches += 1
                
                pbar.set_postfix({
                    'Total': f'{total_loss.item():.4f}',
                    'Task': f'{task_loss.item():.4f}',
                    'Cons': f'{cons_loss.item():.4f}',
                    'Reg': f'{reg_loss.item():.4f}'
                })
                
            except Exception as e:
                print(f"\n⚠ Error in batch: {e}")
                continue
        
        # Epoch summary
        if num_batches > 0:
            avg_total = epoch_total_loss / num_batches
            avg_task = epoch_task_loss / num_batches
            avg_cons = epoch_cons_loss / num_batches
            avg_reg = epoch_reg_loss / num_batches
            
            print(f"Epoch {epoch+1}/{args.epochs}: "
                  f"Total: {avg_total:.4f}, Task: {avg_task:.4f}, "
                  f"Cons: {avg_cons:.4f}, Reg: {avg_reg:.4f}")


def evaluate_difficulty_consistency(model, processor, model_name, dataloader, 
                                   dataset_root, split, device):
    """
    난이도별 일관성 평가
    
    핵심 지표:
    1. Overall metrics: 전체 일관성
    2. Difficulty-wise metrics: 난이도별 일관성
    3. Difficulty separation: 난이도 그룹 간 분리도
    
    Returns:
        metrics: 평가 지표 dict
    """
    model.eval()
    
    # 데이터 수집
    all_losses = []
    all_labels = []
    label_losses = {}
    
    criterion = nn.CrossEntropyLoss(reduction='none')
    
    print(f"Evaluating difficulty-aware consistency for {model_name}...")
    
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
                
                # Forward pass
                if "blip" in model_name.lower():
                    inputs = processor(images=images, text=questions, return_tensors="pt", 
                                     padding=True, truncation=True).to(device)
                    labels = processor(text=answer_texts, return_tensors="pt", padding=True, truncation=True).input_ids.to(device)
                    outputs = model(**inputs, labels=labels)
                    batch_loss = outputs.loss.item()
                    losses_batch = np.array([batch_loss] * len(images))
                    
                elif "vilt" in model_name.lower():
                    inputs = processor(images=images, text=questions, return_tensors="pt", 
                                     padding=True, truncation=True, max_length=40).to(device)
                    outputs = model(**inputs)
                    logits = outputs.logits
                    losses_batch = criterion(logits, answers).cpu().numpy()
                
                # 수집
                all_losses.extend(losses_batch)
                all_labels.extend(answers.cpu().numpy())
                
                # 레이블별 그룹화
                for loss, answer in zip(losses_batch, answers.cpu().numpy()):
                    answer = int(answer)
                    if answer not in label_losses:
                        label_losses[answer] = []
                    label_losses[answer].append(loss)
                
            except Exception as e:
                print(f"\n⚠ Error: {e}")
                continue
    
    all_losses = np.array(all_losses)
    all_labels = np.array(all_labels)
    
    # === Overall metrics ===
    
    # 1. Label-wise variance
    label_variances = []
    label_means = []
    label_stds = []
    label_cvs = []
    
    for label, losses in label_losses.items():
        if len(losses) > 1:
            label_variances.append(np.var(losses))
            label_means.append(np.mean(losses))
            label_stds.append(np.std(losses))
            
            mean_loss = np.mean(losses)
            std_loss = np.std(losses)
            if mean_loss > 0:
                cv = std_loss / mean_loss
                label_cvs.append(cv)
    
    # === Difficulty-wise metrics ===
    
    # Percentile로 난이도 그룹 분할
    p33 = np.percentile(all_losses, 33)
    p66 = np.percentile(all_losses, 66)
    
    difficulty_groups = np.zeros(len(all_losses), dtype=int)
    difficulty_groups[all_losses >= p33] = 1
    difficulty_groups[all_losses >= p66] = 2
    
    difficulty_names = ['Easy', 'Medium', 'Hard']
    difficulty_metrics = {}
    
    for diff_idx in [0, 1, 2]:
        mask = (difficulty_groups == diff_idx)
        if mask.sum() > 1:
            diff_losses = all_losses[mask]
            diff_labels = all_labels[mask]
            
            # 이 난이도 내에서 레이블별 일관성
            diff_label_losses = {}
            for loss, label in zip(diff_losses, diff_labels):
                label = int(label)
                if label not in diff_label_losses:
                    diff_label_losses[label] = []
                diff_label_losses[label].append(loss)
            
            # 일관성 지표
            diff_cvs = []
            for label, losses in diff_label_losses.items():
                if len(losses) > 1:
                    mean_loss = np.mean(losses)
                    std_loss = np.std(losses)
                    if mean_loss > 0:
                        cv = std_loss / mean_loss
                        diff_cvs.append(cv)
            
            difficulty_metrics[difficulty_names[diff_idx]] = {
                'num_samples': int(mask.sum()),
                'mean_loss': float(diff_losses.mean()),
                'std_loss': float(diff_losses.std()),
                'consistency_cv': float(np.mean(diff_cvs)) if diff_cvs else 0.0
            }
    
    # === Difficulty separation ===
    # 난이도 그룹 간 loss 차이
    easy_losses = all_losses[difficulty_groups == 0]
    medium_losses = all_losses[difficulty_groups == 1]
    hard_losses = all_losses[difficulty_groups == 2]
    
    separation = {
        'easy_mean': float(easy_losses.mean()) if len(easy_losses) > 0 else 0.0,
        'medium_mean': float(medium_losses.mean()) if len(medium_losses) > 0 else 0.0,
        'hard_mean': float(hard_losses.mean()) if len(hard_losses) > 0 else 0.0,
    }
    
    metrics = {
        'overall': {
            'label_wise_loss_variance': float(np.mean(label_variances)) if label_variances else 0.0,
            'label_wise_mean_loss': float(np.mean(label_means)) if label_means else 0.0,
            'label_wise_std': float(np.mean(label_stds)) if label_stds else 0.0,
            'label_consistency_cv': float(np.mean(label_cvs)) if label_cvs else 0.0,
            'num_labels_evaluated': int(len(label_losses))
        },
        'difficulty_wise': difficulty_metrics,
        'difficulty_separation': separation
    }
    
    # Print
    print(f"\n{'='*60}")
    print(f"Difficulty-Aware Consistency Metrics for {model_name}")
    print(f"{'='*60}")
    print(f"Overall:")
    print(f"  Label-wise Variance: {metrics['overall']['label_wise_loss_variance']:.6f}")
    print(f"  Label Consistency (CV): {metrics['overall']['label_consistency_cv']:.4f}")
    print(f"  Mean Loss: {metrics['overall']['label_wise_mean_loss']:.4f}")
    print(f"\nDifficulty-wise:")
    for diff_name, diff_metrics in difficulty_metrics.items():
        print(f"  {diff_name}:")
        print(f"    Samples: {diff_metrics['num_samples']}")
        print(f"    Mean Loss: {diff_metrics['mean_loss']:.4f}")
        print(f"    Consistency (CV): {diff_metrics['consistency_cv']:.4f}")
    print(f"\nDifficulty Separation:")
    print(f"  Easy: {separation['easy_mean']:.4f}")
    print(f"  Medium: {separation['medium_mean']:.4f}")
    print(f"  Hard: {separation['hard_mean']:.4f}")
    print(f"{'='*60}\n")
    
    return metrics

def main():
    parser = argparse.ArgumentParser(description='Difficulty-Aware Consistency Training for VQA')
    parser.add_argument('--dataset_root', type=str, default='D:/VQA/cocoqa', help='Dataset root path')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size (larger = better consistency)')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers')
    parser.add_argument('--epochs', type=int, default=2, help='Training epochs (짧게: 2-3)')
    parser.add_argument('--lr', type=float, default=1e-6, help='Learning rate (매우 낮게)')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay')
    parser.add_argument('--lambda_reg', type=float, default=0.1, help='Weight importance regularization')
    parser.add_argument('--lambda_consistency', type=float, default=0.5, help='Difficulty-aware consistency weight')
    parser.add_argument('--aux_ratio', type=float, default=0.3, help='Ratio of auxiliary data')
    parser.add_argument('--output_dir', type=str, default='./consistency/consistency_models', help='Output directory')
    parser.add_argument('--models', type=str, default='vilt', help='Models to train (comma-separated): vilt, blip')
    parser.add_argument('--recompute_importance', action='store_true', help='Recompute importance dict')
    parser.add_argument('--recompute_metrics', action='store_true', help='Recompute original metrics')
    
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
    models_to_train = {k: v for k, v in model_dict.items() if k in selected_models}
    
    print(f"\n{'='*60}")
    print(f"DIFFICULTY-AWARE CONSISTENCY TRAINING")
    print(f"{'='*60}")
    print(f"Dataset: {args.dataset_root}")
    print(f"Auxiliary ratio: {args.aux_ratio}")
    print(f"Models: {list(models_to_train.values())}")
    print(f"Output: {args.output_dir}")
    print(f"Epochs: {args.epochs}, LR: {args.lr}")
    print(f"Lambda (consistency): {args.lambda_consistency}, Lambda (reg): {args.lambda_reg}")
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
    
    # Split into auxiliary and retain
    total_size = len(full_dataset)
    aux_size = int(total_size * args.aux_ratio)
    retain_size = total_size - aux_size
    
    print(f"Dataset Split -> Auxiliary: {aux_size} samples, Retain: {retain_size} samples\n")
    
    aux_dataset, _ = random_split(full_dataset, [aux_size, retain_size])
    
    aux_loader = DataLoader(
        dataset=aux_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=args.num_workers
    )
    
    # === Train each model ===
    for model_key, model_name in models_to_train.items():
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
        
        # === Evaluate BEFORE training ===
        original_metrics_path = os.path.join(args.output_dir, model_key, 'original_metrics.json')
        if os.path.exists(original_metrics_path) and not args.recompute_metrics:
            print(f"\n>>> Loading pre-computed ORIGINAL metrics...")
            original_metrics = load_metrics(original_metrics_path)
        else:
            print(f"\n>>> Evaluating ORIGINAL model...")
            original_metrics = evaluate_difficulty_consistency(
                original_model, processor, model_name, aux_loader,
                args.dataset_root, 'train', device
            )
            save_metrics(original_metrics, original_metrics_path)
        
        # Clone model for training
        model = copy.deepcopy(original_model)
        
        # Compute or load weight importance
        importance_path = os.path.join(args.output_dir, model_key, 'importance_dict.pkl')
        if os.path.exists(importance_path) and not args.recompute_importance:
            print(f"Loading pre-computed importance dict...")
            importance_dict = load_importance_dict(importance_path, device)
        else:
            print(f"Computing weight importance...")
            importance_dict = compute_weight_importance_mas(
                model, aux_loader, processor, model_name, 
                args.dataset_root, 'train', device
            )
            save_importance_dict(importance_dict, importance_path)
        
        # Save initial parameters
        initial_params = {}
        for name, param in model.named_parameters():
            initial_params[name] = param.data.clone()
        
        # Perform consistency training
        train_consistency_model(
            model, processor, model_name, aux_loader,
            importance_dict, initial_params, args, device
        )
        
        # Save trained model
        save_dir = os.path.join(args.output_dir, model_key)
        os.makedirs(save_dir, exist_ok=True)
        
        model_save_path = os.path.join(save_dir, 'pytorch_model.pth')
        torch.save(model.state_dict(), model_save_path)
        print(f"✓ Model state_dict saved to {model_save_path}")
        
        try:
            processor.save_pretrained(save_dir)
            print(f"✓ Processor saved to {save_dir}")
        except Exception as e:
            print(f"⚠ Processor save failed: {e}")
        
        print(f"\n✓ Consistency-trained {model_name} saved to {save_dir}")
        
        # === Evaluate AFTER training ===
        print(f"\n>>> Evaluating TRAINED model...")
        trained_metrics = evaluate_difficulty_consistency(
            model, processor, model_name, aux_loader,
            args.dataset_root, 'train', device
        )
        
        # Save evaluation comparison
        save_evaluation_results(original_metrics, trained_metrics, model_key, args.output_dir)
        
        # Clean up
        del model, original_model, processor, importance_dict, initial_params
        torch.cuda.empty_cache()
    
    print(f"\n{'='*60}")
    print(f"All models trained successfully!")
    print(f"Results saved to {args.output_dir}")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()

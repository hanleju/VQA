"""
Simple Fine-tuning for VQA models
복잡한 consistency loss 없이 기본 CrossEntropyLoss만 사용하여 fine-tuning

사용법:
python ./generalization/finetune.py --models vilt --batch_size 8 --epochs 3 --lr 1e-5
"""
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from transformers import (ViltProcessor, ViltForQuestionAnswering,
                          BertTokenizer)
from functools import partial
from tqdm import tqdm
import numpy as np
from PIL import Image
import argparse
import json

from data import VQADataset, collate_fn_with_tokenizer

torch.manual_seed(42)


def train_simple_finetune(model, processor, model_name, dataloader, 
                         args, device, split='test'):
    """
    Simple fine-tuning with CrossEntropyLoss only
    
    Args:
        model: VQA model
        processor: Model processor
        model_name: Model name
        dataloader: Training data loader
        args: Arguments
        device: Device
        split: Dataset split ('train' or 'test')
    """
    model.train()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()
    
    print(f"\n{'='*60}")
    print(f"Fine-tuning {model_name}")
    print(f"{'='*60}")
    print(f"Learning rate: {args.lr}")
    print(f"Weight decay: {args.weight_decay}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"{'='*60}\n")
    
    for epoch in range(args.epochs):
        epoch_loss = 0.0
        num_batches = 0
        correct = 0
        total = 0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")
        
        for batch in pbar:
            try:
                # 이미지 로드
                image_paths = []
                for img_id in batch['image_id']:
                    base_path = os.path.join(args.dataset_root, split, 'images', img_id)
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
                
                # Forward pass
                inputs = processor(images=images, text=questions, return_tensors="pt", 
                                 padding=True, truncation=True, max_length=40).to(device)
                outputs = model(**inputs)
                logits = outputs.logits
                
                # Loss
                loss = criterion(logits, answers)
                
                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                # Statistics
                epoch_loss += loss.item()
                num_batches += 1
                
                _, predicted = torch.max(logits, 1)
                total += answers.size(0)
                correct += (predicted == answers).sum().item()
                
                pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{100 * correct / total:.2f}%'
                })
                
            except Exception as e:
                print(f"\n⚠ Error in batch: {e}")
                continue
        
        # Epoch summary
        if num_batches > 0:
            avg_loss = epoch_loss / num_batches
            accuracy = 100 * correct / total
            
            print(f"Epoch {epoch+1}/{args.epochs}: "
                  f"Loss: {avg_loss:.4f}, "
                  f"Accuracy: {accuracy:.2f}%")


def evaluate_model(model, processor, model_name, dataloader, 
                   dataset_root, split, device):
    """
    Evaluate model on given data
    
    Returns:
        metrics: Dictionary with evaluation metrics
    """
    model.eval()
    criterion = nn.CrossEntropyLoss()
    
    total_loss = 0.0
    correct = 0
    total = 0
    num_batches = 0
    
    print(f"\nEvaluating {model_name}...")
    
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
                
                if len(image_paths) == 0:
                    continue
                
                images = [Image.open(path).convert('RGB') for path in image_paths]
                questions = batch['question'][:len(images)]
                answers = batch['answer'][:len(images)].to(device)
                
                # Forward pass
                if "vilt" in model_name.lower():
                    inputs = processor(images=images, text=questions, return_tensors="pt", 
                                     padding=True, truncation=True, max_length=40).to(device)
                    outputs = model(**inputs)
                    logits = outputs.logits
                
                # Loss
                loss = criterion(logits, answers)
                total_loss += loss.item()
                num_batches += 1
                
                # Accuracy
                _, predicted = torch.max(logits, 1)
                total += answers.size(0)
                correct += (predicted == answers).sum().item()
                
            except Exception as e:
                print(f"\n⚠ Error: {e}")
                continue
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    accuracy = 100 * correct / total if total > 0 else 0.0
    
    metrics = {
        'loss': float(avg_loss),
        'accuracy': float(accuracy),
        'num_samples': int(total)
    }
    
    print(f"\n{'='*60}")
    print(f"Evaluation Results for {model_name}")
    print(f"{'='*60}")
    print(f"Loss: {avg_loss:.4f}")
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"Samples: {total}")
    print(f"{'='*60}\n")
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description='Simple Fine-tuning for VQA')
    parser.add_argument('--dataset_root', type=str, default='D:/VQA/cocoqa', help='Dataset root path')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers')
    parser.add_argument('--epochs', type=int, default=3, help='Training epochs')
    parser.add_argument('--lr', type=float, default=1e-5, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay')
    parser.add_argument('--train_ratio', type=float, default=0.7, help='Ratio of training data')
    parser.add_argument('--output_dir', type=str, default='./generalization/output_finetune', help='Output directory')
    parser.add_argument('--models', type=str, default='vilt', help='Models to train (comma-separated): vilt')
    
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Model names
    model_dict = {
        'vilt': 'dandelin/vilt-b32-finetuned-vqa'
    }
    
    selected_models = [m.strip() for m in args.models.split(',')]
    models_to_train = {k: v for k, v in model_dict.items() if k in selected_models}
    
    print(f"\n{'='*60}")
    print(f"SIMPLE FINE-TUNING")
    print(f"{'='*60}")
    print(f"Dataset: {args.dataset_root}")
    print(f"Train ratio: {args.train_ratio}")
    print(f"Models: {list(models_to_train.values())}")
    print(f"Output: {args.output_dir}")
    print(f"Epochs: {args.epochs}, LR: {args.lr}")
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
    
    full_dataset = VQADataset(root_dir=args.dataset_root, split='test', transform=image_transform)
    
    # Split into train and eval
    total_size = len(full_dataset)
    train_size = int(total_size * args.train_ratio)
    eval_size = total_size - train_size
    
    print(f"Dataset Split -> Train: {train_size} samples, Eval: {eval_size} samples\n")
    
    train_dataset, eval_dataset = random_split(full_dataset, [train_size, eval_size])
    
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=args.num_workers
    )
    
    eval_loader = DataLoader(
        dataset=eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
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
            model = ViltForQuestionAnswering.from_pretrained(model_name, use_safetensors=True).to(device)
        
        print(f"✓ Model loaded")
        
        # === Evaluate BEFORE fine-tuning ===
        print(f"\n>>> Evaluating ORIGINAL model on EVAL data...")
        original_metrics = evaluate_model(
            model, processor, model_name, eval_loader,
            args.dataset_root, 'test', device
        )
        
        # === Fine-tune ===
        print(f"\n>>> Fine-tuning on TRAIN data...")
        train_simple_finetune(
            model, processor, model_name, train_loader,
            args, device, split='test'
        )
        
        # === Evaluate AFTER fine-tuning ===
        print(f"\n>>> Evaluating FINE-TUNED model on EVAL data...")
        finetuned_metrics = evaluate_model(
            model, processor, model_name, eval_loader,
            args.dataset_root, 'test', device
        )
        
        # Save model
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
        
        # Save metrics comparison
        comparison = {
            'original': original_metrics,
            'finetuned': finetuned_metrics,
            'improvement': {
                'loss_change': finetuned_metrics['loss'] - original_metrics['loss'],
                'accuracy_change': finetuned_metrics['accuracy'] - original_metrics['accuracy']
            }
        }
        
        metrics_path = os.path.join(save_dir, 'evaluation_metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(comparison, f, indent=2)
        print(f"✓ Evaluation metrics saved to {metrics_path}")
        
        # Print comparison
        print(f"\n{'='*60}")
        print(f"Metrics Comparison for {model_name}")
        print(f"{'='*60}")
        print(f"Original  -> Loss: {original_metrics['loss']:.4f}, Acc: {original_metrics['accuracy']:.2f}%")
        print(f"Finetuned -> Loss: {finetuned_metrics['loss']:.4f}, Acc: {finetuned_metrics['accuracy']:.2f}%")
        print(f"Change    -> Loss: {comparison['improvement']['loss_change']:+.4f}, Acc: {comparison['improvement']['accuracy_change']:+.2f}%")
        print(f"{'='*60}\n")
        
        # Clean up
        del model, processor
        torch.cuda.empty_cache()
    
    print(f"\n{'='*60}")
    print(f"All models fine-tuned successfully!")
    print(f"Results saved to {args.output_dir}")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()

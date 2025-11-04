import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, random_split
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    auc,
    confusion_matrix,
)
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import yaml
import argparse
from functools import partial
from torchvision import transforms
from transformers import BertTokenizer
from tqdm import tqdm

from data.data import VQADataset, collate_fn_with_tokenizer
from model.vision_encoder import CNN, ResNet50, SwinTransformer
from model.text_encoder import Bert, RoBerta, BertQLoRA, RoBertaQLoRA
from model.model import VQAModel

torch.manual_seed(42)

def parse_args():
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument('--cfg', '-c', type=str, default=None, help='path to YAML config file')
    p.add_argument('--weights', '-w', type=str, help='path to model weights file')
    p.add_argument('--threshold', type=float, default=0.6, help='confidence threshold for membership')
    known, remaining = p.parse_known_args()

    cfg_from_file = {}
    if known.cfg:
        cfg_path = Path(known.cfg)
        if cfg_path.exists():
            with open(cfg_path, 'r', encoding='utf-8') as f:
                cfg_from_file = yaml.safe_load(f) or {}
        else:
            raise FileNotFoundError(f"Config file not found: {known.cfg}")
        
    args_obj = argparse.Namespace(**cfg_from_file)
    args_obj.weights = known.weights
    args_obj.threshold = known.threshold
    return args_obj

def get_confidence_and_pred(outputs):
    """
    소프트맥스 출력에서 confidence(최대 확률)와 예측 클래스를 반환
    """
    probs = torch.softmax(outputs, dim=1)
    confidence, predictions = torch.max(probs, dim=1)
    return confidence, predictions

# entropy/loss removed — attack uses confidence only

def evaluate_privacy(model, dataloader, device, threshold=0.6, is_member=True):
    """
    데이터셋에 대한 confidence 계산 (entropy/loss removed)
    Returns dict with confidences, ground_truth and predictions (by threshold).
    """
    model.eval()
    confidences = []
    ground_truth = []  # 실제 membership (1 for member, 0 for non-member)
    predictions = []   # 예측된 membership (True/False)

    with torch.no_grad():
        for batch in tqdm(dataloader):
            images = batch['image'].to(device)
            inputs = batch['inputs'].to(device)
            answers = batch['answer'].to(device)

            outputs = model(
                images=images,
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask']
            )

            batch_confidence, _ = get_confidence_and_pred(outputs)
            confidences.extend(batch_confidence.cpu().numpy())

            pred_member = batch_confidence.cpu().numpy() >= threshold
            predictions.extend(pred_member)
            ground_truth.extend([1 if is_member else 0] * len(images))

    return {
        'confidences': np.array(confidences),
        'ground_truth': np.array(ground_truth),
        'predictions': np.array(predictions)
    }

def plot_distributions(member_results, nonmember_results, save_dir):
    """
    member/nonmember data confidence distribution visualization (confidence only)
    """
    os.makedirs(save_dir, exist_ok=True)

    plt.figure(figsize=(10, 6))
    sns.kdeplot(member_results['confidences'], label='Member')
    sns.kdeplot(nonmember_results['confidences'], label='Non-member')
    plt.xlabel('Confidence')
    plt.ylabel('Density')
    plt.title('Confidence Distribution: Member vs Non-member')
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'confidence_distribution.png'))
    plt.close()

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    weight_name = Path(args.weights).stem
    result_dir = os.path.join(os.path.dirname(args.weights), 'privacy_analysis', weight_name)
    os.makedirs(result_dir, exist_ok=True)
    
    image_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225])
    ])
    
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    full_train_dataset = VQADataset(
        root_dir=args.dataset_root,
        split='train',
        transform=image_transform
    )

    test_dataset = VQADataset(
        root_dir=args.dataset_root,
        split='test',
        transform=image_transform
    )

    total_size = len(full_train_dataset)
    val_size = int(total_size * getattr(args, 'val_split_ratio', 0.1))
    train_size = total_size - val_size

    g = torch.Generator()
    g.manual_seed(42)
    train_dataset, _ = random_split(full_train_dataset, [train_size, val_size], generator=g)
    
    collate_fn = partial(collate_fn_with_tokenizer, tokenizer=tokenizer)
    
    train_len = len(train_dataset)
    test_len = len(test_dataset)
    desired = min(train_len, test_len)

    g2 = torch.Generator()
    g2.manual_seed(42)

    if train_len > desired:
        perm_train = torch.randperm(train_len, generator=g2)
        train_indices = perm_train[:desired].tolist()
        train_subset = Subset(train_dataset, train_indices)
    else:
        train_subset = train_dataset

    if test_len > desired:
        perm_test = torch.randperm(test_len, generator=g2)
        test_indices = perm_test[:desired].tolist()
        test_subset = Subset(test_dataset, test_indices)
    else:
        test_subset = test_dataset

    train_loader = DataLoader(
        dataset=train_subset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        dataset=test_subset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    VISION_MODELS = {
        "CNN": CNN,
        "ResNet50": ResNet50,
        "SwinTransformer": SwinTransformer
    }
    TEXT_MODELS = {
        "Bert": Bert,
        "RoBerta": RoBerta,
        "BertQLoRA": BertQLoRA, 
        "RoBertaQLoRA": RoBertaQLoRA
    }
    
    vision_class = VISION_MODELS.get(args.Vision)
    text_class = TEXT_MODELS.get(args.Text)
    
    model = VQAModel(
        vision=vision_class,
        text=text_class,
        fusion_type=args.fusion_type,
        num_classes=args.num_classes
    ).to(device)
    
    model.load_state_dict(torch.load(args.weights, map_location=device))
    
    print("Evaluating member (train) data...")
    member_results = evaluate_privacy(
        model, train_loader, device,
        threshold=args.threshold, is_member=True
    )
    
    print("Evaluating non-member (test) data...")
    nonmember_results = evaluate_privacy(
        model, test_loader, device,
        threshold=args.threshold, is_member=False
    )

    # visualization
    plot_distributions(member_results, nonmember_results, result_dir)

    scores_member = member_results['confidences']
    scores_nonmember = nonmember_results['confidences']
    scores = np.concatenate([scores_member, scores_nonmember])
    labels = np.concatenate([member_results['ground_truth'], nonmember_results['ground_truth']])

    # ROC / AUC
    try:
        roc_auc = roc_auc_score(labels, scores)
        fpr, tpr, _ = roc_curve(labels, scores)
        plt.figure()
        plt.plot(fpr, tpr, label=f'ROC AUC={roc_auc:.4f}')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('FPR'); plt.ylabel('TPR'); plt.title('ROC Curve')
        plt.legend()
        plt.savefig(os.path.join(result_dir, 'roc_curve.png'))
        plt.close()
    except Exception as e:
        roc_auc = None

    # PR curve
    try:
        prec_vals, recall_vals, _ = precision_recall_curve(labels, scores)
        pr_auc = auc(recall_vals, prec_vals)
        plt.figure()
        plt.plot(recall_vals, prec_vals, label=f'PR AUC={pr_auc:.4f}')
        plt.xlabel('Recall'); plt.ylabel('Precision'); plt.title('Precision-Recall Curve')
        plt.legend()
        plt.savefig(os.path.join(result_dir, 'pr_curve.png'))
        plt.close()
    except Exception:
        pr_auc = None

    preds_at_thresh = (scores >= args.threshold).astype(int)
    cm = confusion_matrix(labels, preds_at_thresh)
    plt.figure(figsize=(4, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted'); plt.ylabel('True'); plt.title(f'Confusion Matrix @thr={args.threshold:.2f}')
    plt.savefig(os.path.join(result_dir, f'confusion_at_{args.threshold:.2f}.png'))
    plt.close()

    accuracy = accuracy_score(labels, preds_at_thresh)
    precision = precision_score(labels, preds_at_thresh, zero_division=0)
    recall = recall_score(labels, preds_at_thresh, zero_division=0)
    f1 = f1_score(labels, preds_at_thresh, zero_division=0)

    with open(os.path.join(result_dir, 'privacy_metrics.txt'), 'w') as f:
        f.write(f"Privacy Analysis Results for {weight_name}\n")
        f.write("-" * 50 + "\n")
        f.write(f"Confidence Threshold: {args.threshold}\n\n")
        if roc_auc is not None:
            f.write(f"ROC AUC: {roc_auc:.4f}\n")
        if pr_auc is not None:
            f.write(f"PR AUC: {pr_auc:.4f}\n")
        f.write("Membership Inference Attack Metrics (at threshold):\n")
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall: {recall:.4f}\n")
        f.write(f"F1 Score: {f1:.4f}\n\n")
        f.write("Average Confidence:\n")
        f.write(f"  Member: {np.mean(scores_member):.4f}\n")
        f.write(f"  Non-member: {np.mean(scores_nonmember):.4f}\n")

    print(f"\nResults saved to {result_dir}")
    if roc_auc is not None:
        print(f"ROC AUC: {roc_auc:.4f}")
    if pr_auc is not None:
        print(f"PR AUC: {pr_auc:.4f}")
    print(f"Attack Accuracy: {accuracy:.4f}")
    print(f"Attack Precision: {precision:.4f}")
    print(f"Attack Recall: {recall:.4f}")
    print(f"Attack F1 Score: {f1:.4f}")

if __name__ == '__main__':
    main()

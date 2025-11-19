"""
공통 유틸리티 함수들
Confidence 기반 및 Loss 기반 공격에서 공통으로 사용되는 함수들
utils/util.py의 함수들을 재사용하고 attack 전용 함수들만 정의
"""
import os
import argparse
import torch
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
from functools import partial
from torchvision import transforms
from transformers import BertTokenizer

from data.data import VQADataset, collate_fn_with_tokenizer
from utils.util import parse_args, create_model, load_weights


def parse_args_with_config(extra_args=None):
    """
    YAML 설정 파일과 커맨드라인 인자를 파싱
    utils/util.py의 parse_args를 확장하여 추가 인자를 받을 수 있도록 함
    
    Args:
        extra_args: 추가 커맨드라인 인자 리스트 [(name, type, default, help), ...]
    """
    # 기본 파서 생성
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument('--cfg', '-c', type=str, default=None, help='path to YAML config file')
    p.add_argument('--weights', '-w', type=str, help='path to model weights file')
    
    # 추가 인자가 있으면 파서에 추가
    if extra_args:
        for arg_name, arg_type, arg_default, arg_help in extra_args:
            p.add_argument(arg_name, type=arg_type, default=arg_default, help=arg_help)
    
    # 기본 args 파싱 (utils/util.py의 parse_args 재사용)
    args_obj = parse_args()
    
    # 추가 인자들을 args_obj에 추가
    if extra_args:
        known, _ = p.parse_known_args()
        for arg_name, _, _, _ in extra_args:
            setattr(args_obj, arg_name.lstrip('-'), getattr(known, arg_name.lstrip('-')))
    
    return args_obj


def setup_data_loaders(args, seed=42):
    """
    Train/Test 데이터로더 설정 (동일한 크기로 샘플링)
    
    Returns:
        train_loader, test_loader, tokenizer, image_transform
    """
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
    g.manual_seed(seed)
    train_dataset, _ = random_split(full_train_dataset, [train_size, val_size], generator=g)
    
    collate_fn = partial(collate_fn_with_tokenizer, tokenizer=tokenizer)
    
    train_len = len(train_dataset)
    test_len = len(test_dataset)
    desired = min(train_len, test_len)

    g2 = torch.Generator()
    g2.manual_seed(seed)

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
    
    return train_loader, test_loader, tokenizer, image_transform



def load_model(args, device):
    """
    YAML 설정에 따라 모델 로드
    utils/util.py의 create_model과 load_weights를 재사용
    
    Returns:
        model, model_type
    """
    # utils/util.py의 create_model 사용
    model = create_model(args, device)
    model_type = getattr(args, 'model', 'VQAModel')
    
    # utils/util.py의 load_weights 사용
    model = load_weights(model, args.weights, device)
    
    return model, model_type


def plot_roc_curve(labels, scores, save_path):
    """ROC Curve 시각화"""
    try:
        roc_auc = roc_auc_score(labels, scores)
        fpr, tpr, _ = roc_curve(labels, scores)
        plt.figure()
        plt.plot(fpr, tpr, label=f'ROC AUC={roc_auc:.4f}')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('FPR')
        plt.ylabel('TPR')
        plt.title('ROC Curve')
        plt.legend()
        plt.savefig(save_path)
        plt.close()
        return roc_auc
    except Exception as e:
        print(f"Error plotting ROC curve: {e}")
        return None


def plot_pr_curve(labels, scores, save_path):
    """Precision-Recall Curve 시각화"""
    try:
        prec_vals, recall_vals, _ = precision_recall_curve(labels, scores)
        pr_auc = auc(recall_vals, prec_vals)
        plt.figure()
        plt.plot(recall_vals, prec_vals, label=f'PR AUC={pr_auc:.4f}')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend()
        plt.savefig(save_path)
        plt.close()
        return pr_auc
    except Exception as e:
        print(f"Error plotting PR curve: {e}")
        return None


def plot_confusion_matrix(labels, predictions, threshold, save_path):
    """Confusion Matrix 시각화"""
    cm = confusion_matrix(labels, predictions)
    plt.figure(figsize=(4, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix @thr={threshold:.2f}')
    plt.savefig(save_path)
    plt.close()
    return cm


def calculate_metrics(labels, predictions):
    """분류 메트릭 계산"""
    accuracy = accuracy_score(labels, predictions)
    precision = precision_score(labels, predictions, zero_division=0)
    recall = recall_score(labels, predictions, zero_division=0)
    f1 = f1_score(labels, predictions, zero_division=0)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


def save_privacy_metrics(result_dir, weight_name, threshold, roc_auc, pr_auc, 
                        metrics, scores_member, scores_nonmember, metric_name="Confidence"):
    """프라이버시 분석 결과 저장"""
    with open(os.path.join(result_dir, 'privacy_metrics.txt'), 'w') as f:
        f.write(f"Privacy Analysis Results for {weight_name}\n")
        f.write("-" * 50 + "\n")
        f.write(f"{metric_name} Threshold: {threshold}\n\n")
        if roc_auc is not None:
            f.write(f"ROC AUC: {roc_auc:.4f}\n")
        if pr_auc is not None:
            f.write(f"PR AUC: {pr_auc:.4f}\n")
        f.write("Membership Inference Attack Metrics (at threshold):\n")
        f.write(f"Accuracy: {metrics['accuracy']:.4f}\n")
        f.write(f"Precision: {metrics['precision']:.4f}\n")
        f.write(f"Recall: {metrics['recall']:.4f}\n")
        f.write(f"F1 Score: {metrics['f1']:.4f}\n\n")
        f.write(f"Average {metric_name}:\n")
        f.write(f"  Member: {np.mean(scores_member):.4f}\n")
        f.write(f"  Non-member: {np.mean(scores_nonmember):.4f}\n")


def print_results(roc_auc, pr_auc, metrics):
    """결과 출력"""
    if roc_auc is not None:
        print(f"ROC AUC: {roc_auc:.4f}")
    if pr_auc is not None:
        print(f"PR AUC: {pr_auc:.4f}")
    print(f"Attack Accuracy: {metrics['accuracy']:.4f}")
    print(f"Attack Precision: {metrics['precision']:.4f}")
    print(f"Attack Recall: {metrics['recall']:.4f}")
    print(f"Attack F1 Score: {metrics['f1']:.4f}")

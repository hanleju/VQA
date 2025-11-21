"""
Minimal Difficulty Calibration 기반 Membership Inference Attack
기초적인 난이도 보정 기법: 분류기 학습 없이 calibrated loss로 threshold 기반 공격 수행

핵심 아이디어:
- Shadow model들로 샘플 난이도(평균 loss) 추정
- Target model의 loss에서 난이도를 빼 상대적 loss 계산
- Calibrated loss를 membership score로 사용 (낮을수록 member)
- Threshold 기반 이진 분류 (Logistic Regression 학습 제거)

사용법:
python ./attack/calibration.py -c ./cfg/baseline/SwinT_BERTLoRA_coco/concat.yaml -w ./checkpoints/baseline/SwinT_BERTLoRA_coco/concat/best_model.pth --threshold 0.5
"""
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# 프로젝트 루트를 sys.path에 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

from attack.metric_src import (
    parse_args_with_config, setup_data_loaders, load_model,
    plot_roc_curve, plot_pr_curve, plot_confusion_matrix,
    calculate_metrics, save_privacy_metrics, print_results
)
from attack.cali_src import (
    collect_target_losses, collect_shadow_difficulties, calibrate_loss
)

torch.manual_seed(42)
np.random.seed(42)

# ============================================================
# Threshold 기반 멤버십 예측
# (공통 함수들은 cali_src.py에서 import)
# ============================================================

def evaluate_calibrated_attack(calibrated_train_losses, calibrated_test_losses, threshold):
    """
    Calibrated loss를 기반으로 threshold 기반 공격 수행
    
    Args:
        calibrated_train_losses: calibrated train loss
        calibrated_test_losses: calibrated test loss
        threshold: membership threshold (이보다 낮으면 member)
        
    Returns:
        scores_member, scores_nonmember, labels, predictions
    """
    # Calibrated loss가 낮을수록 member
    # Score는 -calibrated_loss (높을수록 member)
    scores_member = -calibrated_train_losses
    scores_nonmember = -calibrated_test_losses
    
    scores = np.concatenate([scores_member, scores_nonmember])
    labels = np.concatenate([
        np.ones(len(scores_member)),
        np.zeros(len(scores_nonmember))
    ])
    
    # Threshold 적용: -calibrated_loss >= -threshold => calibrated_loss <= threshold
    predictions = (scores >= -threshold).astype(int)
    
    return scores_member, scores_nonmember, scores, labels, predictions


def plot_loss_distributions(train_losses, test_losses, output_dir, filename, title):
    """Loss 분포 시각화"""
    plt.figure(figsize=(10, 6))
    plt.hist(train_losses, bins=50, alpha=0.6, label='Member (Train)', color='blue', density=True)
    plt.hist(test_losses, bins=50, alpha=0.6, label='Non-member (Test)', color='red', density=True)
    plt.xlabel('Calibrated Loss', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(alpha=0.3)
    save_path = os.path.join(output_dir, filename)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Distribution plot saved: {save_path}")


def main():
    extra_args = [
        ('--threshold', float, 0.5, 'calibrated loss threshold (lower = member)'),
        ('--shadow_models', str, None, 'Comma-separated shadow model names'),
    ]
    args = parse_args_with_config(extra_args)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    weight_name = Path(args.weights).stem
    result_dir = os.path.join(os.path.dirname(args.weights), 'privacy_analysis', 'calibration')
    os.makedirs(result_dir, exist_ok=True)
    
    shadow_models = None
    if hasattr(args, 'shadow_models') and args.shadow_models:
        shadow_models = [m.strip() for m in args.shadow_models.split(',')]
    
    print(f"\n{'='*60}")
    print("MINIMAL DIFFICULTY CALIBRATION MIA")
    print("(No classifier training, threshold-based)")
    print(f"{'='*60}")
    print(f"Model: {args.model}")
    print(f"Weights: {args.weights}")
    print(f"Threshold: {args.threshold}")
    print(f"Output: {result_dir}")
    print(f"{'='*60}\n")
    
    # 1. Shadow model들로부터 난이도 추정
    print("\n=== Step 1: Estimating Sample Difficulty ===")
    train_difficulty, test_difficulty = collect_shadow_difficulties(args, device, shadow_models)
    print(f"Train difficulty - Mean: {train_difficulty.mean():.4f}, Std: {train_difficulty.std():.4f}")
    print(f"Test difficulty - Mean: {test_difficulty.mean():.4f}, Std: {test_difficulty.std():.4f}")
    
    # 2. Target model의 loss 수집
    print("\n=== Step 2: Collecting Target Model Losses ===")
    train_loader, test_loader, _, _ = setup_data_loaders(args, seed=42)
    model, model_type = load_model(args, device)
    
    target_train_losses = collect_target_losses(model, train_loader, device, model_type)
    target_test_losses = collect_target_losses(model, test_loader, device, model_type)
    print(f"Target train loss - Mean: {target_train_losses.mean():.4f}, Std: {target_train_losses.std():.4f}")
    print(f"Target test loss - Mean: {target_test_losses.mean():.4f}, Std: {target_test_losses.std():.4f}")
    
    # 3. Loss 보정
    print("\n=== Step 3: Calibrating Losses ===")
    calibrated_train_losses = calibrate_loss(target_train_losses, train_difficulty)
    calibrated_test_losses = calibrate_loss(target_test_losses, test_difficulty)
    print(f"Calibrated train loss - Mean: {calibrated_train_losses.mean():.4f}, Std: {calibrated_train_losses.std():.4f}")
    print(f"Calibrated test loss - Mean: {calibrated_test_losses.mean():.4f}, Std: {calibrated_test_losses.std():.4f}")
    
    # 4. Threshold 기반 공격 수행
    print("\n=== Step 4: Performing Threshold-based Attack ===")
    scores_member, scores_nonmember, scores, labels, predictions = evaluate_calibrated_attack(
        calibrated_train_losses, calibrated_test_losses, args.threshold
    )
    
    # 5. 평가 및 시각화
    print("\n=== Step 5: Evaluation and Visualization ===")
    
    # Distribution plot
    plot_loss_distributions(
        calibrated_train_losses, calibrated_test_losses,
        result_dir, "calibrated_loss_distribution.png",
        "Calibrated Loss Distribution: Member vs Non-member"
    )
    
    # ROC curve
    roc_auc = plot_roc_curve(labels, scores, os.path.join(result_dir, 'roc_curve.png'))
    
    # PR curve
    pr_auc = plot_pr_curve(labels, scores, os.path.join(result_dir, 'pr_curve.png'))
    
    # Confusion matrix
    plot_confusion_matrix(labels, predictions, args.threshold, 
                         os.path.join(result_dir, f'confusion_at_{args.threshold:.2f}.png'))
    
    # 메트릭 계산
    metrics = calculate_metrics(labels, predictions)
    
    # 결과 저장
    save_privacy_metrics(
        result_dir, weight_name, args.threshold, roc_auc, pr_auc,
        metrics, calibrated_train_losses, calibrated_test_losses,
        metric_name="Calibrated Loss"
    )
    
    print(f"\n{'='*60}")
    print("Results Summary")
    print(f"{'='*60}")
    print_results(roc_auc, pr_auc, metrics)
    print(f"\nAll results saved to: {result_dir}")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()

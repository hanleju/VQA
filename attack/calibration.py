"""
Minimal Difficulty Calibration 기반 Membership Inference Attack
기초적인 난이도 보정 기법: 분류기 학습 없이 calibrated loss로 threshold 기반 공격 수행

핵심 아이디어:
- Shadow model들로 샘플 난이도(평균 loss) 추정
- Target model의 loss에서 난이도를 빼 상대적 loss 계산
- Calibrated loss를 membership score로 사용 (낮을수록 member)
- Threshold 기반 이진 분류 (Logistic Regression 학습 제거)

python ./attack/calibration.py -c ./cfg/cocoqa/Res_Bert_Lora_dp.yaml -w ./checkpoints/cocoqa/Res_Bert_Lora_dp/best_model.pth --threshold 0.5

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


def run_threshold_sweep(calibrated_train_losses, calibrated_test_losses, result_dir):
    """0.1~0.9까지 threshold를 변경하며 공격 성능 평가"""
    thresholds = np.arange(-1.0, 3.0, 0.1)
    results = []
    
    print("\n=== Threshold Sweep Analysis ===")
    print(f"Testing thresholds: {thresholds}")
    
    # Score 계산 (calibrated loss가 낮을수록 member)
    scores_member = -calibrated_train_losses
    scores_nonmember = -calibrated_test_losses
    scores = np.concatenate([scores_member, scores_nonmember])
    labels = np.concatenate([
        np.ones(len(scores_member)),
        np.zeros(len(scores_nonmember))
    ])
    
    # 각 threshold에 대해 평가
    for threshold in thresholds:
        preds = (scores >= -threshold).astype(int)
        metrics = calculate_metrics(labels, preds)
        results.append({
            'threshold': threshold,
            'accuracy': metrics['accuracy'],
            'precision': metrics['precision'],
            'recall': metrics['recall'],
            'f1': metrics['f1']
        })
        print(f"Threshold {threshold:.1f}: Accuracy={metrics['accuracy']:.4f}, "
              f"Precision={metrics['precision']:.4f}, Recall={metrics['recall']:.4f}")
    
    # 결과를 txt로 저장
    txt_path = os.path.join(result_dir, 'threshold_sweep_results.txt')
    with open(txt_path, 'w') as f:
        f.write("Threshold Sweep Results\n")
        f.write("=" * 80 + "\n")
        f.write(f"{'Threshold':<12}{'Accuracy':<12}{'Precision':<12}{'Recall':<12}{'F1':<12}\n")
        f.write("=" * 80 + "\n")
        for r in results:
            f.write(f"{r['threshold']:<12.1f}{r['accuracy']:<12.4f}{r['precision']:<12.4f}"
                   f"{r['recall']:<12.4f}{r['f1']:<12.4f}\n")
    print(f"\nThreshold sweep results saved to: {txt_path}")
    
    # 막대그래프 생성
    fig, ax = plt.subplots(figsize=(12, 6))
    x = [r['threshold'] for r in results]
    accuracies = [r['accuracy'] for r in results]
    
    bars = ax.bar(x, accuracies, width=0.08, alpha=0.8, color='mediumseagreen', edgecolor='black')
    ax.set_xlabel('Threshold', fontsize=12, fontweight='bold')
    ax.set_ylabel('Attack Accuracy', fontsize=12, fontweight='bold')
    ax.set_title('Calibration-based MIA: Attack Accuracy vs Threshold', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{t:.1f}' for t in x])
    ax.set_ylim([0, 1.0])
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # 각 막대 위에 값 표시
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{acc:.3f}',
                ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plot_path = os.path.join(result_dir, 'threshold_sweep_accuracy.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Threshold sweep plot saved to: {plot_path}")
    
    return results


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
        ('--sweep_thresholds', int, 0, 'Run threshold sweep from 0.0 to 3.0 with step 0.3 (1=True, 0=False)'),
        ('--shadow_models', str, None, 'Comma-separated shadow model names (blip, vilt, git)'),
        ('--reference_weights', str, None, 'Path to reference model weights (for fine-tuned models)'),
        ('--output_dir', str, 'calibration', 'Output directory name under privacy_analysis/'),
    ]
    args = parse_args_with_config(extra_args)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    weight_name = Path(args.weights).stem
    output_subdir = getattr(args, 'output_dir', 'calibration')
    result_dir = os.path.join(os.path.dirname(args.weights), 'privacy_analysis', output_subdir)
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
    reference_weights = getattr(args, 'reference_weights', None)
    train_difficulty, test_difficulty = collect_shadow_difficulties(args, device, shadow_models, reference_weights)
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
    
    # Threshold sweep 모드
    if hasattr(args, 'sweep_thresholds') and args.sweep_thresholds:
        print("\n*** Threshold Sweep Mode ***")
        run_threshold_sweep(calibrated_train_losses, calibrated_test_losses, result_dir)
        print("\nThreshold sweep completed. Exiting.")
        return
    
    # 기본 모드: 단일 threshold 평가
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
    roc_auc, tpr_at_low_fpr = plot_roc_curve(labels, scores, os.path.join(result_dir, 'roc_curve.png'))
    
    # PR curve
    pr_auc = plot_pr_curve(labels, scores, os.path.join(result_dir, 'pr_curve.png'))
    
    # Confusion matrix
    plot_confusion_matrix(labels, predictions, args.threshold, 
                         os.path.join(result_dir, 'confusion_matrix.png'))
    
    # 메트릭 계산
    metrics = calculate_metrics(labels, predictions)
    
    # 결과 저장
    save_privacy_metrics(
        result_dir, weight_name, args.threshold, roc_auc, pr_auc,
        metrics, calibrated_train_losses, calibrated_test_losses,
        metric_name="Calibrated Loss", tpr_at_low_fpr=tpr_at_low_fpr
    )
    
    print(f"\n{'='*60}")
    print("Results Summary")
    print(f"{'='*60}")
    print_results(roc_auc, pr_auc, metrics, tpr_at_low_fpr)
    print(f"\nAll results saved to: {result_dir}")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()

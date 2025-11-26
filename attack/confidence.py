"""
Confidence 기반 Membership Inference Attack
모델의 confidence(최대 softmax 확률)를 기반으로 member/non-member를 구분하는 공격

python ./attack/confidence.py -c ./cfg/cocoqa/Res_Bert_Lora_dp.yaml -w ./checkpoints/cocoqa/Res_Bert_Lora_dp/best_model.pth --threshold 0.6

"""
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm
import sys

# 프로젝트 루트를 sys.path에 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

from attack.metric_src import (
    parse_args_with_config, setup_data_loaders, 
    load_model, plot_roc_curve, plot_pr_curve,
    plot_confusion_matrix, calculate_metrics, save_privacy_metrics, 
    print_results, get_confidence_and_pred, evaluate_privacy_confidence
)

torch.manual_seed(42)

def run_threshold_sweep(model, train_loader, test_loader, device, model_type, result_dir, weight_name):
    """0.1~0.9까지 threshold를 변경하며 공격 성능 평가"""
    thresholds = np.arange(0.1, 1.0, 0.1)
    results = []
    
    print("\n=== Threshold Sweep Analysis ===")
    print(f"Testing thresholds: {thresholds}")
    
    # 한 번만 confidence 계산
    print("Collecting confidences...")
    member_results = evaluate_privacy_confidence(
        model, train_loader, device,
        threshold=0.5, is_member=True, model_type=model_type
    )
    nonmember_results = evaluate_privacy_confidence(
        model, test_loader, device,
        threshold=0.5, is_member=False, model_type=model_type
    )
    
    scores_member = member_results['confidences']
    scores_nonmember = nonmember_results['confidences']
    scores = np.concatenate([scores_member, scores_nonmember])
    labels = np.concatenate([member_results['ground_truth'], nonmember_results['ground_truth']])
    
    # 각 threshold에 대해 평가
    for threshold in thresholds:
        preds = (scores >= threshold).astype(int)
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
    
    bars = ax.bar(x, accuracies, width=0.08, alpha=0.8, color='steelblue', edgecolor='black')
    ax.set_xlabel('Threshold', fontsize=12, fontweight='bold')
    ax.set_ylabel('Attack Accuracy', fontsize=12, fontweight='bold')
    ax.set_title('Confidence-based MIA: Attack Accuracy vs Threshold', fontsize=14, fontweight='bold')
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


def main():
    # threshold 인자 추가
    extra_args = [
        ('--threshold', float, 0.6, 'confidence threshold for membership'),
        ('--sweep_thresholds', int, 0, 'Run threshold sweep from 0.1 to 0.9 (1=True, 0=False)')
    ]
    args = parse_args_with_config(extra_args)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    weight_name = Path(args.weights).stem
    result_dir = os.path.join(os.path.dirname(args.weights), 'privacy_analysis', "confidence")
    os.makedirs(result_dir, exist_ok=True)
    
    # 데이터 로더 설정 (항상 내부 7:2:1 분할: 70% member, 10% non-member)
    train_loader, test_loader, tokenizer, image_transform = setup_data_loaders(args)
    
    # 모델 로드
    model, model_type = load_model(args, device)
    
    # Threshold sweep 모드
    if hasattr(args, 'sweep_thresholds') and args.sweep_thresholds:
        print("\n*** Threshold Sweep Mode ***")
        run_threshold_sweep(model, train_loader, test_loader, device, model_type, result_dir, weight_name)
        print("\nThreshold sweep completed. Exiting.")
        return
    
    # 기본 모드: 단일 threshold 평가
    print("Evaluating member (train) data...")
    member_results = evaluate_privacy_confidence(
        model, train_loader, device,
        threshold=args.threshold, is_member=True, model_type=model_type
    )
    
    print("Evaluating non-member (test) data...")
    nonmember_results = evaluate_privacy_confidence(
        model, test_loader, device,
        threshold=args.threshold, is_member=False, model_type=model_type
    )

    # Visualization
    os.makedirs(result_dir, exist_ok=True)

    plt.figure(figsize=(10, 6))
    sns.kdeplot(member_results['confidences'], label='Member')
    sns.kdeplot(nonmember_results['confidences'], label='Non-member')
    plt.xlabel('Confidence')
    plt.ylabel('Density')
    plt.title('Confidence Distribution: Member vs Non-member')
    plt.legend()
    plt.savefig(os.path.join(result_dir, 'confidence_distribution.png'))
    plt.close()
    scores_member = member_results['confidences']
    scores_nonmember = nonmember_results['confidences']
    scores = np.concatenate([scores_member, scores_nonmember])
    labels = np.concatenate([member_results['ground_truth'], nonmember_results['ground_truth']])

    # ROC / AUC
    roc_auc, tpr_at_low_fpr = plot_roc_curve(labels, scores, os.path.join(result_dir, 'roc_curve.png'))

    # PR curve
    pr_auc = plot_pr_curve(labels, scores, os.path.join(result_dir, 'pr_curve.png'))

    # Confusion Matrix
    preds_at_thresh = (scores >= args.threshold).astype(int)
    plot_confusion_matrix(labels, preds_at_thresh, args.threshold, 
                         os.path.join(result_dir, f'confusion_at_{args.threshold:.2f}.png'))

    # 메트릭 계산
    metrics = calculate_metrics(labels, preds_at_thresh)

    # 결과 저장
    save_privacy_metrics(
        result_dir, weight_name, args.threshold, roc_auc, pr_auc,
        metrics, scores_member, scores_nonmember,
        metric_name="Confidence", tpr_at_low_fpr=tpr_at_low_fpr
    )

    print(f"\nResults saved to {result_dir}")
    print_results(roc_auc, pr_auc, metrics, tpr_at_low_fpr)


if __name__ == '__main__':
    main()

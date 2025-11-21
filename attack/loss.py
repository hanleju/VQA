"""
Loss 기반 Membership Inference Attack
Loss 값을 기반으로 member/non-member를 구분하는 공격

python ./attack/loss.py -c ./cfg/IB/SwinT_BERTLoRA_coco/coattention.yaml -w ./checkpoints/IB/SwinT_BERTLoRA_coco_3/coattention/best_model.pth --threshold 0.6


"""
import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm
import sys

# 프로젝트 루트를 sys.path에 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

from attack.metric_src import (
    parse_args_with_config, setup_data_loaders, load_model, 
    plot_roc_curve, plot_pr_curve, plot_confusion_matrix, 
    calculate_metrics, save_privacy_metrics, print_results,
    compute_loss, evaluate_privacy_loss
)

torch.manual_seed(42)

def main():
    # threshold 인자 추가
    extra_args = [
        ('--threshold', float, 1.0, 'loss threshold for membership (lower loss = member)')
    ]
    args = parse_args_with_config(extra_args)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    weight_name = Path(args.weights).stem
    result_dir = os.path.join(os.path.dirname(args.weights), 'privacy_analysis', "loss")
    os.makedirs(result_dir, exist_ok=True)
    
    # 데이터 로더 설정 (항상 내부 7:2:1 분할 사용)
    train_loader, test_loader, tokenizer, image_transform = setup_data_loaders(args)
    
    # 모델 로드
    model, model_type = load_model(args, device)
    
    print("Evaluating member (train) data...")
    member_results = evaluate_privacy_loss(
        model, train_loader, device,
        threshold=args.threshold, is_member=True, model_type=model_type
    )
    
    print("Evaluating non-member (test) data...")
    nonmember_results = evaluate_privacy_loss(
        model, test_loader, device,
        threshold=args.threshold, is_member=False, model_type=model_type
    )

    # Visualization
    os.makedirs(result_dir, exist_ok=True)

    plt.figure(figsize=(10, 6))
    sns.kdeplot(member_results['losses'], label='Member')
    sns.kdeplot(nonmember_results['losses'], label='Non-member')
    plt.xlabel('Loss')
    plt.ylabel('Density')
    plt.title('Loss Distribution: Member vs Non-member')
    plt.legend()
    plt.savefig(os.path.join(result_dir, 'loss_distribution.png'))
    plt.close()

    # Loss는 낮을수록 member이므로, score를 반전 (-loss 또는 1/loss 사용)
    # 여기서는 -loss를 사용 (높은 값이 member)
    scores_member = -member_results['losses']
    scores_nonmember = -nonmember_results['losses']
    scores = np.concatenate([scores_member, scores_nonmember])
    labels = np.concatenate([member_results['ground_truth'], nonmember_results['ground_truth']])

    # ROC / AUC
    roc_auc = plot_roc_curve(labels, scores, os.path.join(result_dir, 'roc_curve.png'))

    # PR curve
    pr_auc = plot_pr_curve(labels, scores, os.path.join(result_dir, 'pr_curve.png'))

    # Confusion Matrix
    preds_at_thresh = (scores >= -args.threshold).astype(int)
    plot_confusion_matrix(labels, preds_at_thresh, args.threshold, 
                         os.path.join(result_dir, f'confusion_at_{args.threshold:.2f}.png'))

    # 메트릭 계산
    metrics = calculate_metrics(labels, preds_at_thresh)

    # 결과 저장
    save_privacy_metrics(
        result_dir, weight_name, args.threshold, roc_auc, pr_auc,
        metrics, member_results['losses'], nonmember_results['losses'],
        metric_name="Loss"
    )

    print(f"\nResults saved to {result_dir}")
    print_results(roc_auc, pr_auc, metrics)


if __name__ == '__main__':
    main()

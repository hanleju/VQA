"""
Loss 기반 Membership Inference Attack
Loss 값을 기반으로 member/non-member를 구분하는 공격
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

from attack.src import (
    parse_args_with_config, setup_data_loaders, load_model, 
    plot_roc_curve, plot_pr_curve, plot_confusion_matrix, 
    calculate_metrics, save_privacy_metrics, print_results
)

torch.manual_seed(42)

def compute_loss(outputs, targets, criterion):
    """
    배치의 각 샘플에 대한 Loss 계산
    
    Args:
        outputs: 모델 출력 (batch_size, num_classes)
        targets: 정답 레이블 (batch_size,)
        criterion: Loss 함수
        
    Returns:
        losses: 각 샘플의 loss 값 (batch_size,)
    """
    # reduction='none'으로 설정하여 각 샘플별 loss 계산
    criterion_no_reduction = nn.CrossEntropyLoss(reduction='none')
    losses = criterion_no_reduction(outputs, targets)
    return losses

def evaluate_privacy_loss(model, dataloader, device, threshold=1.0, is_member=True, model_type="VQAModel"):
    """
    데이터셋에 대한 loss 계산
    Loss가 threshold보다 낮으면 member로 예측
    
    Args:
        model: VQAModel 또는 VQAModel_IB
        dataloader: 데이터 로더
        device: 디바이스
        threshold: loss threshold (이보다 낮으면 member)
        is_member: member 데이터인지 여부
        model_type: "VQAModel" 또는 "VQAModel_IB"
        
    Returns:
        dict with losses, ground_truth and predictions
    """
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction='none')
    
    losses = []
    ground_truth = []  # 실제 membership (1 for member, 0 for non-member)
    predictions = []   # 예측된 membership (True/False)

    with torch.no_grad():
        for batch in tqdm(dataloader):
            images = batch['image'].to(device)
            inputs = batch['inputs'].to(device)
            answers = batch['answer'].to(device)

            # 다양한 모델 출력을 지원: logits 또는 (logits, ...)
            out = model(
                images=images,
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask']
            )
            outputs = out[0] if isinstance(out, tuple) else out

            # 각 샘플의 loss 계산
            batch_losses = criterion(outputs, answers)
            losses.extend(batch_losses.cpu().numpy())

            # Loss가 낮을수록 member일 가능성이 높음
            pred_member = batch_losses.cpu().numpy() <= threshold
            predictions.extend(pred_member)
            ground_truth.extend([1 if is_member else 0] * len(images))

    return {
        'losses': np.array(losses),
        'ground_truth': np.array(ground_truth),
        'predictions': np.array(predictions)
    }

def main():
    # threshold 인자 추가
    extra_args = [
        ('--threshold', float, 1.0, 'loss threshold for membership (lower loss = member)')
    ]
    args = parse_args_with_config(extra_args)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    weight_name = Path(args.weights).stem
    result_dir = os.path.join(os.path.dirname(args.weights), 'privacy_analysis', "loss_based")
    os.makedirs(result_dir, exist_ok=True)
    
    # 데이터 로더 설정
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

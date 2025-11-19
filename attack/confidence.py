"""
Confidence 기반 Membership Inference Attack
모델의 confidence(최대 softmax 확률)를 기반으로 member/non-member를 구분하는 공격
"""
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm

from attack.src import (
    parse_args_with_config, setup_data_loaders, 
    load_model, plot_roc_curve, plot_pr_curve,
    plot_confusion_matrix, calculate_metrics, save_privacy_metrics, 
    print_results
)

torch.manual_seed(42)

def get_confidence_and_pred(outputs):
    """
    소프트맥스 출력에서 confidence(최대 확률)와 예측 클래스를 반환
    """
    probs = torch.softmax(outputs, dim=1)
    confidence, predictions = torch.max(probs, dim=1)
    return confidence, predictions


def evaluate_privacy(model, dataloader, device, threshold=0.6, is_member=True, model_type="VQAModel"):
    """
    데이터셋에 대한 confidence 계산
    Confidence가 threshold보다 높으면 member로 예측
    
    Args:
        model: VQAModel 또는 VQAModel_IB
        dataloader: 데이터 로더
        device: 디바이스
        threshold: confidence threshold
        is_member: member 데이터인지 여부
        model_type: "VQAModel" 또는 "VQAModel_IB"
        
    Returns:
        dict with confidences, ground_truth and predictions
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

            # 다양한 모델 출력을 지원: logits 또는 (logits, ...)
            out = model(
                images=images,
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask']
            )
            outputs = out[0] if isinstance(out, tuple) else out

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

def main():
    # threshold 인자 추가
    extra_args = [
        ('--threshold', float, 0.6, 'confidence threshold for membership')
    ]
    args = parse_args_with_config(extra_args)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    weight_name = Path(args.weights).stem
    result_dir = os.path.join(os.path.dirname(args.weights), 'privacy_analysis', weight_name)
    os.makedirs(result_dir, exist_ok=True)
    
    # 데이터 로더 설정
    train_loader, test_loader, tokenizer, image_transform = setup_data_loaders(args)
    
    # 모델 로드
    model, model_type = load_model(args, device)
    
    print("Evaluating member (train) data...")
    member_results = evaluate_privacy(
        model, train_loader, device,
        threshold=args.threshold, is_member=True, model_type=model_type
    )
    
    print("Evaluating non-member (test) data...")
    nonmember_results = evaluate_privacy(
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
    roc_auc = plot_roc_curve(labels, scores, os.path.join(result_dir, 'roc_curve.png'))

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
        metric_name="Confidence"
    )

    print(f"\nResults saved to {result_dir}")
    print_results(roc_auc, pr_auc, metrics)


if __name__ == '__main__':
    main()

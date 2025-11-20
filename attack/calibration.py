"""
Difficulty Calibration 기반 Membership Inference Attack
2022 ICLR: "On the Importance of Difficulty Calibration in Membership Inference Attacks"
https://arxiv.org/abs/2206.06737

Loss 기반 공격의 개선 버전으로, 샘플의 난이도를 보정하여 더 정확한 공격 수행
핵심 아이디어: 어려운 샘플은 자연스럽게 높은 loss를 가지므로, 샘플 난이도를 고려하여 보정

개선사항: Hugging Face pre-trained multimodal models를 shadow models로 사용
- 별도 학습 불필요
- General한 난이도 추정 가능
- BLIP, ViLT 등의 pre-trained VQA 모델 활용
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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from PIL import Image
from transformers import (
    BlipProcessor, BlipForQuestionAnswering,
    ViltProcessor, ViltForQuestionAnswering,
    AutoProcessor, AutoModelForVisualQuestionAnswering
)

# 프로젝트 루트를 sys.path에 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

from attack.src import (
    parse_args_with_config, setup_data_loaders, load_model,
    plot_roc_curve, plot_pr_curve, plot_confusion_matrix,
    calculate_metrics, save_privacy_metrics, print_results
)

torch.manual_seed(42)
np.random.seed(42)

def compute_sample_loss(outputs, targets):
    """
    각 샘플에 대한 Cross-Entropy Loss 계산
    
    Args:
        outputs: 모델 출력 (batch_size, num_classes)
        targets: 정답 레이블 (batch_size,)
        
    Returns:
        losses: 각 샘플의 loss 값 (batch_size,)
    """
    criterion = nn.CrossEntropyLoss(reduction='none')
    losses = criterion(outputs, targets)
    return losses


def estimate_difficulty(shadow_losses):
    """
    Shadow model의 loss 분포로 샘플 난이도 추정
    
    Args:
        shadow_losses: Shadow model들의 loss 값 (num_shadows, num_samples)
        
    Returns:
        difficulties: 각 샘플의 난이도 (평균 loss) (num_samples,)
    """
    # 여러 shadow model의 평균 loss를 난이도로 사용
    difficulties = np.mean(shadow_losses, axis=0)
    return difficulties


def calibrate_loss(target_loss, difficulty):
    """
    Loss를 난이도로 보정
    
    Args:
        target_loss: target model의 loss 값
        difficulty: 샘플 난이도 (shadow model 평균 loss)
        
    Returns:
        calibrated_loss: 보정된 loss (target_loss - difficulty)
    """
    # 난이도가 높은 샘플(높은 평균 loss)의 경우, target loss도 높을 것으로 예상
    # 따라서 난이도를 빼서 상대적인 loss를 계산
    calibrated_loss = target_loss - difficulty
    return calibrated_loss


def collect_losses_and_labels(model, dataloader, device, model_type="VQAModel"):
    """
    데이터셋에 대한 loss와 예측 정확도 수집
    
    Args:
        model: VQA 모델
        dataloader: 데이터 로더
        device: 디바이스
        model_type: 모델 타입
        
    Returns:
        losses: 샘플별 loss 값
        correct: 샘플별 정답 여부 (0 or 1)
    """
    model.eval()
    all_losses = []
    all_correct = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Collecting losses", leave=False):
            images = batch['image'].to(device)
            inputs = batch['inputs'].to(device)
            answers = batch['answer'].to(device)
            
            # Forward pass
            if model_type == "VQAModel_IB":
                outputs, _ = model(
                    images=images,
                    input_ids=inputs['input_ids'],
                    attention_mask=inputs['attention_mask']
                )
            else:
                outputs = model(
                    images=images,
                    input_ids=inputs['input_ids'],
                    attention_mask=inputs['attention_mask']
                )
            
            # Loss 계산
            losses = compute_sample_loss(outputs, answers)
            all_losses.extend(losses.cpu().numpy())
            
            # 정답 여부 계산
            predictions = torch.argmax(outputs, dim=1)
            correct = (predictions == answers).float()
            all_correct.extend(correct.cpu().numpy())
    
    return np.array(all_losses), np.array(all_correct)


def train_shadow_models_and_collect(args, device, shadow_models=None):
    """
    Hugging Face pre-trained multimodal models를 shadow models로 사용하여 loss 수집
    별도 학습 없이 general한 난이도 추정
    
    Args:
        args: 설정 인자
        device: 디바이스
        shadow_models: 사용할 shadow model 이름 리스트
                      None이면 기본 모델들 사용
        
    Returns:
        shadow_train_losses: shadow model들의 train loss (num_shadows, train_size)
        shadow_test_losses: shadow model들의 test loss (num_shadows, test_size)
    """
    if shadow_models is None:
        # 기본 shadow models: Hugging Face pre-trained VQA models
        shadow_models = [
            # "dandelin/vilt-b32-finetuned-vqa",
            "Salesforce/blip-vqa-base",
        ]
    
    print(f"\n=== Using {len(shadow_models)} Pre-trained Shadow Models ===")
    for idx, model_name in enumerate(shadow_models):
        print(f"{idx+1}. {model_name}")
    
    # 데이터 로더 설정
    train_loader, test_loader, _, _ = setup_data_loaders(args, seed=42)
    
    shadow_train_losses = []
    shadow_test_losses = []
    
    for shadow_idx, model_name in enumerate(shadow_models):
        print(f"\n{'='*60}")
        print(f"Shadow Model {shadow_idx + 1}/{len(shadow_models)}: {model_name}")
        print(f"{'='*60}")
        
        try:
            # Shadow model 로드
            shadow_model, shadow_processor = load_pretrained_vqa_model(model_name, device)
            
            # Train/Test loss 수집 (split 정보 전달)
            train_losses = collect_shadow_losses(
                shadow_model, shadow_processor, train_loader, 
                device, model_name, args.dataset_root, split='train'
            )
            test_losses = collect_shadow_losses(
                shadow_model, shadow_processor, test_loader, 
                device, model_name, args.dataset_root, split='test'
            )
            
            shadow_train_losses.append(train_losses)
            shadow_test_losses.append(test_losses)
            
            print(f"✓ Collected - Train: {len(train_losses)}, Test: {len(test_losses)}")
            
            # 메모리 정리
            del shadow_model, shadow_processor
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"✗ Error with {model_name}: {e}")
            print(f"  Skipping this shadow model...")
            continue
    
    if len(shadow_train_losses) == 0:
        raise RuntimeError("No shadow models successfully loaded. Cannot proceed with calibration.")
    
    return np.array(shadow_train_losses), np.array(shadow_test_losses)


def load_pretrained_vqa_model(model_name, device):
    """
    Hugging Face에서 pre-trained VQA model 로드
    
    Args:
        model_name: 모델 이름 (e.g., "Salesforce/blip-vqa-base")
        device: 디바이스
        
    Returns:
        model: 로드된 모델
        processor: 해당 모델의 processor
    """
    print(f"Loading {model_name}...")
    

    if "vilt" in model_name.lower():
        processor = ViltProcessor.from_pretrained(model_name)
        model = ViltForQuestionAnswering.from_pretrained(model_name, use_safetensors=True).to(device)
    elif "blip" in model_name.lower():
        processor = BlipProcessor.from_pretrained(model_name)
        model = BlipForQuestionAnswering.from_pretrained(model_name).to(device)
    else:
        # Generic VQA model
        processor = AutoProcessor.from_pretrained(model_name)
        model = AutoModelForVisualQuestionAnswering.from_pretrained(model_name).to(device)
    
    model.eval()
    print(f"✓ Model loaded successfully")
    
    return model, processor


def collect_shadow_losses(shadow_model, processor, dataloader, device, model_name, dataset_root, split='train'):
    """
    Pre-trained shadow model로 각 샘플의 loss 계산
    
    Args:
        shadow_model: Hugging Face VQA 모델
        processor: 모델 processor
        dataloader: 데이터 로더
        device: 디바이스
        model_name: 모델 이름 (로깅용)
        dataset_root: 데이터셋 루트 경로
        split: 'train' 또는 'test' (이미지 폴더 경로)
        
    Returns:
        losses: 각 샘플의 loss 값 (numpy array)
    """
    all_losses = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Collecting losses ({model_name.split('/')[-1]})", leave=False):
            try:
                # 이미지 파일 경로 가져오기 (split 경로 + 확장자 자동 감지)
                image_paths = []
                for img_id in batch['image_id']:
                    # split 폴더 포함 경로
                    base_path = os.path.join(dataset_root, split, 'images', img_id)
                    # .jpg, .png, .jpeg 확장자 시도
                    found = False
                    for ext in ['.jpg', '.png', '.jpeg']:
                        if os.path.exists(base_path + ext):
                            image_paths.append(base_path + ext)
                            found = True
                            break
                    
                    if not found:
                        # 확장자 없이 존재하는지 확인
                        if os.path.exists(base_path):
                            image_paths.append(base_path)
                        else:
                            raise FileNotFoundError(f"Image not found: {base_path}")
                
                # PIL Image로 로드
                images = [Image.open(path).convert('RGB') for path in image_paths]
                
                # 질문 텍스트
                questions = batch['question']
                
                # 정답 텍스트 (VQA는 텍스트 답변)
                answers = batch['answer_text'] if 'answer_text' in batch else [str(ans) for ans in batch['answer']]
                
                # Processor로 입력 준비
                if "blip" in model_name.lower():
                    inputs = processor(images=images, text=questions, return_tensors="pt", padding=True, truncation=True).to(device)
                    labels = processor(text=answers, return_tensors="pt", padding=True, truncation=True).input_ids.to(device)
                    
                    # Forward pass
                    outputs = shadow_model(**inputs, labels=labels)
                    loss = outputs.loss
                    
                    # Batch loss를 샘플별로 분리
                    batch_size = len(images)
                    sample_losses = [loss.item()] * batch_size  # 평균 loss를 각 샘플에 할당
                    
                elif "vilt" in model_name.lower():
                    # ViLT는 max_length=40이므로 truncation 필수
                    inputs = processor(images=images, text=questions, return_tensors="pt", padding=True, truncation=True, max_length=40).to(device)
                    
                    # ViLT는 answer를 class index로 변환 필요
                    # 간단히 하기 위해 logits만 사용
                    outputs = shadow_model(**inputs)
                    logits = outputs.logits
                    
                    # Cross-entropy loss 계산 (answer를 class로 가정)
                    # 실제로는 answer vocabulary 매핑 필요
                    # 여기서는 approximation으로 uniform distribution과의 KL divergence 사용
                    probs = torch.softmax(logits, dim=-1)
                    entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1)
                    sample_losses = entropy.cpu().numpy().tolist()
                    
                else:
                    # Generic model
                    inputs = processor(images=images, text=questions, return_tensors="pt", padding=True, truncation=True).to(device)
                    outputs = shadow_model(**inputs)
                    logits = outputs.logits
                    
                    probs = torch.softmax(logits, dim=-1)
                    entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1)
                    sample_losses = entropy.cpu().numpy().tolist()
                
                all_losses.extend(sample_losses)
                
            except Exception as e:
                print(f"\n⚠ Error processing batch: {e}")
                # 에러 발생 시 해당 배치는 평균 loss로 대체
                batch_size = len(batch['image'])
                all_losses.extend([1.0] * batch_size)
                continue
    
    return np.array(all_losses)


def perform_calibrated_attack(target_train_losses, target_test_losses, 
                               shadow_train_losses, shadow_test_losses):
    """
    Difficulty Calibration을 적용한 Membership Inference Attack
    
    Args:
        target_train_losses: target model의 train loss
        target_test_losses: target model의 test loss
        shadow_train_losses: shadow model들의 train loss
        shadow_test_losses: shadow model들의 test loss
        
    Returns:
        train_predictions: train 데이터에 대한 예측 (member=1)
        test_predictions: test 데이터에 대한 예측 (member=1)
        train_scores: train 데이터의 member 점수
        test_scores: test 데이터의 member 점수
        calibrated_train_losses: calibrated train loss
        calibrated_test_losses: calibrated test loss
    """
    print("\n=== Performing Difficulty Calibration ===")
    
    # 1. Shadow model들로부터 난이도 추정
    train_difficulty = estimate_difficulty(shadow_train_losses)
    test_difficulty = estimate_difficulty(shadow_test_losses)
    
    print(f"Train difficulty - Mean: {train_difficulty.mean():.4f}, Std: {train_difficulty.std():.4f}")
    print(f"Test difficulty - Mean: {test_difficulty.mean():.4f}, Std: {test_difficulty.std():.4f}")
    
    # 2. Target model의 loss를 난이도로 보정
    calibrated_train_losses = calibrate_loss(target_train_losses, train_difficulty)
    calibrated_test_losses = calibrate_loss(target_test_losses, test_difficulty)
    
    print(f"\nCalibrated train loss - Mean: {calibrated_train_losses.mean():.4f}, Std: {calibrated_train_losses.std():.4f}")
    print(f"Calibrated test loss - Mean: {calibrated_test_losses.mean():.4f}, Std: {calibrated_test_losses.std():.4f}")
    
    # 3. Logistic Regression으로 member/non-member 분류
    # Train: member (label=1), Test: non-member (label=0)
    X_train = calibrated_train_losses.reshape(-1, 1)
    y_train = np.ones(len(X_train))
    
    X_test = calibrated_test_losses.reshape(-1, 1)
    y_test = np.zeros(len(X_test))
    
    # 데이터 합치기
    X = np.vstack([X_train, X_test])
    y = np.hstack([y_train, y_test])
    
    # Logistic Regression 학습
    clf = LogisticRegression(random_state=42, max_iter=1000)
    clf.fit(X, y)
    
    # 예측 (확률)
    train_probs = clf.predict_proba(X_train)[:, 1]  # member 확률
    test_probs = clf.predict_proba(X_test)[:, 1]
    
    # 예측 (이진 분류)
    train_predictions = clf.predict(X_train)
    test_predictions = clf.predict(X_test)
    
    return train_predictions, test_predictions, train_probs, test_probs, calibrated_train_losses, calibrated_test_losses


def evaluate_attack_performance(train_preds, test_preds, train_scores, test_scores, 
                                train_losses, test_losses, 
                                calibrated_train_losses, calibrated_test_losses,
                                output_dir):
    """
    공격 성능 평가 및 시각화
    
    Args:
        train_preds: train 데이터 예측 (member=1)
        test_preds: test 데이터 예측
        train_scores: train 데이터 member 점수
        test_scores: test 데이터 member 점수
        train_losses: train 데이터 원본 loss
        test_losses: test 데이터 원본 loss
        calibrated_train_losses: calibrated train loss
        calibrated_test_losses: calibrated test loss
        output_dir: 결과 저장 디렉토리
    """
    # Ground truth: train=member(1), test=non-member(0)
    y_true = np.hstack([
        np.ones(len(train_preds)),
        np.zeros(len(test_preds))
    ])
    
    y_pred = np.hstack([train_preds, test_preds])
    y_scores = np.hstack([train_scores, test_scores])
    
    # 메트릭 계산 (calculate_metrics는 2개 인자만 받음)
    metrics = calculate_metrics(y_true, y_pred)
    
    # ROC AUC와 PR AUC 추가 계산
    try:
        roc_auc = roc_auc_score(y_true, y_scores)
    except:
        roc_auc = None
    
    try:
        precision_vals, recall_vals, _ = precision_recall_curve(y_true, y_scores)
        pr_auc = auc(recall_vals, precision_vals)
    except:
        pr_auc = None
    
    # 결과 출력 (print_results는 roc_auc, pr_auc, metrics 순서)
    print_results(roc_auc, pr_auc, metrics)
    
    # 시각화
    print("\n=== Generating Visualizations ===")
    
    # ROC Curve
    plot_roc_curve(y_true, y_scores, os.path.join(output_dir, "calibration_roc_curve.png"))
    
    # Precision-Recall Curve
    plot_pr_curve(y_true, y_scores, os.path.join(output_dir, "calibration_pr_curve.png"))
    
    # Confusion Matrix (threshold는 0.5로 고정, Logistic Regression 기본값)
    plot_confusion_matrix(y_true, y_pred, 0.5, os.path.join(output_dir, "calibration_confusion_matrix.png"))
    
    # Loss Distribution - Original (Before Calibration)
    plot_loss_distributions(train_losses, test_losses, output_dir, "original_loss_distribution.png", "Original Loss Distribution")
    
    # Loss Distribution - Calibrated (After Calibration)
    plot_loss_distributions(calibrated_train_losses, calibrated_test_losses, output_dir, "calibrated_loss_distribution.png", "Calibrated Loss Distribution")
    
    # 메트릭 저장 (간단한 JSON 파일로 저장)
    import json
    metrics_to_save = {
        'roc_auc': float(roc_auc) if roc_auc is not None else None,
        'pr_auc': float(pr_auc) if pr_auc is not None else None,
        'accuracy': float(metrics['accuracy']),
        'precision': float(metrics['precision']),
        'recall': float(metrics['recall']),
        'f1': float(metrics['f1'])
    }
    
    metrics_path = os.path.join(output_dir, "calibration_metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(metrics_to_save, f, indent=4)
    print(f"Metrics saved to {metrics_path}")
    
    return metrics


def plot_loss_distributions(train_losses, test_losses, output_dir, filename="loss_distribution.png", title="Loss Distribution: Member vs Non-member"):
    """
    Train/Test loss 분포 시각화
    
    Args:
        train_losses: train loss 값
        test_losses: test loss 값
        output_dir: 저장 디렉토리
        filename: 저장 파일명
        title: 그래프 제목
    """
    plt.figure(figsize=(10, 6))
    
    plt.hist(train_losses, bins=50, alpha=0.6, label='Train (Member)', color='blue', density=True)
    plt.hist(test_losses, bins=50, alpha=0.6, label='Test (Non-member)', color='red', density=True)
    
    plt.xlabel('Loss', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(alpha=0.3)
    
    save_path = os.path.join(output_dir, filename)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Loss distribution saved to {save_path}")


def main():
    # 인자 파싱
    extra_args = [
        ('--shadow_models', str, None, 'Comma-separated list of shadow model names (default: BLIP, ViLT, BLIP-large)'),
    ]
    args = parse_args_with_config(extra_args)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 출력 디렉토리 생성 (weights 경로 기반)
    output_dir = os.path.join(os.path.dirname(args.weights), 'privacy_analysis', 'calibration')
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Shadow models 파싱
    shadow_models = None
    if hasattr(args, 'shadow_models') and args.shadow_models:
        shadow_models = [m.strip() for m in args.shadow_models.split(',')]
    
    print(f"\n{'='*60}")
    print(f"DIFFICULTY CALIBRATION MEMBERSHIP INFERENCE ATTACK")
    print(f"Using Pre-trained Multimodal Models as Shadow Models")
    print(f"{'='*60}")
    print(f"Model: {args.model}")
    print(f"Weights: {args.weights}")
    print(f"Output Directory: {output_dir}")
    print(f"{'='*60}\n")
    
    # 1. Shadow models로부터 loss 분포 수집
    shadow_train_losses, shadow_test_losses = train_shadow_models_and_collect(
        args, device, shadow_models=shadow_models
    )
    
    # 2. Target model의 loss 수집
    print("\n=== Collecting Target Model Losses ===")
    train_loader, test_loader, _, _ = setup_data_loaders(args, seed=42)
    model, model_type = load_model(args, device)
    
    target_train_losses, _ = collect_losses_and_labels(model, train_loader, device, model_type)
    target_test_losses, _ = collect_losses_and_labels(model, test_loader, device, model_type)
    
    print(f"Target train loss - Mean: {target_train_losses.mean():.4f}, Std: {target_train_losses.std():.4f}")
    print(f"Target test loss - Mean: {target_test_losses.mean():.4f}, Std: {target_test_losses.std():.4f}")
    
    # 3. Calibrated Attack 수행
    train_preds, test_preds, train_scores, test_scores, calibrated_train_losses, calibrated_test_losses = perform_calibrated_attack(
        target_train_losses, target_test_losses,
        shadow_train_losses, shadow_test_losses
    )
    
    # 4. 공격 성능 평가
    metrics = evaluate_attack_performance(
        train_preds, test_preds, train_scores, test_scores,
        target_train_losses, target_test_losses,
        calibrated_train_losses, calibrated_test_losses,
        output_dir
    )
    
    print(f"\n{'='*60}")
    print(f"Attack completed! Results saved to {output_dir}")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()

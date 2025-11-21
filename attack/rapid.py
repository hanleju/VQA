"""
RAPID: Advanced Difficulty Calibration with MLP Attack Model
3-layer MLP를 사용하여 calibrated loss와 target loss를 입력으로 membership 판단

핵심 개선사항:
- Logistic Regression 대신 3-layer MLP 사용
- 2차원 입력: [calibrated_loss, target_loss]
- 더 복잡한 비선형 패턴 학습 가능

사용법:
python ./attack/rapid.py -c ./cfg/baseline/SwinT_BERTLoRA_coco/concat.yaml -w ./checkpoints/baseline/SwinT_BERTLoRA_coco/concat/best_model.pth --epochs 50 --lr 0.001
"""
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
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
    collect_target_losses, calibrate_loss, collect_shadow_difficulties
)

torch.manual_seed(42)
np.random.seed(42)


# ============================================================
# MLP Attack Model
# ============================================================

class MLPAttackModel(nn.Module):
    """
    3-layer MLP for membership inference
    Input: [calibrated_loss, target_loss] (2D)
    Output: membership probability (1D)
    """
    def __init__(self, hidden_dim=64):
        super(MLPAttackModel, self).__init__()
        self.fc1 = nn.Linear(2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        """
        Args:
            x: (batch_size, 2) - [calibrated_loss, target_loss]
        Returns:
            (batch_size, 1) - membership probability
        """
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x


def train_mlp_attack_model(calibrated_train_losses, calibrated_test_losses,
                           target_train_losses, target_test_losses,
                           device, epochs=30, lr=0.001, batch_size=64):
    """
    MLP attack model 학습
    
    Args:
        calibrated_train_losses: calibrated train loss
        calibrated_test_losses: calibrated test loss
        target_train_losses: target train loss (original)
        target_test_losses: target test loss (original)
        device: 디바이스
        epochs: 학습 에폭 수
        lr: 학습률
        batch_size: 배치 크기
        
    Returns:
        trained_model, train_probs, test_probs
    """
    # Feature 구성: [calibrated_loss, target_loss]
    X_train = np.stack([calibrated_train_losses, target_train_losses], axis=1)
    y_train = np.ones(len(X_train))
    
    X_test = np.stack([calibrated_test_losses, target_test_losses], axis=1)
    y_test = np.zeros(len(X_test))
    
    # 데이터 합치기
    X = np.vstack([X_train, X_test])
    y = np.hstack([y_train, y_test])
    
    # Tensor 변환
    X_tensor = torch.FloatTensor(X).to(device)
    y_tensor = torch.FloatTensor(y).unsqueeze(1).to(device)
    
    # DataLoader
    dataset = TensorDataset(X_tensor, y_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # 모델 초기화
    model = MLPAttackModel(hidden_dim=64).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    print(f"\n=== Training MLP Attack Model ===")
    print(f"Epochs: {epochs}, LR: {lr}, Batch Size: {batch_size}")
    print(f"Input features: [calibrated_loss, target_loss]")
    
    # 학습
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch_x, batch_y in dataloader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        if (epoch + 1) % 10 == 0:
            avg_loss = epoch_loss / len(dataloader)
            print(f"Epoch [{epoch+1}/{epochs}] - Loss: {avg_loss:.4f}")
    
    # 예측
    model.eval()
    with torch.no_grad():
        X_train_tensor = torch.FloatTensor(X_train).to(device)
        X_test_tensor = torch.FloatTensor(X_test).to(device)
        
        train_probs = model(X_train_tensor).cpu().numpy().flatten()
        test_probs = model(X_test_tensor).cpu().numpy().flatten()
    
    print(f"✓ MLP training completed")
    
    return model, train_probs, test_probs


def evaluate_mlp_attack(train_probs, test_probs, threshold=0.5):
    """
    MLP 공격 평가
    
    Args:
        train_probs: train member 확률
        test_probs: test non-member 확률
        threshold: 이진 분류 threshold
        
    Returns:
        scores, labels, predictions
    """
    scores = np.concatenate([train_probs, test_probs])
    labels = np.concatenate([
        np.ones(len(train_probs)),
        np.zeros(len(test_probs))
    ])
    predictions = (scores >= threshold).astype(int)
    
    return scores, labels, predictions


def plot_loss_distributions(train_losses, test_losses, output_dir, filename, title):
    """Loss 분포 시각화"""
    plt.figure(figsize=(10, 6))
    plt.hist(train_losses, bins=50, alpha=0.6, label='Member (Train)', color='blue', density=True)
    plt.hist(test_losses, bins=50, alpha=0.6, label='Non-member (Test)', color='red', density=True)
    plt.xlabel('Loss', fontsize=12)
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
        ('--shadow_models', str, None, 'Comma-separated shadow model names'),
        ('--epochs', int, 50, 'MLP training epochs'),
        ('--lr', float, 0.001, 'MLP learning rate'),
        ('--batch_size', int, 64, 'MLP training batch size'),
        ('--threshold', float, 0.5, 'Binary classification threshold'),
    ]
    args = parse_args_with_config(extra_args)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    weight_name = Path(args.weights).stem
    result_dir = os.path.join(os.path.dirname(args.weights), 'privacy_analysis', 'rapid')
    os.makedirs(result_dir, exist_ok=True)
    
    shadow_models = None
    if hasattr(args, 'shadow_models') and args.shadow_models:
        shadow_models = [m.strip() for m in args.shadow_models.split(',')]
    
    print(f"\n{'='*60}")
    print("RAPID: MLP-based Difficulty Calibration MIA")
    print("(3-layer MLP with [calibrated_loss, target_loss] input)")
    print(f"{'='*60}")
    print(f"Model: {args.model}")
    print(f"Weights: {args.weights}")
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
    
    # 4. MLP Attack Model 학습
    print("\n=== Step 4: Training MLP Attack Model ===")
    mlp_model, train_probs, test_probs = train_mlp_attack_model(
        calibrated_train_losses, calibrated_test_losses,
        target_train_losses, target_test_losses,
        device, epochs=args.epochs, lr=args.lr, batch_size=args.batch_size
    )
    
    # 5. 공격 평가
    print("\n=== Step 5: Evaluating MLP Attack ===")
    scores, labels, predictions = evaluate_mlp_attack(train_probs, test_probs, args.threshold)
    
    # 6. 시각화 및 메트릭
    print("\n=== Step 6: Visualization and Metrics ===")
    
    # Distribution plots
    plot_loss_distributions(
        calibrated_train_losses, calibrated_test_losses,
        result_dir, "calibrated_loss_distribution.png",
        "Calibrated Loss Distribution: Member vs Non-member"
    )
    plot_loss_distributions(
        target_train_losses, target_test_losses,
        result_dir, "target_loss_distribution.png",
        "Target Loss Distribution: Member vs Non-member"
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
        metrics, train_probs, test_probs,
        metric_name="MLP Score"
    )
    
    # MLP 모델 저장
    mlp_save_path = os.path.join(result_dir, 'mlp_attack_model.pth')
    torch.save(mlp_model.state_dict(), mlp_save_path)
    print(f"MLP model saved: {mlp_save_path}")
    
    print(f"\n{'='*60}")
    print("Results Summary")
    print(f"{'='*60}")
    print_results(roc_auc, pr_auc, metrics)
    print(f"\nAll results saved to: {result_dir}")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()

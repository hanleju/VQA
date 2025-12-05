"""
Optuna를 사용한 Multi-objective 하이퍼파라미터 튜닝
목표: Accuracy 최대화 + MIA Attack Accuracy 최소화 (Privacy 강건성)

python optuna_tune.py -c ./cfg/12.04/base_config.yaml --n_trials 50
"""
import os
import sys
import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from transformers import BertTokenizer
from functools import partial
from pathlib import Path
import numpy as np
import optuna
from optuna.visualization import plot_pareto_front, plot_optimization_history, plot_param_importances

# 프로젝트 루트를 sys.path에 추가
sys.path.insert(0, str(Path(__file__).parent))

from data.data import VQADataset, collate_fn_with_tokenizer
from utils.src import train, validate
from utils.util import create_model
from attack.metric_src import evaluate_privacy_loss, calculate_metrics

torch.manual_seed(42)


class OptunaTrainer:
    """Optuna 기반 하이퍼파라미터 튜닝 클래스"""
    
    def __init__(self, base_config_path, device, n_epochs=10):
        """
        Args:
            base_config_path: 기본 YAML 설정 파일 경로
            device: torch device
            n_epochs: 각 trial당 학습 에포크 수
        """
        self.device = device
        self.n_epochs = n_epochs
        
        # 기본 설정 로드
        with open(base_config_path, 'r', encoding='utf-8') as f:
            self.base_config = yaml.safe_load(f)
        
        # 데이터 로더 준비 (한 번만 생성)
        self.train_loader, self.val_loader, self.test_loader = self._setup_dataloaders()
        
    def _setup_dataloaders(self):
        """데이터 로더 설정 (train.py와 동일)"""
        image_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                std=[0.229, 0.224, 0.225])
        ])
        
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        collate_fn = partial(collate_fn_with_tokenizer, tokenizer=tokenizer)
        
        full_dataset = VQADataset(
            root_dir=self.base_config['dataset_root'],
            split='train',
            transform=image_transform
        )
        
        # 7:2:1 분할
        total_size = len(full_dataset)
        train_size = int(total_size * 0.7)
        val_size = int(total_size * 0.2)
        test_size = total_size - train_size - val_size
        
        train_dataset, val_dataset, test_dataset = random_split(
            full_dataset, [train_size, val_size, test_size]
        )
        
        batch_size = self.base_config.get('batch_size', 32)
        num_workers = self.base_config.get('num_workers', 4)
        
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True,
            collate_fn=collate_fn, num_workers=num_workers, pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False,
            collate_fn=collate_fn, num_workers=num_workers, pin_memory=True
        )
        
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False,
            collate_fn=collate_fn, num_workers=num_workers, pin_memory=True
        )
        
        return train_loader, val_loader, test_loader
    
    def _create_args_from_config(self, config):
        """config dict를 argparse.Namespace로 변환"""
        args = argparse.Namespace(**config)
        return args
    
    def _train_model(self, trial_config):
        """모델 학습 및 validation accuracy 반환"""
        args = self._create_args_from_config(trial_config)
        
        # 모델 생성
        model = create_model(args, self.device)
        
        optimizer = optim.AdamW(
            model.parameters(),
            lr=trial_config['lr'],
            weight_decay=trial_config['weight_decay']
        )
        criterion = nn.CrossEntropyLoss()
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
        
        best_val_acc = 0.0
        
        # 학습
        for epoch in range(self.n_epochs):
            train_loss, train_acc = train(
                model, self.train_loader, optimizer, criterion, self.device, epoch
            )
            val_loss, val_acc = validate(
                model, self.val_loader, criterion, self.device
            )
            scheduler.step()
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
        
        return model, best_val_acc
    
    def _evaluate_privacy(self, model):
        """Privacy 강건성 평가 (MIA Attack Accuracy)"""
        model.eval()
        
        # Member (train) 데이터에 대한 loss
        member_results = evaluate_privacy_loss(
            model, self.train_loader, self.device,
            threshold=1.0, is_member=True, model_type='VQAModel'
        )
        
        # Non-member (test) 데이터에 대한 loss
        nonmember_results = evaluate_privacy_loss(
            model, self.test_loader, self.device,
            threshold=1.0, is_member=False, model_type='VQAModel'
        )
        
        # Loss 기반 MIA attack
        # Loss가 낮으면 member로 예측 -> score를 반전 (-loss)
        scores_member = -member_results['losses']
        scores_nonmember = -nonmember_results['losses']
        scores = np.concatenate([scores_member, scores_nonmember])
        labels = np.concatenate([
            member_results['ground_truth'],
            nonmember_results['ground_truth']
        ])
        
        # Threshold를 중앙값으로 설정
        threshold = np.median(scores)
        preds = (scores >= threshold).astype(int)
        
        # MIA Attack Accuracy 계산
        metrics = calculate_metrics(labels, preds)
        mia_accuracy = metrics['accuracy']
        
        return mia_accuracy
    
    def objective(self, trial):
        """
        Optuna objective function
        
        Returns:
            (val_accuracy, mia_accuracy) - 둘 다 최대화하고 싶지만,
            실제로는 val_accuracy는 maximize, mia_accuracy는 minimize
        """
        # 하이퍼파라미터 샘플링
        prune_ratio = trial.suggest_float('prune_ratio', 0.3, 0.6, step=0.1)
        adv_ratio = trial.suggest_float('adv_ratio', 0.3, 0.7, step=0.1)
        mixup_alpha = trial.suggest_float('mixup_alpha', 0.05, 0.4, log=True)
        lr = trial.suggest_float('lr', 1e-5, 1e-3, log=True)
        weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True)
        
        # 설정 업데이트
        trial_config = self.base_config.copy()
        trial_config.update({
            'use_token_pruning': True,
            'prune_ratio': prune_ratio,
            'adv_ratio': adv_ratio,
            'mixup_alpha': mixup_alpha,
            'lr': lr,
            'weight_decay': weight_decay
        })
        
        print(f"\n{'='*60}")
        print(f"Trial {trial.number}")
        print(f"{'='*60}")
        print(f"Prune Ratio: {prune_ratio}")
        print(f"Adv Ratio: {adv_ratio}")
        print(f"Mixup Alpha: {mixup_alpha:.4f}")
        print(f"LR: {lr:.6f}")
        print(f"Weight Decay: {weight_decay:.6f}")
        print(f"{'='*60}\n")
        
        # 모델 학습
        model, val_accuracy = self._train_model(trial_config)
        
        # Privacy 평가
        mia_accuracy = self._evaluate_privacy(model)
        
        print(f"\n{'='*60}")
        print(f"Trial {trial.number} Results")
        print(f"{'='*60}")
        print(f"Validation Accuracy: {val_accuracy:.4f}")
        print(f"MIA Attack Accuracy: {mia_accuracy:.4f}")
        print(f"{'='*60}\n")
        
        # GPU 메모리 정리
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Multi-objective: (accuracy 최대화, MIA 최소화)
        return val_accuracy, mia_accuracy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, required=True,
                       help='Base YAML config file')
    parser.add_argument('--n_trials', type=int, default=50,
                       help='Number of Optuna trials')
    parser.add_argument('--n_epochs', type=int, default=15,
                       help='Epochs per trial')
    parser.add_argument('--study_name', type=str, default='vqa_privacy_tuning',
                       help='Optuna study name')
    parser.add_argument('--storage', type=str, default=None,
                       help='Optuna storage (e.g., sqlite:///optuna.db)')
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Optuna Trainer 초기화
    trainer = OptunaTrainer(args.config, device, n_epochs=args.n_epochs)
    
    # Optuna Study 생성 (Multi-objective)
    study = optuna.create_study(
        study_name=args.study_name,
        storage=args.storage,
        directions=['maximize', 'minimize'],  # [val_accuracy↑, mia_accuracy↓]
        sampler=optuna.samplers.TPESampler(seed=42),
        load_if_exists=True
    )
    
    print(f"\n{'='*60}")
    print(f"Starting Optuna Optimization")
    print(f"{'='*60}")
    print(f"Study Name: {args.study_name}")
    print(f"Number of Trials: {args.n_trials}")
    print(f"Epochs per Trial: {args.n_epochs}")
    print(f"Objectives: [Accuracy ↑, MIA Accuracy ↓]")
    print(f"{'='*60}\n")
    
    # 최적화 실행
    study.optimize(trainer.objective, n_trials=args.n_trials)
    
    # 결과 출력
    print(f"\n{'='*60}")
    print(f"Optimization Complete!")
    print(f"{'='*60}\n")
    
    print(f"Number of finished trials: {len(study.trials)}")
    print(f"\nBest trials (Pareto Front):")
    
    for i, trial in enumerate(study.best_trials):
        print(f"\n--- Trial {trial.number} (Pareto Optimal #{i+1}) ---")
        print(f"  Validation Accuracy: {trial.values[0]:.4f}")
        print(f"  MIA Attack Accuracy: {trial.values[1]:.4f}")
        print(f"  Parameters:")
        for key, value in trial.params.items():
            print(f"    {key}: {value}")
    
    # 결과 저장 디렉토리
    result_dir = Path('./optuna_results') / args.study_name
    result_dir.mkdir(parents=True, exist_ok=True)
    
    # 시각화 저장
    try:
        # Pareto front
        fig = plot_pareto_front(study, target_names=['Accuracy', 'MIA Accuracy'])
        fig.write_html(str(result_dir / 'pareto_front.html'))
        print(f"\nPareto front plot saved to: {result_dir / 'pareto_front.html'}")
        
        # Optimization history
        fig = plot_optimization_history(study)
        fig.write_html(str(result_dir / 'optimization_history.html'))
        print(f"Optimization history saved to: {result_dir / 'optimization_history.html'}")
        
        # Parameter importance (각 objective별로)
        for i, obj_name in enumerate(['Accuracy', 'MIA Accuracy']):
            fig = plot_param_importances(study, target=lambda t: t.values[i])
            fig.write_html(str(result_dir / f'param_importance_{obj_name.lower().replace(" ", "_")}.html'))
    except Exception as e:
        print(f"Warning: Could not save visualizations: {e}")
    
    # Best trials를 YAML로 저장
    print(f"\nSaving best configurations...")
    for i, trial in enumerate(study.best_trials):
        config_path = result_dir / f'best_config_trial_{trial.number}.yaml'
        
        best_config = trainer.base_config.copy()
        best_config.update(trial.params)
        best_config['val_accuracy'] = trial.values[0]
        best_config['mia_accuracy'] = trial.values[1]
        
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(best_config, f, default_flow_style=False)
        
        print(f"  Config saved to: {config_path}")
    
    print(f"\n{'='*60}")
    print(f"All results saved to: {result_dir}")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()

import os
from functools import partial
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from transformers import BertTokenizer
from opacus import PrivacyEngine
from opacus.validators import ModuleValidator

from data.data import VQADataset, collate_fn_with_tokenizer
from utils.src import train, validate
from utils.util import parse_args, create_model

torch.manual_seed(42)

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    os.makedirs(args.model_save_path, exist_ok=True)
    log_path = os.path.join(args.model_save_path, "log.txt")

    image_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225])
    ])
    
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    full_dataset = VQADataset(root_dir=args.dataset_root, split='train', transform=image_transform)

    total_size = len(full_dataset)
    val_size = int(total_size * args.val_split_ratio)
    train_size = total_size - val_size
    
    print(f"Data Split -> Train: {train_size} samples, Validation: {val_size} samples")
    
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    collate_fn = partial(collate_fn_with_tokenizer, tokenizer=tokenizer)
    
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=True
    )

    # 모델 생성
    model = create_model(args, device)
    
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    
    # DP-SGD 설정
    privacy_engine = None
    use_dp_sgd = getattr(args, 'dp_sgd', False)
    
    if use_dp_sgd:
        target_epsilon = float(getattr(args, 'target_epsilon', 3.0))
        target_delta = float(getattr(args, 'target_delta', 1e-5))
        max_grad_norm = float(getattr(args, 'max_grad_norm', 1.0))
        
        print(f"\n=== DP-SGD Configuration ===")
        print(f"Target ε (epsilon): {target_epsilon}, Target δ (delta): {target_delta}, Max gradient norm: {max_grad_norm}")
        print("\nConverting BatchNorm to GroupNorm in classifier...")
        model.classifier = ModuleValidator.fix(model.classifier)
        model = model.to(device)
        
        privacy_engine = PrivacyEngine(secure_mode=False)
        
        # classifier optimizer
        classifier_optimizer = optim.AdamW(model.classifier.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        
        model.classifier, classifier_optimizer, train_loader = privacy_engine.make_private_with_epsilon(
            module=model.classifier,
            optimizer=classifier_optimizer,
            data_loader=train_loader,
            epochs=args.epochs,
            target_epsilon=target_epsilon,
            target_delta=target_delta,
            max_grad_norm=max_grad_norm,
            poisson_sampling=True,
        )
        
        # non classifier optimizer
        non_classifier_params = [
            p for n, p in model.named_parameters() 
            if not n.startswith('classifier.') and p.requires_grad
        ]
        main_optimizer = optim.AdamW(non_classifier_params, lr=args.lr, weight_decay=args.weight_decay)
        
        optimizer = {'main': main_optimizer, 'classifier': classifier_optimizer}
        
        print(f"\n=== DP-SGD Enabled (Classifier Only) ===")
    
    print(f"Vision: {args.Vision}, Text: {args.Text}, Fusion: {args.fusion_type} \n Start Train...")
    
    best_val_acc = 0.0
    patience = 5
    patience_counter = 0
    best_model_save_path = os.path.join(args.model_save_path, "best_model.pth")

    with open(log_path, "w") as log_file:
        for epoch in range(args.epochs):
            print(f"\n--- Epoch {epoch+1}/{args.epochs} ---")

            # DP-SGD with VQAModel_IB는 아직 미구현
            if use_dp_sgd and args.model == "VQAModel_IB":
                raise NotImplementedError("DP-SGD with VQAModel_IB is not yet implemented")
            
            # Train
            lambda_ib = getattr(args, 'lambda_ib', 0.01) if args.model == "VQAModel_IB" else 0.01
            train_loss, train_acc = train(
                model=model,
                train_loader=train_loader,
                optimizer=optimizer,
                criterion=criterion,
                device=device,
                model_type=args.model,
                use_dp_sgd=use_dp_sgd,
                lambda_ib=lambda_ib
            )

            # Validation
            val_loss, val_acc, metrics = validate(
                model=model,
                val_loader=val_loader,
                criterion=criterion,
                device=device,
                model_type=args.model,
                mode="val"
            )
            
            # Logging
            log_msg = f"Epoch {epoch+1} | Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%"
            
            if use_dp_sgd:
                epsilon = privacy_engine.get_epsilon(target_delta)
                log_msg += f" | ε: {epsilon:.2f}"
            
            print(log_msg)
            log_file.write(log_msg + "\n")
            log_file.flush()

            if not use_dp_sgd:  # DP-SGD는 scheduler 사용 안 함
                scheduler.step()

            # save_path = os.path.join(args.model_save_path, f"epoch_{epoch+1}.pth")
            # torch.save(model.state_dict(), save_path)
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), best_model_save_path)
                # print(f"New Best Model: {save_path}")
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                print(f"No improvement in validation accuracy for {patience} consecutive epochs. Early stopping.")
                break

    print(f"\n--- Train End ---")
    print(f"Best Validation ACC: {best_val_acc:.2f}%")
    
    if use_dp_sgd:
        final_epsilon = privacy_engine.get_epsilon(target_delta)
        print(f"\n=== Final Privacy Budget ===")
        print(f"ε (epsilon): {final_epsilon:.2f}")
        print(f"δ (delta): {target_delta}")
        print(f"============================\n")


if __name__ == '__main__':
    main()
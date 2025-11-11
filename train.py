import os
import yaml
import traceback
import argparse
from pathlib import Path
from functools import partial

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from transformers import BertTokenizer, RobertaTokenizer

from data.data import VQADataset, collate_fn_with_tokenizer
from utils.src import train, validate
from model.vision_encoder import CNN, ResNet50, SwinTransformer
from model.text_encoder import Bert, RoBerta, BertQLoRA, RoBertaQLoRA
from model.model import VQAModel

torch.manual_seed(42)

def parse_args():
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument('--cfg', '-c', type=str, default=None, help='path to YAML config file')
    known, remaining = p.parse_known_args()

    cfg_from_file = {}
    if known.cfg:
        cfg_path = Path(known.cfg)
        if cfg_path.exists():
            with open(cfg_path, 'r', encoding='utf-8') as f:
                cfg_from_file = yaml.safe_load(f) or {}
        else:
            raise FileNotFoundError(f"Config file not found: {known.cfg}")
        
    args_obj = argparse.Namespace(**cfg_from_file)
    return args_obj

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

    VISION_MODELS = {
        "CNN": CNN,
        "ResNet50": ResNet50,
        "SwinTransformer": SwinTransformer
    }
    TEXT_MODELS = {
        "Bert": Bert,
        "RoBerta": RoBerta,
        "BertQLoRA": BertQLoRA, 
        "RoBertaQLoRA": RoBertaQLoRA
    }

    vision_class = VISION_MODELS.get(args.Vision)
    text_class = TEXT_MODELS.get(args.Text)

    model = VQAModel(vision=vision_class, text=text_class, fusion_type=args.fusion_type, num_classes=args.num_classes).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    print(f"Vision: {args.Vision}, Text: {args.Text}, Fusion: {args.fusion_type} \n Start Train...")
    
    best_val_acc = 0.0
    patience = 7  # 5 에포크 동안 Val Acc가 개선되지 않으면 중단
    patience_counter = 0
    best_model_save_path = os.path.join(args.model_save_path, "best_model.pth")

    with open(log_path, "w") as log_file:
        for epoch in range(args.epochs):
            print(f"\n--- Epoch {epoch+1}/{args.epochs} ---")

            # Train
            train_loss, train_acc = train(model, train_loader, optimizer, criterion, device)

            # Validation
            val_loss, val_acc, metrics = validate(model, val_loader, criterion, device, mode="val")

            log_msg = f"Epoch {epoch+1} | Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%"
            print(log_msg)
            log_file.write(log_msg + "\n")
            log_file.flush()

            scheduler.step()

            save_path = os.path.join(args.model_save_path, f"epoch_{epoch+1}.pth")
            torch.save(model.state_dict(), save_path)
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), best_model_save_path)
                print(f"New Best Model: {save_path}")
                patience_counter = 0  # Reset counter if validation accuracy improves
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                print(f"No improvement in validation accuracy for {patience} consecutive epochs. Early stopping.")
                break

    print(f"\n--- Train End ---")
    print(f"Best Validation ACC: {best_val_acc:.2f}%")


if __name__ == '__main__':
    main()
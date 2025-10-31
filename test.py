import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from transformers import BertTokenizer
from functools import partial
import argparse
import yaml
from pathlib import Path

from data.data import VQADataset, collate_fn_with_tokenizer
from utils.src import validate
from model.vision_encoder import CNN, ResNet50
from model.text_encoder import Bert
from model.model import VQAModel

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
    
    test_dataset = VQADataset(root_dir=args.dataset_root, 
                                split='test',
                                transform=image_transform)

    # 3. DataLoader 생성
    collate_fn = partial(collate_fn_with_tokenizer, tokenizer=tokenizer)
    
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=True
    )

    VISION_MODELS = {
        "CNN": CNN,
        "ResNet50": ResNet50
    }
    TEXT_MODELS = {
        "Bert": Bert
    }

    vision_class = VISION_MODELS.get(args.Vision)
    text_class = TEXT_MODELS.get(args.Text)

    # --- 모델, 옵티마이저, 손실함수 준비 ---
    model = VQAModel(vision=vision_class, text=text_class, fusion_type=args.fusion_type, num_classes=args.num_classes).to(device)
    criterion = nn.CrossEntropyLoss()

    model.load_state_dict(torch.load('./checkpoints/ResNet50_BERT_attention/best_model_epoch_5_acc_100.00.pth', map_location=device))

    model.eval()

    with torch.no_grad():
        val_loss, val_acc = validate(model, test_loader, criterion, device)
    
    print("\n--- Test Results ---")
    print(f"Model: {model.__class__.__name__}, Vision: {args.Vision}, Text: {args.Text}, Fusion: {args.fusion_type}")
    print(f"Eval Loss: {val_loss:.4f}, Eval Acc: {val_acc:.2f}%")


# --- 7. 스크립트 실행 ---
if __name__ == '__main__':
    main()
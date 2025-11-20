import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from transformers import BertTokenizer
from functools import partial

from data.data import VQADataset, collate_fn_with_tokenizer
from utils.src import validate
from utils.util import parse_args, create_model, load_weights

def main():
    args = parse_args(require_weights=True)  # test에서는 weights 필수
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

    collate_fn = partial(collate_fn_with_tokenizer, tokenizer=tokenizer)
    
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=True
    )

    # 모델 생성
    model = create_model(args, device)
    
    criterion = nn.CrossEntropyLoss()

    # 가중치 로드 (DP-SGD 호환)
    model = load_weights(model, args.weights, device)

    model.eval()

    print(f"Model: {model.__class__.__name__}, Vision: {args.Vision}, Text: {args.Text}, Fusion: {args.fusion_type}")

    mode = "test"

    with torch.no_grad():
        val_loss, val_acc, metrics = validate(
            model=model,
            val_loader=test_loader,
            criterion=criterion,
            device=device,
            model_type=args.model,
            mode=mode,
            weight_path=args.weights
        )

    print("--- Test Results ---")
    print(f"Loss: {metrics['loss']:.4f}, Accuracy: {metrics['accuracy']:.2f}%\n")
    if 'precision' in metrics:
        print(f"Precision: {metrics['precision']:.4f}, Recall: {metrics['recall']:.4f}, F1 Score: {metrics['f1']:.4f}\n")

if __name__ == '__main__':
    main()
"""
Token Pruning Visualization Tool

Visualize which tokens are selected/pruned in both vision and text modalities.
Usage:
    python visualize_pruning.py -c ./cfg/cocoqa/Res_Bert_Lora_pruning.yaml \
        -w ./checkpoints/cocoqa/Res_Bert_Lora_pruning/best_model.pth \
        --num_samples 10
"""
import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from transformers import BertTokenizer
from functools import partial
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

from data.data import VQADataset, collate_fn_with_tokenizer
from utils.util import parse_args, create_model, load_weights

# 한글 폰트 설정 (Windows)
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False


def extract_pruning_info(model, images, input_ids, attention_mask, device):
    """
    모델의 fusion 과정에서 pruning 정보 추출
    
    Returns:
        vision_info: dict with 'selected_indices', 'attention_weights', 'original_shape'
        text_info: dict with 'selected_indices', 'attention_weights', 'tokens'
    """
    model.eval()
    
    vision_info = None
    text_info = None
    
    with torch.no_grad():
        # Encoder 통과
        v_seq, v_global = model.vision_encoder(images)
        q_seq, q_global = model.text_encoder(input_ids, attention_mask)
        
        # Fusion module에서 pruning 정보 추출
        fusion_module = model.fusion_module
        
        if hasattr(fusion_module, 'use_pruning') and fusion_module.use_pruning:
            if hasattr(fusion_module, 'txt_to_vis_attn'):  # CoAttention
                # Text->Vision attention
                q_global_unsq = q_global.unsqueeze(1)
                v_global_unsq = v_global.unsqueeze(1)
                
                # Vision token pruning
                _, v_attn_weights = fusion_module.txt_to_vis_attn(
                    query=q_global_unsq, key=v_seq, value=v_seq
                )
                
                # Pruning module의 로직 재현
                B, N, D = v_seq.shape
                if v_attn_weights.dim() == 3:
                    v_attn_weights = v_attn_weights.squeeze(1)
                
                k_v = max(1, int(N * (1 - fusion_module.v_pruning.prune_ratio)))
                
                # Temperature scaling & top-k
                soft_weights_v = torch.softmax(v_attn_weights / fusion_module.v_pruning.temperature, dim=-1)
                _, v_top_indices = torch.topk(soft_weights_v, k=k_v, dim=-1)
                
                vision_info = {
                    'selected_indices': v_top_indices.cpu().numpy(),
                    'attention_weights': v_attn_weights.cpu().numpy(),
                    'soft_weights': soft_weights_v.cpu().numpy(),
                    'original_shape': v_seq.shape,
                    'num_selected': k_v
                }
                
                # Text token pruning
                _, q_attn_weights = fusion_module.vis_to_txt_attn(
                    query=v_global_unsq, key=q_seq, value=q_seq
                )
                
                B_q, N_q, D_q = q_seq.shape
                if q_attn_weights.dim() == 3:
                    q_attn_weights = q_attn_weights.squeeze(1)
                
                k_q = max(1, int(N_q * (1 - fusion_module.q_pruning.prune_ratio)))
                
                soft_weights_q = torch.softmax(q_attn_weights / fusion_module.q_pruning.temperature, dim=-1)
                _, q_top_indices = torch.topk(soft_weights_q, k=k_q, dim=-1)
                
                text_info = {
                    'selected_indices': q_top_indices.cpu().numpy(),
                    'attention_weights': q_attn_weights.cpu().numpy(),
                    'soft_weights': soft_weights_q.cpu().numpy(),
                    'original_shape': q_seq.shape,
                    'num_selected': k_q
                }
                
            elif hasattr(fusion_module, 'attention'):  # Attention or GatedFusion
                q_global_unsq = q_global.unsqueeze(1)
                
                # Vision token pruning
                _, v_attn_weights = fusion_module.attention(
                    query=q_global_unsq, key=v_seq, value=v_seq
                )
                
                B, N, D = v_seq.shape
                if v_attn_weights.dim() == 3:
                    v_attn_weights = v_attn_weights.squeeze(1)
                
                k_v = max(1, int(N * (1 - fusion_module.pruning.prune_ratio)))
                
                soft_weights_v = torch.softmax(v_attn_weights / fusion_module.pruning.temperature, dim=-1)
                _, v_top_indices = torch.topk(soft_weights_v, k=k_v, dim=-1)
                
                vision_info = {
                    'selected_indices': v_top_indices.cpu().numpy(),
                    'attention_weights': v_attn_weights.cpu().numpy(),
                    'soft_weights': soft_weights_v.cpu().numpy(),
                    'original_shape': v_seq.shape,
                    'num_selected': k_v
                }
    
    return vision_info, text_info


def visualize_vision_pruning(image, vision_info, save_path):
    """
    Vision token pruning 시각화
    
    Args:
        image: PIL Image or numpy array (H, W, 3)
        vision_info: pruning 정보 dict
        save_path: 저장 경로
    """
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    # 원본 spatial grid 크기 계산
    # ResNet50: (224, 224) -> (7, 7) feature map -> 49 tokens (after global pooling 제외시)
    # 실제로는 v_seq가 (B, 784, 512)라면 28x28 grid
    B, N, D = vision_info['original_shape']
    grid_size = int(np.sqrt(N))  # 28 or 7
    
    selected_indices = vision_info['selected_indices'][0]  # 첫 번째 샘플
    attention_weights = vision_info['soft_weights'][0]  # Soft attention weights
    
    # Figure 생성
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # 1. Original Image
    axes[0].imshow(image)
    axes[0].set_title('Original Image', fontsize=14, fontweight='bold')
    axes[0].axis('off')
    
    # 2. Attention Heatmap (모든 토큰)
    attention_map = attention_weights.reshape(grid_size, grid_size)
    im1 = axes[1].imshow(attention_map, cmap='hot', interpolation='nearest')
    axes[1].set_title(f'Attention Weights ({grid_size}x{grid_size} tokens)', 
                      fontsize=14, fontweight='bold')
    axes[1].axis('off')
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    
    # 3. Selected Tokens Visualization
    # 선택된 토큰만 표시
    selected_map = np.zeros((grid_size, grid_size))
    for idx in selected_indices:
        row = idx // grid_size
        col = idx % grid_size
        if row < grid_size and col < grid_size:
            selected_map[row, col] = attention_weights[idx]
    
    im2 = axes[2].imshow(selected_map, cmap='hot', interpolation='nearest')
    
    # 선택된 토큰 위치에 테두리 표시
    for idx in selected_indices:
        row = idx // grid_size
        col = idx % grid_size
        if row < grid_size and col < grid_size:
            rect = patches.Rectangle((col - 0.5, row - 0.5), 1, 1, 
                                     linewidth=1, edgecolor='cyan', facecolor='none')
            axes[2].add_patch(rect)
    
    retention_rate = len(selected_indices) / N * 100
    axes[2].set_title(f'Selected Tokens ({len(selected_indices)}/{N} = {retention_rate:.1f}%)', 
                      fontsize=14, fontweight='bold')
    axes[2].axis('off')
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Vision pruning 시각화 저장: {save_path}")


def visualize_text_pruning(tokens, text_info, save_path, tokenizer):
    """
    Text token pruning 시각화
    
    Args:
        tokens: token IDs (list or tensor)
        text_info: pruning 정보 dict
        save_path: 저장 경로
        tokenizer: BERT tokenizer
    """
    if text_info is None:
        print("⚠ Text pruning 정보 없음 (CoAttention 아닌 경우 text pruning 안 됨)")
        return
    
    # Token ID를 텍스트로 변환
    if isinstance(tokens, torch.Tensor):
        tokens = tokens.cpu().numpy()
    
    token_texts = tokenizer.convert_ids_to_tokens(tokens)
    
    selected_indices = text_info['selected_indices'][0]
    attention_weights = text_info['soft_weights'][0]
    
    num_tokens = len(token_texts)
    
    # Figure 생성
    fig, axes = plt.subplots(2, 1, figsize=(16, 8))
    
    # 1. All Tokens with Attention Weights
    colors_all = plt.cm.Reds(attention_weights / attention_weights.max())
    
    axes[0].barh(range(num_tokens), attention_weights, color=colors_all)
    axes[0].set_yticks(range(num_tokens))
    axes[0].set_yticklabels(token_texts, fontsize=10)
    axes[0].set_xlabel('Attention Weight', fontsize=12)
    axes[0].set_title(f'All Tokens ({num_tokens} tokens)', fontsize=14, fontweight='bold')
    axes[0].invert_yaxis()
    axes[0].grid(axis='x', alpha=0.3)
    
    # 2. Selected Tokens Only
    selected_weights = np.zeros(num_tokens)
    for idx in selected_indices:
        if idx < num_tokens:
            selected_weights[idx] = attention_weights[idx]
    
    colors_selected = ['green' if idx in selected_indices else 'lightgray' 
                      for idx in range(num_tokens)]
    
    axes[1].barh(range(num_tokens), selected_weights, color=colors_selected)
    axes[1].set_yticks(range(num_tokens))
    axes[1].set_yticklabels([token_texts[i] if i in selected_indices else '' 
                             for i in range(num_tokens)], fontsize=10)
    axes[1].set_xlabel('Attention Weight', fontsize=12)
    
    retention_rate = len(selected_indices) / num_tokens * 100
    axes[1].set_title(f'Selected Tokens ({len(selected_indices)}/{num_tokens} = {retention_rate:.1f}%)', 
                      fontsize=14, fontweight='bold')
    axes[1].invert_yaxis()
    axes[1].grid(axis='x', alpha=0.3)
    
    # 선택된 토큰을 텍스트로 표시
    selected_text = ' '.join([token_texts[i] for i in sorted(selected_indices) if i < num_tokens])
    fig.text(0.5, 0.02, f'Selected: {selected_text}', 
             ha='center', fontsize=10, wrap=True, color='green', fontweight='bold')
    
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Text pruning 시각화 저장: {save_path}")


def main():
    parser = argparse.ArgumentParser(description='Token Pruning Visualization')
    parser.add_argument('--cfg', '-c', type=str, required=True, help='Config file path')
    parser.add_argument('--weights', '-w', type=str, required=True, help='Model weights path')
    parser.add_argument('--num_samples', type=int, default=10, help='Number of samples to visualize')
    parser.add_argument('--output_dir', type=str, default='./pruning_visualizations', 
                       help='Output directory for visualizations')
    parser.add_argument('--split', type=str, default='test', choices=['train', 'test'],
                       help='Dataset split to use')
    cmd_args = parser.parse_args()
    
    # Parse config
    import sys
    sys.argv = ['visualize_pruning.py', '--cfg', cmd_args.cfg, '--weights', cmd_args.weights]
    args = parse_args(require_weights=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")
    
    # Check if pruning is enabled
    if not getattr(args, 'use_token_pruning', False):
        print("⚠️ WARNING: use_token_pruning is False in config!")
        print("   Pruning 시각화를 보려면 use_token_pruning: True로 설정해야 합니다.")
        return
    
    os.makedirs(cmd_args.output_dir, exist_ok=True)
    
    # Load data
    image_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225])
    ])
    
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    dataset = VQADataset(root_dir=args.dataset_root, split=cmd_args.split, transform=image_transform)
    
    collate_fn = partial(collate_fn_with_tokenizer, tokenizer=tokenizer)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)
    
    # Load model
    model = create_model(args, device)
    model = load_weights(model, args.weights, device)
    print(f"✓ Model loaded from {args.weights}\n")
    
    # Check fusion type
    print(f"Fusion type: {args.fusion_type}")
    print(f"Prune ratio: {getattr(args, 'prune_ratio', 0.5)}")
    print(f"Noise scale: {getattr(args, 'noise_scale', 0.1)}\n")
    
    # Visualize samples
    print(f"Visualizing {cmd_args.num_samples} samples...\n")
    
    for idx, batch in enumerate(dataloader):
        if idx >= cmd_args.num_samples:
            break
        
        # Load original image (unnormalized)
        img_id = batch['image_id'][0]
        img_path_candidates = [
            os.path.join(args.dataset_root, cmd_args.split, 'images', img_id),
            os.path.join(args.dataset_root, cmd_args.split, 'images', img_id + '.jpg'),
            os.path.join(args.dataset_root, cmd_args.split, 'images', img_id + '.png'),
        ]
        
        original_image = None
        for path in img_path_candidates:
            if os.path.exists(path):
                original_image = Image.open(path).convert('RGB')
                break
        
        if original_image is None:
            print(f"⚠ Image not found: {img_id}")
            continue
        
        # Get tokens and question
        images = batch['image'].to(device)
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        question = batch['question'][0]
        answer_text = batch['answer_text'][0] if 'answer_text' in batch else 'N/A'
        
        # Extract pruning info
        vision_info, text_info = extract_pruning_info(
            model, images, input_ids, attention_mask, device
        )
        
        # Visualize
        sample_name = f"sample_{idx+1:03d}"
        
        if vision_info is not None:
            vision_save_path = os.path.join(cmd_args.output_dir, f"{sample_name}_vision.png")
            visualize_vision_pruning(original_image, vision_info, vision_save_path)
        
        if text_info is not None:
            text_save_path = os.path.join(cmd_args.output_dir, f"{sample_name}_text.png")
            visualize_text_pruning(input_ids[0], text_info, text_save_path, tokenizer)
        
        # Save metadata
        metadata_path = os.path.join(cmd_args.output_dir, f"{sample_name}_info.txt")
        with open(metadata_path, 'w', encoding='utf-8') as f:
            f.write(f"Sample {idx+1}\n")
            f.write(f"="*50 + "\n")
            f.write(f"Image ID: {img_id}\n")
            f.write(f"Question: {question}\n")
            f.write(f"Answer: {answer_text}\n\n")
            
            if vision_info is not None:
                retention = len(vision_info['selected_indices'][0]) / vision_info['original_shape'][1] * 100
                f.write(f"Vision Tokens:\n")
                f.write(f"  Total: {vision_info['original_shape'][1]}\n")
                f.write(f"  Selected: {len(vision_info['selected_indices'][0])}\n")
                f.write(f"  Retention: {retention:.1f}%\n\n")
            
            if text_info is not None:
                retention = len(text_info['selected_indices'][0]) / text_info['original_shape'][1] * 100
                tokens_text = tokenizer.convert_ids_to_tokens(input_ids[0].cpu().numpy())
                selected_tokens = [tokens_text[i] for i in text_info['selected_indices'][0] 
                                  if i < len(tokens_text)]
                
                f.write(f"Text Tokens:\n")
                f.write(f"  Total: {text_info['original_shape'][1]}\n")
                f.write(f"  Selected: {len(text_info['selected_indices'][0])}\n")
                f.write(f"  Retention: {retention:.1f}%\n")
                f.write(f"  Selected tokens: {' '.join(selected_tokens)}\n")
        
        print(f"✓ Sample {idx+1} 완료")
    
    print(f"\n{'='*60}")
    print(f"모든 시각화 완료!")
    print(f"저장 위치: {cmd_args.output_dir}")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()

"""
Calibration 기반 공격에서 공통으로 사용되는 함수들
Shadow model 기반 난이도 추정 및 loss 보정 관련 함수
"""
import os
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from PIL import Image
from transformers import (
    BlipProcessor, BlipForQuestionAnswering,
    ViltProcessor, ViltForQuestionAnswering,
    AutoProcessor, AutoModelForVisualQuestionAnswering,
    AutoProcessor as GitProcessor, AutoModelForCausalLM,
    LxmertTokenizer, LxmertForQuestionAnswering
)

from attack.metric_src import setup_data_loaders


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


def collect_target_losses(model, dataloader, device, model_type="VQAModel"):
    """
    Target 모델의 각 샘플에 대한 loss 수집
    
    Args:
        model: VQA 모델
        dataloader: 데이터 로더
        device: 디바이스
        model_type: 모델 타입
        
    Returns:
        losses: 샘플별 loss 값 (numpy array)
    """
    model.eval()
    all_losses = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Collecting target losses", leave=False):
            images = batch['image'].to(device)
            inputs = batch['inputs'].to(device)
            answers = batch['answer'].to(device)
            
            out = model(
                images=images,
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask']
            )
            outputs = out[0] if isinstance(out, tuple) else out
            losses = compute_sample_loss(outputs, answers)
            all_losses.extend(losses.cpu().numpy())
    
    return np.array(all_losses)


def estimate_difficulty(shadow_losses):
    """
    Shadow model들의 loss 분포로 샘플 난이도 추정
    
    Args:
        shadow_losses: Shadow model들의 loss 값 (num_shadows, num_samples)
        
    Returns:
        difficulties: 각 샘플의 난이도 (평균 loss) (num_samples,)
    """
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
    calibrated_loss = target_loss - difficulty
    return calibrated_loss


def load_pretrained_vqa_model(model_name, device, weights_path=None):
    """
    Hugging Face에서 pre-trained VQA model 로드
    
    Args:
        model_name: 모델 이름
        device: 디바이스
        weights_path: Fine-tuned weights 경로 (optional)
        
    Returns:
        model, processor
    """
    print(f"Loading {model_name}...")
    
    if "vilt" in model_name.lower():
        processor = ViltProcessor.from_pretrained(model_name)
        model = ViltForQuestionAnswering.from_pretrained(model_name, use_safetensors=True).to(device)
    elif "blip" in model_name.lower():
        processor = BlipProcessor.from_pretrained(model_name)
        model = BlipForQuestionAnswering.from_pretrained(model_name).to(device)
    elif "git" in model_name.lower():
        processor = GitProcessor.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    else:
        processor = AutoProcessor.from_pretrained(model_name)
        model = AutoModelForVisualQuestionAnswering.from_pretrained(model_name).to(device)
    
    # Load fine-tuned weights if provided
    if weights_path is not None:
        print(f"Loading fine-tuned weights from {weights_path}...")
        state_dict = torch.load(weights_path, map_location=device, weights_only=False)
        model.load_state_dict(state_dict, strict=False)
        print(f"✓ Fine-tuned weights loaded")
    
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
        model_name: 모델 이름
        dataset_root: 데이터셋 루트 경로
        split: 'train' 또는 'test'
        
    Returns:
        losses: 각 샘플의 loss 값 (numpy array)
    """
    all_losses = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Shadow: {model_name.split('/')[-1]}", leave=False):
            try:
                image_paths = []
                for img_id in batch['image_id']:
                    base_path = os.path.join(dataset_root, split, 'images', img_id)
                    found = False
                    for ext in ['.jpg', '.png', '.jpeg']:
                        if os.path.exists(base_path + ext):
                            image_paths.append(base_path + ext)
                            found = True
                            break
                    if not found:
                        if os.path.exists(base_path):
                            image_paths.append(base_path)
                        else:
                            raise FileNotFoundError(f"Image not found: {base_path}")
                
                images = [Image.open(path).convert('RGB') for path in image_paths]
                questions = batch['question']
                answers = batch['answer_text'] if 'answer_text' in batch else [str(ans) for ans in batch['answer']]
                
                if "blip" in model_name.lower():
                    inputs = processor(images=images, text=questions, return_tensors="pt", padding=True, truncation=True).to(device)
                    labels = processor(text=answers, return_tensors="pt", padding=True, truncation=True).input_ids.to(device)
                    outputs = shadow_model(**inputs, labels=labels)
                    loss = outputs.loss
                    batch_size = len(images)
                    sample_losses = [loss.item()] * batch_size
                    
                elif "vilt" in model_name.lower():
                    inputs = processor(images=images, text=questions, return_tensors="pt", padding=True, truncation=True, max_length=40).to(device)
                    outputs = shadow_model(**inputs)
                    logits = outputs.logits
                    probs = torch.softmax(logits, dim=-1)
                    entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1)
                    sample_losses = entropy.cpu().numpy().tolist()
                    
                elif "git" in model_name.lower():
                    # GIT는 generative model이므로 각 질문에 대해 생성 시도
                    batch_size = len(images)
                    sample_losses = []
                    for img, question in zip(images, questions):
                        inputs = processor(images=img, text=question, return_tensors="pt").to(device)
                        # Generative model이므로 perplexity 기반 난이도 추정
                        with torch.no_grad():
                            outputs = shadow_model.generate(**inputs, max_length=50, return_dict_in_generate=True, output_scores=True)
                            # 생성된 토큰의 평균 확률로 난이도 추정 (낮을수록 어려움)
                            if hasattr(outputs, 'scores') and len(outputs.scores) > 0:
                                avg_score = torch.stack(outputs.scores).mean().item()
                                sample_losses.append(-avg_score)  # negative log prob와 유사하게
                            else:
                                sample_losses.append(1.0)  # fallback
                    
                else:
                    inputs = processor(images=images, text=questions, return_tensors="pt", padding=True, truncation=True).to(device)
                    outputs = shadow_model(**inputs)
                    logits = outputs.logits
                    probs = torch.softmax(logits, dim=-1)
                    entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1)
                    sample_losses = entropy.cpu().numpy().tolist()
                
                all_losses.extend(sample_losses)
                
            except Exception as e:
                print(f"\n⚠ Error processing batch: {e}")
                batch_size = len(batch['image'])
                all_losses.extend([1.0] * batch_size)
                continue
    
    return np.array(all_losses)


# Shadow model 별칭 매핑
SHADOW_MODEL_ALIASES = {
    'blip': 'Salesforce/blip-vqa-base',
    'vilt': 'dandelin/vilt-b32-finetuned-vqa',
    'git': 'microsoft/git-base-vqav2',
}


def resolve_model_name(model_name):
    """
    모델 이름을 별칭에서 전체 이름으로 변환
    
    Args:
        model_name: 짧은 별칭 또는 전체 모델 이름
        
    Returns:
        resolved_name: 전체 Hugging Face model identifier
    """
    # 이미 전체 경로면 그대로 반환
    if '/' in model_name:
        return model_name
    
    # 별칭 매핑 확인
    if model_name.lower() in SHADOW_MODEL_ALIASES:
        return SHADOW_MODEL_ALIASES[model_name.lower()]
    
    # 매칭 안되면 원본 반환 (에러는 나중에 처리)
    return model_name


def collect_shadow_difficulties(args, device, shadow_models=None, reference_weights=None):
    """
    Shadow model들로부터 난이도 추정
    
    Args:
        args: 설정 인자
        device: 디바이스
        shadow_models: Shadow model 이름 리스트 (별칭 또는 전체 이름)
        reference_weights: Reference model의 fine-tuned weights 경로 (optional)
        
    Returns:
        train_difficulty, test_difficulty
    """
    if shadow_models is None:
        shadow_models = ["blip"]
    
    # 별칭을 전체 모델 이름으로 변환
    resolved_models = [resolve_model_name(name) for name in shadow_models]
    
    print(f"\n=== Using {len(shadow_models)} Shadow Model(s) ===")
    for idx, (original, resolved) in enumerate(zip(shadow_models, resolved_models)):
        if original != resolved:
            print(f"{idx+1}. {original} -> {resolved}")
        else:
            print(f"{idx+1}. {resolved}")
    
    train_loader, test_loader, _, _ = setup_data_loaders(args, seed=42)
    
    shadow_train_losses = []
    shadow_test_losses = []
    
    for idx, (original_name, model_name) in enumerate(zip(shadow_models, resolved_models)):
        print(f"\n{'='*60}")
        print(f"Processing: {original_name} ({model_name})")
        print(f"{'='*60}")
        
        try:
            # Reference weights는 첫 번째 모델에만 적용
            weights = reference_weights if idx == 0 and reference_weights else None
            shadow_model, processor = load_pretrained_vqa_model(model_name, device, weights)
            
            train_losses = collect_shadow_losses(
                shadow_model, processor, train_loader,
                device, model_name, args.dataset_root, split='train'
            )
            test_losses = collect_shadow_losses(
                shadow_model, processor, test_loader,
                device, model_name, args.dataset_root, split='train'
            )
            
            shadow_train_losses.append(train_losses)
            shadow_test_losses.append(test_losses)
            
            print(f"✓ Collected - Train: {len(train_losses)}, Test: {len(test_losses)}")
            
            del shadow_model, processor
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"✗ Error with {model_name}: {e}")
            continue
    
    if len(shadow_train_losses) == 0:
        raise RuntimeError("No shadow models loaded. Cannot proceed.")
    
    train_difficulty = estimate_difficulty(np.array(shadow_train_losses))
    test_difficulty = estimate_difficulty(np.array(shadow_test_losses))
    
    return train_difficulty, test_difficulty

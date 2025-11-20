import argparse
import yaml
from pathlib import Path
import torch

from model.vision_encoder import CNN, ResNet50, SwinTransformer
from model.text_encoder import Bert, RoBerta, BertQLoRA, RoBertaQLoRA
from model.model import VQAModel, VQAModel_IB

def parse_args(require_weights=False):
    """
    YAML 설정 파일과 커맨드라인 인자를 파싱
    
    Args:
        require_weights: weights 인자가 필수인지 여부 (기본값: False)
                        - True: test.py 등에서 사용 (weights 필수)
                        - False: train.py 등에서 사용 (weights 선택)
    
    Returns:
        args_obj: 파싱된 인자 객체
    """
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument('--cfg', '-c', type=str, default=None, help='path to YAML config file')
    p.add_argument('--weights','-w', type=str, default=None, help='path to model weights file (optional for training)')
    known, remaining = p.parse_known_args()

    cfg_from_file = {}
    if known.cfg:
        cfg_path = Path(known.cfg)
        if cfg_path.exists():
            with open(cfg_path, 'r', encoding='utf-8') as f:
                cfg_from_file = yaml.safe_load(f) or {}
        else:
            raise FileNotFoundError(f"Config file not found: {known.cfg}")
    
    args_dict = {**cfg_from_file, 'weights': known.weights}
    
    # weights가 필수인 경우에만 체크
    if require_weights and not args_dict['weights']:
        raise ValueError("--weights/-w argument is required")
        
    args_obj = argparse.Namespace(**args_dict)
    return args_obj


# Model registry - 모델 추가 시 여기에만 등록하면 됨
def get_model_registry():
    """모델 레지스트리 반환 - import 순환 문제 해결을 위한 lazy loading"""
    
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
    
    MODEL_CLASSES = {
        "VQAModel": VQAModel,
        "VQAModel_IB": VQAModel_IB
    }
    
    return VISION_MODELS, TEXT_MODELS, MODEL_CLASSES


def create_model(args, device):
    """모델 생성 헬퍼 함수"""
    VISION_MODELS, TEXT_MODELS, MODEL_CLASSES = get_model_registry()
    
    vision_class = VISION_MODELS.get(args.Vision)
    text_class = TEXT_MODELS.get(args.Text)
    model_class = MODEL_CLASSES.get(args.model)
    
    if not vision_class:
        raise ValueError(f"Unknown vision model: {args.Vision}. Available: {list(VISION_MODELS.keys())}")
    if not text_class:
        raise ValueError(f"Unknown text model: {args.Text}. Available: {list(TEXT_MODELS.keys())}")
    if not model_class:
        raise ValueError(f"Unknown model: {args.model}. Available: {list(MODEL_CLASSES.keys())}")
    
    # 모델별 추가 파라미터
    model_kwargs = {
        'vision': vision_class,
        'text': text_class,
        'fusion_type': args.fusion_type,
        'num_classes': args.num_classes
    }
    
    if args.model == "VQAModel_IB":
        model_kwargs['bottleneck_dim'] = getattr(args, 'bottleneck_dim', 256)
        model_kwargs['beta'] = getattr(args, 'beta', 0.1)
    
    return model_class(**model_kwargs).to(device)


def load_weights(model, weight_path, device):
    """가중치 로드 헬퍼 함수 - DP-SGD 호환"""
    state_dict = torch.load(weight_path, map_location=device, weights_only=False)
    
    # DP-SGD로 학습된 모델은 classifier._module.* 형태로 저장됨
    # 일반 모델 형태로 변환
    new_state_dict = {}
    for key, value in state_dict.items():
        # classifier._module.X.weight -> classifier.X.weight
        new_key = key.replace('classifier._module.', 'classifier.')
        new_state_dict[new_key] = value
    
    # strict=False: DP-SGD 모델은 GroupNorm 사용 (BatchNorm running_mean/var 없음)
    model.load_state_dict(new_state_dict, strict=False)
    print(f"Weights loaded from {weight_path}")
    return model


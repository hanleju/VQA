# VQA: Privacy-Preserving Visual Question Answering with Token Pruning and Mixing

**Token Pruning**과 **Token Mixing** 기법을 결합하여 VQA 모델의 **Privacy Robustness**를 향상시키면서도 모델 성능 저하를 최소화하는 방법론을 제안합니다.

## 🔬 Method

### 1. Privacy-Aware Token Pruning
Vision encoder에서 추출된 이미지 토큰들에 대해 attention score 기반으로 중요하지 않은 토큰들을 제거합니다.

### 2. Adversarial Token Mixing
Privacy를 강화하기 위해 선택된 중요 토큰 중 일부를 덜 중요한 토큰과 교체합니다.

### 3. Token Mixup
제거된 토큰들의 정보를 완전히 버리지 않고, 평균화하여 하나의 토큰으로 추가합니다.

### 4. Noise Injection
선택된 중요 토큰에 노이즈를 추가하여 privacy를 더욱 강화합니다.


## 🎯 Metric

Privacy Robustness를 정량적으로 평가하기 위해 **Membership Inference Attack(MIA)** 기법들을 사용합니다.

- Loss based MIA
- Confidence based MIA
- Difficulty Calibration Attack
- RAPID

### 평가 메트릭
- **Attack Accuracy**: MIA의 정확도 (낮을수록 privacy 강건함)
- **Precision/Recall**: Member 탐지 성능
- **ROC-AUC**: 전반적인 공격 성능
- **PR-AUC**: Precision-Recall 곡선 아래 면적

---

## 🛡️ 비교 방어 기법

- DP-SGD (Differentially Private Stochastic Gradient Descent)

---

## 🚀 사용법

### 1. Download
```bash
pip install -r requirements.txt
```

### 2. Dataset
COCO-QA dataset:
```
cocoqa/
├── train/
│   ├── images/
│   └── questions.json
├── test/
│   ├── images/
│   └── questions.json
└── labels.txt
```

### 3. Train
```bash
# Token Pruning + DP-SGD
python train.py -c ./examples/cfg.yaml

# DP-SGD 없이 학습
python train.py -c ./examples/cfg.yaml --use_dp_sgd false
```

### 4. Test
```bash
python test.py -c ./examples/cfg.yaml -w ./checkpoints/best_model.pth
```

### 5. Privacy Evaluation
```bash
# Difficulty Calibration Attack
python ./attack/calibration.py -c ./examples/cfg.yaml -w ./checkpoints/best_model.pth

# RAPID Attack (Metric-based)
python ./attack/rapid.py -c ./examples/cfg.yaml -w ./checkpoints/best_model.pth --shadow_models blip,vilt,git
```

---

## ⚡ 하이퍼파라미터 최적화

### Optuna를 활용한 Multi-Objective Optimization
**파일**: `optuna_tune.py`

**Optuna** 프레임워크를 사용하여 두 가지 목표를 동시에 최적화합니다:
1. **Accuracy 최대화**: VQA 성능 향상
2. **MIA Attack Accuracy 최소화**: Privacy 강건성 향상

**사용법:**
```bash
python optuna_tune.py -c ./examples/optuna.yaml --n_trials 50
```

---

## 📁 Structure

```
VQA/
├── train.py              # 모델 학습 스크립트
├── test.py               # 모델 평가 스크립트
├── optuna_tune.py        # 하이퍼파라미터 최적화
├── requirements.txt      # 의존성 패키지
├── data/
│   └── data.py          # 데이터셋 클래스
├── model/
│   ├── model.py         # VQA 모델 정의
│   ├── vision_encoder.py
│   └── text_encoder.py
├── utils/
│   ├── fusion.py        # Token Pruning & Mixing 구현
│   ├── src.py           # 학습/검증 함수
│   └── util.py          # 유틸리티 함수
├── attack/
│   ├── calibration.py   # Difficulty Calibration Attack
│   ├── rapid.py         # Metric-based Attack (RAPID)
│   ├── metric_src.py    # 공통 평가 함수
│   └── cali_src.py      # Calibration 유틸리티
└── examples/
    ├── cfg.yaml         # 설정 파일 예시
    └── optuna.yaml      # Optuna 설정 예시
```

---
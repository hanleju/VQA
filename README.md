# VQA: Privacy-Preserving Visual Question Answering with Token Pruning and Mixing

This project proposes a methodology that combines **Token Pruning** and **Token Mixing** techniques to enhance the **Privacy Robustness** of VQA models while minimizing performance degradation.

## 🎯 Metric

We use **Membership Inference Attack (MIA)** techniques to quantitatively evaluate Privacy Robustness.

- Loss based MIA
- Confidence based MIA
- Difficulty Calibration Attack
- RAPID

### Evaluation Metrics
- **Attack Accuracy**: Accuracy of MIA (lower is better for privacy)
- **Precision/Recall**: Member detection performance
- **ROC-AUC**: Overall attack performance
- **PR-AUC**: Area under Precision-Recall curve

---

## 🛡️ Comparative Defense Techniques

- DP-SGD (Differentially Private Stochastic Gradient Descent)

---

## 🚀 Usage

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

# Train without DP-SGD
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

## ⚡ Hyperparameter Optimization

### Multi-Objective Optimization with Optuna
**File**: `optuna_tune.py`

Using the **Optuna** framework, we simultaneously optimize two objectives:
1. **Maximize Accuracy**: Improve VQA performance
2. **Minimize MIA Attack Accuracy**: Enhance privacy robustness

**Usage:**
```bash
python optuna_tune.py -c ./examples/optuna.yaml --n_trials 50
```

---

## 📁 Structure

```
VQA/
├── train.py              # Model training script
├── test.py               # Model evaluation script
├── optuna_tune.py        # Hyperparameter optimization
├── requirements.txt      # Dependencies
├── data/
│   └── data.py          # Dataset class
├── model/
│   ├── model.py         # VQA model definition
│   ├── vision_encoder.py
│   └── text_encoder.py
├── utils/
│   ├── fusion.py        # Token Pruning & Mixing implementation
│   ├── src.py           # Training/validation functions
│   └── util.py          # Utility functions
├── attack/
│   ├── calibration.py   # Difficulty Calibration Attack
│   ├── rapid.py         # Metric-based Attack (RAPID)
│   ├── metric_src.py    # Common evaluation functions
│   └── cali_src.py      # Calibration utilities
└── examples/
    ├── cfg.yaml         # Configuration file example
    └── optuna.yaml      # Optuna configuration example
```

---
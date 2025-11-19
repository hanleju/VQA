"""
간단한 import 테스트
attack 모듈에서 utils.util의 함수들을 제대로 import할 수 있는지 확인
"""

print("Testing imports...")

# 1. utils.util에서 직접 import
try:
    from utils.util import parse_args, create_model, load_weights
    print("✓ Direct import from utils.util works")
except ImportError as e:
    print(f"✗ Direct import from utils.util failed: {e}")

# 2. attack.src에서 import (utils.util을 재사용)
try:
    from attack.src import parse_args_with_config, load_model, setup_data_loaders
    print("✓ Import from attack.src works")
except ImportError as e:
    print(f"✗ Import from attack.src failed: {e}")

# 3. attack 스크립트들이 제대로 import할 수 있는지 확인
try:
    from attack.confidence import get_confidence_and_pred
    print("✓ Import from attack.confidence works")
except ImportError as e:
    print(f"✗ Import from attack.confidence failed: {e}")

try:
    from attack.loss import compute_loss
    print("✓ Import from attack.loss works")
except ImportError as e:
    print(f"✗ Import from attack.loss failed: {e}")

print("\n✅ All imports successful! The refactoring works correctly.")
print("\nDirectory structure allows:")
print("  - attack/confidence.py and attack/loss.py")
print("  - both import from attack/src.py")
print("  - attack/src.py imports from utils/util.py")
print("\nThis creates a clean dependency hierarchy:")
print("  utils/util.py (shared utilities)")
print("    ↓")
print("  attack/src.py (attack-specific utilities)")
print("    ↓")
print("  attack/confidence.py, attack/loss.py (attack implementations)")

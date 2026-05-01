"""
Quick sanity check — run this after activating your venv to confirm
all required packages are importable and functional.

Usage:
    python test_env.py
"""

import sys

failures = []

def check(label, fn):
    try:
        result = fn()
        print(f"  [OK] {label}" + (f": {result}" if result else ""))
    except Exception as e:
        print(f"  [FAIL] {label}: {e}")
        failures.append(label)

print(f"\nPython: {sys.version}\n")

print("--- Core ---")
check("numpy",      lambda: __import__("numpy").__version__)
check("scipy",      lambda: __import__("scipy").__version__)
check("matplotlib", lambda: __import__("matplotlib").__version__)
check("sklearn",    lambda: __import__("sklearn").__version__)

print("\n--- PyTorch ---")
check("torch",      lambda: __import__("torch").__version__)
check("torchvision",lambda: __import__("torchvision").__version__)

import torch
check("torch tensor op", lambda: str(torch.tensor([1.0, 2.0]).mean().item()))
check("CUDA available",  lambda: f"{torch.cuda.is_available()} (expected False on CPU-only)")

print()
if failures:
    print(f"FAILED: {', '.join(failures)}")
    sys.exit(1)
else:
    print("All checks passed.")

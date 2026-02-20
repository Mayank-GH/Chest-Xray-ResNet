import sys
print("Python version:", sys.version)

try:
    import torch
    print("✓ PyTorch version:", torch.__version__)
except ImportError as e:
    print("✗ PyTorch failed:", e)

try:
    import torchvision
    print("✓ Torchvision version:", torchvision.__version__)
except ImportError as e:
    print("✗ Torchvision failed:", e)

try:
    import numpy as np
    print("✓ NumPy version:", np.__version__)
except ImportError as e:
    print("✗ NumPy failed:", e)

try:
    import pandas as pd
    print("✓ Pandas version:", pd.__version__)
except ImportError as e:
    print("✗ Pandas failed:", e)

try:
    import matplotlib
    print("✓ Matplotlib version:", matplotlib.__version__)
except ImportError as e:
    print("✗ Matplotlib failed:", e)

try:
    import cv2
    print("✓ OpenCV version:", cv2.__version__)
except ImportError as e:
    print("✗ OpenCV failed:", e)

try:
    import sklearn
    print("✓ Scikit-learn version:", sklearn.__version__)
except ImportError as e:
    print("✗ Scikit-learn failed:", e)

try:
    import streamlit
    print("✓ Streamlit version:", streamlit.__version__)
except ImportError as e:
    print("✗ Streamlit failed:", e)

# ── Dataset Check ──────────────────────────────
import os
print("\n--- Dataset Check ---")
if os.path.exists('data/raw/chest_xray'):
    print("✓ Dataset folder found!")
    train_normal    = len(os.listdir('data/raw/chest_xray/train/NORMAL'))
    train_pneumonia = len(os.listdir('data/raw/chest_xray/train/PNEUMONIA'))
    test_normal     = len(os.listdir('data/raw/chest_xray/test/NORMAL'))
    test_pneumonia  = len(os.listdir('data/raw/chest_xray/test/PNEUMONIA'))
    print(f"  Train → Normal: {train_normal} | Pneumonia: {train_pneumonia}")
    print(f"  Test  → Normal: {test_normal}  | Pneumonia: {test_pneumonia}")
    print(f"  Total Training images: {train_normal + train_pneumonia}")
else:
    print("✗ Dataset not found yet - download it from Kaggle!")

print("\n✓ Setup Complete! Ready to build!")
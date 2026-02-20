import os

# CPU only - no GPU needed
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# ── Paths ──────────────────────────────────────
DATA_DIR        = 'data/raw/chest_xray'
TRAIN_DIR       = f'{DATA_DIR}/train'
TEST_DIR        = f'{DATA_DIR}/test'
VAL_DIR         = f'{DATA_DIR}/val'
MODEL_SAVE_PATH = 'models/chest_xray_model.pth'
PROCESSED_DIR   = 'data/processed'

# ── Model Settings (optimized for 8GB RAM) ─────
BATCH_SIZE    = 16
IMAGE_SIZE    = (224, 224)
EPOCHS        = 10
LEARNING_RATE = 0.0001
NUM_WORKERS   = 0        # Keep 0 for Windows

# ── Classes ────────────────────────────────────
CLASSES     = ['NORMAL', 'PNEUMONIA']
NUM_CLASSES = 2

print("✓ Config loaded!")
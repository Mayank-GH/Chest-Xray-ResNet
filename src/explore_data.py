import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

# ── Setup paths ────────────────────────────────
TRAIN_DIR = 'data/raw/chest_xray/train'
TEST_DIR  = 'data/raw/chest_xray/test'
VAL_DIR   = 'data/raw/chest_xray/val'

# ── Helper: filter only image files ───────────
def get_images(folder):
    valid = ('.jpg', '.jpeg', '.png', '.bmp')
    return [f for f in os.listdir(folder) if f.lower().endswith(valid)]

print("="*50)
print("   CHEST X-RAY DATA EXPLORATION")
print("="*50)

# ── 1. Count images ────────────────────────────
train_normal    = get_images(f'{TRAIN_DIR}/NORMAL')
train_pneumonia = get_images(f'{TRAIN_DIR}/PNEUMONIA')
test_normal     = get_images(f'{TEST_DIR}/NORMAL')
test_pneumonia  = get_images(f'{TEST_DIR}/PNEUMONIA')
val_normal      = get_images(f'{VAL_DIR}/NORMAL')
val_pneumonia   = get_images(f'{VAL_DIR}/PNEUMONIA')

print("\n📊 DATASET SUMMARY")
print("-"*40)
print(f"TRAIN  → Normal: {len(train_normal):>5} | Pneumonia: {len(train_pneumonia):>5} | Total: {len(train_normal)+len(train_pneumonia):>5}")
print(f"TEST   → Normal: {len(test_normal):>5} | Pneumonia: {len(test_pneumonia):>5} | Total: {len(test_normal)+len(test_pneumonia):>5}")
print(f"VAL    → Normal: {len(val_normal):>5} | Pneumonia: {len(val_pneumonia):>5} | Total: {len(val_normal)+len(val_pneumonia):>5}")
total = len(train_normal)+len(train_pneumonia)+len(test_normal)+len(test_pneumonia)+len(val_normal)+len(val_pneumonia)
print(f"TOTAL  → {total} images")
print("-"*40)

# ── 2. Check image sizes ───────────────────────
print("\n📐 SAMPLE IMAGE SIZES")
print("-"*40)
for label, folder, files in [
    ("Normal",    f'{TRAIN_DIR}/NORMAL',    train_normal),
    ("Pneumonia", f'{TRAIN_DIR}/PNEUMONIA', train_pneumonia)
]:
    sample = Image.open(os.path.join(folder, files[0]))
    print(f"{label:>10} sample → Size: {sample.size} | Mode: {sample.mode}")

# ── 3. Plot 1: Dataset Distribution ───────────
print("\n📈 Generating plots...")

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle('Chest X-Ray Dataset Analysis', fontsize=16, fontweight='bold')

# Bar chart
categories       = ['Train', 'Test', 'Val']
normal_counts    = [len(train_normal), len(test_normal), len(val_normal)]
pneumonia_counts = [len(train_pneumonia), len(test_pneumonia), len(val_pneumonia)]

x     = np.arange(len(categories))
width = 0.35
axes[0].bar(x - width/2, normal_counts,    width, label='Normal',    color='#2ecc71', alpha=0.8)
axes[0].bar(x + width/2, pneumonia_counts, width, label='Pneumonia', color='#e74c3c', alpha=0.8)
axes[0].set_title('Image Count per Split')
axes[0].set_xticks(x)
axes[0].set_xticklabels(categories)
axes[0].set_ylabel('Number of Images')
axes[0].legend()
axes[0].grid(axis='y', alpha=0.3)
for i, (n, p) in enumerate(zip(normal_counts, pneumonia_counts)):
    axes[0].text(i - width/2, n + 20, str(n), ha='center', fontsize=9)
    axes[0].text(i + width/2, p + 20, str(p), ha='center', fontsize=9)

# Training pie chart
axes[1].pie(
    [len(train_normal), len(train_pneumonia)],
    labels=['Normal', 'Pneumonia'],
    colors=['#2ecc71', '#e74c3c'],
    autopct='%1.1f%%',
    startangle=90,
    shadow=True
)
axes[1].set_title('Training Set Distribution')

# Overall pie chart
axes[2].pie(
    [sum(normal_counts), sum(pneumonia_counts)],
    labels=['Normal', 'Pneumonia'],
    colors=['#3498db', '#e67e22'],
    autopct='%1.1f%%',
    startangle=90,
    shadow=True
)
axes[2].set_title('Overall Dataset Distribution')

plt.tight_layout()
plt.savefig('data/processed/dataset_distribution.png', dpi=150, bbox_inches='tight')
plt.show()
print("✓ Plot 1 saved!")

# ── 4. Plot 2: Sample X-ray Images ────────────
fig2, axes2 = plt.subplots(2, 5, figsize=(15, 7))
fig2.suptitle('Sample Chest X-Ray Images', fontsize=16, fontweight='bold')

for i in range(5):
    img = Image.open(os.path.join(f'{TRAIN_DIR}/NORMAL', train_normal[i])).convert('L')
    axes2[0, i].imshow(img, cmap='gray')
    axes2[0, i].set_title(f'Normal {i+1}', color='green', fontweight='bold')
    axes2[0, i].axis('off')

for i in range(5):
    img = Image.open(os.path.join(f'{TRAIN_DIR}/PNEUMONIA', train_pneumonia[i])).convert('L')
    axes2[1, i].imshow(img, cmap='gray')
    axes2[1, i].set_title(f'Pneumonia {i+1}', color='red', fontweight='bold')
    axes2[1, i].axis('off')

plt.tight_layout()
plt.savefig('data/processed/sample_images.png', dpi=150, bbox_inches='tight')
plt.show()
print("✓ Plot 2 saved!")

# ── 5. Plot 3: Image Size Distribution ────────
print("\n📐 Analyzing image dimensions...")
widths  = []
heights = []

all_samples = [(f'{TRAIN_DIR}/NORMAL', f) for f in train_normal[:50]] + \
              [(f'{TRAIN_DIR}/PNEUMONIA', f) for f in train_pneumonia[:50]]

for folder, fname in all_samples:
    try:
        img = Image.open(os.path.join(folder, fname))
        widths.append(img.size[0])
        heights.append(img.size[1])
    except:
        pass

fig3, axes3 = plt.subplots(1, 2, figsize=(12, 4))
fig3.suptitle('Image Dimension Analysis', fontsize=14, fontweight='bold')

axes3[0].hist(widths,  bins=20, color='#3498db', alpha=0.7, edgecolor='black')
axes3[0].set_title('Image Width Distribution')
axes3[0].set_xlabel('Width (pixels)')
axes3[0].set_ylabel('Count')
axes3[0].axvline(np.mean(widths),  color='red',  linestyle='--', label=f'Mean: {np.mean(widths):.0f}px')
axes3[0].legend()
axes3[0].grid(alpha=0.3)

axes3[1].hist(heights, bins=20, color='#e74c3c', alpha=0.7, edgecolor='black')
axes3[1].set_title('Image Height Distribution')
axes3[1].set_xlabel('Height (pixels)')
axes3[1].set_ylabel('Count')
axes3[1].axvline(np.mean(heights), color='blue', linestyle='--', label=f'Mean: {np.mean(heights):.0f}px')
axes3[1].legend()
axes3[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('data/processed/image_dimensions.png', dpi=150, bbox_inches='tight')
plt.show()
print("✓ Plot 3 saved!")

print("\n" + "="*50)
print("✓ DATA EXPLORATION COMPLETE!")
print("✓ Check data/processed/ for saved plots")
print("="*50)
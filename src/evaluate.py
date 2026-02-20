import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report,
    roc_curve, auc, accuracy_score
)

sys.path.append('src')
from model import build_model
from dataset import get_dataloaders

# ── Config ─────────────────────────────────────
DATA_DIR   = 'data/raw/chest_xray'
MODEL_PATH = 'models/chest_xray_model.pth'
DEVICE     = torch.device('cpu')
CLASSES    = ['NORMAL', 'PNEUMONIA']

print("="*60)
print("      CHEST X-RAY MODEL EVALUATION")
print("="*60)

# ── Load Model ─────────────────────────────────
print("\n🔧 Loading trained model...")
model = build_model(num_classes=2).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()
print("✓ Model loaded successfully!")

# ── Load Test Data ─────────────────────────────
_, _, test_loader = get_dataloaders(DATA_DIR, batch_size=16)

# ── Make Predictions ───────────────────────────
print("\n🔍 Evaluating on test set...")
all_preds   = []
all_labels  = []
all_probs   = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(DEVICE)
        outputs = model(images)
        probs = torch.softmax(outputs, dim=1)
        _, predicted = outputs.max(1)
        
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.numpy())
        all_probs.extend(probs.cpu().numpy())

all_preds  = np.array(all_preds)
all_labels = np.array(all_labels)
all_probs  = np.array(all_probs)

# ── Calculate Metrics ──────────────────────────
accuracy = accuracy_score(all_labels, all_preds)
conf_matrix = confusion_matrix(all_labels, all_preds)

print("\n" + "="*60)
print("📊 TEST SET RESULTS")
print("="*60)
print(f"\n🎯 Overall Accuracy: {accuracy*100:.2f}%\n")

print("Detailed Classification Report:")
print("-"*60)
print(classification_report(
    all_labels, all_preds,
    target_names=CLASSES,
    digits=3
))

# ── Confusion Matrix ───────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('Model Evaluation Results', fontsize=16, fontweight='bold')

# Plot 1: Confusion Matrix
sns.heatmap(
    conf_matrix, annot=True, fmt='d', cmap='Blues',
    xticklabels=CLASSES, yticklabels=CLASSES,
    ax=axes[0], cbar=True, square=True,
    annot_kws={'size': 16, 'weight': 'bold'}
)
axes[0].set_title('Confusion Matrix', fontsize=14, fontweight='bold')
axes[0].set_ylabel('True Label', fontsize=12)
axes[0].set_xlabel('Predicted Label', fontsize=12)

# Add accuracy text
tn, fp, fn, tp = conf_matrix.ravel()
axes[0].text(
    0.5, -0.15,
    f'True Negatives: {tn} | False Positives: {fp}\n'
    f'False Negatives: {fn} | True Positives: {tp}',
    ha='center', transform=axes[0].transAxes,
    fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
)

# Plot 2: Normalized Confusion Matrix
conf_matrix_norm = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]
sns.heatmap(
    conf_matrix_norm, annot=True, fmt='.2%', cmap='Greens',
    xticklabels=CLASSES, yticklabels=CLASSES,
    ax=axes[1], cbar=True, square=True,
    annot_kws={'size': 16, 'weight': 'bold'}
)
axes[1].set_title('Normalized Confusion Matrix', fontsize=14, fontweight='bold')
axes[1].set_ylabel('True Label', fontsize=12)
axes[1].set_xlabel('Predicted Label', fontsize=12)

# Plot 3: ROC Curve
fpr, tpr, _ = roc_curve(all_labels, all_probs[:, 1])
roc_auc = auc(fpr, tpr)

axes[2].plot(fpr, tpr, color='darkorange', lw=3,
             label=f'ROC curve (AUC = {roc_auc:.3f})')
axes[2].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--',
             label='Random Classifier')
axes[2].set_xlim([0.0, 1.0])
axes[2].set_ylim([0.0, 1.05])
axes[2].set_xlabel('False Positive Rate', fontsize=12)
axes[2].set_ylabel('True Positive Rate', fontsize=12)
axes[2].set_title('ROC Curve', fontsize=14, fontweight='bold')
axes[2].legend(loc="lower right", fontsize=10)
axes[2].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('data/processed/evaluation_results.png', dpi=150, bbox_inches='tight')
plt.show()

# ── Per-Class Accuracy ─────────────────────────
print("\n📈 Per-Class Performance:")
print("-"*60)
for i, class_name in enumerate(CLASSES):
    class_mask = all_labels == i
    class_acc = accuracy_score(all_labels[class_mask], all_preds[class_mask])
    class_total = class_mask.sum()
    class_correct = (all_preds[class_mask] == all_labels[class_mask]).sum()
    print(f"  {class_name:>10}: {class_acc*100:5.2f}% "
          f"({class_correct}/{class_total} correct)")

print("\n" + "="*60)
print("✓ EVALUATION COMPLETE!")
print(f"  Plot saved: data/processed/evaluation_results.png")
print("="*60)
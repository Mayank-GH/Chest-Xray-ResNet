import os
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import StepLR

# Add src to path
sys.path.append('src')
from model   import build_model
from dataset import get_dataloaders

# ── Config ─────────────────────────────────────
DATA_DIR   = 'data/raw/chest_xray'
SAVE_PATH  = 'models/chest_xray_model.pth'
BATCH_SIZE = 16
EPOCHS     = 10
LR         = 0.0001
DEVICE     = torch.device('cpu')

print("="*55)
print("      CHEST X-RAY MODEL TRAINING")
print("="*55)
print(f"  Device     : {DEVICE}")
print(f"  Batch size : {BATCH_SIZE}")
print(f"  Epochs     : {EPOCHS}")
print(f"  LR         : {LR}")
print("="*55)

# ── Load Data ──────────────────────────────────
train_loader, val_loader, test_loader = get_dataloaders(DATA_DIR, BATCH_SIZE)

# ── Build Model ────────────────────────────────
print("\n🔧 Building model...")
model     = build_model(num_classes=2).to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=LR
)
scheduler = StepLR(optimizer, step_size=3, gamma=0.5)

# ── Training History ───────────────────────────
history = {
    'train_loss': [], 'train_acc': [],
    'val_loss':   [], 'val_acc':   []
}

best_val_acc = 0.0

# ── Training Loop ──────────────────────────────
print("\n🚀 Starting training...\n")

for epoch in range(EPOCHS):
    start_time = time.time()

    # ── TRAIN ──
    model.train()
    train_loss = 0.0
    train_correct = 0
    train_total   = 0

    for batch_idx, (images, labels) in enumerate(train_loader):
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss    = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss    += loss.item()
        _, predicted   = outputs.max(1)
        train_total   += labels.size(0)
        train_correct += predicted.eq(labels).sum().item()

        # Print progress every 50 batches
        if (batch_idx + 1) % 50 == 0:
            print(f"  Epoch {epoch+1}/{EPOCHS} | "
                  f"Batch {batch_idx+1}/{len(train_loader)} | "
                  f"Loss: {train_loss/(batch_idx+1):.4f} | "
                  f"Acc: {100.*train_correct/train_total:.1f}%")

    train_loss /= len(train_loader)
    train_acc   = 100. * train_correct / train_total

    # ── VALIDATE ──
    model.eval()
    val_loss    = 0.0
    val_correct = 0
    val_total   = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs        = model(images)
            loss           = criterion(outputs, labels)

            val_loss    += loss.item()
            _, predicted = outputs.max(1)
            val_total   += labels.size(0)
            val_correct += predicted.eq(labels).sum().item()

    val_loss /= len(val_loader)
    val_acc   = 100. * val_correct / val_total

    # Save history
    history['train_loss'].append(train_loss)
    history['train_acc'].append(train_acc)
    history['val_loss'].append(val_loss)
    history['val_acc'].append(val_acc)

    # Save best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), SAVE_PATH)
        saved = "✓ SAVED"
    else:
        saved = ""

    epoch_time = time.time() - start_time
    print(f"\n{'─'*55}")
    print(f"  Epoch {epoch+1}/{EPOCHS} Complete ({epoch_time:.0f}s)")
    print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.1f}%")
    print(f"  Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.1f}% {saved}")
    print(f"{'─'*55}\n")

    scheduler.step()

# ── Plot Training Results ──────────────────────
print("📊 Saving training plots...")

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
fig.suptitle('Training Results', fontsize=14, fontweight='bold')

epochs_range = range(1, EPOCHS + 1)

# Loss plot
axes[0].plot(epochs_range, history['train_loss'], 'b-o', label='Train Loss')
axes[0].plot(epochs_range, history['val_loss'],   'r-o', label='Val Loss')
axes[0].set_title('Loss over Epochs')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].legend()
axes[0].grid(alpha=0.3)

# Accuracy plot
axes[1].plot(epochs_range, history['train_acc'], 'b-o', label='Train Accuracy')
axes[1].plot(epochs_range, history['val_acc'],   'r-o', label='Val Accuracy')
axes[1].set_title('Accuracy over Epochs')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Accuracy (%)')
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('data/processed/training_results.png', dpi=150, bbox_inches='tight')
plt.show()

print("\n" + "="*55)
print(f"✓ TRAINING COMPLETE!")
print(f"  Best Val Accuracy : {best_val_acc:.1f}%")
print(f"  Model saved to    : {SAVE_PATH}")
print(f"  Plot saved to     : data/processed/training_results.png")
print("="*55)
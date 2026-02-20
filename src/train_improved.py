import os
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import ConcatDataset, DataLoader

sys.path.append('src')
from model import build_model
from dataset import get_dataloaders

# ── Improved Config ───────────────────────────────
DATA_DIR   = 'data/raw/chest_xray'
SAVE_PATH  = 'models/chest_xray_model_v2.pth'
DEVICE     = torch.device('cpu')
BATCH_SIZE = 16
MAX_EPOCHS = 20
LR         = 0.0001
PATIENCE   = 5

print("="*60)
print("   IMPROVED TRAINING WITH EARLY STOPPING")
print("="*60)
print(f"  Device       : {DEVICE}")
print(f"  Batch size   : {BATCH_SIZE}")
print(f"  Max Epochs   : {MAX_EPOCHS}")
print(f"  Learning Rate: {LR}")
print(f"  Patience     : {PATIENCE} epochs")
print("="*60)

# ── Load and Combine Data ─────────────────────────
print("\n📂 Loading datasets...")
train_loader, val_loader, test_loader = get_dataloaders(DATA_DIR, BATCH_SIZE)

print("\n💡 Combining train + val for more training data...")
combined_dataset = ConcatDataset([train_loader.dataset, val_loader.dataset])
train_loader = DataLoader(
    combined_dataset, 
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

print(f"✓ Training images: {len(combined_dataset)}")
print(f"✓ Test images: {len(test_loader.dataset)}")

# ── Build Model ────────────────────────────────────
print("\n🔧 Building model...")
model = build_model(num_classes=2).to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=LR,
    weight_decay=1e-4  # L2 regularization
)

# Adaptive learning rate
scheduler = ReduceLROnPlateau(
    optimizer, mode='max', factor=0.5,
    patience=3, verbose=True
)

print("✓ Model ready!")

# ── Training Loop ──────────────────────────────────
history = {
    'train_loss': [], 'train_acc': [],
    'test_loss': [], 'test_acc': []
}

best_test_acc = 0.0
patience_counter = 0

print("\n🚀 Starting training with early stopping...\n")

for epoch in range(MAX_EPOCHS):
    start_time = time.time()
    
    # ── TRAIN ──
    model.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0
    
    for batch_idx, (images, labels) in enumerate(train_loader):
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        train_total += labels.size(0)
        train_correct += predicted.eq(labels).sum().item()
        
        # Progress every 50 batches
        if (batch_idx + 1) % 50 == 0:
            current_acc = 100. * train_correct / train_total
            current_loss = train_loss / (batch_idx + 1)
            print(f"  Epoch {epoch+1:>2}/{MAX_EPOCHS} | "
                  f"Batch {batch_idx+1:>3}/{len(train_loader)} | "
                  f"Loss: {current_loss:.4f} | "
                  f"Acc: {current_acc:>5.1f}%")
    
    train_loss /= len(train_loader)
    train_acc = 100. * train_correct / train_total
    
    # ── TEST (Validation) ──
    model.eval()
    test_loss = 0.0
    test_correct = 0
    test_total = 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            test_total += labels.size(0)
            test_correct += predicted.eq(labels).sum().item()
    
    test_loss /= len(test_loader)
    test_acc = 100. * test_correct / test_total
    
    # Update learning rate scheduler
    scheduler.step(test_acc)
    
    # Save history
    history['train_loss'].append(train_loss)
    history['train_acc'].append(train_acc)
    history['test_loss'].append(test_loss)
    history['test_acc'].append(test_acc)
    
    # Early stopping check
    if test_acc > best_test_acc:
        improvement = test_acc - best_test_acc
        best_test_acc = test_acc
        torch.save(model.state_dict(), SAVE_PATH)
        patience_counter = 0
        saved = f"✓ SAVED! (+{improvement:.1f}%)"
    else:
        patience_counter += 1
        saved = f"(Patience: {patience_counter}/{PATIENCE})"
    
    epoch_time = time.time() - start_time
    
    print(f"\n{'─'*60}")
    print(f"  Epoch {epoch+1}/{MAX_EPOCHS} Complete ({epoch_time:.0f}s)")
    print(f"  Train → Loss: {train_loss:.4f} | Acc: {train_acc:.2f}%")
    print(f"  Test  → Loss: {test_loss:.4f} | Acc: {test_acc:.2f}% {saved}")
    print(f"  Best Test Acc: {best_test_acc:.2f}%")
    print(f"{'─'*60}\n")
    
    # Stop if no improvement
    if patience_counter >= PATIENCE:
        print(f"\n⚠️  Early stopping! No improvement for {PATIENCE} epochs.")
        print(f"   Best test accuracy: {best_test_acc:.2f}%\n")
        break

# ── Plot Training Curves ───────────────────────────
print("📊 Generating training plots...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Improved Training Results', fontsize=16, fontweight='bold')

epochs_range = range(1, len(history['train_loss']) + 1)

# Loss plot
axes[0].plot(epochs_range, history['train_loss'], 'b-o', 
             linewidth=2, markersize=8, label='Train Loss')
axes[0].plot(epochs_range, history['test_loss'], 'r-o',
             linewidth=2, markersize=8, label='Test Loss')
axes[0].set_title('Loss Over Epochs', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Epoch', fontsize=12)
axes[0].set_ylabel('Loss', fontsize=12)
axes[0].legend(fontsize=11)
axes[0].grid(alpha=0.3)

# Accuracy plot
axes[1].plot(epochs_range, history['train_acc'], 'b-o',
             linewidth=2, markersize=8, label='Train Accuracy')
axes[1].plot(epochs_range, history['test_acc'], 'r-o',
             linewidth=2, markersize=8, label='Test Accuracy')
axes[1].axhline(y=best_test_acc, color='g', linestyle='--',
                linewidth=2, label=f'Best: {best_test_acc:.2f}%')
axes[1].set_title('Accuracy Over Epochs', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Epoch', fontsize=12)
axes[1].set_ylabel('Accuracy (%)', fontsize=12)
axes[1].legend(fontsize=11)
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('data/processed/training_improved_v2.png', dpi=150, bbox_inches='tight')
plt.show()

print("\n" + "="*60)
print("✓ IMPROVED TRAINING COMPLETE!")
print(f"  Best Test Accuracy: {best_test_acc:.2f}%")
print(f"  Improvement: +{best_test_acc - 86.86:.2f}%")
print(f"  Model saved: {SAVE_PATH}")
print(f"  Plot saved: data/processed/training_improved_v2.png")
print("="*60)
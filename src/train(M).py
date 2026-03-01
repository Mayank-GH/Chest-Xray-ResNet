import argparse
import copy
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from tqdm import tqdm

# Config
IMG_SIZE    = 224
BATCH_SIZE  = 32
EPOCHS      = 20
LR          = 1e-4
NUM_CLASSES = 2           # NORMAL, PNEUMONIA
SAVE_PATH   = "pneumonia_resnet18.pth"


# preprocessing
train_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),              #resizes image to 224*224
    transforms.RandomHorizontalFlip(),                    #flip
    transforms.RandomRotation(10),                        #rotation
    transforms.ColorJitter(brightness=0.2, contrast=0.2), #randomly altering brightness,contrast etc
    transforms.ToTensor(),                                #converts image to a multidimensional array(tensor) so gpu can better process it 
    transforms.Normalize([0.485, 0.456, 0.406],           #standardization or z-score normalization. was used in the original imagenet
                         [0.229, 0.224, 0.225]),          #output = (input - mean)/std -> ([R,G,B])
])

test_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])


# ──────────────────────────────────────────────
# Weighted loss helper
# ──────────────────────────────────────────────
def compute_class_weights(dataset, device):
    """Compute inverse-frequency weights for CrossEntropyLoss."""
    counts = torch.zeros(NUM_CLASSES)
    for _, label in dataset:
        counts[label] += 1
    weights = 1.0 / counts
    weights = weights / weights.sum() * NUM_CLASSES   # normalise
    print(f"  Class counts  : {counts.tolist()}")
    print(f"  Class weights : {weights.tolist()}")
    return weights.to(device)


# ──────────────────────────────────────────────
# Model
# ──────────────────────────────────────────────
def build_model(device):
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    # Replace the final fully-connected layer
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, NUM_CLASSES)
    return model.to(device)


#  ──────────────────────────────────────────────
# Training loop
# ──────────────────────────────────────────────
def run_epoch(model, loader, criterion, optimizer, device, phase):
    """Run a single epoch for train or test phase. Returns (loss, accuracy)."""
    is_train = phase == "train"
    model.train() if is_train else model.eval()

    running_loss    = 0.0
    running_correct = 0
    total           = len(loader.dataset)

    # tqdm wraps the dataloader — shows a live progress bar per batch
    pbar = tqdm(
        loader,
        desc=f"  {phase:5s}",
        unit="batch",
        leave=False,                  # clears bar after epoch finishes
        dynamic_ncols=True,           # adapts to terminal width
    )

    for inputs, labels in pbar:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        with torch.set_grad_enabled(is_train):
            outputs = model(inputs)
            loss    = criterion(outputs, labels)
            preds   = outputs.argmax(dim=1)

            if is_train:
                loss.backward()
                optimizer.step()

        batch_loss    = loss.item() * inputs.size(0)
        batch_correct = (preds == labels).sum().item()

        running_loss    += batch_loss
        running_correct += batch_correct

        # Update tqdm postfix with live running stats
        pbar.set_postfix({
            "loss": f"{batch_loss / inputs.size(0):.4f}",
            "acc":  f"{batch_correct / inputs.size(0):.4f}",
        })

    epoch_loss = running_loss    / total
    epoch_acc  = running_correct / total
    return epoch_loss, epoch_acc


def train(model, loaders, criterion, optimizer, scheduler, device, epochs):
    best_acc     = 0.0
    best_weights = copy.deepcopy(model.state_dict())

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        print(f"\nEpoch {epoch}/{epochs}  {'─'*45}")

        train_loss, train_acc = run_epoch(
            model, loaders["train"], criterion, optimizer, device, "train"
        )
        test_loss, test_acc = run_epoch(
            model, loaders["test"], criterion, optimizer, device, "test"
        )

        elapsed = time.time() - t0
        scheduler.step(test_loss)

        # Summary line after both phases
        print(
            f"  {'train':5s} | loss: {train_loss:.4f} | acc: {train_acc:.4f}\n"
            f"  {'test':5s}  | loss: {test_loss:.4f}  | acc: {test_acc:.4f}  "
            f"| {elapsed:.1f}s"
        )

        if test_acc > best_acc:
            best_acc     = test_acc
            best_weights = copy.deepcopy(model.state_dict())
            torch.save(best_weights, SAVE_PATH)
            print(f"  ✔ Best model saved  (acc={best_acc:.4f})")

    print(f"\nTraining complete. Best test accuracy: {best_acc:.4f}")
    print(f"Model saved to: {SAVE_PATH}")
    model.load_state_dict(best_weights)
    return model


# ──────────────────────────────────────────────
# Evaluation
# ──────────────────────────────────────────────
def evaluate(model, loader, class_names, device):
    from sklearn.metrics import classification_report, confusion_matrix

    model.eval()
    all_preds, all_labels = [], []

    pbar = tqdm(loader, desc="  Evaluating", unit="batch", dynamic_ncols=True)
    with torch.no_grad():
        for inputs, labels in pbar:
            inputs = inputs.to(device)
            preds  = model(inputs).argmax(dim=1).cpu()
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.tolist())

    print("\n── Final Test Evaluation ──────────────────────────")
    print("Confusion Matrix:")
    print(confusion_matrix(all_labels, all_preds))
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names))


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
def main(data_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_ds = datasets.ImageFolder(os.path.join(data_dir, "train"), train_transforms)
    test_ds  = datasets.ImageFolder(os.path.join(data_dir, "test"),  test_transforms)

    print(f"\nClasses : {train_ds.classes}")
    print(f"Train   : {len(train_ds)} images")
    print(f"Test    : {len(test_ds)} images")

    print("\nComputing class weights...")
    class_weights = compute_class_weights(train_ds, device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    loaders = {
        "train": DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                            num_workers=4, pin_memory=True),
        "test":  DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=4, pin_memory=True),
    }

    model     = build_model(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min",
                                                      factor=0.5, patience=3)

    model = train(model, loaders, criterion, optimizer, scheduler, device, EPOCHS)
    evaluate(model, loaders["test"], train_ds.classes, device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./chest_xray",
                        help="Path to chest_xray folder (contains train/ and test/)")
    args = parser.parse_args()
    main(args.data_dir)
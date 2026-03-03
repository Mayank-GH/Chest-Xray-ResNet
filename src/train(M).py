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
IMG_SIZE     = 224
BATCH_SIZE   = 32
EPOCHS       = 20
LR           = 1e-4
WEIGHT_DECAY = 1e-4
DROPOUT_P    = 0.5
EARLY_STOP_PATIENCE = 5
NUM_CLASSES = 2           # NORMAL, PNEUMONIA
SAVE_PATH   = "pneumonia_resnet18.pth"


# preprocessing
train_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),              #resizes image to 224*224
    transforms.RandomHorizontalFlip(),                    #flip
    transforms.RandomRotation(15),                        #rotation
    transforms.ColorJitter(brightness=0.2, contrast=0.2), #randomly alters brightness,contrast etc
    transforms.ToTensor(),                                #converts image to a multidimensional array(tensor) so gpu can better process it 
    transforms.Normalize([0.485, 0.456, 0.406],           #standardization or z-score normalization. was used in the original imagenet
                         [0.229, 0.224, 0.225]),          #output = (input - mean)/std -> ([R,G,B])
    transforms.RandomErasing(p=0.3, scale=(0.05, 0.20), ratio=(0.3, 3.3)), 
     transforms.RandomAffine(
        degrees=0,
        translate=(0.1, 0.1),                              # random shift up to 10%
        shear=5,                                           # slight shear
    ),
])

test_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])


# Weighted loss helper

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


# Model — with Dropout in classifier head

def build_model(device):
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

    in_features = model.fc.in_features  # 512 for ResNet-18

    # Replace FC with: Dropout → Linear
    # Dropout zeros 50% of the 512 features during each training forward pass,
    # preventing neurons from co-adapting and memorizing training patterns
    model.fc = nn.Sequential(
        nn.Dropout(p=DROPOUT_P),
        nn.Linear(in_features, NUM_CLASSES),
    )

    return model.to(device)

# Early stopping

class EarlyStopping:
    """
    Stops training if test loss doesn't improve for `patience` consecutive
    epochs. Automatically tracks and restores the best weights seen so far.
    """
    def __init__(self, patience=5):
        self.patience     = patience
        self.counter      = 0
        self.best_loss    = float("inf")
        self.best_weights = None

    def step(self, val_loss, model):
        """Call after each epoch. Returns True if training should stop."""
        if val_loss < self.best_loss:
            self.best_loss    = val_loss
            self.best_weights = copy.deepcopy(model.state_dict())
            self.counter      = 0
            return False   # improved — continue training
        else:
            self.counter += 1
            print(f"  ⚠ No improvement. Early stop counter: {self.counter}/{self.patience}")
            return self.counter >= self.patience  # True = stop

    def restore(self, model):
        """Load the best weights back into the model."""
        if self.best_weights:
            model.load_state_dict(self.best_weights)


# Epoch runner

def run_epoch(model, loader, criterion, optimizer, device, phase):
    is_train = phase == "train"
    model.train() if is_train else model.eval()

    running_loss    = 0.0
    running_correct = 0
    total           = len(loader.dataset)

    pbar = tqdm(
        loader,
        desc=f"  {phase:5s}",
        unit="batch",
        leave=False,
        dynamic_ncols=True,
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

        pbar.set_postfix({
            "loss": f"{batch_loss / inputs.size(0):.4f}",
            "acc":  f"{batch_correct / inputs.size(0):.4f}",
        })

    return running_loss / total, running_correct / total


# Training loop

def train(model, loaders, criterion, optimizer, scheduler, device, epochs):
    early_stopping = EarlyStopping(patience=EARLY_STOP_PATIENCE)
    best_acc       = 0.0

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

        print(
            f"  {'train':5s} | loss: {train_loss:.4f} | acc: {train_acc:.4f}\n"
            f"  {'test':5s}  | loss: {test_loss:.4f}  | acc: {test_acc:.4f}  "
            f"| {elapsed:.1f}s"
        )

        # Track best accuracy separately for display
        if test_acc > best_acc:
            best_acc = test_acc
            print(f"  ✔ New best test accuracy: {best_acc:.4f}")

        # Early stopping — also tracks best weights internally
        if early_stopping.step(test_loss, model):
            print(f"\n⛔ Early stopping triggered at epoch {epoch}.")
            break

    # Restore best weights and save
    early_stopping.restore(model)
    torch.save(model.state_dict(), SAVE_PATH)
    print(f"\nTraining complete. Best test accuracy: {best_acc:.4f}")
    print(f"Model saved to: {SAVE_PATH}")
    return model


# Evaluation

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


# Main

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

    model = build_model(device)

    # weight_decay adds L2 penalty to all weights during each update step
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
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
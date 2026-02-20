import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

# ── Filter valid images only ───────────────────
def get_images(folder):
    valid = ('.jpg', '.jpeg', '.png', '.bmp')
    return [f for f in os.listdir(folder) if f.lower().endswith(valid)]


# ── Custom Dataset Class ───────────────────────
class ChestXRayDataset(Dataset):

    def __init__(self, data_dir, transform=None):
        self.data_dir  = data_dir
        self.transform = transform
        self.images    = []
        self.labels    = []

        # Load NORMAL images → label 0
        normal_dir = os.path.join(data_dir, 'NORMAL')
        for fname in get_images(normal_dir):
            self.images.append(os.path.join(normal_dir, fname))
            self.labels.append(0)

        # Load PNEUMONIA images → label 1
        pneumonia_dir = os.path.join(data_dir, 'PNEUMONIA')
        for fname in get_images(pneumonia_dir):
            self.images.append(os.path.join(pneumonia_dir, fname))
            self.labels.append(1)

        print(f"  Loaded {len(self.images)} images "
              f"(Normal: {self.labels.count(0)}, "
              f"Pneumonia: {self.labels.count(1)})")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img   = Image.open(self.images[idx]).convert('RGB')
        label = self.labels[idx]
        if self.transform:
            img = self.transform(img)
        return img, label


# ── Transforms ────────────────────────────────
def get_transforms():

    # Training: add augmentation to improve model
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    # Validation/Test: no augmentation
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    return train_transform, val_transform


# ── Create DataLoaders ─────────────────────────
def get_dataloaders(data_dir, batch_size=16):

    train_transform, val_transform = get_transforms()

    print("\n📂 Loading datasets...")
    train_dataset = ChestXRayDataset(f'{data_dir}/train', train_transform)
    val_dataset   = ChestXRayDataset(f'{data_dir}/val',   val_transform)
    test_dataset  = ChestXRayDataset(f'{data_dir}/test',  val_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0      # Keep 0 for Windows
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    return train_loader, val_loader, test_loader


# ── Test this file ─────────────────────────────
if __name__ == '__main__':
    print("="*50)
    print("   TESTING DATA LOADER")
    print("="*50)

    DATA_DIR = 'data/raw/chest_xray'
    train_loader, val_loader, test_loader = get_dataloaders(DATA_DIR, batch_size=16)

    print(f"\n✓ DataLoaders created!")
    print(f"  Train batches : {len(train_loader)}")
    print(f"  Val batches   : {len(val_loader)}")
    print(f"  Test batches  : {len(test_loader)}")

    # Test one batch
    images, labels = next(iter(train_loader))
    print(f"\n✓ Sample batch loaded!")
    print(f"  Batch shape : {images.shape}")
    print(f"  Labels      : {labels.tolist()}")
    print(f"  Label 0 = NORMAL | Label 1 = PNEUMONIA")
    print("\n✓ Dataset is ready for training!")
import torch
import torch.nn as nn
import torchvision.models as models

def build_model(num_classes=2):
    """
    Uses ResNet18 pretrained model
    Fine-tuned for chest x-ray classification
    Optimized for CPU and 8GB RAM
    """
    # Load pretrained ResNet18
    model = models.resnet18(weights='IMAGENET1K_V1')

    # Freeze early layers (saves memory and training time)
    for name, param in model.named_parameters():
        if 'layer4' not in name and 'fc' not in name:
            param.requires_grad = False

    # Replace final layer for our 2 classes
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(256, num_classes)
    )

    return model


def count_parameters(model):
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


if __name__ == '__main__':
    print("="*50)
    print("   BUILDING CHEST X-RAY MODEL")
    print("="*50)

    model = build_model(num_classes=2)

    total, trainable = count_parameters(model)
    print(f"\n✓ Model built successfully!")
    print(f"  Architecture  : ResNet18 (pretrained)")
    print(f"  Total params  : {total:,}")
    print(f"  Trainable     : {trainable:,}")
    print(f"  Frozen params : {total - trainable:,}")
    print(f"  Classes       : NORMAL, PNEUMONIA")

    # Test with a dummy image
    dummy = torch.randn(1, 3, 224, 224)
    output = model(dummy)
    print(f"\n✓ Test forward pass successful!")
    print(f"  Input shape  : {dummy.shape}")
    print(f"  Output shape : {output.shape}")
    print(f"  Output values: {output.detach().numpy()}")
    print("\n✓ Model is ready for training!")
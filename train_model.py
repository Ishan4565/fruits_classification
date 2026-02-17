import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
import os

print("=" * 60)
print("FRUIT RECOGNITION - IMPROVED TRAINING")
print("=" * 60)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nUsing device: {device}")

data_dir = "archive/train/train"

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

full_dataset = datasets.ImageFolder(data_dir, transform=train_transform)
num_classes = len(full_dataset.classes)

print(f"\nDataset loaded:")
print(f"  Total images: {len(full_dataset)}")
print(f"  Number of classes: {num_classes}")
print(f"  Sample classes: {full_dataset.classes[:10]}")

val_size = int(0.2 * len(full_dataset))
train_size = len(full_dataset) - val_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

val_dataset.dataset.transform = val_transform

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)

print(f"\nData split:")
print(f"  Training: {train_size}")
print(f"  Validation: {val_size}")

print("\nBuilding model...")

model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

for param in model.parameters():
    param.requires_grad = True

model.fc = nn.Linear(model.fc.in_features, num_classes)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)

print("✓ Model created with all layers unfrozen")
print(f"✓ Output classes: {num_classes}")

num_epochs = 15

print(f"\nStarting training for {num_epochs} epochs...")
print("=" * 60)

best_val_acc = 0.0
prev_lr = optimizer.param_groups[0]['lr']

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    train_correct = 0
    train_total = 0

    for batch_idx, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        
        _, predicted = torch.max(outputs.data, 1)
        train_total += labels.size(0)
        train_correct += (predicted == labels).sum().item()

        if batch_idx % 20 == 0:
            print(f"  Epoch [{epoch+1}/{num_epochs}] Batch [{batch_idx}/{len(train_loader)}] Loss: {loss.item():.4f}")

    train_loss = running_loss / len(train_loader)
    train_acc = 100 * train_correct / train_total

    model.eval()
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()

    val_acc = 100 * val_correct / val_total

    print(f"\nEpoch [{epoch+1}/{num_epochs}] Summary:")
    print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
    print(f"  Val Acc: {val_acc:.2f}%")
    
    current_lr = optimizer.param_groups[0]['lr']
    if current_lr != prev_lr:
        print(f"  Learning rate reduced to: {current_lr}")
        prev_lr = current_lr
    
    print("=" * 60)

    scheduler.step(val_acc)

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save({
            "model_state_dict": model.state_dict(),
            "classes": full_dataset.classes,
            "num_classes": num_classes,
            "best_val_acc": best_val_acc
        }, "fruit_model_best.pth")
        print(f"  ✓ New best model saved! Accuracy: {best_val_acc:.2f}%\n")

torch.save({
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'classes': full_dataset.classes
}, f"checkpoint_epoch_{epoch}.pth")

print("\n" + "=" * 60)
print("TRAINING COMPLETE!")
print("=" * 60)
print(f"\nFinal Results:")
print(f"  Best Validation Accuracy: {best_val_acc:.2f}%")
print(f"  Final Validation Accuracy: {val_acc:.2f}%")
print(f"\nModels saved:")
print(f"  - fruit_model.pth (final model)")
print(f"  - fruit_model_best.pth (best model)")
print("\nNext: streamlit run app.py")
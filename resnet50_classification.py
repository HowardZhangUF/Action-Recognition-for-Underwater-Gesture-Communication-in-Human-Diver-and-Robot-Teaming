import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns
from torchvision import models
from torchvision.models import ResNet50_Weights
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
# ----------------------------
# 1) DATA LOADING (AUTO SPLIT)
# ----------------------------
DATA_PATH = "image_data/horizontal"  
# Directory structure:
# diver_action_image/horizontal_vertical_pose/
#   ├── action_1/
#   │    ├── img1.jpg
#   │    ├── img2.jpg
#   ├── action_2/
#   │    ├── ...
#   └── action_11/

batch_size = 32
img_size = 224

transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                         std=[0.229, 0.224, 0.225])
])

# Load the dataset once
full_dataset = datasets.ImageFolder(root=DATA_PATH, transform=transform)

# Stratified split (80/20)
targets = [label for _, label in full_dataset.samples]  # labels for stratification
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(sss.split(np.zeros(len(targets)), targets))

train_dataset = Subset(full_dataset, train_idx)
test_dataset = Subset(full_dataset, test_idx)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

class_names = full_dataset.classes
NUM_CLASSES = len(class_names)
print("Classes:", class_names)

# ----------------------------
# 2) MODEL: RESNET50
# ----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
for param in model.parameters():
    param.requires_grad = False  # freeze backbone

# Replace final FC layer
model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
model = model.to(device)

# ----------------------------
# 3) LOSS & OPTIMIZER
# ----------------------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=1e-3)

# ----------------------------
# 4) TRAINING LOOP
# ----------------------------
# ----------------------------
# 4) TRAINING LOOP
# ----------------------------
def train_model(model, train_loader, test_loader, criterion, optimizer, epochs=10, log_dir="runs"):
    writer = SummaryWriter(log_dir)   # ✅ Create TensorBoard writer

    train_losses, test_losses, train_accs, test_accs = [], [], [], []
    all_preds, all_labels = [], []   # ✅ move outside loop so it persists across epochs

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss, correct, total = 0.0, 0, 0

        # ✅ wrap training loop with tqdm
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", unit="batch"):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_loss = running_loss / total
        train_acc = correct / total

        # ---- Evaluate ----
        model.eval()
        running_loss, correct, total = 0.0, 0, 0
        all_preds, all_labels = [], []   # reset for test set each epoch
        with torch.no_grad():
            for images, labels in tqdm(test_loader, desc=f"Eval {epoch}/{epochs}", unit="batch"):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                running_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        test_loss = running_loss / total
        test_acc = correct / total

        train_losses.append(train_loss)
        test_losses.append(test_loss)
        train_accs.append(train_acc)
        test_accs.append(test_acc)

        # ✅ Log to TensorBoard
        writer.add_scalar("Loss/Train", train_loss, epoch)
        writer.add_scalar("Loss/Test", test_loss, epoch)
        writer.add_scalar("Accuracy/Train", train_acc, epoch)
        writer.add_scalar("Accuracy/Test", test_acc, epoch)

        print(f"\nEpoch {epoch}/{epochs} | Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | "
              f"Test Loss: {test_loss:.4f}, Acc: {test_acc:.4f}")

    writer.close()  # ✅ close writer
    return train_losses, test_losses, train_accs, test_accs, all_preds, all_labels

# ----------------------------
# 5) RUN TRAINING
# ----------------------------
if __name__ == "__main__":
    train_losses, test_losses, train_accs, test_accs, preds, labels = train_model(
        model, train_loader, test_loader, criterion, optimizer, epochs=10
    )
    # Save the trained model
    torch.save(model.state_dict(), 'resnet50_diver_action_model.pth')
    print("Model saved as 'resnet50_diver_action_model.pth'")

    # ----------------------------
    # 6) PLOTS & CONFUSION MATRIX
    # ----------------------------
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.plot(train_losses, label="Train Loss")
    plt.plot(test_losses, label="Test Loss")
    plt.legend(); plt.title("Loss Curve")

    plt.subplot(1,2,2)
    plt.plot(train_accs, label="Train Acc")
    plt.plot(test_accs, label="Test Acc")
    plt.legend(); plt.title("Accuracy Curve")
    plt.show()

    cm = confusion_matrix(labels, preds, labels=list(range(NUM_CLASSES)), normalize="true")
    plt.figure(figsize=(7,6))
    sns.heatmap(cm, annot=True, cmap="Blues", xticklabels=class_names, yticklabels=class_names, fmt=".2f")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()


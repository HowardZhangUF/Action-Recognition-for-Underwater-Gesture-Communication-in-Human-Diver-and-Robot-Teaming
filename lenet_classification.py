import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import confusion_matrix
import seaborn as sns
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

# ----------------------------
# 1) DATA LOADING (AUTO SPLIT)
# ----------------------------
DATA_PATH = "image_data/horizontal"      # root folder of your images
batch_size = 32
img_size   = 224
actions         = ['ASCEND','DESCEND','ME','STOP','RIGHT','BUDDY_UP',
                   'FOLLOW_ME','OKAY','LEFT','YOU','LEVEL']
transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

full_dataset = datasets.ImageFolder(root=DATA_PATH, transform=transform)

# Stratified 80 / 20 split
targets = [label for _, label in full_dataset.samples]
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
train_idx, test_idx = next(sss.split(np.zeros(len(targets)), targets))

train_dataset = Subset(full_dataset, train_idx)
test_dataset  = Subset(full_dataset,  test_idx)

train_loader = DataLoader(train_dataset, batch_size=batch_size,
                          shuffle=True,  num_workers=4)
test_loader  = DataLoader(test_dataset,  batch_size=batch_size,
                          shuffle=False, num_workers=4)

class_names = full_dataset.classes
NUM_CLASSES = len(class_names)
print("Classes:", class_names)


class CNN(nn.Module):
    """
   
    """
    def __init__(self, num_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),  # 224×224 → 224×224
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                 # 224×224 → 112×112

            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                 # 112×112 → 56×56

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool2d((1, 1))     # 56×56 → 1×1
        )
        self.classifier = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.features(x)          # (B, 64, 1, 1)
        x = torch.flatten(x, 1)       # (B, 64)
        return self.classifier(x)     # (B, num_classes)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model  = CNN(NUM_CLASSES).to(device)

# ----------------------------
# 3) LOSS & OPTIMIZER
# ----------------------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=5e-3)  

# ----------------------------
# 4) TRAINING LOOP
# ----------------------------
def train_model(model, train_loader, test_loader,
                criterion, optimizer, epochs=10, log_dir="runs"):
    writer = SummaryWriter(log_dir)

    train_losses, test_losses = [], []
    train_accs,  test_accs  = [], []
    all_preds,   all_labels = [], []

    for epoch in range(1, epochs + 1):
        # ---- Train ----
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for images, labels in tqdm(train_loader,
                                   desc=f"Epoch {epoch}/{epochs}",
                                   unit="batch"):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            outputs = model(images)
            loss    = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total   += labels.size(0)

        train_loss = running_loss / total
        train_acc  = correct / total

        # ---- Evaluate ----
        model.eval()
        running_loss, correct, total = 0.0, 0, 0
        all_preds, all_labels = [], []
        with torch.no_grad():
            for images, labels in tqdm(test_loader,
                                       desc=f"Eval  {epoch}/{epochs}",
                                       unit="batch"):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss    = criterion(outputs, labels)

                running_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total   += labels.size(0)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        test_loss = running_loss / total
        test_acc  = correct / total

        # ---- Log ----
        train_losses.append(train_loss); test_losses.append(test_loss)
        train_accs.append(train_acc);    test_accs.append(test_acc)

        writer.add_scalar("Loss/Train",     train_loss, epoch)
        writer.add_scalar("Loss/Test",      test_loss,  epoch)
        writer.add_scalar("Accuracy/Train", train_acc,  epoch)
        writer.add_scalar("Accuracy/Test",  test_acc,   epoch)

        print(f"\nEpoch {epoch}/{epochs}"
              f" | Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f}"
              f" | Test Loss: {test_loss:.4f},  Acc: {test_acc:.4f}")

    writer.close()
    return train_losses, test_losses, train_accs, test_accs, all_preds, all_labels

# ----------------------------
# 5) RUN TRAINING
# ----------------------------
if __name__ == "__main__":
    tr_loss, te_loss, tr_acc, te_acc, preds, labels = train_model(
        model, train_loader, test_loader,
        criterion, optimizer, epochs=10
    )

    # ----------------------------
    # 6) PLOTS & CONFUSION MATRIX
    # ----------------------------
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(tr_loss, label="Train Loss")
    plt.plot(te_loss, label="Test  Loss")
    plt.legend(); plt.title("Loss Curve")

    plt.subplot(1, 2, 2)
    plt.plot(tr_acc, label="Train Acc")
    plt.plot(te_acc, label="Test  Acc")
    plt.legend(); plt.title("Accuracy Curve")
    plt.show()

    
    
    cm = confusion_matrix(labels, preds, normalize="true") * 100

    
    cm_labels = np.array([["{:.1f} %".format(v) for v in row] for row in cm])

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm.T,                  
        annot=cm_labels,       
        fmt="",
        cmap="Blues",
        xticklabels=actions,   
        yticklabels=actions    
    )
    plt.ylabel("True")
    plt.xlabel("Predicted")
    plt.title("Confusion Matrix (%)")
    plt.tight_layout()
    plt.show()
